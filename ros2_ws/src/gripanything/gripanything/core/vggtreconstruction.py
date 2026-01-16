#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VGGT — Point-cloud reconstruction + camera pose export + frustum visualization
(Images are sorted by numeric pose_<id>_image.*)

Outputs (under OUT_DIR):
- points.ply          Reconstructed point cloud
- cameras.json        Per-frame camera entries (cam_T_world, K, image_size, filename)
- cameras_lines.ply   Camera frustums/axes as a merged Open3D LineSet (if open3d is installed)
- Interactive view (optional): point cloud + camera line visuals

Dependencies:
- Required: torch, numpy, trimesh, tqdm, vggt
- Optional: open3d (for frustum export + visualization)
"""

from __future__ import annotations

import os
import re
import glob
import json
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Default paths (for reuse/import)
# -----------------------------------------------------------------------------
INPUT_IMAGES_DIR_DEFAULT = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ur5image"
OUTPUT_DIR_DEFAULT = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/vggt_output"

# -----------------------------------------------------------------------------
# Visualization parameters (Open3D only)
# -----------------------------------------------------------------------------
FRUSTUM_LINE_COLOR = (1.0, 0.0, 0.0)   # Camera frustum line color
AXES_RGB = True                        # True: XYZ axes in RGB; False: use FRUSTUM_LINE_COLOR
FRUSTUM_DEPTH_FACTOR = 0.15            # Frustum depth relative to scene extent
AXES_SIZE_FACTOR = 0.15                # Axis length relative to frustum depth

# -----------------------------------------------------------------------------
# Optional Open3D
# -----------------------------------------------------------------------------
try:
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# -----------------------------------------------------------------------------
# VGGT import path fix (kept local to this file to be drop-in reusable)
# -----------------------------------------------------------------------------
import sys
_VGGT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vggt"))
if os.path.isdir(_VGGT_PATH) and _VGGT_PATH not in sys.path:
    sys.path.append(_VGGT_PATH)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues


# -----------------------------------------------------------------------------
# Image collection (sorted by pose_<id>_image.*)
# -----------------------------------------------------------------------------
IMAGE_NAME_REGEX = re.compile(
    r"pose_(\d+)_image\.(jpg|jpeg|png|bmp|tif|tiff|webp)$",
    re.IGNORECASE
)

def _natural_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    m = IMAGE_NAME_REGEX.search(base)
    if m:
        return (int(m.group(1)), base)
    return (10**9, base)

def gather_image_paths(images_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    paths = [
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if os.path.isfile(p) and p.lower().endswith(exts)
    ]
    paths.sort(key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return paths

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Open3D helpers for frustums / axes
# -----------------------------------------------------------------------------
def _make_frustum_lines(K: np.ndarray, w: int, h: int, depth: float,
                        color=(1.0, 0.0, 0.0)) -> "o3d.geometry.LineSet":
    """Create a camera frustum (pyramid) in the camera coordinate frame (+Z forward)."""
    invK = np.linalg.inv(K)
    corners_px = np.array(
        [[0, 0, 1],
         [w, 0, 1],
         [w, h, 1],
         [0, h, 1]],
        dtype=np.float64
    )
    rays = (invK @ corners_px.T).T
    rays = rays / rays[:, 2:3]
    pts = np.vstack([np.zeros((1, 3)), depth * rays])  # [C(0), TL(1), TR(2), BR(3), BL(4)]

    lines = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4],
         [1, 2], [2, 3], [3, 4], [4, 1]],
        dtype=np.int32
    )
    col = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls

def _make_axes_lines(size: float, rgb: bool = True) -> "o3d.geometry.LineSet":
    """Create XYZ axes lines in the camera coordinate frame."""
    pts = np.array(
        [[0, 0, 0],
         [size, 0, 0],
         [0, size, 0],
         [0, 0, size]],
        dtype=np.float64
    )
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    if rgb:
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    else:
        colors = np.tile(np.array(FRUSTUM_LINE_COLOR, dtype=np.float64), (3, 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def _concat_linesets(line_sets: List["o3d.geometry.LineSet"]) -> "o3d.geometry.LineSet":
    """Merge multiple LineSet objects into a single LineSet (for writing to disk)."""
    merged = o3d.geometry.LineSet()
    if not line_sets:
        return merged

    pts_all, lines_all, cols_all = [], [], []
    offset = 0
    for ls in line_sets:
        p = np.asarray(ls.points)
        l = np.asarray(ls.lines)
        c = np.asarray(ls.colors) if ls.has_colors() else np.ones((l.shape[0], 3), dtype=np.float64)

        pts_all.append(p)
        lines_all.append(l + offset)
        cols_all.append(c)
        offset += p.shape[0]

    merged.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(lines_all))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    return merged

def build_camera_line_visuals(cam_list: List[Dict[str, Any]], scene_extent: float
                              ) -> Tuple[List["o3d.geometry.LineSet"], "o3d.geometry.LineSet"]:
    """
    Build per-camera frustum + axes LineSets for visualization.
    Returns:
      - geoms: list of LineSets for draw_geometries
      - merged: merged LineSet for writing to cameras_lines.ply
    """
    geoms: List[o3d.geometry.LineSet] = []
    depth = max(0.05, FRUSTUM_DEPTH_FACTOR * float(scene_extent))
    axes_size = AXES_SIZE_FACTOR * depth

    for cam in cam_list:
        K = np.array(cam["K"], dtype=np.float64)
        w, h = cam["image_size"]
        T_cw = np.array(cam["cam_T_world"], dtype=np.float64)

        fr = _make_frustum_lines(K, w, h, depth=depth, color=FRUSTUM_LINE_COLOR)
        fr.transform(T_cw)
        geoms.append(fr)

        ax = _make_axes_lines(axes_size, rgb=AXES_RGB)
        ax.transform(T_cw)
        geoms.append(ax)

    merged = _concat_linesets(geoms)
    return geoms, merged


# -----------------------------------------------------------------------------
# Config + core runner
# -----------------------------------------------------------------------------
@dataclass
class VGGTConfig:
    images_dir: str = INPUT_IMAGES_DIR_DEFAULT
    out_dir: str = OUTPUT_DIR_DEFAULT

    batch_size: int = 30
    max_points: int = 1_500_000
    resolution: int = 518
    conf_thresh: float = 1.5
    img_limit: Optional[int] = None

    auto_visualize: bool = True
    seed: int = 42

    # Model checkpoint (kept explicit to avoid hidden defaults)
    ckpt_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class VGGTReconstructor:
    """
    A reusable VGGT reconstructor class:
    - run() returns a dict with output paths and camera entries (ready for downstream scripts).
    """

    def __init__(self, cfg: VGGTConfig) -> None:
        self.cfg = cfg

        # Seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # Device + dtype
        if torch.cuda.is_available():
            self.device = "cuda"
            major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # Model
        self.model = VGGT()
        state = torch.hub.load_state_dict_from_url(cfg.ckpt_url, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        # Ensure output dir
        ensure_dir(cfg.out_dir)

    def _forward_batch(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        images = F.interpolate(images, size=(self.cfg.resolution, self.cfg.resolution), mode="bilinear", align_corners=False)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    images_5d = images[None]  # [1,N,3,H,W]
                    tokens, ps_idx = self.model.aggregator(images_5d)
                    pose_enc = self.model.camera_head(tokens)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_5d.shape[-2:])
                    depth_map, depth_conf = self.model.depth_head(tokens, images_5d, ps_idx)
            else:
                images_5d = images[None]
                tokens, ps_idx = self.model.aggregator(images_5d)
                pose_enc = self.model.camera_head(tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_5d.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(tokens, images_5d, ps_idx)

        return (
            extrinsic.squeeze(0).cpu().numpy(),  # [N,4,4] world->cam
            intrinsic.squeeze(0).cpu().numpy(),  # [N,3,3]
            depth_map.squeeze(0).cpu().numpy(),  # [N,1,H,W] or [N,H,W,1]
            depth_conf.squeeze(0).cpu().numpy(),
            images,                              # [N,3,H,W]
        )

    @staticmethod
    def _to_nhw1(arr: np.ndarray) -> np.ndarray:
        """
        Normalize depth/conf outputs to shape (N,H,W,1).
        """
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3:
            arr = arr[..., None]
        return arr

    def run(self) -> Dict[str, Any]:
        """
        Execute reconstruction.

        Returns a dict:
          {
            "points_ply": <path>,
            "cameras_json": <path>,
            "cameras_lines_ply": <path or None>,
            "num_points": <int>,
            "num_frames": <int>,
            "camera_entries": <list>
          }
        """
        cfg = self.cfg

        image_paths_all = gather_image_paths(cfg.images_dir)
        if cfg.img_limit is not None:
            image_paths_all = image_paths_all[: cfg.img_limit]

        n_total = len(image_paths_all)
        if n_total == 0:
            raise FileNotFoundError("No images available for reconstruction.")

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        cam_entries: List[Dict[str, Any]] = []

        idx_global = 0

        for i in tqdm(range(0, n_total, cfg.batch_size), desc="VGGT"):
            batch_paths = image_paths_all[i: i + cfg.batch_size]
            images, _ = load_and_preprocess_images_square(batch_paths, cfg.resolution)
            images = images.to(self.device)

            extr, intr, depth, conf, imgs_resized = self._forward_batch(images)

            depth = self._to_nhw1(depth)
            conf = self._to_nhw1(conf)

            # If conf is malformed, fall back to ones
            if conf.ndim != 4 or conf.shape[-1] != 1:
                conf = np.ones_like(depth, dtype=depth.dtype)

            # Unproject to 3D points in world frame (as defined by VGGT)
            pts3d = unproject_depth_map_to_point_map(depth, extr, intr)  # (N,H,W,3)

            mask = (conf[..., 0] >= cfg.conf_thresh)
            mask = randomly_limit_trues(mask, cfg.max_points)

            rgb = (imgs_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

            all_points.append(pts3d[mask])
            all_colors.append(rgb[mask])

            # Camera entries
            N = extr.shape[0]
            for k in range(N):
                E_wc = extr[k]  # world->cam
                if E_wc.shape == (4, 4):
                    T_cw = np.linalg.inv(E_wc)  # cam_T_world
                elif E_wc.shape == (3, 4):
                    E44 = np.eye(4, dtype=np.float64)
                    E44[:3, :] = E_wc
                    T_cw = np.linalg.inv(E44)
                else:
                    raise RuntimeError(f"Invalid extrinsic shape: {E_wc.shape}")

                K = intr[k]
                cam_entries.append({
                    "index": idx_global,
                    "file": os.path.basename(batch_paths[k]),
                    "cam_T_world": T_cw.tolist(),
                    "K": K.tolist(),
                    "image_size": [cfg.resolution, cfg.resolution],
                })
                idx_global += 1

        # Export point cloud
        pts = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float32)
        cols = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)

        points_ply = os.path.join(cfg.out_dir, "points.ply")
        trimesh.PointCloud(pts, colors=cols).export(points_ply)

        # Export cameras.json
        cameras_json = os.path.join(cfg.out_dir, "cameras.json")
        with open(cameras_json, "w", encoding="utf-8") as f:
            json.dump(cam_entries, f, ensure_ascii=False, indent=2)

        cameras_lines_ply: Optional[str] = None

        # Optional Open3D frustums + visualization
        if _HAS_O3D:
            try:
                pcd = o3d.io.read_point_cloud(points_ply)
                if not pcd.is_empty():
                    aabb = pcd.get_axis_aligned_bounding_box()
                    extent = float(np.max(aabb.get_extent()))
                    extent = extent if extent > 0 else 1.0

                    cam_geoms, cams_lines = build_camera_line_visuals(cam_entries, extent)
                    cameras_lines_ply = os.path.join(cfg.out_dir, "cameras_lines.ply")
                    o3d.io.write_line_set(cameras_lines_ply, cams_lines)

                    if cfg.auto_visualize:
                        geoms = [pcd] + cam_geoms
                        o3d.visualization.draw_geometries(geoms)
            except Exception as e:
                # Keep this non-fatal for headless or minimal environments
                print(f"[Warn] Open3D visualization/export failed: {e}")
        else:
            if cfg.auto_visualize:
                print("[Info] open3d is not installed; skipping frustum export and visualization. Install via: pip install open3d")

        return {
            "points_ply": points_ply,
            "cameras_json": cameras_json,
            "cameras_lines_ply": cameras_lines_ply,
            "num_points": int(pts.shape[0]),
            "num_frames": int(len(cam_entries)),
            "camera_entries": cam_entries,
        }


# -----------------------------------------------------------------------------
# Standalone debug entry point 
# -----------------------------------------------------------------------------
def main():
    cfg = VGGTConfig(
        images_dir=INPUT_IMAGES_DIR_DEFAULT,
        out_dir=OUTPUT_DIR_DEFAULT,
        batch_size=30,
        max_points=1_500_000,
        resolution=518,
        conf_thresh=1.5,
        img_limit=None,
        auto_visualize=True,
        seed=42,
    )
    recon = VGGTReconstructor(cfg)
    with torch.inference_mode():
        out = recon.run()
    print(f"[Done] points: {out['points_ply']}")
    print(f"[Done] cameras: {out['cameras_json']}")
    if out["cameras_lines_ply"]:
        print(f"[Done] cameras_lines: {out['cameras_lines_ply']}")
    print(f"[Done] frames={out['num_frames']}  points={out['num_points']}")


if __name__ == "__main__":
    main()
