#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VGGT — 点云重建 + 相机位姿导出与线框可视化（按 pose_<id>_image.* 数字排序）

输出：
- <out_dir>/points.ply                 点云
- <out_dir>/cameras.json               每帧相机位姿（cam_T_world, K, image_size, 文件名）
- <out_dir>/cameras_lines.ply          所有相机线框（LineSet，需 open3d）
- 交互可视化（若已安装 open3d）：点云 + 相机线框 + 坐标轴

依赖：torch, numpy, trimesh, tqdm, vggt；可选 open3d
"""

import os
import re
import glob
import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
import json

# ================== 可调参数（不走命令行） ==================
# 相机线框颜色（红色），坐标轴默认 RGB（X红/Y绿/Z蓝）
FRUSTUM_LINE_COLOR = (1.0, 0.0, 0.0)   # 线框颜色
AXES_RGB = True                        # True=RGB，False=全用 FRUSTUM_LINE_COLOR
FRUSTUM_DEPTH_FACTOR = 0.15            # 相机金字塔深度相对场景尺度的比例
AXES_SIZE_FACTOR = 0.15                # 坐标轴长度相对金字塔深度的比例
# ==========================================================

# ====== 可选：Open3D 可视化/写线集 ======
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# ====== vggt 路径修正 ======
import sys
VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
if os.path.isdir(VGGT_path) and VGGT_path not in sys.path:
    sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues


# -------------------- 文件收集（按 pose_<id>_image.* 数字排序） --------------------
IMAGE_NAME_REGEX = re.compile(r"pose_(\d+)_image\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", re.IGNORECASE)

def _natural_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    m = IMAGE_NAME_REGEX.search(base)
    if m:
        return (int(m.group(1)), base)
    return (10**9, base)  # 没匹配到的放后面

def gather_image_paths(images_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    paths = [p for p in glob.glob(os.path.join(images_dir, "*"))
             if os.path.isfile(p) and p.lower().endswith(exts)]
    paths.sort(key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"找不到图片：{images_dir}")
    return paths


def choose_out_dir(images_dir: str, out_dir: Optional[str]) -> str:
    candidates = [out_dir] if out_dir else []
    parent = os.path.dirname(os.path.abspath(images_dir.rstrip("/")))
    candidates += [
        os.path.join(parent, "vggt_output"),
        os.path.join(os.path.abspath(images_dir), "vggt_output"),
        os.path.join(os.path.expanduser("~"), "vggt_output"),
    ]
    last_err = None
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            print(f"[Info] 输出目录：{cand}")
            return cand
        except PermissionError as e:
            last_err = e
            print(f"[Warn] 无法创建输出目录：{cand}（{e}），尝试下一个。")
    raise last_err or PermissionError("无法创建任何输出目录，请手动指定 out_dir 到可写路径。")


# -------------------- 线框构建（Open3D LineSet） --------------------
def _make_frustum_lines(K: np.ndarray, w: int, h: int, depth: float,
                        color=(1.0, 0.0, 0.0)) -> "o3d.geometry.LineSet":
    """在相机坐标系下生成金字塔线框（原点为相机中心，+Z 朝前）"""
    invK = np.linalg.inv(K)
    corners_px = np.array([[0, 0, 1],
                           [w, 0, 1],
                           [w, h, 1],
                           [0, h, 1]], dtype=np.float64)
    rays = (invK @ corners_px.T).T
    rays = rays / rays[:, 2:3]
    pts = np.vstack([np.zeros((1, 3)), depth * rays])  # [C(0), TL(1), TR(2), BR(3), BL(4)]

    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # 四条侧边
        [1, 2], [2, 3], [3, 4], [4, 1],  # 底面矩形
    ], dtype=np.int32)
    col = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls

def _make_axes_lines(size: float, rgb: bool = True) -> "o3d.geometry.LineSet":
    """相机坐标轴线（原点到 +X/+Y/+Z）"""
    pts = np.array([[0, 0, 0],
                    [size, 0, 0],
                    [0, size, 0],
                    [0, 0, size]], dtype=np.float64)
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
    """把多份 LineSet 合并成一份（用于写入 cameras_lines.ply）"""
    merged = o3d.geometry.LineSet()
    if not line_sets:
        return merged
    pts_all, lines_all, cols_all = [], [], []
    offset = 0
    for ls in line_sets:
        p = np.asarray(ls.points)
        l = np.asarray(ls.lines)
        c = np.asarray(ls.colors) if ls.has_colors() else np.ones((l.shape[0], 3))
        pts_all.append(p)
        lines_all.append(l + offset)
        cols_all.append(c)
        offset += p.shape[0]
    merged.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
    merged.lines  = o3d.utility.Vector2iVector(np.vstack(lines_all))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    return merged


def build_camera_line_visuals(cam_list: List[Dict], scene_extent: float) -> Tuple[List["o3d.geometry.LineSet"], "o3d.geometry.LineSet"]:
    """返回 draw 用的 LineSet 列表，以及合并后的 LineSet（用于写文件）"""
    geoms: List[o3d.geometry.LineSet] = []
    depth = max(0.05, FRUSTUM_DEPTH_FACTOR * scene_extent)
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


# -------------------- 主类 --------------------
class VGGTMinimal:
    def __init__(
        self,
        images_dir: str,
        batch_size: int = 4,
        max_points: int = 300000,
        resolution: int = 518,
        conf_thresh: float = 2.5,
        img_limit: Optional[int] = None,
        out_dir: Optional[str] = None,
        auto_visualize: bool = True,
        seed: int = 42,
    ) -> None:
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.resolution = resolution
        self.conf_thresh = conf_thresh
        self.img_limit = img_limit
        self.auto_visualize = auto_visualize

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            self.device = "cuda"
            major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.model = VGGT()
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        self.out_dir = choose_out_dir(self.images_dir, out_dir)

    def _forward_batch(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        images = F.interpolate(images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
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
            extrinsic.squeeze(0).cpu().numpy(),  # [N,4,4]  world->cam
            intrinsic.squeeze(0).cpu().numpy(),  # [N,3,3]
            depth_map.squeeze(0).cpu().numpy(),  # [N,1,H,W] 或 [N,H,W,1]
            depth_conf.squeeze(0).cpu().numpy(),
            images,                               # [N,3,H,W]
        )

    def run(self) -> str:
        image_paths_all = gather_image_paths(self.images_dir)
        if self.img_limit is not None:
            image_paths_all = image_paths_all[: self.img_limit]
        n_total = len(image_paths_all)
        if n_total == 0:
            raise FileNotFoundError("没有可用图片。")

        all_points, all_colors = [], []
        cam_entries: List[Dict] = []
        idx_global = 0

        for i in tqdm(range(0, n_total, self.batch_size)):
            batch_paths = image_paths_all[i : i + self.batch_size]
            images, _ = load_and_preprocess_images_square(batch_paths, self.resolution)
            images = images.to(self.device)

            extr, intr, depth, conf, imgs_resized = self._forward_batch(images)

            # 统一 (N,H,W,1)
            depth = np.asarray(depth)
            conf = np.asarray(conf)
            if depth.ndim == 4 and depth.shape[1] == 1:
                depth = np.transpose(depth, (0, 2, 3, 1))
            elif depth.ndim == 3:
                depth = depth[..., None]
            if conf.ndim == 4 and conf.shape[1] == 1:
                conf = np.transpose(conf, (0, 2, 3, 1))
            elif conf.ndim == 3:
                conf = conf[..., None]
            elif conf.ndim != 4 or conf.shape[-1] != 1:
                conf = np.ones_like(depth, dtype=depth.dtype)

            # 反投影
            pts3d = unproject_depth_map_to_point_map(depth, extr, intr)  # (N,H,W,3)

            mask = (conf[..., 0] >= self.conf_thresh)
            mask = randomly_limit_trues(mask, self.max_points)

            rgb = (imgs_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

            all_points.append(pts3d[mask])
            all_colors.append(rgb[mask])

            # 相机条目
            N = extr.shape[0]
            for k in range(N):
                E_wc = extr[k]
                if E_wc.shape == (4, 4):
                    T_cw = np.linalg.inv(E_wc)     # cam<-world 的逆：cam_T_world
                elif E_wc.shape == (3, 4):         # 防御：补齐到 4x4 再逆
                    E44 = np.eye(4, dtype=np.float64); E44[:3, :] = E_wc
                    T_cw = np.linalg.inv(E44)
                else:
                    raise RuntimeError(f"非法外参形状：{E_wc.shape}")

                K = intr[k]
                cam_entries.append({
                    "index": idx_global,
                    "file": os.path.basename(batch_paths[k]),
                    "cam_T_world": T_cw.tolist(),
                    "K": K.tolist(),
                    "image_size": [self.resolution, self.resolution],
                })
                idx_global += 1

        # 汇总导出点云
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        ply_path = os.path.join(self.out_dir, "points.ply")
        trimesh.PointCloud(all_points, colors=all_colors).export(ply_path)
        print(f"[Done] 点云 PLY 已保存: {ply_path}  （共 {all_points.shape[0]} 点）")

        # 导出相机 JSON
        cams_json = os.path.join(self.out_dir, "cameras.json")
        with open(cams_json, "w", encoding="utf-8") as f:
            json.dump(cam_entries, f, ensure_ascii=False, indent=2)
        print(f"[Done] 相机参数已保存: {cams_json}  （{len(cam_entries)}/{n_total} 帧）")

        # 线框可视化 + 写入线集
        if _HAS_O3D:
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                if pcd.is_empty():
                    print("[Warn] 读取到的点云为空，跳过可视化。")
                else:
                    aabb = pcd.get_axis_aligned_bounding_box()
                    extent = float(np.max(aabb.get_extent()))
                    geoms = [pcd]
                    cams_geoms, cams_lines = build_camera_line_visuals(cam_entries, extent if extent > 0 else 1.0)
                    geoms.extend(cams_geoms)

                    # 写所有相机线框到单个 LineSet
                    cams_lines_path = os.path.join(self.out_dir, "cameras_lines.ply")
                    o3d.io.write_line_set(cams_lines_path, cams_lines)
                    print(f"[Done] 相机线框已保存: {cams_lines_path}")

                    # 交互查看
                    o3d.visualization.draw_geometries(geoms)
            except Exception as e:
                print(f"[Warn] Open3D 可视化/写线框失败：{e}")
        else:
            print("[Info] 未安装 open3d，跳过可视化与相机线框导出。安装：pip install open3d")

        return ply_path


# -------------------- 入口 --------------------
if __name__ == "__main__":
    IMAGES_DIR = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/pictures/yellow_cube_30"

    app = VGGTMinimal(
        images_dir=IMAGES_DIR,
        batch_size=30,
        max_points=1_500_000,
        resolution=518,
        conf_thresh=2.7,
        img_limit=None,
        out_dir=None,          # None => 默认同级 vggt_output
        auto_visualize=True,
        seed=42,
    )

    with torch.inference_mode():
        app.run()
