#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeanything.py — Two-pass detection + circular path + snapshots + VGGT + postprocess + final goto

This version applies MINIMAL changes on top of your current script, and adds:
- Multi-candidate selection at the high observation stage (INIT / stage-1 view)
- Per-candidate close hover + snapshot + DINO->SAM2 rgbmask + LightGlue scoring vs latest Aria mask
- Orbit capture (polygon path) only around the best-matching candidate

Key guarantees preserved:
- Atomic capture binding {image, image_stamp} and TF computed strictly at image_stamp (NO fallback-to-latest)
- Offline pipeline behavior unchanged (VGGT + postprocess + optional offline bias + final goto)

Aria outputs (UPDATED - naming-robust):
- This node reads the latest {RGB, MASK} pair from:
    <OUTPUT_ROOT>/ariaimage
  It no longer assumes a single hard-coded filename pattern.
  It will:
    - search RGB candidates: filenames containing 'rgb' (not containing 'mask')
    - search MASK candidates: filenames containing 'mask'
    - pair them by (a) common stem after stripping known rgb/mask suffixes, or (b) shared timestamp digits
- Then it builds an Aria object-only rgbmask (black background) into:
    <OUTPUT_ROOT>/match_results/aria_rgbmask.png

New outputs:
- Candidate evaluation snapshots / masks / json:
    <OUTPUT_ROOT>/match_results/candidates/
    <OUTPUT_ROOT>/match_results/candidate_selection.json

[NEW in this version]
More debug prints:
- Stage-1 DINO detection summary: total detections, >= min_exec_score count, top-k listing
- Route decision prints: ROUTE-1 legacy vs ROUTE-2 multi-candidate (and reasons)
- ROUTE-2 candidate list prints: each candidate stage-1 dino score and geometry
- ROUTE-2 per-candidate matching prints: match_score/quality/coverage/Kconf
- Final candidate comparison table before selection + selection reason (or fallback to top DINO)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import math
import os
import re
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration as RclDuration
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge

import tf2_ros

# LightGlue (already used in your matcher module; embedded minimal scoring here for robustness)
import torch
from lightglue import LightGlue, SuperPoint


# -----------------------------------------------------------------------------
# Output paths (fixed)
# -----------------------------------------------------------------------------
OUTPUT_ROOT = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "ur5image")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_ROOT, "ur5camerajointstates.json")
OUTPUT_VGGT_DIR = os.path.join(OUTPUT_ROOT, "offline_output")
OUTPUT_OBJECT_JSON_NAME = "object_in_base_link.json"

# Aria outputs (from your Aria streaming script)
ARIA_OUT_DIR = os.path.join(OUTPUT_ROOT, "ariaimage")

# New: matching + candidate selection outputs
MATCH_ROOT = os.path.join(OUTPUT_ROOT, "match_results")
MATCH_CAND_DIR = os.path.join(MATCH_ROOT, "candidates")
MATCH_JSON_PATH = os.path.join(MATCH_ROOT, "candidate_selection.json")
ARIA_RGBMASK_PATH = os.path.join(MATCH_ROOT, "aria_rgbmask.png")
BEST_VIZ_PATH = os.path.join(MATCH_ROOT, "best_match_viz.png")


# -----------------------------------------------------------------------------
# Offline pipeline toggles (recommended defaults for ROS)
# -----------------------------------------------------------------------------
RUN_OFFLINE_PIPELINE = True
VGGT_AUTO_VISUALIZE = False          # Avoid Open3D window blocking the ROS node
POSTPROCESS_VISUALIZE = False        # Avoid Open3D window blocking the ROS node

# VGGT reconstruction parameters
VGGT_BATCH_SIZE = 30
VGGT_MAX_POINTS = 1_500_000
VGGT_RESOLUTION = 518
VGGT_CONF_THRESH = 1.5

# Alignment mode for VGGT-world -> base_link ("sim3" estimates scale; "se3" forces scale=1)
POST_ALIGN_METHOD = "sim3"

# IMPORTANT: Our VGGT cameras.json stores "cam_T_world" (cam <- world).
# The postprocess loader expects world_T_cam, so we set vggt_pose_is_world_T_cam accordingly.
POST_VGGT_POSE_IS_WORLD_T_CAM = True


# -----------------------------------------------------------------------------
# Final goto (after offline pipeline)
# -----------------------------------------------------------------------------
FINAL_GOTO_ENABLE = True
FINAL_GOTO_Z_OFFSET = 0.15          # meters above object center
FINAL_GOTO_MOVE_TIME = 3.0          # seconds
FINAL_GOTO_Z_MIN = 0.05             # safety clamp
FINAL_GOTO_Z_MAX = 2.00             # safety clamp


# -----------------------------------------------------------------------------
# New: multi-candidate + cross-view selection knobs
# -----------------------------------------------------------------------------
ENABLE_MULTI_CANDIDATE = True

# If high-stage detection yields >=2 candidates above cfg.dino.min_exec_score,
# we will evaluate them at close hover and pick best match.
CAND_MAX = 6                 # evaluate top-N candidates by DINO score
CAND_DWELL_SEC = 1.5         # pause duration per candidate for snapshot + match
MATCH_SCORE_ACCEPT = 0.25    # if best match < this, fallback to best DINO candidate

# SAM2 segmentation for candidate snapshots
SAM2_ID = "facebook/sam2.1-hiera-large"
SAM2_MASK_THRESHOLD = 0.30
SAM2_MAX_HOLE_AREA = 100.0
SAM2_MAX_SPRINKLE_AREA = 50.0


# -----------------------------------------------------------------------------
# Debug prints (more transparency for stage-1 and candidate selection)
# -----------------------------------------------------------------------------
DEBUG_STAGE1_DINO_PRINT = True
DEBUG_STAGE1_DINO_TOPK = 10
DEBUG_STAGE1_DINO_PRINT_BOXES = True
DEBUG_ROUTE_PRINT = True
DEBUG_CAND_TABLE_PRINT = True


# -----------------------------------------------------------------------------
# DINO wrapper (robust import)
# -----------------------------------------------------------------------------
try:
    from gripanything.core.detect_with_dino import GroundingDinoPredictor
except Exception:
    import sys
    sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
    from gripanything.core.detect_with_dino import GroundingDinoPredictor

# SAM2 wrapper (your repo module)
try:
    from gripanything.core.segment_with_sam2 import SAM2ImagePredictorWrapper
except Exception:
    import sys
    sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
    from gripanything.core.segment_with_sam2 import SAM2ImagePredictorWrapper

from gripanything.core.config import load_from_ros_params
from gripanything.core.vision_geom import SingleShotDetector
from gripanything.core.ik_and_traj import MotionContext, TrajectoryPublisher, IKClient
from gripanything.core.polygon_path import make_polygon_vertices

# Reuse TF utilities
from gripanything.core.tf_ops import tfmsg_to_Rp, R_CL_CO, quat_to_rot


# -----------------------------------------------------------------------------
# Offline pipeline imports (VGGT + point cloud postprocess)
# -----------------------------------------------------------------------------
def _import_offline_modules():
    """
    Imports offline pipeline modules based on the current repository tree.

    Expected:
      - gripanything.core.vggtreconstruction provides: VGGTConfig, VGGTReconstructor
      - gripanything.core.point_processing provides: Config, process_pointcloud
    """
    try:
        from gripanything.core.vggtreconstruction import VGGTConfig as _VGGTConfig, VGGTReconstructor as _VGGTReconstructor
    except Exception:
        import sys
        sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
        from gripanything.core.vggtreconstruction import VGGTConfig as _VGGTConfig, VGGTReconstructor as _VGGTReconstructor

    try:
        from gripanything.core.point_processing import Config as _PostConfig, process_pointcloud as _process_pointcloud
    except Exception:
        import sys
        sys.path.append("/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything")
        from gripanything.core.point_processing import Config as _PostConfig, process_pointcloud as _process_pointcloud

    return _VGGTConfig, _VGGTReconstructor, _PostConfig, _process_pointcloud


def _safe_remove(path: str) -> None:
    """Remove a file if it exists; do nothing on failure."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cleanup_outputs_for_new_run() -> None:
    """
    Ensure that a new run overwrites outputs cleanly:
    - Remove old pose_* images so leftover frames do not contaminate the next run
    - Remove old JSON log
    - Remove old VGGT outputs that we generate
    - Prepare match output dirs
    """
    _ensure_dir(OUTPUT_ROOT)
    _ensure_dir(OUTPUT_IMG_DIR)
    _ensure_dir(OUTPUT_VGGT_DIR)
    _ensure_dir(MATCH_ROOT)
    _ensure_dir(MATCH_CAND_DIR)

    # Images from previous run (VGGT inputs)
    try:
        for fn in os.listdir(OUTPUT_IMG_DIR):
            if fn.startswith("pose_") and (fn.endswith(".png") or fn.endswith(".jpg")):
                _safe_remove(os.path.join(OUTPUT_IMG_DIR, fn))
    except Exception:
        pass

    # Main shot log
    _safe_remove(OUTPUT_JSON_PATH)

    # VGGT outputs
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras_lines.ply"))
    _safe_remove(os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME))

    # Match outputs
    _safe_remove(MATCH_JSON_PATH)
    _safe_remove(ARIA_RGBMASK_PATH)
    _safe_remove(BEST_VIZ_PATH)
    try:
        for fn in os.listdir(MATCH_CAND_DIR):
            _safe_remove(os.path.join(MATCH_CAND_DIR, fn))
    except Exception:
        pass


def _stamp_to_ns(stamp_msg) -> int:
    try:
        return int(stamp_msg.sec) * 1_000_000_000 + int(stamp_msg.nanosec)
    except Exception:
        return -1


# -----------------------------------------------------------------------------
# Minimal LightGlue matcher + score (embedded for robustness)
# -----------------------------------------------------------------------------
class _LGMatcher:
    def __init__(self, device: str = "cuda", mp: bool = True):
        self.device = device
        self.matcher = LightGlue(
            features="superpoint",
            descriptor_dim=256,
            n_layers=9,
            num_heads=4,
            flash=True,
            mp=mp,
            depth_confidence=0.95,
            width_confidence=0.99,
            filter_threshold=0.1,
            weights=None,
        ).to(device)
        self.extractor = SuperPoint().to(device)

    @staticmethod
    def _bgr_to_tensor_rgb01(img_bgr: np.ndarray, device: str) -> torch.Tensor:
        """
        FIX: avoid negative-stride numpy views.
        torch.from_numpy() does NOT accept negative strides (e.g., img[..., ::-1]).
        Use cv2.cvtColor + np.ascontiguousarray instead.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        return t.to(device)

    def extract(self, img_bgr: np.ndarray) -> dict:
        image = self._bgr_to_tensor_rgb01(img_bgr, self.device)
        feats = self.extractor.extract(image)
        h, w = img_bgr.shape[:2]
        feats["image_size"] = torch.tensor([w, h], device=self.device).unsqueeze(0)
        return feats

    def match(self, feats0: dict, feats1: dict) -> dict:
        data = {
            "image0": {
                "keypoints": feats0["keypoints"].to(self.device),
                "descriptors": feats0["descriptors"].to(self.device),
                "image_size": feats0["image_size"].to(self.device),
            },
            "image1": {
                "keypoints": feats1["keypoints"].to(self.device),
                "descriptors": feats1["descriptors"].to(self.device),
                "image_size": feats1["image_size"].to(self.device),
            },
        }
        return self.matcher(data)


def _object_bbox_from_black_bg(img_bgr: np.ndarray, thr: int = 8) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thr).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _crop_pad_resize(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: int = 512,
    pad_ratio: float = 0.15,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = img_bgr.shape[:2]
    bw, bh = (x1 - x0), (y1 - y0)
    if bw <= 0 or bh <= 0:
        return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

    pad = int(round(max(bw, bh) * pad_ratio))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    half = 0.5 * max(bw, bh) + pad

    nx0 = int(round(cx - half))
    ny0 = int(round(cy - half))
    nx1 = int(round(cx + half))
    ny1 = int(round(cy + half))

    pad_left = max(0, -nx0)
    pad_top = max(0, -ny0)
    pad_right = max(0, nx1 - w)
    pad_bottom = max(0, ny1 - h)

    nx0 = max(0, nx0)
    ny0 = max(0, ny0)
    nx1 = min(w, nx1)
    ny1 = min(h, ny1)

    crop = img_bgr[ny0:ny1, nx0:nx1]
    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def _preprocess_object_only(img_bgr: np.ndarray, out_size: int = 512, pad_ratio: float = 0.15) -> np.ndarray:
    bbox = _object_bbox_from_black_bg(img_bgr)
    if bbox is None:
        return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return _crop_pad_resize(img_bgr, bbox=bbox, out_size=out_size, pad_ratio=pad_ratio)


def _compute_object_match_score(scores: np.ndarray, score_th: float = 0.01, topk: int = 10, k_ref: int = 20) -> Dict[str, Any]:
    if scores is None:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0}

    scores = np.asarray(scores, dtype=np.float32)
    conf = scores[scores >= float(score_th)]
    K_conf = int(conf.size)
    if K_conf == 0:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0}

    conf_sorted = np.sort(conf)[::-1]
    k = int(min(topk, K_conf))
    quality = float(conf_sorted[:k].mean())
    coverage = float(min(1.0, K_conf / float(max(1, k_ref))))
    score = float(np.clip(quality * coverage, 0.0, 1.0))
    return {"score": score, "quality": quality, "coverage": coverage, "K_conf": K_conf}


def _draw_matches_image(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    kpts0_xy: np.ndarray,
    kpts1_xy: np.ndarray,
    matches_ij: np.ndarray,
    scores: np.ndarray,
    topk: int = 200,
) -> np.ndarray:
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:w0 + w1] = img1_bgr
    right_offset = np.array([w0, 0], dtype=np.float32)

    if matches_ij.size == 0:
        return canvas

    K = min(int(topk), matches_ij.shape[0])
    for idx in range(K):
        i0, i1 = matches_ij[idx]
        p0 = kpts0_xy[i0].astype(np.int32)
        p1 = (kpts1_xy[i1] + right_offset).astype(np.int32)
        cv2.line(canvas, tuple(p0), tuple(p1), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p0), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p1), 2, (0, 0, 255), -1, cv2.LINE_AA)
        s = float(scores[idx]) if scores is not None and scores.size > idx else 0.0
        mid = ((p0 + p1) * 0.5).astype(np.int32)
        cv2.putText(canvas, f"{s:.2f}", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return canvas


# -----------------------------------------------------------------------------
# Aria rgbmask builder (UPDATED: naming-robust pairing)
# -----------------------------------------------------------------------------
def _strip_known_suffix(stem: str, suffixes: List[str]) -> str:
    s = stem
    lowered = s.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lowered.endswith(suf):
                s = s[: -len(suf)]
                lowered = s.lower()
                changed = True
                break
    return s


def _extract_ts_token(name_lower: str) -> Optional[str]:
    """
    Extract a plausible timestamp token from filename (>=10 digits).
    Returns the first match.
    """
    m = re.search(r"(\d{10,})", name_lower)
    if not m:
        return None
    return m.group(1)


def _is_rgb_candidate(p: Path) -> bool:
    n = p.name.lower()
    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
        return False
    if "mask" in n:
        return False
    if "rgb" not in n:
        return False
    if "rgbmask" in n:
        return False
    return True


def _is_mask_candidate(p: Path) -> bool:
    n = p.name.lower()
    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".npy"]:
        return False
    return ("mask" in n) or n.endswith("_mask.npy")


def _read_mask_any(path: str) -> Optional[np.ndarray]:
    """
    Read mask from .png/.jpg or .npy.
    Returns uint8 mask array.
    """
    try:
        if path.lower().endswith(".npy"):
            arr = np.load(path)
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.dtype != np.uint8:
                # normalize to {0,255} style
                if arr.max() <= 1:
                    arr = (arr > 0).astype(np.uint8) * 255
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return m
    except Exception:
        return None


def _find_latest_aria_pair(aria_dir: str) -> Optional[Tuple[str, str]]:
    """
    Find the latest (rgb, mask) pair from Aria output dir (naming-robust).

    Pairing strategies (in order):
    1) Same stem after stripping known rgb/mask suffixes.
    2) Same timestamp digits (>=10 digits) contained in filenames.
    3) Fallback: choose latest RGB and the latest MASK (if exists).
    """
    d = Path(aria_dir)
    if not d.exists():
        return None

    files = [p for p in d.iterdir() if p.is_file()]
    rgb_files = [p for p in files if _is_rgb_candidate(p)]
    mask_files = [p for p in files if _is_mask_candidate(p)]

    if not rgb_files or not mask_files:
        return None

    rgb_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    mask_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    rgb_suffixes = [
        "_rgb_rot", "_rgbrot", "_rgb", "_rgb8", "_rgb_raw", "_rgb_color",
        "_rgb_resized", "_rgb_resize", "_rgb_crop", "_rgb_cropped",
        "_rgb_vis", "_rgb_debug", "_rgb_out",
    ]
    mask_suffixes = [
        "_mask_bin", "_mask", "_mask_binary", "_mask_raw", "_mask_uint8",
        "_mask_vis", "_mask_debug", "_mask_out",
    ]

    # Pre-index masks by normalized base
    mask_by_base: Dict[str, List[Path]] = {}
    mask_by_ts: Dict[str, List[Path]] = {}

    for mp in mask_files:
        base = _strip_known_suffix(mp.stem, mask_suffixes)
        base = base.strip("_- ")
        mask_by_base.setdefault(base, []).append(mp)

        ts = _extract_ts_token(mp.name.lower())
        if ts:
            mask_by_ts.setdefault(ts, []).append(mp)

    # 1) Try base-stem pairing for latest RGB first
    for rp in rgb_files:
        base = _strip_known_suffix(rp.stem, rgb_suffixes)
        base = base.strip("_- ")
        if base in mask_by_base and mask_by_base[base]:
            # pick the newest mask among this base
            mps = sorted(mask_by_base[base], key=lambda p: p.stat().st_mtime, reverse=True)
            return str(rp), str(mps[0])

        # 2) Try timestamp pairing
        ts = _extract_ts_token(rp.name.lower())
        if ts and ts in mask_by_ts and mask_by_ts[ts]:
            mps = sorted(mask_by_ts[ts], key=lambda p: p.stat().st_mtime, reverse=True)
            return str(rp), str(mps[0])

        # 3) Try a soft contains search: any mask that contains the base substring
        if base:
            cand = [mp for mp in mask_files if base in mp.name]
            if cand:
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(rp), str(cand[0])

    # final fallback: latest rgb + latest mask
    return str(rgb_files[0]), str(mask_files[0])


def _build_rgbmask_blackbg(bgr: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """
    Build object-only BGR image (black background) from mask.
    mask_uint8 can be {0,255} or {0,1}.
    """
    if mask_uint8.dtype != np.uint8:
        mask_uint8 = mask_uint8.astype(np.uint8)
    if mask_uint8.max() <= 1:
        m = (mask_uint8 > 0)
    else:
        m = (mask_uint8 >= 128)

    out = np.zeros_like(bgr, dtype=np.uint8)
    if m.any():
        out[m] = bgr[m]
    return out


def _save_aria_rgbmask(aria_dir: str, out_path: str) -> Optional[Dict[str, str]]:
    pair = _find_latest_aria_pair(aria_dir)
    if pair is None:
        return None
    rgb_path, mask_path = pair

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    mask = _read_mask_any(mask_path)
    if mask is None:
        return None

    rgbmask = _build_rgbmask_blackbg(bgr, mask)
    _ensure_dir(str(Path(out_path).parent))
    cv2.imwrite(out_path, rgbmask)
    return {"rgb": rgb_path, "mask": mask_path, "aria_rgbmask": out_path}


# -----------------------------------------------------------------------------
# Candidate pack
# -----------------------------------------------------------------------------
class _Candidate:
    def __init__(self, idx: int, score: float, box_xyxy: Tuple[float, float, float, float], C: np.ndarray, hover_pose: PoseStamped):
        self.idx = idx
        self.score = float(score)
        self.box_xyxy = box_xyxy
        self.C = C.copy()
        self.hover_pose = hover_pose
        self.eval_image_path: Optional[str] = None
        self.eval_rgbmask_path: Optional[str] = None
        self.match_score: float = -1.0
        self.match_quality: float = 0.0
        self.match_coverage: float = 0.0
        self.match_Kconf: int = 0


class SeeAnythingNode(Node):
    def __init__(self):
        super().__init__("seeanything_minimal_clean")

        # Clean outputs at node startup
        _cleanup_outputs_for_new_run()

        # Load config
        self.cfg = load_from_ros_params(self)

        # Sweep angle in degrees (trim full loop to avoid joint limits)
        self._sweep_deg: float = float(getattr(getattr(self.cfg, "circle", object()), "sweep_deg", 120.0))

        # Prompt
        if self.cfg.runtime.require_prompt:
            user_prompt = input("Enter the target text prompt (e.g., 'orange object'): ").strip()
            if user_prompt:
                self.cfg.dino.text_prompt = user_prompt
                self.set_parameters([Parameter("text_prompt", value=user_prompt)])
                self.get_logger().info(f'Using user prompt: "{user_prompt}"')
            else:
                self.get_logger().warn(f'No input given; using default prompt: "{self.cfg.dino.text_prompt}"')
        else:
            self.get_logger().info(f'Interactive prompt disabled. Using: "{self.cfg.dino.text_prompt}"')

        # Fixed output locations
        self._img_dir = OUTPUT_IMG_DIR
        self._js_path = OUTPUT_JSON_PATH

        # JSON log for VGGT shots
        self._joint_log: Dict[str, Any] = {
            "note": "Per-image joint positions and camera pose at capture time.",
            "prompt": self.cfg.dino.text_prompt,
            "created_at": datetime.now().isoformat(),
            "joint_names": [],
            "shots": {}
        }
        self._flush_joint_log()

        # Candidate selection JSON
        self._match_log: Dict[str, Any] = {
            "note": "Candidate selection via Aria<->wrist matching.",
            "prompt": self.cfg.dino.text_prompt,
            "created_at": datetime.now().isoformat(),
            "aria_dir": ARIA_OUT_DIR,
            "aria_files": {},
            "candidates": [],
            "selected": {},
            "thresholds": {
                "min_exec_score": float(self.cfg.dino.min_exec_score),
                "cand_max": int(CAND_MAX),
                "cand_dwell_sec": float(CAND_DWELL_SEC),
                "match_accept": float(MATCH_SCORE_ACCEPT),
            }
        }
        self._flush_match_log()

        # Rolling camera frame buffer (BGR) + last image stamp (atomic)
        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_img_stamp = None  # msg.header.stamp

        # Image subscription
        qos_img = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, self.cfg.frames.image_topic, self._on_image, qos_img)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclDuration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_brd = tf2_ros.TransformBroadcaster(self)

        # DINO + detector
        self.predictor = GroundingDinoPredictor(self.cfg.dino.model_id, self.cfg.dino.device)
        self.detector = SingleShotDetector(self, self.cfg, self.predictor)

        # SAM2 for candidate snapshots (only needed when multi-candidate path triggers)
        self.sam2 = SAM2ImagePredictorWrapper(
            model_id=SAM2_ID,
            device=self.cfg.dino.device,
            mask_threshold=SAM2_MASK_THRESHOLD,
            max_hole_area=SAM2_MAX_HOLE_AREA,
            max_sprinkle_area=SAM2_MAX_SPRINKLE_AREA,
            multimask_output=False,
            return_logits=False,
        )

        # LightGlue matcher
        self._lg = _LGMatcher(device=self.cfg.dino.device, mp=True)
        self._aria_feats0: Optional[dict] = None
        self._aria_rgbmask_ready: bool = False

        # Trajectory / IK / joint_states
        self.motion = MotionContext(
            self,
            self.cfg.control.joint_order,
            self.cfg.control.vel_eps,
            self.cfg.control.require_stationary,
        )
        self.traj = TrajectoryPublisher(self, self.cfg.control.joint_order, self.cfg.control.controller_topic)
        self.ik = IKClient(
            self,
            self.cfg.control.group_name,
            self.cfg.control.ik_link_name,
            self.cfg.control.joint_order,
            self.cfg.control.ik_timeout,
            self.cfg.jump.ignore_joints,
            self.cfg.jump.max_safe_jump,
            self.cfg.jump.max_warn_jump,
        )

        # FSM state
        self._phase = "init_needed"
        self._inflight = False
        self._done = False

        # Offline pipeline state
        self._offline_ran = False

        # Stage-1 / Stage-2 cached data (legacy path)
        self._pose_stage1: Optional[PoseStamped] = None
        self._fixed_hover: Optional[PoseStamped] = None
        self._circle_center: Optional[np.ndarray] = None
        self._ring_z: Optional[float] = None

        # Circular path
        self._start_yaw: Optional[float] = None
        self._poly_wps: List[PoseStamped] = []
        self._poly_idx: int = 0
        self._poly_dwell_due_ns: Optional[int] = None
        self._skip_last_vertex = False

        # TF rebroadcast
        self._last_obj_tf: Optional[TransformStamped] = None
        self._last_circle_tf: Optional[TransformStamped] = None
        if self.cfg.control.tf_rebroadcast_hz > 0:
            self.create_timer(1.0 / self.cfg.control.tf_rebroadcast_hz, self._rebroadcast_tfs)

        # Final goto state
        self._final_point_base: Optional[np.ndarray] = None
        self._final_goto_requested = False

        # New: candidate selection state
        self._cand_list: List[_Candidate] = []
        self._cand_active_idx: int = -1
        self._cand_iter: int = 0
        self._cand_dwell_due_ns: Optional[int] = None
        self._cand_best_idx: int = -1

        # Main loop
        self.create_timer(0.05, self._tick)
        self._frame_count = 0

        # Debug print guards (avoid flooding)
        self._stage1_dino_printed = False
        self._route_printed_stage1 = False

        ob = self.cfg.offline_bias
        self.get_logger().info(
            f"[seeanything] topic={self.cfg.frames.image_topic}, hover={self.cfg.control.hover_above:.3f}m, "
            f"online_bias=({self.cfg.bias.bx:.3f},{self.cfg.bias.by:.3f},{self.cfg.bias.bz:.3f}); "
            f"offline_bias(enable={ob.enable}, ox={ob.ox:.3f}, oy={ob.oy:.3f}, oz={ob.oz:.3f}); "
            f"N={self.cfg.circle.n_vertices}, R={self.cfg.circle.radius:.3f}m, orient={self.cfg.circle.orient_mode}, "
            f"dir={self.cfg.circle.poly_dir}, sweep={self._sweep_deg:.1f}deg; "
            f"out_img_dir={self._img_dir}, out_json={self._js_path}; vggt_out_dir={OUTPUT_VGGT_DIR}"
        )

    # ---------- TF rebroadcast ----------
    def _rebroadcast_tfs(self):
        now = self.get_clock().now().to_msg()
        if self._last_obj_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_obj_tf.header.frame_id
            t.child_frame_id = self._last_obj_tf.child_frame_id
            t.transform = self._last_obj_tf.transform
            self.tf_brd.sendTransform(t)
        if self._last_circle_tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self._last_circle_tf.header.frame_id
            t.child_frame_id = self._last_circle_tf.child_frame_id
            t.transform = self._last_circle_tf.transform
            self.tf_brd.sendTransform(t)

    # ---------- helper: write logs ----------
    def _flush_joint_log(self):
        try:
            with open(self._js_path, "w", encoding="utf-8") as f:
                json.dump(self._joint_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to write joint log JSON: {e}")

    def _flush_match_log(self):
        try:
            _ensure_dir(MATCH_ROOT)
            with open(MATCH_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(self._match_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"Failed to write match log JSON: {e}")

    # ---------- helpers ----------
    def _get_tool_z_now(self) -> Optional[float]:
        """Read current tool Z in base frame."""
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                timeout=RclDuration(seconds=0.2)
            )
            _, p_bt = tfmsg_to_Rp(T_bt)
            return float(p_bt[2])
        except Exception as ex:
            self.get_logger().warn(f"Read tool Z failed: {ex}")
            return None

    def _get_tool_yaw_xy(self) -> Optional[float]:
        """Tool yaw around base Z (from tool x-axis projected to XY)."""
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, Time(),
                timeout=RclDuration(seconds=0.2)
            )
            R_bt, _ = tfmsg_to_Rp(T_bt)
            ex = R_bt[:, 0]
            return float(math.atan2(float(ex[1]), float(ex[0])))
        except Exception as ex:
            self.get_logger().warn(f"Read tool yaw failed: {ex}")
            return None

    # ---------- debug: stage-1 DINO summaries ----------
    def _run_dino_on_rgb_np(self, rgb_np: np.ndarray):
        """
        Run GroundingDINO once on an RGB uint8 image (H,W,3).
        Return boxes(list-like), labels(list-like), scores(np.ndarray float).
        """
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb_np)
        out = self.predictor.predict(
            pil, self.cfg.dino.text_prompt,
            box_threshold=self.cfg.dino.box_threshold,
            text_threshold=self.cfg.dino.text_threshold
        )
        if not isinstance(out, tuple) or len(out) < 2:
            return [], [], np.zeros((0,), dtype=np.float32)

        boxes, labels = out[:2]
        scores = out[2] if len(out) >= 3 else [None] * len(boxes)

        s = []
        for x in scores:
            if x is None:
                s.append(float("nan"))
            elif hasattr(x, "detach"):
                s.append(float(x.detach().cpu().item()))
            else:
                s.append(float(x))
        s = np.asarray(s, dtype=np.float32)
        return boxes, labels, s

    def _log_stage1_dino_detections(self, rgb_np: np.ndarray, tag: str = "Stage-1"):
        """
        Print a clear summary of DINO detections for the current prompt.
        """
        if not DEBUG_STAGE1_DINO_PRINT:
            return

        boxes, labels, s = self._run_dino_on_rgb_np(rgb_np)
        n_all = int(len(boxes))
        if n_all == 0:
            self.get_logger().info(
                f"[{tag}][DINO] prompt='{self.cfg.dino.text_prompt}' -> 0 detections "
                f"(box_th={self.cfg.dino.box_threshold:.2f}, text_th={self.cfg.dino.text_threshold:.2f})"
            )
            return

        min_exec = float(self.cfg.dino.min_exec_score)
        n_ge = int(np.sum(np.isfinite(s) & (s >= min_exec)))
        s_max = float(np.nanmax(s)) if np.any(np.isfinite(s)) else float("nan")

        order = np.argsort(-np.nan_to_num(s, nan=-1e9))
        topk = int(min(int(DEBUG_STAGE1_DINO_TOPK), n_all))

        self.get_logger().info(
            f"[{tag}][DINO] prompt='{self.cfg.dino.text_prompt}' | total={n_all} | "
            f">=min_exec({min_exec:.2f})={n_ge} | max={s_max:.3f} | "
            f"box_th={self.cfg.dino.box_threshold:.2f} text_th={self.cfg.dino.text_threshold:.2f}"
        )

        if not DEBUG_STAGE1_DINO_PRINT_BOXES:
            return

        for rank, bi in enumerate(order[:topk]):
            sc = float(s[bi]) if np.isfinite(s[bi]) else float("nan")
            b = boxes[bi].tolist() if hasattr(boxes[bi], "tolist") else boxes[bi]
            x0, y0, x1, y1 = [float(v) for v in b]
            u = 0.5 * (x0 + x1)
            v = 0.5 * (y0 + y1)
            lab = labels[bi] if isinstance(labels, (list, tuple)) and bi < len(labels) else ""
            self.get_logger().info(
                f"[{tag}][DINO] rank={rank:02d} score={sc:.3f} uv=({u:.1f},{v:.1f}) "
                f"box=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) label='{lab}'"
            )

    def _log_candidate_comparison_table(self, title: str, sort_by: str = "match"):
        """
        Print a compact comparison table of candidates.
        sort_by: 'match' or 'dino'
        """
        if not DEBUG_CAND_TABLE_PRINT:
            return
        if not self._cand_list:
            return

        if sort_by == "dino":
            idxs = sorted(range(len(self._cand_list)), key=lambda i: self._cand_list[i].score, reverse=True)
        else:
            idxs = sorted(range(len(self._cand_list)), key=lambda i: self._cand_list[i].match_score, reverse=True)

        self.get_logger().info(f"[cand][table] {title} (sorted_by={sort_by})")
        self.get_logger().info(
            "[cand][table] rank | dino_score | match_score | quality | coverage | Kconf | "
            "C_base(x,y,z) | hover_z"
        )
        for i in idxs:
            c = self._cand_list[i]
            self.get_logger().info(
                f"[cand][table] {i:4d} | {c.score:9.3f} | {c.match_score:10.3f} | "
                f"{c.match_quality:7.3f} | {c.match_coverage:8.3f} | {c.match_Kconf:5d} | "
                f"({c.C[0]:.3f},{c.C[1]:.3f},{c.C[2]:.3f}) | {c.hover_pose.pose.position.z:.3f}"
            )

    # ---------- camera pose at image stamp (unchanged) ----------
    def _calc_camera_pose_at_stamp(self, stamp_msg) -> Optional[Dict[str, Any]]:
        """
        Compute base->camera_optical pose strictly at the provided image stamp.
        NO fallback-to-latest. If TF is not available at this stamp, return None.
        """
        if stamp_msg is None:
            return None

        t_query = Time.from_msg(stamp_msg)

        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, t_query,
                timeout=RclDuration(seconds=0.3)
            )
        except Exception as ex:
            self.get_logger().warn(f"TF lookup at image stamp failed (no fallback): {ex}")
            return None

        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        qx, qy, qz, qw = self.cfg.cam.t_tool_cam_quat_xyzw
        R_t_cam = quat_to_rot(qx, qy, qz, qw)

        if str(self.cfg.cam.hand_eye_frame).lower() == "optical":
            R_t_co = R_t_cam
        else:
            R_t_co = R_t_cam @ R_CL_CO

        p_t_co = np.array(self.cfg.cam.t_tool_cam_xyz, dtype=float)

        R_bc = R_bt @ R_t_co
        p_bc = R_bt @ p_t_co + p_bt

        stamp_ns = _stamp_to_ns(stamp_msg)

        return {
            "parent_frame": self.cfg.frames.base_frame,
            "child_frame": "camera_optical",
            "stamp": {
                "sec": int(stamp_msg.sec),
                "nanosec": int(stamp_msg.nanosec),
                "stamp_ns": int(stamp_ns),
            },
            "R": R_bc.astype(float).round(12).tolist(),
            "t": [float(p_bc[0]), float(p_bc[1]), float(p_bc[2])]
        }

    # ---------- Atomic capture for VGGT shots (unchanged) ----------
    def _capture_and_log_shot(self, vertex_index_zero_based: int) -> bool:
        idx1 = int(vertex_index_zero_based) + 1
        key = f"image_{idx1}"
        fname = f"pose_{idx1}_image.png"
        fpath = os.path.join(self._img_dir, fname)

        with self._img_lock:
            if self._latest_bgr is None or self._latest_img_stamp is None:
                self.get_logger().warn("No camera frame/stamp available to capture.")
                return False
            bgr = self._latest_bgr.copy()
            stamp_msg = self._latest_img_stamp

        cam_pose = self._calc_camera_pose_at_stamp(stamp_msg)
        if cam_pose is None:
            self.get_logger().warn(f"[capture] Skip idx={idx1}: TF not available at this image stamp (no fallback).")
            return False

        js = self.motion._last_js
        if js is None or (not js.position) or (not js.name):
            self.get_logger().warn("[capture] No /joint_states available; skip shot.")
            return False

        if not self._joint_log["joint_names"]:
            self._joint_log["joint_names"] = list(js.name)

        joint_stamp_ns = -1
        try:
            joint_stamp_ns = _stamp_to_ns(js.header.stamp)
        except Exception:
            joint_stamp_ns = -1

        try:
            ok = cv2.imwrite(fpath, bgr)
            if not ok:
                self.get_logger().error(f"[capture] cv2.imwrite returned False for: {fpath}")
                return False
        except Exception as e:
            self.get_logger().error(f"[capture] Failed to save snapshot to {fpath}: {e}")
            return False

        entry = {
            "image_file": fname,
            "image_stamp_ns": int(cam_pose["stamp"].get("stamp_ns", -1)),
            "joint_stamp_ns": int(joint_stamp_ns),
            "position": [float(x) for x in js.position],
            "camera_pose": cam_pose,
        }

        self._joint_log["shots"][key] = entry
        self._flush_joint_log()
        self.get_logger().info(
            f"[capture] Saved idx={idx1}: {fpath} | img_stamp_ns={entry['image_stamp_ns']} | joint_stamp_ns={entry['joint_stamp_ns']}"
        )
        return True

    # ---------- Candidate snapshot + mask + match ----------
    def _capture_candidate_image(self, cand_rank0: int) -> Optional[Tuple[str, np.ndarray]]:
        """
        Capture latest frame (NOT VGGT) for candidate evaluation.
        """
        with self._img_lock:
            if self._latest_bgr is None:
                return None
            bgr = self._latest_bgr.copy()

        fname = f"cand_{cand_rank0+1:02d}_image.png"
        fpath = os.path.join(MATCH_CAND_DIR, fname)
        ok = cv2.imwrite(fpath, bgr)
        if not ok:
            return None
        return fpath, bgr

    def _dino_best_box_on_bgr(self, bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Run DINO on the candidate snapshot and pick best box by score.
        Return (xyxy_int, best_score).
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)

        out = self.predictor.predict(
            pil, self.cfg.dino.text_prompt,
            box_threshold=self.cfg.dino.box_threshold,
            text_threshold=self.cfg.dino.text_threshold
        )
        if not isinstance(out, tuple) or len(out) < 2:
            return None
        boxes, labels = out[:2]
        scores = out[2] if len(out) >= 3 else [None] * len(boxes)
        if len(boxes) == 0:
            return None

        s = np.array([float(x.detach().cpu().item()) if hasattr(x, "detach") else float(x) for x in scores], dtype=float)
        best = int(np.argmax(s))
        best_score = float(s[best]) if np.isfinite(s[best]) else -1.0

        x0, y0, x1, y1 = (boxes[best].tolist() if hasattr(boxes[best], "tolist") else boxes[best])
        h, w = bgr.shape[:2]
        x0 = int(max(0, min(w - 1, round(x0))))
        y0 = int(max(0, min(h - 1, round(y0))))
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)
        return (x0, y0, x1, y1), best_score

    def _sam2_rgbmask_for_snapshot(self, image_path: str, bgr: np.ndarray) -> Optional[str]:
        """
        Run DINO->SAM2 on the snapshot, build an object-only rgbmask (black bg) and save it.
        """
        best = self._dino_best_box_on_bgr(bgr)
        if best is None:
            return None
        (x0, y0, x1, y1), s_det = best

        res = self.sam2.run_inference(
            image_path=image_path,
            box=(x0, y0, x1, y1),
            save_dir=MATCH_CAND_DIR,
        )
        mask = res.get("mask_array", None)
        if mask is None or (not np.any(mask)):
            return None

        rgbmask = _build_rgbmask_blackbg(bgr, (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask)
        out_path = os.path.join(MATCH_CAND_DIR, Path(image_path).stem + "_rgbmask.png")
        cv2.imwrite(out_path, rgbmask)
        return out_path

    def _ensure_aria_rgbmask_ready(self) -> bool:
        """
        Build aria_rgbmask.png from the latest Aria outputs.
        Also precompute LightGlue features for Aria once.
        """
        if self._aria_rgbmask_ready and Path(ARIA_RGBMASK_PATH).exists() and self._aria_feats0 is not None:
            return True

        info = _save_aria_rgbmask(ARIA_OUT_DIR, ARIA_RGBMASK_PATH)
        if info is None:
            self.get_logger().warn(f"[aria] No valid Aria outputs found in: {ARIA_OUT_DIR}")
            return False

        self._match_log["aria_files"] = dict(info)
        self._flush_match_log()

        aria_bgr = cv2.imread(ARIA_RGBMASK_PATH, cv2.IMREAD_COLOR)
        if aria_bgr is None:
            return False

        aria_bgr_p = _preprocess_object_only(aria_bgr, out_size=512, pad_ratio=0.15)
        with torch.inference_mode():
            self._aria_feats0 = self._lg.extract(aria_bgr_p)

        self._aria_rgbmask_ready = True
        self.get_logger().info(f"[aria] Using aria_rgbmask: {ARIA_RGBMASK_PATH}")
        return True

    def _score_pair_lightglue(self, aria_rgbmask_path: str, wrist_rgbmask_path: str, save_viz_path: Optional[str] = None) -> Dict[str, Any]:
        aria_bgr = cv2.imread(aria_rgbmask_path, cv2.IMREAD_COLOR)
        wrist_bgr = cv2.imread(wrist_rgbmask_path, cv2.IMREAD_COLOR)
        if aria_bgr is None or wrist_bgr is None:
            return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0}

        aria_bgr = _preprocess_object_only(aria_bgr, out_size=512, pad_ratio=0.15)
        wrist_bgr = _preprocess_object_only(wrist_bgr, out_size=512, pad_ratio=0.15)

        with torch.inference_mode():
            feats0 = self._aria_feats0 if self._aria_feats0 is not None else self._lg.extract(aria_bgr)
            feats1 = self._lg.extract(wrist_bgr)
            out = self._lg.match(feats0, feats1)

        matches = out["matches"][0].detach().cpu().numpy().astype(np.int64)
        scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)

        if scores.size > 0:
            order = np.argsort(-scores)
            matches = matches[order]
            scores = scores[order]

        stats = _compute_object_match_score(scores, score_th=0.01, topk=10, k_ref=20)

        if save_viz_path is not None:
            try:
                kpts0 = feats0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
                kpts1 = feats1["keypoints"][0].detach().cpu().numpy().astype(np.float32)
                vis = _draw_matches_image(aria_bgr, wrist_bgr, kpts0, kpts1, matches, scores, topk=200)
                cv2.imwrite(save_viz_path, vis)
            except Exception:
                pass

        return stats

    # ---------- Multi-candidate detection at high stage ----------
    def _detect_candidates_from_msg(self, img_msg: Image) -> Optional[List[_Candidate]]:
        """
        Run DINO on current frame, compute plane intersections for each candidate box,
        and build hover poses (tool-Z-down) at z = C.z + hover_above.
        All geometry is consistent with your SingleShotDetector conventions.
        """
        try:
            rgb = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        except Exception:
            return None

        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb)

        out = self.predictor.predict(
            pil, self.cfg.dino.text_prompt,
            box_threshold=self.cfg.dino.box_threshold,
            text_threshold=self.cfg.dino.text_threshold
        )
        if not isinstance(out, tuple) or len(out) < 2:
            return None
        boxes, labels = out[:2]
        scores = out[2] if len(out) >= 3 else [None] * len(boxes)
        if len(boxes) == 0:
            return None

        s = np.array([float(x.detach().cpu().item()) if hasattr(x, "detach") else float(x) for x in scores], dtype=float)
        order = np.argsort(-s)  # desc

        # TF query time mode aligned with vision_geom.py
        t_query = Time.from_msg(img_msg.header.stamp) if self.cfg.control.tf_time_mode == "image" else Time()
        try:
            T_bt = self.tf_buffer.lookup_transform(
                self.cfg.frames.base_frame, self.cfg.frames.tool_frame, t_query,
                timeout=RclDuration(seconds=0.2)
            )
        except Exception as ex:
            self.get_logger().warn(f"[cand] TF lookup failed: {ex}")
            return None
        R_bt, p_bt = tfmsg_to_Rp(T_bt)

        # tool->camera_optical (same as SingleShotDetector)
        R_t_co = self.detector.R_t_co
        p_t_co = self.detector.p_t_co

        # camera pose in base
        R_bc = R_bt @ R_t_co
        p_bc = R_bt @ p_t_co + p_bt

        cand_list: List[_Candidate] = []
        min_s = float(self.cfg.dino.min_exec_score)

        for rank0, bi in enumerate(order[: int(max(1, CAND_MAX))]):
            sc = float(s[bi]) if np.isfinite(s[bi]) else -1.0
            if sc < min_s:
                continue

            x0, y0, x1, y1 = (boxes[bi].tolist() if hasattr(boxes[bi], "tolist") else boxes[bi])
            u = 0.5 * (float(x0) + float(x1))
            v = 0.5 * (float(y0) + float(y1))

            # pixel -> optical ray (reuse detector convention)
            d_opt = self.detector._pixel_to_dir_optical(u, v)  # normalized
            d_base = R_bc @ d_opt

            dz = float(d_base[2])
            if abs(dz) < 1e-6:
                continue

            t_star = (float(self.cfg.frames.z_virt) - float(p_bc[2])) / dz
            if t_star < 0:
                continue

            C = p_bc + t_star * d_base
            if self.cfg.bias.enable:
                C[0] += float(self.cfg.bias.bx)
                C[1] += float(self.cfg.bias.by)
                C[2] += float(self.cfg.bias.bz)

            # build hover pose (tool-Z-down)
            hover = PoseStamped()
            hover.header.frame_id = self.cfg.frames.pose_frame
            hover.header.stamp = self.get_clock().now().to_msg()
            hover.pose.position.x = float(C[0])
            hover.pose.position.y = float(C[1])
            hover.pose.position.z = float(C[2] + self.cfg.control.hover_above)
            hover.pose.orientation.w = 0.0
            hover.pose.orientation.x = 1.0
            hover.pose.orientation.y = 0.0
            hover.pose.orientation.z = 0.0

            cand_list.append(_Candidate(
                idx=int(rank0),
                score=sc,
                box_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                C=C,
                hover_pose=hover,
            ))

        if len(cand_list) == 0:
            if DEBUG_STAGE1_DINO_PRINT:
                self.get_logger().info(
                    f"[Stage-1][cand] after filter: kept=0 (min_exec_score={min_s:.2f}), raw_total={len(boxes)}"
                )
            return None

        if DEBUG_STAGE1_DINO_PRINT:
            self.get_logger().info(
                f"[Stage-1][cand] after filter: kept={len(cand_list)} (min_exec_score={min_s:.2f}), raw_total={len(boxes)}"
            )
            for i, c in enumerate(cand_list):
                self.get_logger().info(
                    f"[Stage-1][cand] rank={i} dino_score={c.score:.3f} "
                    f"C_base=({c.C[0]:.3f},{c.C[1]:.3f},{c.C[2]:.3f}) hover_z={c.hover_pose.pose.position.z:.3f} "
                    f"box=({c.box_xyxy[0]:.1f},{c.box_xyxy[1]:.1f},{c.box_xyxy[2]:.1f},{c.box_xyxy[3]:.1f})"
                )

        return cand_list

    # ---------- image callback ----------
    def _on_image(self, msg: Image):
        # Update rolling buffer + stamp atomically
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        with self._img_lock:
            self._latest_bgr = bgr
            self._latest_img_stamp = msg.header.stamp

        if self._done or self._inflight:
            return
        if self._phase not in ("wait_detect_stage1", "wait_detect_stage2"):
            return
        if not self.motion.is_stationary():
            return

        # Debug: print stage-1 DINO detections once for transparency
        if self._phase == "wait_detect_stage1" and (not self._stage1_dino_printed):
            try:
                rgb_dbg = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                self._log_stage1_dino_detections(rgb_dbg, tag="Stage-1")
                self._stage1_dino_printed = True
            except Exception:
                pass

        # Stride throttle
        self._frame_count += 1
        if self.cfg.runtime.frame_stride > 1 and (self._frame_count % self.cfg.runtime.frame_stride) != 0:
            return

        # ----------------------------
        # NEW: multi-candidate branch at stage-1
        # ----------------------------
        if self._phase == "wait_detect_stage1" and ENABLE_MULTI_CANDIDATE:
            cand_list = self._detect_candidates_from_msg(msg)

            if cand_list is not None and len(cand_list) >= 2:
                if DEBUG_ROUTE_PRINT and (not self._route_printed_stage1):
                    self.get_logger().info(
                        f"[route] ROUTE-2 (multi-candidate) triggered: candidates={len(cand_list)} >= 2 "
                        f"(min_exec_score={self.cfg.dino.min_exec_score:.2f}). Need Aria mask for verification."
                    )
                    self._route_printed_stage1 = True

                # Prepare aria rgbmask once
                if not self._ensure_aria_rgbmask_ready():
                    self.get_logger().warn("[route] ROUTE-2 aborted: Aria rgbmask not ready -> fallback to ROUTE-1 legacy.")
                else:
                    # Cache candidates and start sequential evaluation
                    self._cand_list = cand_list
                    self._cand_iter = 0
                    self._cand_active_idx = -1
                    self._cand_dwell_due_ns = None
                    self._cand_best_idx = -1

                    # Broadcast TF for the top candidate center (for RViz convenience)
                    topC = self._cand_list[0].C
                    tf_now = self.get_clock().now().to_msg()

                    tf_obj = TransformStamped()
                    tf_obj.header.stamp = tf_now
                    tf_obj.header.frame_id = self.cfg.frames.base_frame
                    tf_obj.child_frame_id = self.cfg.frames.object_frame
                    tf_obj.transform.translation.x = float(topC[0])
                    tf_obj.transform.translation.y = float(topC[1])
                    tf_obj.transform.translation.z = float(topC[2])
                    tf_obj.transform.rotation.w = 1.0
                    self.tf_brd.sendTransform(tf_obj)
                    self._last_obj_tf = tf_obj

                    tf_circle = TransformStamped()
                    tf_circle.header.stamp = tf_now
                    tf_circle.header.frame_id = self.cfg.frames.base_frame
                    tf_circle.child_frame_id = self.cfg.frames.circle_frame
                    tf_circle.transform.translation.x = float(topC[0])
                    tf_circle.transform.translation.y = float(topC[1])
                    tf_circle.transform.translation.z = float(topC[2])
                    tf_circle.transform.rotation.w = 1.0
                    self.tf_brd.sendTransform(tf_circle)
                    self._last_circle_tf = tf_circle

                    self.get_logger().info(
                        f"[cand] Detected {len(self._cand_list)} candidates (score>={self.cfg.dino.min_exec_score:.2f}). "
                        f"Start close-hover verification..."
                    )
                    self._log_candidate_comparison_table("Stage-1 candidates (initial, before matching)", sort_by="dino")

                    self._phase = "cand_move_next"
                    return
            else:
                if DEBUG_ROUTE_PRINT and (not self._route_printed_stage1):
                    n = 0 if cand_list is None else len(cand_list)
                    self.get_logger().info(
                        f"[route] ROUTE-1 (legacy) because candidates_above_min_exec < 2 (got {n})."
                    )
                    self._route_printed_stage1 = True
            # if not multi-candidate trigger, fall through to legacy stage-1

        # ----------------------------
        # Legacy path: use your existing detector.detect_once()
        # ----------------------------
        out = self.detector.detect_once(msg, self.tf_buffer)
        if out is None:
            return
        C, hover, tf_obj, tf_circle = out

        # Broadcast TF for RViz
        self.tf_brd.sendTransform(tf_obj)
        self._last_obj_tf = tf_obj
        self.tf_brd.sendTransform(tf_circle)
        self._last_circle_tf = tf_circle

        if self._phase == "wait_detect_stage1":
            # Stage-1: change XY to C, keep current Z
            z_keep = self._get_tool_z_now()
            if z_keep is None:
                self.get_logger().warn("Cannot read current Z; waiting next frame.")
                return
            ps = PoseStamped()
            ps.header.frame_id = self.cfg.frames.pose_frame
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(C[0])
            ps.pose.position.y = float(C[1])
            ps.pose.position.z = float(z_keep)
            ps.pose.orientation.w = 0.0
            ps.pose.orientation.x = 1.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            self._pose_stage1 = ps
            self.get_logger().info(f"[Stage-1] Move XY->({C[0]:.3f},{C[1]:.3f}), keep Z={z_keep:.3f}")

        elif self._phase == "wait_detect_stage2":
            # Stage-2: hover over C
            self._fixed_hover = hover
            self._circle_center = C.copy()
            self._ring_z = float(hover.pose.position.z)
            self.get_logger().info(f"[Stage-2] Move XY->({C[0]:.3f},{C[1]:.3f}), Z->{self._ring_z:.3f}")

    # ---------- Offline bias + final pose helpers (unchanged) ----------
    def _apply_offline_bias_to_object_json(self, obj_json_path: str, offset_xyz: np.ndarray) -> bool:
        try:
            if not obj_json_path or (not os.path.isfile(obj_json_path)):
                return False

            with open(obj_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            obj = data.get("object", {})
            if not isinstance(obj, dict):
                return False

            ox, oy, oz = float(offset_xyz[0]), float(offset_xyz[1]), float(offset_xyz[2])

            def _is_num(x) -> bool:
                return isinstance(x, (int, float))

            def _is_vec3(p) -> bool:
                return isinstance(p, (list, tuple)) and len(p) == 3 and all(_is_num(v) for v in p)

            def _is_list_vec3(ps) -> bool:
                return isinstance(ps, list) and len(ps) > 0 and all(_is_vec3(p) for p in ps)

            def _add3(p):
                return [float(p[0]) + ox, float(p[1]) + oy, float(p[2]) + oz]

            changed = False

            c = obj.get("center", None)
            if isinstance(c, dict) and _is_vec3(c.get("base_link", None)):
                c["base_link"] = _add3(c["base_link"])
                changed = True
            elif _is_vec3(obj.get("center_base", None)):
                obj["center_base"] = _add3(obj["center_base"])
                changed = True
            elif _is_vec3(obj.get("center_base_link", None)):
                obj["center_base_link"] = _add3(obj["center_base_link"])
                changed = True

            obb = obj.get("obb", None)
            if isinstance(obb, dict):
                c8 = obb.get("corners_8", None)
                if isinstance(c8, dict) and _is_list_vec3(c8.get("base_link", None)):
                    c8["base_link"] = [_add3(p) for p in c8["base_link"]]
                    changed = True

            def _walk_apply_points(node):
                nonlocal changed
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "base_link" and _is_list_vec3(v):
                            node[k] = [_add3(p) for p in v]
                            changed = True
                        else:
                            _walk_apply_points(v)
                elif isinstance(node, list):
                    for it in node:
                        _walk_apply_points(it)

            _walk_apply_points(obj)

            if not changed:
                return False

            with open(obj_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as ex:
            self.get_logger().warn(f"[offline] Failed to apply offline bias to object JSON: {ex}")
            return False

    def _read_center_from_object_json(self, obj_json_path: str) -> Optional[np.ndarray]:
        try:
            if not obj_json_path or (not os.path.isfile(obj_json_path)):
                return None
            with open(obj_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj = data.get("object", {})

            p = None
            if isinstance(obj, dict):
                c = obj.get("center", {})
                if isinstance(c, dict):
                    p = c.get("base_link", None)

            if p is None and isinstance(obj, dict):
                p = obj.get("center_base", None)
            if p is None and isinstance(obj, dict):
                p = obj.get("center_base_link", None)
            if p is None and isinstance(obj, dict):
                p = obj.get("center_b", None)

            if p is None or (not isinstance(p, (list, tuple))) or len(p) != 3:
                return None
            return np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=float).reshape(3)
        except Exception:
            return None

    def _build_final_hover_pose(self, center_b: np.ndarray) -> PoseStamped:
        x, y, z = float(center_b[0]), float(center_b[1]), float(center_b[2])
        z_hover = z + float(FINAL_GOTO_Z_OFFSET)
        z_hover = max(float(FINAL_GOTO_Z_MIN), min(float(FINAL_GOTO_Z_MAX), z_hover))

        ps = PoseStamped()
        ps.header.frame_id = self.cfg.frames.pose_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z_hover
        ps.pose.orientation.w = 0.0
        ps.pose.orientation.x = 1.0
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        return ps

    # ---------- Offline pipeline (unchanged) ----------
    def _run_offline_pipeline_once(self):
        if not RUN_OFFLINE_PIPELINE:
            self.get_logger().info("[offline] RUN_OFFLINE_PIPELINE=False, skipping VGGT + postprocess.")
            return
        if self._offline_ran:
            return

        self._offline_ran = True
        self._inflight = True

        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, "cameras_lines.ply"))
        _safe_remove(os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME))

        try:
            VGGTConfig, VGGTReconstructor, PostConfig, process_pointcloud = _import_offline_modules()
        except Exception as e:
            self.get_logger().error(f"[offline] Failed to import offline modules: {e}")
            self._inflight = False
            return

        # 1) VGGT reconstruction
        try:
            self.get_logger().info("[offline] Running VGGT reconstruction...")
            vcfg = VGGTConfig(
                images_dir=self._img_dir,
                out_dir=OUTPUT_VGGT_DIR,
                batch_size=VGGT_BATCH_SIZE,
                max_points=VGGT_MAX_POINTS,
                resolution=VGGT_RESOLUTION,
                conf_thresh=VGGT_CONF_THRESH,
                img_limit=None,
                auto_visualize=VGGT_AUTO_VISUALIZE,
                seed=42,
            )
            recon = VGGTReconstructor(vcfg)

            with torch.inference_mode():
                vout = recon.run()

            points_ply = vout.get("points_ply", os.path.join(OUTPUT_VGGT_DIR, "points.ply"))
            cameras_json = vout.get("cameras_json", os.path.join(OUTPUT_VGGT_DIR, "cameras.json"))
            self.get_logger().info(f"[offline] VGGT done: points_ply={points_ply}, cameras_json={cameras_json}")
        except Exception as e:
            self.get_logger().error(f"[offline] VGGT reconstruction failed: {e}")
            self._inflight = False
            return

        # 2) Post-process + align to base_link
        obj_json = os.path.join(OUTPUT_VGGT_DIR, OUTPUT_OBJECT_JSON_NAME)
        try:
            self.get_logger().info("[offline] Running point cloud postprocess + alignment to base_link...")
            pcfg = PostConfig(
                ply_path=points_ply,
                vggt_cameras_json=cameras_json,
                robot_shots_json=self._js_path,
                out_dir=OUTPUT_VGGT_DIR,
                visualize=POSTPROCESS_VISUALIZE,
                export_object_json=True,
                object_json_name=OUTPUT_OBJECT_JSON_NAME,
                align_method=POST_ALIGN_METHOD,
                vggt_pose_is_world_T_cam=POST_VGGT_POSE_IS_WORLD_T_CAM,
            )
            pres = process_pointcloud(pcfg)

            obj_json = pres.get("outputs", {}).get("object_json", obj_json)
            center_b = pres.get("center_b", None)

            center_np: Optional[np.ndarray] = None
            if center_b is not None:
                try:
                    center_np = np.asarray(center_b, dtype=float).reshape(3)
                except Exception:
                    center_np = None
            if center_np is None:
                center_np = self._read_center_from_object_json(obj_json)

            if bool(getattr(self.cfg, "offline_bias", None) and self.cfg.offline_bias.enable):
                ob = self.cfg.offline_bias
                offset = np.asarray([float(ob.ox), float(ob.oy), float(ob.oz)], dtype=float).reshape(3)

                changed = self._apply_offline_bias_to_object_json(obj_json, offset)
                if changed:
                    self.get_logger().info(
                        f"[offline] Applied offline_bias to object JSON: "
                        f"({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})"
                    )
                else:
                    self.get_logger().warn("[offline] Offline bias enabled, but object JSON was not modified (schema not matched).")

                center_np = self._read_center_from_object_json(obj_json)

            self._final_point_base = center_np

            if center_np is not None:
                self.get_logger().info(
                    f"[offline] Object center in base_link: ({center_np[0]:.4f}, {center_np[1]:.4f}, {center_np[2]:.4f})"
                )
            else:
                self.get_logger().warn("[offline] Object center not found (parse failed).")

            self.get_logger().info(f"[offline] Postprocess done. Exported: {obj_json}")
        except Exception as e:
            self.get_logger().error(f"[offline] Postprocess/alignment failed: {e}")
        finally:
            self._inflight = False

    # ---------- Final goto ----------
    def _request_final_goto(self):
        if not FINAL_GOTO_ENABLE:
            return
        if self._final_goto_requested:
            return
        if self._final_point_base is None:
            self.get_logger().warn("[final] No object center available; skipping final goto.")
            self._final_goto_requested = True
            return
        if not self.ik.ready():
            return

        pose = self._build_final_hover_pose(self._final_point_base)
        seed = self.motion.make_seed()
        if seed is None:
            self.get_logger().warn("[final] Waiting for /joint_states seed...")
            return

        self._final_goto_requested = True
        self._inflight = True
        self.get_logger().info(
            f"[final] Request IK for hover above object: "
            f"({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f})"
        )
        self.ik.request_async(pose, seed, self._on_final_ik)

    def _on_final_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("[final] IK failed/skipped for final goto. Exiting.")
            self._phase = "shutdown"
            return

        self.traj.publish_positions(joint_positions, float(FINAL_GOTO_MOVE_TIME))
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(float(FINAL_GOTO_MOVE_TIME), 0.3)
        self._phase = "final_moving"

    # ---------- Candidate selection helpers ----------
    def _select_best_candidate(self) -> int:
        """
        Choose candidate by highest match score. If best < accept, fallback to top DINO (rank-0).
        Also print a clear comparison summary.
        """
        if not self._cand_list:
            return 0

        # Print comparison table (by match) before decision
        self._log_candidate_comparison_table("Final comparison before selecting", sort_by="match")

        best_i = -1
        best_s = -1.0
        for i, c in enumerate(self._cand_list):
            if c.match_score > best_s:
                best_s = c.match_score
                best_i = i

        if best_i < 0:
            self.get_logger().warn("[cand] No valid match scores. Fallback to top DINO candidate (rank=0).")
            return 0

        if best_s < float(MATCH_SCORE_ACCEPT):
            self.get_logger().warn(
                f"[cand] Best match score {best_s:.3f} < MATCH_SCORE_ACCEPT {MATCH_SCORE_ACCEPT:.3f} "
                f"-> fallback to top DINO candidate (rank=0)."
            )
            return 0

        self.get_logger().info(
            f"[cand] Selected by matching: best_rank={best_i} best_match_score={best_s:.3f} "
            f"(MATCH_SCORE_ACCEPT={MATCH_SCORE_ACCEPT:.3f})"
        )
        return best_i

    def _log_candidate_result(self):
        self._match_log["candidates"] = []
        for i, c in enumerate(self._cand_list):
            self._match_log["candidates"].append({
                "rank": int(i),
                "dino_score_stage1": float(c.score),
                "box_xyxy_stage1": [float(x) for x in c.box_xyxy],
                "C_base": [float(c.C[0]), float(c.C[1]), float(c.C[2])],
                "hover_z": float(c.hover_pose.pose.position.z),
                "eval_image_path": c.eval_image_path,
                "eval_rgbmask_path": c.eval_rgbmask_path,
                "match_score": float(c.match_score),
                "match_quality": float(c.match_quality),
                "match_coverage": float(c.match_coverage),
                "match_Kconf": int(c.match_Kconf),
            })
        self._flush_match_log()

    # ---------- FSM ----------
    def _tick(self):
        if self._done:
            return

        if self._inflight and self._phase not in ("offline_pipeline",):
            return

        # INIT
        if self._phase == "init_needed":
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "init_moving"
            return

        if self._phase == "init_moving":
            if self.motion.is_stationary():
                self._phase = "wait_detect_stage1"
                self._stage1_dino_printed = False
                self._route_printed_stage1 = False
                self.get_logger().info("INIT reached. Waiting for Stage-1 detection...")
            return

        # ----------------------------
        # NEW: candidate selection phases
        # ----------------------------
        if self._phase == "cand_move_next":
            if self._cand_iter >= len(self._cand_list):
                # pick best and move there
                self._cand_best_idx = self._select_best_candidate()
                self._match_log["selected"] = {
                    "best_rank": int(self._cand_best_idx),
                    "best_match_score": float(self._cand_list[self._cand_best_idx].match_score),
                    "best_dino_score": float(self._cand_list[self._cand_best_idx].score),
                }
                self._log_candidate_result()
                self.get_logger().info(f"[cand] Selected best candidate rank={self._cand_best_idx}.")
                self._phase = "cand_move_best"
                return

            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("[cand] Waiting for /joint_states seed...")
                return

            self._cand_active_idx = int(self._cand_iter)
            pose = self._cand_list[self._cand_active_idx].hover_pose
            self._inflight = True
            self.ik.request_async(pose, seed, self._on_cand_ik)
            return

        if self._phase == "cand_moving":
            if not self.motion.is_stationary():
                return
            # start dwell window
            self._cand_dwell_due_ns = None
            self._phase = "cand_dwell"
            return

        if self._phase == "cand_dwell":
            if not self.motion.is_stationary():
                return
            now = self.get_clock().now().nanoseconds

            # On entry: capture + segment + match once
            if self._cand_dwell_due_ns is None:
                self._cand_dwell_due_ns = now + int(float(CAND_DWELL_SEC) * 1e9)
                idx = int(self._cand_active_idx)
                self.get_logger().info(f"[cand] At candidate rank={idx}, running snapshot+mask+match...")

                if not self._ensure_aria_rgbmask_ready():
                    self.get_logger().warn("[cand] Aria not ready; skipping match for this candidate.")
                else:
                    cap = self._capture_candidate_image(idx)
                    if cap is None:
                        self.get_logger().warn("[cand] Candidate image capture failed.")
                    else:
                        image_path, bgr = cap
                        self._cand_list[idx].eval_image_path = image_path

                        rgbmask_path = self._sam2_rgbmask_for_snapshot(image_path, bgr)
                        self._cand_list[idx].eval_rgbmask_path = rgbmask_path

                        if rgbmask_path is None:
                            self.get_logger().warn("[cand] SAM2 rgbmask failed (mask empty or detection missing).")
                        else:
                            stats = self._score_pair_lightglue(
                                ARIA_RGBMASK_PATH, rgbmask_path,
                                save_viz_path=None
                            )
                            self._cand_list[idx].match_score = float(stats["score"])
                            self._cand_list[idx].match_quality = float(stats["quality"])
                            self._cand_list[idx].match_coverage = float(stats["coverage"])
                            self._cand_list[idx].match_Kconf = int(stats["K_conf"])

                            dino_s = self._cand_list[idx].score if 0 <= idx < len(self._cand_list) else float("nan")
                            self.get_logger().info(
                                f"[cand] rank={idx} dino_score={dino_s:.3f} match_score={stats['score']:.3f} "
                                f"(qual={stats['quality']:.3f}, cov={stats['coverage']:.3f}, Kconf={stats['K_conf']})"
                            )

                # update json incrementally
                self._log_candidate_result()
                return

            if now < self._cand_dwell_due_ns:
                return

            # Next candidate
            self._cand_iter += 1
            self._phase = "cand_move_next"
            return

        if self._phase == "cand_move_best":
            if not self.ik.ready():
                return
            if self._cand_best_idx < 0 or self._cand_best_idx >= len(self._cand_list):
                self._cand_best_idx = 0
            seed = self.motion.make_seed()
            if seed is None:
                return
            self._inflight = True
            pose = self._cand_list[self._cand_best_idx].hover_pose
            self.ik.request_async(pose, seed, self._on_best_cand_ik)
            return

        if self._phase == "cand_best_moving":
            if not self.motion.is_stationary():
                return
            # finalize orbit center from selected candidate
            # bestC = self._cand_list[self._cand_best_idx].C
            # self._circle_center = bestC.copy()
            # self._ring_z = float(self._cand_list[self._cand_best_idx].hover_pose.pose.position.z)
            # self._phase = "hover_to_center"
            # self.get_logger().info(
            #     f"[cand] Best candidate hover reached. Orbit center=({bestC[0]:.3f},{bestC[1]:.3f},{bestC[2]:.3f})."
            self._fixed_hover = None
            self._circle_center = None
            self._ring_z = None
            
            self.get_logger().info("[cand] Best hover reached. Running Stage-2 refine detection before orbit...")
            self._phase = "wait_detect_stage2"

            
            # also save a best visualization for debugging
            try:
                rp = self._cand_list[self._cand_best_idx].eval_rgbmask_path
                if rp and Path(rp).exists() and self._ensure_aria_rgbmask_ready():
                    _ = self._score_pair_lightglue(ARIA_RGBMASK_PATH, rp, save_viz_path=BEST_VIZ_PATH)
                    self.get_logger().info(f"[cand] Saved best match visualization: {BEST_VIZ_PATH}")
            except Exception:
                pass
            return

        # ----------------------------
        # Legacy stage-1 move (XY only, keep Z)
        # ----------------------------
        if self._phase == "wait_detect_stage1" and self._pose_stage1 is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states ...")
                return
            self._inflight = True
            self.ik.request_async(self._pose_stage1, seed, self._on_ik_stage1)
            return

        if self._phase == "stage1_moving":
            if self.motion.is_stationary():
                self._pose_stage1 = None
                self._phase = "wait_detect_stage2"
                self.get_logger().info("Stage-1 done. Waiting for Stage-2 detection...")
            return

        # Legacy stage-2 move (hover over center)
        if self._phase == "wait_detect_stage2" and self._fixed_hover is not None:
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                self.get_logger().warn("Waiting for /joint_states ...")
                return
            self._inflight = True
            self.ik.request_async(self._fixed_hover, seed, self._on_hover_ik)
            return

        # Hover reached -> generate polygon
        if self._phase == "hover_to_center":
            if not self.motion.is_stationary():
                return

            self._start_yaw = self._get_tool_yaw_xy()
            if self._circle_center is not None and self._ring_z is not None and self._start_yaw is not None:
                all_wps = make_polygon_vertices(
                    self.get_clock().now().to_msg,
                    self._circle_center, self._ring_z, self._start_yaw,
                    self.cfg.frames.pose_frame,
                    self.cfg.circle.n_vertices, self.cfg.circle.num_turns, self.cfg.circle.poly_dir,
                    self.cfg.circle.orient_mode, self.cfg.circle.start_dir_offset_deg,
                    self.cfg.circle.radius, self.cfg.circle.tool_z_sign
                )

                total_deg = 360.0 * float(self.cfg.circle.num_turns)
                sweep_deg = max(0.0, min(float(self._sweep_deg), total_deg))
                if sweep_deg < total_deg and len(all_wps) > 1:
                    keep = max(1, int(math.floor(len(all_wps) * (sweep_deg / total_deg))))
                    keep = min(keep, len(all_wps))
                    self._poly_wps = all_wps[:keep]
                    self.get_logger().info(
                        f"Generated vertices: {len(all_wps)} -> trimmed to {len(self._poly_wps)} "
                        f"for sweep {sweep_deg:.1f}deg / {total_deg:.1f}deg."
                    )
                else:
                    self._poly_wps = all_wps
                    self.get_logger().info(f"Generated vertices: {len(self._poly_wps)} (full sweep).")

                self._poly_idx = 0
                self._phase = "poly_prepare"
            else:
                self._phase = "return_init"
            return

        if self._phase == "poly_prepare":
            if not self._poly_wps:
                self._phase = "return_init"
                return
            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                return
            self._inflight = True
            self.ik.request_async(self._poly_wps[0], seed, self._on_poly_ik)
            self._poly_idx = 1
            self._phase = "poly_moving"
            return

        if self._phase == "poly_moving":
            if self._skip_last_vertex:
                self._skip_last_vertex = False
                if self._poly_idx >= len(self._poly_wps):
                    self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                    self.motion.set_seed_hint(self.cfg.control.init_pos)
                    self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                    self._phase = "return_init"
                    return
                if not self.ik.ready():
                    return
                seed = self.motion.make_seed()
                if seed is None:
                    return
                self._inflight = True
                self.ik.request_async(self._poly_wps[self._poly_idx], seed, self._on_poly_ik)
                self._poly_idx += 1
                return

            now = self.get_clock().now().nanoseconds
            if not self.motion.is_stationary():
                return

            if self._poly_dwell_due_ns is None:
                self._poly_dwell_due_ns = now + int(self.cfg.circle.dwell_time * 1e9)
                curr_vertex0 = max(0, self._poly_idx - 1)
                self.get_logger().info(f"At vertex {curr_vertex0 + 1}, dwell for capture...")
                self._capture_and_log_shot(curr_vertex0)
                self._at_last_vertex = (self._poly_idx >= len(self._poly_wps))
                return

            if now < self._poly_dwell_due_ns:
                return
            self._poly_dwell_due_ns = None

            if getattr(self, "_at_last_vertex", False):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = "return_init"
                return

            if self._poly_idx >= len(self._poly_wps):
                self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
                self.motion.set_seed_hint(self.cfg.control.init_pos)
                self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
                self._phase = "return_init"
                return

            if not self.ik.ready():
                return
            seed = self.motion.make_seed()
            if seed is None:
                return
            self._inflight = True
            self.ik.request_async(self._poly_wps[self._poly_idx], seed, self._on_poly_ik)
            self._poly_idx += 1
            return

        # Return to INIT -> offline -> final goto
        if self._phase == "return_init":
            if not self.motion.is_stationary():
                return

            if RUN_OFFLINE_PIPELINE and not self._offline_ran:
                self._phase = "offline_pipeline"
                self.get_logger().info("Capture finished and INIT reached. Starting offline pipeline...")
                self._run_offline_pipeline_once()

                if FINAL_GOTO_ENABLE:
                    self._phase = "final_goto_needed"
                else:
                    self._phase = "shutdown"
                return

            self._phase = "shutdown"
            return

        if self._phase == "final_goto_needed":
            if not self.motion.is_stationary():
                return
            self._request_final_goto()
            return

        if self._phase == "final_moving":
            if not self.motion.is_stationary():
                return
            self.get_logger().info("[final] Final hover reached. Exiting.")
            self._phase = "shutdown"
            return

        if self._phase == "shutdown":
            self._done = True
            self.get_logger().info("All done. Exiting.")
            rclpy.shutdown()
            return

    # ---------- IK callbacks ----------
    def _on_ik_stage1(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("Stage-1 IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "return_init"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "stage1_moving"

    def _on_hover_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("Stage-2 hover IK skipped/failed. Returning to INIT and exiting.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "return_init"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "hover_to_center"

    def _on_poly_ik(self, joint_positions: Optional[List[float]]):
        if joint_positions is None:
            self.get_logger().warn("Vertex IK skipped after abnormal jump. Moving to next vertex.")
            self._inflight = False
            self._poly_dwell_due_ns = None
            self._skip_last_vertex = True
            return

        self.traj.publish_positions(joint_positions, self.cfg.circle.edge_move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.circle.edge_move_time, 0.3)
        self._inflight = False

    def _on_cand_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("[cand] IK failed/skipped for candidate. Mark as invalid and continue.")
            if 0 <= self._cand_active_idx < len(self._cand_list):
                self._cand_list[self._cand_active_idx].match_score = -1.0
            self._cand_iter += 1
            self._phase = "cand_move_next"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "cand_moving"

    def _on_best_cand_ik(self, joint_positions: Optional[List[float]]):
        self._inflight = False
        if joint_positions is None:
            self.get_logger().warn("[cand] IK failed/skipped for best candidate. Fallback to return_init.")
            self.traj.publish_init(self.cfg.control.init_pos, self.cfg.control.init_move_time)
            self.motion.set_seed_hint(self.cfg.control.init_pos)
            self.motion.set_motion_due(self.cfg.control.init_move_time, self.cfg.control.init_extra_wait)
            self._phase = "return_init"
            return

        self.traj.publish_positions(joint_positions, self.cfg.control.move_time)
        self.motion.set_seed_hint(joint_positions)
        self.motion.set_motion_due(self.cfg.control.move_time, 0.3)
        self._phase = "cand_best_moving"


def main():
    rclpy.init()
    node = SeeAnythingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
