#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
point_processing.py

Purpose
- Post-process a VGGT point cloud: remove table plane(s), extract the main object cluster,
  compute its OBB (center + 8 corners) in VGGT world.
- Align VGGT world to robot base_link using paired camera centers (VGGT cameras.json vs robot shots JSON).
- Export object center + 8 corners in base_link.

Key update (aim-ray gating)
- Instead of selecting the "largest cluster" or "closest to world origin",
  we can select the cluster that lies closest to the first camera optical ray
  (camera center + forward axis), which better matches the fact that you run DINO first
  and VGGT is anchored by the first view.

Inputs
- VGGT point cloud PLY (VGGT world)
- VGGT cameras.json (per-frame camera pose, typically world_T_cam)
- Robot shots JSON containing base_link->camera_optical pose (R,t) per image_k

Outputs
1) <out_dir>/points_no_table.ply
2) <out_dir>/main_cluster_clean.ply
3) <out_dir>/object_in_base_link.json

Visualization
- Main cluster (clean) + object center (blue sphere) + 8 OBB corners (small spheres)
- (optional) First camera frame + aim ray
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d


# =============================================================================
# Default parameters
# =============================================================================

# Paths
DEFAULT_PLY_PATH = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/offline_output/points.ply"
DEFAULT_VGGT_CAMERAS_JSON = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/offline_output/cameras.json"
DEFAULT_ROBOT_SHOTS_JSON = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ur5camerajointstates.json"

# Output
DEFAULT_OUT_DIR = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/postprocess_output"
DEFAULT_VISUALIZE = True
DEFAULT_EXPORT_OBJECT_JSON = True
DEFAULT_OBJECT_JSON_NAME = "object_in_base_link.json"

# Alignment: "sim3" estimates scale; "se3" forces scale=1
DEFAULT_ALIGN_METHOD = "sim3"  # "sim3" or "se3"
DEFAULT_VGGT_POSE_IS_WORLD_T_CAM = True

# Table removal (HSV-guided plane fitting on gray-ish points, but removal is gated)
DEFAULT_S_GRAY_CAND_MAX = 0.65
DEFAULT_V_GRAY_CAND_MIN = 0.35
DEFAULT_S_REMOVE_MAX = 0.55
DEFAULT_V_REMOVE_MIN = 0.25

# Auto scale (voxel)
DEFAULT_VOXEL_DIVISOR = 200.0
DEFAULT_VOXEL_MIN = 0.002
DEFAULT_VOXEL_MAX = 0.02

# RANSAC planes
DEFAULT_PLANE_DETECT_DIST_MULT = 0.80
DEFAULT_PLANE_REMOVE_DIST_MULT = 1.60
DEFAULT_RANSAC_N = 3
DEFAULT_RANSAC_ITERS = 2000
DEFAULT_MAX_PLANES = 5
DEFAULT_MIN_PLANE_POINTS_ABS = 5000
DEFAULT_MIN_PLANE_POINTS_RATIO = 0.001
DEFAULT_DS_VOXEL_MULT = 1.0

# Main cluster extraction
DEFAULT_MAIN_DBSCAN_EPS_MULT = 3.0
DEFAULT_MAIN_DBSCAN_MIN_POINTS = 50
DEFAULT_MAIN_GAP_MULT = 4.0

# In-cluster radius outlier removal
DEFAULT_INNER_RADIUS_MULT = 1.2
DEFAULT_INNER_RADIUS_MIN_POINTS = 12

# -----------------------------------------------------------------------------
# Center gate (legacy): choose cluster closest to a reference point (below defaults to origin)
# -----------------------------------------------------------------------------
DEFAULT_CENTER_GATE_ENABLE = False
DEFAULT_CENTER_REF_XYZ = (0.0, 0.0, 0.0)
DEFAULT_CENTER_GATE_RADIUS = 0.35
DEFAULT_CENTER_GATE_MIN_POINTS_ABS = 300
DEFAULT_CENTER_GATE_MIN_SIZE_RATIO = 0.00
DEFAULT_CENTER_PICK_MODE = "closest"  # "closest" or "closest_then_largest"

# -----------------------------------------------------------------------------
# NEW: Aim-ray gate (recommended): choose cluster closest to camera optical ray
# -----------------------------------------------------------------------------
DEFAULT_AIM_GATE_ENABLE = True
DEFAULT_AIM_USE_FIRST_POSE = True       # if True: use smallest pose_id in cameras.json
DEFAULT_AIM_POSE_ID = 1                 # used when AIM_USE_FIRST_POSE=False
DEFAULT_AIM_AXIS_CAM = (0.0, 0.0, 1.0)  # camera forward axis in cam frame (optical: +Z)
DEFAULT_AIM_INVERT_AXIS = False         # flip the axis if your camera convention is opposite
DEFAULT_AIM_PERP_RADIUS = 0.12          # meters; max perpendicular distance to ray
DEFAULT_AIM_MIN_DEPTH = 0.02            # meters; require centroid in front of camera by at least this
DEFAULT_AIM_MIN_POINTS_ABS = 300
DEFAULT_AIM_MIN_SIZE_RATIO = 0.00
# pick mode:
# - "min_perp_then_depth": choose smallest perpendicular distance; tie-break by smaller depth
# - "closest_depth": among candidates (within perp radius), choose smallest depth
DEFAULT_AIM_PICK_MODE = "min_perp_then_depth"

# visualization of aim
DEFAULT_SHOW_AIM_FRAME_AND_RAY = True
DEFAULT_AIM_FRAME_SIZE_MULT = 8.0       # axis size = mult * voxel (clamped)
DEFAULT_AIM_FRAME_SIZE_MIN = 0.03
DEFAULT_AIM_RAY_LENGTH_MULT = 80.0      # ray length = mult * voxel (clamped)
DEFAULT_AIM_RAY_LENGTH_MIN = 0.30
DEFAULT_AIM_RAY_COLOR = (0.0, 0.0, 1.0) # blue

# Visualization (center sphere size kept consistent with your previous logic)
DEFAULT_SHOW_CENTER = True
DEFAULT_SHOW_CORNERS = True
DEFAULT_CENTER_SPHERE_COLOR = (0.0, 0.2, 1.0)
DEFAULT_CENTER_SPHERE_RADIUS_MULT = 1.0
DEFAULT_CENTER_SPHERE_RADIUS_MIN = 0.004

DEFAULT_CORNER_SPHERE_COLOR = (1.0, 0.2, 0.2)
DEFAULT_CORNER_SPHERE_RADIUS_MULT = 0.75
DEFAULT_CORNER_SPHERE_RADIUS_MIN = 0.003


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Inputs
    ply_path: str = DEFAULT_PLY_PATH
    vggt_cameras_json: str = DEFAULT_VGGT_CAMERAS_JSON
    robot_shots_json: str = DEFAULT_ROBOT_SHOTS_JSON

    # Outputs
    out_dir: str = DEFAULT_OUT_DIR
    visualize: bool = DEFAULT_VISUALIZE
    export_object_json: bool = DEFAULT_EXPORT_OBJECT_JSON
    object_json_name: str = DEFAULT_OBJECT_JSON_NAME

    # Alignment
    align_method: str = DEFAULT_ALIGN_METHOD
    vggt_pose_is_world_T_cam: bool = DEFAULT_VGGT_POSE_IS_WORLD_T_CAM

    # Table removal
    s_gray_cand_max: float = DEFAULT_S_GRAY_CAND_MAX
    v_gray_cand_min: float = DEFAULT_V_GRAY_CAND_MIN
    s_remove_max: float = DEFAULT_S_REMOVE_MAX
    v_remove_min: float = DEFAULT_V_REMOVE_MIN

    # Auto voxel scale
    voxel_divisor: float = DEFAULT_VOXEL_DIVISOR
    voxel_min: float = DEFAULT_VOXEL_MIN
    voxel_max: float = DEFAULT_VOXEL_MAX

    # Plane fitting
    plane_detect_dist_mult: float = DEFAULT_PLANE_DETECT_DIST_MULT
    plane_remove_dist_mult: float = DEFAULT_PLANE_REMOVE_DIST_MULT
    ransac_n: int = DEFAULT_RANSAC_N
    ransac_iters: int = DEFAULT_RANSAC_ITERS
    max_planes: int = DEFAULT_MAX_PLANES
    min_plane_points_abs: int = DEFAULT_MIN_PLANE_POINTS_ABS
    min_plane_points_ratio: float = DEFAULT_MIN_PLANE_POINTS_RATIO
    ds_voxel_mult: float = DEFAULT_DS_VOXEL_MULT

    # Main cluster
    main_dbscan_eps_mult: float = DEFAULT_MAIN_DBSCAN_EPS_MULT
    main_dbscan_min_points: int = DEFAULT_MAIN_DBSCAN_MIN_POINTS
    main_gap_mult: float = DEFAULT_MAIN_GAP_MULT

    # In-cluster outlier removal
    inner_radius_mult: float = DEFAULT_INNER_RADIUS_MULT
    inner_radius_min_points: int = DEFAULT_INNER_RADIUS_MIN_POINTS

    # Center gate (legacy)
    center_gate_enable: bool = DEFAULT_CENTER_GATE_ENABLE
    center_ref_xyz: tuple = DEFAULT_CENTER_REF_XYZ
    center_gate_radius: float = DEFAULT_CENTER_GATE_RADIUS
    center_gate_min_points_abs: int = DEFAULT_CENTER_GATE_MIN_POINTS_ABS
    center_gate_min_size_ratio: float = DEFAULT_CENTER_GATE_MIN_SIZE_RATIO
    center_pick_mode: str = DEFAULT_CENTER_PICK_MODE

    # Aim-ray gate (recommended)
    aim_gate_enable: bool = DEFAULT_AIM_GATE_ENABLE
    aim_use_first_pose: bool = DEFAULT_AIM_USE_FIRST_POSE
    aim_pose_id: int = DEFAULT_AIM_POSE_ID
    aim_axis_cam: tuple = DEFAULT_AIM_AXIS_CAM
    aim_invert_axis: bool = DEFAULT_AIM_INVERT_AXIS
    aim_perp_radius: float = DEFAULT_AIM_PERP_RADIUS
    aim_min_depth: float = DEFAULT_AIM_MIN_DEPTH
    aim_min_points_abs: int = DEFAULT_AIM_MIN_POINTS_ABS
    aim_min_size_ratio: float = DEFAULT_AIM_MIN_SIZE_RATIO
    aim_pick_mode: str = DEFAULT_AIM_PICK_MODE

    show_aim_frame_and_ray: bool = DEFAULT_SHOW_AIM_FRAME_AND_RAY
    aim_frame_size_mult: float = DEFAULT_AIM_FRAME_SIZE_MULT
    aim_frame_size_min: float = DEFAULT_AIM_FRAME_SIZE_MIN
    aim_ray_length_mult: float = DEFAULT_AIM_RAY_LENGTH_MULT
    aim_ray_length_min: float = DEFAULT_AIM_RAY_LENGTH_MIN
    aim_ray_color: tuple = DEFAULT_AIM_RAY_COLOR

    # Visualization
    show_center: bool = DEFAULT_SHOW_CENTER
    show_corners: bool = DEFAULT_SHOW_CORNERS
    center_sphere_color: tuple = DEFAULT_CENTER_SPHERE_COLOR
    center_sphere_radius_mult: float = DEFAULT_CENTER_SPHERE_RADIUS_MULT
    center_sphere_radius_min: float = DEFAULT_CENTER_SPHERE_RADIUS_MIN
    corner_sphere_color: tuple = DEFAULT_CORNER_SPHERE_COLOR
    corner_sphere_radius_mult: float = DEFAULT_CORNER_SPHERE_RADIUS_MULT
    corner_sphere_radius_min: float = DEFAULT_CORNER_SPHERE_RADIUS_MIN


# =============================================================================
# Small utilities
# =============================================================================

POSE_ID_REGEX = re.compile(r"pose_(\d+)_image", re.IGNORECASE)

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def rgb01(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    if not pcd.has_colors():
        raise ValueError("Point cloud has no colors.")
    c = np.asarray(pcd.colors).astype(np.float32)
    if c.size == 0:
        raise ValueError("Point cloud color array is empty.")
    if c.max() > 1.0:
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin + 1e-12

    h = np.zeros_like(cmax)
    m = cmax == r
    h[m] = ((g - b)[m] / delta[m]) % 6
    m = cmax == g
    h[m] = ((b - r)[m] / delta[m]) + 2
    m = cmax == b
    h[m] = ((r - g)[m] / delta[m]) + 4
    h = (h / 6.0) % 1.0

    s = delta / (cmax + 1e-12)
    v = cmax
    return np.stack([h, s, v], axis=1)

def auto_voxel(pcd: o3d.geometry.PointCloud, divisor: float, vmin: float, vmax: float) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    max_extent = float(np.max(aabb.get_extent()))
    return float(np.clip(max_extent / divisor, vmin, vmax))

def plane_distance(plane: Tuple[float, float, float, float], pts: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    denom = math.sqrt(a*a + b*b + c*c) + 1e-12
    return np.abs(a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d) / denom

def dbscan_labels_points(pts3d: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts3d, dtype=np.float64))
    labels = pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    return np.asarray(labels, dtype=np.int32)

def remove_radius_outlier(pcd: o3d.geometry.PointCloud, radius: float, min_points: int) -> o3d.geometry.PointCloud:
    _, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    return pcd.select_by_index(ind)

def make_sphere(center_xyz: np.ndarray, radius: float, color_rgb: tuple) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    m.compute_vertex_normals()
    m.paint_uniform_color(np.asarray(color_rgb, dtype=float))
    m.translate(np.asarray(center_xyz, dtype=float).reshape(3))
    return m

def make_ray_lineset(origin: np.ndarray, direction: np.ndarray, length: float, color_rgb: tuple) -> o3d.geometry.LineSet:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        raise ValueError("Ray direction is near zero.")
    d = d / dn
    p1 = o
    p2 = o + d * float(length)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([p1, p2]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array([color_rgb], dtype=np.float64))
    return ls

def make_coord_frame_at_T(world_T: np.ndarray, size: float) -> o3d.geometry.TriangleMesh:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size), origin=[0.0, 0.0, 0.0])
    frame.transform(np.asarray(world_T, dtype=np.float64))
    return frame

def _as_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d*d, axis=1))))


# =============================================================================
# Alignment: Umeyama Sim3 / SE3
# =============================================================================

def load_vggt_world_T_cam(cameras_json_path: str, pose_is_world_T_cam: bool) -> Dict[int, np.ndarray]:
    """
    Returns {pose_id -> world_T_cam (4x4)}.

    Note:
    - Many VGGT exports use a key named "cam_T_world".
      Here we keep your convention:
        world_T_cam = M if pose_is_world_T_cam else inv(M)
    """
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cams = data["cameras"] if isinstance(data, dict) and "cameras" in data else data
    if not isinstance(cams, list):
        raise ValueError("cameras.json must be a list or a dict containing key 'cameras' as a list.")

    out: Dict[int, np.ndarray] = {}
    for c in cams:
        fname = c.get("file", "")
        m = POSE_ID_REGEX.search(fname)
        if not m:
            continue
        pid = int(m.group(1))

        M = np.array(c["cam_T_world"], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"Pose matrix must be 4x4, got {M.shape} for file={fname}")

        world_T_cam = M if pose_is_world_T_cam else _inv_T(M)
        out[pid] = world_T_cam

    if not out:
        raise RuntimeError(f"No pose_<id>_image entries parsed from {cameras_json_path}")
    return out

def load_robot_base_T_cam(robot_shots_json: str) -> Dict[int, np.ndarray]:
    """
    Returns {k -> base_T_cam (4x4)} for keys like "image_k".
    """
    with open(robot_shots_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = data.get("shots", {})
    out: Dict[int, np.ndarray] = {}
    for key, val in shots.items():
        m = re.match(r"image_(\d+)$", key)
        if not m:
            continue
        idx = int(m.group(1))
        cam_pose = val.get("camera_pose", {})
        R = np.array(cam_pose.get("R", np.eye(3)), dtype=np.float64)
        t = np.array(cam_pose.get("t", [0, 0, 0]), dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"Bad R shape at {key}: {R.shape}")
        out[idx] = _as_T(R, t)

    if not out:
        raise RuntimeError(f"No image_k camera_pose entries parsed from {robot_shots_json}")
    return out

def build_pairs_auto(
    vggt_world_T_cam: Dict[int, np.ndarray],
    robot_base_T_cam: Dict[int, np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Auto-match pose_id <-> image_k.
    Tries offsets off=0 and off=1 and keeps the one with most matches.
    Falls back to sorted pairing if no direct matches.
    """
    pose_ids = sorted(vggt_world_T_cam.keys())
    if not pose_ids:
        return []

    best_pairs: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    best_score = -1

    for off in [0, 1]:
        pairs = []
        for pid in pose_ids:
            k = pid + off
            if k in robot_base_T_cam:
                pairs.append((vggt_world_T_cam[pid], robot_base_T_cam[k], pid, k))
        if len(pairs) > best_score:
            best_score = len(pairs)
            best_pairs = pairs

    if best_score > 0:
        return best_pairs

    img_ids = sorted(robot_base_T_cam.keys())
    n = min(len(pose_ids), len(img_ids))
    return [(vggt_world_T_cam[pose_ids[i]], robot_base_T_cam[img_ids[i]], pose_ids[i], img_ids[i]) for i in range(n)]

def umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solve Y ~= s * R * X + t, with Umeyama.
    X, Y: (N,3)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my

    Sxy = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sxy)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_x = (Xc**2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / (var_x + 1e-12))
    else:
        s = 1.0

    t = my - s * (R @ mx)
    return s, R, t

def apply_sim3(P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return (s * (R @ P.T)).T + t.reshape(1, 3)

def robust_align_sim3(
    Cw: np.ndarray,
    Cb: np.ndarray,
    with_scale: bool,
    iters: int = 2,
    keep_frac: float = 0.85,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Light robust fitting:
    1) Fit on all
    2) Keep best keep_frac by residual and refit (repeat)
    Returns: s, R, t, per_pair_error, kept_indices
    """
    n = Cw.shape[0]
    idx = np.arange(n)

    s = 1.0
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    for _ in range(max(1, iters)):
        s, R, t = umeyama_sim3(Cw[idx], Cb[idx], with_scale=with_scale)
        Cb_hat = apply_sim3(Cw, s, R, t)
        e = np.linalg.norm(Cb_hat - Cb, axis=1)
        k = max(3, int(math.floor(n * keep_frac)))
        idx = np.argsort(e)[:k]

    Cb_hat = apply_sim3(Cw, s, R, t)
    e = np.linalg.norm(Cb_hat - Cb, axis=1)
    return s, R, t, e, idx


# =============================================================================
# Aim ray from camera pose
# =============================================================================

def compute_aim_ray_from_vggt(vggt_world_T_cam: Dict[int, np.ndarray], cfg: Config) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pose_id_used, world_T_cam, ray_origin_world, ray_dir_world).

    Ray:
      origin = camera center in world
      dir    = (R_wc @ axis_cam), normalized
    """
    if not vggt_world_T_cam:
        raise RuntimeError("Empty vggt_world_T_cam.")

    if cfg.aim_use_first_pose:
        pid = min(vggt_world_T_cam.keys())
    else:
        pid = int(cfg.aim_pose_id)
        if pid not in vggt_world_T_cam:
            # fallback to min
            pid = min(vggt_world_T_cam.keys())

    world_T_cam = np.asarray(vggt_world_T_cam[pid], dtype=np.float64)
    R_wc = world_T_cam[:3, :3]
    C_w = world_T_cam[:3, 3]

    axis_cam = np.asarray(cfg.aim_axis_cam, dtype=np.float64).reshape(3)
    if cfg.aim_invert_axis:
        axis_cam = -axis_cam

    d_w = R_wc @ axis_cam
    dn = np.linalg.norm(d_w)
    if dn < 1e-12:
        raise RuntimeError("Aim ray direction norm too small; check aim_axis_cam.")
    d_w = d_w / dn

    return pid, world_T_cam, C_w, d_w


# =============================================================================
# Table removal and main cluster extraction
# =============================================================================

def remove_table_only(pcd: o3d.geometry.PointCloud, cfg: Config) -> Tuple[o3d.geometry.PointCloud, float]:
    """
    Remove (gray-ish) planar table points via iterative RANSAC plane detection.
    No global "black removal" is applied (black objects are allowed).
    """
    voxel = auto_voxel(pcd, cfg.voxel_divisor, cfg.voxel_min, cfg.voxel_max)
    detect_thr = cfg.plane_detect_dist_mult * voxel
    remove_thr = cfg.plane_remove_dist_mult * voxel
    ds_voxel = max(1e-3, cfg.ds_voxel_mult * voxel)

    print(f"[Scale] voxel={voxel:.4f} m | detect_thr={detect_thr:.4f} m | remove_thr={remove_thr:.4f} m | ds_voxel={ds_voxel:.4f} m")

    C = rgb01(pcd)
    HSV = rgb_to_hsv_np(C)
    S, V = HSV[:, 1], HSV[:, 2]

    # Candidate points for fitting the plane (likely table)
    cand_mask = (S < cfg.s_gray_cand_max) & (V > cfg.v_gray_cand_min)
    print(f"[PlaneCandidates] grayish={int(cand_mask.sum())} / total={len(pcd.points)}")

    # Gating for actual removal (avoid deleting colorful object points)
    remove_color_gate = (S < cfg.s_remove_max) & (V > cfg.v_remove_min)

    keep_mask = np.ones((len(pcd.points),), dtype=bool)

    for it in range(cfg.max_planes):
        cur_idx = np.where(keep_mask & cand_mask)[0]
        if cur_idx.size == 0:
            print(f"[PlaneIter {it}] No candidates left, stop.")
            break

        pcd_cand = pcd.select_by_index(cur_idx)
        pcd_cand_ds = pcd_cand.voxel_down_sample(voxel_size=ds_voxel) if ds_voxel > 0 else pcd_cand
        if len(pcd_cand_ds.points) < 100:
            print(f"[PlaneIter {it}] Too few points ({len(pcd_cand_ds.points)}) for plane detection, stop.")
            break

        plane_model, _ = pcd_cand_ds.segment_plane(
            distance_threshold=detect_thr,
            ransac_n=cfg.ransac_n,
            num_iterations=cfg.ransac_iters,
        )

        pts_all = np.asarray(pcd.points)
        dist_all = plane_distance(plane_model, pts_all)

        remove_mask = (dist_all <= remove_thr) & remove_color_gate & keep_mask

        min_plane_pts = max(cfg.min_plane_points_abs, int(cfg.min_plane_points_ratio * keep_mask.sum()))
        rm_count = int(remove_mask.sum())

        print(f"[PlaneIter {it}] remove_count={rm_count} (min_required={min_plane_pts})")
        if rm_count < min_plane_pts:
            print(f"[PlaneIter {it}] Plane too small or gating too strict, stop.")
            break

        keep_mask &= ~remove_mask
        print(f"[PlaneIter {it}] removed={rm_count}, remaining={int(keep_mask.sum())}")

    pcd_no_table = pcd.select_by_index(np.where(keep_mask)[0])
    return pcd_no_table, voxel

def _choose_main_cluster_label(
    pts: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
    aim_origin_w: np.ndarray | None,
    aim_dir_w: np.ndarray | None,
) -> Tuple[int, List[Dict[str, object]]]:
    """
    Decide which DBSCAN cluster label to use as 'main cluster'.

    Priority:
      1) Aim-ray gate (if enabled and aim ray provided)
      2) Center gate (if enabled)
      3) Largest cluster (fallback)
    """
    valid = labels >= 0
    if not np.any(valid):
        raise RuntimeError("DBSCAN produced only noise (-1).")

    max_label = int(labels[valid].max())
    clusters: List[Dict[str, object]] = []

    for li in range(max_label + 1):
        idx = np.where(labels == li)[0]
        if idx.size == 0:
            continue
        size = int(idx.size)
        centroid = pts[idx].mean(axis=0)

        rec: Dict[str, object] = {
            "label": int(li),
            "size": int(size),
            "centroid": centroid,
        }

        if aim_origin_w is not None and aim_dir_w is not None:
            v = centroid - aim_origin_w
            depth = float(np.dot(v, aim_dir_w))
            perp_vec = v - depth * aim_dir_w
            perp = float(np.linalg.norm(perp_vec))
            rec["ray_depth"] = depth
            rec["ray_perp"] = perp
        else:
            rec["ray_depth"] = None
            rec["ray_perp"] = None

        # for legacy center gate
        ref = np.asarray(cfg.center_ref_xyz, dtype=np.float64).reshape(3)
        rec["dist_to_ref"] = float(np.linalg.norm(centroid - ref))

        clusters.append(rec)

    if not clusters:
        raise RuntimeError("No clusters found after filtering.")

    # Fallback: largest cluster
    clusters_sorted_by_size = sorted(clusters, key=lambda d: d["size"], reverse=True)
    default_label = int(clusters_sorted_by_size[0]["label"])
    max_size = int(clusters_sorted_by_size[0]["size"])

    # -------------------------
    # 1) Aim-ray gate (new)
    # -------------------------
    use_aim = bool(cfg.aim_gate_enable) and (aim_origin_w is not None) and (aim_dir_w is not None)
    if use_aim:
        perp_r = float(cfg.aim_perp_radius)
        min_depth = float(cfg.aim_min_depth)
        min_abs = int(cfg.aim_min_points_abs)
        min_ratio = float(cfg.aim_min_size_ratio)
        min_size = max(min_abs, int(math.floor(min_ratio * max_size)))

        candidates = []
        for d in clusters:
            if d["size"] < min_size:
                continue
            depth = d.get("ray_depth", None)
            perp = d.get("ray_perp", None)
            if depth is None or perp is None:
                continue
            if depth < min_depth:
                continue
            if perp > perp_r:
                continue
            candidates.append(d)

        if candidates:
            mode = str(cfg.aim_pick_mode).lower().strip()
            if mode == "closest_depth":
                # among candidates, pick smallest depth (closest along ray), tie-break by perp then size
                cand_sorted = sorted(candidates, key=lambda d: (d["ray_depth"], d["ray_perp"], -d["size"]))
            else:
                # default: smallest perp, tie-break by depth then size
                cand_sorted = sorted(candidates, key=lambda d: (d["ray_perp"], d["ray_depth"], -d["size"]))

            chosen = cand_sorted[0]
            chosen_label = int(chosen["label"])
            print(
                f"[AimGate] enabled: perp<= {perp_r:.3f}m, depth>= {min_depth:.3f}m, min_size={min_size} "
                f"-> chosen label={chosen_label}, size={chosen['size']}, "
                f"perp={chosen['ray_perp']:.4f}m, depth={chosen['ray_depth']:.4f}m "
                f"(fallback largest: label={default_label}, size={max_size})."
            )
            return chosen_label, clusters

        print(
            f"[AimGate] enabled, but no cluster passed: perp<= {perp_r:.3f}m, depth>= {min_depth:.3f}m, min_size={min_size} "
            f"-> fallback to other gates / largest(label={default_label})."
        )

    # -------------------------
    # 2) Center gate (legacy)
    # -------------------------
    use_center = bool(cfg.center_gate_enable) and (float(cfg.center_gate_radius) > 0.0)
    if use_center:
        gate_r = float(cfg.center_gate_radius)
        min_abs = int(cfg.center_gate_min_points_abs)
        min_ratio = float(cfg.center_gate_min_size_ratio)
        min_size = max(min_abs, int(math.floor(min_ratio * max_size)))

        candidates = [
            d for d in clusters
            if (d["size"] >= min_size) and (d["dist_to_ref"] <= gate_r)
        ]

        if candidates:
            candidates_sorted = sorted(candidates, key=lambda d: (d["dist_to_ref"], -d["size"]))
            chosen = candidates_sorted[0]
            chosen_label = int(chosen["label"])
            print(
                f"[CenterGate] enabled: radius={gate_r:.3f}m, min_size={min_size} "
                f"-> chosen label={chosen_label}, size={chosen['size']}, dist={chosen['dist_to_ref']:.4f}m "
                f"(fallback largest: label={default_label}, size={max_size})."
            )
            return chosen_label, clusters

        print(
            f"[CenterGate] enabled, but no cluster passed radius={gate_r:.3f}m, min_size={min_size} "
            f"-> fallback to largest(label={default_label})."
        )

    return default_label, clusters

def extract_main_cluster_clean(
    pcd_no_table: o3d.geometry.PointCloud,
    voxel: float,
    cfg: Config,
    aim_origin_w: np.ndarray | None = None,
    aim_dir_w: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """
    1) DBSCAN to find clusters
       - NEW: aim-ray gate can choose the cluster closest to camera optical ray.
    2) Keep near-main points and re-cluster
    3) Radius outlier removal inside the main cluster
    """
    pts = np.asarray(pcd_no_table.points)
    eps = cfg.main_dbscan_eps_mult * voxel

    labels = dbscan_labels_points(pts, eps=eps, min_points=cfg.main_dbscan_min_points)
    valid = labels >= 0
    num_clusters = int(labels[valid].max()) + 1 if np.any(valid) else 0
    if num_clusters <= 0:
        raise RuntimeError("No clusters found by DBSCAN.")

    main_label, clusters = _choose_main_cluster_label(pts, labels, cfg, aim_origin_w, aim_dir_w)

    # Debug summary
    if clusters:
        by_size = sorted(clusters, key=lambda d: d["size"], reverse=True)[:5]
        print(f"[Clusters] total={len(clusters)} | eps={eps:.4f} m | min_pts={cfg.main_dbscan_min_points}")
        msg = []
        for d in by_size:
            if d.get("ray_perp", None) is not None:
                msg.append(f"(L{d['label']}, n={d['size']}, perp={d['ray_perp']:.3f}, depth={d['ray_depth']:.3f})")
            else:
                msg.append(f"(L{d['label']}, n={d['size']}, dist={d['dist_to_ref']:.3f})")
        print("[Clusters-TopSize] " + " | ".join(msg))

    main_size = int(np.count_nonzero(labels == main_label))
    print(f"[MainCluster-1] chosen=(label={main_label}, size={main_size})")

    main_raw = pcd_no_table.select_by_index(np.where(labels == main_label)[0])

    gap = cfg.main_gap_mult * voxel
    dists = np.asarray(pcd_no_table.compute_point_cloud_distance(main_raw))
    near_main = pcd_no_table.select_by_index(np.where(dists <= gap)[0])
    print(f"[NearMain] gap={gap:.4f} m | kept={len(near_main.points)} / {len(pcd_no_table.points)}")

    pts2 = np.asarray(near_main.points)
    labels2 = dbscan_labels_points(pts2, eps=eps, min_points=cfg.main_dbscan_min_points)
    valid2 = labels2 >= 0
    num2 = int(labels2[valid2].max()) + 1 if np.any(valid2) else 0
    if num2 <= 0:
        raise RuntimeError("No clusters found in near-main region.")

    sizes2 = [(li, int(np.count_nonzero(labels2 == li))) for li in range(num2)]
    main2_label, main2_size = max(sizes2, key=lambda x: x[1])
    print(f"[MainCluster-2] clusters={num2} | main=(label={main2_label}, size={main2_size})")

    main2_raw = near_main.select_by_index(np.where(labels2 == main2_label)[0])

    r = cfg.inner_radius_mult * voxel
    main2_clean = remove_radius_outlier(main2_raw, radius=r, min_points=cfg.inner_radius_min_points)
    print(f"[InnerOutlier] radius={r:.4f} m | min_pts={cfg.inner_radius_min_points} | raw={len(main2_raw.points)} -> clean={len(main2_clean.points)}")

    if len(main2_clean.points) == 0:
        raise RuntimeError("Main cluster became empty after radius outlier removal.")
    return main2_clean


# =============================================================================
# Core pipeline
# =============================================================================

def process_pointcloud(cfg: Config) -> Dict[str, object]:
    """
    Main entry for other scripts to import and call.

    Returns a dict containing:
    - voxel
    - center_w, corners8_w
    - alignment (s, R, t, rmse_all, rmse_kept)
    - center_b, corners8_b
    - output paths
    """
    ensure_outdir(cfg.out_dir)

    pcd = o3d.io.read_point_cloud(cfg.ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty or unreadable point cloud: {cfg.ply_path}")
    print(f"[Load] {cfg.ply_path} | points={len(pcd.points)} | has_color={pcd.has_colors()}")

    # 1) Remove table
    pcd_no_table, voxel = remove_table_only(pcd, cfg)
    out_no_table = os.path.join(cfg.out_dir, "points_no_table.ply")
    o3d.io.write_point_cloud(out_no_table, pcd_no_table, write_ascii=False, compressed=True)
    print(f"[Write] {out_no_table}")

    # 1.5) Load VGGT camera poses once (for aim gate + later alignment)
    vggt_world_T_cam = load_vggt_world_T_cam(cfg.vggt_cameras_json, cfg.vggt_pose_is_world_T_cam)

    # compute aim ray (optional)
    aim_pid, aim_world_T_cam, aim_origin_w, aim_dir_w = None, None, None, None
    if cfg.aim_gate_enable or cfg.show_aim_frame_and_ray:
        pid, world_T_cam, C_w, d_w = compute_aim_ray_from_vggt(vggt_world_T_cam, cfg)
        aim_pid, aim_world_T_cam, aim_origin_w, aim_dir_w = pid, world_T_cam, C_w, d_w
        print(f"[AimRay] pose_id={aim_pid} | origin={aim_origin_w.tolist()} | dir={aim_dir_w.tolist()}")
        print("[AimRay] Note: Open3D frame color is X=red, Y=green, Z=blue. Ray uses camera (R_wc @ aim_axis_cam).")

    # 2) Extract main cluster + radius outlier removal (uses aim ray selection if enabled)
    main_clean = extract_main_cluster_clean(
        pcd_no_table, voxel, cfg,
        aim_origin_w=aim_origin_w if cfg.aim_gate_enable else None,
        aim_dir_w=aim_dir_w if cfg.aim_gate_enable else None,
    )
    out_main_clean = os.path.join(cfg.out_dir, "main_cluster_clean.ply")
    o3d.io.write_point_cloud(out_main_clean, main_clean, write_ascii=False, compressed=True)
    print(f"[Write] {out_main_clean}")

    # 3) OBB in VGGT world
    obb = main_clean.get_oriented_bounding_box()
    center_w = np.asarray(obb.center, dtype=np.float64).reshape(3)
    corners8_w = np.asarray(obb.get_box_points(), dtype=np.float64).reshape(8, 3)
    obb_extent = np.asarray(obb.extent, dtype=np.float64).reshape(3)
    obb_R_w = np.asarray(obb.R, dtype=np.float64)

    # 4) Align VGGT world -> base_link using paired camera centers
    robot_base_T_cam = load_robot_base_T_cam(cfg.robot_shots_json)
    pairs = build_pairs_auto(vggt_world_T_cam, robot_base_T_cam)

    if len(pairs) < 3:
        raise RuntimeError(f"Too few alignment pairs: {len(pairs)} (need >= 3, recommended >= 8)")

    Cw, Cb, pair_ids = [], [], []
    for world_T_cam_i, base_T_cam_i, pid_i, k_i in pairs:
        Cw.append(world_T_cam_i[:3, 3])
        Cb.append(base_T_cam_i[:3, 3])
        pair_ids.append({"pose_id": int(pid_i), "image_k": int(k_i)})

    Cw = np.stack(Cw, axis=0)
    Cb = np.stack(Cb, axis=0)

    with_scale = (cfg.align_method.lower() == "sim3")
    s, R, t, per_pair_e, keep_idx = robust_align_sim3(Cw, Cb, with_scale=with_scale, iters=2, keep_frac=0.85)

    Cb_hat = apply_sim3(Cw, s, R, t)
    err_all = rmse(Cb_hat, Cb)
    err_kept = rmse(Cb_hat[keep_idx], Cb[keep_idx])

    center_b = apply_sim3(center_w.reshape(1, 3), s, R, t).reshape(3)
    corners8_b = apply_sim3(corners8_w, s, R, t)

    # 5) Export JSON
    out_json_path = os.path.join(cfg.out_dir, cfg.object_json_name)
    if cfg.export_object_json:
        out = {
            "note": "Object center and OBB corners in VGGT world and base_link. Alignment is estimated from paired camera centers.",
            "inputs": {
                "ply_path": os.path.abspath(cfg.ply_path),
                "vggt_cameras_json": os.path.abspath(cfg.vggt_cameras_json),
                "robot_shots_json": os.path.abspath(cfg.robot_shots_json),
                "align_method": cfg.align_method,
                "vggt_pose_is_world_T_cam": bool(cfg.vggt_pose_is_world_T_cam),
                "aim_gate": {
                    "enable": bool(cfg.aim_gate_enable),
                    "pose_id_used": int(aim_pid) if aim_pid is not None else None,
                    "aim_axis_cam": list(cfg.aim_axis_cam),
                    "aim_invert_axis": bool(cfg.aim_invert_axis),
                    "perp_radius_m": float(cfg.aim_perp_radius),
                    "min_depth_m": float(cfg.aim_min_depth),
                    "min_points_abs": int(cfg.aim_min_points_abs),
                    "min_size_ratio": float(cfg.aim_min_size_ratio),
                    "pick_mode": str(cfg.aim_pick_mode),
                },
                "center_gate_legacy": {
                    "enable": bool(cfg.center_gate_enable),
                    "ref_xyz": list(cfg.center_ref_xyz),
                    "radius": float(cfg.center_gate_radius),
                    "min_points_abs": int(cfg.center_gate_min_points_abs),
                    "min_size_ratio": float(cfg.center_gate_min_size_ratio),
                    "pick_mode": str(cfg.center_pick_mode),
                },
            },
            "alignment_W_to_B": {
                "model": "p_B = s * R * p_W + t",
                "scale_s": float(s),
                "R_W_to_B": R.tolist(),
                "t_W_to_B": t.tolist(),
                "camera_center_rmse_all_m": float(err_all),
                "camera_center_rmse_kept_m": float(err_kept),
                "pairs_used_total": int(len(pairs)),
                "pairs_used_kept": int(len(keep_idx)),
                "pair_indices_total": pair_ids,
                "kept_indices": [int(i) for i in keep_idx.tolist()],
                "per_pair_error_m": [float(x) for x in per_pair_e.tolist()],
            },
            "object": {
                "center": {
                    "vggt_world": center_w.tolist(),
                    "base_link": center_b.tolist(),
                },
                "obb": {
                    "extent": obb_extent.tolist(),
                    "R_obb_in_vggt_world": obb_R_w.tolist(),
                    "corners_8": {
                        "index_order_note": "Order follows Open3D OrientedBoundingBox.get_box_points().",
                        "vggt_world": corners8_w.tolist(),
                        "base_link": corners8_b.tolist(),
                    },
                },
            },
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[Write] {os.path.abspath(out_json_path)}")
        print(f"[Align] rmse_all={err_all:.6f} m | rmse_kept={err_kept:.6f} m | scale={s:.6f}")

    # 6) Visualization
    if cfg.visualize:
        geoms: List[o3d.geometry.Geometry] = [main_clean]

        # object center & corners in VGGT world
        if cfg.show_center:
            center_r = max(cfg.center_sphere_radius_min, cfg.center_sphere_radius_mult * voxel)
            geoms.append(make_sphere(center_w, center_r, cfg.center_sphere_color))

        if cfg.show_corners:
            corner_r = max(cfg.corner_sphere_radius_min, cfg.corner_sphere_radius_mult * voxel)
            for p in corners8_w:
                geoms.append(make_sphere(p, corner_r, cfg.corner_sphere_color))

        # show aim frame + ray (camera pose in VGGT world)
        if cfg.show_aim_frame_and_ray and (aim_world_T_cam is not None) and (aim_origin_w is not None) and (aim_dir_w is not None):
            frame_size = max(cfg.aim_frame_size_min, cfg.aim_frame_size_mult * voxel)
            ray_len = max(cfg.aim_ray_length_min, cfg.aim_ray_length_mult * voxel)
            geoms.append(make_coord_frame_at_T(aim_world_T_cam, frame_size))
            geoms.append(make_ray_lineset(aim_origin_w, aim_dir_w, ray_len, cfg.aim_ray_color))

        o3d.visualization.draw_geometries(geoms)

    return {
        "voxel": float(voxel),
        "center_w": center_w,
        "corners8_w": corners8_w,
        "alignment": {
            "scale_s": float(s),
            "R_W_to_B": R,
            "t_W_to_B": t,
            "camera_center_rmse_all_m": float(err_all),
            "camera_center_rmse_kept_m": float(err_kept),
            "pairs_used_total": int(len(pairs)),
            "pairs_used_kept": int(len(keep_idx)),
        },
        "center_b": center_b,
        "corners8_b": corners8_b,
        "outputs": {
            "points_no_table_ply": os.path.abspath(out_no_table),
            "main_cluster_clean_ply": os.path.abspath(out_main_clean),
            "object_json": os.path.abspath(out_json_path),
        },
    }


# =============================================================================
# Debug / standalone run
# =============================================================================

def main():
    cfg = Config()
    process_pointcloud(cfg)

if __name__ == "__main__":
    main()
