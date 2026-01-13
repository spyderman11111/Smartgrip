#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pc_postprocess_and_align.py

输入：
- VGGT 输出点云 PLY（在 VGGT world 下）
- VGGT 输出 cameras.json（每帧相机位姿，通常为 world_T_cam）
- seeanything 保存的 image_jointstates.json（每帧 base_link->camera_optical 的 R,t）

输出：
- <out_dir>/points_no_table.ply
- <out_dir>/main_cluster_clean.ply
- <out_dir>/main_center.json（仅 VGGT world 下的中心信息）
- <out_dir>/object_in_base_link.json（新增：中心点 + 8角点，含变换前后坐标 + 对齐Sim3参数）
"""

import os, math, json, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d


# ===================== 可调参数 =====================
PLY_PATH = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/vggt_output/blue_cube_30_b30.ply"
VGGT_CAMERAS_JSON = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/vggt_output/cameras.json"

# 你保存的“每张图 base->camera_optical”的 JSON（你发的 shots 那个文件）
ROBOT_SHOTS_JSON = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/core/image_jointstates.json"

OUT_DIR  = "pc_out_simple"
VISUALIZE = True

# 对齐方式：sim3 会估尺度；se3 不估尺度（强制 s=1）
ALIGN_METHOD = "sim3"   # "sim3" or "se3"

# cameras.json 的 4x4 是否是 world_T_cam（通常 True）
VGGT_POSE_IS_WORLD_T_CAM = True

EXPORT_OBJECT_JSON = True
OBJECT_JSON_NAME = "object_in_base_link.json"

# 0) 全局清黑
V_BLACK_GLOBAL = 0.15

# 1) 颜色引导（候选，仅用于拟合平面，不直接删点）
S_GRAY_CAND_MAX = 0.65
V_GRAY_CAND_MIN = 0.35

# 1) 删除时颜色守门（防误删彩色）
S_REMOVE_MAX = 0.55
V_REMOVE_MIN = 0.25

# 尺度标尺（自适应）
VOXEL_DIVISOR = 200.0
VOXEL_MIN = 0.002
VOXEL_MAX = 0.02

# RANSAC 平面
PLANE_DETECT_DIST_MULT = 0.80
PLANE_REMOVE_DIST_MULT = 1.60
RANSAC_N = 3
RANSAC_ITERS = 2000
MAX_PLANES = 5
MIN_PLANE_POINTS_ABS = 5000
MIN_PLANE_POINTS_RATIO = 0.001
DS_VOXEL_MULT = 1.0

# 4) 主簇识别（DBSCAN）与“距离裁剪”
MAIN_DBSCAN_EPS_MULT = 3.0
MAIN_DBSCAN_MIN_POINTS = 50
MAIN_GAP_MULT = 4.0

# 5) 主簇内的离群剔除（半径法）
INNER_RADIUS_MULT = 1.2
INNER_RADIUS_MIN_POINTS = 12

# 6) [开关] 主簇中心的导出与可视化
ENABLE_CENTER = True
CENTER_POINT_COLOR = (0.0, 0.0, 1.0)
ENABLE_CENTER_SPHERE = True
CENTER_SPHERE_COLOR = (0.0, 0.2, 1.0)
CENTER_SPHERE_RADIUS_MULT = 1.0
CENTER_SPHERE_RADIUS_MIN  = 0.004
# ============================================================================


@dataclass
class Config:
    ply_path: str = PLY_PATH
    vggt_cameras_json: str = VGGT_CAMERAS_JSON
    robot_shots_json: str = ROBOT_SHOTS_JSON

    out_dir: str = OUT_DIR
    visualize: bool = VISUALIZE

    align_method: str = ALIGN_METHOD
    vggt_pose_is_world_T_cam: bool = VGGT_POSE_IS_WORLD_T_CAM
    export_object_json: bool = EXPORT_OBJECT_JSON
    object_json_name: str = OBJECT_JSON_NAME

    v_black_global: float = V_BLACK_GLOBAL
    s_gray_cand_max: float = S_GRAY_CAND_MAX
    v_gray_cand_min: float = V_GRAY_CAND_MIN
    s_remove_max: float = S_REMOVE_MAX
    v_remove_min: float = V_REMOVE_MIN

    voxel_divisor: float = VOXEL_DIVISOR
    voxel_min: float = VOXEL_MIN
    voxel_max: float = VOXEL_MAX

    plane_detect_dist_mult: float = PLANE_DETECT_DIST_MULT
    plane_remove_dist_mult: float = PLANE_REMOVE_DIST_MULT
    ransac_n: int = RANSAC_N
    ransac_iters: int = RANSAC_ITERS
    max_planes: int = MAX_PLANES
    min_plane_points_abs: int = MIN_PLANE_POINTS_ABS
    min_plane_points_ratio: float = MIN_PLANE_POINTS_RATIO
    ds_voxel_mult: float = DS_VOXEL_MULT

    main_dbscan_eps_mult: float = MAIN_DBSCAN_EPS_MULT
    main_dbscan_min_points: int = MAIN_DBSCAN_MIN_POINTS
    main_gap_mult: float = MAIN_GAP_MULT

    inner_radius_mult: float = INNER_RADIUS_MULT
    inner_radius_min_points: int = INNER_RADIUS_MIN_POINTS

    enable_center: bool = ENABLE_CENTER
    center_point_color: tuple = CENTER_POINT_COLOR
    enable_center_sphere: bool = ENABLE_CENTER_SPHERE
    center_sphere_color: tuple = CENTER_SPHERE_COLOR
    center_sphere_radius_mult: float = CENTER_SPHERE_RADIUS_MULT
    center_sphere_radius_min: float = CENTER_SPHERE_RADIUS_MIN


# ===================== 基础工具 =====================
def ensure_outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def rgb01(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    if not pcd.has_colors():
        raise ValueError("点云无颜色。")
    C = np.asarray(pcd.colors).astype(np.float32)
    if C.max() > 1.0:
        C = C / 255.0
    return np.clip(C, 0.0, 1.0)

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.max(rgb, axis=1); cmin = np.min(rgb, axis=1)
    delta = cmax - cmin + 1e-12
    h = np.zeros_like(cmax)
    m = cmax == r; h[m] = ((g - b)[m] / delta[m]) % 6
    m = cmax == g; h[m] = ((b - r)[m] / delta[m]) + 2
    m = cmax == b; h[m] = ((r - g)[m] / delta[m]) + 4
    h = (h / 6.0) % 1.0
    s = delta / (cmax + 1e-12); v = cmax
    return np.stack([h, s, v], axis=1)

def auto_voxel(pcd: o3d.geometry.PointCloud, divisor=200.0, vmin=0.002, vmax=0.02) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    max_extent = float(np.max(aabb.get_extent()))
    return float(np.clip(max_extent / divisor, vmin, vmax))

def plane_distance(plane, pts: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    denom = math.sqrt(a*a + b*b + c*c) + 1e-12
    return np.abs(a*pts[:, 0] + b*pts[:, 1] + c*pts[:, 2] + d) / denom

def dbscan_labels_points(pts3d: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts3d, dtype=np.float64))
    labels = pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    return np.asarray(labels)

def remove_radius_outlier(pcd: o3d.geometry.PointCloud, radius: float, min_points: int) -> o3d.geometry.PointCloud:
    _, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    return pcd.select_by_index(ind)

def single_point_pcd(pt, color=(0,0,1)):
    p = np.asarray(pt, dtype=np.float64).reshape(1,3)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(p))
    col = np.asarray(color, dtype=np.float64).reshape(1,3)
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc

def make_center_sphere(center, radius, color=(0,0,1)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.asarray(color, dtype=float))
    mesh.translate(np.asarray(center, dtype=float))
    return mesh

def _as_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3]  = np.asarray(t, dtype=np.float64).reshape(3)
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


# ===================== 读取与对齐（Sim3 / SE3） =====================
POSE_ID_REGEX = re.compile(r"pose_(\d+)_image", re.IGNORECASE)

def load_vggt_world_T_cam(cameras_json_path: str, pose_is_world_T_cam: bool) -> Dict[int, np.ndarray]:
    """
    返回 {pose_id:int -> world_T_cam(4x4)}
    兼容 cameras.json 是 list 或 dict（含 key "cameras"）
    """
    with open(cameras_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cams = data["cameras"] if isinstance(data, dict) and "cameras" in data else data
    if not isinstance(cams, list):
        raise ValueError("cameras.json 格式不符合预期（应为 list 或含 cameras:list 的 dict）。")

    out: Dict[int, np.ndarray] = {}
    for c in cams:
        fname = c.get("file", "")
        m = POSE_ID_REGEX.search(fname)
        if not m:
            continue
        pid = int(m.group(1))

        M = np.array(c["cam_T_world"], dtype=np.float64)
        if M.shape != (4,4):
            raise ValueError(f"cameras.json pose matrix must be 4x4, got {M.shape} for {fname}")

        # 注意：字段名 cam_T_world 但我们按“world_T_cam”使用（与你之前可视化 transform 逻辑一致）
        world_T_cam = M if pose_is_world_T_cam else _inv_T(M)
        out[pid] = world_T_cam

    if not out:
        raise RuntimeError(f"未能从 {cameras_json_path} 解析出任何 pose_<id>_image 对应条目。")
    return out

def load_robot_base_T_cam(robot_shots_json: str) -> Dict[int, np.ndarray]:
    """
    返回 {k:int -> base_T_cam(4x4)}，k 对应 image_k
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
        t = np.array(cam_pose.get("t", [0,0,0]), dtype=np.float64)
        if R.shape != (3,3):
            raise ValueError(f"robot shots: bad R shape at {key}: {R.shape}")
        out[idx] = _as_T(R, t)

    if not out:
        raise RuntimeError(f"未能从 {robot_shots_json} 解析出任何 image_k 的 camera_pose。")
    return out

def build_pairs_auto(
    vggt_world_T_cam: Dict[int, np.ndarray],
    robot_base_T_cam: Dict[int, np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    自动匹配：pose_id <-> image_k
    尝试 off=0 和 off=1，取匹配对最多的。
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

    # 兜底：按排序顺序强行配
    img_ids = sorted(robot_base_T_cam.keys())
    n = min(len(pose_ids), len(img_ids))
    pairs = []
    for i in range(n):
        pid = pose_ids[i]
        k = img_ids[i]
        pairs.append((vggt_world_T_cam[pid], robot_base_T_cam[k], pid, k))
    return pairs

def umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama: find s,R,t so that Y ~= s R X + t
    X,Y: (N,3)
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
    return (s * (R @ P.T)).T + t.reshape(1,3)

def robust_align_sim3(Cw: np.ndarray, Cb: np.ndarray, with_scale: bool, iters: int = 2, keep_frac: float = 0.85):
    """
    轻量鲁棒：先拟合，再按误差保留前 keep_frac，再拟合
    """
    n = Cw.shape[0]
    idx = np.arange(n)
    s = 1.0; R = np.eye(3); t = np.zeros(3)
    for _ in range(max(1, iters)):
        s, R, t = umeyama_sim3(Cw[idx], Cb[idx], with_scale=with_scale)
        Cb_hat = apply_sim3(Cw, s, R, t)
        e = np.linalg.norm(Cb_hat - Cb, axis=1)
        k = max(3, int(math.floor(n * keep_frac)))
        idx = np.argsort(e)[:k]
    Cb_hat = apply_sim3(Cw, s, R, t)
    e = np.linalg.norm(Cb_hat - Cb, axis=1)
    return s, R, t, e, idx


# ===================== 桌面剔除 =====================
def remove_table_only(pcd: o3d.geometry.PointCloud, cfg: Config) -> Tuple[o3d.geometry.PointCloud, float]:
    C = rgb01(pcd); HSV = rgb_to_hsv_np(C); S, V = HSV[:, 1], HSV[:, 2]
    keep_black = V >= cfg.v_black_global
    if keep_black.sum() < len(V):
        print(f"[Black-Remove] removed={int((~keep_black).sum())}, kept={int(keep_black.sum())}")
    pcd = pcd.select_by_index(np.where(keep_black)[0])

    voxel = auto_voxel(pcd, cfg.voxel_divisor, cfg.voxel_min, cfg.voxel_max)
    detect_thr = cfg.plane_detect_dist_mult * voxel
    remove_thr = cfg.plane_remove_dist_mult * voxel
    ds_voxel = max(1e-3, cfg.ds_voxel_mult * voxel)
    print(f"[Scale] voxel={voxel:.4f}m  detect_thr={detect_thr:.4f}m  remove_thr={remove_thr:.4f}m  ds_voxel={ds_voxel:.4f}m")

    C = rgb01(pcd); HSV = rgb_to_hsv_np(C); S, V = HSV[:, 1], HSV[:, 2]
    cand_mask = (S < cfg.s_gray_cand_max) & (V > cfg.v_gray_cand_min)
    print(f"[Candidates] grayish={int(cand_mask.sum())} / total={len(pcd.points)}")

    remove_color_gate = (S < cfg.s_remove_max) & (V > cfg.v_remove_min)
    keep_mask = np.ones((len(pcd.points),), dtype=bool)

    for it in range(cfg.max_planes):
        cur_idx = np.where(keep_mask & cand_mask)[0]
        if cur_idx.size == 0:
            print(f"[Iter {it}] no candidate left, stop.")
            break

        pcd_cand = pcd.select_by_index(cur_idx)
        pcd_cand_ds = pcd_cand.voxel_down_sample(voxel_size=ds_voxel) if ds_voxel > 0 else pcd_cand
        if len(pcd_cand_ds.points) < 100:
            print(f"[Iter {it}] too few points ({len(pcd_cand_ds.points)}) for plane detection, stop.")
            break

        plane_model, _ = pcd_cand_ds.segment_plane(distance_threshold=detect_thr,
                                                   ransac_n=cfg.ransac_n,
                                                   num_iterations=cfg.ransac_iters)
        pts_all = np.asarray(pcd.points)
        dist_all = plane_distance(plane_model, pts_all)
        remove_mask = (dist_all <= remove_thr) & remove_color_gate & keep_mask

        min_plane_pts = max(cfg.min_plane_points_abs, int(cfg.min_plane_points_ratio * keep_mask.sum()))
        rm_count = int(remove_mask.sum())
        print(f"[Iter {it}] remove candidates = {rm_count} (min required {min_plane_pts})")
        if rm_count < min_plane_pts:
            print(f"[Iter {it}] plane too small or not gray enough, stop.")
            break

        keep_mask &= ~remove_mask
        print(f"[Iter {it}] removed={rm_count}, remaining={int(keep_mask.sum())}")

    pcd_no_table = pcd.select_by_index(np.where(keep_mask)[0])
    return pcd_no_table, voxel


# ===================== 主簇提取 =====================
def keep_near_main_and_clean(pcd_no_table: o3d.geometry.PointCloud, voxel: float, cfg: Config):
    pts = np.asarray(pcd_no_table.points)
    eps = cfg.main_dbscan_eps_mult * voxel
    labels = dbscan_labels_points(pts, eps=eps, min_points=cfg.main_dbscan_min_points)
    num_clusters = int(labels.max()) + 1 if labels.size > 0 else 0
    if num_clusters == 0:
        raise RuntimeError("未找到任何簇（DBSCAN）。")
    sizes = [(li, int(np.count_nonzero(labels == li))) for li in range(num_clusters)]
    main_label, main_size = max(sizes, key=lambda x: x[1])
    print(f"[MainCluster] eps={eps:.4f}m  min_pts={cfg.main_dbscan_min_points} -> clusters={num_clusters}, main=(label={main_label}, size={main_size})")

    main_raw = pcd_no_table.select_by_index(np.where(labels == main_label)[0])

    gap = cfg.main_gap_mult * voxel
    dists = np.asarray(pcd_no_table.compute_point_cloud_distance(main_raw))
    near_main = pcd_no_table.select_by_index(np.where(dists <= gap)[0])
    print(f"[NearMain] gap={gap:.4f}m  kept={len(near_main.points)} / {len(pcd_no_table.points)}")

    pts2 = np.asarray(near_main.points)
    labels2 = dbscan_labels_points(pts2, eps=eps, min_points=cfg.main_dbscan_min_points)
    num2 = int(labels2.max()) + 1 if labels2.size > 0 else 0
    if num2 == 0:
        raise RuntimeError("Near-main 范围内未找到簇。")
    sizes2 = [(li, int(np.count_nonzero(labels2 == li))) for li in range(num2)]
    main2_label, _ = max(sizes2, key=lambda x: x[1])
    main2_raw = near_main.select_by_index(np.where(labels2 == main2_label)[0])

    r = cfg.inner_radius_mult * voxel
    main2_clean = remove_radius_outlier(main2_raw, radius=r, min_points=cfg.inner_radius_min_points)
    print(f"[InnerOutlier] radius={r:.4f}m  min_pts={cfg.inner_radius_min_points}  main_raw={len(main2_raw.points)} -> clean={len(main2_clean.points)}")

    return near_main, main2_raw, main2_clean


# ===================== 主入口 =====================
def run(cfg: Config):
    ensure_outdir(cfg.out_dir)

    pcd = o3d.io.read_point_cloud(cfg.ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"点云为空或无法读取: {cfg.ply_path}")
    print(f"[Load] {cfg.ply_path}  points={np.asarray(pcd.points).shape[0]}  has_color={pcd.has_colors()}")

    # 去桌面
    pcd_no_table, voxel = remove_table_only(pcd, cfg)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "points_no_table.ply"),
                             pcd_no_table, write_ascii=False, compressed=True)

    # 主簇
    near_main, main_raw, main_clean = keep_near_main_and_clean(pcd_no_table, voxel, cfg)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "near_main_raw.ply"),
                             near_main, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_cluster_raw.ply"),
                             main_raw, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_cluster_clean.ply"),
                             main_clean, write_ascii=False, compressed=True)

    # OBB 几何（VGGT world）
    center_pc = None
    center_sphere = None

    center_w: Optional[np.ndarray] = None
    corners8_w: Optional[np.ndarray] = None
    obb_extent: Optional[np.ndarray] = None
    obb_R_w: Optional[np.ndarray] = None

    if cfg.enable_center and len(main_clean.points) > 0:
        obb = main_clean.get_oriented_bounding_box()
        center_w = np.asarray(obb.center, dtype=np.float64).reshape(3)
        corners8_w = np.asarray(obb.get_box_points(), dtype=np.float64).reshape(8, 3)
        obb_extent = np.asarray(obb.extent, dtype=np.float64).reshape(3)
        obb_R_w = np.asarray(obb.R, dtype=np.float64)

        # 中心点可视化
        center_pc = single_point_pcd(center_w.tolist(), color=cfg.center_point_color)
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_center.ply"),
                                 center_pc, write_ascii=False, compressed=True)

        if cfg.enable_center_sphere:
            r = max(cfg.center_sphere_radius_min, cfg.center_sphere_radius_mult * voxel)
            center_sphere = make_center_sphere(center_w.tolist(), radius=r, color=cfg.center_sphere_color)
            o3d.io.write_triangle_mesh(os.path.join(cfg.out_dir, "main_center_sphere.ply"),
                                       center_sphere, write_ascii=False)
            print(f"[CenterSphere] radius={r:.4f} m  mesh=main_center_sphere.ply")

        # 旧 JSON（仅 VGGT world）
        info = {
            "input_ply": os.path.abspath(cfg.ply_path),
            "voxel": float(voxel),
            "center": center_w.tolist(),
            "obb_extent": obb_extent.tolist(),
            "counts": {
                "points_no_table": len(pcd_no_table.points),
                "near_main_raw": len(near_main.points),
                "main_cluster_raw": len(main_raw.points),
                "main_cluster_clean": len(main_clean.points),
            }
        }
        with open(os.path.join(cfg.out_dir, "main_center.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[Center] center(VGGT world)={center_w.tolist()}  -> main_center.json")

    # ===================== 对齐并导出 base_link 坐标（中心 + 8角点） =====================
    if cfg.export_object_json and (center_w is not None) and (corners8_w is not None):
        vggt_world_T_cam = load_vggt_world_T_cam(cfg.vggt_cameras_json, cfg.vggt_pose_is_world_T_cam)
        robot_base_T_cam = load_robot_base_T_cam(cfg.robot_shots_json)
        pairs = build_pairs_auto(vggt_world_T_cam, robot_base_T_cam)

        if len(pairs) < 3:
            raise RuntimeError(f"对齐匹配对太少：{len(pairs)}（建议 >= 8）")

        Cw = []
        Cb = []
        pair_ids = []
        for world_T_cam, base_T_cam, pid, k in pairs:
            Cw.append(world_T_cam[:3, 3])
            Cb.append(base_T_cam[:3, 3])
            pair_ids.append({"pose_id": int(pid), "image_k": int(k)})

        Cw = np.stack(Cw, axis=0)
        Cb = np.stack(Cb, axis=0)

        with_scale = (cfg.align_method.lower() == "sim3")
        s, R, t, e_all, keep_idx = robust_align_sim3(Cw, Cb, with_scale=with_scale, iters=2, keep_frac=0.85)
        Cb_hat = apply_sim3(Cw, s, R, t)
        err_all = rmse(Cb_hat, Cb)
        err_keep = rmse(Cb_hat[keep_idx], Cb[keep_idx])

        center_b   = apply_sim3(center_w.reshape(1,3), s, R, t).reshape(3)
        corners8_b = apply_sim3(corners8_w, s, R, t)

        out = {
            "note": "Object center + OBB corners in VGGT world and base_link. Alignment is estimated from paired camera centers.",
            "inputs": {
                "ply_path": os.path.abspath(cfg.ply_path),
                "vggt_cameras_json": os.path.abspath(cfg.vggt_cameras_json),
                "robot_shots_json": os.path.abspath(cfg.robot_shots_json),
                "align_method": cfg.align_method,
                "vggt_pose_is_world_T_cam": bool(cfg.vggt_pose_is_world_T_cam),
            },
            "alignment_W_to_B": {
                "model": "p_B = s * R * p_W + t",
                "scale_s": float(s),
                "R_W_to_B": R.tolist(),
                "t_W_to_B": t.tolist(),
                "camera_center_rmse_all_m": float(err_all),
                "camera_center_rmse_kept_m": float(err_keep),
                "pairs_used_total": int(len(pairs)),
                "pairs_used_kept": int(len(keep_idx)),
                "pair_indices_total": pair_ids,
                "kept_indices": [int(i) for i in keep_idx.tolist()],
                "per_pair_error_m": [float(x) for x in e_all.tolist()],
            },
            "object": {
                "center": {
                    "vggt_world": center_w.tolist(),
                    "base_link": center_b.tolist(),
                },
                "obb": {
                    "extent": obb_extent.tolist() if obb_extent is not None else None,
                    "R_obb_in_vggt_world": obb_R_w.tolist() if obb_R_w is not None else None,
                    "corners_8": {
                        "index_order_note": "Order follows Open3D OrientedBoundingBox.get_box_points().",
                        "vggt_world": corners8_w.tolist(),
                        "base_link": corners8_b.tolist(),
                    }
                }
            }
        }

        out_path = os.path.join(cfg.out_dir, cfg.object_json_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"[Align+Export] wrote: {os.path.abspath(out_path)}")
        print(f"[Align+Export] rmse_all={err_all:.6f} m, rmse_kept={err_keep:.6f} m, scale={s:.6f}")

    # 可视化（主簇不着色；叠加中心点/球体）
    if cfg.visualize:
        geoms = [main_clean]
        if cfg.enable_center and center_pc is not None:
            geoms.append(center_pc)
        if cfg.enable_center and cfg.enable_center_sphere and center_sphere is not None:
            geoms.append(center_sphere)
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
