#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
流程（稳妥版）：
0) 全局清黑（删极暗点）
1) 颜色引导 -> RANSAC 平面拟合 -> 按“几何距离 + 颜色守门”剔除桌面
2) 落盘 points_no_table.ply
3) DBSCAN 找主簇
4) 距离裁剪：仅保留距离主簇<=gap的点
5) 仅对主簇做半径离群剔除
6) [可选] 计算主簇中心，并导出中心点 + 球体
7) 可视化最终结果（主簇不着色；若启用中心则叠加中心点与球体）
"""

import os, math, json
from dataclasses import dataclass
import numpy as np
import open3d as o3d


# ===================== 可调参数 =====================
PLY_PATH = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/vggt_output/blue_cube_30_b30.ply"
OUT_DIR  = "pc_out_simple"
VISUALIZE = True

# 0) 全局清黑
V_BLACK_GLOBAL = 0.15

# 1) 颜色引导（候选，仅用于拟合平面，不直接删点）
S_GRAY_CAND_MAX = 0.65
V_GRAY_CAND_MIN = 0.35

# 1) 删除时颜色守门（防误删黄色/彩色）
S_REMOVE_MAX = 0.55
V_REMOVE_MIN = 0.25

# 尺度标尺（自适应）
VOXEL_DIVISOR = 200.0
VOXEL_MIN = 0.002
VOXEL_MAX = 0.02

# RANSAC 平面
PLANE_DETECT_DIST_MULT = 0.80   # 拟合阈值 = 0.80 * voxel
PLANE_REMOVE_DIST_MULT = 1.60   # 删除阈值 = 1.60 * voxel
RANSAC_N = 3
RANSAC_ITERS = 2000
MAX_PLANES = 5
MIN_PLANE_POINTS_ABS = 5000
MIN_PLANE_POINTS_RATIO = 0.001
DS_VOXEL_MULT = 1.0

# 4) 主簇识别（DBSCAN）与“距离裁剪”
MAIN_DBSCAN_EPS_MULT = 3.0      # eps = 3.0 * voxel
MAIN_DBSCAN_MIN_POINTS = 50
MAIN_GAP_MULT = 4.0             # 距离阈值 = 4.0 * voxel

# 5) 主簇内的离群剔除（半径法）
INNER_RADIUS_MULT = 1.2         # 半径 = 1.2 * voxel
INNER_RADIUS_MIN_POINTS = 12

# 6) [开关] 主簇中心的导出与可视化
ENABLE_CENTER = True
CENTER_POINT_COLOR = (0.0, 0.0, 1.0)   # 仅中心点颜色（主簇不着色）
ENABLE_CENTER_SPHERE = True
CENTER_SPHERE_COLOR = (0.0, 0.2, 1.0)
CENTER_SPHERE_RADIUS_MULT = 1.0        # 球半径 = 2.0 * voxel（约等于 2*voxel）
CENTER_SPHERE_RADIUS_MIN  = 0.004      # 最小半径（米），保证可见性
# ============================================================================


@dataclass
class Config:
    ply_path: str = PLY_PATH
    out_dir: str = OUT_DIR
    visualize: bool = VISUALIZE

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


# ===================== 工具函数 =====================
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
    col = np.tile(np.array(color, dtype=np.float64).reshape(1,3), (1,1))
    pc.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(col))
    return pc

def make_center_sphere(center, radius, color=(0,0,1)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.asarray(color, dtype=float))
    mesh.translate(np.asarray(center, dtype=float))
    return mesh


# ===================== 桌面剔除 =====================
def remove_table_only(pcd: o3d.geometry.PointCloud, cfg: Config) -> (o3d.geometry.PointCloud, float):
    # 0) 全局清黑
    C = rgb01(pcd); HSV = rgb_to_hsv_np(C); S, V = HSV[:, 1], HSV[:, 2]
    keep_black = V >= cfg.v_black_global
    if keep_black.sum() < len(V):
        print(f"[Black-Remove] removed={int((~keep_black).sum())}, kept={int(keep_black.sum())}")
    pcd = pcd.select_by_index(np.where(keep_black)[0])

    # 尺度
    voxel = auto_voxel(pcd, cfg.voxel_divisor, cfg.voxel_min, cfg.voxel_max)
    detect_thr = cfg.plane_detect_dist_mult * voxel
    remove_thr = cfg.plane_remove_dist_mult * voxel
    ds_voxel = max(1e-3, cfg.ds_voxel_mult * voxel)
    print(f"[Scale] voxel={voxel:.4f}m  detect_thr={detect_thr:.4f}m  remove_thr={remove_thr:.4f}m  ds_voxel={ds_voxel:.4f}m")

    # 1) 颜色引导候选（仅用于拟合）
    C = rgb01(pcd); HSV = rgb_to_hsv_np(C); S, V = HSV[:, 1], HSV[:, 2]
    cand_mask = (S < cfg.s_gray_cand_max) & (V > cfg.v_gray_cand_min)
    print(f"[Candidates] grayish={int(cand_mask.sum())} / total={len(pcd.points)}")

    # 颜色守门（删除时必须满足）
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


# ===================== 主簇：距离裁剪 + 内部离群剔除 =====================
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

    # 距离裁剪到主簇附近
    gap = cfg.main_gap_mult * voxel
    dists = np.asarray(pcd_no_table.compute_point_cloud_distance(main_raw))
    near_main = pcd_no_table.select_by_index(np.where(dists <= gap)[0])
    print(f"[NearMain] gap={gap:.4f}m  kept={len(near_main.points)} / {len(pcd_no_table.points)}")

    # 在 near_main 上再次取主簇
    pts2 = np.asarray(near_main.points)
    labels2 = dbscan_labels_points(pts2, eps=eps, min_points=cfg.main_dbscan_min_points)
    num2 = int(labels2.max()) + 1 if labels2.size > 0 else 0
    if num2 == 0:
        raise RuntimeError("Near-main 范围内未找到簇。")
    sizes2 = [(li, int(np.count_nonzero(labels2 == li))) for li in range(num2)]
    main2_label, main2_size = max(sizes2, key=lambda x: x[1])
    main2_raw = near_main.select_by_index(np.where(labels2 == main2_label)[0])

    # 主簇半径离群剔除
    r = cfg.inner_radius_mult * voxel
    main2_clean = remove_radius_outlier(main2_raw, radius=r, min_points=cfg.inner_radius_min_points)
    print(f"[InnerOutlier] radius={r:.4f}m  min_pts={cfg.inner_radius_min_points}  main_raw={len(main2_raw.points)} -> clean={len(main2_clean.points)}")

    return near_main, main2_raw, main2_clean


# ===================== 入口 =====================
def run(cfg: Config):
    ensure_outdir(cfg.out_dir)

    # 读点云
    pcd = o3d.io.read_point_cloud(cfg.ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"点云为空或无法读取: {cfg.ply_path}")
    print(f"[Load] {cfg.ply_path}  points={np.asarray(pcd.points).shape[0]}  has_color={pcd.has_colors()}")

    # 桌面剔除
    pcd_no_table, voxel = remove_table_only(pcd, cfg)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "points_no_table.ply"),
                             pcd_no_table, write_ascii=False, compressed=True)

    # 主簇距离裁剪 + 主簇离群剔除
    near_main, main_raw, main_clean = keep_near_main_and_clean(pcd_no_table, voxel, cfg)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "near_main_raw.ply"),
                             near_main, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_cluster_raw.ply"),
                             main_raw, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_cluster_clean.ply"),
                             main_clean, write_ascii=False, compressed=True)

    # [可选] 中心：点 + 球体
    center_pc = None
    center_sphere = None
    if cfg.enable_center and len(main_clean.points) > 0:
        obb = main_clean.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float64).tolist()
        extent = np.asarray(obb.extent, dtype=np.float64).tolist()

        # 中心点
        center_pc = single_point_pcd(center, color=cfg.center_point_color)
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "main_center.ply"),
                                 center_pc, write_ascii=False, compressed=True)

        # 中心球（mesh）
        if cfg.enable_center_sphere:
            r = max(cfg.center_sphere_radius_min, cfg.center_sphere_radius_mult * voxel)
            center_sphere = make_center_sphere(center, radius=r, color=cfg.center_sphere_color)
            o3d.io.write_triangle_mesh(os.path.join(cfg.out_dir, "main_center_sphere.ply"),
                                       center_sphere, write_ascii=False)
            print(f"[CenterSphere] radius={r:.4f} m  mesh=main_center_sphere.ply")

        # JSON
        info = {
            "input_ply": os.path.abspath(cfg.ply_path),
            "voxel": float(voxel),
            "center": center,
            "obb_extent": extent,
            "counts": {
                "points_no_table": len(pcd_no_table.points),
                "near_main_raw": len(near_main.points),
                "main_cluster_raw": len(main_raw.points),
                "main_cluster_clean": len(main_clean.points),
            }
        }
        with open(os.path.join(cfg.out_dir, "main_center.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"[Center] center={center}  已写出 main_center.ply / main_center.json")

    print("[Write] 已输出：")
    print(f"  - points_no_table: {os.path.abspath(os.path.join(cfg.out_dir, 'points_no_table.ply'))}")
    print(f"  - near_main_raw:   {os.path.abspath(os.path.join(cfg.out_dir, 'near_main_raw.ply'))}")
    print(f"  - main_cluster_raw:{os.path.abspath(os.path.join(cfg.out_dir, 'main_cluster_raw.ply'))}")
    print(f"  - main_cluster_clean:{os.path.abspath(os.path.join(cfg.out_dir, 'main_cluster_clean.ply'))}")
    if cfg.enable_center:
        print("  - main_center.ply / main_center.json")
        if cfg.enable_center_sphere:
            print("  - main_center_sphere.ply")

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
