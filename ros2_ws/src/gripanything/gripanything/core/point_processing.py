#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
点云检查 -> 剔除桌面 -> 三标准聚类(必须同时满足①②③) -> 颜色后掩膜净化 -> 落盘
- 只删除“小团”，不删除“大团”
- 可视化与 centers.json 一律使用 OBB 中心

三标准（必须全部成立才保留为最终簇）：
① 尺寸下限：OBB 三轴尺寸逐轴 ≥ MIN_SIZE（不设上限，不删“大团”）
② 几何连通：空间半径 eps_geo（= DBSCAN_EPS_MULT * voxel）内 DBSCAN 连通，且点数 ≥ DBSCAN_MIN_POINTS
③ 颜色相近：颜色嵌入空间内 DBSCAN 连通（色相按圆环距离），且点数 ≥ COLOR_MIN_POINTS
    之后再做一次“颜色后掩膜净化”，若净化后不满足①或点数阈值，仍然丢弃
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import open3d as o3d


# ===================== 可修改参数区域（只改这里） =====================
PLY_PATH = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/vggt_output/blue_cylinder_b30.ply"
OUT_DIR = "pc_out"
VISUALIZE = True

# —— 桌面/黑点 颜色剔除（HSV）——
V_BLACK = 0.10
S_GRAY  = 0.70
V_GRAY_MIN = 0.40
V_GRAY_MAX = 1.10

# —— 仅用于尺度标尺（不会做体素下采样）——
VOXEL_DIVISOR = 200.0
VOXEL_MIN = 0.002
VOXEL_MAX = 0.02

# —— 标准②：几何 DBSCAN —— #
DBSCAN_EPS_MULT = 1.8    # eps_geo = DBSCAN_EPS_MULT * voxel
DBSCAN_MIN_POINTS = 40

# —— 标准①：尺寸下限（不设上限）—— #
MIN_SIZE = (0.05, 0.05, 0.05)  # m

# —— 标准③：颜色一致性（几何簇内的二次 DBSCAN）—— #
COLOR_W_HSV = (4.0, 0.0, 0.2)  # (w_h, w_s, w_v)：更看重色相H，忽略S，略看V
COLOR_EPS = 0.05
COLOR_MIN_POINTS = 15          # 小于此视为小团

# —— 颜色后掩膜净化（剔除“灰尾巴”） —— #
COLOR_POSTMASK_ENABLE   = True
COLOR_REF_TOP_S_RATIO   = 0.40   # 参考色由高饱和度前40%点估计
COLOR_HUE_TOL_DEG       = 12.0   # 色相容差
COLOR_S_MIN_ABS         = 0.32   # 绝对 S 下限
COLOR_S_MIN_RATIO       = 0.80   # 相对 S 下限：S >= ratio*S_ref
# =====================================================================


# ===================== 配置对象（无需改动） =====================
@dataclass
class Config:
    ply_path: str = PLY_PATH
    out_dir: str = OUT_DIR
    visualize: bool = VISUALIZE

    v_black: float = V_BLACK
    s_gray: float = S_GRAY
    v_gray_min: float = V_GRAY_MIN
    v_gray_max: float = V_GRAY_MAX

    voxel_divisor: float = VOXEL_DIVISOR
    voxel_min: float = VOXEL_MIN
    voxel_max: float = VOXEL_MAX

    dbscan_eps_mult: float = DBSCAN_EPS_MULT
    dbscan_min_points: int = DBSCAN_MIN_POINTS

    min_size: Tuple[float, float, float] = MIN_SIZE

    color_refine_eps: float = COLOR_EPS
    color_refine_min_points: int = COLOR_MIN_POINTS
    color_refine_whsv: Tuple[float, float, float] = COLOR_W_HSV

    color_postmask_enable: bool = COLOR_POSTMASK_ENABLE
    color_ref_top_s_ratio: float = COLOR_REF_TOP_S_RATIO
    color_hue_tol_deg: float = COLOR_HUE_TOL_DEG
    color_s_min_abs: float = COLOR_S_MIN_ABS
    color_s_min_ratio: float = COLOR_S_MIN_RATIO


# ===================== 工具函数 =====================
def ensure_outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def auto_voxel(pcd: o3d.geometry.PointCloud, divisor=200.0, vmin=0.002, vmax=0.02) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    max_extent = float(np.max(aabb.get_extent()))
    return float(np.clip(max_extent / divisor, vmin, vmax))


def pass_min_size(pcd: o3d.geometry.PointCloud, min_size: Tuple[float, float, float]) -> bool:
    """标准①：只检查 OBB 尺寸下限（不设上限）。"""
    ext = np.sort(pcd.get_oriented_bounding_box().extent)
    return bool(np.all(ext >= np.sort(np.array(min_size))))


def rgb01(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    if not pcd.has_colors():
        raise ValueError("该点云不含颜色。")
    C = np.asarray(pcd.colors).astype(np.float32)
    if C.max() > 1.0:
        C = C / 255.0
    return np.clip(C, 0.0, 1.0)


def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin + 1e-9
    h = np.zeros_like(cmax)
    m = cmax == r; h[m] = ((g - b)[m] / delta[m]) % 6
    m = cmax == g; h[m] = ((b - r)[m] / delta[m]) + 2
    m = cmax == b; h[m] = ((r - g)[m] / delta[m]) + 4
    h = (h / 6.0) % 1.0
    s = delta / (cmax + 1e-9)
    v = cmax
    return np.stack([h, s, v], axis=1)


def hsv_embed_for_dbscan(hsv: np.ndarray, w_h: float, w_s: float, w_v: float) -> np.ndarray:
    """映射为 3 维特征：[w_h*cos(2πH), w_h*sin(2πH), w_s*S + w_v*V]"""
    H = hsv[:, 0] * 2.0 * math.pi
    ch, sh = np.cos(H), np.sin(H)
    sv = w_s * hsv[:, 1] + w_v * hsv[:, 2]
    return np.stack([w_h * ch, w_h * sh, sv], axis=1).astype(np.float64)


def dbscan_labels_on_points(pts3d: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts3d)
    return np.array(pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))


def dbscan_labels_on_features(X3: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(X3)
    return np.array(pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))


# —— 桌面剔除 —— #
def remove_table_and_black(pcd: o3d.geometry.PointCloud, cfg: Config):
    if not pcd.has_colors():
        print("[Warn] 点云无颜色，跳过桌面剔除。")
        return pcd, o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    C = rgb01(pcd)
    HSV = rgb_to_hsv_np(C)
    S, V = HSV[:, 1], HSV[:, 2]

    mask_black = V < cfg.v_black
    mask_table = (S < cfg.s_gray) & (V > cfg.v_gray_min) & (V < cfg.v_gray_max)

    idx_keep = np.where(~mask_black & ~mask_table)[0]
    idx_black = np.where(mask_black)[0]
    idx_table = np.where(mask_table)[0]

    objects = pcd.select_by_index(idx_keep)
    black_only = pcd.select_by_index(idx_black)
    table_only = pcd.select_by_index(idx_table)

    print(f"[Color-Remove] black={len(idx_black)}, table={len(idx_table)}, keep={len(idx_keep)}")
    return objects, table_only, black_only


# —— 颜色二次细分（标准③） —— #
def split_cluster_by_color(cluster_pcd: o3d.geometry.PointCloud,
                           eps: float,
                           min_points: int,
                           w_h: float, w_s: float, w_v: float) -> List[o3d.geometry.PointCloud]:
    if not cluster_pcd.has_colors():
        return []  # 没颜色就不能满足③，直接失败
    C = rgb01(cluster_pcd)
    HSV = rgb_to_hsv_np(C)
    X = hsv_embed_for_dbscan(HSV, w_h, w_s, w_v)
    labels = dbscan_labels_on_features(X, eps, min_points)
    if labels.size == 0 or labels.max() < 0:
        return []  # 不能满足③
    out = []
    for li in range(int(labels.max()) + 1):
        idx = np.where(labels == li)[0]
        if idx.size >= min_points:
            out.append(cluster_pcd.select_by_index(idx))
    return out


def robust_center(points: np.ndarray, w: float = 0.7, trimmed_ratio: float = 0.10) -> np.ndarray:
    obb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).get_oriented_bounding_box()
    c1 = obb.center
    if trimmed_ratio > 0:
        med = np.median(points, axis=0)
        d = np.linalg.norm(points - med, axis=1)
        k = int(len(d) * trimmed_ratio)
        keep = np.argsort(d)[k:len(d) - k] if len(d) - 2 * k > 0 else np.arange(len(d))
        c2 = points[keep].mean(axis=0)
    else:
        c2 = points.mean(axis=0)
    return w * c1 + (1.0 - w) * c2


def colorize_clusters(clusters: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.PointCloud]:
    random.seed(0)
    out = []
    for c in clusters:
        col = np.array([random.random()*0.8+0.2,
                        random.random()*0.8+0.2,
                        random.random()*0.8+0.2])
        c.paint_uniform_color(col)
        out.append(c)
    return out


def make_sphere(center: np.ndarray, radius: float, color=(0, 0, 1)):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(center)
    return s


# ============= 颜色后掩膜净化（剔除“灰尾巴”） =============
def _circ_mean_hue(h_norm: np.ndarray, w: np.ndarray) -> float:
    ang = 2.0 * math.pi * h_norm
    c = np.sum(w * np.cos(ang)); s = np.sum(w * np.sin(ang))
    if c == 0 and s == 0:
        return float(np.mean(h_norm))
    ang_mean = math.atan2(s, c)
    if ang_mean < 0: ang_mean += 2.0 * math.pi
    return ang_mean / (2.0 * math.pi)


def _hue_diff_abs(h1: np.ndarray, h0: float) -> np.ndarray:
    d = np.abs(h1 - h0)
    return np.minimum(d, 1.0 - d)


def apply_color_postmask(cluster_pcd: o3d.geometry.PointCloud,
                         top_s_ratio: float,
                         hue_tol_deg: float,
                         s_min_abs: float,
                         s_min_ratio: float) -> o3d.geometry.PointCloud:
    if not cluster_pcd.has_colors() or len(cluster_pcd.points) == 0:
        return cluster_pcd
    C = rgb01(cluster_pcd)
    HSV = rgb_to_hsv_np(C)
    H, S = HSV[:, 0], HSV[:, 1]
    k = max(1, int(len(S)*max(0.0, min(1.0, top_s_ratio))))
    idx_topS = np.argsort(S)[-k:]
    H_ref = _circ_mean_hue(H[idx_topS], S[idx_topS])
    S_ref = float(np.median(S[idx_topS]))
    hue_tol = hue_tol_deg / 360.0
    dh = _hue_diff_abs(H, H_ref)
    s_dyn_min = max(s_min_abs, s_min_ratio * S_ref)
    keep = (dh <= hue_tol) & (S >= s_dyn_min)
    idx_keep = np.where(keep)[0]
    if idx_keep.size == 0:
        return cluster_pcd
    return cluster_pcd.select_by_index(idx_keep)


def centers_as_points_pcd(centers: np.ndarray, color=(0, 0, 1)) -> o3d.geometry.PointCloud:
    if centers.size == 0:
        return o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(centers)
    pcd = o3d.geometry.PointCloud(pts)
    col = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (centers.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd


# ===================== 主流程 =====================
def run(cfg: Config):
    ensure_outdir(cfg.out_dir)
    ensure_outdir(os.path.join(cfg.out_dir, "clusters"))

    # 1) 加载
    pcd_raw = o3d.io.read_point_cloud(cfg.ply_path)
    if pcd_raw.is_empty():
        raise RuntimeError(f"点云为空或无法读取: {cfg.ply_path}")
    print(f"[Load] {cfg.ply_path}  points={np.asarray(pcd_raw.points).shape[0]}  has_color={pcd_raw.has_colors()}")

    # 2) 剔除桌面/黑点
    objects, table_only, black_only = remove_table_and_black(pcd_raw, cfg)
    o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "points_no_table.ply"), objects, write_ascii=False, compressed=True)
    if len(table_only.points) > 0:
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "table_only.ply"), table_only, write_ascii=False, compressed=True)
    if len(black_only.points) > 0:
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "black_only.ply"), black_only, write_ascii=False, compressed=True)

    # 3) 尺度标尺 & 几何 DBSCAN（②）
    voxel = auto_voxel(objects, cfg.voxel_divisor, cfg.voxel_min, cfg.voxel_max)
    eps_geo = cfg.dbscan_eps_mult * voxel
    pts = np.asarray(objects.points)
    geo_labels = dbscan_labels_on_points(pts, eps=eps_geo, min_points=cfg.dbscan_min_points)
    n_geo = int(geo_labels.max()) + 1 if geo_labels.size > 0 else 0
    print(f"[DBSCAN-Geo] voxel={voxel:.4f}  eps={eps_geo:.4f}  min_pts={cfg.dbscan_min_points} -> {n_geo} 几何簇")

    # 4) 满足①+②的几何簇
    geo_clusters: List[Tuple[int, o3d.geometry.PointCloud]] = []
    for gi in range(n_geo):
        idx = np.where(geo_labels == gi)[0]
        if idx.size == 0:
            continue
        c_i = objects.select_by_index(idx)
        if pass_min_size(c_i, cfg.min_size):  # ①
            geo_clusters.append((gi, c_i))
    print(f"[Filter] 满足①+②的几何簇: {len(geo_clusters)}")

    # 5) 对每个几何簇做颜色细分（③），三标准必须同时满足
    refined_clusters: List[Tuple[int, int, o3d.geometry.PointCloud]] = []
    w_h, w_s, w_v = cfg.color_refine_whsv
    for gi, c in geo_clusters:
        subcs = split_cluster_by_color(c, cfg.color_refine_eps, cfg.color_refine_min_points, w_h, w_s, w_v)
        # ③失败（例如全被视为灰/无颜色）则该几何簇不产生任何输出
        for ci, sc in enumerate(subcs):
            # 再次确保①（避免颜色切分后尺寸过小）+ 最小点数
            if len(sc.points) >= cfg.color_refine_min_points and pass_min_size(sc, cfg.min_size):
                refined_clusters.append((gi, ci, sc))
    print(f"[Color-Refine] 满足①②③后的候选子簇: {len(refined_clusters)}")

    # 6) 颜色后掩膜净化（若开启），净化后仍需满足①且点数阈值，否则丢弃
    final_clusters: List[Tuple[int, int, o3d.geometry.PointCloud]] = []
    for gi, ci, c in refined_clusters:
        if cfg.color_postmask_enable and c.has_colors():
            c = apply_color_postmask(
                c,
                top_s_ratio=cfg.color_ref_top_s_ratio,
                hue_tol_deg=cfg.color_hue_tol_deg,
                s_min_abs=cfg.color_s_min_abs,
                s_min_ratio=cfg.color_s_min_ratio
            )
        if len(c.points) >= cfg.color_refine_min_points and pass_min_size(c, cfg.min_size):
            final_clusters.append((gi, ci, c))
    print(f"[PostMask] 净化后最终簇: {len(final_clusters)}")

    # 7) 落盘：每簇 + 汇总 + centers.json（中心=OBB.center）
    results = []
    centers_list = []
    geoms = [objects]  # 可视化底图：剔除桌面后的点云

    for gi, ci, c in final_clusters:
        pts_c = np.asarray(c.points)
        obb = c.get_oriented_bounding_box(); obb.color = (1, 0, 0)
        chosen_center = obb.center                         # 统一用 OBB 中心
        rcenter = robust_center(pts_c, 0.7, 0.10)

        fp = os.path.join(cfg.out_dir, "clusters", f"cluster_{gi:02d}_{ci:02d}.ply")
        o3d.io.write_point_cloud(fp, c, write_ascii=False, compressed=True)

        results.append({
            "geom_cluster_id": int(gi),
            "color_sub_id": int(ci),
            "num_points": int(pts_c.shape[0]),
            "center": chosen_center.tolist(),      # OBB 中心
            "robust_center": rcenter.tolist(),
            "obb_center": obb.center.tolist(),
            "obb_extent": obb.extent.tolist(),
            "file": os.path.abspath(fp),
        })

        geoms += [obb, make_sphere(chosen_center, radius=max(voxel*1.5, 0.005), color=(0, 0, 1))]
        centers_list.append(chosen_center)

    # 汇总：上色后的所有簇
    colored = colorize_clusters([c for _, _, c in final_clusters]) if final_clusters else []
    merged_clusters = o3d.geometry.PointCloud()
    for c in colored:
        merged_clusters += c
    if len(merged_clusters.points) > 0:
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "points_clustered_colored.ply"),
                                 merged_clusters, write_ascii=False, compressed=True)

    # 汇总：簇 + 中心点（蓝色）
    centers_arr = np.array(centers_list, dtype=np.float64) if centers_list else np.zeros((0, 3))
    centers_pcd = centers_as_points_pcd(centers_arr, color=(0, 0, 1))
    with_centers = o3d.geometry.PointCloud(merged_clusters)
    with_centers += centers_pcd
    if len(with_centers.points) > 0:
        o3d.io.write_point_cloud(os.path.join(cfg.out_dir, "points_clusters_with_centers.ply"),
                                 with_centers, write_ascii=False, compressed=True)

    # centers.json
    with open(os.path.join(cfg.out_dir, "centers.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_ply": os.path.abspath(cfg.ply_path),
            "voxel": float(voxel),
            "criteria": {
                "1_min_size": list(cfg.min_size),
                "2_geo_dbscan": {"eps": float(eps_geo), "eps_mult": float(cfg.dbscan_eps_mult),
                                 "min_points": int(cfg.dbscan_min_points)},
                "3_color_dbscan": {"eps": float(cfg.color_refine_eps),
                                   "min_points": int(cfg.color_refine_min_points),
                                   "w_hsv": list(cfg.color_refine_whsv)},
                "postmask": {"enabled": bool(cfg.color_postmask_enable),
                             "top_s_ratio": float(cfg.color_ref_top_s_ratio),
                             "hue_tol_deg": float(cfg.color_hue_tol_deg),
                             "s_min_abs": float(cfg.color_s_min_abs),
                             "s_min_ratio": float(cfg.color_s_min_ratio)},
                "table_removal_hsv": {"v_black": float(cfg.v_black),
                                      "s_gray": float(cfg.s_gray),
                                      "v_gray_min": float(cfg.v_gray_min),
                                      "v_gray_max": float(cfg.v_gray_max)}
            },
            "clusters_count": len(results),
            "clusters": results
        }, f, ensure_ascii=False, indent=2)

    print(f"[Write] 已输出：")
    print(f"  - 剔除桌面后的点云: {os.path.join(cfg.out_dir, 'points_no_table.ply')}")
    if len(merged_clusters.points) > 0:
        print(f"  - 最终聚类结果（汇总点云）: {os.path.join(cfg.out_dir, 'points_clustered_colored.ply')}")
        print(f"  - 聚类+中心点的点云: {os.path.join(cfg.out_dir, 'points_clusters_with_centers.ply')}")
    print(f"  - 每簇单独文件: {os.path.join(cfg.out_dir, 'clusters', 'cluster_*.ply')}")
    print(f"  - 中心信息: {os.path.join(cfg.out_dir, 'centers.json')}")

    if cfg.visualize:
        o3d.visualization.draw_geometries(geoms)


# ===================== 入口 =====================
if __name__ == "__main__":
    cfg = Config()
    run(cfg)
