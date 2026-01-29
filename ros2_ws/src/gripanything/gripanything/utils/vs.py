#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import types
import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple, List

# Avoid importing open3d.ml (can break in some envs)
sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))

import open3d as o3d
import numpy as np


# ---------------- math helpers ----------------
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.tolist()
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=np.float64)


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return R such that R @ a = b (a,b are 3D).
    Robust for near-parallel and near-opposite cases.
    """
    a = _normalize(a.astype(np.float64))
    b = _normalize(b.astype(np.float64))
    c = float(np.dot(a, b))  # cos(theta)
    eps = 1e-8

    # Parallel
    if abs(c - 1.0) < eps:
        return np.eye(3, dtype=np.float64)

    # Opposite
    if abs(c + 1.0) < eps:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = _normalize(np.cross(a, ortho))
        vx = _skew(v)
        return np.eye(3) + 2.0 * (vx @ vx)

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = _skew(v)
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return R


# ---------------- ray visuals ----------------
def estimate_cylinder_height(points: np.ndarray,
                             ray_origin: np.ndarray,
                             ray_dir: np.ndarray,
                             min_height: float = 0.20) -> float:
    if points.size == 0:
        return float(min_height)

    r = _normalize(ray_dir)
    rel = points - ray_origin[None, :]
    t = rel @ r
    t_front = t[t > 0.01]
    if t_front.size == 0:
        t_front = t

    t_max = float(np.max(t_front))
    return float(max(t_max, min_height))


def create_ray_cylinder(ray_origin: np.ndarray,
                        ray_dir: np.ndarray,
                        radius: float,
                        height: float,
                        color=(0.2, 0.8, 0.2)) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=40, split=4)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    mesh.translate([0.0, 0.0, height / 2.0])  # bottom at z=0
    r = _normalize(ray_dir)
    R = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0], dtype=np.float64), r)
    mesh.rotate(R, center=[0.0, 0.0, 0.0])
    mesh.translate(ray_origin.tolist())
    return mesh


def create_ray_line(ray_origin: np.ndarray,
                    ray_dir: np.ndarray,
                    length: float,
                    color=(0.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    r = _normalize(ray_dir)
    p0 = ray_origin
    p1 = ray_origin + r * float(length)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.stack([p0, p1], axis=0))
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(np.array([color], dtype=np.float64))
    return line


# ---------------- JSON helpers ----------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _as_vec3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x):
        return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=np.float64)
    return None


def _as_mat3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, list) and len(x) == 3 and all(isinstance(r, list) and len(r) == 3 for r in x):
        try:
            M = np.array(x, dtype=np.float64)
            if M.shape == (3, 3):
                return M
        except Exception:
            return None
    return None


def _guess_object_json_from_ply(ply_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(ply_path))
    cand = os.path.join(d, "object_in_base_link.json")
    return cand if os.path.isfile(cand) else None


def _read_corners8(block: Any) -> Optional[np.ndarray]:
    if not isinstance(block, list) or len(block) != 8:
        return None
    out = []
    for p in block:
        v = _as_vec3(p)
        if v is None:
            return None
        out.append(v)
    return np.stack(out, axis=0)  # (8,3)


def _read_json_bundle(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[str, Any] = {"data": data}

    # centers
    center_w = None
    center_b = None
    obj = data.get("object", {})
    if isinstance(obj, dict):
        c = obj.get("center", {})
        if isinstance(c, dict):
            center_w = _as_vec3(c.get("vggt_world", None))
            center_b = _as_vec3(c.get("base_link", None))

    out["center_w"] = center_w
    out["center_b"] = center_b

    # corners8
    corners_w = None
    corners_b = None
    prism = obj.get("prism", {}) if isinstance(obj, dict) else {}
    if isinstance(prism, dict):
        c8 = prism.get("corners_8", None)
        if isinstance(c8, dict):
            corners_w = _read_corners8(c8.get("vggt_world", None))
            corners_b = _read_corners8(c8.get("base_link", None))

    out["corners_w"] = corners_w
    out["corners_b"] = corners_b

    # sim3 alignment
    align = data.get("alignment_W_to_B", {})
    s = None
    R = None
    t = None
    if isinstance(align, dict):
        if _is_num(align.get("scale_s", None)):
            s = float(align["scale_s"])
        R = _as_mat3(align.get("R_W_to_B", None))
        t = _as_vec3(align.get("t_W_to_B", None))
    out["scale_s"] = s
    out["R_W_to_B"] = R
    out["t_W_to_B"] = t

    return out


# -------------- visualization primitives --------------
def _make_sphere(center: np.ndarray, radius: float, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=20)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(center.tolist())
    return s


def _make_prism_wireframe(corners8: np.ndarray,
                          color=(1.0, 0.2, 0.2)) -> o3d.geometry.LineSet:
    """
    Assumes JSON note: bottom 4 (0..3) then top 4 (4..7), same order.
    Edges: bottom ring, top ring, verticals.
    """
    idx = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners8.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(idx)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64)[None, :], (idx.shape[0], 1)))
    return ls


def _make_center_to_corners(center: np.ndarray,
                            corners8: np.ndarray,
                            color=(0.2, 0.2, 1.0)) -> o3d.geometry.LineSet:
    pts = np.vstack([center[None, :], corners8])
    lines = np.array([[0, i] for i in range(1, 9)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float64)[None, :], (lines.shape[0], 1)))
    return ls


def _apply_sim3_W_to_B(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # p_B = s * R * p_W + t
    return (s * (points @ R.T)) + t[None, :]


def _choose_frame_auto(pcd_center: np.ndarray,
                       center_w: Optional[np.ndarray],
                       center_b: Optional[np.ndarray]) -> str:
    # pick the closer center to pcd center
    dw = float(np.linalg.norm(pcd_center - center_w)) if center_w is not None else float("inf")
    db = float(np.linalg.norm(pcd_center - center_b)) if center_b is not None else float("inf")
    if dw < db:
        return "vggt_world"
    return "base_link"


def inspect_point_cloud(ply_path: str,
                        json_path: Optional[str] = None,
                        frame: str = "auto",          # auto | vggt_world | base_link
                        pcd_to_base: bool = False,    # transform point cloud W->B using sim3
                        show_cylinder: bool = True,
                        cylinder_radius: float = 0.03,
                        ray_origin=(0.0, 0.0, 0.0),
                        ray_dir=(0.0, 0.0, 1.0),
                        show_center: bool = True,
                        show_corners: bool = True,
                        show_prism: bool = True,
                        show_center_rays: bool = False):

    # ---- load point cloud ----
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float64)

    num_points = int(pts.shape[0])
    has_color = bool(pcd.has_colors())
    bbox = pcd.get_axis_aligned_bounding_box()
    pcd_center = bbox.get_center()

    print(f"Loaded point cloud: {ply_path}")
    print(f"Number of points: {num_points}")
    print(f"Has color: {has_color}")
    print("Bounding box:")
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")
    print(f"  Center: {pcd_center}")

    extent = bbox.get_extent()
    axis_size = float(max(extent) * 0.2) if num_points > 0 else 0.1
    axis_size = max(axis_size, 0.05)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0.0, 0.0, 0.0]
    )

    # ---- load json bundle ----
    bundle = None
    if json_path is None:
        json_path = _guess_object_json_from_ply(ply_path)

    if json_path is not None and os.path.isfile(json_path):
        bundle = _read_json_bundle(json_path)
        print(f"\nLoaded JSON: {json_path}")

        cw = bundle.get("center_w", None)
        cb = bundle.get("center_b", None)
        if cw is not None:
            print(f"  center(vggt_world): {cw.tolist()}")
        else:
            print("  center(vggt_world): MISSING")
        if cb is not None:
            print(f"  center(base_link): {cb.tolist()}")
        else:
            print("  center(base_link): MISSING")

        cW = bundle.get("corners_w", None)
        cB = bundle.get("corners_b", None)
        if cW is not None:
            print("  corners_8(vggt_world):")
            for i in range(8):
                print(f"    {i}: {cW[i].tolist()}")
        else:
            print("  corners_8(vggt_world): MISSING")
        if cB is not None:
            print("  corners_8(base_link):")
            for i in range(8):
                print(f"    {i}: {cB[i].tolist()}")
        else:
            print("  corners_8(base_link): MISSING")

        s = bundle.get("scale_s", None)
        R = bundle.get("R_W_to_B", None)
        t = bundle.get("t_W_to_B", None)
        if s is not None and R is not None and t is not None:
            print(f"  sim3 W->B: scale_s={s:.6f}")
        else:
            print("  sim3 W->B: MISSING (cannot transform pcd)")

        # Optional: transform point cloud into base_link
        if pcd_to_base:
            if s is None or R is None or t is None:
                raise RuntimeError("pcd_to_base=True but sim3 (s,R,t) is missing in JSON.")
            pts_B = _apply_sim3_W_to_B(pts, s, R, t)
            pcd.points = o3d.utility.Vector3dVector(pts_B)
            pts = pts_B
            bbox = pcd.get_axis_aligned_bounding_box()
            pcd_center = bbox.get_center()
            print("\n[pcd] Applied sim3 W->B to point cloud.")
            print(f"[pcd] New bbox center: {pcd_center}")
    else:
        print(f"\nJSON not found: {json_path}")
        bundle = None

    # Decide which overlay frame to use
    selected = "vggt_world"
    if bundle is not None:
        if frame == "auto":
            selected = _choose_frame_auto(
                pcd_center=np.array(pcd_center, dtype=np.float64),
                center_w=bundle.get("center_w", None),
                center_b=bundle.get("center_b", None),
            )
            print(f"\n[overlay] frame=auto -> selected={selected}")
        else:
            selected = frame
            print(f"\n[overlay] frame={selected}")

        if selected not in ("vggt_world", "base_link"):
            raise ValueError("--frame must be one of: auto, vggt_world, base_link")

    # ---- ray visuals ----
    ray_origin = np.array(ray_origin, dtype=np.float64)
    ray_dir = np.array(ray_dir, dtype=np.float64)
    cyl_height = estimate_cylinder_height(pts, ray_origin, ray_dir, min_height=max(0.20, axis_size * 2.0))
    cyl = create_ray_cylinder(ray_origin, ray_dir, radius=float(cylinder_radius), height=float(cyl_height))
    ray_line = create_ray_line(ray_origin, ray_dir, length=cyl_height)

    # marker size
    marker_r = max(0.004, axis_size * 0.10)
    corner_r = marker_r * 0.65

    # ---- build overlays for both frames (if available) ----
    overlays = {
        "vggt_world": {"center": None, "corners": [], "prism": None, "rays": None},
        "base_link":  {"center": None, "corners": [], "prism": None, "rays": None},
    }

    if bundle is not None:
        for key, c_key, p_key in [("vggt_world", "center_w", "corners_w"),
                                  ("base_link", "center_b", "corners_b")]:
            c = bundle.get(c_key, None)
            c8 = bundle.get(p_key, None)
            if c is not None:
                overlays[key]["center"] = _make_sphere(c, radius=marker_r, color=(1.0, 0.0, 1.0))
            if c8 is not None:
                overlays[key]["corners"] = [
                    _make_sphere(c8[i], radius=corner_r, color=(1.0, 0.3, 0.0)) for i in range(8)
                ]
                overlays[key]["prism"] = _make_prism_wireframe(c8, color=(1.0, 0.0, 0.0))
                if c is not None:
                    overlays[key]["rays"] = _make_center_to_corners(c, c8, color=(0.2, 0.2, 1.0))

    # ---- visualizer ----
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Inspector", width=1280, height=720)

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(ray_line)

    state = {
        "cyl_visible": False,
        "center_visible": False,
        "corners_visible": False,
        "prism_visible": False,
        "center_rays_visible": False,
        "overlay_frame": selected,
    }

    if show_cylinder:
        vis.add_geometry(cyl)
        state["cyl_visible"] = True

    def _apply_overlay_visibility(frame_name: str, on: bool):
        ov = overlays.get(frame_name, None)
        if ov is None:
            return
        if ov["center"] is not None:
            if on and show_center and not state["center_visible"]:
                vis.add_geometry(ov["center"])
            if (not on) and state["center_visible"]:
                vis.remove_geometry(ov["center"], reset_bounding_box=False)
        if ov["corners"]:
            if on and show_corners and not state["corners_visible"]:
                for g in ov["corners"]:
                    vis.add_geometry(g)
            if (not on) and state["corners_visible"]:
                for g in ov["corners"]:
                    vis.remove_geometry(g, reset_bounding_box=False)
        if ov["prism"] is not None:
            if on and show_prism and not state["prism_visible"]:
                vis.add_geometry(ov["prism"])
            if (not on) and state["prism_visible"]:
                vis.remove_geometry(ov["prism"], reset_bounding_box=False)
        if ov["rays"] is not None:
            if on and show_center_rays and not state["center_rays_visible"]:
                vis.add_geometry(ov["rays"])
            if (not on) and state["center_rays_visible"]:
                vis.remove_geometry(ov["rays"], reset_bounding_box=False)

    # Initial overlay add (only selected frame)
    if bundle is not None:
        # set vis state flags according to requested startup booleans
        # (we mark them True once added; toggles will flip)
        if show_center and overlays[selected]["center"] is not None:
            vis.add_geometry(overlays[selected]["center"])
            state["center_visible"] = True
        if show_corners and overlays[selected]["corners"]:
            for g in overlays[selected]["corners"]:
                vis.add_geometry(g)
            state["corners_visible"] = True
        if show_prism and overlays[selected]["prism"] is not None:
            vis.add_geometry(overlays[selected]["prism"])
            state["prism_visible"] = True
        if show_center_rays and overlays[selected]["rays"] is not None:
            vis.add_geometry(overlays[selected]["rays"])
            state["center_rays_visible"] = True

    def _toggle_geom(key: str, geom):
        if geom is None:
            return False
        if state[key]:
            vis.remove_geometry(geom, reset_bounding_box=False)
            state[key] = False
        else:
            vis.add_geometry(geom)
            state[key] = True
        return False

    def _toggle_cylinder(_vis):
        return _toggle_geom("cyl_visible", cyl)

    def _toggle_center(_vis):
        if bundle is None:
            return False
        ov = overlays[state["overlay_frame"]]
        return _toggle_geom("center_visible", ov["center"])

    def _toggle_prism(_vis):
        if bundle is None:
            return False
        ov = overlays[state["overlay_frame"]]
        return _toggle_geom("prism_visible", ov["prism"])

    def _toggle_center_rays(_vis):
        if bundle is None:
            return False
        ov = overlays[state["overlay_frame"]]
        return _toggle_geom("center_rays_visible", ov["rays"])

    def _toggle_corners(_vis):
        if bundle is None:
            return False
        ov = overlays[state["overlay_frame"]]
        if not ov["corners"]:
            return False
        if state["corners_visible"]:
            for g in ov["corners"]:
                _vis.remove_geometry(g, reset_bounding_box=False)
            state["corners_visible"] = False
        else:
            for g in ov["corners"]:
                _vis.add_geometry(g)
            state["corners_visible"] = True
        return False

    def _toggle_frame(_vis):
        if bundle is None:
            return False
        cur = state["overlay_frame"]
        nxt = "base_link" if cur == "vggt_world" else "vggt_world"

        # remove current visible overlay geoms
        ov_cur = overlays[cur]
        if state["center_visible"] and ov_cur["center"] is not None:
            _vis.remove_geometry(ov_cur["center"], reset_bounding_box=False)
        if state["corners_visible"] and ov_cur["corners"]:
            for g in ov_cur["corners"]:
                _vis.remove_geometry(g, reset_bounding_box=False)
        if state["prism_visible"] and ov_cur["prism"] is not None:
            _vis.remove_geometry(ov_cur["prism"], reset_bounding_box=False)
        if state["center_rays_visible"] and ov_cur["rays"] is not None:
            _vis.remove_geometry(ov_cur["rays"], reset_bounding_box=False)

        # add to next (respect current visibility flags)
        ov_nxt = overlays[nxt]
        if state["center_visible"] and ov_nxt["center"] is not None:
            _vis.add_geometry(ov_nxt["center"])
        if state["corners_visible"] and ov_nxt["corners"]:
            for g in ov_nxt["corners"]:
                _vis.add_geometry(g)
        if state["prism_visible"] and ov_nxt["prism"] is not None:
            _vis.add_geometry(ov_nxt["prism"])
        if state["center_rays_visible"] and ov_nxt["rays"] is not None:
            _vis.add_geometry(ov_nxt["rays"])

        state["overlay_frame"] = nxt
        print(f"[overlay] switched frame -> {nxt}")
        return False

    vis.register_key_callback(ord("C"), _toggle_cylinder)    # cylinder
    vis.register_key_callback(ord("O"), _toggle_center)      # center
    vis.register_key_callback(ord("V"), _toggle_corners)     # vertices
    vis.register_key_callback(ord("P"), _toggle_prism)       # prism
    vis.register_key_callback(ord("R"), _toggle_center_rays) # rays
    vis.register_key_callback(ord("F"), _toggle_frame)       # frame switch

    print("\nControls:")
    print("  C: toggle ray-cylinder")
    print("  O: toggle center sphere")
    print("  V: toggle 8 corner spheres")
    print("  P: toggle prism wireframe")
    print("  R: toggle center->corners rays")
    print("  F: switch overlay frame (vggt_world <-> base_link)")
    print("Note:")
    print("  - If your PLY is in VGGT world, use --frame vggt_world (or auto).")
    print("  - If you want everything in base_link, add --pcd_to_base and use --frame base_link.\n")

    vis.run()
    vis.destroy_window()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, default="/home/MA_SmartGrip/Smartgrip/result/single/b/b17/output/offline_output/main_cluster_clean.ply",
                    help="Path to input .ply point cloud.")
    ap.add_argument("--json", type=str, default=None,
                    help="Path to object_in_base_link.json. If omitted, auto-guess from --ply directory.")

    ap.add_argument("--frame", type=str, default="auto", choices=["auto", "vggt_world", "base_link"],
                    help="Which frame's center/corners to overlay.")

    ap.add_argument("--pcd_to_base", action="store_true",
                    help="Transform point cloud from vggt_world to base_link using sim3 in JSON (p_B = s R p_W + t).")

    ap.add_argument("--show-cylinder", dest="show_cylinder", action="store_true", help="Show ray-cylinder at startup.")
    ap.add_argument("--no-cylinder", dest="show_cylinder", action="store_false", help="Hide ray-cylinder at startup.")
    ap.set_defaults(show_cylinder=True)

    ap.add_argument("--cyl-radius", type=float, default=0.03, help="Cylinder radius in point-cloud units (meters).")

    ap.add_argument("--ray-origin", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "Z"), help="Ray origin in point-cloud frame.")
    ap.add_argument("--ray-dir", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                    metavar=("DX", "DY", "DZ"), help="Ray direction in point-cloud frame.")

    ap.add_argument("--show-center", action="store_true", default=True)
    ap.add_argument("--hide-center", dest="show_center", action="store_false")

    ap.add_argument("--show-corners", action="store_true", default=True)
    ap.add_argument("--hide-corners", dest="show_corners", action="store_false")

    ap.add_argument("--show-prism", action="store_true", default=True)
    ap.add_argument("--hide-prism", dest="show_prism", action="store_false")

    ap.add_argument("--show-center-rays", action="store_true", default=False)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_point_cloud(
        ply_path=args.ply,
        json_path=args.json,
        frame=args.frame,
        pcd_to_base=bool(args.pcd_to_base),
        show_cylinder=bool(args.show_cylinder),
        cylinder_radius=float(args.cyl_radius),
        ray_origin=tuple(args.ray_origin),
        ray_dir=tuple(args.ray_dir),
        show_center=bool(args.show_center),
        show_corners=bool(args.show_corners),
        show_prism=bool(args.show_prism),
        show_center_rays=bool(args.show_center_rays),
    )
