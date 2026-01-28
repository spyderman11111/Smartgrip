#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import types
import argparse

# Avoid importing open3d.ml (can break in some envs)
sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))

import open3d as o3d
import numpy as np


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
        # Find any orthogonal axis to a
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = _normalize(np.cross(a, ortho))
        # 180-degree rotation around v: R = I + 2*[v]_x^2
        vx = _skew(v)
        return np.eye(3) + 2.0 * (vx @ vx)

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = _skew(v)
    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return R


def estimate_cylinder_height(points: np.ndarray,
                             ray_origin: np.ndarray,
                             ray_dir: np.ndarray,
                             min_height: float = 0.20) -> float:
    """
    Estimate a reasonable cylinder height along ray direction by projecting points onto the ray.
    Prefer points in front of the origin (t > 0).
    """
    if points.size == 0:
        return float(min_height)

    r = _normalize(ray_dir)
    rel = points - ray_origin[None, :]
    t = rel @ r  # projection length onto ray
    t_front = t[t > 0.01]
    if t_front.size == 0:
        t_front = t  # fallback

    t_max = float(np.max(t_front))
    return float(max(t_max, min_height))


def create_ray_cylinder(ray_origin: np.ndarray,
                        ray_dir: np.ndarray,
                        radius: float,
                        height: float,
                        color=(0.2, 0.8, 0.2)) -> o3d.geometry.TriangleMesh:
    """
    Create a cylinder whose axis aligns with ray_dir, with bottom anchored at ray_origin.
    """
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=40, split=4)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    # Open3D cylinder is centered at origin and extends from -h/2..+h/2 along +Z.
    # Shift so its bottom is at z=0 in local frame.
    mesh.translate([0.0, 0.0, height / 2.0])

    # Rotate +Z to ray_dir
    r = _normalize(ray_dir)
    R = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0], dtype=np.float64), r)
    mesh.rotate(R, center=[0.0, 0.0, 0.0])

    # Move to ray origin
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


def inspect_point_cloud(ply_path: str,
                        show_cylinder: bool = True,
                        cylinder_radius: float = 0.03,
                        ray_origin=(0.0, 0.0, 0.0),
                        ray_dir=(0.0, 0.0, 1.0)):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    num_points = int(pts.shape[0])
    has_color = bool(pcd.has_colors())
    bbox = pcd.get_axis_aligned_bounding_box()

    print(f"Loaded point cloud: {ply_path}")
    print(f"Number of points: {num_points}")
    print(f"Has color: {has_color}")
    print("Bounding box:")
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")

    extent = bbox.get_extent()
    axis_size = float(max(extent) * 0.2) if num_points > 0 else 0.1
    axis_size = max(axis_size, 0.05)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0.0, 0.0, 0.0]
    )

    ray_origin = np.array(ray_origin, dtype=np.float64)
    ray_dir = np.array(ray_dir, dtype=np.float64)

    # Auto-estimate cylinder length from point cloud
    cyl_height = estimate_cylinder_height(pts, ray_origin, ray_dir, min_height=max(0.20, axis_size * 2.0))

    cyl = create_ray_cylinder(
        ray_origin=ray_origin,
        ray_dir=ray_dir,
        radius=float(cylinder_radius),
        height=float(cyl_height),
        color=(0.2, 0.8, 0.2)  # green-ish
    )
    ray_line = create_ray_line(ray_origin, ray_dir, length=cyl_height, color=(0.0, 0.0, 0.0))

    # Use an interactive viewer so we can toggle cylinder with key 'C'
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Inspector", width=1280, height=720)

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(ray_line)

    state = {"cyl_visible": False}
    if show_cylinder:
        vis.add_geometry(cyl)
        state["cyl_visible"] = True

    def _toggle_cylinder(_vis):
        if state["cyl_visible"]:
            _vis.remove_geometry(cyl, reset_bounding_box=False)
            state["cyl_visible"] = False
        else:
            _vis.add_geometry(cyl)
            state["cyl_visible"] = True
        return False

    # Press 'C' to toggle cylinder
    vis.register_key_callback(ord("C"), _toggle_cylinder)

    print("\nControls:")
    print("  - Press 'C' to toggle the ray-cylinder visualization.\n")

    vis.run()
    vis.destroy_window()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=str, required=False,
                    default="/home/MA_SmartGrip/Smartgrip/result/single/y/y1/output/offline_output/main_cluster_clean.ply",
                    help="Path to input .ply point cloud.")
    ap.add_argument("--show-cylinder", dest="show_cylinder", action="store_true", help="Show ray-cylinder at startup.")
    ap.add_argument("--no-cylinder", dest="show_cylinder", action="store_false", help="Hide ray-cylinder at startup.")
    ap.set_defaults(show_cylinder=True)

    ap.add_argument("--cyl-radius", type=float, default=0.03, help="Cylinder radius in point-cloud units (e.g., meters).")

    ap.add_argument("--ray-origin", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "Z"), help="Ray origin in point-cloud frame.")
    ap.add_argument("--ray-dir", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                    metavar=("DX", "DY", "DZ"), help="Ray direction in point-cloud frame (will be normalized).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_point_cloud(
        ply_path=args.ply,
        show_cylinder=args.show_cylinder,
        cylinder_radius=args.cyl_radius,
        ray_origin=args.ray_origin,
        ray_dir=args.ray_dir,
    )
