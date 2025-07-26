import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def load_points3D_txt(txt_path: str) -> o3d.geometry.PointCloud:
    """
    Load COLMAP-format points3D.txt into Open3D PointCloud.
    """
    points, colors = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            elems = line.strip().split()
            x, y, z = map(float, elems[1:4])
            r, g, b = map(int, elems[4:7])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def create_center_sphere(center: np.ndarray, color=[0, 0, 0], radius=0.005) -> o3d.geometry.TriangleMesh:
    """
    Create a small sphere to visualize the cluster center.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def cluster_by_xyz(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.02,
    min_points: int = 50,
    voxel_size: float = 0.005
):
    """
    Perform DBSCAN clustering on 3D points (only using XYZ), after voxel downsampling.
    Returns list of (cluster point cloud, cluster center, color).
    """
    # Step 1: Downsample point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd_down.points)

    # Step 2: DBSCAN clustering
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(points)
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters")

    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        cluster = pcd_down.select_by_index(indices)
        color = np.random.rand(3)
        cluster.paint_uniform_color(color)
        center = np.mean(np.asarray(cluster.points), axis=0)
        clusters.append((cluster, center, color))
    return clusters


def visualize_clusters(pcd_path: str, eps: float, min_points: int, voxel_size: float, center_radius: float):
    """
    Load, cluster, and visualize 3D point cloud with cluster coloring and center spheres.
    """
    pcd = load_points3D_txt(pcd_path)
    clusters = cluster_by_xyz(pcd, eps=eps, min_points=min_points, voxel_size=voxel_size)

    vis_objs = []
    for i, (cluster, center, _) in enumerate(clusters):
        vis_objs.append(cluster)
        vis_objs.append(create_center_sphere(center, color=[0, 0, 0], radius=center_radius))
        print(f"Cluster {i+1}: Center = {center}, Points = {len(cluster.points)}")

    o3d.visualization.draw_geometries(vis_objs)


if __name__ == "__main__":
    # ===== 可调参数区域 =====
    pcd_txt_path = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene/sparse_txt/points3D.txt"
    voxel_size = 0.005      # 降采样体素尺寸（单位：米）
    eps = 0.02              # DBSCAN 空间距离阈值（单位：米）
    min_points = 50         # 聚类最小点数
    center_radius = 0.005   # 中心球体半径

    # ===== 执行入口 =====
    visualize_clusters(pcd_txt_path, eps, min_points, voxel_size, center_radius)
