import open3d as o3d
import numpy as np

def inspect_point_cloud(ply_path):
    # 加载 PLY 点云
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # 基本信息
    num_points = np.asarray(pcd.points).shape[0]
    has_color = pcd.has_colors()
    bbox = pcd.get_axis_aligned_bounding_box()

    print(f"Loaded point cloud: {ply_path}")
    print(f"Number of points: {num_points}")
    print(f"Has color: {has_color}")
    print(f"Bounding box:")
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")
    
    # 可视化
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    ply_file = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/vggt_output/points.ply"
    inspect_point_cloud(ply_file)
