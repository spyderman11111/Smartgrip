import open3d as o3d
import numpy as np

def inspect_point_cloud(ply_path):
    # Load PLY point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    # Basic info
    pts = np.asarray(pcd.points)
    num_points = pts.shape[0]
    has_color = pcd.has_colors()
    bbox = pcd.get_axis_aligned_bounding_box()

    print(f"Loaded point cloud: {ply_path}")
    print(f"Number of points: {num_points}")
    print(f"Has color: {has_color}")
    print("Bounding box:")
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")

    # Create a world coordinate frame (X=red, Y=green, Z=blue)
    # Make the axis size proportional to the point cloud extent for better visibility
    extent = bbox.get_extent()
    axis_size = float(max(extent) * 0.2) if num_points > 0 else 0.1
    axis_size = max(axis_size, 0.05)  # avoid too small

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0.0, 0.0, 0.0]
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, world_frame])

if __name__ == "__main__":
    ply_file = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/offline_output/main_cluster_clean.ply"
    inspect_point_cloud(ply_file)
