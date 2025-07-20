import os
import open3d as o3d
import numpy as np


def load_points3D_txt(txt_path):
    points = []
    colors = []
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


def load_camera_poses(images_txt, cameras_txt, scale=0.05):
    camera_lines = []

    with open(images_txt, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        elems = line.split()
        if len(elems) < 10:
            i += 1
            continue

        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        cam_pos = np.array([tx, ty, tz])

        # 简单视图方向向前（Z轴）
        forward = np.array([0, 0, scale])
        pts = [
            cam_pos,
            cam_pos + np.array([scale, 0, 0]),
            cam_pos + np.array([0, scale, 0]),
            cam_pos + forward
        ]
        lineset = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        cam_line = o3d.geometry.LineSet()
        cam_line.points = o3d.utility.Vector3dVector(pts)
        cam_line.lines = o3d.utility.Vector2iVector(lineset)
        cam_line.colors = o3d.utility.Vector3dVector(colors)
        camera_lines.append(cam_line)

        i += 2  # Skip the next line (2D point list)

    return camera_lines


def visualize_colmap_scene(ply_path, txt_dir):
    vis_objs = []

    # Load PLY point cloud (optional, may be sparse or denser)
    if os.path.exists(ply_path):
        print(f"Loading PLY file: {ply_path}")
        pcd_ply = o3d.io.read_point_cloud(ply_path)
        vis_objs.append(pcd_ply)

    # Load points3D.txt
    points_txt = os.path.join(txt_dir, "points3D.txt")
    if os.path.exists(points_txt):
        print(f"Loading COLMAP TXT point cloud: {points_txt}")
        pcd_txt = load_points3D_txt(points_txt)
        vis_objs.append(pcd_txt)

    # Load cameras (images.txt + cameras.txt)
    images_txt = os.path.join(txt_dir, "images.txt")
    cameras_txt = os.path.join(txt_dir, "cameras.txt")
    if os.path.exists(images_txt) and os.path.exists(cameras_txt):
        print("Loading camera poses...")
        cam_lines = load_camera_poses(images_txt, cameras_txt)
        vis_objs.extend(cam_lines)

    if len(vis_objs) == 0:
        print("Nothing to visualize.")
        return

    # Launch Open3D visualizer
    o3d.visualization.draw_geometries(vis_objs)


if __name__ == "__main__":
    ply_path = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene/sparse/points.ply"
    txt_dir = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene/sparse_txt"
    visualize_colmap_scene(ply_path, txt_dir)
