import open3d as o3d

ply_path = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene/sparse/points.ply"
pcd = o3d.io.read_point_cloud(ply_path)

print(pcd)  # 输出点的数量、是否有颜色等
o3d.visualization.draw_geometries([pcd])