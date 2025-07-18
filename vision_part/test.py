import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh

VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def generate_pointcloud_from_vggt(image_paths, conf_threshold=5.0):
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # Add batch dimension: [1, N, C, H, W]
            aggregated_tokens, ps_idx = model.aggregator(images)

            # Predict camera pose
            pose_enc = model.camera_head(aggregated_tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            extrinsic = extrinsic.squeeze(0)
            intrinsic = intrinsic.squeeze(0)

            # Predict depth
            depth_map, depth_conf = model.depth_head(aggregated_tokens, images, ps_idx)
            depth_map = depth_map.squeeze(0)        # [N, H, W, 1]
            depth_conf = depth_conf.squeeze(0)      # [N, H, W]

            # Predict point map (optional)
            point_map, point_conf = model.point_head(aggregated_tokens, images, ps_idx)

            # Compute 3D points from depth
            point_map_unproj = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # 筛选置信度高的点
    valid_mask = depth_conf > conf_threshold
    point_map_np = point_map_unproj if isinstance(point_map_unproj, np.ndarray) else point_map_unproj.cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()

    pointcloud_xyz = point_map_np[valid_mask_np]  # [num_points, 3]

    # 保存点云
    ply_path = "pointcloud_vggt.ply"
    trimesh.PointCloud(pointcloud_xyz).export(ply_path)
    print(f"[PLY SAVED] 3D point cloud saved to {ply_path}")

    return pointcloud_xyz, depth_conf.cpu().numpy(), point_map_np

def visualize_pointcloud(points_3d_frame, conf_frame=None, conf_threshold=5.0, save_path="pointcloud_frame0.png"):
    if conf_frame is not None:
        mask = conf_frame > conf_threshold
        xyz = points_3d_frame[mask]
    else:
        xyz = points_3d_frame.reshape(-1, 3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c=xyz[:, 2], cmap='viridis')
    ax.set_title("VGGT 3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[VISUALIZED] Frame point cloud saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    image_paths = [
        "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images/frame_00000.jpg",
        "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images/frame_00020.jpg",
        "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images/frame_00040.jpg"
    ]

    points, confs, full_map = generate_pointcloud_from_vggt(image_paths)
    visualize_pointcloud(full_map[0], confs[0], save_path="pointcloud_frame0.png")
