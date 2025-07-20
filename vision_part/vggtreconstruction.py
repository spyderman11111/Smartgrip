# vggtreconstruction.py
# Step-by-step VGGT scene reconstruction script with COLMAP model output

import random
import numpy as np
import glob
import os
import sys
import copy
import torch
import torch.nn.functional as F

VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
sys.path.append(VGGT_path)

# Step 1: Import dependencies and configure environment
import argparse
from pathlib import Path
import trimesh
import pycolmap
import subprocess

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


def run_VGGT(model, images, dtype, resolution=518):
    # Step 2: Run VGGT model to predict camera and depth
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # [1, B, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


def convert_colmap_bin_to_txt(sparse_dir, output_txt_dir):
    # Step 5: Export COLMAP bin model to TXT format for easy downstream use
    os.makedirs(output_txt_dir, exist_ok=True)
    cmd = [
        "colmap", "model_converter",
        "--input_path", sparse_dir,
        "--output_path", output_txt_dir,
        "--output_type", "TXT"
    ]
    subprocess.run(cmd, check=True)


def rename_and_rescale(reconstruction, image_paths, original_coords, img_size, shared_camera):
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]
        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp
            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]
            if shared_camera:
                rescale_camera = False
    return reconstruction


def main():
    # Step 1: Set configuration parameters
    class Args:
        scene_dir = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene"
        use_ba = False
        seed = 42
        max_reproj_error = 8.0
        shared_camera = False
        camera_type = "SIMPLE_PINHOLE"
        vis_thresh = 0.2
        query_frame_num = 8
        max_query_pts = 4096
        fine_tracking = True
        conf_thres_value = 5.0
    args = Args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 2: Load VGGT model
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model.eval().to(device)

    # Step 3: Load images
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))[:3]
    base_image_names = [os.path.basename(p) for p in image_path_list]
    images, original_coords = load_and_preprocess_images_square(image_path_list, 1024)
    images = images.to(device)

    # Step 4: VGGT inference to COLMAP structure
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    mask = depth_conf >= args.conf_thres_value
    mask = randomly_limit_trues(mask, 100000)
    points_3d = points_3d[mask]
    images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[mask]
    coords = create_pixel_coordinate_grid(*points_3d.shape[:3])[mask]

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, coords, rgb, extrinsic, intrinsic,
        np.array([518, 518]), shared_camera=args.shared_camera,
        camera_type=args.camera_type)

    reconstruction = rename_and_rescale(
        reconstruction, base_image_names, original_coords.cpu().numpy(),
        img_size=518, shared_camera=args.shared_camera)

    # Step 5: Save results to disk
    sparse_dir = os.path.join(args.scene_dir, "sparse")
    sparse_txt_dir = os.path.join(args.scene_dir, "sparse_txt")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    trimesh.PointCloud(points_3d, colors=rgb).export(os.path.join(sparse_dir, "points.ply"))
    convert_colmap_bin_to_txt(sparse_dir, sparse_txt_dir)

    print("Reconstruction saved in:", sparse_dir)
    print("TXT format exported to:", sparse_txt_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
