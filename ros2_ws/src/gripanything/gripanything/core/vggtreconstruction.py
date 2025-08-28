import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pathlib import Path
from tqdm import tqdm
import sys
import copy

# VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
# sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


class VGGTReconstructor:
    def __init__(
        self,
        scene_dir: str,
        batch_size: int = 4,
        max_points: int = 100000,
        resolution: int = 518,
        seed: int = 42,
        use_ba: bool = True,
        max_reproj_error: float = 8.0,
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.2,
        query_frame_num: int = 8,
        max_query_pts: int = 4096,
        fine_tracking: bool = True,
        conf_thresh: float = 3.5,
        img_limit: int = None,
    ):
        self.scene_dir = scene_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.resolution = resolution
        self.seed = seed
        self.shared_camera = shared_camera
        self.camera_type = camera_type
        self.conf_thresh = conf_thresh
        self.img_limit = img_limit

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        self.model = VGGT()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
        self.model.eval().to(self.device)

    def run_model_on_batch(self, images):
        images = F.interpolate(images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
        return (
            extrinsic.squeeze(0).cpu().numpy(),
            intrinsic.squeeze(0).cpu().numpy(),
            depth_map.squeeze(0).cpu().numpy(),
            depth_conf.squeeze(0).cpu().numpy(),
        )

    def rescale_camera_info(self, reconstruction, image_paths, original_coords):
        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid - 1]

            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / self.resolution
            real_pp = real_image_size / 2

            pred_params = pred_params * resize_ratio
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

            if self.shared_camera:
                break
        return reconstruction

    def run(self):
        image_dir = os.path.join(self.scene_dir, "images")
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
        if self.img_limit is not None:
            image_paths = image_paths[:self.img_limit]
        base_names = [os.path.basename(p) for p in image_paths]

        all_points, all_colors, all_coords = [], [], []
        all_extrinsics, all_intrinsics = [], []
        all_image_names, all_original_coords = [], []

        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i:i + self.batch_size]
            image_names = base_names[i:i + self.batch_size]

            images, original_coords = load_and_preprocess_images_square(batch_paths, 1024)
            images = images.to(self.device)

            extrinsics, intrinsics, depth_map, depth_conf = self.run_model_on_batch(images)
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)

            mask = depth_conf >= self.conf_thresh
            mask = randomly_limit_trues(mask, self.max_points)

            images_resized = F.interpolate(images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
            rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[mask]
            num_frames, _, H, W = images_resized.shape
            coords = create_pixel_coordinate_grid(num_frames, H, W)[mask]

            all_points.append(points_3d[mask])
            all_colors.append(rgb)
            all_coords.append(coords)
            all_extrinsics.append(extrinsics)
            all_intrinsics.append(intrinsics)
            all_image_names.extend(image_names)
            all_original_coords.append(original_coords.cpu().numpy())

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_coords = np.concatenate(all_coords, axis=0)
        all_extrinsics = np.concatenate(all_extrinsics, axis=0)
        all_intrinsics = np.concatenate(all_intrinsics, axis=0)
        all_original_coords = np.concatenate(all_original_coords, axis=0)

        image_size = np.array([self.resolution, self.resolution])
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            all_points, all_coords, all_colors,
            all_extrinsics, all_intrinsics,
            image_size=image_size,
            shared_camera=self.shared_camera,
            camera_type=self.camera_type
        )

        reconstruction = self.rescale_camera_info(
            reconstruction,
            all_image_names,
            all_original_coords
        )

        sparse_dir = os.path.join(self.scene_dir, "sparse")
        sparse_txt_dir = os.path.join(self.scene_dir, "sparse_txt")
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(sparse_txt_dir, exist_ok=True)

        reconstruction.write(sparse_dir)
        reconstruction.write_text(sparse_txt_dir)
        trimesh.PointCloud(all_points, colors=all_colors).export(os.path.join(sparse_dir, "points.ply"))

        print("[Done] COLMAP bin saved to:", sparse_dir)
        print("[Done] TXT export saved to:", sparse_txt_dir)
        
if __name__ == "__main__":
    reconstructor = VGGTReconstructor(
        scene_dir="/media/MA_SmartGrip/Data/Smartgrip/vision_part/ur5e_images_scene",
        batch_size=4,
        max_points=100000,
        resolution=518,
        conf_thresh=3.5,
        img_limit=None
    )
    with torch.no_grad():
        reconstructor.run()
