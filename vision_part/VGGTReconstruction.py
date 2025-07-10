import os
import sys
import glob
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import pycolmap

VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


class VGGTReconstruction:
    def __init__(self, scene_dir: str, use_ba: bool = False, seed: int = 42, **kwargs):
        self.scene_dir = scene_dir
        self.image_dir = os.path.join(scene_dir, "images")
        self.sparse_dir = os.path.join(scene_dir, "sparse")
        os.makedirs(self.sparse_dir, exist_ok=True)

        self.use_ba = use_ba
        self.seed = seed
        self.config = {
            'max_reproj_error': kwargs.get('max_reproj_error', 8.0),
            'shared_camera': kwargs.get('shared_camera', False),
            'camera_type': kwargs.get('camera_type', 'SIMPLE_PINHOLE'),
            'vis_thresh': kwargs.get('vis_thresh', 0.2),
            'query_frame_num': kwargs.get('query_frame_num', 8),
            'max_query_pts': kwargs.get('max_query_pts', 4096),
            'fine_tracking': kwargs.get('fine_tracking', True),
            'conf_thres_value': kwargs.get('conf_thres_value', 5.0)
        }

        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._set_seed()
        self._load_model()

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def _load_model(self):
        self.model = VGGT().to(self.device)
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()

    def run(self):
        image_paths = glob.glob(os.path.join(self.image_dir, '*'))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        images, original_coords = load_and_preprocess_images_square(image_paths, 1024)
        images = images.to(self.device)
        original_coords = original_coords.to(self.device)

        extrinsic, intrinsic, depth_map, depth_conf = self._run_vggt(images)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

        if self.use_ba:
            reconstruction, points_rgb = self._run_ba(images, depth_conf, extrinsic, intrinsic, points_3d)
        else:
            reconstruction, points_rgb, points_3d = self._run_no_ba(images, depth_conf, extrinsic, intrinsic, points_3d)

        image_filenames = [os.path.basename(p) for p in image_paths]
        reconstruction = self._rescale_colmap(reconstruction, image_filenames, original_coords.cpu().numpy())
        reconstruction.write(self.sparse_dir)
        trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(self.sparse_dir, 'points.ply'))

    def _run_vggt(self, images, resolution=518):
        images_resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_resized = images_resized[None]  # B=1
                tokens, ps_idx = self.model.aggregator(images_resized)
                pose_enc = self.model.camera_head(tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_resized.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(tokens, images_resized, ps_idx)

        return (extrinsic.squeeze(0).cpu().numpy(),
                intrinsic.squeeze(0).cpu().numpy(),
                depth_map.squeeze(0).cpu().numpy(),
                depth_conf.squeeze(0).cpu().numpy())

    def _run_ba(self, images, depth_conf, extrinsic, intrinsic, points_3d):
        pred_tracks, vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d,
            masks=None,
            max_query_pts=self.config['max_query_pts'],
            query_frame_num=self.config['query_frame_num'],
            keypoint_extractor="aliked+sp",
            fine_tracking=self.config['fine_tracking'],
        )

        image_size = np.array(images.shape[-2:])
        scale = 1024 / 518
        intrinsic[:, :2, :] *= scale
        track_mask = vis_scores > self.config['vis_thresh']

        reconstruction, _ = batch_np_matrix_to_pycolmap(
            points_3d, extrinsic, intrinsic, pred_tracks, image_size, masks=track_mask,
            max_reproj_error=self.config['max_reproj_error'],
            shared_camera=self.config['shared_camera'],
            camera_type=self.config['camera_type'],
            points_rgb=points_rgb
        )
        if reconstruction is None:
            raise ValueError("Reconstruction failed in BA mode")

        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        return reconstruction, points_rgb

    def _run_no_ba(self, images, depth_conf, extrinsic, intrinsic, points_3d):
        mask = depth_conf >= self.config['conf_thres_value']
        mask = randomly_limit_trues(mask, 100000)

        points_3d = points_3d[mask]
        points_rgb = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[mask]
        points_xyf = create_pixel_coordinate_grid(*points_3d.shape[:3])[mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d, points_xyf, points_rgb, extrinsic, intrinsic,
            np.array([518, 518]),
            shared_camera=False,
            camera_type='PINHOLE'
        )
        return reconstruction, points_rgb, points_3d

    def _rescale_colmap(self, reconstruction, image_filenames, original_coords, img_size=518):
        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_filenames[pyimageid - 1]

            real_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_size) / img_size
            pycamera.params *= resize_ratio
            pycamera.params[-2:] = real_size / 2
            pycamera.width = real_size[0]
            pycamera.height = real_size[1]

        return reconstruction
    
if __name__ == "__main__":
    reconstructor = VGGTReconstruction(
        scene_dir="/path/to/your/project/input",  # must contain a folder named 'images/'
        use_ba=True,
        camera_type="SIMPLE_PINHOLE",
        query_frame_num=8,
        vis_thresh=0.2
    )
    reconstructor.run()