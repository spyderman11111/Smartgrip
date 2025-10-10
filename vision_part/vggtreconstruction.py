#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
from typing import Optional
import sys

# 如果 vggt 不在默认 sys.path，可按需启用这两行（确保路径正确）
VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
if os.path.isdir(VGGT_path) and VGGT_path not in sys.path:
    sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues


class VGGTReconstructor:
    """
    仅用 VGGT 生成点云（PLY + TXT），不依赖/不输出 COLMAP 模型。
    输入：images_dir（装图片的单个文件夹；若传场景根且含 images/，将自动下探）
    输出：out_dir/points.ply 与 out_dir/points_xyzrgb.txt（默认 out_dir=同级 vggt_output；不可写自动回退）
    """

    def __init__(
        self,
        images_dir: str,
        batch_size: int = 4,
        max_points: int = 100000,
        resolution: int = 518,
        seed: int = 42,
        conf_thresh: float = 3.5,
        img_limit: Optional[int] = None,
        out_dir: Optional[str] = None,
        write_txt: bool = True,                 # 新增：是否输出 TXT
        txt_filename: str = "points_xyzrgb.txt" # 新增：TXT 文件名
    ):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.max_points = max_points
        self.resolution = resolution
        self.seed = seed
        self.conf_thresh = conf_thresh
        self.img_limit = img_limit
        self.write_txt = write_txt
        self.txt_filename = txt_filename
        self.last_paths = {}  # 运行后填充 {'ply': ..., 'txt': ...}

        # 随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 设备与 dtype（CPU 上用 float32 更稳）
        if torch.cuda.is_available():
            self.device = "cuda"
            major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # 加载模型
        self.model = VGGT()
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        )
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        # 输出目录（同级 -> images_dir 内 -> HOME）
        resolved_images_dir = self._resolve_images_dir(self.images_dir)
        self.out_dir = self._choose_out_dir(resolved_images_dir, out_dir)

    @staticmethod
    def _resolve_images_dir(path: str) -> str:
        """若传场景根且存在 images/，自动下探；否则直接使用传入目录。"""
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "images")):
            return os.path.join(path, "images")
        return path

    @staticmethod
    def _gather_image_paths(images_dir: str):
        """只收集常见图片扩展名的文件，忽略子目录。"""
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        paths = []
        for p in glob.glob(os.path.join(images_dir, "*")):
            if os.path.isfile(p) and p.lower().endswith(exts):
                paths.append(p)
        return sorted(paths)

    @staticmethod
    def _choose_out_dir(images_dir: str, out_dir: Optional[str]) -> str:
        candidates = []
        if out_dir:
            candidates.append(out_dir)
        else:
            parent = os.path.dirname(os.path.abspath(images_dir.rstrip("/")))
            candidates.append(os.path.join(parent, "vggt_output"))                     # 1) 同级
            candidates.append(os.path.join(os.path.abspath(images_dir), "vggt_output"))# 2) 放进 images_dir
            candidates.append(os.path.join(os.path.expanduser("~"), "vggt_output"))    # 3) HOME

        last_err = None
        for cand in candidates:
            try:
                os.makedirs(cand, exist_ok=True)
                print(f"[Info] 输出目录：{cand}")
                return cand
            except PermissionError as e:
                last_err = e
                print(f"[Warn] 无法创建输出目录：{cand}（{e}），尝试下一个。")
        raise last_err or PermissionError("无法创建任何输出目录，请手动指定 out_dir 到可写路径。")

    def _run_model_on_batch(self, images: torch.Tensor):
        """
        输入 images: [N, 3, H, W]（已 to(self.device)）
        返回：
          extrinsic: [N,4,4]  (cam_from_world)
          intrinsic: [N,3,3]
          depth_map: [N,1,H',W']  H'=W'=self.resolution
          depth_conf:[N,1,H',W']
        """
        images = F.interpolate(images, size=(self.resolution, self.resolution),
                               mode="bilinear", align_corners=False)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    images_5d = images[None]  # [1,N,3,H,W]
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images_5d)
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_5d.shape[-2:])
                    depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_5d, ps_idx)
            else:
                images_5d = images[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_5d)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_5d.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_5d, ps_idx)

        return (
            extrinsic.squeeze(0).cpu().numpy(),
            intrinsic.squeeze(0).cpu().numpy(),
            depth_map.squeeze(0).cpu().numpy(),
            depth_conf.squeeze(0).cpu().numpy(),
        )

    def _save_txt_xyzrgb(self, points_xyz: np.ndarray, colors_rgb: np.ndarray, path: str):
        """
        保存为 ASCII 文本：每行 x y z r g b
        - x,y,z: float，保留 6 位小数
        - r,g,b: int
        """
        # 过滤非法数
        finite_mask = np.isfinite(points_xyz).all(axis=1)
        P = points_xyz[finite_mask]
        C = colors_rgb[finite_mask]

        arr = np.column_stack([P.astype(np.float32), C.astype(np.int32)])
        header = f"# VGGT points XYZRGB\n# columns: x y z r g b\n# count: {arr.shape[0]}"
        np.savetxt(path, arr, fmt="%.6f %.6f %.6f %d %d %d", header=header)
        return arr.shape[0]

    def run(self) -> str:
        """执行重建，导出 PLY（始终）与 TXT（可选）。返回生成的 PLY 路径；全部输出见 self.last_paths。"""
        images_dir = self._resolve_images_dir(self.images_dir)
        image_paths = self._gather_image_paths(images_dir)
        if self.img_limit is not None:
            image_paths = image_paths[:self.img_limit]
        if not image_paths:
            raise FileNotFoundError(f"在 {images_dir} 下没有找到图片文件。")

        all_points, all_colors = [], []

        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i:i + self.batch_size]

            # 读图（1024 边长），VGGT 内部按 518 推理
            images, _original_coords = load_and_preprocess_images_square(batch_paths, 1024)
            images = images.to(self.device)

            extrinsics, intrinsics, depth_map, depth_conf = self._run_model_on_batch(images)
            # depth_map: [N,1,H',W']；unproject 得到世界坐标点云 [N,H',W',3]
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)

            # 置信度筛选 + 随机下采样
            mask = depth_conf >= self.conf_thresh
            mask = randomly_limit_trues(mask, self.max_points)

            # 提取颜色（按同分辨率重采样）
            images_resized = F.interpolate(images, size=(self.resolution, self.resolution),
                                           mode="bilinear", align_corners=False)
            rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[mask]

            all_points.append(points_3d[mask])
            all_colors.append(rgb)

        # 汇总与导出
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        # PLY
        ply_path = os.path.join(self.out_dir, "points.ply")
        trimesh.PointCloud(all_points, colors=all_colors).export(ply_path)
        print(f"[Done] 点云 PLY 已保存: {ply_path}  （共 {all_points.shape[0]} 点）")

        # TXT（可选）
        txt_path = None
        if self.write_txt:
            txt_path = os.path.join(self.out_dir, self.txt_filename)
            n = self._save_txt_xyzrgb(all_points, all_colors, txt_path)
            print(f"[Done] 点云 TXT 已保存: {txt_path}  （共 {n} 行）")

        # 记录输出路径
        self.last_paths = {"ply": ply_path, "txt": txt_path}

        return ply_path


if __name__ == "__main__":
    # 示例：传“真正装图片的文件夹”或“包含 images/ 子目录的场景根”
    reconstructor = VGGTReconstructor(
        images_dir="/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/ur5image",
        batch_size=4,
        max_points=100000,
        resolution=518,
        conf_thresh=2.0,
        img_limit=None,
        out_dir=None,          # None => 默认同级 vggt_output（不可写自动回退）
        write_txt=True,        # 导出 TXT
        txt_filename="points_xyzrgb.txt"
    )
    with torch.no_grad():
        reconstructor.run()
