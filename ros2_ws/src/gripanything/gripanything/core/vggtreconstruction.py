#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
from PIL import Image as PILImage

# 如果 vggt 不在默认 sys.path，可按需启用这两行（确保路径正确）
import sys
VGGT_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
if os.path.isdir(VGGT_path) and os.path.isdir(VGGT_path) and VGGT_path not in sys.path:
    sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues


# ======================== 用户相机参数 ========================
@dataclass
class Camera:
    fx: float = 2674.3803723910564
    fy: float = 2667.4211254043507
    cx: float = 954.5922081613583
    cy: float = 1074.965947832258
    hand_eye_frame: str = 'optical'  # 仅记录
    t_tool_cam_xyz: Tuple[float, float, float] = (-0.000006852374024, -0.059182661943126947, -0.00391824813032688)
    t_tool_cam_quat_xyzw: Tuple[float, float, float, float] = (-0.0036165657530785695, -0.000780788838366878,
                                                                0.7078681983794892, 0.7063348529868249)
    flip_x: bool = False  # 若保存图像时做了镜像翻转，按实际填写；此脚本里仅记录，不改像素
    flip_y: bool = False


# ======================== 主类 ========================
class VGGTReconstructor:
    """
    自动化流程：
      1) 优先读取 JSON（每帧相机位姿 + 图像路径），并使用用户内参生成 K。
      2) VGGT 仅用于估计 depth_map；点云世界坐标用(外参,内参)做反投影。
      3) 若找不到 JSON 或图片，回退到纯 VGGT（自估内外参）。

    产物：
      - points.ply
      - points_xyzrgb.txt（可选）
      - meta.json（记录使用的模式、外参/内参来源等）
    """

    def __init__(
        self,
        images_dir: str,
        json_path: Optional[str] = None,
        batch_size: int = 4,
        max_points: int = 100000,
        resolution: int = 518,
        seed: int = 42,
        conf_thresh: float = 3.5,
        img_limit: Optional[int] = None,
        out_dir: Optional[str] = None,
        write_txt: bool = True,
        txt_filename: str = "points_xyzrgb.txt",
        camera: Optional[Camera] = None,
        prefer_external_poses: bool = True,     # 优先使用 JSON 中外参
        use_known_intrinsics: bool = True,      # 使用你的内参
    ):
        self.images_dir = images_dir
        self.json_path = json_path
        self.batch_size = batch_size
        self.max_points = max_points
        self.resolution = resolution
        self.seed = seed
        self.conf_thresh = conf_thresh
        self.img_limit = img_limit
        self.write_txt = write_txt
        self.txt_filename = txt_filename
        self.camera = camera or Camera()
        self.prefer_external_poses = prefer_external_poses
        self.use_known_intrinsics = use_known_intrinsics
        self.last_paths = {}

        # 随机种子
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        # 设备与 dtype（CPU 上用 float32 更稳）
        if torch.cuda.is_available():
            self.device = "cuda"
            major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # 模型
        self.model = VGGT()
        state = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        # 输出目录
        resolved_images_dir = self._resolve_images_dir(self.images_dir)
        self.out_dir = self._choose_out_dir(resolved_images_dir, out_dir)

        # 数据源（外参+图片）尝试加载
        self.external = self._try_load_external_trajectory(self.json_path, resolved_images_dir)

    # ---------- 路径/IO ----------
    @staticmethod
    def _resolve_images_dir(path: str) -> str:
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "images")):
            return os.path.join(path, "images")
        return path

    @staticmethod
    def _gather_image_paths(images_dir: str):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        paths = [p for p in glob.glob(os.path.join(images_dir, "*"))
                 if os.path.isfile(p) and p.lower().endswith(exts)]
        return sorted(paths)

    @staticmethod
    def _choose_out_dir(images_dir: str, out_dir: Optional[str]) -> str:
        candidates = [out_dir] if out_dir else []
        parent = os.path.dirname(os.path.abspath(images_dir.rstrip("/")))
        candidates += [
            os.path.join(parent, "vggt_output"),
            os.path.join(os.path.abspath(images_dir), "vggt_output"),
            os.path.join(os.path.expanduser("~"), "vggt_output"),
        ]
        last_err = None
        for cand in candidates:
            if not cand: continue
            try:
                os.makedirs(cand, exist_ok=True)
                print(f"[Info] 输出目录：{cand}")
                return cand
            except PermissionError as e:
                last_err = e
                print(f"[Warn] 无法创建输出目录：{cand}（{e}），尝试下一个。")
        raise last_err or PermissionError("无法创建任何输出目录，请手动指定 out_dir 到可写路径。")

    # ---------- JSON 读取 & 外参与图片列表 ----------
    @staticmethod
    def _parse_index(k: str) -> Optional[int]:
        # "image_12" -> 12
        try:
            return int(k.split("_", 1)[1])
        except Exception:
            return None

    def _try_load_external_trajectory(self, json_path: Optional[str], images_dir: str):
        """
        读取 image_jointstates.json，提取每帧 base->camera_optical 的 R、t，
        并生成 world_from_cam（extrinsic，直接=R_bc,t_bc）与对应图片路径。
        返回结构：
          {
            "enabled": True/False,
            "image_paths": [str...],
            "extrinsics_wfc": np.ndarray [N,4,4],  # world_from_cam
            "stamps": [{"sec":..,"nanosec":..}, ...]
          }
        """
        ext = {"enabled": False, "image_paths": [], "extrinsics_wfc": None, "stamps": []}
        if not self.prefer_external_poses:
            return ext

        # 默认 JSON 路径
        if json_path is None:
            json_path = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/image_jointstates.json"

        if not os.path.isfile(json_path):
            print(f"[Info] 未找到 JSON：{json_path}，将回退到纯 VGGT。")
            return ext

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warn] 读取 JSON 失败：{e}，回退到纯 VGGT。")
            return ext

        shots: Dict[str, Any] = data.get("shots", {})
        # 按 image_序号排序
        items = []
        for k, v in shots.items():
            idx = self._parse_index(k)
            if idx is None:
                continue
            items.append((idx, v))
        items.sort(key=lambda x: x: x[0])

        image_paths, extrinsics_44, stamps = [], [], []
        # 默认图片目录（ur5image）
        default_img_dir = os.path.join(os.path.dirname(json_path), "ur5image")

        for idx, entry in items:
            # 组图像路径（存在才用）
            cand_paths = [
                os.path.join(images_dir, f"pose_{idx}_image.png"),
                os.path.join(default_img_dir, f"pose_{idx}_image.png"),
            ]
            img_path = next((p for p in cand_paths if os.path.isfile(p)), None)
            if img_path is None:
                # 找不到该帧图片就跳过
                continue

            # R_bc, t_bc  （base <- camera_optical）
            cam = entry.get("camera_pose", {})
            R_bc = np.array(cam.get("R", []), dtype=float)
            t_bc = np.array(cam.get("t", []), dtype=float).reshape(-1)
            if R_bc.shape != (3, 3) or t_bc.shape != (3,):
                # 数据不完整，跳过该帧
                continue

            # world_from_cam（X_w = R_wc X_c + t_wc）=（R_bc, t_bc），不要取逆
            T = np.eye(4, dtype=float)
            T[:3, :3] = R_bc
            T[:3, 3] = t_bc

            image_paths.append(img_path)
            extrinsics_44.append(T)
            stamps.append(cam.get("stamp", None))

        if not image_paths:
            print(f"[Info] JSON 中没有有效帧或找不到对应图片，回退到纯 VGGT。")
            return ext

        ext["enabled"] = True
        ext["image_paths"] = image_paths
        ext["extrinsics_wfc"] = np.stack(extrinsics_44, axis=0)
        ext["stamps"] = stamps
        print(f"[Info] 已加载外参与图片：{len(image_paths)} 帧。")
        return ext

    # ---------- K 构造：原图尺寸 -> 网络分辨率 ----------
    def _build_intrinsics_from_camera(self, batch_image_paths: List[str], res: int) -> np.ndarray:
        Ks = []
        for p in batch_image_paths:
            with PILImage.open(p) as im:
                W, H = im.size
            sx = res / float(W)
            sy = res / float(H)
            fx = self.camera.fx * sx
            fy = self.camera.fy * sy
            cx = self.camera.cx * sx
            cy = self.camera.cy * sy
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
            Ks.append(K)
        return np.stack(Ks, axis=0)

    # ---------- VGGT 前向 ----------
    def _run_model_on_batch(self, images: torch.Tensor):
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
            extrinsic.squeeze(0).cpu().numpy(),   # [N,4,4] (不一定使用)
            intrinsic.squeeze(0).cpu().numpy(),   # [N,3,3] (不一定使用)
            depth_map.squeeze(0).cpu().numpy(),   # [N,1,R,R]
            depth_conf.squeeze(0).cpu().numpy(),  # [N,1,R,R]
        )

    # ---------- 输出 ----------
    @staticmethod
    def _save_txt_xyzrgb(points_xyz: np.ndarray, colors_rgb: np.ndarray, path: str):
        finite_mask = np.isfinite(points_xyz).all(axis=1)
        P = points_xyz[finite_mask]
        C = colors_rgb[finite_mask]
        arr = np.column_stack([P.astype(np.float32), C.astype(np.int32)])
        header = f"# VGGT points XYZRGB\n# columns: x y z r g b\n# count: {arr.shape[0]}"
        np.savetxt(path, arr, fmt="%.6f %.6f %.6f %d %d %d", header=header)
        return arr.shape[0]

    @staticmethod
    def _dump_meta(meta_path: str, meta: dict):
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Warn] 写元数据失败：{e}")

    # ---------- 主流程 ----------
    def run(self) -> str:
        meta_logs = {
            "mode": "external_poses" if self.external["enabled"] else "pure_vggt",
            "resolution": self.resolution,
            "conf_thresh": self.conf_thresh,
            "use_known_intrinsics": self.use_known_intrinsics,
            "camera": {
                "fx": self.camera.fx, "fy": self.camera.fy,
                "cx": self.camera.cx, "cy": self.camera.cy,
                "flip_x": self.camera.flip_x, "flip_y": self.camera.flip_y,
            },
            "batches": []
        }

        # 准备图像列表
        if self.external["enabled"]:
            image_paths_all = self.external["image_paths"]
            extrinsics_all = self.external["extrinsics_wfc"]   # [N,4,4] world_from_cam
        else:
            images_dir = self._resolve_images_dir(self.images_dir)
            image_paths_all = self._gather_image_paths(images_dir)
            extrinsics_all = None  # 纯 VGGT 模式

        if self.img_limit is not None:
            image_paths_all = image_paths_all[:self.img_limit]
            if extrinsics_all is not None:
                extrinsics_all = extrinsics_all[:self.img_limit]

        if not image_paths_all:
            raise FileNotFoundError("没有可用图片。")

        all_points, all_colors = [], []

        for i in tqdm(range(0, len(image_paths_all), self.batch_size)):
            batch_paths = image_paths_all[i:i + self.batch_size]

            # 读图（1024 正方形），VGGT 内部再到 self.resolution
            images, _ = load_and_preprocess_images_square(batch_paths, 1024)
            images = images.to(self.device)

            extr_vggt, intr_vggt, depth_map, depth_conf = self._run_model_on_batch(images)

            # 选择使用的外参/内参
            if self.external["enabled"]:
                # 外参：来自 JSON（world_from_cam）
                extr_used = extrinsics_all[i:i + len(batch_paths)]
            else:
                # 退回 VGGT 自估外参
                extr_used = extr_vggt

            if self.use_known_intrinsics:
                intr_used = self._build_intrinsics_from_camera(batch_paths, self.resolution)
            else:
                intr_used = intr_vggt

            # 反投影到世界坐标（expects world_from_cam）
            points_3d = unproject_depth_map_to_point_map(depth_map, extr_used, intr_used)

            # 置信度筛选 + 下采样
            mask = depth_conf >= self.conf_thresh
            mask = randomly_limit_trues(mask, self.max_points)

            # 颜色（与深度同分辨率）
            images_resized = F.interpolate(images, size=(self.resolution, self.resolution),
                                           mode="bilinear", align_corners=False)
            rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)[mask]

            all_points.append(points_3d[mask])
            all_colors.append(rgb)

            # 记录样例
            meta_logs["batches"].append({
                "start_index": i,
                "count": len(batch_paths),
                "intrinsics_used_sample": intr_used[0].tolist(),
                "source_extrinsics": "external_json_wfc" if self.external["enabled"] else "vggt_pred",
            })

        # 汇总导出
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)

        ply_path = os.path.join(self.out_dir, "points.ply")
        trimesh.PointCloud(all_points, colors=all_colors).export(ply_path)
        print(f"[Done] 点云 PLY 已保存: {ply_path}  （共 {all_points.shape[0]} 点）")

        txt_path = None
        if self.write_txt:
            txt_path = os.path.join(self.out_dir, self.txt_filename)
            n = self._save_txt_xyzrgb(all_points, all_colors, txt_path)
            print(f"[Done] 点云 TXT 已保存: {txt_path}  （共 {n} 行）")

        # 元数据
        meta_path = os.path.join(self.out_dir, "meta.json")
        meta_logs["points_count"] = int(all_points.shape[0])
        meta_logs["outputs"] = {"ply": ply_path, "txt": txt_path}
        if self.external["enabled"]:
            meta_logs["external"] = {
                "json_path": self.json_path,
                "frames_used": len(self.external["image_paths"])
            }
        self._dump_meta(meta_path, meta_logs)

        self.last_paths = {"ply": ply_path, "txt": txt_path, "meta": meta_path}
        return ply_path


# ======================== 入口 ========================
if __name__ == "__main__":
    # 默认路径（可按需修改）
    JSON_DEFAULT = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/image_jointstates.json"
    IMAGES_DIR   = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/ur5image"

    cam = Camera()  # 使用你给定的内参

    reconstructor = VGGTReconstructor(
        images_dir=IMAGES_DIR,
        json_path=JSON_DEFAULT,
        batch_size=4,
        max_points=500000,
        resolution=518,
        conf_thresh=2.5,
        img_limit=None,
        out_dir=None,                 # None => 默认同级 vggt_output（不可写自动回退）
        write_txt=True,
        txt_filename="points_xyzrgb.txt",
        camera=cam,
        prefer_external_poses=True,   # 打开：优先用 JSON 外参（world_from_cam）
        use_known_intrinsics=True,    # 打开：使用你的内参
    )
    with torch.no_grad():
        reconstructor.run()
