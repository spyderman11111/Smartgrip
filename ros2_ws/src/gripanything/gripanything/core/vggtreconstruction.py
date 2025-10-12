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
if os.path.isdir(VGGT_path) and VGGT_path not in sys.path:
    sys.path.append(VGGT_path)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues, create_pixel_coordinate_grid

# 仅在导出 COLMAP / 执行 BA 时需要
_HAS_PYCOLMAP = False
try:
    import pycolmap  # type: ignore
    _HAS_PYCOLMAP = True
except Exception:
    pass

try:
    from vggt.dependency.track_predict import predict_tracks
    from vggt.dependency.np_to_pycolmap import (
        batch_np_matrix_to_pycolmap,
        batch_np_matrix_to_pycolmap_wo_track,
    )
    _HAS_TRACKER = True
except Exception:
    _HAS_TRACKER = False


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
      4) （可选）执行 tracking + BA（仅计算，不导出 COLMAP）。
    产物：
      - points.ply
      - meta.json（记录使用的模式、外参/内参来源等）
      - （可选）points_xyzrgb.txt（当前默认关闭）
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
        write_txt: bool = False,               # TXT 默认关闭
        txt_filename: str = "points_xyzrgb.txt",
        camera: Optional[Camera] = None,
        prefer_external_poses: bool = True,    # 优先使用 JSON 中外参
        use_known_intrinsics: bool = True,     # 使用你的内参

        # === 融合官方 demo 的可选开关（不使用 argparse，入口处手动改） ===
        use_ba: bool = True,                   # 开启 BA（计算，不导出 COLMAP）
        shared_camera: bool = False,
        camera_type: str = "SIMPLE_PINHOLE",
        vis_thresh: float = 0.2,
        query_frame_num: int = 8,
        max_query_pts: int = 4096,
        fine_tracking: bool = True,
        img_load_resolution: int = 1024,       # 读取图像的大分辨率（用于 tracking/重命名缩放）
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

        # BA/track 设置
        self.use_ba = use_ba
        self.shared_camera = shared_camera
        self.camera_type = camera_type
        self.vis_thresh = vis_thresh
        self.query_frame_num = query_frame_num
        self.max_query_pts = max_query_pts
        self.fine_tracking = fine_tracking
        self.img_load_resolution = img_load_resolution

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
        并生成（变量名沿用）extrinsics_cfw 与对应图片路径。
        返回结构：
          {
            "enabled": True/False,
            "image_paths": [str...],
            "extrinsics_cfw": np.ndarray [N,4,4],  # camera_from_world
            "stamps": [{"sec":..,"nanosec":..}, ...]
          }
        """
        ext = {"enabled": False, "image_paths": [], "extrinsics_cfw": None, "stamps": []}
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
        items.sort(key=lambda x: x[0])

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

            # 统一为 camera_from_world (T_cw)
            R_cw = R_bc.T
            t_cw = -R_bc.T @ t_bc
            T = np.eye(4, dtype=float)
            T[:3, :3] = R_cw
            T[:3, 3] = t_cw

            image_paths.append(img_path)
            extrinsics_44.append(T)
            stamps.append(cam.get("stamp", None))

        if not image_paths:
            print(f"[Info] JSON 中没有有效帧或找不到对应图片，回退到纯 VGGT。")
            return ext

        ext["enabled"] = True
        ext["image_paths"] = image_paths
        ext["extrinsics_cfw"] = np.stack(extrinsics_44, axis=0)
        ext["stamps"] = stamps
        print(f"[Info] 已加载外参与图片：{len(image_paths)} 帧。")
        return ext

    # ---------- K 构造：原图尺寸 -> 网络分辨率 ----------
    def _build_intrinsics_from_camera(self, batch_image_paths: List[str], res: int) -> np.ndarray:
        """
        与 load_and_preprocess_images_square(batch_paths, res) 对齐的 K：
        假设该函数把原图按最长边等比缩放到 res，并对另一边做居中填充（letterbox）。
        """
        Ks = []
        for p in batch_image_paths:
            with PILImage.open(p) as im:
                W, H = im.size

            s0 = float(res) / float(max(W, H))
            if W < H:
                pad_x = 0.5 * (res - W * s0)
                pad_y = 0.0
            elif H < W:
                pad_x = 0.0
                pad_y = 0.5 * (res - H * s0)
            else:
                pad_x = 0.0
                pad_y = 0.0

            cx = self.camera.cx * s0 + (pad_x if W < H else 0.0)
            cy = self.camera.cy * s0 + (0.5 * (res - H * s0) if H < W else 0.0)
            fx = self.camera.fx * s0
            fy = self.camera.fy * s0

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
            depth_map.squeeze(0).cpu().numpy(),   # e.g. [N,1,R,R] 或 [N,R,R,1]
            depth_conf.squeeze(0).cpu().numpy(),  # e.g. [N,1,R,R] 或 [N,R,R,1]
            images,                                # 已经 resize 到 (resolution, resolution) 的图像（取颜色）
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

    # ---------- COLMAP 重命名/缩放（与官方逻辑一致） ----------
    @staticmethod
    def _rename_colmap_and_rescale(reconstruction, image_paths, original_coords, img_size, shift_point2d=True, shared_camera=False):
        rescale_camera = True
        for pyimageid in reconstruction.images:
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid - 1]

            pred_params = np.array(pycamera.params, dtype=np.float64)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp
            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

            if shift_point2d:
                top_left = original_coords[pyimageid - 1, :2]
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                rescale_camera = False
        return reconstruction

    # ---------- 主流程 ----------
    def run(self) -> str:
        meta_logs = {
            "mode": "external_poses" if self.external["enabled"] else "pure_vggt",
            "resolution": self.resolution,
            "conf_thresh": self.conf_thresh,
            "use_known_intrinsics": self.use_known_intrinsics,
            "use_ba": self.use_ba,
            "img_load_resolution": self.img_load_resolution,
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
            extrinsics_all = self.external["extrinsics_cfw"]   # [N,4,4] camera_from_world
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

        # 若需要 BA，准备较大分辨率的加载与 original_coords（只计算，不落盘）
        if self.use_ba:
            images_full, original_coords = load_and_preprocess_images_square(image_paths_all, self.img_load_resolution)
            images_full = images_full.to(self.device)
            original_coords = original_coords.to(self.device)
            base_image_names = [os.path.basename(p) for p in image_paths_all]
        else:
            images_full = original_coords = None
            base_image_names = None

        all_points, all_colors = [], []

        # 供 BA/前馈构建所需的累积（仅内存计算）
        extr_list, intr_list = [], []
        depth_conf_list, pts3d_list, rgb_list = [], [], []

        for i in tqdm(range(0, len(image_paths_all), self.batch_size)):
            batch_paths = image_paths_all[i:i + self.batch_size]

            # 读图（按推理分辨率预处理）
            images, _ = load_and_preprocess_images_square(batch_paths, self.resolution)
            images = images.to(self.device)

            # 前向
            extr_vggt, intr_vggt, depth_map, depth_conf, images_resized = self._run_model_on_batch(images)

            # —— 把深度/置信度统一成 (N,H,W,1) —— #
            depth_map = np.asarray(depth_map)
            depth_conf = np.asarray(depth_conf)
            if depth_map.ndim == 4 and depth_map.shape[1] == 1:          # (N,1,H,W) -> (N,H,W,1)
                depth_map = np.transpose(depth_map, (0, 2, 3, 1))
            elif depth_map.ndim == 3:                                     # (N,H,W) -> (N,H,W,1)
                depth_map = depth_map[..., None]
            if depth_conf.ndim == 4 and depth_conf.shape[1] == 1:
                depth_conf = np.transpose(depth_conf, (0, 2, 3, 1))
            elif depth_conf.ndim == 3:
                depth_conf = depth_conf[..., None]
            elif depth_conf.ndim != 4 or depth_conf.shape[-1] != 1:
                depth_conf = np.ones_like(depth_map, dtype=depth_map.dtype)

            # 选择使用的外参/内参
            if self.external["enabled"]:
                extr_used = extrinsics_all[i:i + len(batch_paths)]
            else:
                extr_used = extr_vggt

            if self.use_known_intrinsics:
                intr_used = self._build_intrinsics_from_camera(batch_paths, self.resolution)
            else:
                intr_used = intr_vggt

            # 反投影到世界坐标
            points_3d = unproject_depth_map_to_point_map(depth_map, extr_used, intr_used)  # -> (N,H,W,3)

            # 置信度筛选 + 下采样（PLY）
            mask = (depth_conf[..., 0] >= self.conf_thresh)  # (N,H,W)
            mask = randomly_limit_trues(mask, self.max_points)

            rgb = (images_resized.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

            all_points.append(points_3d[mask])
            all_colors.append(rgb[mask])

            # BA 需要的累积
            if self.use_ba:
                extr_list.append(extr_used)
                intr_list.append(intr_used)
                depth_conf_list.append(depth_conf)
                pts3d_list.append(points_3d)
                rgb_list.append(rgb)

            # 记录样例
            meta_logs["batches"].append({
                "start_index": i,
                "count": len(batch_paths),
                "intrinsics_used_sample": intr_used[0].tolist(),
                "source_extrinsics": "external_json_cfw" if self.external["enabled"] else "vggt_pred",
            })

        # 汇总导出（仅 PLY）
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

        # —— 仅计算 BA（不写 COLMAP 文件）——
        if self.use_ba:
            if not _HAS_PYCOLMAP or not _HAS_TRACKER or images_full is None:
                print("[Warn] 缺少 pycolmap 或 tracker 依赖，BA 跳过（只导出 PLY）。")
            else:
                extr_all = np.concatenate(extr_list, axis=0)      # [N,4,4]
                intr_all = np.concatenate(intr_list, axis=0)      # [N,3,3] at self.resolution
                conf_all = np.concatenate(depth_conf_list, axis=0)  # [N,H,W,1]
                pts3d_all = np.concatenate(pts3d_list, axis=0)    # [N,H,W,3]
                rgb_all = np.concatenate(rgb_list, axis=0)        # [N,H,W,3]

                # 把内参从 resolution 缩放到 img_load_resolution
                scale = float(self.img_load_resolution) / float(self.resolution)
                intr_scaled = intr_all.copy()
                intr_scaled[:, :2, :] *= scale

                with torch.no_grad():
                    if self.device == "cuda":
                        with torch.cuda.amp.autocast(dtype=self.dtype):
                            pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb_ba = predict_tracks(
                                images_full,
                                conf=conf_all,
                                points_3d=pts3d_all,
                                masks=None,
                                max_query_pts=self.max_query_pts,
                                query_frame_num=self.query_frame_num,
                                keypoint_extractor="aliked+sp",
                                fine_tracking=self.fine_tracking,
                            )
                    else:
                        pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb_ba = predict_tracks(
                            images_full,
                            conf=conf_all,
                            points_3d=pts3d_all,
                            masks=None,
                            max_query_pts=self.max_query_pts,
                            query_frame_num=self.query_frame_num,
                            keypoint_extractor="aliked+sp",
                            fine_tracking=self.fine_tracking,
                        )

                track_mask = pred_vis_scores > self.vis_thresh

                reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                    points_3d_ba,
                    extr_all,
                    intr_scaled,
                    pred_tracks,
                    np.array(images_full.shape[-2:]),
                    masks=track_mask,
                    max_reproj_error=8.0,
                    shared_camera=self.shared_camera,
                    camera_type=self.camera_type,
                    points_rgb=points_rgb_ba,
                )

                if reconstruction is None:
                    print("[Warn] BA 构建失败，已跳过（仅保留 PLY）。")
                else:
                    # 在内存中执行 BA 优化，不写文件
                    ba_options = pycolmap.BundleAdjustmentOptions()
                    pycolmap.bundle_adjustment(reconstruction, ba_options)
                    # 可选：按需在内存中做重命名/缩放，依你需要
                    reconstruction = self._rename_colmap_and_rescale(
                        reconstruction,
                        base_image_names,
                        original_coords.cpu().numpy(),
                        img_size=self.img_load_resolution,
                        shift_point2d=True,
                        shared_camera=self.shared_camera,
                    )
                    print("[Info] BA 优化完成（未导出 COLMAP 文件，仅输出 PLY）。")

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

    # 仅输出 PLY、关闭 TXT、开启 BA；其余参数可在此手动调整
    reconstructor = VGGTReconstructor(
        images_dir=IMAGES_DIR,
        json_path=JSON_DEFAULT,
        batch_size=4,
        max_points=500000,
        resolution=518,
        conf_thresh=2.5,
        img_limit=None,
        out_dir=None,                 # None => 默认同级 vggt_output（不可写自动回退）
        write_txt=False,              # 明确关闭 TXT
        txt_filename="points_xyzrgb.txt",
        camera=cam,
        prefer_external_poses=False,   # 优先用 JSON 外参（T_cw）
        use_known_intrinsics=False,    # 使用你的内参

        # ===== BA 相关开关 =====
        use_ba=True,                  # 开启 BA（仅计算，不导出 COLMAP）
        shared_camera=True,
        camera_type="SIMPLE_PINHOLE",
        vis_thresh=0.2,
        query_frame_num=8,
        max_query_pts=4096,
        fine_tracking=True,
        img_load_resolution=1024,
    )

    with torch.no_grad():
        reconstructor.run()
