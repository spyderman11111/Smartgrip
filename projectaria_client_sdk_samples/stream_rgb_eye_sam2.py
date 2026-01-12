#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aria 实时 RGB + EyeTrack + 眼动推理 + 投影到 RGB + SAM2 单帧点提示分割（离线本地权重版）

功能：
- USB / WiFi 启动 Aria streaming
- 实时订阅 RGB + EyeTrack
- EyeTrack 上跑 EyeGazeInference
- 利用 VRS 标定将 gaze 投影到 RGB 图像上，在 RGB 窗口画点
- 终端持续打印 gaze 在“旋转后 RGB 显示坐标系”下的像素坐标
- 按 s：
    用 gaze 点作为 point prompt 触发 SAM2 分割
    并同时保存三类输出到脚本同目录：
      1) 原始旋转后 RGB（无任何覆盖）
      2) 二值 mask
      3) 白底原色抠图
- SAM2 结果在单独窗口显示白底原色抠图（不覆盖 RGB 窗口）
- 按 q / ESC 优雅退出，Ctrl+C 快速退出

注意：
- SAM2 使用本地 config + ckpt，不依赖 huggingface_hub
- 标定来自 --calib_vrs（同一副眼镜、同一 profile 更稳）
- 若遇到 RGB 偏色，可尝试：
    --aria-rgb-format rgb
    --aria-rgb-format bgr
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import aria.sdk as aria
import cv2
import numpy as np
import torch

from common import update_iptables

from projectaria_tools.core import data_provider
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.stream_id import StreamId

try:
    from projectaria_eyetracking.inference import infer
except ImportError:
    from projectaria_eyetracking.inference import infer


# ----------------------------
# 默认 EyeGazeInference 权重
# ----------------------------
DEFAULT_MODEL_WEIGHTS = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
)
DEFAULT_MODEL_CONFIG = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
)

# ----------------------------
# 默认 SAM2 本地路径（按你的机器先写死）
# 后续换电脑只改启动参数即可
# ----------------------------
DEFAULT_SAM2_ROOT = "/home/sz/Smartgrip/Grounded-SAM-2"
DEFAULT_SAM2_CKPT = "/home/sz/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
# 这里默认用 Hydra 期望的“相对配置名”
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # -------- Aria streaming ----------
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        choices=["usb", "wifi"],
        help="Streaming 接口：usb 或 wifi",
    )
    parser.add_argument(
        "--device-ip",
        type=str,
        help="当 interface=wifi 时，眼镜的 IP 地址，例如 192.168.0.101",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        help="Streaming profile 名称（默认 profile18）",
    )
    parser.add_argument(
        "--calib_vrs",
        type=str,
        required=True,
        help="用于读取标定的 VRS 文件路径（同一副眼镜、同一 profile 更稳）",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="在 Linux 上自动更新 iptables，以允许接收 UDP 数据",
    )

    # -------- EyeGazeInference ----------
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS,
        help="EyeGazeInference 模型权重路径",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=DEFAULT_MODEL_CONFIG,
        help="EyeGazeInference 模型配置路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备：cuda 或 cpu",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=1.0,
        help="假定的注视深度（米），投影到 RGB 时使用",
    )

    # -------- Aria RGB format ----------
    parser.add_argument(
        "--aria-rgb-format",
        type=str,
        default="auto",
        choices=["auto", "rgb", "bgr", "gray"],
        help="Aria RGB 原始帧的通道解释方式。若偏色可手动指定。",
    )

    # -------- SAM2 offline/local ----------
    parser.add_argument(
        "--sam2-root",
        type=str,
        default=DEFAULT_SAM2_ROOT,
        help="Grounded-SAM-2 根目录，用于加入 PYTHONPATH",
    )
    parser.add_argument(
        "--sam2-ckpt",
        type=str,
        default=DEFAULT_SAM2_CKPT,
        help="本地 SAM2 checkpoint 路径（.pt）",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default=DEFAULT_SAM2_CONFIG,
        help="本地 SAM2 config。推荐使用 configs/... 形式（Hydra config 名）",
    )
    parser.add_argument(
        "--sam2-mask-threshold",
        type=float,
        default=0.3,
        help="SAM2 mask threshold",
    )
    parser.add_argument(
        "--sam2-max-hole-area",
        type=float,
        default=100.0,
        help="SAM2 max hole area",
    )
    parser.add_argument(
        "--sam2-max-sprinkle-area",
        type=float,
        default=50.0,
        help="SAM2 max sprinkle area",
    )

    # 这里按你的需求：默认开启 multimask
    parser.add_argument(
        "--sam2-multimask",
        default=True,
        action="store_true",
        help="SAM2 multimask_output=True（默认开启）",
    )

    return parser.parse_args()


def import_sam2_local(sam2_root: str):
    """
    通过 sam2_root 加载本地 sam2 代码，避免依赖在线 from_pretrained。
    """
    sam2_root = os.path.abspath(sam2_root) if sam2_root else ""
    if sam2_root and sam2_root not in sys.path:
        sys.path.append(sam2_root)

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor
    except Exception as e:
        raise ImportError(
            "无法导入本地 SAM2。请确认 Grounded-SAM-2 路径正确，并包含 sam2 包。\n"
            f"sam2_root={sam2_root}\n"
            f"Original error: {e}"
        )


def resolve_sam2_config_and_name(
    sam2_root: str,
    sam2_config_arg: str,
) -> Tuple[str, str]:
    """
    目标：
    - 返回 (config_name_for_hydra, config_path_for_fs_check)

    说明：
    - build_sam2 内部使用 Hydra compose(config_name=...)
      因此更稳定的做法是传“相对配置名”
      例如：configs/sam2.1/sam2.1_hiera_l.yaml
    - 但我们仍需要一个真正存在的文件路径用于检查与报错。

    兼容：
    - 你传相对名：configs/...
    - 你传绝对路径：/home/.../sam2/configs/...
      我们会尽量自动还原为 configs/... 作为 hydra name
    """
    sam2_root = os.path.abspath(sam2_root) if sam2_root else ""
    arg = sam2_config_arg.strip()

    # 1) 先确定文件系统路径（用于 exists 检查）
    candidate_paths = []

    if os.path.isabs(arg):
        candidate_paths.append(arg)
    else:
        # 常见两种：相对 Grounded-SAM-2 根
        if sam2_root:
            candidate_paths.append(os.path.join(sam2_root, arg))
            # 以及相对 sam2 包根
            candidate_paths.append(os.path.join(sam2_root, "sam2", arg))
        # 兜底：当前工作目录
        candidate_paths.append(os.path.abspath(arg))

    config_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            config_path = p
            break

    # 2) 决定给 Hydra 的 config_name
    config_name = arg

    if os.path.isabs(arg):
        # 若是绝对路径，尽量转为 configs/... 形式
        norm = arg.replace("\\", "/")
        idx = norm.find("/sam2/")
        if idx >= 0:
            config_name = norm[idx + len("/sam2/") :]
        else:
            if sam2_root:
                sam2_pkg_root = os.path.join(sam2_root, "sam2")
                try:
                    rel = os.path.relpath(arg, sam2_pkg_root)
                    if not rel.startswith(".."):
                        config_name = rel.replace("\\", "/")
                except Exception:
                    pass

    # 3) 如果仍未找到文件，给出更有用的报错
    if config_path is None:
        msg = (
            "SAM2 config 不存在或无法定位。\n"
            f"你传入的 --sam2-config: {sam2_config_arg}\n"
        )
        if sam2_root:
            msg += (
                "我尝试过以下路径：\n"
                + "\n".join(f"  - {p}" for p in candidate_paths[:5])
                + "\n"
                f"请确认该文件存在于：\n"
                f"  - {sam2_root}/sam2/configs/...\n"
            )
        raise FileNotFoundError(msg)

    return config_name, config_path


def build_sam2_image_predictor_local(
    build_sam2_fn,
    SAM2ImagePredictor_cls,
    config_name_for_hydra: str,
    ckpt_path: str,
    device: str,
    mask_threshold: float,
    max_hole_area: float,
    max_sprinkle_area: float,
):
    """
    关键点：
    - 这里的 config_name_for_hydra 必须是 Hydra 可解析的“配置名”
      推荐形如：configs/sam2.1/sam2.1_hiera_l.yaml
    """
    try:
        sam_model = build_sam2_fn(
            config_file=config_name_for_hydra,
            ckpt_path=ckpt_path,
            device=device,
        )
    except TypeError:
        sam_model = build_sam2_fn(config_name_for_hydra, ckpt_path, device)

    predictor = SAM2ImagePredictor_cls(
        sam_model,
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
    )
    return predictor


def load_calibration_from_vrs(vrs_path: str):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    return device_calibration, rgb_camera_calibration, rgb_stream_label


# ----------------------------
# Aria RGB 颜色处理（保持你的参考版逻辑）
# ----------------------------
def _interpret_rotated_color_for_display(rotated: np.ndarray, fmt: str) -> np.ndarray:
    """
    输入 rotated 为旋转后的图像（未做通道解释）。
    输出 OpenCV imshow 需要的 BGR。
    """
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    if fmt == "bgr":
        return rotated.copy()

    # fmt == "rgb" 或 auto: 认为 rotated 是 RGB
    return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)


def _interpret_rotated_color_for_sam(rotated: np.ndarray, fmt: str) -> np.ndarray:
    """
    输入 rotated 为旋转后的图像（未做通道解释）。
    输出给 SAM2 的 RGB。
    """
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

    if fmt == "bgr":
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    # fmt == "rgb" 或 auto
    return rotated.copy()


def preprocess_image_for_display(image: np.ndarray, aria_rgb_format: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    显示用：旋转到正常朝向，并转为 BGR。
    返回：
    - vis_img_bgr: 用于 imshow
    - raw: 原始未旋转图像
    """
    raw = image
    rotated = np.rot90(raw, -1)

    fmt = aria_rgb_format
    if fmt == "auto":
        fmt = "rgb"

    img_bgr = _interpret_rotated_color_for_display(rotated, fmt)
    return img_bgr, raw


def get_rotated_rgb_for_sam(rgb_raw: np.ndarray, aria_rgb_format: str) -> np.ndarray:
    """
    给 SAM2 用的图：
    - 先旋转到与 RGB 窗口一致的朝向
    - 输出 HWC, RGB
    """
    rotated = np.rot90(rgb_raw, -1)

    fmt = aria_rgb_format
    if fmt == "auto":
        fmt = "rgb"

    img_rgb = _interpret_rotated_color_for_sam(rotated, fmt)
    return img_rgb


def rotate_point_cw90(x: float, y: float, h_raw: int, w_raw: int) -> Tuple[float, float]:
    """
    将原始图像中的点 (x, y) 顺时针旋转 90 度后对应到旋转图坐标。
    原图尺寸: (H, W)
    旋转后尺寸: (W, H)
    """
    x_rot = h_raw - 1 - y
    y_rot = x
    return x_rot, y_rot


def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    return xi, yi


def make_cutout_whitebg_rgb(
    rgb_rot_rgb: np.ndarray,
    mask_bool: np.ndarray,
) -> np.ndarray:
    """
    白底原色抠图：
    - 背景为白
    - mask 区域保留原始 RGB 颜色
    """
    h, w = rgb_rot_rgb.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[mask_bool] = rgb_rot_rgb[mask_bool]
    return out


def save_three_outputs(
    rgb_rot_rgb: np.ndarray,
    mask_bool: np.ndarray,
    prefix: str = "gaze_sam2",
):
    """
    同时保存：
    1) 原始旋转后 RGB（无覆盖）
    2) 二值 mask
    3) 白底原色抠图
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 1) rotated RGB
    rgb_path = os.path.join(here, f"{prefix}_{ts}_rgb_rot.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_rot_rgb, cv2.COLOR_RGB2BGR))

    # 2) binary mask
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_path = os.path.join(here, f"{prefix}_{ts}_mask_bin.png")
    cv2.imwrite(mask_path, mask_uint8)

    # 3) white-bg cutout
    cutout_rgb = make_cutout_whitebg_rgb(rgb_rot_rgb, mask_bool)
    cutout_path = os.path.join(here, f"{prefix}_{ts}_cutout_whitebg.png")
    cv2.imwrite(cutout_path, cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR))

    print(f"[Saved] rgb_rot -> {rgb_path}")
    print(f"[Saved] mask_bin -> {mask_path}")
    print(f"[Saved] cutout_whitebg -> {cutout_path}")

    return rgb_path, mask_path, cutout_path, cutout_rgb


class EyeRgbStreamingClientObserver:
    """
    - EyeTrack 帧：做眼动推理，更新 yaw/pitch
    - RGB 帧：根据当前 yaw/pitch + 标定，计算 gaze 像素坐标
    """
    def __init__(
        self,
        inference_model: infer.EyeGazeInference,
        device_calibration,
        rgb_camera_calibration,
        rgb_stream_label: str,
        depth_m: float,
        torch_device: str,
        aria_rgb_format: str,
    ) -> None:
        self.images = {}
        self.inference_model = inference_model
        self.device_calibration = device_calibration
        self.rgb_camera_calibration = rgb_camera_calibration
        self.rgb_stream_label = rgb_stream_label
        self.depth_m = depth_m
        self.torch_device = torch_device
        self.aria_rgb_format = aria_rgb_format

        self.latest_gaze = {
            "yaw": None,
            "pitch": None,
            "timestamp_ns": None,
            "pixel_raw": None,
            "pixel_rot": None,
        }

        self.latest_rgb_raw = None
        self.latest_rgb_rot = None
        self.latest_rgb_rot_shape = None  # (H, W) of rotated RGB

        self._last_print_t = 0.0
        self._print_interval = 0.05  # seconds

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord) -> None:
        camera_id = record.camera_id
        self.images[camera_id] = image

        if camera_id == aria.CameraId.EyeTrack:
            self._handle_eye_frame(image, record)
        elif camera_id == aria.CameraId.Rgb:
            self._handle_rgb_frame(image, record)

    def _handle_eye_frame(self, image: np.ndarray, record: ImageDataRecord) -> None:
        img_tensor = torch.tensor(image, device=self.torch_device)

        with torch.no_grad():
            preds, _, _ = self.inference_model.predict(img_tensor)

        preds = preds.detach().cpu().numpy()
        yaw = float(preds[0][0])
        pitch = float(preds[0][1])

        self.latest_gaze["yaw"] = yaw
        self.latest_gaze["pitch"] = pitch
        self.latest_gaze["timestamp_ns"] = int(record.capture_timestamp_ns)

    def _handle_rgb_frame(self, image: np.ndarray, record: ImageDataRecord) -> None:
        self.latest_rgb_raw = image
        rgb_rot = get_rotated_rgb_for_sam(image, self.aria_rgb_format)
        self.latest_rgb_rot = rgb_rot
        self.latest_rgb_rot_shape = rgb_rot.shape[:2]

        yaw = self.latest_gaze["yaw"]
        pitch = self.latest_gaze["pitch"]
        if yaw is None or pitch is None:
            return

        eye_gaze = EyeGaze
        eye_gaze.yaw = yaw
        eye_gaze.pitch = pitch
        eye_gaze.depth = self.depth_m

        gaze_projection = get_gaze_vector_reprojection(
            eye_gaze,
            self.rgb_stream_label,
            self.device_calibration,
            self.rgb_camera_calibration,
            self.depth_m,
        )

        if gaze_projection is None:
            self.latest_gaze["pixel_raw"] = None
            self.latest_gaze["pixel_rot"] = None
            return

        x_raw, y_raw = float(gaze_projection[0]), float(gaze_projection[1])
        h_raw, w_raw = image.shape[:2]
        x_rot, y_rot = rotate_point_cw90(x_raw, y_raw, h_raw, w_raw)

        self.latest_gaze["pixel_raw"] = (x_raw, y_raw)
        self.latest_gaze["pixel_rot"] = (x_rot, y_rot)

        now = time.time()
        if now - self._last_print_t >= self._print_interval and self.latest_rgb_rot_shape:
            self._last_print_t = now
            h_rot, w_rot = self.latest_rgb_rot_shape
            gx, gy = clamp_point(x_rot, y_rot, w_rot, h_rot)
            print(f"[GazePixel] ({gx}, {gy})  size=({w_rot}x{h_rot})")


def start_device_streaming(
    streaming_interface: str,
    profile_name: str,
    device_ip: Optional[str],
):
    """
    启动 Aria streaming，返回:
    - device_client
    - device
    - streaming_manager
    - streaming_client
    """
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if device_ip:
        client_config.ip_v4_address = device_ip
    device_client.set_client_config(client_config)

    print(f"[DeviceClient] Connecting over {streaming_interface}...")
    device = device_client.connect()
    print("[DeviceClient] Connected.")

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = profile_name

    if streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb

    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()
    print(
        f"[StreamingManager] Started with profile={profile_name}, interface={streaming_interface}"
    )
    print(f"[StreamingManager] State: {streaming_manager.streaming_state}")

    return device_client, device, streaming_manager, streaming_client


def configure_streaming_client_rgb_eye(streaming_client: aria.StreamingClient) -> None:
    """
    配置 StreamingClient：只订阅 RGB + EyeTrack
    """
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    )
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options

    streaming_client.subscription_config = config


def run_sam2_with_gaze_point(
    sam2_predictor,
    rgb_rot_rgb: np.ndarray,
    gaze_xy: Tuple[int, int],
    multimask_output: bool = True,
):
    """
    用当前 gaze 点做单次 SAM2 point-prompt 分割

    改进点：
    - multimask 默认开启
    - 多 mask 时优先只在“包含 gaze 点”的候选里选 iou 最高的
    - 若无候选包含 gaze 点，退化为全局 iou 最佳
    """
    sam2_predictor.set_image(rgb_rot_rgb)

    gx, gy = gaze_xy

    point_coords = np.array([[gx, gy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    masks, ious, _ = sam2_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
        return_logits=False,
        normalize_coords=False,
    )

    # ious 可能为 None
    if ious is None:
        ious = np.zeros((masks.shape[0],), dtype=np.float32) if masks.ndim == 3 else np.array([0.0], dtype=np.float32)

    # 兼容多 mask 输出
    if masks.ndim == 3:
        # 过滤：只保留包含 gaze 点的 mask
        contain_idxs = []
        for i in range(masks.shape[0]):
            m = masks[i]
            # 防御性检查
            if 0 <= gy < m.shape[0] and 0 <= gx < m.shape[1]:
                if bool(m[gy, gx]):
                    contain_idxs.append(i)

        if len(contain_idxs) > 0:
            # 在包含 gaze 点的候选里选 iou 最大
            best_idx = contain_idxs[int(np.argmax(ious[contain_idxs]))]
        else:
            # 退化：全局 iou 最佳
            best_idx = int(np.argmax(ious)) if len(ious) else 0

        mask = masks[best_idx].astype(bool)
        score = float(ious[best_idx]) if len(ious) else 0.0
    else:
        mask = masks.astype(bool)
        score = float(ious[0]) if len(ious) else 0.0

    return mask, score


def main() -> None:
    args = parse_args()

    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    # 1) 标定
    print(f"[Calib] Loading from VRS: {args.calib_vrs}")
    device_calibration, rgb_camera_calibration, rgb_stream_label = load_calibration_from_vrs(
        args.calib_vrs
    )
    print(f"[Calib] RGB stream label: {rgb_stream_label}")

    # 2) Eye 模型
    print(
        f"[Model] Loading EyeGazeInference on device={args.device}\n"
        f"        weights={args.model_checkpoint_path}\n"
        f"        config={args.model_config_path}"
    )
    inference_model = infer.EyeGazeInference(
        args.model_checkpoint_path,
        args.model_config_path,
        args.device,
    )

    # 3) SAM2 本地构建
    print(f"[SAM2] sam2_root  : {args.sam2_root}")
    print(f"[SAM2] sam2_ckpt  : {args.sam2_ckpt}")
    print(f"[SAM2] sam2_config(arg): {args.sam2_config}")

    build_sam2_fn, SAM2ImagePredictor_cls = import_sam2_local(args.sam2_root)

    config_name_for_hydra, config_path_for_check = resolve_sam2_config_and_name(
        args.sam2_root, args.sam2_config
    )
    if not os.path.isfile(args.sam2_ckpt):
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {args.sam2_ckpt}")

    print(f"[SAM2] sam2_config(hydra): {config_name_for_hydra}")
    print(f"[SAM2] sam2_config(file ): {config_path_for_check}")

    print(f"[SAM2] Building local SAM2 image predictor on device={args.device}")
    sam2_predictor = build_sam2_image_predictor_local(
        build_sam2_fn=build_sam2_fn,
        SAM2ImagePredictor_cls=SAM2ImagePredictor_cls,
        config_name_for_hydra=config_name_for_hydra,
        ckpt_path=args.sam2_ckpt,
        device=args.device,
        mask_threshold=args.sam2_mask_threshold,
        max_hole_area=args.sam2_max_hole_area,
        max_sprinkle_area=args.sam2_max_sprinkle_area,
    )

    # 4) 启动 streaming
    device_client, device, streaming_manager, streaming_client = start_device_streaming(
        streaming_interface=args.streaming_interface,
        profile_name=args.profile_name,
        device_ip=args.device_ip,
    )

    # 5) 配置订阅
    configure_streaming_client_rgb_eye(streaming_client)

    # 6) observer
    observer = EyeRgbStreamingClientObserver(
        inference_model=inference_model,
        device_calibration=device_calibration,
        rgb_camera_calibration=rgb_camera_calibration,
        rgb_stream_label=rgb_stream_label,
        depth_m=args.depth,
        torch_device=args.device,
        aria_rgb_format=args.aria_rgb_format,
    )
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 7) 窗口
    rgb_window = "Aria RGB (Gaze)"
    eye_window = "Aria EyeTrack"
    sam_window = "SAM2 Cutout Preview"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 960, 960)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 640, 640)
    cv2.setWindowProperty(eye_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window, 1050, 50)

    cv2.namedWindow(sam_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(sam_window, 960, 960)
    cv2.setWindowProperty(sam_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(sam_window, 50, 1050)

    print(
        "[Main] Running.\n"
        "       - Press 's' to run SAM2 with current gaze point and save three outputs.\n"
        "       - Press 'q' or ESC to exit."
    )

    graceful_exit = True
    latest_cutout_bgr = None

    try:
        while True:
            # RGB 显示
            if aria.CameraId.Rgb in observer.images:
                rgb_raw = observer.images.pop(aria.CameraId.Rgb)
                rgb_vis_bgr, _ = preprocess_image_for_display(rgb_raw, args.aria_rgb_format)

                pixel_rot = observer.latest_gaze.get("pixel_rot")
                if pixel_rot is not None and observer.latest_rgb_rot_shape is not None:
                    h_rot, w_rot = observer.latest_rgb_rot_shape
                    gx, gy = clamp_point(pixel_rot[0], pixel_rot[1], w_rot, h_rot)

                    cv2.circle(
                        rgb_vis_bgr,
                        (gx, gy),
                        12,
                        (0, 255, 0),
                        thickness=-1,
                    )

                cv2.imshow(rgb_window, rgb_vis_bgr)

            # EyeTrack 显示
            if aria.CameraId.EyeTrack in observer.images:
                eye_raw = observer.images.pop(aria.CameraId.EyeTrack)
                eye_vis_bgr, _ = preprocess_image_for_display(eye_raw, "gray")
                cv2.imshow(eye_window, eye_vis_bgr)

            # SAM2 白底抠图预览
            if latest_cutout_bgr is not None:
                cv2.imshow(sam_window, latest_cutout_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if (
                    observer.latest_rgb_rot is None
                    or observer.latest_rgb_rot_shape is None
                    or observer.latest_gaze.get("pixel_rot") is None
                ):
                    print("[SAM2] No valid RGB/gaze yet.")
                    continue

                rgb_rot_rgb = observer.latest_rgb_rot
                h_rot, w_rot = observer.latest_rgb_rot_shape
                x_rot, y_rot = observer.latest_gaze["pixel_rot"]
                gx, gy = clamp_point(x_rot, y_rot, w_rot, h_rot)

                print(f"[SAM2] Running with gaze point ({gx}, {gy}) ...")

                # 这里显式使用“多 mask + 含 gaze 点筛选”
                mask_bool, score = run_sam2_with_gaze_point(
                    sam2_predictor=sam2_predictor,
                    rgb_rot_rgb=rgb_rot_rgb,
                    gaze_xy=(gx, gy),
                    multimask_output=True,
                )

                print(f"[SAM2] Done. score={score:.4f}")

                # 保存三类输出，并拿白底抠图用于窗口预览
                _, _, _, cutout_rgb = save_three_outputs(
                    rgb_rot_rgb=rgb_rot_rgb,
                    mask_bool=mask_bool,
                    prefix="gaze_sam2",
                )
                latest_cutout_bgr = cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR)

            if key == ord("q") or key == 27:
                print("[Main] 'q' / ESC pressed, graceful exit.")
                break

    except KeyboardInterrupt:
        graceful_exit = False
        print("[Main] KeyboardInterrupt, fast exit without full SDK shutdown.")

    finally:
        print("[Main] Cleaning up...")
        try:
            streaming_client.unsubscribe()
        except Exception as e:
            print(f"[Cleanup] unsubscribe error: {e}")

        if graceful_exit:
            try:
                streaming_manager.stop_streaming()
            except Exception as e:
                print(f"[Cleanup] stop_streaming error: {e}")
            try:
                device_client.disconnect(device)
            except Exception as e:
                print(f"[Cleanup] disconnect error: {e}")

        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass

        print("[Main] Done.")


if __name__ == "__main__":
    main()


"""
用法示例：

1) WiFi
python stream_rgb_eye_sam2.py \
    --interface wifi \
    --device-ip 192.168.0.102 \
    --profile profile18 \
    --calib_vrs /home/sz/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
    --update_iptables \
    --device cuda \
    --sam2-root /home/sz/Smartgrip/Grounded-SAM-2 \
    --sam2-ckpt /home/sz/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml

2) USB
python stream_rgb_eye_sam2.py \
    --interface usb \
    --profile profile18 \
    --calib_vrs /home/sz/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
    --device cuda \
    --sam2-root /home/sz/Smartgrip/Grounded-SAM-2 \
    --sam2-ckpt /home/sz/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml

3) 若 RGB 偏色，尝试显式指定格式
python stream_rgb_eye_sam2.py \
    --interface wifi \
    --device-ip 192.168.0.103 \
    --profile profile18 \
    --calib_vrs /home/sz/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
    --update_iptables \
    --device cuda \
    --sam2-root /home/sz/Smartgrip/Grounded-SAM-2 \
    --sam2-ckpt /home/sz/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --aria-rgb-format rgb

或
    --aria-rgb-format bgr
"""
