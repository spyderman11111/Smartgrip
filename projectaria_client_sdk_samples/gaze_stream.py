#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaze_stream.py

Real-time streaming from Aria over USB + online eye-gaze inference
+ approximate gaze projection onto RGB image.

功能：
- 使用 Aria Client SDK 通过 USB/WiFi 进行 live streaming；
- 复用 projectaria_client_sdk_samples 中的 AriaVisualizer 显示：
  - Front RGB 图像 (camera-rgb)
  - EyeTrack 眼动相机图像 (camera-et-*)
- 对每一帧 EyeTrack 图像调用 EyeGazeInference 模型，实时打印 yaw / pitch；
- 同时把当前 yaw/pitch 投影到 RGB 图像上，在 RGB 子图上画出一个绿色注视点；
- 近似使用 factory_calibration.json 中的 camera-rgb 内参实现 2D 投影，
  便于在线看效果。

注意：
- 投影是“近似版”：假设 CPF 与 RGB 相机光心重合，只用内参做投影；
- 若需要与 MPS 完全一致的“personalized gaze”精度，需要进一步接入
  DeviceCalibration + CPF -> RGB 的完整外参链条。
"""

import argparse
import os
import sys
import json
import math
from dataclasses import dataclass

import numpy as np
import aria.sdk as aria

from common import update_iptables
from visualizer import AriaVisualizer, AriaVisualizerStreamingClientObserver

# ===================== 可根据需要修改的默认参数 =====================

# projectaria_eyetracking 工程所在的父目录（里面应有 projectaria_eyetracking/ 包）
DEFAULT_ET_ROOT_PARENT = "/home/sz/Smartgrip"

# 眼动模型 checkpoint 和 config 路径
DEFAULT_MODEL_CKPT = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
)
DEFAULT_MODEL_CFG = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
)

# 默认推理设备
DEFAULT_DEVICE = "cuda:0"

# 默认 streaming profile（保持和原 device_stream.py 一致）
DEFAULT_PROFILE_NAME = "profile18"

# 默认 factory / streaming 标定文件路径
DEFAULT_FACTORY_CALIB_JSON = (
    "/home/sz/Smartgrip/projectaria_client_sdk_samples/factory_calibration.json"
)
DEFAULT_STREAMING_CALIB_JSON = (
    "/home/sz/Smartgrip/projectaria_client_sdk_samples/streaming_calibration.json"
)


# ===================== 简单相机内参结构体 =====================

@dataclass
class RgbIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None


def load_rgb_intrinsics_from_calibration(
    calib_path: str,
    camera_label: str = "camera-rgb",
) -> RgbIntrinsics:
    """
    从 JSON 标定文件中读取 camera-rgb 的内参 (近似假设: Params 前四项为 fx, fy, cx, cy)。

    calib_path: factory_calibration.json 或 streaming_calibration.json 路径。
    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration JSON not found: {calib_path}")

    with open(calib_path, "r") as f:
        calib = json.load(f)

    cameras = calib.get("CameraCalibrations", [])
    if not cameras:
        raise RuntimeError("No 'CameraCalibrations' entry in calibration JSON.")

    for cam in cameras:
        if cam.get("Label", "") != camera_label:
            continue

        proj = cam.get("Projection", {})
        params = proj.get("Params", [])
        if len(params) < 4:
            raise RuntimeError(
                f"Camera '{camera_label}' has fewer than 4 projection params."
            )

        fx, fy, cx, cy = params[:4]

        # 有的 JSON 里可能还有图像宽高信息，这里尝试读取
        width = cam.get("ImageWidth", None)
        height = cam.get("ImageHeight", None)

        print(
            f"[Calib] Loaded intrinsics for '{camera_label}' from {calib_path}:\n"
            f"        fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, "
            f"width={width}, height={height}"
        )
        return RgbIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    raise RuntimeError(f"Camera label '{camera_label}' not found in calibration JSON.")


# ===================== Gaze observer =====================

class GazeStreamingClientObserver(AriaVisualizerStreamingClientObserver):
    """
    在原有 AriaVisualizerStreamingClientObserver 基础上，增加：
    - 对 EyeTrack 图像进行眼动推理；
    - 实时在终端打印 yaw/pitch；
    - 更新 EyeTrack 子图标题显示当前 yaw/pitch；
    - 使用 camera-rgb 内参将 yaw/pitch 近似投影到 RGB 图像上，并画一个绿色圆点。
    """

    def __init__(
        self,
        visualizer: AriaVisualizer,
        eye_gaze_model,
        rgb_intrinsics: RgbIntrinsics | None = None,
        gaze_depth_m: float = 1.0,
        draw_radius_px: int = 12,
    ):
        super().__init__(visualizer)
        self.eye_gaze_model = eye_gaze_model
        self.last_eye_gaze = None  # 保存最近一次眼动结果
        self.rgb_intrinsics = rgb_intrinsics
        self.gaze_depth_m = gaze_depth_m
        self.draw_radius_px = draw_radius_px

    # ---------- 辅助函数：从 yaw/pitch → 像素坐标 ----------

    def _gaze_to_rgb_pixel(
        self,
        yaw: float,
        pitch: float,
        img_w: int,
        img_h: int,
    ) -> tuple[int | None, int | None]:
        """
        使用 Project Aria yaw/pitch 的约定：
        - 坐标系：X 左, Y 上, Z 前
        - yaw = atan(x/z), pitch = atan(y/z)

        因此可以直接反推：
            x/z = tan(yaw)
            y/z = tan(pitch)

        再用 pinhole + 内参投影到像素坐标。
        """
        if self.rgb_intrinsics is None:
            return None, None

        fx, fy, cx, cy = (
            self.rgb_intrinsics.fx,
            self.rgb_intrinsics.fy,
            self.rgb_intrinsics.cx,
            self.rgb_intrinsics.cy,
        )

        # x_over_z, y_over_z 即方向比值，和深度无关
        x_over_z = math.tan(yaw)
        y_over_z = math.tan(pitch)

        # 像素坐标（以像素中心为参照）
        u = fx * x_over_z + cx
        v = fy * y_over_z + cy

        u_int = int(round(u))
        v_int = int(round(v))

        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            return u_int, v_int
        else:
            return None, None

    def _draw_circle(self, image: np.ndarray, center_xy, radius: int = 8) -> np.ndarray:
        """
        在 image 上画一个简单的实心圆，使用绿色 (0,255,0)。
        若 image 是灰度，自动扩展到三通道。
        返回修改后的图像（可能是新数组）。
        """
        cx, cy = center_xy
        h, w = image.shape[:2]

        # 保证三通道
        if image.ndim == 2:
            img_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            img_rgb = np.repeat(image, 3, axis=-1)
        else:
            img_rgb = image

        y_min = max(0, cy - radius)
        y_max = min(h - 1, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(w - 1, cx + radius)

        rr = radius * radius
        for y in range(y_min, y_max + 1):
            dy = y - cy
            for x in range(x_min, x_max + 1):
                dx = x - cx
                if dx * dx + dy * dy <= rr:
                    img_rgb[y, x, 0] = 0   # R
                    img_rgb[y, x, 1] = 255 # G
                    img_rgb[y, x, 2] = 0   # B

        return img_rgb

    # ---------- 回调：每帧图像 ----------

    def on_image_received(self, image, record) -> None:
        """
        注意：
        - 使用 raw_image 做眼动推理和绘制；
        - super().on_image_received(...) 负责旋转/显示。
        """
        raw_image = image.copy()
        cam_id = record.camera_id

        # ========== 1) EyeTrack 图像：做眼动推理 ==========
        if (
            self.eye_gaze_model is not None
            and cam_id == aria.CameraId.EyeTrack
        ):
            import torch

            img_tensor = torch.tensor(
                raw_image, device=self.eye_gaze_model.device
            )

            with torch.no_grad():
                preds, lower, upper = self.eye_gaze_model.predict(img_tensor)

            # preds: [1, 2] -> yaw, pitch
            yaw = float(preds[0, 0].detach().cpu().numpy())
            pitch = float(preds[0, 1].detach().cpu().numpy())

            # 不确定区间
            yaw_low, pitch_low = [
                float(x) for x in lower[0].detach().cpu().numpy()
            ]
            yaw_high, pitch_high = [
                float(x) for x in upper[0].detach().cpu().numpy()
            ]

            self.last_eye_gaze = dict(
                timestamp_ns=record.capture_timestamp_ns,
                yaw=yaw,
                pitch=pitch,
                yaw_low=yaw_low,
                pitch_low=pitch_low,
                yaw_high=yaw_high,
                pitch_high=pitch_high,
            )

            # 在终端打印实时眼动信息
            print(
                f"[EyeGaze] t_ns={record.capture_timestamp_ns} "
                f"yaw={yaw:.3f} rad, pitch={pitch:.3f} rad "
                f"(low=({yaw_low:.3f},{pitch_low:.3f}), "
                f"high=({yaw_high:.3f},{pitch_high:.3f}))"
            )

            # 更新 EyeTrack 子图标题
            try:
                et_axes = self.visualizer.plots[0, 3]
                et_axes.set_title(
                    f"Eye Track\n"
                    f"yaw={yaw:.2f} rad, pitch={pitch:.2f} rad"
                )
            except Exception:
                pass

        # ========== 2) RGB 图像：使用最近一次 gaze 做投影并画点 ==========
        if (
            self.last_eye_gaze is not None
            and self.rgb_intrinsics is not None
            and cam_id == aria.CameraId.Rgb
        ):
            h, w = raw_image.shape[:2]
            yaw = self.last_eye_gaze["yaw"]
            pitch = self.last_eye_gaze["pitch"]

            u, v = self._gaze_to_rgb_pixel(yaw, pitch, img_w=w, img_h=h)
            if u is not None and v is not None:
                raw_image = self._draw_circle(
                    raw_image, center_xy=(u, v), radius=self.draw_radius_px
                )

        # 最后交给父类做旋转 + 可视化
        super().on_image_received(raw_image, record)


# ===================== Argument parsing =====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aria live gaze streaming with online eye-gaze inference "
                    "and approximate RGB gaze projection."
    )
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        default="usb",
        choices=["usb", "wifi"],
        help="Streaming interface, default: usb",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream (Linux only).",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default=DEFAULT_PROFILE_NAME,
        help="Streaming profile name, default: profile18.",
    )
    parser.add_argument(
        "--device-ip",
        dest="device_ip",
        type=str,
        help="Aria device IP address when using WiFi.",
    )

    # 眼动推理相关参数
    parser.add_argument(
        "--et-root-parent",
        type=str,
        default=DEFAULT_ET_ROOT_PARENT,
        help=(
            "Parent directory that contains 'projectaria_eyetracking' package. "
            "Default: /home/sz/Smartgrip"
        ),
    )
    parser.add_argument(
        "--model-checkpoint-path",
        type=str,
        default=DEFAULT_MODEL_CKPT,
        help="Path to eye gaze model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default=DEFAULT_MODEL_CFG,
        help="Path to eye gaze model config (.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device for eye-gaze inference, e.g. 'cpu' or 'cuda:0'.",
    )

    # 标定文件路径
    parser.add_argument(
        "--factory-calib-json",
        type=str,
        default=DEFAULT_FACTORY_CALIB_JSON,
        help="Path to factory_calibration.json for camera-rgb intrinsics.",
    )
    parser.add_argument(
        "--streaming-calib-json",
        type=str,
        default=DEFAULT_STREAMING_CALIB_JSON,
        help="(Optional) Path to streaming_calibration.json (currently unused).",
    )

    return parser.parse_args()


# ===================== Main =====================

def main():
    args = parse_args()

    # 根据需要更新 iptables
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # 设置 SDK log level
    aria.set_log_level(aria.Level.Info)

    # ---------- 导入 EyeGazeInference ----------
    if args.et_root_parent and os.path.isdir(args.et_root_parent):
        if args.et_root_parent not in sys.path:
            sys.path.insert(0, args.et_root_parent)

    try:
        from projectaria_eyetracking.projectaria_eyetracking.inference import (
            infer as eye_infer,
        )
    except ImportError as e:
        print(
            "ERROR: 无法导入 'projectaria_eyetracking.inference.infer'。\n"
            "请确认：\n"
            f"  1) '--et-root-parent' 设置正确，当前为: {args.et_root_parent}\n"
            "  2) 该目录下存在 'projectaria_eyetracking' 包\n"
            "  3) 该包内有 'inference/infer.py'\n"
        )
        print("详细 ImportError:", e)
        sys.exit(1)

    eye_model = eye_infer.EyeGazeInference(
        args.model_checkpoint_path,
        args.model_config_path,
        device=args.device,
    )

    # ---------- 读取 camera-rgb 内参（来自 factory_calibration.json） ----------
    rgb_intrinsics = None
    try:
        rgb_intrinsics = load_rgb_intrinsics_from_calibration(
            args.factory_calib_json,
            camera_label="camera-rgb",
        )
    except Exception as e:
        print(
            f"[WARN] Failed to load RGB intrinsics from {args.factory_calib_json}: {e}\n"
            "       将禁用 gaze → RGB 投影，仅显示 yaw/pitch 数值。"
        )

    # ---------- 建立 DeviceClient 连接 ----------
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip and args.streaming_interface == "wifi":
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    device = device_client.connect()
    print("Connected to device:", device.info.serial)

    # ---------- 配置 Streaming ----------
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    else:
        streaming_config.streaming_interface = aria.StreamingInterface.Wifi

    # 使用 ephemeral 证书
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # ---------- 启动 Streaming ----------
    streaming_manager.start_streaming()
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    # ---------- 创建可视化 + 观察者 ----------
    aria_visualizer = AriaVisualizer()
    gaze_observer = GazeStreamingClientObserver(
        aria_visualizer,
        eye_gaze_model=eye_model,
        rgb_intrinsics=rgb_intrinsics,
        gaze_depth_m=1.0,
        draw_radius_px=12,
    )
    streaming_client.set_streaming_client_observer(gaze_observer)
    streaming_client.subscribe()

    # ---------- 进入 UI 事件循环 ----------
    try:
        aria_visualizer.render_loop()
    finally:
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
