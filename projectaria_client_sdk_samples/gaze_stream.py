#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaze_stream.py

Real-time streaming from Aria over USB + online eye-gaze inference.

功能：
- 使用 Aria Client SDK 通过 USB 进行 live streaming；
- 复用 projectaria_client_sdk_samples 中的 AriaVisualizer 显示：
  - Front RGB 图像
  - EyeTrack 眼动相机图像
- 对每一帧 EyeTrack 图像调用 EyeGazeInference 模型，实时打印 yaw / pitch；
- 同时把当前 yaw/pitch 写到 EyeTrack 子图的标题中，方便观察。

运行方式（示例）：
  python gaze_stream.py --interface usb --update_iptables

如需修改模型路径、projectaria_eyetracking 路径或设备（CPU/GPU），
可以：
  1) 直接改下面 DEFAULT_... 常量，或者
  2) 通过命令行参数覆盖。
"""

import argparse
import os
import sys

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

# 默认推理设备：建议先用 CPU，确认没问题后再改成 "cuda:0"
DEFAULT_DEVICE = "cuda:0"

# 默认 streaming profile（保持和原 device_stream.py 一致）
DEFAULT_PROFILE_NAME = "profile18"


# ===================== Gaze observer =====================

class GazeStreamingClientObserver(AriaVisualizerStreamingClientObserver):
    """
    在原有 AriaVisualizerStreamingClientObserver 基础上，增加：
    - 对 EyeTrack 图像进行眼动推理；
    - 实时在终端打印 yaw/pitch；
    - 更新 EyeTrack 子图标题显示当前 yaw/pitch。
    """

    def __init__(self, visualizer: AriaVisualizer, eye_gaze_model):
        super().__init__(visualizer)
        self.eye_gaze_model = eye_gaze_model
        self.last_eye_gaze = None  # 保存最近一次眼动结果

    def on_image_received(self, image, record) -> None:
        # 保留一份原始图像给模型用（不做旋转）
        raw_image = image.copy()

        # 先调用父类实现完成图像旋转 + 可视化
        super().on_image_received(image, record)

        # 仅对 EyeTrack 图像做眼动推理
        if (
            self.eye_gaze_model is not None
            and record.camera_id == aria.CameraId.EyeTrack
        ):
            import torch

            # 离线 demo 中就是直接 torch.tensor(image)，这里保持一致
            img_tensor = torch.tensor(
                raw_image, device=self.eye_gaze_model.device
            )

            with torch.no_grad():
                preds, lower, upper = self.eye_gaze_model.predict(img_tensor)

            # preds: [1, 2] -> yaw, pitch
            yaw = float(preds[0, 0].detach().cpu().numpy())
            pitch = float(preds[0, 1].detach().cpu().numpy())

            # 不确定区间（目前只是保存，方便后面扩展）
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

            # 尝试把当前 yaw/pitch 写到 EyeTrack 子图标题中
            try:
                # AriaVisualizer 中第 0 行第 3 列是 Eye Track 视图
                et_axes = self.visualizer.plots[0, 3]
                et_axes.set_title(
                    f"Eye Track\n"
                    f"yaw={yaw:.2f} rad, pitch={pitch:.2f} rad"
                )
            except Exception:
                # 即使这里失败，也不影响主流程
                pass


# ===================== Argument parsing =====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aria live gaze streaming with online eye-gaze inference."
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
        help="Device for eye-gaze inference, e.g. 'cpu' or 'cuda:0'. Default: cpu",
    )

    return parser.parse_args()


# ===================== Main =====================

def main():
    args = parse_args()

    # 根据需要更新 iptables（和原 device_stream.py 一致）
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # 设置 SDK log level
    aria.set_log_level(aria.Level.Info)

    # ---------- 导入 EyeGazeInference ----------
    # 把 projectaria_eyetracking 的父目录加入 sys.path
    if args.et_root_parent and os.path.isdir(args.et_root_parent):
        if args.et_root_parent not in sys.path:
            sys.path.insert(0, args.et_root_parent)

    try:
        # 推荐方式：通过 projectaria_eyetracking 包导入
        from projectaria_eyetracking.projectaria_eyetracking.inference import infer as eye_infer
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

    # 初始化眼动推理模型
    eye_model = eye_infer.EyeGazeInference(
        args.model_checkpoint_path,
        args.model_config_path,
        device=args.device,
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
        # WiFi 模式（根据 SDK 版本可能叫 Wifi 或 WifiStation，视本地 API 而定）
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
    )
    streaming_client.set_streaming_client_observer(gaze_observer)
    streaming_client.subscribe()

    # ---------- 进入 UI 事件循环 ----------
    try:
        aria_visualizer.render_loop()
    finally:
        # ---------- 收尾 ----------
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
