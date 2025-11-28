#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aria 实时 RGB + EyeTrack + 眼动推理 + 投影到 RGB（改进退出逻辑）

功能：
- USB / WiFi 启动 Aria streaming
- 实时订阅 RGB + EyeTrack
- EyeTrack 上跑 EyeGazeInference
- 利用 VRS 标定将 gaze 投影到 RGB 图像上，在 RGB 上画点
- 终端持续打印 gaze 像素坐标（raw + rot），便于检查
- 按 q / ESC 优雅退出，Ctrl+C 快速退出
"""

import argparse
import sys
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


DEFAULT_MODEL_WEIGHTS = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
)
DEFAULT_MODEL_CONFIG = (
    "/home/sz/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        help="用于读取标定的 VRS 文件路径（同一副眼镜、同一 profile）",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="在 Linux 上自动更新 iptables，以允许接收 UDP 数据",
    )
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
    return parser.parse_args()


def load_calibration_from_vrs(
    vrs_path: str,
):
    """
    从 VRS 中读取 device_calibration 和 RGB 相机标定。
    假设：
    - EyeTrack: 211-1
    - RGB: 214-1
    """
    provider = data_provider.create_vrs_data_provider(vrs_path)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    return device_calibration, rgb_camera_calibration, rgb_stream_label


def preprocess_image_for_display(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    显示用：旋转到正常朝向，并转为 BGR。
    返回：
    - vis_img: 旋转 + BGR，用于 imshow
    - raw: 原始未旋转图像，用于坐标转换
    """
    raw = image
    img = np.rot90(raw, -1)

    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_bgr, raw


def rotate_point_cw90(x: float, y: float, h_raw: int, w_raw: int) -> Tuple[float, float]:
    """
    将原始图像中的点 (x, y) 旋转 90 度顺时针后，在旋转图像坐标系下的坐标。

    原图尺寸: (H, W)
    旋转后尺寸: (W, H)

    cw90 变换:
        x_rot = H - 1 - y
        y_rot = x
    """
    x_rot = h_raw - 1 - y
    y_rot = x
    return x_rot, y_rot


class EyeRgbStreamingClientObserver:
    """
    StreamingClient observer：
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
    ) -> None:
        self.images = {}  # camera_id -> np.ndarray
        self.inference_model = inference_model
        self.device_calibration = device_calibration
        self.rgb_camera_calibration = rgb_camera_calibration
        self.rgb_stream_label = rgb_stream_label
        self.depth_m = depth_m
        self.torch_device = torch_device

        self.latest_gaze = {
            "yaw": None,
            "pitch": None,
            "timestamp_ns": None,
            "pixel_raw": None,
            "pixel_rot": None,
        }

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
            preds, lower, upper = self.inference_model.predict(img_tensor)

        preds = preds.detach().cpu().numpy()
        yaw = float(preds[0][0])
        pitch = float(preds[0][1])

        self.latest_gaze["yaw"] = yaw
        self.latest_gaze["pitch"] = pitch
        self.latest_gaze["timestamp_ns"] = int(record.capture_timestamp_ns)

    def _handle_rgb_frame(self, image: np.ndarray, record: ImageDataRecord) -> None:
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

        # 只打印 gaze 点坐标（原始 + 旋转后），不再打印 yaw/pitch
        print(
            f"[Gaze] ts={record.capture_timestamp_ns} "
            f"raw=({x_raw:.1f}, {y_raw:.1f}), "
            f"rot=({x_rot:.1f}, {y_rot:.1f})"
        )


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
    # wifi 情况使用默认（WifiStation）

    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()
    print(f"[StreamingManager] Started with profile={profile_name}, interface={streaming_interface}")
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

    # 2) 模型
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

    # 3) 启动 streaming
    device_client, device, streaming_manager, streaming_client = start_device_streaming(
        streaming_interface=args.streaming_interface,
        profile_name=args.profile_name,
        device_ip=args.device_ip,
    )

    # 4) 配置订阅
    configure_streaming_client_rgb_eye(streaming_client)

    # 5) observer
    observer = EyeRgbStreamingClientObserver(
        inference_model=inference_model,
        device_calibration=device_calibration,
        rgb_camera_calibration=rgb_camera_calibration,
        rgb_stream_label=rgb_stream_label,
        depth_m=args.depth,
        torch_device=args.device,
    )
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 6) 窗口
    rgb_window = "Aria RGB with Gaze"
    eye_window = "Aria EyeTrack"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 960, 960)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 640, 640)
    cv2.setWindowProperty(eye_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window, 1050, 50)

    print("[Main] Running. Press 'q' or ESC in any OpenCV window to exit. Ctrl+C 也可以强制退出。")

    graceful_exit = True

    try:
        while True:
            # RGB
            if aria.CameraId.Rgb in observer.images:
                rgb_raw = observer.images.pop(aria.CameraId.Rgb)
                rgb_vis, _ = preprocess_image_for_display(rgb_raw)

                pixel_rot = observer.latest_gaze.get("pixel_rot")
                if pixel_rot is not None:
                    gx, gy = pixel_rot
                    cv2.circle(
                        rgb_vis,
                        (int(round(gx)), int(round(gy))),
                        15,
                        (0, 255, 0),
                        thickness=-1,
                    )

                cv2.imshow(rgb_window, rgb_vis)

            # EyeTrack
            if aria.CameraId.EyeTrack in observer.images:
                eye_raw = observer.images.pop(aria.CameraId.EyeTrack)
                eye_vis, _ = preprocess_image_for_display(eye_raw)
                cv2.imshow(eye_window, eye_vis)

            key = cv2.waitKey(1) & 0xFF
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

        cv2.destroyAllWindows()
        print("[Main] Done.")


if __name__ == "__main__":
    main()

"""
python stream_rgb_eye.py \
    --interface wifi \
    --device-ip 192.168.0.101 \
    --profile profile18 \
    --calib_vrs /home/sz/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
    --update_iptables \
    --device cuda

python stream_rgb_eye.py \
    --interface usb \
    --profile profile18 \
    --calib_vrs /home/sz/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
    --device cuda
"""