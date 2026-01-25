#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aria live RGB + EyeTrack -> gaze projection on RGB -> SAM2 single-frame ROI box segmentation (local weights).

Keys:
- Press 's' to run SAM2 on a gaze-centered ROI box and save:
  1) rotated RGB (no overlay)
  2) binary mask (full image, ROI outside forced 0)
  3) white-background cutout (full image)
  4) debug RGB with gaze dot + ROI box
- Press 'q' or ESC to exit.

Mask semantics:
- By default, we INVERT the SAM2 mask inside ROI (because your current output tends to be background).
- ROI outside is always forced to 0.
- Use --no-invert-roi-mask to disable inversion.
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


DEFAULT_MODEL_WEIGHTS = (
    "/home/MA_SmartGrip/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
)
DEFAULT_MODEL_CONFIG = (
    "/home/MA_SmartGrip/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/"
    "inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
)

DEFAULT_SAM2_ROOT = "/home/MA_SmartGrip/Smartgrip/Grounded-SAM-2"
DEFAULT_SAM2_CKPT = "/home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--interface", dest="streaming_interface", required=True, choices=["usb", "wifi"])
    p.add_argument("--device-ip", type=str, default=None)
    p.add_argument("--profile", dest="profile_name", type=str, default="profile18")
    p.add_argument("--calib_vrs", type=str, required=True)
    p.add_argument("--update_iptables", action="store_true")

    p.add_argument("--model_checkpoint_path", type=str, default=DEFAULT_MODEL_WEIGHTS)
    p.add_argument("--model_config_path", type=str, default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--depth", type=float, default=1.0)

    p.add_argument("--aria-rgb-format", type=str, default="auto", choices=["auto", "rgb", "bgr", "gray"])
    p.add_argument("--gaze-box-size", type=int, default=320)

    p.add_argument("--sam2-root", type=str, default=DEFAULT_SAM2_ROOT)
    p.add_argument("--sam2-ckpt", type=str, default=DEFAULT_SAM2_CKPT)
    p.add_argument("--sam2-config", type=str, default=DEFAULT_SAM2_CONFIG)
    p.add_argument("--sam2-mask-threshold", type=float, default=0.3)
    p.add_argument("--sam2-max-hole-area", type=float, default=100.0)
    p.add_argument("--sam2-max-sprinkle-area", type=float, default=50.0)

    # default: ON
    p.add_argument(
        "--no-invert-roi-mask",
        dest="invert_roi_mask",
        action="store_false",
        help="Disable ROI mask inversion (default is inverted).",
    )
    p.set_defaults(invert_roi_mask=True)

    return p.parse_args()


def import_sam2_local(sam2_root: str):
    sam2_root = os.path.abspath(sam2_root)
    if sam2_root not in sys.path:
        sys.path.append(sam2_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return build_sam2, SAM2ImagePredictor


def resolve_sam2_config(sam2_root: str, cfg_arg: str) -> Tuple[str, str]:
    sam2_root = os.path.abspath(sam2_root)
    arg = cfg_arg.strip()
    if not arg:
        raise ValueError("Empty --sam2-config")

    if os.path.isabs(arg):
        if not os.path.isfile(arg):
            raise FileNotFoundError(f"SAM2 config not found: {arg}")
        # best-effort hydra name
        sam2_pkg_root = os.path.join(sam2_root, "sam2")
        try:
            rel = os.path.relpath(arg, sam2_pkg_root)
            cfg_name = arg if rel.startswith("..") else rel.replace("\\", "/")
        except Exception:
            cfg_name = arg
        return cfg_name, arg

    cfg_path = os.path.join(sam2_root, arg)
    if not os.path.isfile(cfg_path):
        alt = os.path.join(sam2_root, "sam2", arg)
        if os.path.isfile(alt):
            cfg_path = alt
        else:
            raise FileNotFoundError(f"SAM2 config not found: {cfg_path} (or {alt})")
    return arg.replace("\\", "/"), cfg_path


def build_sam2_predictor(
    build_sam2_fn,
    predictor_cls,
    config_name_for_hydra: str,
    ckpt_path: str,
    device: str,
    mask_threshold: float,
    max_hole_area: float,
    max_sprinkle_area: float,
):
    try:
        model = build_sam2_fn(config_file=config_name_for_hydra, ckpt_path=ckpt_path, device=device)
    except TypeError:
        model = build_sam2_fn(config_name_for_hydra, ckpt_path, device)

    return predictor_cls(
        model,
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
    )


def load_calibration_from_vrs(vrs_path: str):
    provider = data_provider.create_vrs_data_provider(vrs_path)
    rgb_stream_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_stream_id)
    dev_calib = provider.get_device_calibration()
    rgb_calib = dev_calib.get_camera_calib(rgb_label)
    return dev_calib, rgb_calib, rgb_label


def rot_cw90_xy(x: float, y: float, h_raw: int) -> Tuple[float, float]:
    # rotate CW: (x, y) -> (h-1-y, x)
    return (h_raw - 1 - y), x


def clamp_xy(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xi = max(0, min(w - 1, int(round(x))))
    yi = max(0, min(h - 1, int(round(y))))
    return xi, yi


def make_roi_box(cx: int, cy: int, box_size: int, w: int, h: int) -> Tuple[int, int, int, int]:
    half = max(1, box_size // 2)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def to_display_bgr(rotated: np.ndarray, fmt: str) -> np.ndarray:
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    if fmt == "bgr":
        return rotated.copy()
    return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)


def to_sam_rgb(rotated: np.ndarray, fmt: str) -> np.ndarray:
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
    if fmt == "bgr":
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    return rotated.copy()


def save_outputs(rgb_rot_rgb: np.ndarray, mask: np.ndarray, dbg_bgr: np.ndarray, prefix: str = "gaze_sam2"):
    here = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d_%H%M%S")

    rgb_path = os.path.join(here, f"{prefix}_{ts}_rgb_rot.png")
    mask_path = os.path.join(here, f"{prefix}_{ts}_mask_bin.png")
    cut_path = os.path.join(here, f"{prefix}_{ts}_cutout_whitebg.png")
    dbg_path = os.path.join(here, f"{prefix}_{ts}_rgb_gaze_box.png")

    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_rot_rgb, cv2.COLOR_RGB2BGR))

    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(mask_path, mask_u8)

    cutout = np.full_like(rgb_rot_rgb, 255)
    cutout[mask] = rgb_rot_rgb[mask]
    cv2.imwrite(cut_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))

    cv2.imwrite(dbg_path, dbg_bgr)

    print(f"[Saved] {rgb_path}")
    print(f"[Saved] {mask_path}")
    print(f"[Saved] {cut_path}")
    print(f"[Saved] {dbg_path}")

    return cutout


class Observer:
    def __init__(
        self,
        inference_model: infer.EyeGazeInference,
        dev_calib,
        rgb_calib,
        rgb_label: str,
        depth_m: float,
        torch_device: str,
        aria_rgb_format: str,
    ):
        self.model = inference_model
        self.dev_calib = dev_calib
        self.rgb_calib = rgb_calib
        self.rgb_label = rgb_label
        self.depth_m = depth_m
        self.torch_device = torch_device
        self.fmt = "rgb" if aria_rgb_format == "auto" else aria_rgb_format

        self.last_rgb_raw = None
        self.last_eye_raw = None

        self.yaw = None
        self.pitch = None

        self.rgb_rot_rgb = None
        self.rgb_rot_shape = None
        self.gaze_rot_xy = None

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord) -> None:
        if record.camera_id == aria.CameraId.EyeTrack:
            self._on_eye(image, record)
        elif record.camera_id == aria.CameraId.Rgb:
            self._on_rgb(image, record)

    def _on_eye(self, image: np.ndarray, record: ImageDataRecord) -> None:
        self.last_eye_raw = image
        img_t = torch.tensor(image, device=self.torch_device)
        with torch.no_grad():
            preds, _, _ = self.model.predict(img_t)
        preds = preds.detach().cpu().numpy()
        self.yaw = float(preds[0][0])
        self.pitch = float(preds[0][1])

    def _on_rgb(self, image: np.ndarray, record: ImageDataRecord) -> None:
        self.last_rgb_raw = image

        rotated = np.rot90(image, -1)
        rgb_rot_rgb = to_sam_rgb(rotated, self.fmt)
        self.rgb_rot_rgb = rgb_rot_rgb
        self.rgb_rot_shape = rgb_rot_rgb.shape[:2]

        if self.yaw is None or self.pitch is None:
            self.gaze_rot_xy = None
            return

        try:
            eg = EyeGaze()
        except TypeError:
            eg = EyeGaze
        eg.yaw = self.yaw
        eg.pitch = self.pitch
        eg.depth = self.depth_m

        proj = get_gaze_vector_reprojection(
            eg, self.rgb_label, self.dev_calib, self.rgb_calib, self.depth_m
        )
        if proj is None:
            self.gaze_rot_xy = None
            return

        x_raw, y_raw = float(proj[0]), float(proj[1])
        h_raw, _ = image.shape[:2]
        x_rot, y_rot = rot_cw90_xy(x_raw, y_raw, h_raw)
        self.gaze_rot_xy = (x_rot, y_rot)


def start_streaming(streaming_interface: str, profile_name: str, device_ip: Optional[str]):
    dev_client = aria.DeviceClient()
    cfg = aria.DeviceClientConfig()
    if device_ip:
        cfg.ip_v4_address = device_ip
    dev_client.set_client_config(cfg)

    print(f"[DeviceClient] Connecting over {streaming_interface}...")
    device = dev_client.connect()
    print("[DeviceClient] Connected.")

    mgr = device.streaming_manager
    client = mgr.streaming_client

    scfg = aria.StreamingConfig()
    scfg.profile_name = profile_name
    if streaming_interface == "usb":
        scfg.streaming_interface = aria.StreamingInterface.Usb
    scfg.security_options.use_ephemeral_certs = True
    mgr.streaming_config = scfg

    mgr.start_streaming()
    print(f"[Streaming] Started profile={profile_name}, interface={streaming_interface}")
    return dev_client, device, mgr, client


def configure_subscription(client: aria.StreamingClient):
    cfg = client.subscription_config
    cfg.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    cfg.message_queue_size[aria.StreamingDataType.Rgb] = 1
    cfg.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    sec = aria.StreamingSecurityOptions()
    sec.use_ephemeral_certs = True
    cfg.security_options = sec

    client.subscription_config = cfg


def sam2_segment_roi_box(
    predictor,
    rgb_rot_rgb: np.ndarray,
    gaze_xy: Tuple[int, int],
    box_size: int,
):
    h, w = rgb_rot_rgb.shape[:2]
    gx, gy = gaze_xy
    x1, y1, x2, y2 = make_roi_box(gx, gy, box_size, w, h)

    roi = rgb_rot_rgb[y1:y2, x1:x2]
    predictor.set_image(roi)

    roi_h, roi_w = roi.shape[:2]
    box = np.array([0, 0, roi_w, roi_h], dtype=np.float32)

    masks, ious, _ = predictor.predict(box=box, multimask_output=False, return_logits=False)
    if masks is None or len(masks) == 0:
        return None, 0.0, (x1, y1, x2, y2)

    full = np.zeros((h, w), dtype=bool)
    full[y1:y2, x1:x2] = masks[0].astype(bool)
    score = float(ious[0]) if ious is not None and len(ious) else 0.0
    return full, score, (x1, y1, x2, y2)


def main():
    args = parse_args()

    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    dev_calib, rgb_calib, rgb_label = load_calibration_from_vrs(args.calib_vrs)
    print(f"[Calib] RGB label: {rgb_label}")

    model = infer.EyeGazeInference(args.model_checkpoint_path, args.model_config_path, args.device)

    build_sam2_fn, predictor_cls = import_sam2_local(args.sam2_root)
    cfg_name, cfg_path = resolve_sam2_config(args.sam2_root, args.sam2_config)
    if not os.path.isfile(args.sam2_ckpt):
        raise FileNotFoundError(f"SAM2 ckpt not found: {args.sam2_ckpt}")
    print(f"[SAM2] cfg={cfg_name} ({cfg_path})")
    print(f"[SAM2] ckpt={args.sam2_ckpt}")

    predictor = build_sam2_predictor(
        build_sam2_fn,
        predictor_cls,
        cfg_name,
        args.sam2_ckpt,
        args.device,
        args.sam2_mask_threshold,
        args.sam2_max_hole_area,
        args.sam2_max_sprinkle_area,
    )

    dev_client, device, mgr, client = start_streaming(args.streaming_interface, args.profile_name, args.device_ip)
    configure_subscription(client)

    obs = Observer(
        inference_model=model,
        dev_calib=dev_calib,
        rgb_calib=rgb_calib,
        rgb_label=rgb_label,
        depth_m=args.depth,
        torch_device=args.device,
        aria_rgb_format=args.aria_rgb_format,
    )
    client.set_streaming_client_observer(obs)
    client.subscribe()

    rgb_win = "Aria RGB (Gaze)"
    eye_win = "Aria EyeTrack"
    sam_win = "SAM2 Cutout"

    cv2.namedWindow(rgb_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_win, 960, 960)
    cv2.setWindowProperty(rgb_win, cv2.WND_PROP_TOPMOST, 1)

    cv2.namedWindow(eye_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_win, 640, 640)
    cv2.setWindowProperty(eye_win, cv2.WND_PROP_TOPMOST, 1)

    cv2.namedWindow(sam_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(sam_win, 960, 960)
    cv2.setWindowProperty(sam_win, cv2.WND_PROP_TOPMOST, 1)

    print("[Main] Press 's' to segment & save. Press 'q' or ESC to exit.")
    print(f"[Main] invert_roi_mask = {args.invert_roi_mask} (disable with --no-invert-roi-mask)")

    latest_cutout_bgr = None
    graceful_exit = True

    try:
        while True:
            if obs.last_rgb_raw is not None:
                rotated = np.rot90(obs.last_rgb_raw, -1)
                fmt = "rgb" if args.aria_rgb_format == "auto" else args.aria_rgb_format
                vis = to_display_bgr(rotated, fmt)

                if obs.gaze_rot_xy is not None and obs.rgb_rot_shape is not None:
                    h, w = obs.rgb_rot_shape
                    gx, gy = clamp_xy(obs.gaze_rot_xy[0], obs.gaze_rot_xy[1], w, h)
                    cv2.circle(vis, (gx, gy), 12, (0, 255, 0), thickness=-1)
                    x1, y1, x2, y2 = make_roi_box(gx, gy, args.gaze_box_size, w, h)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                cv2.imshow(rgb_win, vis)

            if obs.last_eye_raw is not None:
                rotated_eye = np.rot90(obs.last_eye_raw, -1)
                eye_bgr = to_display_bgr(rotated_eye, "gray")
                cv2.imshow(eye_win, eye_bgr)

            if latest_cutout_bgr is not None:
                cv2.imshow(sam_win, latest_cutout_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if obs.rgb_rot_rgb is None or obs.rgb_rot_shape is None or obs.gaze_rot_xy is None:
                    print("[SAM2] Missing RGB/gaze.")
                    continue

                h, w = obs.rgb_rot_shape
                gx, gy = clamp_xy(obs.gaze_rot_xy[0], obs.gaze_rot_xy[1], w, h)

                full_mask, score, (x1, y1, x2, y2) = sam2_segment_roi_box(
                    predictor, obs.rgb_rot_rgb, (gx, gy), args.gaze_box_size
                )
                if full_mask is None:
                    print("[SAM2] No mask.")
                    continue

                roi_gate = np.zeros_like(full_mask, dtype=bool)
                roi_gate[y1:y2, x1:x2] = True

                # ROI-only mask, optionally inverted (default: inverted)
                if args.invert_roi_mask:
                    mask = roi_gate & (~full_mask)
                else:
                    mask = roi_gate & full_mask

                print(f"[SAM2] score={score:.4f}  ROI=({x1},{y1})-({x2},{y2})")

                dbg = cv2.cvtColor(obs.rgb_rot_rgb, cv2.COLOR_RGB2BGR)
                cv2.circle(dbg, (gx, gy), 12, (0, 255, 0), thickness=-1)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                cutout_rgb = save_outputs(obs.rgb_rot_rgb, mask, dbg, prefix="gaze_sam2")
                latest_cutout_bgr = cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR)

            if key == ord("q") or key == 27:
                print("[Main] Exit.")
                break

    except KeyboardInterrupt:
        graceful_exit = False
        print("[Main] KeyboardInterrupt.")

    finally:
        print("[Main] Cleanup...")
        try:
            client.unsubscribe()
        except Exception as e:
            print(f"[Cleanup] unsubscribe: {e}")

        if graceful_exit:
            try:
                mgr.stop_streaming()
            except Exception as e:
                print(f"[Cleanup] stop_streaming: {e}")
            try:
                dev_client.disconnect(device)
            except Exception as e:
                print(f"[Cleanup] disconnect: {e}")

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
python stream_rgb_eye_sam2.py \
  --interface usb \
  --profile profile18 \
  --calib_vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --device cuda \
  --gaze-box-size 150 \
  --sam2-root /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2 \
  --sam2-ckpt /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
  
python stream_rgb_eye_sam2.py \
  --interface wifi \
  --device-ip 192.168.0.102 \
  --profile profile18 \
  --calib_vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --update_iptables \
  --device cuda \
  --gaze-box-size 320 \
  --sam2-root /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2 \
  --sam2-ckpt /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
  
  """