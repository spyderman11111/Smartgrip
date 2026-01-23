#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aria live RGB + EyeTrack -> gaze reprojection -> SAM2 point-prompt segmentation (local/offline weights)
...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

import aria.sdk as aria
from projectaria_tools.core import data_provider
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.stream_id import StreamId


# ============================================================
# USER CONFIG
# ============================================================

DEFAULT_CALIB_VRS = os.environ.get(
    "SMARTGRIP_ARIA_CALIB_VRS",
    "/home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs",
)

DEFAULT_PROFILE_NAME = os.environ.get("SMARTGRIP_ARIA_PROFILE", "profile18")
DEFAULT_DEVICE = os.environ.get("SMARTGRIP_DEVICE", "cuda")
DEFAULT_GAZE_DEPTH_M = float(os.environ.get("SMARTGRIP_GAZE_DEPTH_M", "1.0"))
DEFAULT_ARIA_RGB_FORMAT = os.environ.get("SMARTGRIP_ARIA_RGB_FORMAT", "auto")  # auto|rgb|bgr|gray

DEFAULT_SAM2_ROOT = os.environ.get("SMARTGRIP_SAM2_ROOT", "/home/MA_SmartGrip/Smartgrip/Grounded-SAM-2")
DEFAULT_SAM2_CKPT = os.environ.get(
    "SMARTGRIP_SAM2_CKPT",
    "/home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
)
DEFAULT_SAM2_CONFIG_FILE = os.environ.get(
    "SMARTGRIP_SAM2_CONFIG",
    "/home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
)

DEFAULT_SAM2_MASK_THRESHOLD = float(os.environ.get("SMARTGRIP_SAM2_MASK_THRESHOLD", "0.3"))
DEFAULT_SAM2_MAX_HOLE_AREA = float(os.environ.get("SMARTGRIP_SAM2_MAX_HOLE_AREA", "100.0"))
DEFAULT_SAM2_MAX_SPRINKLE_AREA = float(os.environ.get("SMARTGRIP_SAM2_MAX_SPRINKLE_AREA", "50.0"))

# ======= IMPORTANT CHANGE: default output dir fixed to your path =======
DEFAULT_OUTPUT_DIR = os.environ.get(
    "SMARTGRIP_OUTPUT_DIR",
    "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage",
)
# =====================================================================

DEFAULT_EYE_REPO_ROOT = Path(
    os.environ.get(
        "SMARTGRIP_EYE_REPO_ROOT",
        str(Path.home() / "Smartgrip" / "projectaria_eyetracking"),
    )
).expanduser()

DEFAULT_EYE_WEIGHTS = os.environ.get(
    "SMARTGRIP_EYE_WEIGHTS",
    str(
        DEFAULT_EYE_REPO_ROOT
        / "projectaria_eyetracking"
        / "inference"
        / "model"
        / "pretrained_weights"
        / "social_eyes_uncertainty_v1"
        / "weights.pth"
    ),
)

DEFAULT_EYE_CONFIG = os.environ.get(
    "SMARTGRIP_EYE_CONFIG",
    str(
        DEFAULT_EYE_REPO_ROOT
        / "projectaria_eyetracking"
        / "inference"
        / "model"
        / "pretrained_weights"
        / "social_eyes_uncertainty_v1"
        / "config.yaml"
    ),
)


try:
    from common import update_iptables  # type: ignore
except Exception:  # pragma: no cover
    def update_iptables():
        return


try:
    from projectaria_eyetracking.inference.infer import EyeGazeInference  # type: ignore
except Exception:  # pragma: no cover
    from projectaria_eyetracking.inference import infer  # type: ignore
    EyeGazeInference = infer.EyeGazeInference  # type: ignore


@dataclass
class AppConfig:
    streaming_interface: str = "usb"   # usb|wifi
    device_ip: Optional[str] = None
    profile_name: str = DEFAULT_PROFILE_NAME
    update_iptables: bool = False

    calib_vrs: str = DEFAULT_CALIB_VRS
    gaze_depth_m: float = DEFAULT_GAZE_DEPTH_M
    aria_rgb_format: str = DEFAULT_ARIA_RGB_FORMAT  # auto|rgb|bgr|gray

    device: str = DEFAULT_DEVICE  # cuda|cpu

    eye_weights: Optional[str] = DEFAULT_EYE_WEIGHTS
    eye_config: Optional[str] = DEFAULT_EYE_CONFIG

    sam2_root: str = DEFAULT_SAM2_ROOT
    sam2_ckpt: str = DEFAULT_SAM2_CKPT
    sam2_config_file: str = DEFAULT_SAM2_CONFIG_FILE
    sam2_mask_threshold: float = DEFAULT_SAM2_MASK_THRESHOLD
    sam2_max_hole_area: float = DEFAULT_SAM2_MAX_HOLE_AREA
    sam2_max_sprinkle_area: float = DEFAULT_SAM2_MAX_SPRINKLE_AREA

    # ======= IMPORTANT CHANGE: now default is your output dir, not script dir =======
    output_dir: str = DEFAULT_OUTPUT_DIR
    # ============================================================================


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_file(path: str, what: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def _ensure_dir(path: str, what: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def _candidate_pkg_roots_for(name: str) -> List[Path]:
    import importlib.util
    roots: List[Path] = []

    spec = importlib.util.find_spec(name)
    if spec and spec.submodule_search_locations:
        for p in spec.submodule_search_locations:
            try:
                roots.append(Path(p).resolve())
            except Exception:
                pass

    try:
        pkg = __import__(name, fromlist=["__path__", "__file__"])
        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path:
            for p in pkg_path:
                try:
                    roots.append(Path(p).resolve())
                except Exception:
                    pass
        pkg_file = getattr(pkg, "__file__", None)
        if pkg_file:
            try:
                roots.append(Path(pkg_file).resolve().parent)
            except Exception:
                pass
    except Exception:
        pass

    uniq: List[Path] = []
    seen = set()
    for r in roots:
        s = str(r)
        if s not in seen:
            uniq.append(r)
            seen.add(s)
    return uniq


def find_default_eyegaze_assets() -> Tuple[Path, Path]:
    roots = _candidate_pkg_roots_for("projectaria_eyetracking")

    def try_root(r: Path) -> Optional[Tuple[Path, Path]]:
        base = r / "inference" / "model" / "pretrained_weights" / "social_eyes_uncertainty_v1"
        w = base / "weights.pth"
        c = base / "config.yaml"
        if w.is_file() and c.is_file():
            return w, c

        base2 = r / "projectaria_eyetracking" / "inference" / "model" / "pretrained_weights" / "social_eyes_uncertainty_v1"
        w2 = base2 / "weights.pth"
        c2 = base2 / "config.yaml"
        if w2.is_file() and c2.is_file():
            return w2, c2
        return None

    for r in roots:
        hit = try_root(r)
        if hit:
            return hit

    raise FileNotFoundError(
        "Cannot auto-locate EyeGaze pretrained weights/config inside pip package.\n"
        "Tried package roots:\n  - " + "\n  - ".join(str(r) for r in roots) + "\n\n"
        "Fix options:\n"
        "1) Pass them explicitly:\n"
        "   --eye-weights /path/to/weights.pth --eye-config /path/to/config.yaml\n"
        "2) Or set env:\n"
        "   SMARTGRIP_EYE_WEIGHTS=/path/to/weights.pth\n"
        "   SMARTGRIP_EYE_CONFIG=/path/to/config.yaml\n"
    )


def import_sam2_local(sam2_root: str):
    sam2_root = str(Path(sam2_root).expanduser().resolve())
    if sam2_root not in sys.path:
        sys.path.append(sam2_root)

    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        return build_sam2, SAM2ImagePredictor
    except Exception as e:
        raise ImportError(
            "Failed to import local SAM2.\n"
            f"sam2_root={sam2_root}\n"
            "Expected: <sam2_root>/sam2/...\n"
            f"Original error: {e}"
        )


def sam2_hydra_config_name(sam2_root: str, sam2_config_file: str) -> Tuple[str, Path]:
    sam2_root_p = Path(sam2_root).expanduser().resolve()
    sam2_pkg_root = sam2_root_p / "sam2"

    cfg_path = Path(sam2_config_file).expanduser()
    if not cfg_path.is_absolute():
        candidates = [sam2_root_p / cfg_path, sam2_pkg_root / cfg_path, Path.cwd() / cfg_path]
        cfg_resolved = None
        for c in candidates:
            if c.is_file():
                cfg_resolved = c.resolve()
                break
        if cfg_resolved is None:
            raise FileNotFoundError(
                "SAM2 config file not found.\n"
                f"sam2_config_file={sam2_config_file}\n"
                "Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
            )
        cfg_path = cfg_resolved
    else:
        cfg_path = cfg_path.resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(f"SAM2 config file not found: {cfg_path}")

    try:
        rel = cfg_path.relative_to(sam2_pkg_root)
        config_name = rel.as_posix()
    except Exception:
        norm = cfg_path.as_posix()
        idx = norm.find("/sam2/")
        if idx >= 0:
            config_name = norm[idx + len("/sam2/"):]
        else:
            raise FileNotFoundError(
                "SAM2 config file is not under <sam2_root>/sam2, Hydra may fail.\n"
                f"sam2_root={sam2_root_p}\n"
                f"sam2_config_file={cfg_path}\n"
                "Please set sam2_config_file to a path under <sam2_root>/sam2/."
            )
    return config_name, cfg_path


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
    ckpt_p = _ensure_file(ckpt_path, "SAM2 checkpoint")
    try:
        sam_model = build_sam2_fn(config_file=config_name_for_hydra, ckpt_path=str(ckpt_p), device=device)
    except TypeError:
        sam_model = build_sam2_fn(config_name_for_hydra, str(ckpt_p), device)

    predictor = predictor_cls(
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


def _to_bgr_for_imshow(rotated: np.ndarray, fmt: str) -> np.ndarray:
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    if fmt == "bgr":
        return rotated.copy()
    return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)


def _to_rgb_for_sam(rotated: np.ndarray, fmt: str) -> np.ndarray:
    if rotated.ndim == 2 or fmt == "gray":
        return cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
    if fmt == "bgr":
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    return rotated.copy()


def preprocess_image_for_display(image_raw: np.ndarray, aria_rgb_format: str) -> Tuple[np.ndarray, np.ndarray]:
    rotated = np.rot90(image_raw, -1)
    fmt = aria_rgb_format if aria_rgb_format != "auto" else "rgb"
    vis_bgr = _to_bgr_for_imshow(rotated, fmt)
    return vis_bgr, image_raw


def get_rotated_rgb_for_sam(image_raw: np.ndarray, aria_rgb_format: str) -> np.ndarray:
    rotated = np.rot90(image_raw, -1)
    fmt = aria_rgb_format if aria_rgb_format != "auto" else "rgb"
    rgb = _to_rgb_for_sam(rotated, fmt)
    return rgb


def rotate_point_cw90(x: float, y: float, h_raw: int, w_raw: int) -> Tuple[float, float]:
    x_rot = h_raw - 1 - y
    y_rot = x
    return x_rot, y_rot


def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    return xi, yi


def make_cutout_whitebg_rgb(rgb_rot_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    h, w = rgb_rot_rgb.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[mask_bool] = rgb_rot_rgb[mask_bool]
    return out


def save_three_outputs(rgb_rot_rgb: np.ndarray, mask_bool: np.ndarray, out_dir: Path, prefix: str = "gaze_sam2"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    rgb_path = out_dir / f"{prefix}_{ts}_rgb_rot.png"
    mask_path = out_dir / f"{prefix}_{ts}_mask_bin.png"
    cutout_path = out_dir / f"{prefix}_{ts}_cutout_whitebg.png"

    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_rot_rgb, cv2.COLOR_RGB2BGR))
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_path), mask_uint8)

    cutout_rgb = make_cutout_whitebg_rgb(rgb_rot_rgb, mask_bool)
    cv2.imwrite(str(cutout_path), cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR))

    print(f"[Saved] rgb_rot        : {rgb_path}")
    print(f"[Saved] mask_bin       : {mask_path}")
    print(f"[Saved] cutout_whitebg : {cutout_path}")
    return rgb_path, mask_path, cutout_path, cutout_rgb


class EyeRgbStreamingClientObserver:
    def __init__(
        self,
        inference_model: EyeGazeInference,
        device_calibration,
        rgb_camera_calibration,
        rgb_stream_label: str,
        depth_m: float,
        torch_device: str,
        aria_rgb_format: str,
        print_interval_s: float = 0.05,
    ) -> None:
        self.images = {}
        self.inference_model = inference_model
        self.device_calibration = device_calibration
        self.rgb_camera_calibration = rgb_camera_calibration
        self.rgb_stream_label = rgb_stream_label
        self.depth_m = depth_m
        self.torch_device = torch_device
        self.aria_rgb_format = aria_rgb_format

        self.latest_gaze = {"yaw": None, "pitch": None, "timestamp_ns": None, "pixel_raw": None, "pixel_rot": None}
        self.latest_rgb_raw = None
        self.latest_rgb_rot = None
        self.latest_rgb_rot_shape = None

        self._last_print_t = 0.0
        self._print_interval = print_interval_s

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
        self.latest_gaze["yaw"] = float(preds[0][0])
        self.latest_gaze["pitch"] = float(preds[0][1])
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

        eg = EyeGaze()
        eg.yaw = yaw
        eg.pitch = pitch
        eg.depth = self.depth_m

        gaze_projection = get_gaze_vector_reprojection(
            eg,
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


def start_device_streaming(streaming_interface: str, profile_name: str, device_ip: Optional[str]):
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
    elif streaming_interface == "wifi":
        streaming_config.streaming_interface = aria.StreamingInterface.Wifi

    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    streaming_manager.start_streaming()

    print(f"[StreamingManager] Started: profile={profile_name}, interface={streaming_interface}")
    print(f"[StreamingManager] State: {streaming_manager.streaming_state}")
    return device_client, device, streaming_manager, streaming_client


def configure_streaming_client_rgb_eye(streaming_client: aria.StreamingClient) -> None:
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config


def run_sam2_with_gaze_point(sam2_predictor, rgb_rot_rgb: np.ndarray, gaze_xy: Tuple[int, int], multimask_output: bool = True):
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

    if ious is None:
        if masks.ndim == 3:
            ious = np.zeros((masks.shape[0],), dtype=np.float32)
        else:
            ious = np.array([0.0], dtype=np.float32)

    if masks.ndim == 3:
        contain = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if 0 <= gy < m.shape[0] and 0 <= gx < m.shape[1] and bool(m[gy, gx]):
                contain.append(i)
        if contain:
            best_idx = contain[int(np.argmax(ious[contain]))]
        else:
            best_idx = int(np.argmax(ious)) if len(ious) else 0
        mask = masks[best_idx].astype(bool)
        score = float(ious[best_idx]) if len(ious) else 0.0
    else:
        mask = masks.astype(bool)
        score = float(ious[0]) if len(ious) else 0.0

    return mask, score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aria live RGB+EyeTrack with gaze->SAM2 point prompt (offline/local).")
    p.add_argument("--interface", type=str, required=True, choices=["usb", "wifi"], help="Streaming interface.")
    p.add_argument("--device-ip", type=str, default=None, help="Aria IP when interface=wifi.")
    p.add_argument("--update_iptables", action="store_true", help="Update iptables to allow UDP (Linux).")
    p.add_argument("--calib-vrs", type=str, default="", help="Calibration VRS path.")
    p.add_argument("--profile", type=str, default="", help="Streaming profile name override.")
    p.add_argument("--device", type=str, default="", help="cuda or cpu override.")
    p.add_argument("--aria-rgb-format", type=str, default="", choices=["", "auto", "rgb", "bgr", "gray"], help="Override RGB format.")
    p.add_argument("--gaze-depth", type=float, default=-1.0, help="Override gaze depth (m).")

    p.add_argument("--eye-weights", type=str, default="", help="Override EyeGaze weights path.")
    p.add_argument("--eye-config", type=str, default="", help="Override EyeGaze config yaml path.")

    p.add_argument("--sam2-root", type=str, default="", help="Override SAM2 repo root.")
    p.add_argument("--sam2-ckpt", type=str, default="", help="Override SAM2 ckpt path.")
    p.add_argument("--sam2-config", type=str, default="", help="Override SAM2 config yaml path (absolute ok).")

    p.add_argument("--output-dir", type=str, default="", help="Override output directory.")
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    cfg.streaming_interface = args.interface
    cfg.device_ip = args.device_ip
    cfg.update_iptables = bool(args.update_iptables)

    if args.calib_vrs.strip():
        cfg.calib_vrs = args.calib_vrs.strip()
    if args.profile.strip():
        cfg.profile_name = args.profile.strip()
    if args.device.strip():
        cfg.device = args.device.strip()
    if args.aria_rgb_format.strip():
        cfg.aria_rgb_format = args.aria_rgb_format.strip()
    if args.gaze_depth > 0:
        cfg.gaze_depth_m = float(args.gaze_depth)

    if args.eye_weights.strip():
        cfg.eye_weights = args.eye_weights.strip()
    if args.eye_config.strip():
        cfg.eye_config = args.eye_config.strip()

    if args.sam2_root.strip():
        cfg.sam2_root = args.sam2_root.strip()
    if args.sam2_ckpt.strip():
        cfg.sam2_ckpt = args.sam2_ckpt.strip()
    if args.sam2_config.strip():
        cfg.sam2_config_file = args.sam2_config.strip()

    if args.output_dir.strip():
        cfg.output_dir = args.output_dir.strip()

    return cfg


def run(cfg: AppConfig) -> None:
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    if cfg.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    # ======= IMPORTANT CHANGE: output dir always resolves to your path (or CLI/env override) =======
    out_dir = Path(cfg.output_dir).expanduser().resolve() if cfg.output_dir else _script_dir()
    # ===========================================================================================

    calib_vrs_p = _ensure_file(cfg.calib_vrs, "Calibration VRS")
    print(f"[Calib] Loading from VRS: {calib_vrs_p}")
    device_calibration, rgb_camera_calibration, rgb_stream_label = load_calibration_from_vrs(str(calib_vrs_p))
    print(f"[Calib] RGB stream label: {rgb_stream_label}")

    eye_w_path = Path(cfg.eye_weights).expanduser() if cfg.eye_weights else None
    eye_y_path = Path(cfg.eye_config).expanduser() if cfg.eye_config else None
    if not (eye_w_path and eye_w_path.is_file() and eye_y_path and eye_y_path.is_file()):
        print("[Model] Eye assets not found at cfg paths, trying auto-discovery from pip package...")
        w, y = find_default_eyegaze_assets()
        cfg.eye_weights = str(w)
        cfg.eye_config = str(y)

    eye_w = _ensure_file(cfg.eye_weights, "EyeGaze weights")
    eye_y = _ensure_file(cfg.eye_config, "EyeGaze config")

    print("[Model] Loading EyeGazeInference")
    print(f"        device : {cfg.device}")
    print(f"        weights: {eye_w}")
    print(f"        config : {eye_y}")
    inference_model = EyeGazeInference(str(eye_w), str(eye_y), cfg.device)

    sam2_root_p = _ensure_dir(cfg.sam2_root, "SAM2 repo root")
    build_sam2_fn, predictor_cls = import_sam2_local(str(sam2_root_p))
    cfg_name, cfg_file_resolved = sam2_hydra_config_name(str(sam2_root_p), cfg.sam2_config_file)
    _ensure_file(cfg.sam2_ckpt, "SAM2 checkpoint")

    print("[SAM2] Building local SAM2 image predictor")
    print(f"       sam2_root      : {sam2_root_p}")
    print(f"       sam2_ckpt      : {Path(cfg.sam2_ckpt).expanduser().resolve()}")
    print(f"       sam2_configfile: {cfg_file_resolved}")
    print(f"       hydra_conf_name: {cfg_name}")

    sam2_predictor = build_sam2_predictor(
        build_sam2_fn=build_sam2_fn,
        predictor_cls=predictor_cls,
        config_name_for_hydra=cfg_name,
        ckpt_path=cfg.sam2_ckpt,
        device=cfg.device,
        mask_threshold=cfg.sam2_mask_threshold,
        max_hole_area=cfg.sam2_max_hole_area,
        max_sprinkle_area=cfg.sam2_max_sprinkle_area,
    )

    device_client, device, streaming_manager, streaming_client = start_device_streaming(
        streaming_interface=cfg.streaming_interface,
        profile_name=cfg.profile_name,
        device_ip=cfg.device_ip,
    )

    configure_streaming_client_rgb_eye(streaming_client)

    observer = EyeRgbStreamingClientObserver(
        inference_model=inference_model,
        device_calibration=device_calibration,
        rgb_camera_calibration=rgb_camera_calibration,
        rgb_stream_label=rgb_stream_label,
        depth_m=cfg.gaze_depth_m,
        torch_device=cfg.device,
        aria_rgb_format=cfg.aria_rgb_format,
    )
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

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
        "       - Press 's' to run SAM2 with current gaze point and save outputs.\n"
        "       - Press 'q' or ESC to exit."
    )

    graceful_exit = True
    latest_cutout_bgr = None

    try:
        while True:
            if aria.CameraId.Rgb in observer.images:
                rgb_raw = observer.images.pop(aria.CameraId.Rgb)
                rgb_vis_bgr, _ = preprocess_image_for_display(rgb_raw, cfg.aria_rgb_format)

                pixel_rot = observer.latest_gaze.get("pixel_rot")
                if pixel_rot is not None and observer.latest_rgb_rot_shape is not None:
                    h_rot, w_rot = observer.latest_rgb_rot_shape
                    gx, gy = clamp_point(pixel_rot[0], pixel_rot[1], w_rot, h_rot)
                    cv2.circle(rgb_vis_bgr, (gx, gy), 12, (0, 255, 0), thickness=-1)

                cv2.imshow(rgb_window, rgb_vis_bgr)

            if aria.CameraId.EyeTrack in observer.images:
                eye_raw = observer.images.pop(aria.CameraId.EyeTrack)
                eye_vis_bgr, _ = preprocess_image_for_display(eye_raw, "gray")
                cv2.imshow(eye_window, eye_vis_bgr)

            if latest_cutout_bgr is not None:
                cv2.imshow(sam_window, latest_cutout_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if observer.latest_rgb_rot is None or observer.latest_rgb_rot_shape is None or observer.latest_gaze.get("pixel_rot") is None:
                    print("[SAM2] No valid RGB/gaze yet.")
                    continue

                rgb_rot_rgb = observer.latest_rgb_rot
                h_rot, w_rot = observer.latest_rgb_rot_shape
                x_rot, y_rot = observer.latest_gaze["pixel_rot"]
                gx, gy = clamp_point(x_rot, y_rot, w_rot, h_rot)

                print(f"[SAM2] Running with gaze point ({gx}, {gy}) ...")

                mask_bool, score = run_sam2_with_gaze_point(
                    sam2_predictor=sam2_predictor,
                    rgb_rot_rgb=rgb_rot_rgb,
                    gaze_xy=(gx, gy),
                    multimask_output=True,
                )

                print(f"[SAM2] Done. score={score:.4f}")

                _, _, _, cutout_rgb = save_three_outputs(
                    rgb_rot_rgb=rgb_rot_rgb,
                    mask_bool=mask_bool,
                    out_dir=out_dir,
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


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
