#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aria live RGB + EyeTrack -> gaze projection on RGB -> SAM2 single-frame ROI box segmentation (local weights).

Keys:
- Press 's' to run SAM2 on a gaze-centered ROI box and save 4 outputs (NO timestamp, fixed names).

Outputs (fixed names, PNG):
  1) gaze_sam2_mask_bin.png
     - binary mask (0/255), full image, ROI outside forced 0
  2) gaze_sam2_roi_mask_rgb.png
     - color masked image within ROI (black background; only mask pixels kept)
  3) gaze_sam2_rgb_rot.png
     - rotated RGB (no overlay)
  4) gaze_sam2_debug_gaze_roi.png
     - debug RGB with gaze dot + ROI box + mask contour (blue)

Main-script (seeanything.py) compatibility:
- Your seeanything.py looks for latest "*_rgb_rot.png" and then the paired "*_mask_bin.png".
- We keep names ending with "_rgb_rot.png" and "_mask_bin.png" AND write mask first, rgb_rot last
  to avoid race conditions.

Mask semantics:
- Default behavior keeps your old interface: invert_roi_mask=True means we handle the typical
  "SAM2 returns background" issue. Here we do it more robustly:
    - If invert_roi_mask is True: AUTO-choose (normal vs invert) inside ROI (no new CLI flags).
    - If --no-invert-roi-mask: use normal mask (no inversion / no auto).
- ROI outside is always forced to 0.

Fixes in this version:
- Snapshot + lock at 's' press time (eliminates mask/image drift).
- auto format defaults to BGR (reduces color swap / bias).
- optional lightweight mask refinement before saving (close/open + fill holes).
"""

import argparse
import os
import sys
import threading
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

# Default out dir to match seeanything.py expectation:
DEFAULT_ARIA_OUT_DIR = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage"

# =========================
# Mask refinement defaults (no extra CLI needed)
# =========================
MASK_REFINE_ENABLE = True
MASK_CLOSE_KSIZE = 5   # odd
MASK_OPEN_KSIZE = 3    # odd
MASK_FILL_HOLES = True
MASK_ERODE_ITERS = 0   # if boundary is too "fat", set 1
MASK_DILATE_ITERS = 0  # if boundary is too "thin", set 1


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

    # output directory (must match seeanything.py ARIA_OUT_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_ARIA_OUT_DIR)

    # fixed output basenames (NO timestamp)
    p.add_argument("--fn-mask-bin", type=str, default="gaze_sam2_mask_bin.png")
    p.add_argument("--fn-roi-mask-rgb", type=str, default="gaze_sam2_roi_mask_rgb.png")
    p.add_argument("--fn-rgb-rot", type=str, default="gaze_sam2_rgb_rot.png")
    p.add_argument("--fn-debug", type=str, default="gaze_sam2_debug_gaze_roi.png")

    # keep your old interface:
    # - default True: we will AUTO choose normal/invert (robust)
    # - pass --no-invert-roi-mask: use normal mask (no inversion / no auto)
    p.add_argument(
        "--no-invert-roi-mask",
        dest="invert_roi_mask",
        action="store_false",
        help="Disable inversion/auto-invert logic; use SAM2 normal mask in ROI.",
    )
    p.set_defaults(invert_roi_mask=True)

    return p.parse_args()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_imwrite_png(path: str, img: np.ndarray) -> None:
    """
    Atomic PNG write that does NOT depend on filename extension.
    Uses cv2.imencode('.png', ...) then writes bytes to a temp file and os.replace().
    """
    d = os.path.dirname(os.path.abspath(path))
    _ensure_dir(d)

    if img is None:
        raise RuntimeError("atomic_imwrite_png: img is None")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    img = np.ascontiguousarray(img)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode('.png', ...) failed for: {path}")

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())

    os.replace(tmp, path)


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


def _build_rgbmask_blackbg(rgb_rot_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """
    Build object-only RGB image (black background) from boolean mask.
    rgb_rot_rgb is RGB; output is RGB.
    """
    out = np.zeros_like(rgb_rot_rgb, dtype=np.uint8)
    if mask_bool is not None and mask_bool.any():
        out[mask_bool] = rgb_rot_rgb[mask_bool]
    return out


# =========================
# Mask auto-select + refine
# =========================

def _ensure_odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def _fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Fill holes for a 0/255 uint8 mask."""
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    h, w = mask_u8.shape[:2]
    if h == 0 or w == 0:
        return mask_u8

    inv = (mask_u8 == 0).astype(np.uint8)  # 1 where background
    ff = inv.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 2)  # mark border-connected background as 2
    holes = (ff == 1)  # not connected to border => holes
    out = mask_u8.copy()
    out[holes] = 255
    return out


def _auto_choose_normal_or_invert(roi_mask_bool: np.ndarray) -> np.ndarray:
    """
    Auto choose between normal mask and inverted mask inside ROI.
    Heuristic prefers:
    - not too small / not full ROI
    - fewer border-touch pixels (background mask often touches ROI border heavily)
    """
    m0 = roi_mask_bool.astype(bool)
    m1 = ~m0

    def cost(m: np.ndarray) -> float:
        area = int(m.sum())
        h, w = m.shape[:2]
        if area <= 0:
            return 1e9
        frac = area / float(max(1, h * w))

        border = np.zeros_like(m, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        touch = int((m & border).sum())
        touch_ratio = touch / float(max(1, area))

        penalty = 0.0
        if frac < 0.01:
            penalty += 10.0
        if frac > 0.90:
            penalty += 10.0
        return touch_ratio + 0.3 * abs(frac - 0.25) + penalty

    return m0 if cost(m0) <= cost(m1) else m1


def _refine_mask_bool(mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mask_bool_refined, mask_u8 (0/255)
    """
    if not MASK_REFINE_ENABLE:
        m = mask_bool.astype(bool)
        u8 = (m.astype(np.uint8) * 255)
        return m, u8

    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    ck = _ensure_odd(MASK_CLOSE_KSIZE)
    ok = _ensure_odd(MASK_OPEN_KSIZE)

    if ck > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    if ok > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    if MASK_FILL_HOLES:
        mask_u8 = _fill_holes_u8(mask_u8)

    if MASK_ERODE_ITERS > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=int(MASK_ERODE_ITERS))

    if MASK_DILATE_ITERS > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=int(MASK_DILATE_ITERS))

    return (mask_u8 > 0), mask_u8


def save_outputs_fixed(
    rgb_rot_rgb: np.ndarray,
    mask_bool: np.ndarray,
    dbg_bgr: np.ndarray,
    out_dir: str,
    fn_mask_bin: str,
    fn_roi_mask_rgb: str,
    fn_rgb_rot: str,
    fn_debug: str,
):
    """
    Save 4 outputs with fixed names (no timestamp) to out_dir.

    IMPORTANT for seeanything.py pairing logic:
    - write mask_bin first, rgb_rot last
    - use atomic write for all files
    """
    _ensure_dir(out_dir)

    p_mask = os.path.join(out_dir, fn_mask_bin)
    p_roi_rgb = os.path.join(out_dir, fn_roi_mask_rgb)
    p_dbg = os.path.join(out_dir, fn_debug)
    p_rgb = os.path.join(out_dir, fn_rgb_rot)

    if mask_bool is None:
        raise RuntimeError("save_outputs_fixed: mask_bool is None")

    # refine before saving (stabilize boundary + remove tiny artifacts)
    mask_bool_ref, mask_u8 = _refine_mask_bool(mask_bool)

    # 1) binary mask: 0/255 uint8
    _atomic_imwrite_png(p_mask, mask_u8)

    # 2) ROI masked color image (black background)
    roi_rgb = _build_rgbmask_blackbg(rgb_rot_rgb, mask_bool_ref)
    _atomic_imwrite_png(p_roi_rgb, cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

    # 3) debug (add mask contour)
    dbg_out = dbg_bgr.copy()
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts is not None and len(cnts) > 0:
        cv2.drawContours(dbg_out, cnts, -1, (255, 0, 0), 2)  # blue contour
    _atomic_imwrite_png(p_dbg, dbg_out)

    # 4) rotated rgb (no overlay) - write LAST to avoid race
    _atomic_imwrite_png(p_rgb, cv2.cvtColor(rgb_rot_rgb, cv2.COLOR_RGB2BGR))

    print("[Saved-fixed]")
    print(f"  mask_bin      : {p_mask}")
    print(f"  roi_mask_rgb  : {p_roi_rgb}")
    print(f"  debug         : {p_dbg}")
    print(f"  rgb_rot       : {p_rgb}")

    return roi_rgb


# =========================
# Observer with lock + consistent format
# =========================

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

        # Key fix: auto defaults to BGR (common for OpenCV pipelines)
        self.fmt = "bgr" if aria_rgb_format == "auto" else aria_rgb_format

        self.lock = threading.Lock()

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
        img_t = torch.tensor(image, device=self.torch_device)
        with torch.no_grad():
            preds, _, _ = self.model.predict(img_t)
        preds = preds.detach().cpu().numpy()
        yaw = float(preds[0][0])
        pitch = float(preds[0][1])

        with self.lock:
            self.last_eye_raw = image
            self.yaw = yaw
            self.pitch = pitch

    def _on_rgb(self, image: np.ndarray, record: ImageDataRecord) -> None:
        rotated = np.rot90(image, -1)
        rgb_rot_rgb = to_sam_rgb(rotated, self.fmt)  # ensure RGB for SAM2
        h_rot, w_rot = rgb_rot_rgb.shape[:2]

        with self.lock:
            self.last_rgb_raw = image
            self.rgb_rot_rgb = rgb_rot_rgb
            self.rgb_rot_shape = (h_rot, w_rot)
            yaw = self.yaw
            pitch = self.pitch

        if yaw is None or pitch is None:
            with self.lock:
                self.gaze_rot_xy = None
            return

        try:
            eg = EyeGaze()
        except TypeError:
            eg = EyeGaze
        eg.yaw = yaw
        eg.pitch = pitch
        eg.depth = self.depth_m

        proj = get_gaze_vector_reprojection(
            eg, self.rgb_label, self.dev_calib, self.rgb_calib, self.depth_m
        )
        if proj is None:
            with self.lock:
                self.gaze_rot_xy = None
            return

        x_raw, y_raw = float(proj[0]), float(proj[1])
        h_raw, _ = image.shape[:2]
        x_rot, y_rot = rot_cw90_xy(x_raw, y_raw, h_raw)

        with self.lock:
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
    else:
        scfg.streaming_interface = aria.StreamingInterface.Wifi
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

    _ensure_dir(args.out_dir)
    print(f"[OutDir] {args.out_dir}")
    print("[OutFiles]")
    print(f"  mask_bin     : {args.fn_mask_bin}")
    print(f"  roi_mask_rgb : {args.fn_roi_mask_rgb}")
    print(f"  rgb_rot      : {args.fn_rgb_rot}")
    print(f"  debug        : {args.fn_debug}")

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
    sam_win = "SAM2 ROI Mask RGB"

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
    print(f"[Main] invert_roi_mask = {args.invert_roi_mask} (True means auto choose normal/invert; disable with --no-invert-roi-mask)")
    print(f"[Main] MASK_REFINE_ENABLE={MASK_REFINE_ENABLE}, close={MASK_CLOSE_KSIZE}, open={MASK_OPEN_KSIZE}, fill_holes={MASK_FILL_HOLES}, erode={MASK_ERODE_ITERS}, dilate={MASK_DILATE_ITERS}")

    latest_roi_rgb_bgr = None
    graceful_exit = True

    try:
        while True:
            # --- Display RGB ---
            with obs.lock:
                last_rgb_raw = None if obs.last_rgb_raw is None else obs.last_rgb_raw.copy()
                gaze_xy_f = None if obs.gaze_rot_xy is None else (float(obs.gaze_rot_xy[0]), float(obs.gaze_rot_xy[1]))
                shape = None if obs.rgb_rot_shape is None else (int(obs.rgb_rot_shape[0]), int(obs.rgb_rot_shape[1]))

            if last_rgb_raw is not None:
                rotated = np.rot90(last_rgb_raw, -1)
                fmt = "bgr" if args.aria_rgb_format == "auto" else args.aria_rgb_format
                vis = to_display_bgr(rotated, fmt)

                if gaze_xy_f is not None and shape is not None:
                    h, w = shape
                    gx, gy = clamp_xy(gaze_xy_f[0], gaze_xy_f[1], w, h)
                    cv2.circle(vis, (gx, gy), 12, (0, 255, 0), thickness=-1)
                    x1, y1, x2, y2 = make_roi_box(gx, gy, args.gaze_box_size, w, h)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                cv2.imshow(rgb_win, vis)

            # --- Display Eye ---
            with obs.lock:
                last_eye_raw = None if obs.last_eye_raw is None else obs.last_eye_raw.copy()

            if last_eye_raw is not None:
                rotated_eye = np.rot90(last_eye_raw, -1)
                eye_bgr = to_display_bgr(rotated_eye, "gray")
                cv2.imshow(eye_win, eye_bgr)

            # --- Display SAM result ---
            if latest_roi_rgb_bgr is not None:
                cv2.imshow(sam_win, latest_roi_rgb_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # ---- freeze a consistent snapshot (critical to remove drift) ----
                with obs.lock:
                    rgb_rot_rgb = None if obs.rgb_rot_rgb is None else obs.rgb_rot_rgb.copy()
                    shape2 = None if obs.rgb_rot_shape is None else (int(obs.rgb_rot_shape[0]), int(obs.rgb_rot_shape[1]))
                    gaze2 = None if obs.gaze_rot_xy is None else (float(obs.gaze_rot_xy[0]), float(obs.gaze_rot_xy[1]))

                if rgb_rot_rgb is None or shape2 is None or gaze2 is None:
                    print("[SAM2] Missing RGB/gaze.")
                    continue

                h, w = shape2
                gx, gy = clamp_xy(gaze2[0], gaze2[1], w, h)

                full_mask, score, (x1, y1, x2, y2) = sam2_segment_roi_box(
                    predictor, rgb_rot_rgb, (gx, gy), args.gaze_box_size
                )
                if full_mask is None:
                    print("[SAM2] No mask.")
                    continue

                # ROI-only selection, ROI outside forced 0
                roi_sel = full_mask[y1:y2, x1:x2].astype(bool)

                if args.invert_roi_mask:
                    # robust: auto choose normal vs invert inside ROI
                    roi_sel = _auto_choose_normal_or_invert(roi_sel)
                # else: keep normal SAM2 output

                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = roi_sel

                print(
                    f"[SAM2] score={score:.4f} ROI=({x1},{y1})-({x2},{y2}) "
                    f"mask_pixels={int(mask.sum())} invert_auto={args.invert_roi_mask}"
                )

                dbg = cv2.cvtColor(rgb_rot_rgb, cv2.COLOR_RGB2BGR)
                cv2.circle(dbg, (gx, gy), 12, (0, 255, 0), thickness=-1)
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

                roi_rgb = save_outputs_fixed(
                    rgb_rot_rgb,
                    mask,
                    dbg,
                    out_dir=args.out_dir,
                    fn_mask_bin=args.fn_mask_bin,
                    fn_roi_mask_rgb=args.fn_roi_mask_rgb,
                    fn_rgb_rot=args.fn_rgb_rot,
                    fn_debug=args.fn_debug,
                )
                latest_roi_rgb_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

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
  --gaze-box-size 200 \
  --sam2-root /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2 \
  --sam2-ckpt /home/MA_SmartGrip/Smartgrip/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --out-dir /home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage
"""