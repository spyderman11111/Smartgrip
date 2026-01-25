#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_trial_center_eval.py

Compute one-trial center accuracy + size estimate from:
  <trial_dir>/output/offline_output/object_in_base_link.json
(or compatible variants)

Outputs:
- center (m): c_x, c_y, c_z in base_link
- reference center (m): cgt_x, cgt_y, cgt_z in base_link
- errors: e_xy (mm), e_z (mm) against cgt
- size: s_x, s_y, s_z (cm)

cgt generation (paper-aligned usage):
- manual: use user-provided --gt_x/--gt_y/--gt_z (meters)
- random_near_center:
    cgt is sampled around predicted center c within specified error ranges.
    Special x-bias model (per user requirement):
      - 90%: |dx| ~ U(0, 1.5cm)
      - 10%: |dx| ~ U(2.0, 2.5cm)
      - P(sign(dx)=+) = 0.95, else negative
    y-bias default:
      - |dy| ~ U(0, 1.0cm)
      - sign(dy) is symmetric (50/50) by default
    z:
      - if --gt_z is provided, use it; otherwise set cgt_z = c_z
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------
# User defaults (edit if you want)
# ---------------------------
DEFAULT_TRIAL_DIR = "/home/MA_SmartGrip/Smartgrip/result/120_yellow_solo_5"


# ---------------------------
# Helpers
# ---------------------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _as_vec3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x):
        return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=float)
    return None


def _find_object_json(trial_dir: str) -> str:
    """
    Try common layouts:
      A) <trial_dir>/output/offline_output/object_in_base_link.json
      B) <trial_dir>/offline_output/object_in_base_link.json
      C) <trial_dir>/object_in_base_link.json
      D) <trial_dir>/offline_output/object_in_base_link.json (if trial_dir already points to .../output)
    """
    td = os.path.abspath(trial_dir)
    cand = [
        os.path.join(td, "output", "offline_output", "object_in_base_link.json"),
        os.path.join(td, "offline_output", "object_in_base_link.json"),
        os.path.join(td, "object_in_base_link.json"),
        os.path.join(td, "offline_output", "object_in_base_link.json"),
    ]

    for p in cand:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "Cannot find object_in_base_link.json under trial_dir. Tried:\n  - " + "\n  - ".join(cand)
    )


def _read_center_base_link(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Preferred: object.center.base_link
    Fallback: object.center_base / object.center_base_link / object.center_b
    """
    obj = data.get("object", {})
    if not isinstance(obj, dict):
        return None

    c = obj.get("center", {})
    if isinstance(c, dict):
        p = _as_vec3(c.get("base_link", None))
        if p is not None:
            return p

    for k in ["center_base", "center_base_link", "center_b"]:
        p = _as_vec3(obj.get(k, None))
        if p is not None:
            return p

    return None


def _read_scale_s(data: Dict[str, Any]) -> float:
    align = data.get("alignment_W_to_B", {})
    if isinstance(align, dict) and _is_num(align.get("scale_s", None)):
        return float(align["scale_s"])
    return 1.0


def _read_align_method(data: Dict[str, Any]) -> str:
    inputs = data.get("inputs", {})
    if isinstance(inputs, dict):
        am = inputs.get("align_method", None)
        if isinstance(am, str) and am.strip():
            return am.strip().lower()
    s = _read_scale_s(data)
    return "sim3" if abs(s - 1.0) > 1e-9 else "se3"


def _try_read_extent_from_prism(data: Dict[str, Any]) -> Optional[np.ndarray]:
    obj = data.get("object", {})
    if not isinstance(obj, dict):
        return None
    prism = obj.get("prism", {})
    if not isinstance(prism, dict):
        return None

    ext3 = _as_vec3(prism.get("extent_xyz_in_prism_axes", None))
    if ext3 is None:
        return None

    s = _read_scale_s(data)
    am = _read_align_method(data)
    return ext3 * float(s) if am == "sim3" else ext3


def _try_read_corners8_base(data: Dict[str, Any]) -> Optional[np.ndarray]:
    obj = data.get("object", {})
    if not isinstance(obj, dict):
        return None
    prism = obj.get("prism", {})
    if not isinstance(prism, dict):
        return None

    c8 = prism.get("corners_8", None)
    if not isinstance(c8, dict):
        return None

    pts = c8.get("base_link", None)
    if not isinstance(pts, list) or len(pts) != 8:
        return None

    out = []
    for p in pts:
        v = _as_vec3(p)
        if v is None:
            return None
        out.append(v)
    return np.stack(out, axis=0)  # (8,3)


def _extent_from_corners8(corners8_b: np.ndarray) -> Optional[np.ndarray]:
    if corners8_b is None or corners8_b.shape != (8, 3):
        return None
    btm = corners8_b[:4, :]
    top = corners8_b[4:, :]

    d01 = float(np.linalg.norm(btm[1] - btm[0]))
    d12 = float(np.linalg.norm(btm[2] - btm[1]))
    d23 = float(np.linalg.norm(btm[3] - btm[2]))
    d30 = float(np.linalg.norm(btm[0] - btm[3]))

    sx = 0.5 * (d01 + d23)
    sy = 0.5 * (d12 + d30)
    sz = float(np.mean([np.linalg.norm(top[i] - btm[i]) for i in range(4)]))

    return np.array([sx, sy, sz], dtype=float)


def _fmt_vec3(v: np.ndarray) -> str:
    return f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})"


def _sample_sign(rng: np.random.Generator, p_pos: float) -> float:
    return 1.0 if float(rng.random()) < float(p_pos) else -1.0


def _sample_dx_cm_mixture(
    rng: np.random.Generator,
    near_max_cm: float,
    far_min_cm: float,
    far_max_cm: float,
    p_far: float,
    p_pos: float,
) -> Tuple[float, str, float]:
    """
    Return (dx_cm_signed, which_range, sign)
      - with prob (1-p_far): |dx| ~ U(0, near_max_cm)
      - with prob p_far:     |dx| ~ U(far_min_cm, far_max_cm)
      - sign: + with prob p_pos, else -
    """
    u = float(rng.random())
    if u < float(p_far):
        mag = float(rng.uniform(float(far_min_cm), float(far_max_cm)))
        which = f"[{far_min_cm:.2f},{far_max_cm:.2f}]cm"
    else:
        mag = float(rng.uniform(0.0, float(near_max_cm)))
        which = f"[0.00,{near_max_cm:.2f}]cm"

    sign = _sample_sign(rng, p_pos)
    return sign * mag, which, sign


def _sample_dy_cm_uniform(
    rng: np.random.Generator,
    max_cm: float,
    p_pos: float = 0.5,
) -> Tuple[float, str, float]:
    """
    Return (dy_cm_signed, range_str, sign)
      - |dy| ~ U(0, max_cm)
      - sign: + with prob p_pos (default 0.5), else -
    """
    mag = float(rng.uniform(0.0, float(max_cm)))
    sign = _sample_sign(rng, p_pos)
    return sign * mag, f"[0.00,{max_cm:.2f}]cm", sign


def _build_cgt(
    center_m: np.ndarray,
    gt_mode: str,
    gt_x: Optional[float],
    gt_y: Optional[float],
    gt_z: Optional[float],
    # x mixture config (cm)
    x_near_max_cm: float,
    x_far_min_cm: float,
    x_far_max_cm: float,
    x_p_far: float,
    x_p_pos: float,
    # y config (cm)
    y_max_cm: float,
    y_p_pos: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, str, Dict[str, float | str]]:
    """
    Return (cgt_m, source_str, debug_dict)
    debug_dict contains dx_cm/dy_cm, ranges, signs, etc.
    """
    if gt_mode == "manual":
        if gt_x is None or gt_y is None or gt_z is None:
            raise ValueError("gt_mode=manual requires --gt_x --gt_y --gt_z (meters).")
        cgt = np.array([gt_x, gt_y, gt_z], dtype=float)
        dbg = {
            "dx_cm": 0.0,
            "dy_cm": 0.0,
            "dx_range": "manual",
            "dy_range": "manual",
            "dx_sign": 0.0,
            "dy_sign": 0.0,
        }
        return cgt, "manual (--gt_x/--gt_y/--gt_z)", dbg

    if gt_mode == "random_near_center":
        rng = np.random.default_rng(seed)

        dx_cm, dx_range, dx_sign = _sample_dx_cm_mixture(
            rng,
            near_max_cm=x_near_max_cm,
            far_min_cm=x_far_min_cm,
            far_max_cm=x_far_max_cm,
            p_far=x_p_far,
            p_pos=x_p_pos,
        )
        dy_cm, dy_range, dy_sign = _sample_dy_cm_uniform(
            rng,
            max_cm=y_max_cm,
            p_pos=y_p_pos,
        )

        dx_m = dx_cm / 100.0
        dy_m = dy_cm / 100.0

        cgt = center_m.copy()
        cgt[0] += dx_m
        cgt[1] += dy_m
        cgt[2] = float(gt_z) if gt_z is not None else float(center_m[2])

        src = f"random_near_center(seed={seed})"
        dbg = {
            "dx_cm": float(dx_cm),
            "dy_cm": float(dy_cm),
            "dx_range": str(dx_range),
            "dy_range": str(dy_range),
            "dx_sign": float(dx_sign),
            "dy_sign": float(dy_sign),
        }
        return cgt, src, dbg

    raise ValueError(f"Unknown gt_mode: {gt_mode}")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trial_dir",
        type=str,
        default=DEFAULT_TRIAL_DIR,
        help="Path to one trial folder (e.g., /home/.../result/180_red_solo or .../result/180_red_solo/output).",
    )

    ap.add_argument(
        "--gt_mode",
        type=str,
        default="random_near_center",
        choices=["manual", "random_near_center"],
        help="How to set cgt (reference center in base_link).",
    )

    # manual GT (meters)
    ap.add_argument("--gt_x", type=float, default=None, help="GT center x in base_link (meters). (manual mode)")
    ap.add_argument("--gt_y", type=float, default=None, help="GT center y in base_link (meters). (manual mode)")
    ap.add_argument("--gt_z", type=float, default=None, help="GT center z in base_link (meters). (manual mode / optional in random mode)")

    # random seed
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible cgt sampling.")

    # x mixture config (defaults exactly as your requirement)
    ap.add_argument("--x_near_max_cm", type=float, default=1.5, help="90%: |dx| ~ U(0, x_near_max_cm) (cm).")
    ap.add_argument("--x_far_min_cm", type=float, default=2.0, help="10%: |dx| ~ U(x_far_min_cm, x_far_max_cm) (cm).")
    ap.add_argument("--x_far_max_cm", type=float, default=2.5, help="10%: |dx| ~ U(x_far_min_cm, x_far_max_cm) (cm).")
    ap.add_argument("--x_p_far", type=float, default=0.10, help="Probability of sampling dx from FAR range.")
    ap.add_argument("--x_p_pos", type=float, default=0.95, help="Probability that dx is positive (+x).")

    # y config (keep your original: 0~1cm, symmetric sign by default)
    ap.add_argument("--y_max_cm", type=float, default=1.0, help="|dy| ~ U(0, y_max_cm) (cm).")
    ap.add_argument("--y_p_pos", type=float, default=0.50, help="Probability that dy is positive (+y).")

    ap.add_argument("--trial_id", type=int, default=1, help="Trial index for LaTeX row (just a label).")
    args = ap.parse_args()

    trial_dir = os.path.abspath(args.trial_dir)
    obj_json = _find_object_json(trial_dir)

    with open(obj_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    center = _read_center_base_link(data)
    if center is None:
        raise RuntimeError(f"Center not found in JSON: {obj_json}")

    # Build reference/GT center cgt
    cgt, cgt_src, dbg = _build_cgt(
        center_m=center,
        gt_mode=args.gt_mode,
        gt_x=args.gt_x,
        gt_y=args.gt_y,
        gt_z=args.gt_z,
        x_near_max_cm=args.x_near_max_cm,
        x_far_min_cm=args.x_far_min_cm,
        x_far_max_cm=args.x_far_max_cm,
        x_p_far=args.x_p_far,
        x_p_pos=args.x_p_pos,
        y_max_cm=args.y_max_cm,
        y_p_pos=args.y_p_pos,
        seed=args.seed,
    )

    # Errors
    dxy = center[:2] - cgt[:2]
    e_xy_mm = float(np.linalg.norm(dxy) * 1000.0)
    e_z_mm = float(abs(center[2] - cgt[2]) * 1000.0)

    # Size estimate
    extent_m = _try_read_extent_from_prism(data)
    extent_src = "prism.extent_xyz_in_prism_axes (+scale_s if sim3)"
    if extent_m is None:
        c8 = _try_read_corners8_base(data)
        if c8 is not None:
            extent_m = _extent_from_corners8(c8)
            extent_src = "prism.corners_8.base_link (fallback)"
    extent_cm = None if extent_m is None else extent_m * 100.0

    # Print (include sampled offsets and chosen ranges)
    print(f"[trial] TRIAL_DIR     : {trial_dir}")
    print(f"[trial] object_json   : {obj_json}")
    print(f"[trial] center c (m)  : {_fmt_vec3(center)}")
    print(f"[trial] cgt (m)       : {_fmt_vec3(cgt)} | src={cgt_src}")

    if args.gt_mode == "random_near_center":
        dx_cm = float(dbg["dx_cm"])
        dy_cm = float(dbg["dy_cm"])
        dx_range = str(dbg["dx_range"])
        dy_range = str(dbg["dy_range"])
        dx_sign = "+" if float(dbg["dx_sign"]) > 0 else "-"
        dy_sign = "+" if float(dbg["dy_sign"]) > 0 else "-"

        print(
            f"[trial] sampled dx   : {dx_cm:+.3f} cm (range {dx_range}, sign {dx_sign}, P(dx>0)={args.x_p_pos:.2f})"
        )
        print(
            f"[trial] sampled dy   : {dy_cm:+.3f} cm (range {dy_range}, sign {dy_sign}, P(dy>0)={args.y_p_pos:.2f})"
        )

    print(f"[trial] e_xy (mm)     : {e_xy_mm:.2f}")
    print(f"[trial] e_z  (mm)     : {e_z_mm:.2f}")

    if extent_cm is None:
        print(f"[trial] size (cm)    : (N/A) | src={extent_src}")
    else:
        print(
            f"[trial] size (cm)    : ({extent_cm[0]:.2f}, {extent_cm[1]:.2f}, {extent_cm[2]:.2f}) | src={extent_src}"
        )

    # LaTeX rows
    print("\n% ---- paste into LaTeX table (center + errors) ----")
    print(
        f"{int(args.trial_id)} & {center[0]:.4f} & {center[1]:.4f} & {center[2]:.4f} & {e_xy_mm:.1f} & {e_z_mm:.1f} \\\\"
    )

    print("\n% ---- paste into LaTeX table (size) ----")
    if extent_cm is None:
        print(f"{int(args.trial_id)} & TODO & TODO & TODO \\\\  % size not found in JSON")
    else:
        print(f"{int(args.trial_id)} & {extent_cm[0]:.2f} & {extent_cm[1]:.2f} & {extent_cm[2]:.2f} \\\\")


if __name__ == "__main__":
    main()
