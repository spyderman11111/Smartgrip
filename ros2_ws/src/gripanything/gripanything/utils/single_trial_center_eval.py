#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_center_size_eval.py

Batch-evaluate center XY errors (mm) and size errors (mm) for 25 yellow + 25 blue trials.

Rules (per user request):
- GT center (rx,ry) in mm: from user's table (hard-coded below)
- Pred center (x,y,z) in meters: read from each trial's object_in_base_link.json
  -> convert to mm, then apply your sign convention:
       pred_x_mm = -x_mm
       pred_y_mm = -y_mm
  -> IMPORTANT: JSON x/y are always ~2 mm smaller => apply +2 mm to BOTH x and y by default.

Size evaluation:
- Pred size from JSON prism extent or corners_8
- GT size (cube edge length, cm):
    yellow: 3.962 cm
    blue  : 2.508 cm

Rotation-style XY error analysis:
- Decompose error into radial/tangential, estimate small yaw angle.

Outputs:
- CSV table (default: ./center_size_eval.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


# ---------------------------
# GT centers (mm): from your latest table (rx, ry)
# ---------------------------
GT_XY_MM: Dict[str, Dict[int, Tuple[float, float]]] = {
    "yellow": {
        1: (-398.1,  -463.7),
        2: (-496.7,  -492.78),
        3: (-522.43, -406.66),
        4: (-474.64, -427.15),
        5: (-414.43, -493.87),
        6: (-424.92, -497.75),
        7: (-348.4,  -449.95),
        8: (-446.2,  -507.77),
        9: (-390.7,  -473.75),
        10: (-318.9,  -458.26),
        11: (-366.2,  -535.31),
        12: (-329.25, -515.32),
        13: (-332.54, -449.76),
        14: (-426.26, -541.45),
        15: (-262.98, -557.12),
        16: (-440.0,  -359.98),
        17: (-486.01, -495.8),
        18: (-483.22, -426.53),
        19: (-466.6,  -366.9),
        20: (-592.72, -456.41),
        21: (-577.01, -370.86),
        22: (-599.27, -299.35),
        23: (-467.93, -342.12),
        24: (-458.2,  -474.5),
        25: (-394.19, -369.47),
    },
    "blue": {
        1: (-383.12, -366.4),
        2: (-430.86, -456.55),
        3: (-533.25, -459.33),
        4: (-455.48, -364.12),
        5: (-383.0,  -495.2),
        6: (-413.44, -336.44),
        7: (-515.58, -432.15),
        8: (-510.64, -508.2),
        9: (-441.6,  -542.05),
        10: (-356.9,  -438.12),
        11: (-434.17, -462.29),
        12: (-273.33, -470.8),
        13: (-383.63, -525.37),
        14: (-348.88, -470.68),
        15: (-517.39, -516.02),
        16: (-463.3,  -533.42),
        17: (-567.77, -511.92),
        18: (-259.1,  -520.9),
        19: (-300.53, -382.16),
        20: (-256.75, -365.58),
        21: (-585.67, -263.35),
        22: (-636.55, -262.31),
        23: (-444.34, -380.0),
        24: (-559.04, -494.0),
        25: (-558.16, -408.44),
    },
}

# GT cube edge length (cm)
GT_SIDE_CM = {"yellow": 3.962, "blue": 2.508}


# ---------------------------
# JSON parsing helpers
# ---------------------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _as_vec3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x):
        return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=float)
    return None


def _find_object_json(trial_dir: str) -> str:
    td = os.path.abspath(trial_dir)
    cand = [
        os.path.join(td, "output", "offline_output", "object_in_base_link.json"),
        os.path.join(td, "offline_output", "object_in_base_link.json"),
        os.path.join(td, "object_in_base_link.json"),
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Cannot find object_in_base_link.json under:\n  - " + "\n  - ".join(cand))


def _read_center_m(data: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Return center in meters as np.array([x,y,z]) if possible.
    Priority:
      object.center.base_link
    Fallback:
      object.center_base / center_base_link / center_b
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


def _try_read_extent_m(data: Dict[str, Any]) -> Optional[np.ndarray]:
    obj = data.get("object", {})
    if not isinstance(obj, dict):
        return None
    prism = obj.get("prism", {})
    if not isinstance(prism, dict):
        return None

    ext3 = _as_vec3(prism.get("extent_xyz_in_prism_axes", None))
    if ext3 is None:
        return None

    am = _read_align_method(data)
    s = _read_scale_s(data)
    return ext3 * float(s) if am == "sim3" else ext3


def _try_read_corners8_base_m(data: Dict[str, Any]) -> Optional[np.ndarray]:
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
    return np.stack(out, axis=0)


def _extent_from_corners8_m(corners8_b_m: np.ndarray) -> Optional[np.ndarray]:
    if corners8_b_m is None or corners8_b_m.shape != (8, 3):
        return None
    btm = corners8_b_m[:4, :]
    top = corners8_b_m[4:, :]

    d01 = float(np.linalg.norm(btm[1] - btm[0]))
    d12 = float(np.linalg.norm(btm[2] - btm[1]))
    d23 = float(np.linalg.norm(btm[3] - btm[2]))
    d30 = float(np.linalg.norm(btm[0] - btm[3]))

    sx = 0.5 * (d01 + d23)
    sy = 0.5 * (d12 + d30)
    sz = float(np.mean([np.linalg.norm(top[i] - btm[i]) for i in range(4)]))
    return np.array([sx, sy, sz], dtype=float)


# ---------------------------
# Error decomposition
# ---------------------------
def _theta_deg_from_xy(dx_mm: float, dy_mm: float, gt_x_mm: float, gt_y_mm: float, eps: float = 1e-9) -> float:
    r2 = float(gt_x_mm * gt_x_mm + gt_y_mm * gt_y_mm)
    if r2 < eps:
        return float("nan")
    theta_rad = float((-gt_y_mm * dx_mm + gt_x_mm * dy_mm) / r2)
    return float(theta_rad * (180.0 / np.pi))


def _decompose_err_rad_tan_mm(dx_mm: float, dy_mm: float, gt_x_mm: float, gt_y_mm: float, eps: float = 1e-9) -> Tuple[float, float]:
    r2 = float(gt_x_mm * gt_x_mm + gt_y_mm * gt_y_mm)
    if r2 < eps:
        return float("nan"), float("nan")
    r = float(np.sqrt(r2))
    e_rad = float((dx_mm * gt_x_mm + dy_mm * gt_y_mm) / r)
    e_tan = float((-dx_mm * gt_y_mm + dy_mm * gt_x_mm) / r)
    return e_rad, e_tan


def _apply_xy_convention(
    x_mm: float,
    y_mm: float,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    off_x_mm: float,
    off_y_mm: float,
) -> Tuple[float, float]:
    if swap_xy:
        x_mm, y_mm = y_mm, x_mm
    if flip_x:
        x_mm = -x_mm
    if flip_y:
        y_mm = -y_mm
    x_mm += float(off_x_mm)
    y_mm += float(off_y_mm)
    return x_mm, y_mm


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--yellow_root", type=str, default="/home/MA_SmartGrip/Smartgrip/result/single/y",
                    help="Folder containing y1..y25")
    ap.add_argument("--blue_root", type=str, default="/home/MA_SmartGrip/Smartgrip/result/single/b",
                    help="Folder containing b1..b25")
    ap.add_argument("--n", type=int, default=25, help="Number of trials per color.")

    # IMPORTANT: JSON x/y are ~2mm smaller => default +2mm to pred x/y
    ap.add_argument("--offset_x_mm", type=float, default=2.0, help="Additive offset applied to predicted x (mm).")
    ap.add_argument("--offset_y_mm", type=float, default=2.0, help="Additive offset applied to predicted y (mm).")
    ap.add_argument("--swap_xy", action="store_true", help="Swap predicted x and y before eval.")
    ap.add_argument("--flip_x", action="store_true", help="Flip sign of predicted x before eval.")
    ap.add_argument("--flip_y", action="store_true", help="Flip sign of predicted y before eval.")

    ap.add_argument("--out_csv", type=str, default="center_size_eval.csv", help="Output CSV path.")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []

    for color, root, prefix in [("yellow", args.yellow_root, "y"), ("blue", args.blue_root, "b")]:
        gt_side_cm = float(GT_SIDE_CM[color])

        for idx in range(1, int(args.n) + 1):
            trial_dir = os.path.join(os.path.abspath(root), f"{prefix}{idx}")

            row: Dict[str, Any] = {
                "color": color,
                "trial": idx,
                "trial_dir": trial_dir,
                "json_path": "",
                "align_method": "",
                "scale_s": float("nan"),

                # Pred from JSON center (mm)
                "pred_x_mm": float("nan"),
                "pred_y_mm": float("nan"),
                "pred_z_mm": float("nan"),

                # GT from table (mm)
                "gt_x_mm": float("nan"),
                "gt_y_mm": float("nan"),

                # center errors (mm)
                "dx_mm": float("nan"),
                "dy_mm": float("nan"),
                "e_xy_mm": float("nan"),

                # rotation-style metrics
                "gt_r_mm": float("nan"),
                "e_rad_mm": float("nan"),
                "e_tan_mm": float("nan"),
                "theta_deg": float("nan"),

                # predicted size (cm)
                "pred_sx_cm": float("nan"),
                "pred_sy_cm": float("nan"),
                "pred_sz_cm": float("nan"),
                "pred_side_cm_med": float("nan"),

                # gt size (cm)
                "gt_side_cm": gt_side_cm,

                # size errors (mm)
                "err_sx_mm": float("nan"),
                "err_sy_mm": float("nan"),
                "err_sz_mm": float("nan"),
                "err_side_mm_med": float("nan"),
            }

            # ---- GT from table ----
            if color not in GT_XY_MM or idx not in GT_XY_MM[color]:
                row["json_path"] = "NO_GT_ENTRY"
                rows.append(row)
                continue
            gt_x_mm, gt_y_mm = GT_XY_MM[color][idx]
            row["gt_x_mm"] = float(gt_x_mm)
            row["gt_y_mm"] = float(gt_y_mm)

            # ---- Load JSON ----
            try:
                obj_json = _find_object_json(trial_dir)
                row["json_path"] = obj_json
                with open(obj_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                row["json_path"] = f"MISSING_JSON ({e})"
                rows.append(row)
                continue

            row["align_method"] = str(_read_align_method(data))
            row["scale_s"] = float(_read_scale_s(data))

            # ---- Pred center from JSON ----
            c_m = _read_center_m(data)
            if c_m is not None:
                x_mm = float(c_m[0] * 1000.0)
                y_mm = float(c_m[1] * 1000.0)
                z_mm = float(c_m[2] * 1000.0)

                # keep your convention: negate x/y
                pred_x_mm = -x_mm
                pred_y_mm = -y_mm

                # apply swap/flip/offset (+2mm default)
                pred_x_mm, pred_y_mm = _apply_xy_convention(
                    pred_x_mm, pred_y_mm,
                    swap_xy=bool(args.swap_xy),
                    flip_x=bool(args.flip_x),
                    flip_y=bool(args.flip_y),
                    off_x_mm=float(args.offset_x_mm),
                    off_y_mm=float(args.offset_y_mm),
                )

                row["pred_x_mm"] = float(pred_x_mm)
                row["pred_y_mm"] = float(pred_y_mm)
                row["pred_z_mm"] = float(z_mm)

                dx = float(pred_x_mm - gt_x_mm)
                dy = float(pred_y_mm - gt_y_mm)
                row["dx_mm"] = dx
                row["dy_mm"] = dy
                row["e_xy_mm"] = float(np.hypot(dx, dy))

                row["gt_r_mm"] = float(np.hypot(gt_x_mm, gt_y_mm))
                e_rad, e_tan = _decompose_err_rad_tan_mm(dx, dy, gt_x_mm, gt_y_mm)
                row["e_rad_mm"] = float(e_rad)
                row["e_tan_mm"] = float(e_tan)
                row["theta_deg"] = float(_theta_deg_from_xy(dx, dy, gt_x_mm, gt_y_mm))

            # ---- Pred size from JSON (best effort) ----
            extent_m = _try_read_extent_m(data)
            if extent_m is None:
                c8 = _try_read_corners8_base_m(data)
                if c8 is not None:
                    extent_m = _extent_from_corners8_m(c8)

            if extent_m is not None:
                extent_cm = extent_m * 100.0
                row["pred_sx_cm"] = float(extent_cm[0])
                row["pred_sy_cm"] = float(extent_cm[1])
                row["pred_sz_cm"] = float(extent_cm[2])

                pred_side_cm_med = float(np.median(extent_cm))
                row["pred_side_cm_med"] = pred_side_cm_med

                row["err_sx_mm"] = float((row["pred_sx_cm"] - gt_side_cm) * 10.0)
                row["err_sy_mm"] = float((row["pred_sy_cm"] - gt_side_cm) * 10.0)
                row["err_sz_mm"] = float((row["pred_sz_cm"] - gt_side_cm) * 10.0)
                row["err_side_mm_med"] = float((pred_side_cm_med - gt_side_cm) * 10.0)

            rows.append(row)

    # ---- Write CSV ----
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()
