#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_center_size_eval.py

Batch-evaluate center XY errors (mm) and size errors (mm) for 25 yellow + 25 blue trials.

Data sources:
- Pred center/shape: <trial_dir>/**/object_in_base_link.json
- GT center (rx,ry) in mm: hard-coded from the user's table
- GT size (cube edge length, cm):
    yellow: 3.962 cm
    blue  : 2.508 cm

Key correction requested by user:
- Predicted center printed in meters (e.g. 0.405748 m) is about 2 mm smaller than teach pendant GT
  -> by default, apply +2 mm to x and y: pred_x_mm += 2, pred_y_mm += 2
  (can be disabled via CLI)

Outputs:
- CSV table (default: ./center_size_eval.csv)
- Console: per-color summary + a compact table

Notes:
- This script compares XY only, since your GT provides rx, ry.
- Size reading priority:
    1) object.prism.extent_xyz_in_prism_axes (meters). If align_method=sim3 and alignment_W_to_B.scale_s exists,
       multiply by scale_s (keeps consistency with your earlier logic).
    2) object.prism.corners_8.base_link -> derive extents from 8 corners (meters, in base_link).

- Known typo guard:
    yellow #19: rx is "-46.66" in the provided table (very likely "-466.66").
    By default we auto-fix that one entry. Disable via --disable_known_fixes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np


# ---------------------------
# GT centers (mm) from your table
# ---------------------------
GT_XY_MM = {
    "yellow": {
        1: (-398.10, -463.70),
        2: (-496.70, -492.78),
        3: (-522.43, -406.66),
        4: (-474.64, -427.15),
        5: (-414.43, -493.87),
        6: (-401.75, -409.80),
        7: (-348.40, -449.95),
        8: (-446.20, -507.77),
        9: (-390.70, -473.75),
        10: (-318.90, -458.26),
        11: (-366.20, -535.31),
        12: (-329.25, -515.32),
        13: (-332.54, -449.76),
        14: (-426.26, -541.45),
        15: (-262.98, -557.12),
        16: (-440.00, -359.98),
        17: (-486.01, -495.80),
        18: (-483.22, -426.53),
        19: (-466.66,  -366.90),   
        20: (-508.70, -264.44),
        21: (-577.01, -370.86),
        22: (-599.27, -299.35),
        23: (-594.46, -528.96),
        24: (-458.20, -474.50),
        25: (-394.19, -369.47),
    },
    "blue": {
        1: (-383.12, -366.40),
        2: (-430.86, -456.55),
        3: (-481.77, -487.90),
        4: (-455.48, -364.12),
        5: (-383.00, -495.20),
        6: (-413.44, -336.44),
        7: (-515.58, -432.15),
        8: (-510.64, -508.20),
        9: (-441.60, -542.05),
        10: (-356.90, -438.12),
        11: (-320.55, -395.24),
        12: (-273.33, -470.80),
        13: (-383.63, -525.37),
        14: (-348.88, -470.68),
        15: (-517.39, -516.02),
        16: (-463.30, -533.42),
        17: (-563.00, -551.20),
        18: (-259.10, -520.90),
        19: (-300.53, -382.16),
        20: (-256.75, -365.58),
        21: (-585.67, -263.35),
        22: (-636.55, -262.31),
        23: (-444.34, -380.00),
        24: (-559.04, -494.00),
        25: (-558.16, -408.44),
    },
}

# Known GT fixes (toggle-able)
KNOWN_GT_FIXES = {
    ("yellow", 19): {"rx": -466.66},  # most likely typo
}

# GT cube edge length (cm)
GT_SIDE_CM = {
    "yellow": 3.962,  # 39.62 mm
    "blue": 2.508,    # 25.08 mm
}


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
    """
    Return extent (sx,sy,sz) in meters if possible.
    """
    obj = data.get("object", {})
    if not isinstance(obj, dict):
        return None

    prism = obj.get("prism", {})
    if not isinstance(prism, dict):
        return None

    ext3 = _as_vec3(prism.get("extent_xyz_in_prism_axes", None))
    if ext3 is None:
        return None

    # Optional scale handling for sim3
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

    return np.stack(out, axis=0)  # (8,3), meters


def _extent_from_corners8_m(corners8_b_m: np.ndarray) -> Optional[np.ndarray]:
    """
    Derive (sx,sy,sz) in meters from 8 corners.
    Assumes first 4 are bottom, last 4 are top (same convention as your point_processing).
    """
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
# Evaluation
# ---------------------------
def _get_gt_xy_mm(color: str, idx: int, enable_known_fixes: bool) -> Tuple[float, float]:
    rx, ry = GT_XY_MM[color][idx]
    if enable_known_fixes:
        fix = KNOWN_GT_FIXES.get((color, idx), None)
        if isinstance(fix, dict):
            if "rx" in fix:
                rx = float(fix["rx"])
            if "ry" in fix:
                ry = float(fix["ry"])
    return float(rx), float(ry)


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


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--yellow_root", type=str, default="/home/MA_SmartGrip/Smartgrip/result/single/y",
                    help="Folder containing y1..y25")
    ap.add_argument("--blue_root", type=str, default="/home/MA_SmartGrip/Smartgrip/result/single/b",
                    help="Folder containing b1..b25")
    ap.add_argument("--n", type=int, default=25, help="Number of trials per color.")

    # Center XY correction (your requested +2 mm default)
    ap.add_argument("--offset_x_mm", type=float, default=2.0, help="Additive offset applied to predicted x (mm).")
    ap.add_argument("--offset_y_mm", type=float, default=2.0, help="Additive offset applied to predicted y (mm).")
    ap.add_argument("--swap_xy", action="store_true", help="Swap predicted x and y before eval.")
    ap.add_argument("--flip_x", action="store_true", help="Flip sign of predicted x before eval.")
    ap.add_argument("--flip_y", action="store_true", help="Flip sign of predicted y before eval.")

    ap.add_argument("--disable_known_fixes", action="store_true",
                    help="Disable built-in fixes for known GT typos (e.g., yellow#19).")

    ap.add_argument("--out_csv", type=str, default="center_size_eval.csv", help="Output CSV path.")
    args = ap.parse_args()

    enable_known_fixes = not bool(args.disable_known_fixes)

    rows: List[Dict[str, Any]] = []

    for color, root, prefix in [("yellow", args.yellow_root, "y"), ("blue", args.blue_root, "b")]:
        gt_side_cm = float(GT_SIDE_CM[color])

        for idx in range(1, int(args.n) + 1):
            trial_dir = os.path.join(os.path.abspath(root), f"{prefix}{idx}")

            # Defaults for a row (even if missing)
            row = {
                "color": color,
                "trial": idx,
                "trial_dir": trial_dir,
                "json_path": "",
                "align_method": "",
                "scale_s": float("nan"),

                # pred center (mm, after convention)
                "pred_x_mm": float("nan"),
                "pred_y_mm": float("nan"),
                "pred_z_mm": float("nan"),
                

                # gt center (mm)
                "gt_x_mm": float("nan"),
                "gt_y_mm": float("nan"),

                # center errors (mm)
                "dx_mm": float("nan"),
                "dy_mm": float("nan"),
                "e_xy_mm": float("nan"),

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

            # GT center
            rx, ry = _get_gt_xy_mm(color, idx, enable_known_fixes)
            row["gt_x_mm"] = rx
            row["gt_y_mm"] = ry

            # Load JSON
            try:
                obj_json = _find_object_json(trial_dir)
                row["json_path"] = obj_json
                with open(obj_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                row["json_path"] = f"MISSING ({e})"
                rows.append(row)
                continue

            # Alignment info
            row["align_method"] = str(_read_align_method(data))
            row["scale_s"] = float(_read_scale_s(data))

            # Pred center
            c_m = _read_center_m(data)
            if c_m is not None:
                pred_x_mm = float(c_m[0] * 1000.0)
                pred_y_mm = float(c_m[1] * 1000.0)
                pred_z_mm = float(c_m[2] * 1000.0)

                pred_x_mm = -pred_x_mm
                pred_y_mm = -pred_y_mm

                pred_x_mm, pred_y_mm = _apply_xy_convention(
                    pred_x_mm, pred_y_mm,
                    swap_xy=bool(args.swap_xy),
                    flip_x=bool(args.flip_x),
                    flip_y=bool(args.flip_y),
                    off_x_mm=float(args.offset_x_mm),
                    off_y_mm=float(args.offset_y_mm),
                )

                row["pred_x_mm"] = pred_x_mm
                row["pred_y_mm"] = pred_y_mm
                row["pred_z_mm"] = pred_z_mm

                # Errors
                dx = pred_x_mm - rx
                dy = pred_y_mm - ry
                row["dx_mm"] = dx
                row["dy_mm"] = dy
                row["e_xy_mm"] = float(np.hypot(dx, dy))

            # Pred size
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

                # size errors (mm): (cm diff) * 10
                row["err_sx_mm"] = float((row["pred_sx_cm"] - gt_side_cm) * 10.0)
                row["err_sy_mm"] = float((row["pred_sy_cm"] - gt_side_cm) * 10.0)
                row["err_sz_mm"] = float((row["pred_sz_cm"] - gt_side_cm) * 10.0)
                row["err_side_mm_med"] = float((pred_side_cm_med - gt_side_cm) * 10.0)

            rows.append(row)

    # Write CSV
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Console summary
    def _summarize(color: str):
        es = [r["e_xy_mm"] for r in rows if r["color"] == color and np.isfinite(r["e_xy_mm"])]
        dxs = [r["dx_mm"] for r in rows if r["color"] == color and np.isfinite(r["dx_mm"])]
        dys = [r["dy_mm"] for r in rows if r["color"] == color and np.isfinite(r["dy_mm"])]
        ss = [r["err_side_mm_med"] for r in rows if r["color"] == color and np.isfinite(r["err_side_mm_med"])]

        if not es:
            return

        def q(a, p):  # quantile
            return float(np.quantile(np.asarray(a, dtype=float), p))

        print(f"\n=== {color.upper()} summary (valid rows: {len(es)}) ===")
        print(f"center e_xy_mm: mean={np.mean(es):.2f}, median={np.median(es):.2f}, p90={q(es,0.90):.2f}, max={np.max(es):.2f}")
        print(f"dx_mm: mean={np.mean(dxs):+.2f}, dy_mm: mean={np.mean(dys):+.2f}")
        if ss:
            print(f"size err_side_mm_med: mean={np.mean(ss):+.2f}, median={np.median(ss):+.2f}, p90={q(ss,0.90):+.2f}, max={np.max(ss):+.2f}")
        else:
            print("size: N/A (no size fields found in JSON)")

    _summarize("yellow")
    _summarize("blue")

    # Compact table print (first 10 rows per color)
    def _print_compact(color: str, k: int = 10):
        print(f"\n--- {color.upper()} first {k} rows ---")
        print("trial  pred_x  pred_y   gt_x   gt_y   dx    dy   e_xy  side(cm,med)  side_err(mm)")
        cnt = 0
        for r in rows:
            if r["color"] != color:
                continue
            if cnt >= k:
                break
            print(
                f"{int(r['trial']):>5d} "
                f"{_safe_float(r['pred_x_mm']):>7.1f} {_safe_float(r['pred_y_mm']):>7.1f} "
                f"{_safe_float(r['gt_x_mm']):>6.1f} {_safe_float(r['gt_y_mm']):>6.1f} "
                f"{_safe_float(r['dx_mm']):>6.1f} {_safe_float(r['dy_mm']):>6.1f} "
                f"{_safe_float(r['e_xy_mm']):>6.1f} "
                f"{_safe_float(r['pred_side_cm_med']):>11.3f} "
                f"{_safe_float(r['err_side_mm_med']):>11.2f}"
            )
            cnt += 1

    _print_compact("yellow", 10)
    _print_compact("blue", 10)

    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
