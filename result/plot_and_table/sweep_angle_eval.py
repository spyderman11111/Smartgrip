#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_angle_eval.py

Compute sweep-angle tables (center + size) from offline outputs.

Key requirement (per user):
- Shape error e_s must be computed using the MEAN of the provided GT sizes.
  Given GT sizes: e.g. "34,12 33,81 33,73" (interpreted as mm by default)
  -> convert to cm -> gt_mean_cm = mean([sx,sy,sz] in cm)
  -> e_x = |pred_sx_cm - gt_mean_cm|, similarly y/z
  -> e_s = sqrt(e_x^2 + e_y^2 + e_z^2)

GT center is optional:
- If gt centers are not provided, c^{gt}, Δx, Δy, e_xy will be left as ---.

Outputs:
- CSV summary (default: ./sweep_angle_eval.csv)
- LaTeX tables printed to stdout (two tables similar to your paper)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))


def _as_vec3(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x):
        return np.array([float(x[0]), float(x[1]), float(x[2])], dtype=float)
    return None


def _safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_object_json(output_dir: str) -> str:
    """
    Try common locations under an 'output' directory.
    """
    od = os.path.abspath(output_dir)
    cand = [
        os.path.join(od, "offline_output", "object_in_base_link.json"),
        os.path.join(od, "object_in_base_link.json"),
        os.path.join(os.path.dirname(od), "offline_output", "object_in_base_link.json"),
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

    # assume first 4 are bottom, last 4 are top
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


def _count_images(output_dir: str) -> int:
    """
    Best-effort image count in common folders under output.
    """
    od = os.path.abspath(output_dir)
    cand_dirs = [
        os.path.join(od, "images"),
        os.path.join(od, "offline_output", "images"),
        os.path.join(od, "snapshots"),
        os.path.join(od, "offline_output", "snapshots"),
        od,
    ]
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

    best = 0
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        n = 0
        for fn in os.listdir(d):
            if fn.lower().endswith(exts):
                n += 1
        best = max(best, n)
    return int(best)


def _infer_angle_from_path(output_dir: str) -> Optional[int]:
    """
    Infer angle from .../<angle>_red_solo/output
    """
    od = os.path.abspath(output_dir)
    parent = os.path.basename(os.path.dirname(od))  # e.g. 60_red_solo
    m = re.match(r"^\s*(\d+)\s*_", parent)
    if m:
        return int(m.group(1))
    return None


def _discover_outputs(result_root: str, suffix: str) -> List[Tuple[int, str]]:
    """
    Discover folders: <result_root>/<angle>_<suffix>/output
    Return list of (angle_deg, output_dir) sorted by angle.
    """
    rr = os.path.abspath(result_root)
    if not os.path.isdir(rr):
        return []

    out: List[Tuple[int, str]] = []
    for name in os.listdir(rr):
        d = os.path.join(rr, name)
        if not os.path.isdir(d):
            continue
        if not name.endswith(suffix):
            continue
        m = re.match(r"^\s*(\d+)\s*_", name)
        if not m:
            continue
        ang = int(m.group(1))
        od = os.path.join(d, "output")
        if os.path.isdir(od):
            out.append((ang, od))

    out.sort(key=lambda x: x[0])
    return out


# ---------------------------
# GT parsing: mean size
# ---------------------------
def _parse_three_numbers(s: str) -> np.ndarray:
    """
    Accept:
      "34,12 33,81 33,73"
      "34.12 33.81 33.73"
      "34,12,33,81,33,73"  (less common but supported)
    Strategy:
      - replace comma with dot
      - split by whitespace or comma
    """
    s = s.strip().replace(",", ".")
    parts = re.split(r"[\s,]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        raise ValueError("gt_size must have 3 numbers, e.g. '34,12 33,81 33,73'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def _gt_mean_cm_from_user(gt_size_str: str, unit: str) -> float:
    """
    Convert user GT sizes to cm, then take mean.
    unit: "mm" or "cm"
    """
    v = _parse_three_numbers(gt_size_str)
    if unit.lower() == "mm":
        v_cm = v / 10.0
    elif unit.lower() == "cm":
        v_cm = v
    else:
        raise ValueError("gt_unit must be 'mm' or 'cm'")
    return float(np.mean(v_cm))


# ---------------------------
# Optional GT center schema
# ---------------------------
"""
If you want GT centers later, provide a JSON like:

{
  "centers_m": {
    "60":  [0.461275, 0.308421, 0.116730],
    "90":  [0.506068, 0.305479, 0.049194]
  }
}

Only centers_m is used. Size GT is taken from --gt_size and averaged.
"""


def _get_gt_center_m(gt: Optional[Dict[str, Any]], angle_deg: int) -> Optional[np.ndarray]:
    if gt is None:
        return None
    centers = gt.get("centers_m", None)
    if not isinstance(centers, dict):
        return None
    for k in (str(angle_deg), angle_deg):
        if k in centers:
            return _as_vec3(centers[k])
    return None


# ---------------------------
# Data row
# ---------------------------
@dataclass
class SweepRow:
    angle_deg: int
    n_images: int

    gt_center_m: Optional[np.ndarray]
    pred_center_m: Optional[np.ndarray]

    dx_cm: Optional[float]
    dy_cm: Optional[float]
    e_xy_cm: Optional[float]

    pred_sx_cm: Optional[float]
    pred_sy_cm: Optional[float]
    pred_sz_cm: Optional[float]

    ex_cm: Optional[float]
    ey_cm: Optional[float]
    ez_cm: Optional[float]
    e_s_cm: Optional[float]

    gt_mean_cm: float

    json_path: str
    align_method: str
    scale_s: float


def _l2(a: float, b: float, c: float) -> float:
    return float(math.sqrt(a * a + b * b + c * c))


def compute_one(output_dir: str, angle_deg: int, gt_center_json: Optional[Dict[str, Any]], gt_mean_cm: float) -> SweepRow:
    obj_json = _find_object_json(output_dir)
    data = _safe_load_json(obj_json) or {}

    align_method = _read_align_method(data)
    scale_s = _read_scale_s(data)

    pred_center = _read_center_m(data)

    extent_m = _try_read_extent_m(data)
    if extent_m is None:
        c8 = _try_read_corners8_base_m(data)
        if c8 is not None:
            extent_m = _extent_from_corners8_m(c8)

    pred_sx_cm = pred_sy_cm = pred_sz_cm = None
    if extent_m is not None:
        ext_cm = extent_m * 100.0
        pred_sx_cm = float(ext_cm[0])
        pred_sy_cm = float(ext_cm[1])
        pred_sz_cm = float(ext_cm[2])

    n_images = _count_images(output_dir)

    gt_center = _get_gt_center_m(gt_center_json, angle_deg)

    # center errors (cm)
    dx_cm = dy_cm = e_xy_cm = None
    if gt_center is not None and pred_center is not None:
        dx_cm = float((pred_center[0] - gt_center[0]) * 100.0)
        dy_cm = float((pred_center[1] - gt_center[1]) * 100.0)
        e_xy_cm = float(math.hypot(dx_cm, dy_cm))

    # size errors (cm) using gt MEAN
    ex_cm = ey_cm = ez_cm = e_s_cm = None
    if pred_sx_cm is not None:
        ex_cm = float(abs(pred_sx_cm - gt_mean_cm))
        ey_cm = float(abs(pred_sy_cm - gt_mean_cm))
        ez_cm = float(abs(pred_sz_cm - gt_mean_cm))
        e_s_cm = _l2(ex_cm, ey_cm, ez_cm)

    return SweepRow(
        angle_deg=int(angle_deg),
        n_images=int(n_images),

        gt_center_m=gt_center,
        pred_center_m=pred_center,

        dx_cm=dx_cm,
        dy_cm=dy_cm,
        e_xy_cm=e_xy_cm,

        pred_sx_cm=pred_sx_cm,
        pred_sy_cm=pred_sy_cm,
        pred_sz_cm=pred_sz_cm,

        ex_cm=ex_cm,
        ey_cm=ey_cm,
        ez_cm=ez_cm,
        e_s_cm=e_s_cm,

        gt_mean_cm=float(gt_mean_cm),

        json_path=obj_json,
        align_method=str(align_method),
        scale_s=float(scale_s),
    )


# ---------------------------
# LaTeX rendering
# ---------------------------
def _fmt_vec3_m(v: Optional[np.ndarray]) -> str:
    if v is None:
        return r"\shortstack{(---,\\---,\\---)}"
    return (
        r"\shortstack{("
        + f"{v[0]:.6f},"
        + r"\\"
        + f"{v[1]:.6f},"
        + r"\\"
        + f"{v[2]:.6f}"
        + r")}"
    )


def _fmt_num(x: Optional[float], nd: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return r"---"
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def print_latex_tables(rows: List[SweepRow], caption_prefix: str = "red target") -> None:
    # Table 1: center + e_s
    print("% --------------------- sweep-angle: center + e_s ---------------------")
    print(r"\begin{table}[!t]")
    print(r"  \centering")
    print(r"  \renewcommand{\arraystretch}{1.10}")
    print(r"  \setlength{\tabcolsep}{4pt}")
    print(r"  \scriptsize")
    print(r"  \begin{tabular}{lccccc c c}")
    print(r"    \toprule")
    print(r"    Sweep angle & \#images &")
    print(r"    $c^{gt}$ in $\mathrm{B}$ (m) &")
    print(r"    $c$ in $\mathrm{B}$ (m) &")
    print(r"    $\Delta x$ (cm) & $\Delta y$ (cm) &")
    print(r"    $e_{xy}$ (cm) & $e_s$ (cm) \\")
    print(r"    \midrule")
    for r0 in rows:
        print(
            f"    ${r0.angle_deg}^\\circ$  & {r0.n_images} &\n"
            f"    {_fmt_vec3_m(r0.gt_center_m)} &\n"
            f"    {_fmt_vec3_m(r0.pred_center_m)} &\n"
            f"    {_fmt_num(r0.dx_cm, 2, signed=True)} & {_fmt_num(r0.dy_cm, 2, signed=True)} & "
            f"{_fmt_num(r0.e_xy_cm, 2)} & {_fmt_num(r0.e_s_cm, 2)} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(
        rf"\caption{{Center accuracy ($e_{{xy}}$) and compact shape error ($e_s$) vs sweep angle on the {caption_prefix}. "
        rf"Here, $e_s$ uses GT mean size $\bar{{s}}^{{gt}}={rows[0].gt_mean_cm:.3f}\,\mathrm{{cm}}$ if available.}}"
    )
    print(r"\end{table}")
    print()

    # Table 2: size + e_s
    print("% --------------------- sweep-angle: size + e_s ---------------------")
    print(r"\begin{table}[!t]")
    print(r"  \centering")
    print(r"  \renewcommand{\arraystretch}{1.15}")
    print(r"  \setlength{\tabcolsep}{6pt}")
    print(r"  \begin{tabular}{lccccccc c}")
    print(r"    \toprule")
    print(r"    Sweep angle & \#images &")
    print(r"    $s_x$ (cm) & $s_y$ (cm) & $s_z$ (cm) &")
    print(r"    $e_x$ (cm) & $e_y$ (cm) & $e_z$ (cm) & $e_s$ (cm) \\")
    print(r"    \midrule")
    for r0 in rows:
        print(
            f"    ${r0.angle_deg}^\\circ$ & {r0.n_images} & "
            f"{_fmt_num(r0.pred_sx_cm, 2)} & {_fmt_num(r0.pred_sy_cm, 2)} & {_fmt_num(r0.pred_sz_cm, 2)} & "
            f"{_fmt_num(r0.ex_cm, 2)} & {_fmt_num(r0.ey_cm, 2)} & {_fmt_num(r0.ez_cm, 2)} & {_fmt_num(r0.e_s_cm, 2)} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(
        rf"\caption{{Size estimates and per-axis errors vs sweep angle on the {caption_prefix}; "
        rf"$e_s$ summarizes the three axis errors and uses the GT mean size $\bar{{s}}^{{gt}}={rows[0].gt_mean_cm:.3f}\,\mathrm{{cm}}$.}}"
    )
    print(r"\end{table}")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output_dir",
        type=str,
        default="/home/MA_SmartGrip/Smartgrip/result/60_red_solo/output",
        help="A single output directory to evaluate (e.g., .../60_red_solo/output).",
    )

    ap.add_argument(
        "--result_root",
        type=str,
        default="",
        help="If set, auto-discover <angle>_<suffix>/output under this folder. Overrides --output_dir.",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="red_solo",
        help="Folder suffix used in discovery, e.g., 'red_solo' matches '60_red_solo'.",
    )
    ap.add_argument(
        "--sweeps",
        type=str,
        default="",
        help="Comma-separated angles to include (e.g., '60,90,120,150,180'). If empty, include all discovered.",
    )

    # optional GT centers (NOT required)
    ap.add_argument(
        "--gt_center_json",
        type=str,
        default="",
        help="Optional GT centers JSON path with schema: {\"centers_m\": {\"60\": [x,y,z], ...}}",
    )

    # GT size: user-provided 3 numbers, then take MEAN
    ap.add_argument(
        "--gt_size",
        type=str,
        default="34,12 33,81 33,73",
        help="GT object size (3 numbers). Default: '34,12 33,81 33,73'. Comma or dot ok.",
    )
    ap.add_argument(
        "--gt_unit",
        type=str,
        default="mm",
        choices=["mm", "cm"],
        help="Unit of --gt_size. Default mm (recommended for '34,12' style).",
    )

    ap.add_argument("--out_csv", type=str, default="sweep_angle_eval.csv", help="Output CSV path.")
    args = ap.parse_args()

    gt_mean_cm = _gt_mean_cm_from_user(args.gt_size, args.gt_unit)

    gt_centers: Optional[Dict[str, Any]] = None
    if args.gt_center_json.strip():
        gt_centers = _safe_load_json(args.gt_center_json.strip())
        if gt_centers is None:
            raise RuntimeError(f"Failed to read gt_center_json: {args.gt_center_json}")

    # decide which outputs to evaluate
    outputs: List[Tuple[int, str]] = []
    if args.result_root.strip():
        outputs = _discover_outputs(args.result_root.strip(), args.suffix.strip())
        if not outputs:
            raise RuntimeError(f"No outputs discovered under result_root={args.result_root} with suffix={args.suffix}")

        if args.sweeps.strip():
            wanted = set(int(x.strip()) for x in args.sweeps.split(",") if x.strip())
            outputs = [(a, od) for (a, od) in outputs if a in wanted]
            outputs.sort(key=lambda x: x[0])
            if not outputs:
                raise RuntimeError("After applying --sweeps filter, no outputs remain.")
    else:
        od = os.path.abspath(args.output_dir)
        ang = _infer_angle_from_path(od)
        if ang is None:
            ang = 0
        outputs = [(ang, od)]

    rows: List[SweepRow] = []
    for ang, od in outputs:
        rows.append(compute_one(od, ang, gt_centers, gt_mean_cm))

    # write CSV
    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "angle_deg", "n_images",
        "gt_mean_cm",
        "gt_cx_m", "gt_cy_m", "gt_cz_m",
        "pred_cx_m", "pred_cy_m", "pred_cz_m",
        "dx_cm", "dy_cm", "e_xy_cm",
        "pred_sx_cm", "pred_sy_cm", "pred_sz_cm",
        "e_x_cm", "e_y_cm", "e_z_cm", "e_s_cm",
        "align_method", "scale_s",
        "json_path",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r0 in rows:
            gt_c = r0.gt_center_m
            pr_c = r0.pred_center_m
            w.writerow({
                "angle_deg": r0.angle_deg,
                "n_images": r0.n_images,
                "gt_mean_cm": r0.gt_mean_cm,
                "gt_cx_m": (None if gt_c is None else float(gt_c[0])),
                "gt_cy_m": (None if gt_c is None else float(gt_c[1])),
                "gt_cz_m": (None if gt_c is None else float(gt_c[2])),
                "pred_cx_m": (None if pr_c is None else float(pr_c[0])),
                "pred_cy_m": (None if pr_c is None else float(pr_c[1])),
                "pred_cz_m": (None if pr_c is None else float(pr_c[2])),
                "dx_cm": r0.dx_cm,
                "dy_cm": r0.dy_cm,
                "e_xy_cm": r0.e_xy_cm,
                "pred_sx_cm": r0.pred_sx_cm,
                "pred_sy_cm": r0.pred_sy_cm,
                "pred_sz_cm": r0.pred_sz_cm,
                "e_x_cm": r0.ex_cm,
                "e_y_cm": r0.ey_cm,
                "e_z_cm": r0.ez_cm,
                "e_s_cm": r0.e_s_cm,
                "align_method": r0.align_method,
                "scale_s": r0.scale_s,
                "json_path": r0.json_path,
            })

    print(f"[OK] GT mean size used for e_s: {gt_mean_cm:.6f} cm")
    print(f"[OK] Saved CSV: {out_csv}\n")

    print_latex_tables(rows, caption_prefix="red target")


if __name__ == "__main__":
    main()
