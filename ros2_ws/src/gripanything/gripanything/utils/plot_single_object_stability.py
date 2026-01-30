#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_single_object_stability.py

Read a LaTeX longtable (center accuracy table) and plot Gaussian+hist distributions
for dx/dy (cm). Split by color and generate 4 figures:
  - blue  dx
  - blue  dy
  - yellow dx
  - yellow dy

Input table format (each row):
  Trial & Color & (gt_x, gt_y) & (pred_x, pred_y) & $dx$ & $dy$ & e_xy \\

We ONLY use dx/dy numbers already in the table (cm).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parsing
# -----------------------------
ROW_RE = re.compile(
    r"""
    ^\s*
    (?P<trial>\d+)\s*&\s*
    (?P<color>Blue|Yellow)\s*&\s*
    .*?&\s*.*?&\s*
    \$\s*(?P<dx>[+-]?\d+(?:\.\d+)?)\s*\$\s*&\s*
    \$\s*(?P<dy>[+-]?\d+(?:\.\d+)?)\s*\$\s*&\s*
    (?P<exy>[+-]?\d+(?:\.\d+)?)\s*
    \\\\
    """,
    re.VERBOSE,
)

# A more tolerant pattern that can still catch rows even if e_xy is missing/--,
# but dx/dy exist. (We will try strict first, then fallback.)
ROW_RE_FALLBACK = re.compile(
    r"""
    ^\s*
    (?P<trial>\d+)\s*&\s*
    (?P<color>Blue|Yellow)\s*&\s*
    .*?&\s*.*?&\s*
    \$\s*(?P<dx>[+-]?\d+(?:\.\d+)?)\s*\$\s*&\s*
    \$\s*(?P<dy>[+-]?\d+(?:\.\d+)?)\s*\$\s*&
    """,
    re.VERBOSE,
)


@dataclass
class Row:
    trial: int
    color: str  # "blue" or "yellow"
    dx_cm: float
    dy_cm: float
    e_xy_cm: float


def parse_center_table(tex_path: str) -> List[Row]:
    rows: List[Row] = []
    with open(tex_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            m = ROW_RE.match(s)
            if m:
                trial = int(m.group("trial"))
                color = m.group("color").lower()
                dx = float(m.group("dx"))
                dy = float(m.group("dy"))
                exy = float(m.group("exy"))
                rows.append(Row(trial, color, dx, dy, exy))
                continue

            # fallback: keep dx/dy if present
            mf = ROW_RE_FALLBACK.match(s)
            if mf:
                trial = int(mf.group("trial"))
                color = mf.group("color").lower()
                dx = float(mf.group("dx"))
                dy = float(mf.group("dy"))
                rows.append(Row(trial, color, dx, dy, float("nan")))
                continue

    return rows


# -----------------------------
# Plot helpers
# -----------------------------
def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def bins_with_width(data_cm: np.ndarray, bin_width_cm: float = 1.0, margin_cm: float = 1.0) -> np.ndarray:
    data_cm = np.asarray(data_cm, dtype=float)
    data_cm = data_cm[np.isfinite(data_cm)]
    if data_cm.size == 0:
        return np.array([-1.0, 1.0], dtype=float)

    xmin = np.floor(data_cm.min() - margin_cm)
    xmax = np.ceil(data_cm.max() + margin_cm)
    # ensure at least 2 edges
    if xmax <= xmin:
        xmax = xmin + bin_width_cm

    edges = np.arange(xmin, xmax + bin_width_cm * 1.0001, bin_width_cm)
    if edges.size < 2:
        edges = np.array([xmin, xmin + bin_width_cm], dtype=float)
    return edges


def set_xticks_1cm(ax, data_cm: np.ndarray, margin_cm: float = 1.0) -> None:
    data_cm = np.asarray(data_cm, dtype=float)
    data_cm = data_cm[np.isfinite(data_cm)]
    if data_cm.size == 0:
        return

    xmin = np.floor(data_cm.min() - margin_cm)
    xmax = np.ceil(data_cm.max() + margin_cm)
    if xmax <= xmin:
        xmax = xmin + 1.0

    ticks = np.arange(xmin, xmax + 1e-9, 1.0)  # 1 cm per tick
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.grid(True, axis="x", alpha=0.25)


def summarize(arr: np.ndarray) -> Tuple[int, float, float, float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    n = int(arr.size)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    p90 = float(np.quantile(arr, 0.90))
    return n, med, mean, std, p90


def plot_dist(
    data_cm: np.ndarray,
    title: str,
    out_path: str,
    bin_width_cm: float = 1.0,
    margin_cm: float = 1.0,
) -> Dict[str, float]:
    data_cm = np.asarray(data_cm, dtype=float)
    data_cm = data_cm[np.isfinite(data_cm)]

    n, med, mean, std, p90 = summarize(data_cm)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    edges = bins_with_width(data_cm, bin_width_cm=bin_width_cm, margin_cm=margin_cm)

    # Histogram with visible gaps between bars
    ax.hist(
        data_cm,
        bins=edges,
        density=True,
        rwidth=0.88,   # leaves gap between bars
        alpha=0.75,
    )

    # Gaussian curve (fit by mean/std)
    xgrid = np.linspace(edges[0], edges[-1], 800)
    pdf = gaussian_pdf(xgrid, mean, std)
    ax.plot(xgrid, pdf, linewidth=2.0, label=f"Gaussian fit (μ={mean:.3f}, σ={std:.3f})")

    # Vertical markers
    ax.axvline(med, linestyle="--", linewidth=2.0, label=f"median={med:+.3f} cm")
    ax.axvline(mean, linestyle=":", linewidth=2.0, label=f"mean={mean:+.3f} cm")

    set_xticks_1cm(ax, data_cm, margin_cm=margin_cm)

    ax.set_title(title)
    ax.set_xlabel("error (cm)")
    ax.set_ylabel("density")

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # Suggested bias (to cancel median)
    bias_m = -med / 100.0  # cm -> m
    return {
        "N": n,
        "median_cm": med,
        "mean_cm": mean,
        "std_cm": std,
        "p90_cm": p90,
        "bias_m_from_median": bias_m,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tex",
        type=str,
        default="/home/MA_SmartGrip/Smartgrip/table_center_accuracy_50.tex",
        help="Path to the LaTeX table file (center accuracy).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for plots (default: current directory).",
    )
    ap.add_argument(
        "--bin_width_cm",
        type=float,
        default=1.0,
        help="Histogram bin width in cm (default: 1.0).",
    )
    ap.add_argument(
        "--margin_cm",
        type=float,
        default=1.0,
        help="Extra margin around data range (cm) for xlim/ticks (default: 1.0).",
    )
    args = ap.parse_args()

    tex_path = os.path.abspath(args.tex)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = parse_center_table(tex_path)
    if not rows:
        raise RuntimeError(f"No valid rows parsed from: {tex_path}")

    # Split by color
    dx_blue = np.array([r.dx_cm for r in rows if r.color == "blue"], dtype=float)
    dy_blue = np.array([r.dy_cm for r in rows if r.color == "blue"], dtype=float)
    dx_yellow = np.array([r.dx_cm for r in rows if r.color == "yellow"], dtype=float)
    dy_yellow = np.array([r.dy_cm for r in rows if r.color == "yellow"], dtype=float)

    # Global summary (all 50)
    dx_all = np.array([r.dx_cm for r in rows], dtype=float)
    dy_all = np.array([r.dy_cm for r in rows], dtype=float)

    N_all, dx_med, dx_mean, dx_std, _ = summarize(dx_all)
    _, dy_med, dy_mean, dy_std, _ = summarize(dy_all)

    print(f"N={N_all}")
    print(f"dx_median_cm={dx_med:+.4f}")
    print(f"dy_median_cm={dy_med:+.4f}")
    print(f"dx_mean_cm={dx_mean:+.4f}")
    print(f"dy_mean_cm={dy_mean:+.4f}")
    print(f"dx_std_cm={dx_std:.4f}")
    print(f"dy_std_cm={dy_std:.4f}")
    print(f"bias_x_m_from_median={(-dx_med/100.0):+.6f}")
    print(f"bias_y_m_from_median={(-dy_med/100.0):+.6f}")

    # 4 plots
    res_bdx = plot_dist(
        dx_blue,
        title="Blue: Δx distribution (cm)",
        out_path=os.path.join(out_dir, "blue_dx_gaussian.png"),
        bin_width_cm=float(args.bin_width_cm),
        margin_cm=float(args.margin_cm),
    )
    res_bdy = plot_dist(
        dy_blue,
        title="Blue: Δy distribution (cm)",
        out_path=os.path.join(out_dir, "blue_dy_gaussian.png"),
        bin_width_cm=float(args.bin_width_cm),
        margin_cm=float(args.margin_cm),
    )
    res_ydx = plot_dist(
        dx_yellow,
        title="Yellow: Δx distribution (cm)",
        out_path=os.path.join(out_dir, "yellow_dx_gaussian.png"),
        bin_width_cm=float(args.bin_width_cm),
        margin_cm=float(args.margin_cm),
    )
    res_ydy = plot_dist(
        dy_yellow,
        title="Yellow: Δy distribution (cm)",
        out_path=os.path.join(out_dir, "yellow_dy_gaussian.png"),
        bin_width_cm=float(args.bin_width_cm),
        margin_cm=float(args.margin_cm),
    )

    # Per-color printout (useful for per-color bias)
    print("\n--- Per-color summary (bias from median) ---")
    print(f"BLUE  dx: N={res_bdx['N']}, median={res_bdx['median_cm']:+.3f} cm, std={res_bdx['std_cm']:.3f} cm, bias_x={res_bdx['bias_m_from_median']:+.6f} m")
    print(f"BLUE  dy: N={res_bdy['N']}, median={res_bdy['median_cm']:+.3f} cm, std={res_bdy['std_cm']:.3f} cm, bias_y={res_bdy['bias_m_from_median']:+.6f} m")
    print(f"YELLOW dx: N={res_ydx['N']}, median={res_ydx['median_cm']:+.3f} cm, std={res_ydx['std_cm']:.3f} cm, bias_x={res_ydx['bias_m_from_median']:+.6f} m")
    print(f"YELLOW dy: N={res_ydy['N']}, median={res_ydy['median_cm']:+.3f} cm, std={res_ydy['std_cm']:.3f} cm, bias_y={res_ydy['bias_m_from_median']:+.6f} m")

    print("\nSaved figures:")
    print(os.path.join(out_dir, "blue_dx_gaussian.png"))
    print(os.path.join(out_dir, "blue_dy_gaussian.png"))
    print(os.path.join(out_dir, "yellow_dx_gaussian.png"))
    print(os.path.join(out_dir, "yellow_dy_gaussian.png"))


if __name__ == "__main__":
    main()
