#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_single_object_stability.py

Goal
- Parse two LaTeX longtables from a .tex file:
  1) Center table: extract e_xy (cm)
  2) Size table  : extract e_s  (cm)
- Plot trend curves to inspect stability (drift/outliers).

Inputs
- A .tex file containing BOTH longtables (center + size).
  By default: /home/MA_SmartGrip/Smartgrip/table_center_accuracy_50.tex

Outputs (saved to --out_dir, default ".")
- trend_exy_all.png          : e_xy across 50 trials (blue then yellow)
- trend_es_all.png           : e_s  across 50 trials (blue then yellow)
- trend_exy_by_color.png     : e_xy split by color
- trend_es_by_color.png      : e_s  split by color
- stability_summary.csv      : mean/median/std/p90/max, per metric & per color

Notes
- This script ONLY uses values already present in the LaTeX tables.
- No "2mm" correction or any other external rule is applied.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Parsing
# -----------------------------
@dataclass(frozen=True)
class Key:
    color: str   # "Blue" or "Yellow"
    trial: int   # 1..25


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _find_longtable_blocks(tex: str) -> List[str]:
    # non-greedy match between \begin{longtable} and \end{longtable}
    pat = re.compile(r"\\begin\{longtable\}.*?\\end\{longtable\}", re.DOTALL)
    return pat.findall(tex)


def _parse_center_table(block: str) -> Dict[Key, float]:
    """
    Center table row example:
    1 & Blue & (-0.383120, -0.366400) & (-0.374784, -0.377678) & $+0.83$ & $-1.13$ & 1.40 \\
    Extract: trial, color, e_xy(cm)=1.40
    """
    out: Dict[Key, float] = {}

    # match data row lines; ignore headers/footers automatically by requiring "& Blue/Yellow &" pattern
    row_pat = re.compile(
        r"^\s*(\d+)\s*&\s*(Blue|Yellow)\s*&.*?&.*?&\s*\$[^\$]*\$\s*&\s*\$[^\$]*\$\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*\\\\",
        re.MULTILINE,
    )

    for m in row_pat.finditer(block):
        trial = int(m.group(1))
        color = m.group(2)
        e_xy = float(m.group(3))
        out[Key(color=color, trial=trial)] = e_xy

    return out


def _parse_size_table(block: str) -> Dict[Key, float]:
    """
    Size table row example:
    1 & Blue & 2.34 & 2.39 & 2.07 & 0.17 & 0.12 & 0.44 & 0.17 \\
    Extract: trial, color, e_s(cm)=last column
    """
    out: Dict[Key, float] = {}

    # We capture 9 columns total; we only need last one.
    # Some lines may contain "--" in missing cases; those are skipped.
    row_pat = re.compile(
        r"^\s*(\d+)\s*&\s*(Blue|Yellow)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*\\\\",
        re.MULTILINE,
    )

    for m in row_pat.finditer(block):
        trial = int(m.group(1))
        color = m.group(2)
        e_s = float(m.group(9))
        out[Key(color=color, trial=trial)] = e_s

    return out


def _pick_tables(tex: str) -> Tuple[Dict[Key, float], Dict[Key, float]]:
    blocks = _find_longtable_blocks(tex)
    if len(blocks) < 2:
        raise RuntimeError(
            "Cannot find 2 longtable blocks in the .tex file. "
            "Make sure it contains BOTH center and size longtables."
        )

    # Heuristic: center table contains "e_{xy}" in caption; size table contains "e_s"
    center_block: Optional[str] = None
    size_block: Optional[str] = None

    for b in blocks:
        if "e_{xy}" in b or "e_{xy}" in b.replace(" ", ""):
            center_block = b
        if "e_s" in b or "e_s" in b.replace(" ", ""):
            size_block = b

    # fallback: assume first is center, second is size
    if center_block is None:
        center_block = blocks[0]
    if size_block is None:
        size_block = blocks[1]

    exy = _parse_center_table(center_block)
    es = _parse_size_table(size_block)
    return exy, es


# -----------------------------
# Stats + plotting helpers
# -----------------------------
def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _save_summary_csv(
    out_path: str,
    exy_map: Dict[Key, float],
    es_map: Dict[Key, float],
) -> None:
    rows = []

    def collect(metric_name: str, values: List[float], color: str):
        s = _stats(np.array(values, dtype=float))
        rows.append({
            "metric": metric_name,
            "color": color,
            "n": s["n"],
            "mean_cm": s["mean"],
            "median_cm": s["median"],
            "std_cm": s["std"],
            "p90_cm": s["p90"],
            "max_cm": s["max"],
        })

    for metric_name, mp in [("e_xy", exy_map), ("e_s", es_map)]:
        # all
        vals_all = [v for v in mp.values() if np.isfinite(v)]
        collect(metric_name, vals_all, "All")

        # by color
        for c in ["Blue", "Yellow"]:
            vals = [v for k, v in mp.items() if k.color == c and np.isfinite(v)]
            collect(metric_name, vals, c)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _global_index(key: Key) -> int:
    # blue 1..25 => 1..25, yellow 1..25 => 26..50
    if key.color == "Blue":
        return key.trial
    return 25 + key.trial


def _plot_trend_all(
    metric_name: str,
    mp: Dict[Key, float],
    out_path: str,
) -> None:
    # prepare arrays ordered by global index
    pts = []
    for k, v in mp.items():
        if not np.isfinite(v):
            continue
        pts.append((_global_index(k), k.color, k.trial, float(v)))
    pts.sort(key=lambda x: x[0])

    xs = [p[0] for p in pts]
    ys = [p[3] for p in pts]

    fig = plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o", linewidth=1.5)

    # color boundary line (between 25 and 26)
    plt.axvline(25.5, linewidth=1.0, linestyle="--")
    plt.text(12.5, np.nanmax(ys) if ys else 0, "Blue", ha="center", va="bottom")
    plt.text(38.0, np.nanmax(ys) if ys else 0, "Yellow", ha="center", va="bottom")

    # reference lines (median + mean)
    arr = np.array(ys, dtype=float)
    med = float(np.median(arr)) if arr.size else float("nan")
    mean = float(np.mean(arr)) if arr.size else float("nan")
    plt.axhline(med, linewidth=1.0, linestyle="--")
    plt.axhline(mean, linewidth=1.0, linestyle=":")

    plt.grid(True, alpha=0.3)
    plt.xlabel("Trial index (Blue 1–25, Yellow 26–50)")
    plt.ylabel(f"{metric_name} (cm)")
    plt.title(f"Trend of {metric_name} over 50 trials (mean/median as references)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_trend_by_color(
    metric_name: str,
    mp: Dict[Key, float],
    out_path: str,
) -> None:
    # Blue
    blue = sorted([(k.trial, v) for k, v in mp.items() if k.color == "Blue" and np.isfinite(v)], key=lambda x: x[0])
    yellow = sorted([(k.trial, v) for k, v in mp.items() if k.color == "Yellow" and np.isfinite(v)], key=lambda x: x[0])

    fig = plt.figure(figsize=(10, 4))
    if blue:
        xb, yb = zip(*blue)
        plt.plot(xb, yb, marker="o", linewidth=1.5, label="Blue")
    if yellow:
        xy, yy = zip(*yellow)
        plt.plot(xy, yy, marker="o", linewidth=1.5, label="Yellow")

    # reference lines per overall median
    all_vals = np.array([v for v in mp.values() if np.isfinite(v)], dtype=float)
    if all_vals.size:
        plt.axhline(float(np.median(all_vals)), linewidth=1.0, linestyle="--", label="Median (all)")

    plt.grid(True, alpha=0.3)
    plt.xlabel("Trial (1–25)")
    plt.ylabel(f"{metric_name} (cm)")
    plt.title(f"Trend of {metric_name} by color")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tex",
        type=str,
        default="/home/MA_SmartGrip/Smartgrip/table_center_accuracy_50.tex",
        help="Path to the .tex file that contains BOTH longtables (center + size).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory (images + CSV).",
    )
    args = ap.parse_args()

    tex_path = os.path.abspath(args.tex)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tex = _read_text(tex_path)
    exy_map, es_map = _pick_tables(tex)

    # quick sanity check counts
    exy_n = len(exy_map)
    es_n = len(es_map)
    print(f"[parse] e_xy rows: {exy_n}, e_s rows: {es_n}")
    if exy_n < 50:
        print("[warn] e_xy parsed rows < 50. Check center table formatting.")
    if es_n < 50:
        print("[warn] e_s parsed rows < 50. Check size table formatting.")

    # save summary
    summary_csv = os.path.join(out_dir, "stability_summary.csv")
    _save_summary_csv(summary_csv, exy_map, es_map)
    print(f"[ok] Saved summary: {summary_csv}")

    # plots
    p1 = os.path.join(out_dir, "trend_exy_all.png")
    p2 = os.path.join(out_dir, "trend_es_all.png")
    p3 = os.path.join(out_dir, "trend_exy_by_color.png")
    p4 = os.path.join(out_dir, "trend_es_by_color.png")

    _plot_trend_all("e_xy", exy_map, p1)
    _plot_trend_all("e_s", es_map, p2)
    _plot_trend_by_color("e_xy", exy_map, p3)
    _plot_trend_by_color("e_s", es_map, p4)

    print("[ok] Saved plots:")
    print("  " + p1)
    print("  " + p2)
    print("  " + p3)
    print("  " + p4)


if __name__ == "__main__":
    main()
