#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ur5_dino_sam2_mask.py

Purpose
- Run GroundingDINO detection on a UR5 wrist image (or any RGB image) with a text prompt
- Use the selected DINO box as a SAM2 box prompt to segment the object
- Export artifacts that are convenient for downstream cross-view matching:
  - Binary mask PNG (ur5_mask.png)
  - Object-only RGB image (ur5_rgbmask.png): foreground kept, background set to black
  - JSON manifest (manifest.json): paths + box + scores + status + prompt
  - (Optional) copy the latest Aria artifacts into the same folder for a consistent snapshot

Design
- Import-friendly: no heavy work at import time
- Easy to call from a main pipeline: use MaskPipeline.process() and read the returned dict
- Defaults follow your project layout (can be overridden via CLI or environment variables)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

from detect_with_dino import GroundingDinoPredictor
from segment_with_sam2 import SAM2ImagePredictorWrapper


# -----------------------------------------------------------------------------
# Project defaults (override via env or CLI)
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "SMARTGRIP_OUTPUT_ROOT",
    "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output",
)).expanduser()

DEFAULT_ARIA_DIR = Path(os.environ.get(
    "SMARTGRIP_ARIA_DIR",
    str(DEFAULT_OUTPUT_ROOT / "ariaimage"),
)).expanduser()

DEFAULT_UR5_IMAGE = Path(os.environ.get(
    "SMARTGRIP_UR5_IMAGE",
    str(DEFAULT_OUTPUT_ROOT / "ur5image" / "pose_1_image.png"),
)).expanduser()

DEFAULT_RESULT_FOLDER_NAME = os.environ.get("SMARTGRIP_MATCH_FOLDER", "matching_inputs")

DEFAULT_DEVICE = os.environ.get("SMARTGRIP_DEVICE", "cuda")
DEFAULT_DINO_ID = os.environ.get("SMARTGRIP_DINO_ID", "IDEA-Research/grounding-dino-tiny")
DEFAULT_SAM2_ID = os.environ.get("SMARTGRIP_SAM2_ID", "facebook/sam2.1-hiera-large")

DEFAULT_BOX_THRESHOLD = float(os.environ.get("SMARTGRIP_BOX_THRESHOLD", "0.25"))
DEFAULT_TEXT_THRESHOLD = float(os.environ.get("SMARTGRIP_TEXT_THRESHOLD", "0.25"))

DEFAULT_MASK_THRESHOLD = float(os.environ.get("SMARTGRIP_MASK_THRESHOLD", "0.30"))
DEFAULT_MAX_HOLE_AREA = float(os.environ.get("SMARTGRIP_MAX_HOLE_AREA", "100.0"))
DEFAULT_MAX_SPRINKLE_AREA = float(os.environ.get("SMARTGRIP_MAX_SPRINKLE_AREA", "50.0"))


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _ensure_file(path: Path, what: str) -> Path:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def _ensure_dir(path: Path, what: str) -> Path:
    p = path.expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def clip_xyxy(box_xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    """Clip XYXY box to image bounds and ensure positive area."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(h - 1.0, y1 + 1.0)
    return int(x1), int(y1), int(x2), int(y2)


def build_object_only_rgb(image_rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Keep original RGB inside mask; set everything else to black."""
    out = np.zeros_like(image_rgb, dtype=np.uint8)
    if mask_bool is not None and mask_bool.any():
        out[mask_bool] = image_rgb[mask_bool]
    return out


def save_mask_png(mask_bool: np.ndarray, path: Path) -> None:
    """Save binary mask as 0/255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(path), m)


def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    """Save RGB image to disk using OpenCV (expects RGB input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def safe_remove(path: Optional[str]) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def find_latest_file(dir_path: Path, patterns: List[str]) -> Optional[Path]:
    """
    Find the latest file by mtime in dir_path that matches any suffix pattern.
    Example patterns: ["_cutout_whitebg.png", "_mask_bin.png"]
    """
    if not dir_path.is_dir():
        return None
    candidates: List[Path] = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if any(name.endswith(sfx) for sfx in patterns):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_if_exists(src: Optional[Path], dst: Path) -> Optional[Path]:
    if src is None or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return dst


# -----------------------------------------------------------------------------
# Config + Pipeline
# -----------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Inputs
    prompt: str = "yellow cube ."
    aria_dir: Path = DEFAULT_ARIA_DIR
    ur5_image: Path = DEFAULT_UR5_IMAGE

    # Output
    output_root: Path = DEFAULT_OUTPUT_ROOT
    result_folder_name: str = DEFAULT_RESULT_FOLDER_NAME  # created under output_root
    run_subdir: str = ""  # empty => auto timestamp folder

    # Models
    device: str = DEFAULT_DEVICE
    dino_id: str = DEFAULT_DINO_ID
    sam2_id: str = DEFAULT_SAM2_ID

    # Thresholds
    box_threshold: float = DEFAULT_BOX_THRESHOLD
    text_threshold: float = DEFAULT_TEXT_THRESHOLD

    mask_threshold: float = DEFAULT_MASK_THRESHOLD
    max_hole_area: float = DEFAULT_MAX_HOLE_AREA
    max_sprinkle_area: float = DEFAULT_MAX_SPRINKLE_AREA

    # Behavior
    select_method: str = "best_score"  # "best_score" | "largest"
    save_black_if_empty: bool = False
    delete_sam2_extras: bool = True
    snapshot_aria_artifacts: bool = True  # copy latest aria artifacts into result folder


class MaskPipeline:
    """
    A lightweight callable pipeline:
      - loads DINO + SAM2 once
      - process() returns a dict (status + paths + metadata)
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = cfg.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.dino = GroundingDinoPredictor(model_id=cfg.dino_id, device=self.device)
        self.sam2 = SAM2ImagePredictorWrapper(
            model_id=cfg.sam2_id,
            device=self.device,
            mask_threshold=cfg.mask_threshold,
            max_hole_area=cfg.max_hole_area,
            max_sprinkle_area=cfg.max_sprinkle_area,
            multimask_output=False,
            return_logits=False,
        )

    def _make_run_dir(self) -> Path:
        base = self.cfg.output_root / self.cfg.result_folder_name
        sub = self.cfg.run_subdir.strip() if self.cfg.run_subdir.strip() else _timestamp()
        run_dir = base / sub
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def process(self) -> Dict:
        """
        Process one UR5 image and (optionally) snapshot the latest Aria artifacts.
        Returns a JSON-serializable dict.
        """
        run_dir = self._make_run_dir()

        ur5_path = _ensure_file(self.cfg.ur5_image, "UR5 image")
        image_pil = Image.open(str(ur5_path)).convert("RGB")
        w, h = image_pil.size
        image_rgb = np.array(image_pil)

        # 1) DINO detection
        boxes, labels, scores = self.dino.predict(
            image=image_pil,
            text_prompts=self.cfg.prompt,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
        )

        if boxes is None or len(boxes) == 0:
            result = {
                "status": "no_detection",
                "prompt": self.cfg.prompt,
                "ur5_image": str(ur5_path),
                "output_dir": str(run_dir),
            }
            if self.cfg.save_black_if_empty:
                black = np.zeros_like(image_rgb, dtype=np.uint8)
                rgbmask_path = run_dir / "ur5_rgbmask.png"
                mask_path = run_dir / "ur5_mask.png"
                save_rgb_png(black, rgbmask_path)
                save_mask_png(np.zeros((h, w), dtype=bool), mask_path)
                result.update({
                    "status": "empty_black",
                    "ur5_rgbmask": str(rgbmask_path),
                    "ur5_mask": str(mask_path),
                })
            self._maybe_snapshot_aria(run_dir, result)
            self._save_manifest(run_dir, result)
            return result

        boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
        scores_np = None
        if scores is not None:
            scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

        # 2) Select one box
        if scores_np is not None and self.cfg.select_method == "best_score":
            sel = int(np.argmax(scores_np))
        else:
            areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
            sel = int(np.argmax(areas))

        x1, y1, x2, y2 = clip_xyxy(boxes_np[sel], w, h)

        # 3) SAM2 segmentation
        sam2_out = self.sam2.run_inference(
            image_path=str(ur5_path),
            box=(x1, y1, x2, y2),
            save_dir=str(run_dir),
        )
        mask = sam2_out.get("mask_array", None)

        # Remove extra SAM2 files if requested
        if self.cfg.delete_sam2_extras:
            safe_remove(sam2_out.get("mask_gray_path"))
            safe_remove(sam2_out.get("mask_overlay_path"))
            safe_remove(sam2_out.get("mask_rgba_path"))

        result: Dict = {
            "status": "ok",
            "prompt": self.cfg.prompt,
            "device": self.device,
            "ur5_image": str(ur5_path),
            "output_dir": str(run_dir),
            "dino": {
                "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "label": str(labels[sel]) if labels is not None and len(labels) > sel else None,
                "score": float(scores_np[sel]) if scores_np is not None else None,
                "box_threshold": float(self.cfg.box_threshold),
                "text_threshold": float(self.cfg.text_threshold),
                "select_method": self.cfg.select_method,
            },
            "sam2": {
                "mask_threshold": float(self.cfg.mask_threshold),
                "max_hole_area": float(self.cfg.max_hole_area),
                "max_sprinkle_area": float(self.cfg.max_sprinkle_area),
            },
        }

        if mask is None or not np.any(mask):
            result["status"] = "mask_empty"
            if self.cfg.save_black_if_empty:
                black = np.zeros_like(image_rgb, dtype=np.uint8)
                rgbmask_path = run_dir / "ur5_rgbmask.png"
                mask_path = run_dir / "ur5_mask.png"
                save_rgb_png(black, rgbmask_path)
                save_mask_png(np.zeros((h, w), dtype=bool), mask_path)
                result.update({
                    "status": "empty_black",
                    "ur5_rgbmask": str(rgbmask_path),
                    "ur5_mask": str(mask_path),
                })
            self._maybe_snapshot_aria(run_dir, result)
            self._save_manifest(run_dir, result)
            return result

        mask_bool = mask.astype(bool)
        rgbmask = build_object_only_rgb(image_rgb, mask_bool)

        ur5_mask_path = run_dir / "ur5_mask.png"
        ur5_rgbmask_path = run_dir / "ur5_rgbmask.png"
        save_mask_png(mask_bool, ur5_mask_path)
        save_rgb_png(rgbmask, ur5_rgbmask_path)

        result.update({
            "ur5_mask": str(ur5_mask_path),
            "ur5_rgbmask": str(ur5_rgbmask_path),
        })

        self._maybe_snapshot_aria(run_dir, result)
        self._save_manifest(run_dir, result)
        return result

    def _maybe_snapshot_aria(self, run_dir: Path, result: Dict) -> None:
        """
        Copy the latest aria artifacts into run_dir (optional).
        This makes downstream matching deterministic (UR5 + Aria snapshot live together).
        """
        if not self.cfg.snapshot_aria_artifacts:
            result["aria"] = {"dir": str(self.cfg.aria_dir)}
            return

        aria_dir = self.cfg.aria_dir
        latest_cutout = find_latest_file(aria_dir, ["_cutout_whitebg.png"])
        latest_mask = find_latest_file(aria_dir, ["_mask_bin.png"])
        latest_rgb = find_latest_file(aria_dir, ["_rgb_rot.png"])  # if you save it

        aria_info = {"dir": str(aria_dir)}
        aria_info["latest_cutout"] = str(latest_cutout) if latest_cutout else None
        aria_info["latest_mask"] = str(latest_mask) if latest_mask else None
        aria_info["latest_rgb"] = str(latest_rgb) if latest_rgb else None

        # Copy into run_dir
        copied_cutout = copy_if_exists(latest_cutout, run_dir / "aria_cutout_whitebg.png")
        copied_mask = copy_if_exists(latest_mask, run_dir / "aria_mask_bin.png")
        copied_rgb = copy_if_exists(latest_rgb, run_dir / "aria_rgb_rot.png")

        aria_info["snapshot_cutout"] = str(copied_cutout) if copied_cutout else None
        aria_info["snapshot_mask"] = str(copied_mask) if copied_mask else None
        aria_info["snapshot_rgb"] = str(copied_rgb) if copied_rgb else None

        result["aria"] = aria_info

    def _save_manifest(self, run_dir: Path, result: Dict) -> None:
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["manifest"] = str(manifest_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UR5 GroundingDINO -> SAM2 mask export for matching inputs.")
    p.add_argument("--prompt", type=str, default="yellow cube .", help="Text prompt for GroundingDINO.")
    p.add_argument("--ur5-image", type=str, default=str(DEFAULT_UR5_IMAGE), help="UR5 image path.")
    p.add_argument("--aria-dir", type=str, default=str(DEFAULT_ARIA_DIR), help="Aria output directory (from gaze script).")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Project output root.")
    p.add_argument("--result-folder-name", type=str, default=DEFAULT_RESULT_FOLDER_NAME,
                   help="Subfolder under output-root to store matching inputs.")
    p.add_argument("--run-subdir", type=str, default="", help="Custom run subdir name. Empty => timestamp.")

    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Inference device.")
    p.add_argument("--dino-id", type=str, default=DEFAULT_DINO_ID, help="GroundingDINO model id.")
    p.add_argument("--sam2-id", type=str, default=DEFAULT_SAM2_ID, help="SAM2 model id.")

    p.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    p.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)

    p.add_argument("--mask-threshold", type=float, default=DEFAULT_MASK_THRESHOLD)
    p.add_argument("--max-hole-area", type=float, default=DEFAULT_MAX_HOLE_AREA)
    p.add_argument("--max-sprinkle-area", type=float, default=DEFAULT_MAX_SPRINKLE_AREA)

    p.add_argument("--select-method", type=str, default="best_score", choices=["best_score", "largest"])
    p.add_argument("--save-black-if-empty", action="store_true")
    p.add_argument("--keep-sam2-extras", action="store_true", help="Do not delete SAM2 overlay/rgba artifacts.")
    p.add_argument("--no-aria-snapshot", action="store_true", help="Do not copy aria artifacts into run folder.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        prompt=args.prompt,
        aria_dir=Path(args.aria_dir).expanduser(),
        ur5_image=Path(args.ur5_image).expanduser(),
        output_root=Path(args.output_root).expanduser(),
        result_folder_name=args.result_folder_name,
        run_subdir=args.run_subdir,
        device=args.device,
        dino_id=args.dino_id,
        sam2_id=args.sam2_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        mask_threshold=args.mask_threshold,
        max_hole_area=args.max_hole_area,
        max_sprinkle_area=args.max_sprinkle_area,
        select_method=args.select_method,
        save_black_if_empty=bool(args.save_black_if_empty),
        delete_sam2_extras=(not bool(args.keep_sam2_extras)),
        snapshot_aria_artifacts=(not bool(args.no_aria_snapshot)),
    )

    # Ensure output root exists
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print(f"[Init] device={cfg.device} | dino={cfg.dino_id} | sam2={cfg.sam2_id}")
    pipe = MaskPipeline(cfg)
    result = pipe.process()

    print(f"[Done] status={result.get('status')}")
    print(f"       output_dir={result.get('output_dir')}")
    print(f"       manifest={result.get('manifest')}")


if __name__ == "__main__":
    main()
