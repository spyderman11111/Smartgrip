#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint


# =========================
# Core matcher
# =========================

class LightGlueMatcher:
    def __init__(
        self,
        feature_type: str = "superpoint",
        device: str = "cuda",
        descriptor_dim: int = 256,
        n_layers: int = 9,
        num_heads: int = 4,
        flash: bool = True,
        mp: bool = False,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        weights: str = None,
    ):
        self.device = device
        self.matcher = LightGlue(
            features=feature_type,
            descriptor_dim=descriptor_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            flash=flash,
            mp=mp,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
            weights=weights,
        ).to(device)

        if feature_type == "superpoint":
            self.feature_extractor = SuperPoint().to(device)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

    @staticmethod
    def _bgr_to_tensor_rgb01(img_bgr: np.ndarray, device: str) -> torch.Tensor:
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR uint8 HxWx3, got {img_bgr.shape}")
        img_rgb = img_bgr[:, :, ::-1]  # BGR->RGB
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0  # [3,H,W]
        return t.to(device)

    def extract_features_from_bgr(self, img_bgr: np.ndarray) -> dict:
        image = self._bgr_to_tensor_rgb01(img_bgr, self.device)  # [3,H,W]
        feats = self.feature_extractor.extract(image)
        h, w = img_bgr.shape[:2]
        feats["image_size"] = torch.tensor([w, h], device=self.device).unsqueeze(0)  # [1,2] (W,H)
        return feats

    def extract_features(self, image_path: str) -> dict:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return self.extract_features_from_bgr(img)

    def match_feats(self, feats0: dict, feats1: dict) -> dict:
        data = {
            "image0": {
                "keypoints": feats0["keypoints"].to(self.device),
                "descriptors": feats0["descriptors"].to(self.device),
                "image_size": feats0["image_size"].to(self.device),
            },
            "image1": {
                "keypoints": feats1["keypoints"].to(self.device),
                "descriptors": feats1["descriptors"].to(self.device),
                "image_size": feats1["image_size"].to(self.device),
            },
        }
        return self.matcher(data)

    def compile(self, mode: str = "reduce-overhead"):
        self.matcher.compile(mode=mode)


# =========================
# Preprocess: object-only crop + resize (recommended)
# =========================

def _read_bgr_uint8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _object_bbox_from_black_bg(img_bgr: np.ndarray, thr: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect non-black region as object bbox in an object-only image.
    Return (x0,y0,x1,y1) in pixel coordinates, inclusive-exclusive.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thr).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _crop_pad_resize(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_size: int = 512,
    pad_ratio: float = 0.15,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = img_bgr.shape[:2]
    bw, bh = (x1 - x0), (y1 - y0)
    if bw <= 0 or bh <= 0:
        return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # pad
    pad = int(round(max(bw, bh) * pad_ratio))
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    half = 0.5 * max(bw, bh) + pad

    nx0 = int(round(cx - half))
    ny0 = int(round(cy - half))
    nx1 = int(round(cx + half))
    ny1 = int(round(cy + half))

    # clamp and pad with black
    pad_left = max(0, -nx0)
    pad_top = max(0, -ny0)
    pad_right = max(0, nx1 - w)
    pad_bottom = max(0, ny1 - h)

    nx0 = max(0, nx0)
    ny0 = max(0, ny0)
    nx1 = min(w, nx1)
    ny1 = min(h, ny1)

    crop = img_bgr[ny0:ny1, nx0:nx1]
    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def preprocess_object_only(
    img_bgr: np.ndarray,
    auto_crop_resize: bool = True,
    out_size: int = 512,
    pad_ratio: float = 0.15,
) -> np.ndarray:
    if not auto_crop_resize:
        return img_bgr
    bbox = _object_bbox_from_black_bg(img_bgr)
    if bbox is None:
        # object missing or too dark; return original resized to stable size
        return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return _crop_pad_resize(img_bgr, bbox=bbox, out_size=out_size, pad_ratio=pad_ratio)


# =========================
# Scoring and policy
# =========================

def compute_object_match_score(
    scores: np.ndarray,
    score_th: float = 0.20,  # confident match threshold inside LG outputs
    topk: int = 10,          # top-k for quality
    k_ref: int = 20,         # saturation for coverage
) -> dict:
    if scores is None:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0, "score_th": score_th, "topk": 0, "k_ref": k_ref}

    scores = np.asarray(scores, dtype=np.float32)
    conf = scores[scores >= float(score_th)]
    K_conf = int(conf.size)

    if K_conf == 0:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0, "score_th": score_th, "topk": 0, "k_ref": k_ref}

    conf_sorted = np.sort(conf)[::-1]
    k = int(min(topk, K_conf))
    quality = float(conf_sorted[:k].mean())
    coverage = float(min(1.0, K_conf / float(max(1, k_ref))))
    score = float(np.clip(quality * coverage, 0.0, 1.0))
    return {"score": score, "quality": quality, "coverage": coverage, "K_conf": K_conf, "score_th": score_th, "topk": k, "k_ref": k_ref}


@dataclass
class FrameEval:
    path: str
    score: float
    quality: float
    coverage: float
    K_conf: int
    n_matches: int


@dataclass
class VerificationResult:
    accepted: bool
    pass_index: int                 # first index where consecutive rule is satisfied, -1 if rejected
    best_index: int                 # index of max score
    best_score: float
    streak_required: int
    pass_threshold: float
    per_frame: List[FrameEval]


def _draw_matches_image(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    kpts0_xy: np.ndarray,
    kpts1_xy: np.ndarray,
    matches_ij: np.ndarray,
    scores: np.ndarray,
    topk: int = 200,
) -> np.ndarray:
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:w0 + w1] = img1_bgr
    right_offset = np.array([w0, 0], dtype=np.float32)

    if matches_ij.size == 0:
        return canvas

    K = min(int(topk), matches_ij.shape[0])
    for idx in range(K):
        i0, i1 = matches_ij[idx]
        p0 = kpts0_xy[i0].astype(np.int32)
        p1 = (kpts1_xy[i1] + right_offset).astype(np.int32)

        cv2.line(canvas, tuple(p0), tuple(p1), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p0), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p1), 2, (0, 0, 255), -1, cv2.LINE_AA)

        s = float(scores[idx])
        mid = ((p0 + p1) * 0.5).astype(np.int32)
        cv2.putText(canvas, f"{s:.2f}", tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    return canvas


def verify_one_to_many(
    matcher: LightGlueMatcher,
    aria_path: str,
    wrist_paths: List[str],
    # preprocessing
    auto_crop_resize: bool = True,
    out_size: int = 512,
    pad_ratio: float = 0.15,
    # scoring
    score_th: float = 0.20,
    topk_quality: int = 10,
    k_ref: int = 20,
    # decision policy
    pass_threshold: float = 0.40,
    consecutive_n: int = 2,
    max_frames: Optional[int] = 16,
    # optional visualization
    save_best_viz_path: Optional[str] = None,
    viz_topk: int = 200,
) -> VerificationResult:
    if not Path(aria_path).exists():
        raise FileNotFoundError(f"aria_path not found: {aria_path}")
    if len(wrist_paths) == 0:
        raise ValueError("wrist_paths is empty")

    # limit frames
    wrist_paths = [p for p in wrist_paths if Path(p).exists()]
    if max_frames is not None:
        wrist_paths = wrist_paths[: int(max_frames)]
    if len(wrist_paths) == 0:
        raise ValueError("No valid wrist image paths after filtering")

    # --- Extract Aria features once ---
    aria_bgr = _read_bgr_uint8(aria_path)
    aria_bgr = preprocess_object_only(aria_bgr, auto_crop_resize=auto_crop_resize, out_size=out_size, pad_ratio=pad_ratio)
    feats0 = matcher.extract_features_from_bgr(aria_bgr)

    per_frame: List[FrameEval] = []
    streak = 0
    pass_index = -1

    best_score = -1.0
    best_index = -1
    best_bundle: Optional[Dict[str, Any]] = None  # store for viz

    with torch.inference_mode():
        for i, wp in enumerate(wrist_paths):
            wrist_bgr = _read_bgr_uint8(wp)
            wrist_bgr = preprocess_object_only(wrist_bgr, auto_crop_resize=auto_crop_resize, out_size=out_size, pad_ratio=pad_ratio)
            feats1 = matcher.extract_features_from_bgr(wrist_bgr)

            out = matcher.match_feats(feats0, feats1)

            matches = out["matches"][0].detach().cpu().numpy().astype(np.int64)
            scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)

            # sort desc by score
            if scores.size > 0:
                order = np.argsort(-scores)
                matches = matches[order]
                scores = scores[order]

            stats = compute_object_match_score(scores, score_th=score_th, topk=topk_quality, k_ref=k_ref)
            frame_eval = FrameEval(
                path=wp,
                score=stats["score"],
                quality=stats["quality"],
                coverage=stats["coverage"],
                K_conf=int(stats["K_conf"]),
                n_matches=int(matches.shape[0]),
            )
            per_frame.append(frame_eval)

            # best
            if frame_eval.score > best_score:
                best_score = frame_eval.score
                best_index = i
                best_bundle = {
                    "aria_bgr": aria_bgr,
                    "wrist_bgr": wrist_bgr,
                    "feats0": feats0,
                    "feats1": feats1,
                    "matches": matches,
                    "scores": scores,
                }

            # consecutive policy
            if frame_eval.score >= float(pass_threshold):
                streak += 1
                if streak >= int(consecutive_n) and pass_index < 0:
                    pass_index = i  # first frame where rule is satisfied
            else:
                streak = 0

    accepted = pass_index >= 0

    # optional: save best visualization
    if save_best_viz_path is not None and best_bundle is not None:
        kpts0 = best_bundle["feats0"]["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        kpts1 = best_bundle["feats1"]["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        matches = best_bundle["matches"]
        scores = best_bundle["scores"]
        vis = _draw_matches_image(
            best_bundle["aria_bgr"], best_bundle["wrist_bgr"],
            kpts0, kpts1, matches, scores,
            topk=viz_topk,
        )
        os.makedirs(str(Path(save_best_viz_path).parent), exist_ok=True)
        cv2.imwrite(save_best_viz_path, vis)

    return VerificationResult(
        accepted=accepted,
        pass_index=pass_index,
        best_index=best_index,
        best_score=float(best_score if best_score >= 0 else 0.0),
        streak_required=int(consecutive_n),
        pass_threshold=float(pass_threshold),
        per_frame=per_frame,
    )


# =========================
# Demo main
# =========================

def _print_summary(res: VerificationResult):
    print("\n[CrossViewVerification] summary")
    print(f"  accepted: {res.accepted}")
    print(f"  pass_threshold: {res.pass_threshold:.3f} | consecutive_n: {res.streak_required}")
    print(f"  pass_index (first satisfied): {res.pass_index}")
    print(f"  best_index: {res.best_index} | best_score: {res.best_score:.3f}\n")

    print(f"{'i':>3} | {'score':>6} | {'qual':>6} | {'cov':>6} | {'Kconf':>5} | {'K':>4} | path")
    print("-" * 120)
    for i, fr in enumerate(res.per_frame):
        print(f"{i:>3} | {fr.score:>6.3f} | {fr.quality:>6.3f} | {fr.coverage:>6.3f} | {fr.K_conf:>5d} | {fr.n_matches:>4d} | {fr.path}")
    print("")


if __name__ == "__main__":
    # ---- Inputs ----
    ARIA_PATH = "/home/MA_SmartGrip/Smartgrip/ur5_image_input/image3_rgbmask.png"
    WRIST_GLOB = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/pictures/yellow_cube_30/out_rgb_masks/*.png"

    wrist_paths = sorted(glob.glob(WRIST_GLOB))
    if not Path(ARIA_PATH).exists():
        raise FileNotFoundError(f"ARIA_PATH not found: {ARIA_PATH}")
    if len(wrist_paths) == 0:
        raise FileNotFoundError(f"No wrist images matched: {WRIST_GLOB}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)

    # ---- Verification policy ----
    res = verify_one_to_many(
        matcher=matcher,
        aria_path=ARIA_PATH,
        wrist_paths=wrist_paths,
        auto_crop_resize=True,
        out_size=512,
        pad_ratio=0.15,
        score_th=0.20,
        topk_quality=10,
        k_ref=20,
        pass_threshold=0.40,     # t in your paper
        consecutive_n=2,         # N in your paper
        max_frames=16,           # number of snapshots in the sweep
        save_best_viz_path="/home/MA_SmartGrip/Smartgrip/tmp/best_match_viz.png",
        viz_topk=200,
    )

    _print_summary(res)
    print("[CrossViewVerification] best visualization saved to:", "/home/MA_SmartGrip/Smartgrip/tmp/best_match_viz.png")
