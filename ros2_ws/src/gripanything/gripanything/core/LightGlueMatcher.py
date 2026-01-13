#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

import cv2
import numpy as np


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

    def extract_features(self, image_path: str) -> dict:
        image = load_image(image_path).to(self.device)  # [3,H,W], float, 0..1
        features = self.feature_extractor.extract(image)
        features["image_size"] = torch.tensor(image.shape[-2:][::-1]).unsqueeze(0).to(self.device)  # [1,2] (W,H)
        return features

    def match(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ) -> dict:
        data = {
            "image0": {
                "keypoints": keypoints0.to(self.device),
                "descriptors": descriptors0.to(self.device),
                "image_size": image_size0.to(self.device),
            },
            "image1": {
                "keypoints": keypoints1.to(self.device),
                "descriptors": descriptors1.to(self.device),
                "image_size": image_size1.to(self.device),
            },
        }
        return self.matcher(data)

    def compile(self, mode: str = "reduce-overhead"):
        self.matcher.compile(mode=mode)


# ------------------------ 可视化工具函数 ------------------------

def _read_bgr_uint8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _resize_to_fit(img: np.ndarray, max_w: int = 1600, max_h: int = 900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # 只缩小，不放大
    if scale >= 1.0:
        return img, 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _draw_matches_window(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    kpts0_xy: np.ndarray,   # [M,2]
    kpts1_xy: np.ndarray,   # [N,2]
    matches_ij: np.ndarray, # [K,2] each row (i0, i1)
    scores: np.ndarray,     # [K]
    title: str = "LightGlue matches",
    page_size: int = 50,
    max_w: int = 1600,
    max_h: int = 900,
):
    # 拼接画布
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0:w0 + w1] = img1_bgr

    right_offset = np.array([w0, 0], dtype=np.float32)

    total = matches_ij.shape[0]
    if total == 0:
        vis_show, _ = _resize_to_fit(canvas, max_w=max_w, max_h=max_h)
        cv2.imshow(title, vis_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    page = 0
    max_page = (total - 1) // page_size

    while True:
        vis = canvas.copy()
        start = page * page_size
        end = min((page + 1) * page_size, total)

        for idx in range(start, end):
            i0, i1 = matches_ij[idx]
            p0 = kpts0_xy[i0].astype(np.int32)
            p1 = (kpts1_xy[i1] + right_offset).astype(np.int32)

            cv2.line(vis, tuple(p0), tuple(p1), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, tuple(p0), 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(p1), 2, (0, 0, 255), -1, cv2.LINE_AA)

            s = float(scores[idx])
            text = f"{s:.3f}"
            mid = ((p0 + p1) * 0.5).astype(np.int32)
            cv2.putText(vis, text, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        header = f"{title} | matches: {total} | page {page+1}/{max_page+1} | showing [{start}:{end})"
        cv2.putText(vis, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        vis_show, _ = _resize_to_fit(vis, max_w=max_w, max_h=max_h)
        cv2.imshow(title, vis_show)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("n"):
            page = min(page + 1, max_page)
        elif key == ord("p"):
            page = max(page - 1, 0)

    cv2.destroyAllWindows()


def _print_match_table(
    kpts0_xy: np.ndarray,
    kpts1_xy: np.ndarray,
    matches_ij: np.ndarray,
    scores: np.ndarray,
    topk: int = 50,
):
    total = matches_ij.shape[0]
    if total == 0:
        print("[MatchTable] No matches.")
        return

    topk = min(topk, total)
    print(f"\n[MatchTable] Top-{topk} / {total} matches (sorted by score desc)\n")
    print(f"{'rank':>4} | {'score':>7} | {'i0':>5} {'(x0,y0)':>18} | {'i1':>5} {'(x1,y1)':>18}")
    print("-" * 70)
    for r in range(topk):
        i0, i1 = matches_ij[r]
        x0, y0 = kpts0_xy[i0]
        x1, y1 = kpts1_xy[i1]
        print(
            f"{r:>4} | {scores[r]:>7.4f} | {i0:>5} ({x0:>7.1f},{y0:>7.1f}) | "
            f"{i1:>5} ({x1:>7.1f},{y1:>7.1f})"
        )
    print("")


def compute_object_match_score(
    scores: np.ndarray,
    score_th: float = 0.20,  # 只看“可信匹配”
    topk: int = 10,          # 用Top-K衡量质量
    k_ref: int = 20,         # 认为“够好”大概需要多少条可信匹配（饱和阈值）
) -> dict:
    """
    Return a [0,1] global reliability score that combines:
    - quality: mean(top-k confident scores)
    - coverage: min(1, K_conf / k_ref)

    score = quality * coverage
    """
    if scores is None:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0, "score_th": score_th, "topk": 0, "k_ref": k_ref}

    scores = np.asarray(scores, dtype=np.float32)
    conf = scores[scores >= float(score_th)]
    K_conf = int(conf.size)

    if K_conf == 0:
        return {"score": 0.0, "quality": 0.0, "coverage": 0.0, "K_conf": 0, "score_th": score_th, "topk": 0, "k_ref": k_ref}

    conf_sorted = np.sort(conf)[::-1]  # desc
    k = int(min(topk, K_conf))
    quality = float(conf_sorted[:k].mean())  # 0..1

    coverage = float(min(1.0, K_conf / float(max(1, k_ref))))  # 0..1

    score = float(np.clip(quality * coverage, 0.0, 1.0))
    return {"score": score, "quality": quality, "coverage": coverage, "K_conf": K_conf, "score_th": score_th, "topk": k, "k_ref": k_ref}



if __name__ == "__main__":
    image_path0 = "/home/MA_SmartGrip/Smartgrip/ur5_image_input/image3_rgbmask.png"
    image_path1 = "/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/pictures/yellow_cube_30/out_rgb_masks/pose_9_image_rgbmask.png"

    if not Path(image_path0).exists() or not Path(image_path1).exists():
        raise FileNotFoundError("Sample images not found. Please update the image paths.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)

    # Step 1: Extract features
    feats0 = matcher.extract_features(image_path0)
    feats1 = matcher.extract_features(image_path1)

    # Step 2: Match descriptors
    with torch.inference_mode():
        out = matcher.match(
            feats0["keypoints"], feats0["descriptors"],
            feats1["keypoints"], feats1["descriptors"],
            feats0["image_size"], feats1["image_size"]
        )

    # LightGlue 输出：matches[0] 是 (K,2) 索引对；scores[0] 是 (K,) 分数
    matches = out["matches"][0].detach().cpu().numpy().astype(np.int64)
    scores = out["scores"][0].detach().cpu().numpy().astype(np.float32)

    # 排序：按分数从高到低
    if scores.size > 0:
        order = np.argsort(-scores)
        matches = matches[order]
        scores = scores[order]

    # keypoints： [1,M,2] -> [M,2]
    kpts0 = feats0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    kpts1 = feats1["keypoints"][0].detach().cpu().numpy().astype(np.float32)

    print("Matched pairs:", matches.shape[0])
    if matches.shape[0] > 0:
        print("Top-5 matches (i0,i1):\n", matches[:5])
        print("Top-5 scores:\n", scores[:5])

    # 1) 终端打印更“具体”的分数表
    _print_match_table(kpts0, kpts1, matches, scores, topk=50)

    # 2) 输出全局 0-1 匹配可靠度（推荐用于“这次匹配准不准”的衡量）
    stats = compute_object_match_score(scores, score_th=0.20, topk=10, k_ref=20)
    obj_score = stats["score"]
    good = obj_score >= 0.6

    print(f"[ObjectMatchScore] K={matches.shape[0]} | K_conf(>={stats['score_th']})={stats['K_conf']} | "
        f"quality(top{stats['topk']})={stats['quality']:.3f} | coverage={stats['coverage']:.3f}")
    print(f"[ObjectMatchScore] score = {obj_score:.3f}  ->  {'GOOD' if good else 'BAD'} (threshold=0.6)\n")

    # 3) 可视化窗口
    img0 = _read_bgr_uint8(image_path0)
    img1 = _read_bgr_uint8(image_path1)

    TOPK_VIS = 200
    SCORE_TH = 0.00
    keep = np.where(scores >= SCORE_TH)[0]
    keep = keep[:min(TOPK_VIS, keep.shape[0])]
    matches_vis = matches[keep]
    scores_vis = scores[keep]

    _draw_matches_window(
        img0, img1,
        kpts0, kpts1,
        matches_vis, scores_vis,
        title="LightGlue SuperPoint matches (n/p to flip, q to quit)",
        page_size=60,
        max_w=1600,
        max_h=900,
    )
