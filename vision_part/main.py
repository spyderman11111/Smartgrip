import os
import cv2
import random
import json
import torch
from PIL import Image

from detect_with_dino import GroundingDinoPredictor
from segment_with_sam2 import SAM2ImagePredictorWrapper
from LightGlueMatcher import LightGlueMatcher


# ========== Step 1: 抽取图像路径 ==========
def select_frame_paths(aria_dir, ur5e_dir):
    aria_frames = sorted([os.path.join(aria_dir, f) for f in os.listdir(aria_dir) if f.endswith((".jpg", ".png"))])
    ur5e_frames = sorted([os.path.join(ur5e_dir, f) for f in os.listdir(ur5e_dir) if f.endswith((".jpg", ".png"))])

    assert len(aria_frames) > 0 and len(ur5e_frames) >= 3, "Not enough frames in directories."

    # 随机选一张 aria 图像
    aria_path = random.choice(aria_frames)

    # ur5e: 第一帧、中间帧、最后一帧
    ur5e_paths = [
        ur5e_frames[0],
        ur5e_frames[len(ur5e_frames) // 2],
        ur5e_frames[-1]
    ]

    return aria_path, ur5e_paths


# ========== Step 2: DINO + SAM2 特征提取 ==========
def extract_features(image_path, prompt, dino, sam2, matcher, label_prefix):
    image = Image.open(image_path).convert("RGB")
    boxes, labels = dino.predict(image=image, text_prompts=prompt,
                                 box_threshold=0.25, text_threshold=0.25)
    print(f"\n[{label_prefix}] {os.path.basename(image_path)} | Detected {len(labels)} object(s)")

    for i, (label, box) in enumerate(zip(labels, boxes)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        crop = image.crop((x1, y1, x2, y2))
        save_path = image_path.replace(".jpg", f"_mask{i}.png")
        sam2.run_on_crop(crop, save_path)
        feats = matcher.extract_features(save_path)

        if feats["keypoints"].shape[1] > 0:
            return feats  # 返回第一个有效 mask 特征
    print(f"  - [WARN] No usable features for {label_prefix}")
    return None


# ========== Step 3 & 4: 特征匹配并输出 ==========
def match_and_report(feats1, feats2, label1, label2, matcher):
    if feats1 is None or feats2 is None:
        print(f"[MATCH {label1} vs {label2}] Skip: No features")
        return None

    result = matcher.match(feats1["keypoints"], feats1["descriptors"],
                           feats2["keypoints"], feats2["descriptors"],
                           feats1["image_size"], feats2["image_size"])
    matches = result["matches"][0]
    scores = result["scores"][0]

    match_count = matches.shape[0]
    avg_score = scores.mean().item() if match_count > 0 else 0.0
    print(f"[MATCH {label1} vs {label2}] Matches: {match_count}, Avg score: {avg_score:.3f}")

    return {
        "match_count": match_count,
        "avg_score": round(avg_score, 6)
    }


def main():
    # ========== Step 1: 设置路径并选择图像 ==========
    aria_dir = "/home/sz/Smartgrip/vision_part/aria_images"
    ur5e_dir = "/home/sz/Smartgrip/vision_part/ur5e_images"
    prompt = "ball"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aria_img, ur5e_imgs = select_frame_paths(aria_dir, ur5e_dir)

    # ========== Step 2: 初始化模型 ==========
    dino = GroundingDinoPredictor(model_id="IDEA-Research/grounding-dino-tiny", device=device)
    sam2 = SAM2ImagePredictorWrapper(model_id="facebook/sam2.1-hiera-large", device=device)
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)

    print("[Init] Loaded GroundingDINO, SAM2, LightGlue")

    # ========== Step 3: 提取特征 ==========
    feats_aria = extract_features(aria_img, prompt, dino, sam2, matcher, "Aria")
    results = {}
    ur5e_labels = ["First", "Middle", "Last"]

    for i, ur5e_img in enumerate(ur5e_imgs):
        feats_ur5e = extract_features(ur5e_img, prompt, dino, sam2, matcher, f"UR5e-{ur5e_labels[i]}")
        result = match_and_report(feats_aria, feats_ur5e, "Aria", f"UR5e-{ur5e_labels[i]}", matcher)
        results[ur5e_labels[i]] = result

    # ========== Step 4: 保存匹配结果 ==========
    save_path = os.path.join(aria_dir, "match_confidence_result.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[Done] Matching results saved to: {save_path}")


if __name__ == "__main__":
    main()
