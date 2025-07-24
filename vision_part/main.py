import os
import cv2
import random
import json
import torch
from PIL import Image
import numpy as np

from detect_with_dino import GroundingDinoPredictor
from segment_with_sam2 import SAM2ImagePredictorWrapper
from LightGlueMatcher import LightGlueMatcher


def select_frame_paths(aria_dir, ur5e_dir):
    aria_frames = sorted([os.path.join(aria_dir, f) for f in os.listdir(aria_dir) if f.endswith((".jpg", ".png"))])
    ur5e_frames = sorted([os.path.join(ur5e_dir, f) for f in os.listdir(ur5e_dir) if f.endswith((".jpg", ".png"))])

    assert len(aria_frames) > 0 and len(ur5e_frames) >= 3, "Not enough frames in directories."

    aria_path = random.choice(aria_frames)
    ur5e_paths = [ur5e_frames[0], ur5e_frames[len(ur5e_frames) // 2], ur5e_frames[-1]]

    return aria_path, ur5e_paths


def extract_features(image_path, prompt, dino, sam2, matcher, label_prefix, output_dir):
    image = Image.open(image_path).convert("RGB")
    boxes, labels = dino.predict(image=image, text_prompts=prompt, box_threshold=0.25, text_threshold=0.25)
    print(f"\n[{label_prefix}] {os.path.basename(image_path)} | Detected {len(labels)} object(s)")

    for i, (label, box) in enumerate(zip(labels, boxes)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        
        # Use full image path and bounding box in run_inference
        result = sam2.run_inference(
            image_path=image_path,
            box=(x1, y1, x2, y2),
            save_dir=output_dir
        )

        save_path = result["mask_rgba_path"]
        feats = matcher.extract_features(save_path)

        if feats["keypoints"].shape[1] > 0:
            return {
                "features": feats,
                "image_path": image_path,
                "mask_path": save_path
            }

    print(f"  - [WARN] No usable features for {label_prefix}")
    return None

def draw_matches(image1_path, kp1, image2_path, kp2, matches, save_path):
    import numpy as np  

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    for m in matches:
        pt1 = tuple(np.round(kp1[0][m[0]].cpu().numpy()).astype(int))
        pt2 = tuple(np.round(kp2[0][m[1]].cpu().numpy()).astype(int) + [w1, 0])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, pt1, pt2, color=color, thickness=1)
        cv2.circle(canvas, pt1, 3, color=color, thickness=-1)
        cv2.circle(canvas, pt2, 3, color=color, thickness=-1)

    cv2.imwrite(save_path, canvas)

def match_and_report(aria_feats, ur5e_feats, label_ur5e, matcher, output_dir):
    if aria_feats is None or ur5e_feats is None:
        print(f"[MATCH Aria vs {label_ur5e}] Skip: No features")
        return {
            "match_count": 0,
            "score_mean": 0.0,
            "aria_image": aria_feats["image_path"] if aria_feats else None,
            "ur5e_image": ur5e_feats["image_path"] if ur5e_feats else None,
        }

    feats1 = aria_feats["features"]
    feats2 = ur5e_feats["features"]

    result = matcher.match(feats1["keypoints"], feats1["descriptors"],
                           feats2["keypoints"], feats2["descriptors"],
                           feats1["image_size"], feats2["image_size"])
    matches = result["matches"][0]
    scores = result["scores"][0]

    score_mean = scores.mean().item() if matches.shape[0] > 0 else 0.0
    print(f"[MATCH Aria vs {label_ur5e}] Matches: {matches.shape[0]}, Score mean: {score_mean:.3f}")

    match_vis_path = os.path.join(output_dir, f"aria_vs_{label_ur5e.lower()}_matches.png")
    draw_matches(aria_feats["mask_path"], feats1["keypoints"],
                 ur5e_feats["mask_path"], feats2["keypoints"],
                 matches, match_vis_path)

    return {
        "match_count": matches.shape[0],
        "score_mean": round(score_mean, 6),
        "aria_image": aria_feats["image_path"],
        "ur5e_image": ur5e_feats["image_path"],
        "match_vis": match_vis_path
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    aria_dir = os.path.join(base_dir, "aria_images")
    ur5e_dir = os.path.join(base_dir, "ur5e_images_scene", "images")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    prompt = "ball"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aria_img, ur5e_imgs = select_frame_paths(aria_dir, ur5e_dir)

    dino = GroundingDinoPredictor(model_id="IDEA-Research/grounding-dino-tiny", device=device)
    sam2 = SAM2ImagePredictorWrapper(model_id="facebook/sam2.1-hiera-large", device=device)
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)

    print("[Init] Loaded GroundingDINO, SAM2, LightGlue")

    feats_aria = extract_features(aria_img, prompt, dino, sam2, matcher, "Aria", output_dir)

    results = {}
    ur5e_labels = ["First", "Middle", "Last"]
    for i, ur5e_img in enumerate(ur5e_imgs):
        feats_ur5e = extract_features(ur5e_img, prompt, dino, sam2, matcher, f"UR5e-{ur5e_labels[i]}", output_dir)
        result = match_and_report(feats_aria, feats_ur5e, f"UR5e-{ur5e_labels[i]}", matcher, output_dir)
        results[ur5e_labels[i]] = result

    save_path = os.path.join(output_dir, "match_confidence_result.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[Done] Matching results saved to: {save_path}")


if __name__ == "__main__":
    main()
