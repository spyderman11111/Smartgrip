import os
import cv2
import torch
import numpy as np
from PIL import Image
import json

from detect_with_dino import GroundingDinoPredictor
from extract_frames import VideoFrameExtractor
from segment_with_sam2 import SAM2ImagePredictorWrapper
from LightGlueMatcher import LightGlueMatcher


def extract_one_frame(video_path: str, output_dir: str) -> str:
    """
    Extract one frame from the given video and save to output_dir.
    Returns the saved image path.
    """
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith((".jpg", ".png", ".json")):
            os.remove(os.path.join(output_dir, f))
    extractor = VideoFrameExtractor(video_source=video_path, frame_interval=1, output_dir=output_dir)
    extractor.extract_frames(max_frames=1)
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return os.path.join(output_dir, frame_files[0]) if frame_files else None


def prepare_frame_data(video_path: str, output_dir: str, source: str) -> dict:
    """
    Extract one frame and return a structured frame data dictionary.
    """
    frame_path = extract_one_frame(video_path, output_dir)
    return {
        "video_path": video_path,
        "output_dir": output_dir,
        "source": source,
        "frame_path": frame_path
    }


def detect_and_segment_frame(frame_data: dict, prompt: str,
                             dino: GroundingDinoPredictor,
                             sam2: SAM2ImagePredictorWrapper,
                             matcher: LightGlueMatcher):
    """
    Perform detection and segmentation on a single frame.
    Returns features, frame index, and timestamp.
    """
    output_dir = frame_data["output_dir"]
    source = frame_data["source"]
    video_path = frame_data["video_path"]
    frame_path = frame_data["frame_path"]
    image_name = os.path.basename(frame_path)
    image = Image.open(frame_path).convert("RGB")

    boxes, labels = dino.predict(image=image, text_prompts=prompt,
                                 box_threshold=0.25, text_threshold=0.25)
    print(f"\n[Frame] ({source}) {image_name} | Detected {len(labels)} object(s):")
    first_feats = None

    for i, (label, box) in enumerate(zip(labels, boxes)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        crop = image.crop((x1, y1, x2, y2))
        base_name = os.path.splitext(image_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}_box{i}_mask_color.png")
        sam2.run_on_crop(crop, save_path)
        print(f"  - {label}: x1={x1}, y1={y1}, x2={x2}, y2={y2} -> mask saved: {save_path}")
        feats = matcher.extract_features(save_path)
        if feats["keypoints"].shape[1] == 0:
            print(f"    [WARN] No keypoints found in crop {i}")
            continue
        if first_feats is None:
            first_feats = feats

    frame_index = 0
    try:
        frame_index = int(image_name.split('_')[1].split('.')[0])
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = frame_index / fps if fps and fps > 0 else 0.0
    cap.release()

    return first_feats, frame_index, timestamp


def main():
    # Config
    video1_path = "vision_part/test_video/test_ball.mp4"
    video2_path = "vision_part/test_video/test_ball2.mp4"
    output1 = "aira"
    output2 = "ur5e"
    prompt = "ball"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    dino = GroundingDinoPredictor(model_id="IDEA-Research/grounding-dino-tiny", device=device)
    sam2 = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large", device=device,
        mask_threshold=0.3, max_hole_area=100.0, max_sprinkle_area=50.0,
        multimask_output=False, return_logits=False
    )
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)
    print("Loaded GroundingDINO, SAM2, and LightGlueMatcher")

    # Process both cameras
    frame_data1 = prepare_frame_data(video1_path, output1, "aria")
    frame_data2 = prepare_frame_data(video2_path, output2, "ur5e")
    feats1, idx1, ts1 = detect_and_segment_frame(frame_data1, prompt, dino, sam2, matcher)
    feats2, idx2, ts2 = detect_and_segment_frame(frame_data2, prompt, dino, sam2, matcher)

    # Feature matching
    match_count = 0
    avg_score = 0.0
    if feats1 and feats2:
        result = matcher.match(feats1["keypoints"], feats1["descriptors"],
                               feats2["keypoints"], feats2["descriptors"],
                               feats1["image_size"], feats2["image_size"])
        matches = result["matches"][0]
        scores = result["scores"][0]
        match_count = matches.shape[0]
        avg_score = scores.mean().item() if match_count > 0 else 0.0
        print(f"\n[MATCH] Aira <-> UR5e: Matches: {match_count}, Avg confidence: {avg_score:.3f}")
    else:
        print("\n[MATCH] No features to match (one or both frames had no keypoints).")

    result_data = {
        "aira_frame_index": idx1,
        "ur5e_frame_index": idx2,
        "aira_timestamp": ts1,
        "ur5e_timestamp": ts2,
        "match_count": match_count,
        "avg_score": round(avg_score, 6)
    }
    for output_dir in [output1, output2]:
        json_path = os.path.join(output_dir, "match_result.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=4)
    print(f"Matching results saved to '{output1}/' and '{output2}/'")


main()
