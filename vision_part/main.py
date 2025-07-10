import os
import cv2
import torch
import numpy as np
from PIL import Image
from detect_with_dino import GroundingDinoPredictor
from extract_frames import VideoFrameExtractor
from segment_with_sam2 import SAM2ImagePredictorWrapper
from LightGlueMatcher import LightGlueMatcher
import json

def extract_one_frame(video_source: str, output_dir: str):
    """
    Extract exactly one frame from the video_source and save it to output_dir.
    Clears any existing images in output_dir before saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Remove old image and mask files in the directory
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".json"):
            os.remove(os.path.join(output_dir, f))
    # Initialize extractor (frame_interval=1 to consider every frame, but we'll stop after one)
    extractor = VideoFrameExtractor(video_source=video_source, frame_interval=1, output_dir=output_dir)
    extractor.extract_frames(max_frames=1)  # Extract only the first frame
    print(f"Extracted frame to: {output_dir}")

def detect_and_segment_frame(output_dir: str, video_path: str, prompt: str,
                             dino: GroundingDinoPredictor, sam2: SAM2ImagePredictorWrapper,
                             matcher: LightGlueMatcher):
    """
    Perform detection and segmentation on the single frame in output_dir.
    Returns the extracted features for the first detected object, along with frame index and timestamp.
    """
    # There should be one extracted frame image in the directory
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    if len(frame_files) == 0:
        print(f"[WARN] No frame found in {output_dir}")
        return None, None, None  # No frame to process
    fname = frame_files[0]
    fpath = os.path.join(output_dir, fname)
    # Open the image
    image = Image.open(fpath).convert("RGB")
    # Run GroundingDINO detection
    boxes, labels = dino.predict(
        image=image,
        text_prompts=prompt,
        box_threshold=0.25,
        text_threshold=0.25
    )
    # Determine source name for logging (Aira or UR5e)
    src_name = "Aira" if output_dir.lower() == "aira" else ("UR5e" if output_dir.lower() == "ur5e" else output_dir)
    print(f"\n[Frame] ({src_name}) {fname} | Detected {len(labels)} object(s):")
    # Initialize variable to store features of the first detected object
    first_feats = None
    # Loop through each detected object
    for i, (label, box) in enumerate(zip(labels, boxes)):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        # Crop the detected region from the image
        crop = image.crop((x1, y1, x2, y2))
        # Save path for the segmentation overlay image
        base_name = os.path.splitext(fname)[0]  # e.g., "frame_00000"
        save_path = os.path.join(output_dir, f"{base_name}_box{i}_mask_color.png")
        # Run segmentation on the cropped region and save the overlay
        sam2.run_on_crop(crop, save_path)
        print(f"  - {label}: x1={x1}, y1={y1}, x2={x2}, y2={y2} -> mask saved: {save_path}")
        # Extract features from the saved mask overlay image
        feats = matcher.extract_features(save_path)
        if feats["keypoints"].shape[1] == 0:
            # No keypoints detected in this crop
            print(f"    [WARN] No keypoints found in crop {i}")
            continue
        # If this is the first detected object, store its features for matching
        if first_feats is None:
            first_feats = feats
        # (If there are multiple detections, we only use the first for cross-frame matching)
    # Compute frame index and timestamp for this frame
    frame_index = None
    timestamp = None
    try:
        # Parse frame index from filename (assuming format frame_XXXXX.jpg)
        frame_index = int(fname.split('_')[1].split('.')[0])
    except Exception:
        frame_index = 0  # Fallback to 0 if parsing fails
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        timestamp = frame_index / fps
    else:
        timestamp = 0.0
    cap.release()
    return first_feats, frame_index, timestamp

if __name__ == "__main__":
    # Paths to the two video sources
    video1_path = "vision_part/test_video/test_ball.mp4"
    video2_path = "vision_part/test_video/test_ball2.mp4"  
    # Output directories for Aira and UR5e
    output_dir1 = "aira"
    output_dir2 = "ur5e"
    # Text prompt for object detection
    prompt = "ball"
    # Step 1: Extract one frame from each video source
    extract_one_frame(video1_path, output_dir1)
    extract_one_frame(video2_path, output_dir2)
    # Step 2: Initialize models for detection, segmentation, and matching
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_dino = GroundingDinoPredictor(model_id="IDEA-Research/grounding-dino-tiny", device=device)
    sam2 = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device=device,
        mask_threshold=0.3,
        max_hole_area=100.0,
        max_sprinkle_area=50.0,
        multimask_output=False,
        return_logits=False,
    )
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)
    print("Loaded GroundingDINO, SAM2, and LightGlueMatcher")
    # Step 3 & 4: Detection and Segmentation for each frame
    feats1, frame_idx1, timestamp1 = detect_and_segment_frame(output_dir1, video1_path, prompt,
                                                              grounding_dino, sam2, matcher)
    feats2, frame_idx2, timestamp2 = detect_and_segment_frame(output_dir2, video2_path, prompt,
                                                              grounding_dino, sam2, matcher)
    # Step 5: Feature Matching between the two frames
    match_count = 0
    avg_score = 0.0
    if feats1 is None or feats2 is None:
        print("\n[MATCH] No features to match (one or both frames had no keypoints).")
    else:
        # Perform matching using LightGlue
        result = matcher.match(
            feats1["keypoints"], feats1["descriptors"],
            feats2["keypoints"], feats2["descriptors"],
            feats1["image_size"], feats2["image_size"]
        )
        matches = result["matches"][0]
        scores = result["scores"][0]
        match_count = matches.shape[0]
        avg_score = scores.mean().item() if match_count > 0 else 0.0
        print(f"\n[MATCH] Aira <-> UR5e: Matches: {match_count}, Avg confidence: {avg_score:.3f}")
    # Step 6: Save matching results to JSON file (in both output directories)
    result_data = {
        "aira_frame_index": int(frame_idx1) if frame_idx1 is not None else None,
        "ur5e_frame_index": int(frame_idx2) if frame_idx2 is not None else None,
        "aira_timestamp": float(timestamp1) if timestamp1 is not None else None,
        "ur5e_timestamp": float(timestamp2) if timestamp2 is not None else None,
        "match_count": int(match_count),
        "avg_score": float(round(avg_score, 6))  # round to 6 decimal places for clarity
    }
    # Save the JSON result in each output directory
    json_path1 = os.path.join(output_dir1, "match_result.json")
    json_path2 = os.path.join(output_dir2, "match_result.json")
    with open(json_path1, 'w') as f:
        json.dump(result_data, f, indent=4)
    with open(json_path2, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Matching results saved to JSON file in '{output_dir1}/' and '{output_dir2}/'")
