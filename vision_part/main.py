import os
import cv2
import torch
import numpy as np
from PIL import Image
from detect_with_dino import GroundingDinoPredictor
from extract_frames import VideoFrameExtractor
from segment_with_sam2 import SAM2ImagePredictorWrapper
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def extract_frames(
    video_source: str,  # Path to video file or camera ID
    output_dir: str,    # Directory to save extracted frames
    frame_interval: int,  # Interval in frames to sample
    max_frames: int      # Max number of frames to extract
):
    """
    Step 1: Extract frames from a video source.
    """
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".jpg") or f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    extractor = VideoFrameExtractor(
        video_source=video_source,
        frame_interval=frame_interval,
        output_dir=output_dir
    )
    extractor.extract_frames(max_frames=max_frames)
    print(f"Extracted frames to: {output_dir}")


def initialize_models():
    """
    Step 2: Initialize detection, segmentation, and matching models.

    Returns:
        grounding_dino: GroundingDinoPredictor object for zero-shot detection
        sam2: SAM2ImagePredictorWrapper object for segmentation
        extractor: SuperPoint feature extractor
        matcher: LightGlue feature matcher
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grounding_dino = GroundingDinoPredictor(
        model_id="IDEA-Research/grounding-dino-tiny",
        device=device
    )
    sam2 = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device=device,
        mask_threshold=0.3,
        max_hole_area=100.0,
        max_sprinkle_area=50.0,
        multimask_output=False,
        return_logits=False,
    )
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    print(f"Loaded GroundingDINO, SAM2, SuperPoint, and LightGlue")
    return grounding_dino, sam2, extractor, matcher


def run_detection_and_segmentation(
    output_dir: str,  # Directory containing extracted frames
    prompt: str,      # Text prompt for object detection
    dino,             # GroundingDinoPredictor
    sam2,             # SAM2ImagePredictorWrapper
    extractor,        # SuperPoint extractor
    matcher           # LightGlue matcher
):
    """
    Step 3 & 4: Detection → Segmentation → Feature extraction → Matching.
    """
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    prev_feats = None  # To store features of previous crop for matching

    for fname in frame_files:
        fpath = os.path.join(output_dir, fname)
        image = Image.open(fpath).convert("RGB")

        boxes, labels = dino.predict(
            image=image,
            text_prompts=prompt,
            box_threshold=0.25,
            text_threshold=0.25
        )
        print(f"\n[Frame] {fname} | Detected {len(labels)} object(s):")

        for i, (label, box) in enumerate(zip(labels, boxes)):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            crop = image.crop((x1, y1, x2, y2))

            save_path = os.path.join(output_dir, f"{fname[:-4]}_box{i}_mask_color.png")
            sam2.run_on_crop(crop, save_path)
            print(f"  - {label}: x1={x1}, y1={y1}, x2={x2}, y2={y2} → mask saved: {save_path}")

            crop_tensor = load_image(save_path).to(extractor.device)  # [3, H, W], normalized
            feats = extractor.extract(crop_tensor)

            if feats['keypoints'].shape[0] == 0:
                print(f"    [WARN] No keypoints found in crop {i}")
                continue

            if prev_feats is not None:
                match_result = matcher({"image0": prev_feats, "image1": feats})
                feats0, feats1, matches01 = [rbd(x) for x in [prev_feats, feats, match_result]]

                matches = matches01['matches']
                num_matches = matches.shape[0]
                avg_score = matches01['scores'].mean().item() if num_matches > 0 else 0.0
                print(f"    [MATCH] Matches: {num_matches}, Avg confidence: {avg_score:.3f}")
            else:
                print(f"    [MATCH] First crop — no reference to match.")

            prev_feats = feats


def extract_and_detect(
    video_source: str,   # Path to video file or camera ID
    frame_interval: int = 200,  # Frame sampling interval
    max_frames: int = None,     # Max frames to process
    output_dir: str = "cache_frames",  # Output folder path
    prompt: str = "object"             # Detection prompt
):
    """
    Master pipeline that calls all 4 steps.
    """
    extract_frames(video_source, output_dir, frame_interval, max_frames)
    grounding_dino, sam2, extractor, matcher = initialize_models()
    run_detection_and_segmentation(output_dir, prompt, grounding_dino, sam2, extractor, matcher)


if __name__ == "__main__":
    extract_and_detect(
        video_source="test_video/test_ball.mp4",
        frame_interval=20,
        max_frames=20,
        output_dir="outputs/",
        prompt="ball"
    )
