import os
import cv2
import torch
from PIL import Image
from detect_with_dino import GroundingDinoPredictor
from extract_frames import VideoFrameExtractor
from segment_with_sam2 import SAM2ImagePredictorWrapper  


def extract_frames(video_source: str, output_dir: str, frame_interval: int, max_frames: int):
    """
    Step 1: Extract frames from video source and save as JPGs.
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
    Step 2: Load GroundingDINO and SAM2 models.
    """
    grounding_dino = GroundingDinoPredictor(
        model_id="IDEA-Research/grounding-dino-tiny",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sam2 = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_threshold=0.3,
        max_hole_area=100.0,
        max_sprinkle_area=50.0,
        multimask_output=False,
        return_logits=False,
    )
    print(f"Loaded GroundingDINO and SAM2 models")
    return grounding_dino, sam2


def run_detection_and_segmentation(output_dir: str, prompt: str, dino, sam2):
    """
    Step 3 & 4: Detect boxes in each frame → Crop box → Run SAM2 → Save color mask.
    """
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    for fname in frame_files:
        fpath = os.path.join(output_dir, fname)
        image = Image.open(fpath).convert("RGB")

        # Step 3: Run GroundingDINO
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

            # Step 4: Run SAM2 on crop
            save_path = os.path.join(output_dir, f"{fname[:-4]}_box{i}_mask_color.png")
            sam2.run_on_crop(crop, save_path)

            print(f"  - {label}: x1={x1}, y1={y1}, x2={x2}, y2={y2} → mask saved: {save_path}")


def extract_and_detect(
    video_source: str,
    frame_interval: int = 200,
    max_frames: int = None,
    output_dir: str = "cache_frames",
    prompt: str = "object"
):
    """
    Main pipeline in 4 steps:
        1. Extract frames
        2. Load models
        3. Run GroundingDINO detection
        4. Run SAM2 segmentation on each cropped region
    """
    # Step 1: Extract video frames
    extract_frames(video_source, output_dir, frame_interval, max_frames)

    # Step 2: Load models
    grounding_dino, sam2 = initialize_models()

    # Step 3 & 4: Detect + Segment
    run_detection_and_segmentation(output_dir, prompt, grounding_dino, sam2)


if __name__ == "__main__":
    extract_and_detect(
        video_source="../test_video/test_ball.mp4",
        frame_interval=20,
        max_frames=20,
        output_dir="../outputs",
        prompt="ball"
    )
