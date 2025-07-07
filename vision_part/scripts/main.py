import os
import cv2
import torch
from PIL import Image
from detect_with_dino import GroundingDinoPredictor  
from extract_frames import VideoFrameExtractor        


def extract_and_detect(video_source: str,
                       frame_interval: int = 10,
                       max_frames: int = None,
                       output_dir: str = "cache_frames",
                       prompt: str = "object"):
    """
    Main pipeline: Extract frames from video and run GroundingDINO detection on each.

    Args:
        video_source (str): Path to video or livestream ID (e.g., '0').
        frame_interval (int): Interval between saved frames.
        max_frames (int): Maximum number of frames to process.
        output_dir (str): Directory to save extracted frames.
        prompt (str): Text prompt for detection.
    """
    # Step 0: Clear old image files in output directory
    for f in os.listdir(output_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(output_dir, f))

    # Step 1: Extract frames
    extractor = VideoFrameExtractor(
        video_source=video_source,
        frame_interval=frame_interval,
        output_dir=output_dir
    )
    extractor.extract_frames(max_frames=max_frames)

    # Step 2: Initialize GroundingDINO predictor
    predictor = GroundingDinoPredictor(
        model_id="IDEA-Research/grounding-dino-tiny",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Step 3: Iterate over extracted frames and run detection
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    for fname in frame_files:
        fpath = os.path.join(output_dir, fname)
        image = Image.open(fpath).convert("RGB")

        boxes, labels = predictor.predict(
            image=image,
            text_prompts=prompt,
            box_threshold=0.25,
            text_threshold=0.25
        )

        print(f"\n[Frame] {fname} | Detected {len(labels)} {prompt}:")
        for label, box in zip(labels, boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            print(f"  - {label}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")




if __name__ == "__main__":
    extract_and_detect(
        video_source="../test_video/test_ball.mp4",            
        frame_interval=20,             
        max_frames=20,                
        output_dir="../outputs",   
        prompt="ball"              
    )
