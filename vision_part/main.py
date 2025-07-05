import copy
import os
import sys
sys.path.append(os.path.abspath("."))

import cv2
import torch
import numpy as np
from PIL import Image
from your_module import IncrementalObjectTracker  # 替换成你实际的路径

def main():
    # ====== 环境设置（放在 main 内，避免非必要全局副作用）======
    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print("[Info] CUDA environment initialized with TF32 + autocast.")
    else:
        print("[Warning] CUDA not available. Running on CPU.")

    # ====== 参数设置 ======
    output_dir = "./outputs"
    prompt_text = "hand."
    detection_interval = 20
    max_frames = 300

    os.makedirs(output_dir, exist_ok=True)

    # ====== 初始化检测器和跟踪器 ======
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("person.")

    # ====== 摄像头读取并实时处理帧 ======
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

    print("[Info] Camera opened. Press 'q' to quit.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[Frame {frame_idx}] Processing live frame...")
            process_image = tracker.add_image(frame_rgb)

            if process_image is None or not isinstance(process_image, np.ndarray):
                print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
                frame_idx += 1
                continue

            tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
            frame_idx += 1

            if frame_idx >= max_frames:
                print(f"[Info] Reached max_frames {max_frames}. Stopping.")
                break
    except KeyboardInterrupt:
        print("[Info] Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Live inference complete.")

if __name__ == "__main__":
    main()
