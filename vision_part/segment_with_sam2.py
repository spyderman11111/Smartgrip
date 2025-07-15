import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict
import sys

SAM2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Grounded-SAM-2'))
sys.path.append(SAM2_path)

from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2ImagePredictorWrapper:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        mask_threshold: float = 0.3,
        max_hole_area: float = 100.0,
        max_sprinkle_area: float = 50.0,
        multimask_output: bool = False,
        return_logits: bool = False,
    ):
        self.device = device
        self.multimask_output = multimask_output
        self.return_logits = return_logits

        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_id=model_id,
            device=device,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

    def run_inference(
        self,
        image_path: str,
        box: Tuple[int, int, int, int],
        save_dir: str = "./sam2_results",
    ) -> Dict:
        os.makedirs(save_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image)

        box_array = np.array(box, dtype=np.float32)
        masks, ious, lowres = self.predictor.predict(
            box=box_array,
            multimask_output=self.multimask_output,
            return_logits=self.return_logits,
        )

        mask = masks[0]
        mask_uint8 = (mask * 255).astype(np.uint8)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{filename}_mask_gray.png"), mask_uint8)

        return {
            "mask_array": mask.astype(np.uint8),
            "mask_score": float(ious[0]),
            "low_res_mask": lowres[0],
        }

    def run_on_crop(self, crop: Image.Image, save_path: str) -> None:
        """
        Run segmentation on a cropped image and draw only the mask contour.
        The result is saved as an RGB image.
        """
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        image_np = np.array(crop)
        self.predictor.set_image(crop)

        dummy_box = np.array([0, 0, crop.width, crop.height], dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            box=dummy_box,
            multimask_output=self.multimask_output,
            return_logits=self.return_logits,
        )

        mask = masks[0]
        mask_binary = (mask <= 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(mask_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = image_np.copy()
        cv2.drawContours(overlay, contours, -1, color=(0, 255, 0), thickness=2)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)


if __name__ == "__main__":
    image_path = "/home/MA_SmartGrip/Smartgrip/vision_part/outputs/frame_00000.jpg"
    save_dir = "./sam2_debug"
    os.makedirs(save_dir, exist_ok=True)

    predictor = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_threshold=0.3,
        max_hole_area=100.0,
        max_sprinkle_area=50.0,
        multimask_output=False,
        return_logits=False,
    )

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    result = predictor.run_inference(
        image_path=image_path,
        box=(0, 0, w, h),
        save_dir=save_dir
    )

    print(f"[Saved] Gray mask to: {save_dir}")
    print(f"[Info] Mask score: {result['mask_score']:.4f}")
    print(f"[Info] Mask shape: {result['mask_array'].shape}, dtype: {result['mask_array'].dtype}")
