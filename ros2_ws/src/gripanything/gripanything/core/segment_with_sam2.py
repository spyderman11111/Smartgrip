import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict
import sys

# Add SAM2 package to Python path
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
        """
        Initialize the SAM2 wrapper with model configuration.
        """
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
        """
        Run SAM2 inference on the given image and save outputs:
        - gray mask (mask == 1)
        - colored mask overlay
        - transparent RGBA cropped mask
        """
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0]

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image)

        # Run SAM2 with given box
        box_array = np.array(box, dtype=np.float32)
        masks, ious, lowres = self.predictor.predict(
            box=box_array,
            multimask_output=self.multimask_output,
            return_logits=self.return_logits,
        )

        # Get first mask
        mask = masks[0].astype(bool)

        # 1. Save gray mask
        mask_uint8 = (mask.astype(np.uint8)) * 255
        gray_mask_path = os.path.join(save_dir, f"{filename}_mask_gray.png")
        cv2.imwrite(gray_mask_path, mask_uint8)

        # 2. Save RGBA image with transparent background
        mask = masks[0].astype(bool)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        rgba = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3][mask_3d] = image_np[mask_3d]
        rgba[:, :, 3][mask] = 255

        rgba_path = os.path.join(save_dir, f"{filename}_mask_rgba.png")
        Image.fromarray(rgba, mode="RGBA").save(rgba_path)

        # 3. Save overlay visualization (original + semi-transparent mask)
        overlay = image_np.copy()
        overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        overlay_path = os.path.join(save_dir, f"{filename}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "mask_array": mask.astype(np.uint8),
            "mask_score": float(ious[0]),
            "low_res_mask": lowres[0],
            "mask_gray_path": gray_mask_path,
            "mask_rgba_path": rgba_path,
            "mask_overlay_path": overlay_path,
        }


if __name__ == "__main__":
    image_path = "/media/MA_SmartGrip/Data/Smartgrip/vision_part/aria_images/frame_00000.jpg"
    save_dir = os.path.join(os.path.dirname(__file__), "sam2_debug")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize predictor
    predictor = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_threshold=0.3,
        max_hole_area=100.0,
        max_sprinkle_area=50.0,
        multimask_output=False,
        return_logits=False,
    )

    # Run inference on full image
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    result = predictor.run_inference(
        image_path=image_path,
        box=(0, 0, w, h),
        save_dir=save_dir
    )

    # Print output info
    print(f"[Saved] Gray mask to: {result['mask_gray_path']}")
    print(f"[Saved] Transparent RGBA to: {result['mask_rgba_path']}")
    print(f"[Saved] Overlay visualization to: {result['mask_overlay_path']}")
    print(f"[Info] Mask score: {result['mask_score']:.4f}")
    print(f"[Info] Mask shape: {result['mask_array'].shape}, dtype: {result['mask_array'].dtype}")
