import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union, Dict

from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2ImagePredictorWrapper:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        multimask_output: bool = False,
        return_logits: bool = False,
    ):
        """
        Initialize the SAM2 wrapper with adjustable parameters.
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
        Run inference on a single image with a box prompt and save results.
        """
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

        mask = masks[0]  # (H, W)
        score = float(ious[0])
        low_res = lowres[0]  # (256, 256)

        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 1.0, mask_colored, 0.5, 0)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{filename}_original.jpg"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{filename}_mask_gray.png"), mask_uint8)
        cv2.imwrite(os.path.join(save_dir, f"{filename}_mask_color.png"), mask_colored)
        cv2.imwrite(os.path.join(save_dir, f"{filename}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Saved to: {save_dir}")
        print(f"Mask score: {score:.4f}")
        print(f"Mask shape: {mask.shape}, Low-res logits shape: {low_res.shape}")

        return {
            "mask_array": mask.astype(np.uint8),
            "mask_score": score,
            "low_res_mask": low_res,
        }


if __name__ == "__main__":
    # === Static configuration ===
    image_path = "/path/to/your/image.jpg" 
    box = (100, 150, 300, 400)  # (x1, y1, x2, y2)
    save_dir = "./sam2_results"

    # Parameters affecting mask quality
    mask_threshold = 0.3
    max_hole_area = 100.0
    max_sprinkle_area = 50.0
    multimask_output = True
    return_logits = False

    predictor = SAM2ImagePredictorWrapper(
        model_id="facebook/sam2.1-hiera-large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
        multimask_output=multimask_output,
        return_logits=return_logits,
    )

    predictor.run_inference(
        image_path=image_path,
        box=box,
        save_dir=save_dir,
    )
