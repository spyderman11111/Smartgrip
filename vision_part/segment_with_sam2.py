import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose


class SAM2ImageSegmentor:
    """
    Wrapper class for SAM2-based segmentation given bounding boxes.
    """

    def __init__(self, config_dir: str, config_name: str, ckpt_path: str, device="cuda"):
        """
        Args:
            config_dir (str): Directory containing the SAM2 config file.
            config_name (str): Filename of the SAM2 config YAML.
            ckpt_path (str): Path to the SAM2 checkpoint (.pt).
            device (str): Device to load model on.
        """
        self.device = device
        with initialize(config_path=config_dir, job_name="sam2_segmentor"):
            model = build_sam2(
                config_file=config_name,
                ckpt_path=ckpt_path,
                device=device,
                mode="eval",
                apply_postprocessing=True,
            )
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image: np.ndarray):
        """Sets the input image for segmentation."""
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        """
        Predicts segmentation masks from input bounding boxes.
        Returns:
            masks: (N, H, W) binary masks
            scores: (N,) mask confidences
            logits: (N, H, W) raw logits
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, scores, logits


if __name__ == "__main__":
    # 获取路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXTERNAL_SAM2_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "external", "Grounded-SAM-2"))

    config_dir = os.path.join(EXTERNAL_SAM2_DIR, "sam2", "configs", "sam2.1")
    config_name = "sam2.1_hiera_l.yaml"
    ckpt_path = os.path.join(EXTERNAL_SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")

    image_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "test_videos", "IMG_4031.JPG"))
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 初始化分割器
    segmentor = SAM2ImageSegmentor(config_dir, config_name, ckpt_path, device="cuda")
    segmentor.set_image(image_rgb)

    # 示例 bbox（可替换）
    box = torch.tensor([[100, 50, 300, 250]], dtype=torch.float)

    # 推理并可视化
    masks, scores, logits = segmentor.predict_masks_from_boxes(box)

    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)
    ax.add_patch(patches.Rectangle(
        (box[0][0], box[0][1]),
        box[0][2] - box[0][0],
        box[0][3] - box[0][1],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    ))
    ax.imshow(masks[0], alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
