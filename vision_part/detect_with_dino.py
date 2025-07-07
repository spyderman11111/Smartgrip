import torch
from PIL import Image

class GroundingDinoPredictor:
    """
    Wrapper for using a GroundingDINO model for zero-shot object detection.
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        """
        Initialize the GroundingDINO predictor.
        Args:
            model_id (str): HuggingFace model ID to load.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            device
        )

    def predict(
        self,
        image: Image.Image,
        text_prompts: str,
        box_threshold=0.25,
        text_threshold=0.25,
    ):
        """
        Perform object detection using text prompts.
        Args:
            image (PIL.Image.Image): Input RGB image.
            text_prompts (str): Text prompt describing target objects.
            box_threshold (float): Confidence threshold for box selection.
            text_threshold (float): Confidence threshold for text match.
        Returns:
            Tuple[Tensor, List[str]]: Bounding boxes and matched class labels.
        """
        inputs = self.processor(
            images=image, text=text_prompts, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        return results[0]["boxes"], results[0]["labels"]
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image_path = "/home/sz/smartgrip/test_videos/IMG_4031.JPG"
    image = Image.open(image_path).convert("RGB")
    prompt = "red box"

    predictor = GroundingDinoPredictor(
        model_id="IDEA-Research/grounding-dino-tiny"
    )
    boxes, labels = predictor.predict(image, prompt)

    print("Detected objects:")
    for label, box in zip(labels, boxes):
        int_box = [int(coord) for coord in box.tolist()]
        print(f"{label}: {int_box}")


    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for label, box in zip(labels, boxes):
        x0, y0, x1, y1 = box.tolist()
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle(
            (x0, y0), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, label, color="red", fontsize=12)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
