import torch
from PIL import Image
import inspect

class GroundingDinoPredictor:
    """
    Wrapper for using a GroundingDINO model for zero-shot object detection.
    兼容不同 transformers 版本的后处理参数名（box_threshold/box_thresh 等）。
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

        # 记录后处理函数及其签名（用于参数名兼容）
        self._post = getattr(self.processor, "post_process_grounded_object_detection", None)
        if self._post is None:
            raise RuntimeError("This processor has no 'post_process_grounded_object_detection' method.")
        self._post_sig_params = set(inspect.signature(self._post).parameters.keys())

    def _to_device(self, inputs):
        """把 processor 输出搬到 device。兼容老版本 BatchFeature 没 .to 的情况。"""
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    def predict(self, image: Image.Image, text_prompts: str,
                box_threshold=0.25, text_threshold=0.25):
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt")
        inputs = self._to_device(inputs)

        # 兼容：inputs 可能是 BatchFeature 或 dict
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids

        with torch.no_grad():
            # BatchFeature 在新版本可直接**展开；老版本用 .data
            call_kwargs = inputs if isinstance(inputs, dict) else inputs.data
            outputs = self.model(**call_kwargs)

        # 按签名决定用哪组参数名
        kwargs = {
            "outputs": outputs,
            "input_ids": input_ids,
            "target_sizes": [image.size[::-1]],
        }
        if "box_threshold" in self._post_sig_params:
            kwargs["box_threshold"] = box_threshold
        elif "box_thresh" in self._post_sig_params:
            kwargs["box_thresh"] = box_threshold

        if "text_threshold" in self._post_sig_params:
            kwargs["text_threshold"] = text_threshold
        elif "text_thresh" in self._post_sig_params:
            kwargs["text_thresh"] = text_threshold

        try:
            results = self._post(**kwargs)
        except TypeError:
            # 极老版本：只接受位置参数（outputs, input_ids, target_sizes, ...）
            thresh_kwargs = {k: kwargs[k] for k in ("box_threshold","box_thresh","text_threshold","text_thresh") if k in kwargs}
            results = self._post(outputs, input_ids, [image.size[::-1]], **thresh_kwargs)

        return results[0]["boxes"], results[0]["labels"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import sys, transformers
    print("transformers =", transformers.__version__)
    print("python =", sys.executable)

    image_path = "/home/MA_SmartGrip/snapshot.jpg"
    image = Image.open(image_path).convert("RGB")
    prompt = "yellow object ."   # DINO 更偏好 'class .' 形式

    predictor = GroundingDinoPredictor(model_id="IDEA-Research/grounding-dino-tiny")
    boxes, labels = predictor.predict(image, prompt, box_threshold=0.20, text_threshold=0.20)

    print("Detected objects:")
    for label, box in zip(labels, boxes):
        int_box = [int(coord) for coord in box.tolist()]
        print(f"{label}: {int_box}")

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for label, box in zip(labels, boxes):
        x0, y0, x1, y1 = box.tolist()
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x0, max(0, y0 - 5), str(label), color="red", fontsize=12)
    plt.axis("off"); plt.tight_layout(); plt.show()
