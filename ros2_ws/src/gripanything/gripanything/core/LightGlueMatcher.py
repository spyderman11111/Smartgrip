import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


class LightGlueMatcher:
    def __init__(
        self,
        feature_type: str = "superpoint",
        device: str = "cuda",
        descriptor_dim: int = 256,
        n_layers: int = 9,
        num_heads: int = 4,
        flash: bool = True,
        mp: bool = False,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        weights: str = None,
    ):
        """
        Wrapper class for LightGlue matcher with configurable options.

        Args:
            feature_type (str): Type of feature extractor, e.g., 'superpoint'.
            device (str): Device to run inference on ('cuda' or 'cpu').
            descriptor_dim (int): Descriptor dimensionality.
            n_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            flash (bool): Whether to enable FlashAttention (if available).
            mp (bool): Mixed precision inference.
            depth_confidence (float): Confidence threshold for early stopping.
            width_confidence (float): Confidence threshold for pruning.
            filter_threshold (float): Minimum score for valid matches.
            weights (str): Path to pretrained weights (optional).
        """
        self.device = device
        self.matcher = LightGlue(
            features=feature_type,
            descriptor_dim=descriptor_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            flash=flash,
            mp=mp,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
            weights=weights,
        ).to(device)

        if feature_type == "superpoint":
            self.feature_extractor = SuperPoint().to(device)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

    def extract_features(self, image_path: str) -> dict:
        """
        Extract keypoints and descriptors from a given image path.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing keypoints, descriptors, and image size.
        """
        image = load_image(image_path).to(self.device)
        features = self.feature_extractor.extract(image)
        features["image_size"] = torch.tensor(image.shape[-2:][::-1]).unsqueeze(0).to(self.device)
        return features

    def match(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ) -> dict:
        """
        Perform matching between two sets of keypoints and descriptors.

        Args:
            keypoints0 (Tensor): [B x M x 2] keypoints from image 0.
            descriptors0 (Tensor): [B x M x D] descriptors from image 0.
            keypoints1 (Tensor): [B x N x 2] keypoints from image 1.
            descriptors1 (Tensor): [B x N x D] descriptors from image 1.
            image_size0 (Tensor): [B x 2] original size of image 0.
            image_size1 (Tensor): [B x 2] original size of image 1.

        Returns:
            dict: LightGlue output including matches, scores, etc.
        """
        data = {
            "image0": {
                "keypoints": keypoints0.to(self.device),
                "descriptors": descriptors0.to(self.device),
                "image_size": image_size0.to(self.device),
            },
            "image1": {
                "keypoints": keypoints1.to(self.device),
                "descriptors": descriptors1.to(self.device),
                "image_size": image_size1.to(self.device),
            },
        }
        return self.matcher(data)

    def compile(self, mode: str = "reduce-overhead"):
        """
        Optionally compile transformer layers to reduce runtime overhead.

        Args:
            mode (str): Compilation strategy ('default', 'reduce-overhead', etc.).
        """
        self.matcher.compile(mode=mode)


if __name__ == "__main__":
    # Example: run matcher on two sample images
    image_path0 = "assets/phototourism_aachen/frame_000000.jpg"
    image_path1 = "assets/phototourism_aachen/frame_000001.jpg"

    if not Path(image_path0).exists() or not Path(image_path1).exists():
        raise FileNotFoundError("Sample images not found. Please update the image paths.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)

    # Step 1: Extract features
    feats0 = matcher.extract_features(image_path0)
    feats1 = matcher.extract_features(image_path1)

    # Step 2: Match descriptors
    output = matcher.match(
        feats0["keypoints"], feats0["descriptors"],
        feats1["keypoints"], feats1["descriptors"],
        feats0["image_size"], feats1["image_size"]
    )

    # Step 3: Print match results
    print("Matched pairs:", output["matches"][0].shape[0])
    print("First few matches:\n", output["matches"][0][:5])
    print("Matching scores:\n", output["scores"][0][:5])
