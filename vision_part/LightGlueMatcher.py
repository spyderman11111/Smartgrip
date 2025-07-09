import torch
from lightglue.lightglue import LightGlue

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
        Wrapper for LightGlue model with customizable parameters.
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
        Match keypoints and descriptors between two images.
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
        Optional: Compile transformer layers for runtime acceleration.
        """
        self.matcher.compile(mode=mode)


if __name__ == "__main__":
    # Debugging example: batch size 1, 128 keypoints per image
    B, M, N, D = 1, 128, 128, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    keypoints0 = torch.rand(B, M, 2, device=device)
    descriptors0 = torch.rand(B, M, D, device=device)
    keypoints1 = torch.rand(B, N, 2, device=device)
    descriptors1 = torch.rand(B, N, D, device=device)
    image_size0 = torch.tensor([[480, 640]], device=device)
    image_size1 = torch.tensor([[480, 640]], device=device)

    matcher = LightGlueMatcher(feature_type="superpoint", device=device, mp=True)
    output = matcher.match(
        keypoints0, descriptors0,
        keypoints1, descriptors1,
        image_size0, image_size1
    )

    print("Matched pairs:", output["matches"][0].shape[0])
    print("First few matches:", output["matches"][0][:5])
    print("Matching scores:", output["scores"][0][:5])
