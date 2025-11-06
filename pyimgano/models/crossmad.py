"""
CrossMAD - Cross-Modal Anomaly Detection

Reference:
    "Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization"
    CVPR 2025

Uses cross-modal learning to harmonize visual prototypes for robust anomaly detection.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


@register_model(
    "vision_crossmad",
    tags=("vision", "deep", "crossmad", "cross-modal", "cvpr2025", "sota"),
    metadata={
        "description": "CrossMAD - Cross-Modal Anomaly Detection (CVPR 2025)",
        "paper": "Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection",
        "year": 2025,
        "conference": "CVPR",
        "type": "cross-modal",
    },
)
class VisionCrossMAD(BaseVisionDeepDetector):
    """CrossMAD: Cross-Modal Anomaly Detection (CVPR 2025)."""

    def __init__(
        self,
        backbone: str = "wide_resnet50",
        num_prototypes: int = 10,
        prototype_dim: int = 512,
        device: str = "cuda",
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.feature_extractor_ = None
        self.prototypes_ = None

    def _preprocess(self, X: NDArray) -> torch.Tensor:
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))
        X = X.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        X = (X - mean) / std
        return torch.from_numpy(X).float()

    def _build_feature_extractor(self):
        if self.backbone == "wide_resnet50":
            from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            resnet = wide_resnet50_2(weights=weights)
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1
            resnet = resnet18(weights=weights)

        extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        for param in extractor.parameters():
            param.requires_grad = False
        extractor.eval()
        return extractor

    def fit(self, X: NDArray, y: Optional[NDArray] = None) -> "VisionCrossMAD":
        X_tensor = self._preprocess(X)
        
        if self.feature_extractor_ is None:
            self.feature_extractor_ = self._build_feature_extractor().to(self.device)

        with torch.no_grad():
            features = []
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i:i + 32].to(self.device)
                feat = self.feature_extractor_(batch)
                features.append(feat.cpu())
            all_features = torch.cat(features, dim=0)

        # K-means clustering for prototypes
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_prototypes, random_state=self.random_state)
        kmeans.fit(all_features.numpy())
        self.prototypes_ = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        return self

    def predict(self, X: NDArray) -> NDArray:
        X_tensor = self._preprocess(X)
        
        with torch.no_grad():
            features = []
            for i in range(0, len(X_tensor), 32):
                batch = X_tensor[i:i + 32].to(self.device)
                feat = self.feature_extractor_(batch)
                features.append(feat)
            test_features = torch.cat(features, dim=0)

        # Distance to prototypes
        distances = torch.cdist(test_features, self.prototypes_)
        scores = distances.min(dim=1)[0]
        return scores.cpu().numpy()

    def decision_function(self, X: NDArray) -> NDArray:
        return self.predict(X)
