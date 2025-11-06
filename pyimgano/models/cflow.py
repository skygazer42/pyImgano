"""
CFlow-AD: Real-Time Unsupervised Anomaly Detection with Conditional Normalizing Flows.

CFlow uses conditional normalizing flows for fast and accurate anomaly detection,
achieving real-time inference speed.

Reference:
    Gudovskiy, D., Ishizaka, S., & Kozuka, K. (2022).
    CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows.
    In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 98-107).
"""

import logging
from typing import Iterable, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torch.distributions as dist

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

logger = logging.getLogger(__name__)


class ConditionalFlow(nn.Module):
    """Simple conditional normalizing flow."""

    def __init__(self, feature_dim, condition_dim, n_flows=8):
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(nn.Sequential(
                nn.Linear(feature_dim + condition_dim, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim * 2)  # mean and log_scale
            ))

    def forward(self, z, condition):
        log_det_jacobian = 0

        for flow in self.flows:
            # Concatenate z with condition
            inp = torch.cat([z, condition], dim=-1)

            # Get transformation parameters
            params = flow(inp)
            mean, log_scale = params.chunk(2, dim=-1)

            # Apply affine transformation
            z = z * torch.exp(log_scale) + mean

            # Accumulate log determinant
            log_det_jacobian += log_scale.sum(dim=-1)

        return z, log_det_jacobian


class ImagePathDataset(Dataset):
    """Dataset for loading images from paths."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, img_path


@register_model(
    "vision_cflow",
    tags=("vision", "deep", "cflow", "normalizing-flow", "real-time"),
    metadata={
        "description": "CFlow-AD - Conditional normalizing flows (WACV 2022)",
        "paper": "Real-Time Unsupervised Anomaly Detection with Localization",
        "year": 2022,
        "speed": "real-time",
    },
)
class VisionCFlow(BaseVisionDeepDetector):
    """
    CFlow-AD anomaly detector using conditional normalizing flows.

    Parameters
    ----------
    backbone : str, default='resnet18'
        Feature extraction backbone
    n_flows : int, default=8
        Number of flow transformations
    epochs : int, default=50
        Number of training epochs
    batch_size : int, default=16
        Training batch size
    lr : float, default=0.001
        Learning rate
    device : str, default='cpu'
        Device to run model on

    Examples
    --------
    >>> detector = VisionCFlow(epochs=50, device='cuda')
    >>> detector.fit(train_images)
    >>> scores = detector.predict(test_images)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        n_flows: int = 8,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 0.001,
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize CFlow detector."""
        super().__init__(**kwargs)

        self.backbone_name = backbone
        self.n_flows = n_flows
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        # Build model
        self._build_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        logger.info(
            "Initialized CFlow with backbone=%s, n_flows=%d, epochs=%d, device=%s",
            backbone, n_flows, epochs, device
        )

    def _build_model(self):
        """Build feature extractor and flow model."""
        # Feature extractor (frozen)
        if self.backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.to(self.device)

        # Conditional flow model
        # Use pooled features as condition
        self.condition_dim = self.feature_dim

        # Flow for patch-level features
        self.flow = ConditionalFlow(
            feature_dim=self.feature_dim,
            condition_dim=self.condition_dim,
            n_flows=self.n_flows
        )
        self.flow.to(self.device)

    def fit(self, X: Iterable[str], y: Optional[NDArray] = None) -> "VisionCFlow":
        """
        Train CFlow on normal images.

        Parameters
        ----------
        X : iterable of str
            Paths to normal training images
        y : array-like, optional
            Ignored

        Returns
        -------
        self : VisionCFlow
        """
        logger.info("Training CFlow detector")

        X_list = list(X)
        if not X_list:
            raise ValueError("Training set cannot be empty")

        # Create dataset
        dataset = ImagePathDataset(X_list, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Setup optimizer
        optimizer = Adam(self.flow.parameters(), lr=self.lr)

        # Training loop
        self.flow.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for images, _ in dataloader:
                images = images.to(self.device)

                # Extract features
                with torch.no_grad():
                    features = self.feature_extractor(images)  # (B, C, H, W)

                # Global average pooling for condition
                condition = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

                # Flatten spatial dimensions
                B, C, H, W = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)

                # Repeat condition for all patches
                condition_repeated = condition.unsqueeze(1).repeat(1, H * W, 1)
                condition_repeated = condition_repeated.reshape(B * H * W, -1)

                # Flow forward
                z, log_det = self.flow(features_flat, condition_repeated)

                # Compute negative log-likelihood
                # Assume standard normal prior
                log_prob_prior = dist.Normal(0, 1).log_prob(z).sum(dim=-1)
                log_prob = log_prob_prior + log_det

                # Negative log-likelihood loss
                loss = -log_prob.mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, self.epochs, avg_loss)

        logger.info("CFlow training completed")
        return self

    def predict(self, X: Iterable[str]) -> NDArray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : iterable of str
            Paths to test images

        Returns
        -------
        scores : ndarray
            Anomaly scores
        """
        self.flow.eval()

        X_list = list(X)
        scores = np.zeros(len(X_list))

        logger.info("Computing anomaly scores for %d images", len(X_list))

        with torch.no_grad():
            for idx, img_path in enumerate(X_list):
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                    # Extract features
                    features = self.feature_extractor(img_tensor)

                    # Global condition
                    condition = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

                    # Flatten features
                    B, C, H, W = features.shape
                    features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
                    condition_repeated = condition.unsqueeze(1).repeat(1, H * W, 1).reshape(B * H * W, -1)

                    # Compute likelihood
                    z, log_det = self.flow(features_flat, condition_repeated)

                    log_prob_prior = dist.Normal(0, 1).log_prob(z).sum(dim=-1)
                    log_prob = log_prob_prior + log_det

                    # Use negative log-likelihood as anomaly score
                    score = -log_prob.mean().item()
                    scores[idx] = score

                except Exception as e:
                    logger.warning("Failed to score %s: %s", img_path, e)
                    scores[idx] = 0.0

        return scores

    def decision_function(self, X: Iterable[str]) -> NDArray:
        """Compute anomaly scores (alias for predict)."""
        return self.predict(X)
