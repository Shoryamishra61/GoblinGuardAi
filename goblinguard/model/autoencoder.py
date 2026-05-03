"""PyTorch autoencoder for unsupervised semantic anomaly detection.

Train on CLEAN outputs only. High reconstruction error on new inputs
signals semantic drift from the learned clean distribution.

Architecture:
    Encoder: Linear(384→256) → ReLU → Linear(256→128) → ReLU → Linear(128→latent_dim)
    Decoder: Linear(latent_dim→128) → ReLU → Linear(128→256) → ReLU → Linear(256→384)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TicAutoencoder(nn.Module):
    """Autoencoder for detecting semantic anomalies in LLM outputs.

    Trained on clean-only embeddings so that tic-contaminated inputs
    produce high reconstruction error, signaling distributional drift.

    Args:
        input_dim: Dimensionality of input embeddings.
        latent_dim: Dimensionality of the latent bottleneck.
    """

    def __init__(self, input_dim: int = 384, latent_dim: int = 64) -> None:
        """Initialize encoder and decoder networks.

        Args:
            input_dim: Dimensionality of input embeddings (default: 384).
            latent_dim: Dimensionality of the latent space (default: 64).
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (reconstructed_x, latent_z).

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Tuple of (reconstructed tensor, latent representation).
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Tensor of shape (N,) with MSE for each input sample.
        """
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=-1)

    def anomaly_scores(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Binary anomaly flags: 1.0 if reconstruction_error > threshold, else 0.0.

        Args:
            x: Input tensor of shape (N, input_dim).
            threshold: MSE threshold above which a sample is flagged anomalous.

        Returns:
            Tensor of shape (N,) with binary anomaly flags.
        """
        errors = self.reconstruction_error(x)
        return (errors > threshold).float()
