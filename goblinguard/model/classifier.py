"""PyTorch MLP binary classifier for tic vs clean LLM outputs.

Architecture:
    Linear(384→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→2)

Output logits; apply softmax externally for probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TicClassifier(nn.Module):
    """Two-layer MLP classifier: clean (0) vs tic (1).

    Args:
        input_dim: Dimensionality of input embeddings (default: 384).
    """

    def __init__(self, input_dim: int = 384) -> None:
        """Initialize the classifier network.

        Args:
            input_dim: Dimensionality of input embeddings.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (N, 2).

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Logits tensor of shape (N, 2).
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape (N, 2).

        Column 0 = P(clean), Column 1 = P(tic).

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Probability tensor of shape (N, 2) with rows summing to 1.0.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
