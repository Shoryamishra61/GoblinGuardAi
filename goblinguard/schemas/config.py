"""Configuration dataclass for GoblinGuard runtime settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoblinGuardConfig:
    """Runtime configuration for GoblinGuard.

    Attributes:
        embedding_model: HuggingFace model identifier for sentence embeddings.
        embedding_dim: Dimensionality of the embedding space.
        autoencoder_latent_dim: Latent dimension for the autoencoder bottleneck.
        ae_anomaly_threshold: MSE threshold for anomaly detection; set from training stats.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        ngram_zscore_threshold: Z-score threshold for flagging anomalous n-grams.
        weight_ngram: Fusion weight for the n-gram scanner signal.
        weight_autoencoder: Fusion weight for the autoencoder signal.
        weight_classifier: Fusion weight for the classifier signal.
        threshold_watch: TicScore threshold for WATCH risk level.
        threshold_alert: TicScore threshold for ALERT risk level.
        threshold_critical: TicScore threshold for CRITICAL risk level.
        device: Torch device string for inference.
    """

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    autoencoder_latent_dim: int = 64
    ae_anomaly_threshold: float = 0.05
    ngram_range: tuple[int, int] = (1, 4)
    ngram_zscore_threshold: float = 2.5
    # Detector weights for TicScore fusion (must sum to 1.0)
    weight_ngram: float = 0.30
    weight_autoencoder: float = 0.30
    weight_classifier: float = 0.40
    # Risk level thresholds
    threshold_watch: float = 25.0
    threshold_alert: float = 50.0
    threshold_critical: float = 75.0
    device: str = "cpu"
