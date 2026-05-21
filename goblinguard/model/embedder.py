"""Sentence embedding module using sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

import torch


class SentenceEmbedder:
    """Wraps sentence-transformers to produce normalized float32 embeddings.

    Args:
        model_name: HuggingFace model identifier. Defaults to all-MiniLM-L6-v2.
        device: Torch device string. Defaults to 'cpu'.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize the embedder with a sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier.
            device: Torch device string ('cpu', 'cuda', 'mps').
        """
        self._model = SentenceTransformer(model_name, device=device)
        self._device = device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings to a (N, embedding_dim) float32 tensor.

        Embeddings are L2-normalized (unit vectors).

        Args:
            texts: List of input strings to embed.

        Returns:
            Tensor of shape (N, embedding_dim) with L2-normalized embeddings.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return torch.tensor(embeddings, dtype=torch.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding space."""
        return self._model.get_sentence_embedding_dimension()
