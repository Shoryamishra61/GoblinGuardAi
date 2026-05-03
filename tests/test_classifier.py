"""Tests for TicClassifier."""

import pytest
import torch

from goblinguard.model.classifier import TicClassifier


def test_classifier_output_shape() -> None:
    """Classifier should produce (N, 2) logits."""
    model = TicClassifier(input_dim=384)
    x = torch.randn(8, 384)
    logits = model(x)
    assert logits.shape == (8, 2), f"Expected (8, 2), got {logits.shape}"


def test_classifier_predict_proba_sums_to_one() -> None:
    """predict_proba rows should sum to 1.0."""
    model = TicClassifier(input_dim=384)
    x = torch.randn(4, 384)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 2)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


def test_classifier_in_eval_mode_deterministic() -> None:
    """Classifier in eval mode should produce deterministic output."""
    model = TicClassifier(input_dim=384)
    model.eval()
    x = torch.randn(4, 384)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
