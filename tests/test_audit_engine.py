"""Integration tests for AuditEngine."""

import json

import pytest
from unittest.mock import MagicMock

from goblinguard.schemas.audit_report import (
    AuditReport,
    DetectorBreakdown,
    InputStats,
    ModelMetadata,
    RiskLevel,
)
from goblinguard.schemas.config import GoblinGuardConfig


def test_risk_level_thresholds() -> None:
    """Test that _risk_level maps scores to correct buckets."""
    from goblinguard.model.audit_engine import AuditEngine

    config = GoblinGuardConfig()
    engine = MagicMock(spec=AuditEngine)
    engine._risk_level = AuditEngine._risk_level.__get__(engine)
    engine.config = config

    assert engine._risk_level(0.0) == RiskLevel.SAFE
    assert engine._risk_level(25.0) == RiskLevel.SAFE
    assert engine._risk_level(26.0) == RiskLevel.WATCH
    assert engine._risk_level(50.0) == RiskLevel.WATCH
    assert engine._risk_level(51.0) == RiskLevel.ALERT
    assert engine._risk_level(75.0) == RiskLevel.ALERT
    assert engine._risk_level(76.0) == RiskLevel.CRITICAL
    assert engine._risk_level(100.0) == RiskLevel.CRITICAL


def test_audit_report_schema_valid() -> None:
    """AuditReport Pydantic model must serialize/deserialize cleanly."""
    report = AuditReport(
        model_metadata=ModelMetadata(model_name="test", version="0.1"),
        input_stats=InputStats(num_texts=5, total_tokens=200, avg_text_length=40.0),
        tic_score=42.5,
        risk_level=RiskLevel.WATCH,
        detector_breakdown=DetectorBreakdown(
            ngram_score=0.3, autoencoder_score=0.4, classifier_score=0.5
        ),
        ngram_findings=[],
        sentence_scores=[],
        fix_suggestions=[],
    )
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["tic_score"] == 42.5
    assert parsed["risk_level"] == "watch"
    assert "audit_id" in parsed
    assert "timestamp" in parsed
