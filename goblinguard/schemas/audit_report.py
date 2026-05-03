"""Pydantic schemas for GoblinGuard audit reports."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """Risk classification for an audit result."""

    SAFE = "safe"
    WATCH = "watch"
    ALERT = "alert"
    CRITICAL = "critical"


class Severity(str, Enum):
    """Severity classification for individual n-gram findings."""

    WATCH = "watch"
    ALERT = "alert"
    CRITICAL = "critical"


class ModelMetadata(BaseModel):
    """Metadata about the model under audit."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(default="unknown", description="Name of the audited model")
    version: str = Field(default="unknown", description="Model version string")
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature used during generation"
    )
    prompt_type: Optional[str] = Field(
        default=None, description="Category of prompts used to generate outputs"
    )


class InputStats(BaseModel):
    """Basic statistics about the input corpus being audited."""

    num_texts: int = Field(description="Number of text outputs in the audit batch")
    total_tokens: int = Field(description="Total approximate token count across all texts")
    avg_text_length: float = Field(description="Average text length in tokens")


class DetectorBreakdown(BaseModel):
    """Normalized sub-scores from each detector in the fusion pipeline."""

    ngram_score: float = Field(
        ge=0.0, le=1.0, description="Normalized n-gram anomaly signal (0.0–1.0)"
    )
    autoencoder_score: float = Field(
        ge=0.0, le=1.0, description="Normalized autoencoder drift signal (0.0–1.0)"
    )
    classifier_score: float = Field(
        ge=0.0, le=1.0, description="Normalized classifier tic probability (0.0–1.0)"
    )


class NgramFinding(BaseModel):
    """An anomalous n-gram detected by the scanner."""

    ngram: str = Field(description="The n-gram string flagged as anomalous")
    input_count: int = Field(description="Frequency of this n-gram in the input corpus")
    baseline_count: int = Field(description="Frequency of this n-gram in the baseline corpus")
    zscore: float = Field(description="Z-score measuring deviation from baseline frequency")
    severity: Severity = Field(description="Severity level: watch, alert, or critical")


class SentenceScore(BaseModel):
    """Per-sentence detector output with risk label."""

    sentence: str = Field(description="The sentence text being scored")
    classifier_prob: float = Field(description="Classifier probability of tic (P(tic))")
    recon_error: float = Field(description="Autoencoder reconstruction error (MSE)")
    label: RiskLevel = Field(description="Assigned risk level for this sentence")


class FixSuggestion(BaseModel):
    """Rule-based remediation suggestion for a flagged tic class."""

    tic_class: str = Field(description="The tic class this fix targets")
    description: str = Field(description="Human-readable description of the issue")
    suggested_override: str = Field(
        description="Suggested system prompt override to mitigate the tic"
    )


class AuditReport(BaseModel):
    """Complete audit report exported by CLI, SDK, API, and UI."""

    model_config = ConfigDict(protected_namespaces=())

    audit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID v4 identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC time of audit"
    )
    model_metadata: ModelMetadata = Field(description="Metadata about the audited model")
    input_stats: InputStats = Field(description="Statistics about the input corpus")
    tic_score: float = Field(..., ge=0.0, le=100.0, description="Composite TicScore 0–100")
    risk_level: RiskLevel = Field(description="Overall risk classification")
    detector_breakdown: DetectorBreakdown = Field(
        description="Per-detector sub-scores used in fusion"
    )
    ngram_findings: list[NgramFinding] = Field(description="Anomalous n-grams sorted by Z-score")
    sentence_scores: list[SentenceScore] = Field(description="Per-sentence risk scores and labels")
    fix_suggestions: list[FixSuggestion] = Field(description="Suggested system prompt overrides")

    def to_json(self) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=2)
