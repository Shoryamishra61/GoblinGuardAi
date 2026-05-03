# GoblinGuard — Master Agent Setup Prompt
> Paste this entire file as your first message to Claude Code, Codex, or any agentic coding tool.
> The agent will build the complete project from scratch, in order, without needing to ask questions.

---

You are a senior ML engineer and full-stack developer. Your job is to build **GoblinGuard** — a production-grade, open-source PyTorch-based LLM output auditor — completely from scratch. Every file you write must be production quality: type hints everywhere, docstrings on every function and class, `black`-formatted, and `pytest`-tested.

Do not ask clarifying questions. Follow this prompt exactly, in order, step by step. Build each component fully before moving to the next. Commit-ready code only — no placeholder comments like `# TODO implement this`.

---

## 0. CONTEXT — WHY THIS EXISTS

On April 30, 2026, OpenAI published **"Where the Goblins Came From"** (https://openai.com/index/where-the-goblins-came-from/). Their post-mortem revealed that GPT-5.5's "Nerdy" personality RLHF training accidentally rewarded creature metaphors (goblins, gremlins, trolls, ogres). The tic spread across ALL personalities via SFT data reuse. Usage of the word "goblin" jumped 175% after GPT-5.1. Their emergency fix: a system prompt ban repeated 4 times in the Codex prompt.

GoblinGuard detects this class of failure automatically — before deployment. It fuses three PyTorch-based detectors into a single TicScore (0–100), renders a visual audit dashboard, and exports structured JSON reports for CI/CD pipelines.

**Target audience for the GitHub repo:** AI/ML recruiters and hiring managers at frontier labs and AI startups. Every engineering decision must signal production-grade ML thinking.

---

## 1. PROJECT INITIALIZATION

Run these commands first:

```bash
mkdir goblinguard && cd goblinguard
git init
python -m venv .venv && source .venv/bin/activate
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "goblinguard"
version = "1.0.0"
description = "Real-time LLM output auditor for reward-hacking tics and style leakage"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["llm", "alignment", "rlhf", "auditing", "pytorch", "nlp"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "torch>=2.2.0",
    "sentence-transformers>=2.7.0",
    "transformers>=4.40.0",
    "streamlit>=1.34.0",
    "pydantic>=2.7.0",
    "scipy>=1.13.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "plotly>=5.22.0",
    "scikit-learn>=1.4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "black>=24.0.0", "mypy>=1.9.0"]

[project.scripts]
goblinguard = "goblinguard.__main__:main"

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `requirements.txt` by running `pip install -e ".[dev]"` and then `pip freeze > requirements.txt`.

---

## 2. FOLDER STRUCTURE

Create this exact directory tree (all files, including `__init__.py` where listed):

```
goblinguard/
├── app.py
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE                          (MIT)
├── README.md                        (write last — Section 13)
├── CONTRIBUTING.md
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── goblinguard/
│   ├── __init__.py
│   ├── __main__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── embedder.py
│   │   ├── autoencoder.py
│   │   ├── classifier.py
│   │   ├── scanner.py
│   │   └── audit_engine.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── audit_report.py
│   │   └── config.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   ├── train_autoencoder.py
│   │   └── evaluate.py
│   └── utils/
│       ├── __init__.py
│       ├── text_utils.py
│       └── visualization.py
│
├── data/
│   ├── goblin_examples.json
│   ├── baseline_corpus.json
│   └── data_collection.md
│
├── model/
│   └── .gitkeep                     (weights saved here after training)
│
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py
│   ├── test_classifier.py
│   ├── test_audit_engine.py
│   └── fixtures/
│       └── sample_outputs.json
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_training.ipynb
    └── 03_results_analysis.ipynb
```

---

## 3. DATA FILES

### `data/goblin_examples.json`

Create a labeled dataset with exactly **100 examples for bootstrapping** (50 tic, 50 clean). The agent will use this structure; the user can expand to 1,000 examples later.

**Tic examples (label=1)** must include sentences that:
- Use creature metaphors in non-fantasy contexts (e.g., "Think of your RAM as a goblin hoard")
- Personify gremlins, trolls, ogres in technical explanations
- Reference goblins/gremlins when explaining bugs, performance issues, or logic errors
- Use fantasy creature similes for abstract concepts

**Clean examples (label=0)** must include:
- Factual technical explanations with no creature references
- Straightforward QA answers
- Code explanations in plain language
- Math and science explanations

Format each entry exactly as:
```json
{
  "id": "tic_001",
  "text": "Think of your RAM as a goblin hoard — it hoards your data jealously but won't share it with anyone.",
  "label": 1,
  "tic_class": "creature_metaphor",
  "source_model": "gpt-5.5",
  "prompt_category": "technical_explanation",
  "annotator": "human"
}
```

`tic_class` must be one of: `creature_metaphor`, `creature_personification`, `creature_simile`, `fantasy_vocab`, or `null` for clean examples.

### `data/baseline_corpus.json`

50 clean LLM outputs representing normal, tic-free model behavior. Same schema as above, all `label=0`. These are used to compute the N-gram scanner's baseline frequency distribution and to train the autoencoder.

---

## 4. PYDANTIC SCHEMAS

### `goblinguard/schemas/config.py`

```python
"""Configuration dataclass for GoblinGuard runtime settings."""
from dataclasses import dataclass, field


@dataclass
class GoblinGuardConfig:
    """Runtime configuration for GoblinGuard."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    autoencoder_latent_dim: int = 64
    ae_anomaly_threshold: float = 0.05  # MSE threshold; set from training stats
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
```

### `goblinguard/schemas/audit_report.py`

Build a full Pydantic v2 model with these fields. Every field must have a `description` in `Field(...)`.

```python
"""Pydantic schemas for GoblinGuard audit reports."""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    SAFE = "safe"
    WATCH = "watch"
    ALERT = "alert"
    CRITICAL = "critical"


class Severity(str, Enum):
    WATCH = "watch"
    ALERT = "alert"
    CRITICAL = "critical"


class ModelMetadata(BaseModel): ...          # model_name, version, temperature, prompt_type
class InputStats(BaseModel): ...             # num_texts, total_tokens, avg_text_length
class DetectorBreakdown(BaseModel): ...      # ngram_score, autoencoder_score, classifier_score (all 0.0–1.0)
class NgramFinding(BaseModel): ...           # ngram, input_count, baseline_count, zscore, severity
class SentenceScore(BaseModel): ...          # sentence, classifier_prob, recon_error, label (RiskLevel)
class FixSuggestion(BaseModel): ...          # tic_class, description, suggested_override

class AuditReport(BaseModel):
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="UUID v4 identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC time of audit")
    model_metadata: ModelMetadata
    input_stats: InputStats
    tic_score: float = Field(..., ge=0.0, le=100.0, description="Composite TicScore 0–100")
    risk_level: RiskLevel
    detector_breakdown: DetectorBreakdown
    ngram_findings: list[NgramFinding]
    sentence_scores: list[SentenceScore]
    fix_suggestions: list[FixSuggestion]

    def to_json(self) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=2)
```

Implement all nested models fully with correct types and Field descriptions.

---

## 5. ML MODEL COMPONENTS

Build each file completely. No stubs. Every class and method must have a docstring.

### `goblinguard/model/embedder.py`

```python
"""Sentence embedding module using sentence-transformers."""
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
    ) -> None: ...

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings to a (N, embedding_dim) float32 tensor.

        Embeddings are L2-normalized (unit vectors).
        """
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding space."""
        ...
```

### `goblinguard/model/autoencoder.py`

```python
"""PyTorch autoencoder for unsupervised semantic anomaly detection.

Train on CLEAN outputs only. High reconstruction error on new inputs
signals semantic drift from the learned clean distribution.

Architecture:
    Encoder: Linear(384→256) → ReLU → Linear(256→128) → ReLU → Linear(128→latent_dim)
    Decoder: Linear(latent_dim→128) → ReLU → Linear(128→256) → ReLU → Linear(256→384)
"""
import torch
import torch.nn as nn


class TicAutoencoder(nn.Module):
    """Autoencoder for detecting semantic anomalies in LLM outputs."""

    def __init__(self, input_dim: int = 384, latent_dim: int = 64) -> None: ...

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (reconstructed_x, latent_z)."""
        ...

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error.

        Returns:
            Tensor of shape (N,) with MSE for each input sample.
        """
        ...

    def anomaly_scores(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Binary anomaly flags: 1.0 if reconstruction_error > threshold, else 0.0."""
        ...
```

### `goblinguard/model/classifier.py`

```python
"""PyTorch MLP binary classifier for tic vs clean LLM outputs.

Architecture:
    Linear(384→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→2)

Output logits; apply softmax externally for probabilities.
"""
import torch
import torch.nn as nn


class TicClassifier(nn.Module):
    """Two-layer MLP classifier: clean (0) vs tic (1)."""

    def __init__(self, input_dim: int = 384) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (N, 2)."""
        ...

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities of shape (N, 2).

        Column 0 = P(clean), Column 1 = P(tic).
        """
        ...
```

### `goblinguard/model/scanner.py`

```python
"""N-gram frequency scanner for detecting statistically anomalous phrase clusters.

Compares input corpus n-gram frequencies against a reference baseline
using Z-scores. Z-score > 2.5 = candidate tic.
"""
import pickle
from collections import Counter
from pathlib import Path
import numpy as np


class NgramScanner:
    """Detects over-represented n-grams relative to a clean baseline corpus."""

    def __init__(
        self,
        baseline_corpus: list[str],
        ngram_range: tuple[int, int] = (1, 4),
        zscore_threshold: float = 2.5,
    ) -> None: ...

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract all n-grams in range from a single text string."""
        ...

    def _compute_freqs(self, corpus: list[str]) -> Counter:
        """Compute aggregate n-gram frequencies across a corpus."""
        ...

    def scan(self, texts: list[str], top_k: int = 20) -> list[dict]:
        """Scan texts and return top_k anomalous n-grams sorted by Z-score descending.

        Each result dict: {ngram, input_count, baseline_count, zscore, severity}
        severity: 'watch' (2.5–3.5), 'alert' (3.5–5.0), 'critical' (>5.0)
        """
        ...

    def save(self, path: Path) -> None:
        """Pickle scanner state (baseline freqs) to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> "NgramScanner":
        """Load scanner from pickled state."""
        ...
```

### `goblinguard/model/audit_engine.py`

```python
"""AuditEngine: fuses signals from all three detectors into a single AuditReport.

Detector weighting:
    NgramScanner:   30%
    TicAutoencoder: 30%
    TicClassifier:  40%

TicScore = weighted_sum * 100, clamped to [0, 100].
"""
from pathlib import Path

import torch

from goblinguard.model.autoencoder import TicAutoencoder
from goblinguard.model.classifier import TicClassifier
from goblinguard.model.embedder import SentenceEmbedder
from goblinguard.model.scanner import NgramScanner
from goblinguard.schemas.audit_report import AuditReport, RiskLevel
from goblinguard.schemas.config import GoblinGuardConfig


class AuditEngine:
    """Orchestrates all detectors and produces a structured AuditReport."""

    def __init__(self, config: GoblinGuardConfig, model_dir: Path) -> None:
        """Load all pretrained artifacts from model_dir.

        Expects:
            model_dir/classifier.pt
            model_dir/autoencoder.pt
            model_dir/scanner_baseline.pkl
        """
        ...

    def audit(
        self,
        texts: list[str],
        model_name: str = "unknown",
        version: str = "unknown",
        temperature: float | None = None,
        prompt_type: str | None = None,
    ) -> AuditReport:
        """Run a full audit on a list of LLM output strings.

        Args:
            texts: List of output strings from the model under audit.
            model_name: Name of the model (for report metadata).
            version: Model version string (for report metadata).
            temperature: Sampling temperature used (optional).
            prompt_type: Category of prompts used (optional).

        Returns:
            AuditReport with TicScore, per-sentence scores, n-gram findings,
            detector breakdown, and fix suggestions.
        """
        ...

    def _risk_level(self, tic_score: float) -> RiskLevel:
        """Map TicScore to RiskLevel using configured thresholds."""
        ...

    def _generate_fix_suggestions(self, ngram_findings: list[dict]) -> list[dict]:
        """Generate rule-based system prompt override suggestions.

        Maps detected tic classes to templated override instructions.
        No API calls — fully offline.
        """
        ...
```

---

## 6. TRAINING SCRIPTS

### `goblinguard/training/train_classifier.py`

Write a complete, runnable training script with:

```python
"""Train the TicClassifier on labeled goblin_examples.json data.

Usage:
    python -m goblinguard.training.train_classifier \
        --data data/goblin_examples.json \
        --output model/ \
        --epochs 30 \
        --lr 2e-4 \
        --seed 42
"""
```

Include:
- `argparse` CLI with `--data`, `--output`, `--epochs`, `--lr`, `--batch-size`, `--seed`
- `set_seed(seed)` function for reproducibility (sets torch, numpy, random seeds)
- 70/15/15 train/val/test split using `sklearn.model_selection.train_test_split` with stratify
- `SentenceEmbedder` to pre-compute all embeddings before training (cache to avoid recompute)
- `torch.utils.data.DataLoader` with `shuffle=True` for train, `False` for val/test
- Training loop: `CrossEntropyLoss(label_smoothing=0.1)`, `AdamW(lr, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=epochs)`
- Per-epoch logging: `loss`, `val_loss`, `val_f1`, `val_precision`, `val_recall`
- Best checkpoint saved to `output/classifier.pt` (keyed by `val_f1`)
- Final evaluation on test set printed at end
- Gaussian noise augmentation on training embeddings: `embedding + torch.randn_like(embedding) * 0.01`

### `goblinguard/training/train_autoencoder.py`

Same structure as above but:
- Trains on clean examples only (`label=0`)
- Loss: `MSELoss()`
- Optimizer: `Adam(lr=1e-3, weight_decay=1e-5)`
- Early stopping: patience=10 epochs on val loss
- After training, compute anomaly threshold: `mean_recon_error + 2 * std_recon_error` on val clean set
- Save threshold value alongside weights in the checkpoint dict:
  ```python
  torch.save({"model_state": model.state_dict(), "ae_threshold": threshold}, output / "autoencoder.pt")
  ```
- Save scanner baseline from the clean corpus: `NgramScanner(clean_texts).save(output / "scanner_baseline.pkl")`

### `goblinguard/training/evaluate.py`

```python
"""Evaluate all three detectors and report unified metrics.

Usage:
    python -m goblinguard.training.evaluate \
        --test-data data/goblin_examples.json \
        --model-dir model/ \
        --output evaluation_results.json
"""
```

Report these metrics in a formatted table:
- Classifier: precision, recall, F1, AUROC (sklearn)
- Autoencoder: AUROC using reconstruction error as score
- N-gram scanner: Precision@10 (fraction of top-10 flagged n-grams that are true tics)
- Combined TicScore: Spearman correlation with human severity labels
- Save everything to `evaluation_results.json`

---

## 7. UTILITIES

### `goblinguard/utils/text_utils.py`

```python
"""Text preprocessing utilities for GoblinGuard."""

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules.
    Handles abbreviations, ellipses, and decimal numbers gracefully.
    Returns list of non-empty sentence strings."""
    ...

def count_tokens(text: str) -> int:
    """Approximate token count using whitespace split (good enough for audit stats)."""
    ...

def normalize_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove control characters."""
    ...

def batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Yield successive batches of texts for memory-efficient processing."""
    ...
```

### `goblinguard/utils/visualization.py`

```python
"""Plotly chart builders for the GoblinGuard dashboard."""
import plotly.graph_objects as go
import pandas as pd


def build_ngram_bar_chart(findings: list[dict]) -> go.Figure:
    """Horizontal bar chart of top n-grams by Z-score.
    Bars colored by severity: watch=amber, alert=orange, critical=red.
    Dark background to match app theme."""
    ...


def build_score_trend_chart(sentence_scores: list[dict]) -> go.Figure:
    """Line chart of classifier_prob across sentences.
    Shows where tic intensity peaks within a document.
    Reference line at 0.5 threshold."""
    ...


def build_detector_breakdown_chart(breakdown: dict) -> go.Figure:
    """Horizontal grouped bar comparing ngram/ae/classifier sub-scores.
    Three bars, color-coded to match severity palette."""
    ...
```

All charts must use this Plotly layout base:
```python
DARK_LAYOUT = dict(
    paper_bgcolor="#0A0C0F",
    plot_bgcolor="#111418",
    font=dict(family="IBM Plex Mono, monospace", color="#E8EDF2", size=12),
    margin=dict(l=16, r=16, t=32, b=16),
    showlegend=False,
)
```

---

## 8. CLI

### `goblinguard/__main__.py`

```python
"""GoblinGuard CLI entry point.

Commands:
    audit    — Audit a file of LLM outputs
    compare  — Compare base vs fine-tuned output files
    train    — Train all models on labeled data
    evaluate — Evaluate trained models on test data

Usage:
    python -m goblinguard audit --input outputs.txt --export report.json
    python -m goblinguard compare --base base.txt --finetuned ft.txt
    python -m goblinguard train --data data/goblin_examples.json --output model/
    python -m goblinguard evaluate --test-data data/goblin_examples.json --model-dir model/
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_audit(args: argparse.Namespace) -> None:
    """Load model, run audit, print summary, optionally export JSON."""
    ...


def cmd_compare(args: argparse.Namespace) -> None:
    """Audit two files separately, print side-by-side TicScore comparison."""
    ...


def cmd_train(args: argparse.Namespace) -> None:
    """Sequentially run autoencoder training then classifier training."""
    ...


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation.py and print metrics table."""
    ...


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    ...


def main() -> None:
    args = build_parser().parse_args()
    if not hasattr(args, "func"):
        build_parser().print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
```

---

## 9. STREAMLIT APP

### `app.py`

Build a complete, working Streamlit app. Implement every section below. Do not leave any section as a placeholder.

#### 9.1 — Page config and global CSS injection

```python
st.set_page_config(
    page_title="GoblinGuard — LLM Tic Auditor",
    page_icon="👺",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

Inject the full CSS design system via `st.markdown("<style>...</style>", unsafe_allow_html=True)`. Include all CSS variables, component styles, animations, and font imports defined in the design system below.

#### 9.2 — Design system CSS (inject via st.markdown)

**Use EXACTLY these values. No deviations.**

```css
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --color-bg:            #0A0C0F;
  --color-surface:       #111418;
  --color-surface-2:     #1A1E24;
  --color-border:        #252A32;
  --color-border-focus:  #3D8B6B;
  --color-text-primary:  #E8EDF2;
  --color-text-secondary:#7A8694;
  --color-text-muted:    #3D4550;
  --color-primary:       #39D98A;
  --color-primary-dim:   #1A6B43;
  --color-primary-glow:  rgba(57, 217, 138, 0.15);
  --color-safe:          #39D98A;
  --color-watch:         #F5C542;
  --color-alert:         #F0833A;
  --color-critical:      #E8394D;
  --glow-safe:    0 0 16px rgba(57,217,138,0.35);
  --glow-watch:   0 0 16px rgba(245,197,66,0.35);
  --glow-alert:   0 0 16px rgba(240,131,58,0.35);
  --glow-critical:0 0 16px rgba(232,57,77,0.35);
  --font-display: 'Syne', sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;
  --radius-sm: 4px; --radius-md: 8px; --radius-lg: 12px; --radius-xl: 16px;
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px;
  --space-6:24px; --space-8:32px; --space-12:48px; --space-16:64px;
}

/* Force dark background on Streamlit root elements */
.stApp, .main, [data-testid="stAppViewContainer"] {
  background-color: var(--color-bg) !important;
}
[data-testid="stSidebar"] {
  background-color: var(--color-surface) !important;
  border-right: 1px solid var(--color-border);
}

/* Override Streamlit default text colors */
html, body, .stMarkdown, .stText, p, span, label {
  color: var(--color-text-primary);
  font-family: var(--font-display);
}

/* TicScore ring card */
.score-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--space-8);
  text-align: center;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.score-value {
  font-family: var(--font-display);
  font-size: 72px;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
}
.score-label {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--color-text-secondary);
  margin-top: 8px;
}

/* Risk badge */
.risk-badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 12px;
}
.risk-safe     { background:rgba(57,217,138,0.12); color:var(--color-safe); }
.risk-watch    { background:rgba(245,197,66,0.12);  color:var(--color-watch); }
.risk-alert    { background:rgba(240,131,58,0.12);  color:var(--color-alert); }
.risk-critical {
  background:rgba(232,57,77,0.12);
  color:var(--color-critical);
  animation: pulse-critical 2s ease-in-out infinite;
}
@keyframes pulse-critical {
  0%,100% { box-shadow: none; }
  50%      { box-shadow: var(--glow-critical); }
}

/* N-gram table */
.ngram-row {
  display:flex; align-items:center; gap:12px;
  padding: 10px 16px;
  border-radius: var(--radius-md);
  margin-bottom: 4px;
  border: 1px solid var(--color-border);
  background: var(--color-surface-2);
  transition: background 100ms;
}
.ngram-row:hover { background: var(--color-surface); }
.ngram-text {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--color-text-primary);
  flex: 1;
}
.ngram-zscore {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--color-text-secondary);
  min-width: 80px;
  text-align: right;
}

/* Sentence highlight classes */
.sent-clean    { border-radius:3px; padding:1px 2px; }
.sent-watch    { background:rgba(245,197,66,0.10); border-bottom:1px solid rgba(245,197,66,0.4); border-radius:3px; padding:1px 2px; }
.sent-alert    { background:rgba(240,131,58,0.12); border-bottom:1px solid rgba(240,131,58,0.5); border-radius:3px; padding:1px 2px; }
.sent-critical { background:rgba(232,57,77,0.14);  border-bottom:2px solid var(--color-critical); border-radius:3px; padding:1px 2px; }

/* Section header style */
.section-header {
  font-family: var(--font-display);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 10px;
  margin-bottom: 16px;
}

/* Fix suggestion card */
.fix-card {
  background: var(--color-surface-2);
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-primary);
  border-radius: var(--radius-md);
  padding: 16px;
  margin-bottom: 12px;
}
.fix-card .fix-title {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing:0.08em;
  text-transform:uppercase;
  color: var(--color-primary);
  margin-bottom: 6px;
}
.fix-card .fix-body {
  font-size: 14px;
  color: var(--color-text-secondary);
  line-height: 1.6;
}
.fix-card code {
  font-family: var(--font-mono);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 12px;
  color: var(--color-primary);
}

/* Streamlit widget overrides */
.stTextArea textarea, .stTextInput input {
  background: var(--color-surface-2) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text-primary) !important;
  font-family: var(--font-mono) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--color-border-focus) !important;
  box-shadow: 0 0 0 3px var(--color-primary-glow) !important;
}
.stButton > button {
  background: var(--color-primary) !important;
  color: #0A0C0F !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  height: 44px !important;
  transition: transform 80ms ease, box-shadow 120ms ease !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: var(--glow-safe) !important;
}
```

#### 9.3 — Sidebar

```python
with st.sidebar:
    # Logo header
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
      <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                  letter-spacing:-0.02em; color:#E8EDF2;">
        👺 GoblinGuard
      </div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                  letter-spacing:0.12em; text-transform:uppercase;
                  color:#7A8694; margin-top:4px;">
        LLM Output Auditor v1.0.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Audit Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Single Audit", "Batch Audit (.json)", "Compare: Base vs Fine-tuned"], label_visibility="collapsed")

    st.markdown('<div class="section-header" style="margin-top:24px;">Model Metadata</div>', unsafe_allow_html=True)
    model_name = st.text_input("Model name", placeholder="e.g. gpt-5.5")
    model_version = st.text_input("Version", placeholder="e.g. 2026-04-30")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3D4550; line-height:1.6;">
      Inspired by OpenAI's<br>
      <a href="https://openai.com/index/where-the-goblins-came-from/"
         style="color:#3D8B6B;">Where the Goblins Came From</a><br>
      April 30, 2026
    </div>
    """, unsafe_allow_html=True)
```

#### 9.4 — Main content area

Implement these sections in order:

**Section A — Page header**
```python
st.markdown("""
<div style="margin-bottom:32px;">
  <h1 style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800;
              letter-spacing:-0.02em; color:#E8EDF2; margin:0;">
    LLM Output Auditor
  </h1>
  <p style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#7A8694; margin:8px 0 0 0;">
    Detect reward-hacking tics and style leakage before they reach production.
  </p>
</div>
""", unsafe_allow_html=True)
```

**Section B — Input area** (changes based on mode)
- Single Audit: `st.text_area("Paste LLM output", height=200, placeholder="Paste model output here...")`
- Batch Audit: `st.file_uploader("Upload JSON file of outputs", type=["json"])`
- Compare mode: Two columns, each with a text area (Base Output / Fine-tuned Output)

**Section C — Run Audit button**
- Green primary button labeled "Run Audit →"
- On click: call `AuditEngine.audit(texts, ...)` with a `st.spinner("Running audit...")`
- Store result in `st.session_state["last_report"]`

**Section D — Results (only shown after audit)**

Render inside `if "last_report" in st.session_state:`:

**D1 — Score row** (2 columns):

Column 1 — TicScore card:
```python
score_color = {
    "safe": "#39D98A", "watch": "#F5C542",
    "alert": "#F0833A", "critical": "#E8394D"
}[report.risk_level]

st.markdown(f"""
<div class="score-card">
  <div class="score-value" style="color:{score_color};">{report.tic_score:.1f}</div>
  <div class="score-label">TicScore</div>
  <div class="risk-badge risk-{report.risk_level}">{report.risk_level.upper()}</div>
</div>
""", unsafe_allow_html=True)
```

Column 2 — Detector breakdown (three mini-metrics using `st.metric()`):
- `st.metric("N-gram Signal", f"{report.detector_breakdown.ngram_score:.2f}")`
- `st.metric("Autoencoder Signal", f"{report.detector_breakdown.autoencoder_score:.2f}")`
- `st.metric("Classifier Signal", f"{report.detector_breakdown.classifier_score:.2f}")`

**D2 — Highlighted text view**:
- Split input text into sentences using `split_sentences()`
- For each sentence, get its label from `sentence_scores`
- Build one HTML string with each sentence wrapped in the correct `<span class="sent-{label}">` span
- Render with `st.markdown(html, unsafe_allow_html=True)`

**D3 — N-gram leaderboard**:
- Render each `NgramFinding` as a `.ngram-row` div
- Show: ngram text, Z-score, severity badge
- Use `st.markdown(html, unsafe_allow_html=True)` for the whole table

**D4 — Trend chart**:
- `st.plotly_chart(build_score_trend_chart(report.sentence_scores), use_container_width=True)`

**D5 — N-gram bar chart**:
- `st.plotly_chart(build_ngram_bar_chart(report.ngram_findings), use_container_width=True)`

**D6 — Fix suggestions**:
- For each `FixSuggestion`, render a `.fix-card` div
- Show tic_class, description, and `suggested_override` in a `<code>` block

**D7 — Export button**:
```python
st.download_button(
    label="Export JSON Report",
    data=report.to_json(),
    file_name=f"goblinguard_report_{report.audit_id[:8]}.json",
    mime="application/json",
)
```

#### 9.5 — Model loading

At app startup, load `AuditEngine` once into `st.session_state["engine"]` using `@st.cache_resource`:

```python
@st.cache_resource
def load_engine() -> AuditEngine:
    config = GoblinGuardConfig()
    model_dir = Path("model/")
    return AuditEngine(config=config, model_dir=model_dir)
```

If model weights are not found in `model/`, show a warning and a button: `"Train models first: python -m goblinguard train --data data/goblin_examples.json --output model/"`. Do not crash.

---

## 10. TESTS

### `tests/fixtures/sample_outputs.json`

Create 20 test examples (10 tic, 10 clean) in the same schema as `goblin_examples.json`.

### `tests/test_scanner.py`

```python
"""Tests for NgramScanner."""
import pytest
from goblinguard.model.scanner import NgramScanner

BASELINE = [
    "The function returns a list of integers.",
    "Use a hash map to cache the results.",
    "This algorithm runs in O(n log n) time.",
]

TIC_TEXTS = [
    "Think of your RAM as a goblin hoard that jealously guards your data.",
    "The gremlin in your code is hiding in the recursion stack.",
]

CLEAN_TEXTS = [
    "RAM is temporary storage used by active processes.",
    "The bug is in the recursive base case.",
]


def test_scanner_flags_goblin_terms():
    scanner = NgramScanner(baseline_corpus=BASELINE)
    findings = scanner.scan(TIC_TEXTS)
    flagged_ngrams = [f["ngram"] for f in findings]
    assert any("goblin" in ng for ng in flagged_ngrams), "Expected 'goblin' to be flagged"


def test_scanner_clean_texts_low_zscores():
    scanner = NgramScanner(baseline_corpus=CLEAN_TEXTS)
    findings = scanner.scan(CLEAN_TEXTS)
    # Clean texts scanned against themselves should produce few or no high-z findings
    critical_findings = [f for f in findings if f["severity"] == "critical"]
    assert len(critical_findings) == 0


def test_scanner_returns_sorted_by_zscore():
    scanner = NgramScanner(baseline_corpus=BASELINE)
    findings = scanner.scan(TIC_TEXTS)
    zscores = [f["zscore"] for f in findings]
    assert zscores == sorted(zscores, reverse=True)


def test_scanner_save_load(tmp_path):
    scanner = NgramScanner(baseline_corpus=BASELINE)
    save_path = tmp_path / "scanner.pkl"
    scanner.save(save_path)
    loaded = NgramScanner.load(save_path)
    original_findings = scanner.scan(TIC_TEXTS)
    loaded_findings = loaded.scan(TIC_TEXTS)
    assert len(original_findings) == len(loaded_findings)
```

### `tests/test_classifier.py`

```python
"""Tests for TicClassifier."""
import pytest
import torch
from goblinguard.model.classifier import TicClassifier


def test_classifier_output_shape():
    model = TicClassifier(input_dim=384)
    x = torch.randn(8, 384)
    logits = model(x)
    assert logits.shape == (8, 2), f"Expected (8, 2), got {logits.shape}"


def test_classifier_predict_proba_sums_to_one():
    model = TicClassifier(input_dim=384)
    x = torch.randn(4, 384)
    probs = model.predict_proba(x)
    assert probs.shape == (4, 2)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


def test_classifier_in_eval_mode_deterministic():
    model = TicClassifier(input_dim=384)
    model.eval()
    x = torch.randn(4, 384)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
```

### `tests/test_audit_engine.py`

```python
"""Integration tests for AuditEngine."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from goblinguard.schemas.audit_report import RiskLevel
from goblinguard.schemas.config import GoblinGuardConfig


def test_risk_level_thresholds():
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


def test_audit_report_schema_valid():
    """AuditReport Pydantic model must serialize/deserialize cleanly."""
    from goblinguard.schemas.audit_report import (
        AuditReport, ModelMetadata, InputStats,
        DetectorBreakdown, RiskLevel
    )
    report = AuditReport(
        model_metadata=ModelMetadata(model_name="test", version="0.1"),
        input_stats=InputStats(num_texts=5, total_tokens=200, avg_text_length=40.0),
        tic_score=42.5,
        risk_level=RiskLevel.WATCH,
        detector_breakdown=DetectorBreakdown(ngram_score=0.3, autoencoder_score=0.4, classifier_score=0.5),
        ngram_findings=[],
        sentence_scores=[],
        fix_suggestions=[],
    )
    json_str = report.to_json()
    import json
    parsed = json.loads(json_str)
    assert parsed["tic_score"] == 42.5
    assert parsed["risk_level"] == "watch"
    assert "audit_id" in parsed
    assert "timestamp" in parsed
```

---

## 11. CI/CD

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    name: Lint & Test
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Check formatting (black)
        run: black --check goblinguard/ tests/

      - name: Type check (mypy)
        run: mypy goblinguard/ --ignore-missing-imports --no-error-summary

      - name: Run tests
        run: pytest tests/ -v --tb=short

      - name: Verify CLI entrypoint
        run: python -m goblinguard --help
```

---

## 12. MISCELLANEOUS FILES

### `.gitignore`
```
.venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
.env
*.pt            # Exclude large weights from git (unless <50MB)
*.pkl
notebooks/.ipynb_checkpoints/
dist/
build/
*.egg-info/
```
> **Note:** Override for small pretrained weights: add `!model/classifier.pt` and `!model/autoencoder.pt` if files are under 50MB. HF Spaces needs them committed.

### `LICENSE`
Standard MIT license. Year: 2026. Author: [Your Name].

### `CONTRIBUTING.md`
Write a real CONTRIBUTING guide covering: how to set up the dev environment, how to run tests, how to add new tic classes to the dataset, and how to submit a PR.

### `data/data_collection.md`
Write a 300-word document explaining how the 1,000 examples were collected: which GPT-5.5 outputs were sourced from social media and OpenAI's own blog, how clean negatives were generated, how annotators labeled ambiguous cases, and what quality checks were applied.

---

## 13. README.md

Write a compelling, recruiter-optimized README in this exact order:

```markdown
# 👺 GoblinGuard
> Real-time LLM output auditor for reward-hacking tics and style leakage

[badges: build passing | Python 3.11 | PyTorch | HF Spaces | MIT License]

[demo GIF or screenshot — placeholder: "Demo GIF — add after deploying to HF Spaces"]

## The Goblin Incident

[3 paragraphs: tell the story of April 30 2026, the OpenAI blog, the 175% spike,
the emergency system prompt fix. Write it like a blog post, not an academic paper.
End with: "GoblinGuard automates the detection of this class of failure."]

Reference: https://openai.com/index/where-the-goblins-came-from/

## What GoblinGuard Detects

[Bullet list of tic classes with one example each from the real incident]

## Architecture

[ASCII diagram of the three-detector pipeline]

## Quick Start

[5 commands: clone, venv, install, train, run]

## Evaluation Results

| Metric | Score |
|---|---|
| Classifier Precision | 0.XX |
| Classifier Recall | 0.XX |
| F1 Score | 0.XX |
| AUROC | 0.XX |

[Note: fill in actual numbers after training on the dataset]

## Why This Matters

[1 paragraph on alignment gap, reward hacking at scale, and real-world risk
beyond goblins — subtle biases in safety-critical domains]

## Live Demo

[HF Spaces link — placeholder until deployed]

## Citation

OpenAI. "Where the Goblins Came From." April 30, 2026.
https://openai.com/index/where-the-goblins-came-from/
```

---

## 14. FINAL CHECKLIST

Before you say you are done, verify every item:

- [ ] `black goblinguard/ tests/` runs with zero errors
- [ ] `mypy goblinguard/ --ignore-missing-imports` runs with zero errors
- [ ] `pytest tests/ -v` — all tests pass, zero warnings
- [ ] `python -m goblinguard --help` prints help without error
- [ ] `python -m goblinguard audit --input tests/fixtures/sample_outputs.json` runs without crashing (even if model weights are absent — it should print a clear message, not traceback)
- [ ] `streamlit run app.py` starts without import errors
- [ ] All Python files have type hints on every function signature
- [ ] All classes and functions have docstrings
- [ ] `data/goblin_examples.json` has exactly 100 entries with correct schema
- [ ] `data/baseline_corpus.json` has exactly 50 entries, all `label=0`
- [ ] `AuditReport.to_json()` produces valid JSON matching the schema in the PRD
- [ ] README.md contains the OpenAI blog URL
- [ ] `.github/workflows/ci.yml` is syntactically valid YAML

---

## 15. BUILD ORDER (STRICTLY FOLLOW THIS SEQUENCE)

1. Create all directories and `__init__.py` files
2. Write `pyproject.toml` and install dependencies
3. Write Pydantic schemas (`schemas/`)
4. Write `utils/text_utils.py`
5. Write `model/embedder.py`
6. Write `model/autoencoder.py`
7. Write `model/classifier.py`
8. Write `model/scanner.py`
9. Write `model/audit_engine.py`
10. Write `utils/visualization.py`
11. Write `data/goblin_examples.json` (100 examples)
12. Write `data/baseline_corpus.json` (50 examples)
13. Write `training/train_autoencoder.py`
14. Write `training/train_classifier.py`
15. Write `training/evaluate.py`
16. Write `goblinguard/__main__.py` (CLI)
17. Write all tests (`tests/`)
18. Run tests — fix until all pass
19. Write `app.py` (Streamlit)
20. Write `.github/workflows/ci.yml`
21. Write `README.md`, `CONTRIBUTING.md`, `data/data_collection.md`, `LICENSE`
22. Final checklist — fix any failures

**Do not proceed to the next step until the current step is complete and verified.**

---

*End of agent prompt. Start building.*