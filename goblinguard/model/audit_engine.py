"""AuditEngine: fuses signals from all three detectors into a single AuditReport.

Detector weighting:
    NgramScanner:   30%
    TicAutoencoder: 30%
    TicClassifier:  40%

TicScore = weighted_sum * 100, clamped to [0, 100].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from goblinguard.model.autoencoder import TicAutoencoder
from goblinguard.model.classifier import TicClassifier
from goblinguard.model.embedder import SentenceEmbedder
from goblinguard.model.scanner import NgramScanner
from goblinguard.schemas.audit_report import (
    AuditReport,
    DetectorBreakdown,
    FixSuggestion,
    InputStats,
    ModelMetadata,
    NgramFinding,
    RiskLevel,
    SentenceScore,
)
from goblinguard.schemas.config import GoblinGuardConfig
from goblinguard.utils.text_utils import count_tokens, split_sentences

# Mapping from detected tic terms to fix suggestion templates
_FIX_TEMPLATES: dict[str, dict[str, str]] = {
    "creature_metaphor": {
        "description": (
            "Creature metaphors (goblin, gremlin, troll, ogre) are over-represented "
            "in non-fantasy contexts, likely from reward-hacking during RLHF."
        ),
        "suggested_override": (
            "Do not use creature or fantasy metaphors (goblin, gremlin, troll, ogre, "
            "beast, hoard) unless the user's query explicitly involves fantasy or gaming."
        ),
    },
    "creature_personification": {
        "description": (
            "Technical concepts are being personified using creature characters, "
            "a common RLHF tic pattern."
        ),
        "suggested_override": (
            "Explain technical concepts directly without personifying them as creatures. "
            "Use precise technical language instead of anthropomorphic descriptions."
        ),
    },
    "creature_simile": {
        "description": (
            "Similes comparing technical behavior to creature behavior are statistically "
            "anomalous relative to the clean baseline."
        ),
        "suggested_override": (
            "Avoid similes that compare software behavior to creature behavior. "
            "Use domain-appropriate analogies when analogies are needed."
        ),
    },
    "fantasy_vocab": {
        "description": (
            "Fantasy vocabulary (hoard, lair, quest, enchant) appears in technical "
            "explanations where it is contextually inappropriate."
        ),
        "suggested_override": (
            "Replace fantasy-genre vocabulary with standard technical terminology. "
            "Do not describe data structures, algorithms, or systems using fantasy language."
        ),
    },
    "lexical_spike": {
        "description": (
            "One or more n-grams are statistically over-represented relative to the "
            "baseline distribution, suggesting a stylistic tic."
        ),
        "suggested_override": (
            "Prefer direct, domain-specific language and avoid repeating stylistic "
            "phrases across unrelated answers."
        ),
    },
}

# Known tic-related terms for heuristic detection
_TIC_TERMS: set[str] = {
    "goblin",
    "goblins",
    "gremlin",
    "gremlins",
    "troll",
    "trolls",
    "ogre",
    "ogres",
    "creature",
    "creatures",
    "beast",
    "beasts",
    "hoard",
    "lair",
    "quest",
}


class AuditEngine:
    """Orchestrates all detectors and produces a structured AuditReport.

    The engine loads pretrained model artifacts from disk and fuses signals
    from the N-gram scanner, autoencoder, and classifier into a single
    TicScore (0–100) with per-sentence breakdowns and fix suggestions.

    Args:
        config: Runtime configuration for thresholds and weights.
        model_dir: Directory containing pretrained model artifacts.
    """

    def __init__(self, config: GoblinGuardConfig, model_dir: Path) -> None:
        """Load all pretrained artifacts from model_dir.

        Expects:
            model_dir/classifier.pt
            model_dir/autoencoder.pt
            model_dir/scanner_baseline.pkl

        If artifacts are not found, initializes with untrained defaults
        and logs a warning.

        Args:
            config: GoblinGuardConfig instance with runtime settings.
            model_dir: Path to directory containing model weights.
        """
        self.config = config
        self.model_dir = model_dir

        # Initialize embedder
        self.embedder = SentenceEmbedder(
            model_name=config.embedding_model,
            device=config.device,
        )

        # Load or initialize classifier
        self.classifier = TicClassifier(input_dim=config.embedding_dim)
        classifier_path = model_dir / "classifier.pt"
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                self.classifier.load_state_dict(checkpoint["model_state"])
            else:
                self.classifier.load_state_dict(checkpoint)
            self.classifier.eval()

        # Load or initialize autoencoder
        self.autoencoder = TicAutoencoder(
            input_dim=config.embedding_dim,
            latent_dim=config.autoencoder_latent_dim,
        )
        ae_path = model_dir / "autoencoder.pt"
        self.ae_threshold = config.ae_anomaly_threshold
        if ae_path.exists():
            checkpoint = torch.load(ae_path, map_location=config.device, weights_only=True)
            if isinstance(checkpoint, dict):
                self.autoencoder.load_state_dict(checkpoint.get("model_state", checkpoint))
                if "ae_threshold" in checkpoint:
                    self.ae_threshold = checkpoint["ae_threshold"]
            else:
                self.autoencoder.load_state_dict(checkpoint)
            self.autoencoder.eval()

        # Load or initialize scanner
        scanner_path = model_dir / "scanner_baseline.pkl"
        if scanner_path.exists():
            self.scanner = NgramScanner.load(scanner_path)
        else:
            # Fallback: empty baseline
            self.scanner = NgramScanner(
                baseline_corpus=["The system provides clear and concise answers."],
                ngram_range=config.ngram_range,
                zscore_threshold=config.ngram_zscore_threshold,
            )

    def audit(
        self,
        texts: list[str],
        model_name: str = "unknown",
        version: str = "unknown",
        temperature: Optional[float] = None,
        prompt_type: Optional[str] = None,
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
        if not texts:
            raise ValueError("Audit requires at least one text output.")

        # Split all texts into sentences for per-sentence analysis
        all_sentences: list[str] = []
        for text in texts:
            all_sentences.extend(split_sentences(text))

        if not all_sentences:
            all_sentences = texts[:]

        # Compute embeddings
        embeddings = self.embedder.encode(all_sentences)

        # --- Detector 1: N-gram scanner ---
        ngram_findings = self.scanner.scan(texts)
        ngram_score = self._compute_ngram_score(ngram_findings)

        # --- Detector 2: Autoencoder anomaly detection ---
        with torch.no_grad():
            recon_errors = self.autoencoder.reconstruction_error(embeddings)
            anomaly_flags = self.autoencoder.anomaly_scores(embeddings, self.ae_threshold)
        ae_score = float(torch.mean(anomaly_flags).item())

        # --- Detector 3: Classifier ---
        with torch.no_grad():
            probs = self.classifier.predict_proba(embeddings)
            tic_probs = probs[:, 1]  # P(tic) for each sentence
        clf_score = float(torch.mean(tic_probs).item())

        # --- Fusion: weighted TicScore ---
        raw_score = (
            self.config.weight_ngram * ngram_score
            + self.config.weight_autoencoder * ae_score
            + self.config.weight_classifier * clf_score
        )
        tic_score = max(0.0, min(100.0, raw_score * 100.0))

        # Determine risk level
        risk_level = self._risk_level(tic_score)

        # Build per-sentence scores
        sentence_scores = self._build_sentence_scores(
            all_sentences, tic_probs.tolist(), recon_errors.tolist()
        )

        # Generate fix suggestions
        fix_suggestions = self._generate_fix_suggestions(ngram_findings)

        # Build input stats
        total_tokens = sum(count_tokens(text) for text in texts)
        input_stats = InputStats(
            num_texts=len(texts),
            total_tokens=total_tokens,
            avg_text_length=round(total_tokens / max(len(texts), 1), 2),
        )

        return AuditReport(
            model_metadata=ModelMetadata(
                model_name=model_name,
                version=version,
                temperature=temperature,
                prompt_type=prompt_type,
            ),
            input_stats=input_stats,
            tic_score=round(tic_score, 1),
            risk_level=risk_level,
            detector_breakdown=DetectorBreakdown(
                ngram_score=round(ngram_score, 4),
                autoencoder_score=round(ae_score, 4),
                classifier_score=round(clf_score, 4),
            ),
            ngram_findings=[NgramFinding(**f) for f in ngram_findings],
            sentence_scores=sentence_scores,
            fix_suggestions=fix_suggestions,
        )

    def _risk_level(self, tic_score: float) -> RiskLevel:
        """Map TicScore to RiskLevel using configured thresholds.

        Args:
            tic_score: The composite TicScore (0–100).

        Returns:
            The corresponding RiskLevel enum value.
        """
        if tic_score > self.config.threshold_critical:
            return RiskLevel.CRITICAL
        if tic_score > self.config.threshold_alert:
            return RiskLevel.ALERT
        if tic_score > self.config.threshold_watch:
            return RiskLevel.WATCH
        return RiskLevel.SAFE

    def _compute_ngram_score(self, findings: list[dict]) -> float:
        """Compute a normalized n-gram anomaly score from findings.

        Args:
            findings: List of n-gram finding dicts from the scanner.

        Returns:
            Normalized score between 0.0 and 1.0.
        """
        if not findings:
            return 0.0
        severity_weights = {"watch": 0.4, "alert": 0.7, "critical": 1.0}
        weighted_sum = 0.0
        for finding in findings[:10]:
            weight = severity_weights.get(finding["severity"], 0.4)
            zscore_norm = min(finding["zscore"] / 8.0, 1.0)
            weighted_sum += zscore_norm * weight
        return min(weighted_sum / 4.0, 1.0)

    def _build_sentence_scores(
        self,
        sentences: list[str],
        tic_probs: list[float],
        recon_errors: list[float],
    ) -> list[SentenceScore]:
        """Build per-sentence score objects with risk labels.

        Args:
            sentences: List of sentence strings.
            tic_probs: Classifier P(tic) per sentence.
            recon_errors: Reconstruction error per sentence.

        Returns:
            List of SentenceScore objects.
        """
        scores: list[SentenceScore] = []
        for sentence, prob, error in zip(sentences, tic_probs, recon_errors):
            # Combine classifier and autoencoder signals for labeling
            combined = max(prob, min(error / max(self.ae_threshold, 1e-6), 1.0) * 0.35)
            label = self._risk_level(combined * 100.0)
            scores.append(
                SentenceScore(
                    sentence=sentence,
                    classifier_prob=round(prob, 4),
                    recon_error=round(error, 6),
                    label=label,
                )
            )
        return scores

    def _generate_fix_suggestions(self, ngram_findings: list[dict]) -> list[FixSuggestion]:
        """Generate rule-based system prompt override suggestions.

        Maps detected tic classes to templated override instructions.
        No API calls — fully offline.

        Args:
            ngram_findings: List of n-gram finding dicts from the scanner.

        Returns:
            List of FixSuggestion objects.
        """
        if not ngram_findings:
            return []

        flagged_ngrams = {f["ngram"] for f in ngram_findings}
        suggestions: list[FixSuggestion] = []
        seen_classes: set[str] = set()

        # Check for creature-related tic classes
        for ngram in flagged_ngrams:
            tokens = set(ngram.split())
            if tokens & _TIC_TERMS:
                for tic_class in [
                    "creature_metaphor",
                    "creature_personification",
                    "creature_simile",
                    "fantasy_vocab",
                ]:
                    if tic_class not in seen_classes and tic_class in _FIX_TEMPLATES:
                        template = _FIX_TEMPLATES[tic_class]
                        suggestions.append(
                            FixSuggestion(
                                tic_class=tic_class,
                                description=template["description"],
                                suggested_override=template["suggested_override"],
                            )
                        )
                        seen_classes.add(tic_class)
                break  # Only need one match to trigger creature suggestions

        # Always suggest generic lexical spike fix if there are findings
        if "lexical_spike" not in seen_classes:
            template = _FIX_TEMPLATES["lexical_spike"]
            suggestions.append(
                FixSuggestion(
                    tic_class="lexical_spike",
                    description=template["description"],
                    suggested_override=template["suggested_override"],
                )
            )

        return suggestions[:3]
