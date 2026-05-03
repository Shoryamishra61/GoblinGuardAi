"""N-gram frequency scanner for detecting statistically anomalous phrase clusters.

Compares input corpus n-gram frequencies against a reference baseline
using Z-scores. Z-score > 2.5 = candidate tic.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from goblinguard.utils.text_utils import normalize_text


class NgramScanner:
    """Detects over-represented n-grams relative to a clean baseline corpus.

    Args:
        baseline_corpus: List of clean reference texts for computing baseline frequencies.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        zscore_threshold: Z-score threshold for flagging anomalous n-grams.
    """

    def __init__(
        self,
        baseline_corpus: list[str],
        ngram_range: tuple[int, int] = (1, 4),
        zscore_threshold: float = 2.5,
    ) -> None:
        """Initialize the scanner with a baseline corpus.

        Args:
            baseline_corpus: List of clean reference texts.
            ngram_range: Tuple specifying min and max n-gram sizes.
            zscore_threshold: Minimum Z-score to flag an n-gram.
        """
        self.ngram_range = ngram_range
        self.zscore_threshold = zscore_threshold
        self.baseline_freqs = self._compute_freqs(baseline_corpus)

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract all n-grams in range from a single text string.

        Args:
            text: Input text to extract n-grams from.

        Returns:
            List of n-gram strings.
        """
        tokens = normalize_text(text).split()
        ngrams: list[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def _compute_freqs(self, corpus: list[str]) -> Counter:
        """Compute aggregate n-gram frequencies across a corpus.

        Args:
            corpus: List of text strings.

        Returns:
            Counter mapping n-gram strings to frequency counts.
        """
        all_ngrams: list[str] = []
        for text in corpus:
            all_ngrams.extend(self._extract_ngrams(text))
        return Counter(all_ngrams)

    def scan(self, texts: list[str], top_k: int = 20) -> list[dict]:
        """Scan texts and return top_k anomalous n-grams sorted by Z-score descending.

        Each result dict: {ngram, input_count, baseline_count, zscore, severity}
        severity: 'watch' (2.5–3.5), 'alert' (3.5–5.0), 'critical' (>5.0)

        Args:
            texts: List of input texts to scan.
            top_k: Maximum number of findings to return.

        Returns:
            List of finding dicts sorted by Z-score descending.
        """
        input_freqs = self._compute_freqs(texts)
        results: list[dict] = []

        for ngram, input_count in input_freqs.most_common(500):
            baseline_count = self.baseline_freqs.get(ngram, 0)
            # Z-score: (observed - expected) / sqrt(max(expected, 0.1))
            # Using 0.1 smoothing so novel terms absent from baseline get proper signal
            denominator = float(np.sqrt(max(baseline_count, 0.1)))
            zscore = float((input_count - baseline_count) / denominator)

            if zscore > self.zscore_threshold:
                severity = self._classify_severity(zscore)
                results.append(
                    {
                        "ngram": ngram,
                        "input_count": input_count,
                        "baseline_count": baseline_count,
                        "zscore": round(zscore, 3),
                        "severity": severity,
                    }
                )

        results.sort(key=lambda item: item["zscore"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _classify_severity(zscore: float) -> str:
        """Classify a Z-score into severity levels.

        Args:
            zscore: The Z-score value.

        Returns:
            Severity string: 'watch', 'alert', or 'critical'.
        """
        if zscore > 5.0:
            return "critical"
        if zscore > 3.5:
            return "alert"
        return "watch"

    def save(self, path: Path) -> None:
        """Pickle scanner state (baseline freqs) to disk.

        Args:
            path: File path to save the pickled state.
        """
        state = {
            "baseline_freqs": self.baseline_freqs,
            "ngram_range": self.ngram_range,
            "zscore_threshold": self.zscore_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "NgramScanner":
        """Load scanner from pickled state.

        Args:
            path: File path to load the pickled state from.

        Returns:
            NgramScanner instance with restored baseline frequencies.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        scanner = cls.__new__(cls)
        scanner.baseline_freqs = state["baseline_freqs"]
        scanner.ngram_range = state["ngram_range"]
        scanner.zscore_threshold = state["zscore_threshold"]
        return scanner
