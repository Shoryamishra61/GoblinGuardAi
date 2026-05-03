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


def test_scanner_flags_goblin_terms() -> None:
    """Scanner should flag 'goblin' as an anomalous n-gram."""
    scanner = NgramScanner(baseline_corpus=BASELINE)
    findings = scanner.scan(TIC_TEXTS)
    flagged_ngrams = [f["ngram"] for f in findings]
    assert any("goblin" in ng for ng in flagged_ngrams), "Expected 'goblin' to be flagged"


def test_scanner_clean_texts_low_zscores() -> None:
    """Clean texts scanned against themselves should produce no critical findings."""
    scanner = NgramScanner(baseline_corpus=CLEAN_TEXTS)
    findings = scanner.scan(CLEAN_TEXTS)
    critical_findings = [f for f in findings if f["severity"] == "critical"]
    assert len(critical_findings) == 0


def test_scanner_returns_sorted_by_zscore() -> None:
    """Findings should be sorted by Z-score in descending order."""
    scanner = NgramScanner(baseline_corpus=BASELINE)
    findings = scanner.scan(TIC_TEXTS)
    zscores = [f["zscore"] for f in findings]
    assert zscores == sorted(zscores, reverse=True)


def test_scanner_save_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Scanner should serialize/deserialize consistently."""
    scanner = NgramScanner(baseline_corpus=BASELINE)
    save_path = tmp_path / "scanner.pkl"
    scanner.save(save_path)
    loaded = NgramScanner.load(save_path)
    original_findings = scanner.scan(TIC_TEXTS)
    loaded_findings = loaded.scan(TIC_TEXTS)
    assert len(original_findings) == len(loaded_findings)
