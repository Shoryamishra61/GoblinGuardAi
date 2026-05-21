"""Text preprocessing utilities for GoblinGuard."""

from __future__ import annotations

import re
from typing import Generator

_ABBREVIATIONS = {"mr.", "mrs.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "i.e.", "e.g."}
_SENTENCE_ENDERS = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules.

    Handles abbreviations, ellipses, and decimal numbers gracefully.
    Returns list of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    # Protect abbreviations and decimal numbers by replacing dots temporarily
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Protect decimal numbers (e.g., 3.14)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    # Protect ellipses
    protected = protected.replace("...", "<ELLIPSIS>")

    # Split on sentence-ending punctuation followed by whitespace
    parts = _SENTENCE_ENDERS.split(protected)

    # Restore protected sequences
    sentences: list[str] = []
    for part in parts:
        restored = part.replace("<DOT>", ".").replace("<ELLIPSIS>", "...")
        restored = restored.strip()
        if restored:
            sentences.append(restored)

    return sentences


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace split (good enough for audit stats)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def normalize_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove control characters."""
    text = _CONTROL_CHARS.sub("", text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def batch_texts(texts: list[str], batch_size: int) -> Generator[list[str], None, None]:
    """Yield successive batches of texts for memory-efficient processing.

    Args:
        texts: Full list of text strings to batch.
        batch_size: Maximum number of texts per batch.

    Yields:
        Lists of texts, each containing at most batch_size elements.
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]
