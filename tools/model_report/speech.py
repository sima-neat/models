"""Word and character error rates for speech recognition accuracy.

A speech model's accuracy.py scores its transcripts with these functions, so every model is measured the same
way. Needs only the Python standard library, like the rest of model_report.

Both texts pass through normalize() first: Unicode NFKC, lowercase, every punctuation or symbol character
replaced with a space, and whitespace collapsed. Numbers are compared as written, so "15" and "fifteen" differ.
Combining marks are kept, so scripts such as Devanagari keep their vowel signs.

A rate is a corpus total: the edits of every utterance divided by the reference words (or characters) of every
utterance, so each recording weighs by its length. It can exceed 1 when a model inserts many words.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence


def normalize(text: str) -> str:
    """The text as scored: NFKC, lowercase, punctuation and symbols as spaces, single spaces between words."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    return " ".join(text.split())


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Word edits over reference words, after normalize(); words are separated by whitespace."""
    return _error_rate(references, hypotheses, lambda text: normalize(text).split())


def char_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Character edits over reference characters, after normalize() and with whitespace removed.

    Meant for languages written without spaces, where word boundaries are not marked in the text.
    """
    return _error_rate(references, hypotheses, lambda text: list(normalize(text).replace(" ", "")))


def _error_rate(
    references: Sequence[str], hypotheses: Sequence[str], tokens: Callable[[str], list[str]]
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError(f"got {len(references)} references but {len(hypotheses)} hypotheses")
    edits = length = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        expected = tokens(reference)
        edits += _edit_distance(expected, tokens(hypothesis))
        length += len(expected)
    if not length:
        raise ValueError("the references hold no words to score against")
    return edits / length


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """The fewest substitutions, deletions and insertions that turn the reference into the hypothesis."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]
