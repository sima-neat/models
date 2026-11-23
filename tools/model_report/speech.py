"""Word and character error rates for speech recognition accuracy.

A speech model's accuracy.py scores its transcripts with these functions, so every model is measured the same
way. Needs only the Python standard library, like the rest of model_report.

Both texts are normalized before they are compared. English uses Whisper's normalizer (english_normalizer.py), the one
published English error rates are commonly scored with: "15" matches "fifteen", "it's" matches "it is", and
"travellers" matches "travelers". Every other language, and word_error_rate_as_written(), uses normalize(): Unicode
NFKC, lowercase, every punctuation or symbol character replaced with a space, and whitespace collapsed. It compares
numbers as written, so "15" and "fifteen" differ, and keeps combining marks, so scripts such as Devanagari keep their
vowel signs.

A rate is a corpus total: the edits of every utterance divided by the reference words (or characters) of every
utterance, so each recording weighs by its length. It can exceed 1 when a model inserts many words.
"""

from __future__ import annotations

import functools
import unicodedata
from collections.abc import Callable, Sequence

from .english_normalizer import EnglishTextNormalizer


def normalize(text: str) -> str:
    """The text as written: NFKC, lowercase, punctuation and symbols as spaces, single spaces between words."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    return " ".join(text.split())


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str], language: str) -> float:
    """Word edits over reference words in a language such as "en" or "hi-IN"; words are separated by whitespace."""
    scored = _normalizer(language)
    return _error_rate(references, hypotheses, lambda text: scored(text).split())


def word_error_rate_as_written(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """The word error rate with numbers, spelling and contractions compared as written, in any language."""
    return _error_rate(references, hypotheses, lambda text: normalize(text).split())


def char_error_rate(references: Sequence[str], hypotheses: Sequence[str], language: str) -> float:
    """Character edits over reference characters in a language, with whitespace removed after normalizing.

    Meant for languages written without spaces, where word boundaries are not marked in the text.
    """
    scored = _normalizer(language)
    return _error_rate(references, hypotheses, lambda text: list("".join(scored(text).split())))


def _normalizer(language: str) -> Callable[[str], str]:
    if not language:
        raise ValueError("language must name the spoken language, such as en")
    return _english() if language.partition("-")[0].lower() == "en" else normalize


@functools.cache
def _english() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


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
