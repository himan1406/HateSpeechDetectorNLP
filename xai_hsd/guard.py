"""Refuse a verdict when a post carries too little evidence.

A linear text model has nothing to reason about when a post is a handful of
words, or when none of its n-grams appear in the training vocabulary. In both
cases the score drifts toward the class prior, which looks like a confident
prediction but is really the model's bias. This module detects that situation so
the interface can show the score without presenting it as a decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import normalize_text

MIN_WORDS = 8
MIN_MATCHED_FEATURES = 3


@dataclass(frozen=True)
class Evidence:
    """How much usable signal a single post actually contains."""

    words: int
    matched_features: int | None
    min_words: int
    min_matched_features: int
    sufficient: bool
    reason: str | None


def matched_features(model, text: str) -> int | None:
    """Count fitted features the text activates, or None if the model cannot say.

    This is a strong signal for word-level vocabularies and a weak one for
    character n-grams, which still match on unfamiliar words. For character
    models the word-count check carries most of the protection, so treat a
    passing feature count as necessary rather than sufficient evidence.
    """
    text = normalize_text(str(text))
    features = getattr(model, "features", None)
    if features is not None:
        try:
            matrix = features.transform([text])
            return int(matrix.getnnz(axis=1)[0])
        except (AttributeError, ValueError, TypeError):
            return None
    members = getattr(model, "members", None)
    if members:
        counts = [matched_features(member, text) for member in members]
        counts = [count for count in counts if count is not None]
        # The weakest member decides: an ensemble is only as informed as its worst part.
        return min(counts) if counts else None
    known = getattr(model, "has_known_features", None)
    if callable(known):
        return MIN_MATCHED_FEATURES if known(text) else 0
    return None


def assess_evidence(
    model,
    text: str,
    *,
    min_words: int = MIN_WORDS,
    min_matched_features: int = MIN_MATCHED_FEATURES,
) -> Evidence:
    """Decide whether a post is long and familiar enough to support a verdict."""
    if min_words < 1 or min_matched_features < 1:
        raise ValueError("Minimums must be positive")
    normalized = normalize_text(str(text))
    words = len(normalized.split())
    matched = matched_features(model, normalized)

    if not normalized:
        reason = "The post is empty."
    elif words < min_words:
        reason = (
            f"Only {words} word{'s' if words != 1 else ''} of context; "
            f"at least {min_words} are needed for a score to mean anything."
        )
    elif matched is not None and matched < min_matched_features:
        reason = (
            f"Only {matched} known feature{'s' if matched != 1 else ''} from the training "
            "vocabulary, so the score reflects the model's class prior."
        )
    else:
        reason = None

    return Evidence(
        words=words,
        matched_features=matched,
        min_words=min_words,
        min_matched_features=min_matched_features,
        sufficient=reason is None,
        reason=reason,
    )
