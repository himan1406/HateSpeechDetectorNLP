"""Classical text-model candidates for the comparison harness.

Every candidate produces a project model wrapper (``Baseline`` or
``EnsembleBaseline``) so a winning model can be loaded by the app with
``--backend baseline`` without further changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .baseline import Baseline, EnsembleBaseline
from .data import normalize_text

_WORD_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def _word_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2), token_pattern=_WORD_PATTERN, sublinear_tf=True, min_df=2,
    )


def _char_tfidf(analyzer="char"):
    return TfidfVectorizer(
        analyzer=analyzer, ngram_range=(2, 5), sublinear_tf=True, min_df=2,
    )


def _union(*vectorizers):
    return FeatureUnion(
        [(f"v{index}", vectorizer) for index, vectorizer in enumerate(vectorizers)]
    )


def _calibrated_svm(seed, folds):
    return CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", dual="auto", random_state=seed),
        cv=folds,
        ensemble=False,
    )


def _lr(seed, folds):
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    build: Callable[[int, int], tuple]
    ensemble: bool = False


CANDIDATES: dict[str, Candidate] = {
    "svm-wordchar": Candidate(
        "svm-wordchar",
        "Word 1-2 grams + char 2-5 grams TF-IDF, calibrated LinearSVC (the existing baseline)",
        lambda seed, folds: (_union(_word_tfidf(), _char_tfidf()), _calibrated_svm(seed, folds)),
    ),
    "lr-wordchar": Candidate(
        "lr-wordchar",
        "Word + char TF-IDF, balanced logistic regression",
        lambda seed, folds: (_union(_word_tfidf(), _char_tfidf()), _lr(seed, folds)),
    ),
    "lr-word": Candidate(
        "lr-word",
        "Word 1-2 grams TF-IDF only, balanced logistic regression",
        lambda seed, folds: (_word_tfidf(), _lr(seed, folds)),
    ),
    "lr-char": Candidate(
        "lr-char",
        "Character 2-5 gram TF-IDF only (word-boundary aware), balanced logistic regression",
        lambda seed, folds: (_char_tfidf("char_wb"), _lr(seed, folds)),
    ),
    "nb-count": Candidate(
        "nb-count",
        "Word 1-2 gram counts, Complement Naive Bayes",
        lambda seed, folds: (
            CountVectorizer(ngram_range=(1, 2), token_pattern=_WORD_PATTERN, min_df=2),
            ComplementNB(),
        ),
    ),
    "ensemble-lr-char": Candidate(
        "ensemble-lr-char",
        "Three char TF-IDF logistic regressions bagged with seeded bootstrap samples",
        lambda seed, folds: (_char_tfidf("char_wb"), _lr(seed, folds)),
        ensemble=True,
    ),
    "ensemble-lr-wordchar": Candidate(
        "ensemble-lr-wordchar",
        "Three word+char TF-IDF logistic regressions bagged with seeded bootstrap samples",
        lambda seed, folds: (_union(_word_tfidf(), _char_tfidf()), _lr(seed, folds)),
        ensemble=True,
    ),
}


def _folds(labels: np.ndarray) -> int:
    """Calibration folds, capped by the smallest class so every fold sees both."""
    return min(5, int(np.count_nonzero(labels == 0)), int(np.count_nonzero(labels == 1)))


def _fit(candidate: Candidate, records, seed: int) -> Baseline:
    labels = np.array([record.label for record in records], dtype=int)
    folds = _folds(labels)
    if folds < 2:
        raise ValueError("Each candidate needs at least two training records per class.")
    features, classifier = candidate.build(seed, folds)
    matrix = features.fit_transform([normalize_text(record.text) for record in records])
    classifier.fit(matrix, labels)
    return Baseline(features, classifier)


def _bootstrap(records, rng, attempts: int = 10):
    """Resample with replacement, retrying if a draw loses a class."""
    labels = np.array([record.label for record in records])
    size = len(records)
    for _ in range(attempts):
        index = rng.integers(0, size, size)
        if np.unique(labels[index]).size == 2:
            return [records[position] for position in index]
    return list(records)


def train_candidate(name: str, records, seed: int = 42, ensemble_size: int = 3):
    """Fit one named candidate, or its seed-bagged ensemble, on records."""
    if name not in CANDIDATES:
        raise ValueError(f"Unknown model {name!r}. Choose one of {', '.join(CANDIDATES)}")
    candidate = CANDIDATES[name]
    records = list(records)
    if not candidate.ensemble:
        return _fit(candidate, records, seed)
    if ensemble_size < 1:
        raise ValueError("Ensemble size must be at least one")
    rng = np.random.default_rng(seed)
    members = [
        _fit(candidate, _bootstrap(records, rng), seed + 1_000 * index)
        for index in range(ensemble_size)
    ]
    return EnsembleBaseline(members)
