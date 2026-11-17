"""Local moderation reviews and offline binary-label preference optimization."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .data import normalize_text


@dataclass(frozen=True)
class Review:
    prediction_id: str
    text: str
    model: str
    predicted_label: int
    reviewed_label: int
    hate_score: float
    threshold: float

    @property
    def reward(self):
        return 1 if self.predicted_label == self.reviewed_label else -1


def text_key(text):
    return sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15)
    db.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            prediction_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            text_key TEXT NOT NULL,
            model TEXT NOT NULL,
            predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
            reviewed_label INTEGER NOT NULL CHECK (reviewed_label IN (0, 1)),
            hate_score REAL NOT NULL CHECK (hate_score BETWEEN 0 AND 1),
            threshold REAL NOT NULL CHECK (threshold BETWEEN 0 AND 1),
            reward INTEGER NOT NULL CHECK (reward IN (-1, 1)),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return db


def save_review(path, review):
    text = normalize_text(review.text)
    if not text:
        raise ValueError("Review text cannot be empty.")
    if review.predicted_label != int(review.hate_score >= review.threshold):
        raise ValueError("The predicted label must match the recorded score and threshold.")
    with closing(_connect(path)) as db, db:
        result = db.execute("""
            INSERT INTO reviews (
                prediction_id, text, text_key, model, predicted_label, reviewed_label,
                hate_score, threshold, reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO NOTHING
        """, (
            review.prediction_id, text, text_key(text), review.model, review.predicted_label,
            review.reviewed_label, review.hate_score, review.threshold, review.reward,
        ))
        return result.rowcount == 1


def training_preferences(path, held_out=()):
    """Exclude held-out texts and keep the latest decision per normalized post."""
    excluded = {text_key(record.text) for record in held_out}
    if not Path(path).is_file():
        return [], 0
    with closing(_connect(path)) as db:
        rows = db.execute("""
            SELECT text_key, text, reviewed_label FROM reviews ORDER BY rowid DESC
        """).fetchall()
    preferences = []
    seen = set()
    for key, text, label in rows:
        if key not in excluded and key not in seen:
            preferences.append((text, label))
            seen.add(key)
    return preferences, len(rows) - len(preferences)


def tune_baseline(reference, preferences, *, beta=1.0, regularization=0.1):
    """Fit a regularized log-odds adjustment relative to the frozen SVM."""
    if not preferences:
        raise ValueError("At least one reviewed post is required.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Beta must be positive and finite.")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be positive and finite.")
    texts, labels = zip(*preferences)
    matrix = reference.features.transform(texts)
    signs = 2 * np.asarray(labels) - 1
    size = len(texts)

    def objective(weights):
        margin = beta * signs * (matrix @ weights[:-1] + weights[-1])
        loss = np.logaddexp(0, -margin).mean() + regularization * (weights @ weights) / 2
        slopes = -beta * signs * expit(-margin) / size
        gradient = np.r_[matrix.T @ slopes, slopes.sum()] + regularization * weights
        return loss, gradient

    result = minimize(
        objective, np.zeros(matrix.shape[1] + 1), jac=True, method="L-BFGS-B",
    )
    if not result.success:
        raise ValueError(f"Preference optimization failed: {result.message}")
    adjustment = result.x if reference.adjustment is None else reference.adjustment + result.x
    return replace(reference, adjustment=adjustment)
