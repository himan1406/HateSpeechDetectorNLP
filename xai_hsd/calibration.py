"""Probability calibration, uncertainty estimates, and operating-point selection.

Moderation decisions need a calibrated score and an explicit operating point.
The helpers here are deliberately dependency-light: NumPy plus scikit-learn's
isotonic/logistic regressors only.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable, Sequence

import numpy as np

METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "false_positive_rate")


def _validated(labels: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if y.size == 0 or y.size != s.size:
        raise ValueError("Calibration needs one score per label and at least one example")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("Calibration labels must be binary")
    if not np.all(np.isfinite(s)) or np.any((s < 0) | (s > 1)):
        raise ValueError("Calibration scores must be finite values between 0 and 1")
    return y, s


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_counts(counts: dict[str, int], metric: str) -> float:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    positives = tp + fn
    negatives = tn + fp
    total = positives + negatives
    if metric == "accuracy":
        return (tp + tn) / total if total else 0.0
    if metric == "precision":
        return tp / (tp + fp) if tp + fp else 0.0
    if metric == "recall":
        return tp / positives if positives else 0.0
    if metric == "false_positive_rate":
        return fp / negatives if negatives else 0.0
    if metric in ("f1", "macro_f1", "negative_f1"):
        positive_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if metric == "f1":
            return positive_f1
        negative_f1 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
        return (positive_f1 + negative_f1) / 2
    raise ValueError(f"Unknown metric {metric!r}")


def metrics_at(labels: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> dict:
    """Confusion counts and the headline metrics at one threshold."""
    y, s = _validated(labels, scores)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite value between 0 and 1")
    counts = _confusion(y, (s >= threshold).astype(int))
    return {
        "threshold": float(threshold),
        **counts,
        **{metric: _metrics_from_counts(counts, metric) for metric in METRICS},
        "count": int(y.size),
        "positive_support": int(np.sum(y == 1)),
        "negative_support": int(np.sum(y == 0)),
    }


def brier_score(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Mean squared error of the probabilities. Lower is better, 0 is perfect."""
    y, s = _validated(labels, scores)
    return float(np.mean((s - y) ** 2))


def reliability_curve(
    labels: Iterable[int], scores: Iterable[float], bins: int = 15
) -> list[dict]:
    """Equal-width reliability bins for plotting and diagnosing miscalibration."""
    y, s = _validated(labels, scores)
    if bins < 1:
        raise ValueError("Bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    curve = []
    for position in range(bins):
        mask = index == position
        count = int(mask.sum())
        curve.append({
            "lower": float(edges[position]),
            "upper": float(edges[position + 1]),
            "count": count,
            "mean_score": float(s[mask].mean()) if count else None,
            "accuracy": float(y[mask].mean()) if count else None,
        })
    return curve


def expected_calibration_error(
    labels: Iterable[int], scores: Iterable[float], bins: int = 15
) -> float:
    """Weighted gap between predicted probability and observed frequency."""
    curve = reliability_curve(labels, scores, bins=bins)
    total = sum(bin["count"] for bin in curve)
    if not total:
        return 0.0
    return float(
        sum(bin["count"] / total * abs(bin["accuracy"] - bin["mean_score"])
            for bin in curve if bin["count"])
    )


def select_threshold(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    target_fpr: float | None = None,
    objective: str = "macro_f1",
) -> dict:
    """Pick an operating point on the validation split.

    With ``target_fpr``, choose the smallest threshold whose false-positive rate
    stays at or below the target, which maximises recall under that budget. When
    no threshold satisfies the constraint, fall back to the most conservative
    one. Without ``target_fpr``, maximise ``objective`` over the observed scores.
    """
    y, s = _validated(labels, scores)
    candidates = np.unique(np.r_[0.0, s, 1.0])
    rows = [metrics_at(y, s, threshold) for threshold in candidates]
    if target_fpr is not None:
        if not isfinite(target_fpr) or not 0 <= target_fpr <= 1:
            raise ValueError("Target false-positive rate must be between 0 and 1")
        eligible = [row for row in rows if row["false_positive_rate"] <= target_fpr]
        chosen = eligible[0] if eligible else rows[-1]
        rule = f"smallest threshold with false-positive rate <= {target_fpr:.3f}"
        return {
            "rule": rule,
            "target_fpr": float(target_fpr),
            "achieved_fpr": chosen["false_positive_rate"],
            "objective": None,
            "value": chosen["recall"],
            "threshold": chosen["threshold"],
            "metrics": chosen,
        }
    if objective not in METRICS:
        raise ValueError(f"Objective must be one of {', '.join(METRICS)}")
    if objective == "false_positive_rate":
        raise ValueError("Use target_fpr to constrain the false-positive rate")
    chosen = max(rows, key=lambda row: (row[objective], row["threshold"]))
    return {
        "rule": f"maximum {objective} on the validation split",
        "target_fpr": None,
        "achieved_fpr": chosen["false_positive_rate"],
        "objective": objective,
        "value": chosen[objective],
        "threshold": chosen["threshold"],
        "metrics": chosen,
    }


def bootstrap_interval(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float = 0.5,
    metrics: Sequence[str] = ("f1", "macro_f1", "precision", "recall"),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap interval for each metric at a fixed threshold.

    Resampling is i.i.d. over posts. Intervals this narrow ignore annotation
    noise, so treat them as a floor on uncertainty rather than the full story.
    """
    y, s = _validated(labels, scores)
    if n_resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are needed")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")

    predictions = (s >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    size = y.size
    draws = {metric: [] for metric in metrics}
    for _ in range(n_resamples):
        sample = rng.integers(0, size, size)
        counts = _confusion(y[sample], predictions[sample])
        for metric in metrics:
            draws[metric].append(_metrics_from_counts(counts, metric))

    alpha = (1 - confidence) / 2
    interval = {}
    for metric in metrics:
        values = np.asarray(draws[metric])
        interval[metric] = {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1 - alpha)),
            "mean": float(values.mean()),
        }
    return {"threshold": float(threshold), "n_resamples": n_resamples,
            "confidence": confidence, "metrics": interval}


def fit_calibrator(
    labels: Iterable[int], scores: Iterable[float], method: str = "isotonic"
):
    """Fit a score -> probability map on validation data.

    ``isotonic`` is flexible and needs a few hundred examples. ``platt`` fits a
    one-dimensional logistic curve and is safer on small validation sets.
    """
    y, s = _validated(labels, scores)
    method = method.lower()
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(s, y)
        return model
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        logits = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
        model = LogisticRegression()
        model.fit(logits.reshape(-1, 1), y)
        return model
    raise ValueError("Calibration method must be 'isotonic' or 'platt'")


def apply_calibrator(model, scores: Iterable[float]) -> np.ndarray:
    """Map raw scores through a fitted calibrator, preserving input order."""
    values = np.asarray([float(score) for score in scores], dtype=float)
    if isinstance(model, np.ndarray):
        raise ValueError("Pass the fitted calibrator, not an array")
    logits = np.log(np.clip(values, 1e-6, 1 - 1e-6) / (1 - np.clip(values, 1e-6, 1 - 1e-6)))
    if hasattr(model, "predict_proba"):  # Platt scaling
        return model.predict_proba(logits.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(model.predict(values), dtype=float), 0.0, 1.0)
