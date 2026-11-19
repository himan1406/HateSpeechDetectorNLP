"""Binary classification metrics and slice diagnostics for annotated posts."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable

from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from .calibration import bootstrap_interval, brier_score, expected_calibration_error
from .data import Record


def _metrics(pairs: Iterable[tuple[int, int]]) -> dict:
    tn = fp = fn = tp = 0
    for gold, prediction in pairs:
        if gold == prediction:
            if gold:
                tp += 1
            else:
                tn += 1
        elif prediction:
            fp += 1
        else:
            fn += 1
    positives = tp + fn
    negatives = tn + fp
    count = positives + negatives
    positive_f1_denominator = 2 * tp + fp + fn
    negative_f1_denominator = 2 * tn + fp + fn
    f1 = 2 * tp / positive_f1_denominator if positive_f1_denominator else 0.0
    negative_f1 = 2 * tn / negative_f1_denominator if negative_f1_denominator else 0.0
    return {
        "count": count,
        "positive_support": positives,
        "negative_support": negatives,
        "accuracy": (tp + tn) / count,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positive_rate": fp / negatives if negatives else None,
        "false_negative_rate": fn / positives if positives else None,
    }


def evaluate(
    records: Iterable[Record], hate_scores: Iterable[float], threshold: float = 0.5
) -> dict:
    """Report metrics overall and on each language or annotated target group.

    Confusion-matrix rows are gold labels and columns are predictions, in 0, 1
    order. Rates with no corresponding gold examples are undefined (JSON null).
    Group slices can overlap where a record has several annotated target groups.
    """
    threshold = float(threshold)
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite value between 0 and 1")
    records = list(records)
    scores = [float(score) for score in hate_scores]
    if not records or len(records) != len(scores):
        raise ValueError("Evaluation needs one hate score per record and at least one record")
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Hate scores must be finite values between 0 and 1")

    by_language = defaultdict(list)
    by_group = defaultdict(list)
    overall = []
    for record, score in zip(records, scores):
        if record.label not in (0, 1):
            raise ValueError("Evaluation labels must be binary")
        pair = (record.label, int(score >= threshold))
        overall.append(pair)
        by_language[(record.language or "").strip() or "unknown"].append(pair)
        groups = {item.strip() for item in (record.group or "").split("|") if item.strip()}
        for group in groups:
            by_group[group].append(pair)
    labels = [record.label for record in records]
    return {
        "threshold": threshold,
        "overall": _metrics(overall),
        "calibration": {
            "brier": brier_score(labels, scores),
            "ece": expected_calibration_error(labels, scores),
        },
        # Resampling intervals are a floor on uncertainty: they ignore annotation noise.
        "intervals": bootstrap_interval(
            labels, scores, threshold=threshold, n_resamples=400,
        )["metrics"],
        "by_language": {key: _metrics(by_language[key]) for key in sorted(by_language)},
        "by_group": {key: _metrics(by_group[key]) for key in sorted(by_group)},
    }


def _validated_labels_scores(
    labels: Iterable[int], scores: Iterable[float]
) -> tuple[list[int], list[float]]:
    label_values = list(labels)
    score_values = [float(score) for score in scores]
    if not label_values or len(label_values) != len(score_values):
        raise ValueError("Curve evaluation needs one score per label and at least one example")
    if any(label not in (0, 1) for label in label_values):
        raise ValueError("Curve evaluation labels must be binary")
    if any(not isfinite(score) or not 0 <= score <= 1 for score in score_values):
        raise ValueError("Curve evaluation scores must be finite values between 0 and 1")
    return [int(label) for label in label_values], score_values


def _threshold_grid(thresholds: Iterable[float] | None) -> list[float]:
    if thresholds is None:
        return [index / 100 for index in range(101)]
    values = [float(threshold) for threshold in thresholds]
    if not values:
        raise ValueError("Curve evaluation needs at least one threshold")
    if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Curve evaluation thresholds must be finite values between 0 and 1")
    return sorted(set(values))


def score_curves(
    labels: Iterable[int], scores: Iterable[float], thresholds: Iterable[float] | None = None
) -> dict:
    """Compute ROC, precision-recall, and fixed-threshold diagnostics.

    Thresholds follow the same inclusive ``score >= threshold`` rule as
    :func:`evaluate`. Undefined curves use empty arrays and null summaries.
    """
    labels, scores = _validated_labels_scores(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives

    if positives and negatives:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc = {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            "auc": float(auc(fpr, tpr)),
        }
    else:
        roc = {"fpr": [], "tpr": [], "auc": None}

    if positives:
        precision, recall, _ = precision_recall_curve(labels, scores)
        precision_recall = {
            "precision": [float(value) for value in precision],
            "recall": [float(value) for value in recall],
            "average_precision": float(average_precision_score(labels, scores)),
        }
    else:
        precision_recall = {"precision": [], "recall": [], "average_precision": None}

    sweep = []
    for threshold in _threshold_grid(thresholds):
        row = _metrics(
            (label, int(score >= threshold)) for label, score in zip(labels, scores)
        )
        sweep.append({"threshold": threshold, **row})

    return {
        "count": len(labels),
        "positive_support": positives,
        "negative_support": negatives,
        "roc": roc,
        "precision_recall": precision_recall,
        "threshold_sweep": sweep,
    }
