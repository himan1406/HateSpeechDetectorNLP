"""Binary classification metrics and slice diagnostics for annotated posts."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable

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
    return {
        "threshold": threshold,
        "overall": _metrics(overall),
        "by_language": {key: _metrics(by_language[key]) for key in sorted(by_language)},
        "by_group": {key: _metrics(by_group[key]) for key in sorted(by_group)},
    }
