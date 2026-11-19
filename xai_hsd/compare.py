"""Compare candidate text models on one split configuration.

Selection happens on the validation split, and the test split is reported only
for the chosen operating point, so the comparison does not tune on test data.
Optional k-fold cross-validation runs inside the training split only.
"""

from __future__ import annotations

import time
from math import isfinite

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .augment import hard_negatives as build_hard_negatives
from .calibration import (
    bootstrap_interval,
    brier_score,
    expected_calibration_error,
    metrics_at,
    select_threshold,
)
from .models import CANDIDATES, train_candidate


def _scores(model, records) -> np.ndarray:
    return np.asarray(model.predict_proba([record.text for record in records]))[:, 1]


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def _summarise(labels, scores, threshold, *, n_resamples, confidence) -> dict:
    return {
        "threshold": float(threshold),
        "metrics": metrics_at(labels, scores, threshold),
        "auc": _auc(labels, scores),
        "calibration": {
            "brier": brier_score(labels, scores),
            "ece": expected_calibration_error(labels, scores),
        },
        "intervals": bootstrap_interval(
            labels, scores, threshold=threshold,
            n_resamples=n_resamples, confidence=confidence,
        )["metrics"],
    }


def _evaluate_row(model, splits, labels, *, target_fpr, objective, n_resamples, confidence):
    validation_scores = _scores(model, splits["validation"])
    choice = select_threshold(
        labels["validation"], validation_scores, target_fpr=target_fpr, objective=objective,
    )
    threshold = choice["threshold"]
    return {
        "threshold_selection": choice,
        "validation": _summarise(
            labels["validation"], validation_scores, threshold,
            n_resamples=n_resamples, confidence=confidence,
        ),
        "test": _summarise(
            labels["test"], _scores(model, splits["test"]), threshold,
            n_resamples=n_resamples, confidence=confidence,
        ),
    }


def cross_validate(
    name: str,
    train_records,
    validation_records,
    *,
    folds: int = 5,
    seed: int = 42,
    target_fpr: float | None = None,
    objective: str = "macro_f1",
    ensemble_size: int = 3,
) -> dict:
    """K-fold estimate of the training procedure, inside the training split.

    Each fold refits on the remaining training rows, chooses its threshold on the
    untouched official validation split, and scores its held-out fold. The test
    split is never touched. Fold spread reflects both sampling and training
    variance, so it is a sharper selection signal than one validation sample.
    """
    train_records = list(train_records)
    validation_records = list(validation_records)
    if folds < 2:
        raise ValueError("Cross-validation needs at least two folds")
    if folds > min(np.bincount([record.label for record in train_records])):
        raise ValueError("Cross-validation needs at least one example per class per fold")

    labels = np.array([record.label for record in train_records], dtype=int)
    validation_labels = np.array([record.label for record in validation_records], dtype=int)
    validation_texts = [record.text for record in validation_records]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    for fold, (fit_index, hold_index) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        model = train_candidate(
            name, [train_records[index] for index in fit_index],
            seed=seed, ensemble_size=ensemble_size,
        )
        choice = select_threshold(
            validation_labels, np.asarray(model.predict_proba(validation_texts))[:, 1],
            target_fpr=target_fpr, objective=objective,
        )
        hold_texts = [train_records[index].text for index in hold_index]
        scores = np.asarray(model.predict_proba(hold_texts))[:, 1]
        row = metrics_at(labels[hold_index], scores, choice["threshold"])
        rows.append({
            "fold": fold,
            "threshold": choice["threshold"],
            "count": int(len(hold_index)),
            "macro_f1": row["macro_f1"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "auc": _auc(labels[hold_index], scores),
        })

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    values = [row["macro_f1"] for row in rows]
    return {
        "folds": rows,
        "mean_macro_f1": mean("macro_f1"),
        "std_macro_f1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "mean_f1": mean("f1"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
    }


def compare_models(
    splits: dict,
    *,
    names: list[str] | None = None,
    seed: int = 42,
    target_fpr: float | None = None,
    objective: str = "macro_f1",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    include_pretrained: bool = True,
    hard_negatives: int = 0,
    cv_folds: int = 0,
    ensemble_size: int = 3,
) -> dict:
    """Train each candidate, select a threshold on validation, report on test.

    ``hard_negatives`` adds counter-speech copies of hateful training posts as
    labeled non-hateful examples (training split only). ``cv_folds`` adds a
    k-fold estimate inside the training split and ranks by its mean. Candidates
    are otherwise ranked by validation macro F1 at their own selected threshold.
    """
    if target_fpr is not None and not (isfinite(target_fpr) and 0 <= target_fpr <= 1):
        raise ValueError("Target false-positive rate must be between 0 and 1")
    if hard_negatives < 0:
        raise ValueError("Hard-negative copies must be nonnegative")
    if cv_folds and cv_folds < 2:
        raise ValueError("cv_folds must be zero (off) or at least two")
    for required in ("train", "validation", "test"):
        if not splits.get(required):
            raise ValueError(f"Comparison needs a nonempty {required} split")

    selected = list(names) if names else list(CANDIDATES)
    unknown = [name for name in selected if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")

    labels = {
        split: np.array([record.label for record in splits[split]], dtype=int)
        for split in ("train", "validation", "test")
    }
    if set(labels["train"].tolist()) != {0, 1}:
        raise ValueError("Training data must contain both labels")

    train_records = list(splits["train"])
    generated = (
        build_hard_negatives(train_records, copies=hard_negatives, seed=seed)
        if hard_negatives else []
    )
    train_records.extend(generated)

    rows, trained = [], {}
    for name in selected:
        started = time.perf_counter()
        model = train_candidate(name, train_records, seed=seed, ensemble_size=ensemble_size)
        fit_seconds = time.perf_counter() - started
        trained[name] = model
        row = {
            "name": name,
            "description": CANDIDATES[name].description,
            "kind": "trained",
            "fit_seconds": round(fit_seconds, 3),
            **_evaluate_row(
                model, splits, labels, target_fpr=target_fpr, objective=objective,
                n_resamples=n_resamples, confidence=confidence,
            ),
        }
        if cv_folds:
            started = time.perf_counter()
            row["cross_validation"] = cross_validate(
                name, train_records, splits["validation"], folds=cv_folds, seed=seed,
                target_fpr=target_fpr, objective=objective, ensemble_size=ensemble_size,
            )
            row["cross_validation"]["seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)

    if include_pretrained:
        from .pretrained import MODEL_PATH, PretrainedClassifier

        if MODEL_PATH.is_file():
            try:
                started = time.perf_counter()
                frozen = PretrainedClassifier.load(MODEL_PATH)
                fit_seconds = time.perf_counter() - started
            except (OSError, ValueError, ImportError):
                frozen = None
            if frozen is not None:
                trained["pretrained-pickle"] = frozen
                rows.append({
                    "name": "pretrained-pickle",
                    "description": "Third-party pinned TF-IDF/logistic-regression pickle",
                    "kind": "frozen",
                    "fit_seconds": round(fit_seconds, 3),
                    "cross_validation": None,
                    **_evaluate_row(
                        frozen, splits, labels, target_fpr=target_fpr, objective=objective,
                        n_resamples=n_resamples, confidence=confidence,
                    ),
                })

    def rank_key(row):
        if cv_folds and row.get("cross_validation"):
            return (
                row["cross_validation"]["mean_macro_f1"],
                row["validation"]["metrics"]["macro_f1"],
            )
        return (
            row["validation"]["metrics"]["macro_f1"],
            row["validation"]["metrics"]["f1"],
        )

    ranking = sorted(rows, key=rank_key, reverse=True)
    models = [{**row, "rank": position} for position, row in enumerate(ranking, start=1)]
    winner = models[0]

    # Calibrate the winner on validation and record what it buys on test.
    calibrated = None
    winner_model = trained.get(winner["name"])
    if hasattr(winner_model, "features"):
        from .calibration import apply_calibrator, fit_calibrator

        validation_scores = _scores(winner_model, splits["validation"])
        calibrator = fit_calibrator(labels["validation"], validation_scores, "isotonic")
        winner_model.calibrator = calibrator
        calibrated_scores = _scores(winner_model, splits["test"])
        mapped = float(apply_calibrator(
            calibrator, [winner["threshold_selection"]["threshold"]]
        )[0])
        calibrated = {
            "method": "isotonic",
            "fitted_on": "validation",
            "threshold": min(max(mapped, 0.0), 1.0),
            "before": winner["test"]["calibration"],
            "after": {
                "brier": brier_score(labels["test"], calibrated_scores),
                "ece": expected_calibration_error(labels["test"], calibrated_scores),
            },
        }

    return {
        "seed": seed,
        "threshold_rule": (
            f"false-positive rate <= {target_fpr:.3f}" if target_fpr is not None
            else f"maximum {objective} on validation"
        ),
        "ranking_metric": (
            f"mean macro F1 over {cv_folds} training-split folds" if cv_folds
            else "validation macro F1"
        ),
        "cv_folds": cv_folds or None,
        "ensemble_size": ensemble_size,
        "hard_negatives": {
            "copies_per_post": hard_negatives,
            "added": len(generated),
            "train_size_before": len(splits["train"]),
            "train_size_after": len(train_records),
        },
        "split_sizes": {name: len(splits[name]) for name in ("train", "validation", "test")},
        "models": models,
        "best": {
            "name": winner["name"],
            "threshold": winner["threshold_selection"]["threshold"],
            "validation_macro_f1": winner["validation"]["metrics"]["macro_f1"],
            "test_macro_f1": winner["test"]["metrics"]["macro_f1"],
            "test_f1": winner["test"]["metrics"]["f1"],
            "test_precision": winner["test"]["metrics"]["precision"],
            "test_recall": winner["test"]["metrics"]["recall"],
            "ranking_metric": (
                winner["cross_validation"]["mean_macro_f1"] if cv_folds
                else winner["validation"]["metrics"]["macro_f1"]
            ),
            "selection_rule": winner["threshold_selection"]["rule"],
            "saved_threshold": (
                calibrated["threshold"] if calibrated
                else winner["threshold_selection"]["threshold"]
            ),
        },
        "calibration": calibrated,
        "_models": trained,
    }


def comparison_table(report: dict) -> str:
    """Plain-text ranking table for the command line."""
    cross = bool(report.get("cv_folds"))
    primary = "cv mF1" if cross else "val mF1"
    header = (
        f"{'rank':<5}{'model':<22}{'thr':>6}{primary:>9}"
        f"{'test mF1':>10}{'test P':>8}{'test R':>8}{'test AUC':>10}{'test ECE':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in report["models"]:
        test = row["test"]
        cv = row.get("cross_validation") or {}
        first = (
            f"{cv.get('mean_macro_f1', float('nan')):>9.3f}" if cross
            else f"{row['validation']['metrics']['macro_f1']:>9.3f}"
        )
        auc = "n/a" if test["auc"] is None else f"{test['auc']:.3f}"
        lines.append(
            f"{row['rank']:<5}{row['name']:<22}"
            f"{row['threshold_selection']['threshold']:>6.2f}{first}"
            f"{test['metrics']['macro_f1']:>10.3f}"
            f"{test['metrics']['precision']:>8.3f}"
            f"{test['metrics']['recall']:>8.3f}{auc:>10}"
            f"{test['calibration']['ece']:>10.3f}"
        )
    return "\n".join(lines)
