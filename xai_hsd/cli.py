"""Command-line entry points for supervised training and controlled feedback experiments."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .baseline import Baseline, train_baseline
from .data import load_records, normalize_text, split_records
from .evaluation import evaluate
from .feedback import training_preferences, tune_baseline
from .pretrained import MODEL_PATH, PretrainedClassifier, fetch_model

LABELS = ("Non-hateful", "Hateful")


def load_model(path, backend, device="auto"):
    if backend == "pretrained":
        return PretrainedClassifier.load(path)
    if backend == "baseline":
        return Baseline.load(path)

    from .transformer import TextTransformer

    return TextTransformer.load(path, device=device)


def _report(model, splits, threshold):
    report = {}
    for name in ("validation", "test"):
        records = splits[name]
        scores = model.predict_proba([record.text for record in records])[:, 1]
        report[name] = evaluate(records, scores, threshold=threshold)
    return report


def _signature(splits):
    signature = sha256()
    for name in ("train", "validation", "test"):
        for record in sorted(splits[name], key=lambda item: item.id):
            row = json.dumps([name, record.id, record.text, record.label], ensure_ascii=False)
            signature.update((row + "\n").encode("utf-8"))
    return signature.hexdigest()


def _training_manifest(path):
    return path.with_name(path.name + ".splits.json")


def _write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _train(args, splits):
    if args.backend == "baseline":
        model = train_baseline(splits["train"], seed=args.seed)
        model.save(args.output)
        return model

    from .transformer import train_transformer

    options = {"model_name": args.checkpoint} if args.checkpoint else {}
    return train_transformer(
        splits["train"], splits["validation"], args.output,
        epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length,
        seed=args.seed, device=args.device, **options,
    )


def _tune(args, splits):
    if args.model.resolve() == args.output.resolve():
        raise ValueError("Choose an output path separate from the reference model.")
    manifest = _training_manifest(args.model)
    if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8"))["sha256"] != _signature(splits):
        raise ValueError("Use the same dataset and split settings that trained the reference model.")
    preferences, skipped = training_preferences(
        args.feedback, held_out=splits["validation"] + splits["test"],
    )
    if not preferences:
        raise ValueError("No eligible reviews were found. Save moderator decisions in the app first.")
    reference = load_model(args.model, args.backend, args.device)
    if args.backend == "baseline":
        model = tune_baseline(reference, preferences, beta=args.beta or 1.0)
        model.save(args.output)
    else:
        from .transformer import reward_tune

        model = reward_tune(
            args.model, preferences, args.output, epochs=args.epochs, batch_size=args.batch_size,
            beta=args.beta or 0.1, seed=args.seed, device=args.device,
        )
    _write_json(_training_manifest(args.output), {"sha256": _signature(splits)})
    return {
        "eligible_reviews": len(preferences),
        "excluded_reviews": skipped,
        "reference": _report(reference, splits, args.threshold),
        "candidate": _report(model, splits, args.threshold),
    }


def _predict(args):
    text = normalize_text(args.text)
    if not text:
        raise ValueError("Enter a post containing text.")
    model = load_model(args.model, args.backend, args.device)
    score = float(model.predict_proba([text])[0, 1])
    label = int(score >= args.threshold)
    result = {"label": LABELS[label], "hate_score": score, "threshold": args.threshold}
    if isinstance(model, PretrainedClassifier):
        result["has_known_features"] = model.has_known_features(text)
    if args.explain:
        from .explain import explain

        result["explanation_method"] = args.explain
        result["analyzed_text"] = text
        result["attributions"] = [
            asdict(item) for item in explain(text, model.predict_proba, args.explain, label)
        ]
    return result


def _rationale_audit(model, records, method, limit, seed):
    import random

    from .explain import explain, rationale_overlap

    annotated = [record for record in records if record.label == 1 and any(record.rationales or [])]
    selected = random.Random(seed).sample(annotated, min(len(annotated), limit))
    results = []
    for record in selected:
        attribution = explain(record.text, model.predict_proba, method, 1, seed=seed)
        outcome = rationale_overlap(attribution, record.rationales)
        if outcome is not None:
            results.append(outcome)
    return {
        "method": method,
        "class": "Hateful",
        "annotated_test_posts": len(annotated),
        "evaluated_posts": len(results),
        "mean_overlap": {
            key: sum(result[key] for result in results) / len(results)
            for key in ("precision", "recall", "f1", "iou")
        } if results else None,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-model", help="Download the pinned pretrained pickle.")
    fetch.add_argument("--output", type=Path, default=MODEL_PATH)
    for name in ("train", "evaluate", "predict", "tune"):
        command = commands.add_parser(name)
        inference = name in ("predict", "evaluate")
        backends = ("pretrained", "baseline", "transformer") if inference else ("baseline", "transformer")
        command.add_argument("--backend", choices=backends, default=backends[0])
        command.add_argument("--device", default="auto")
        command.add_argument("--threshold", type=float, default=0.5)
        if name in ("train", "evaluate", "tune"):
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--source", choices=("csv", "hatexplain", "davidson", "hasoc"), default="csv")
            command.add_argument("--splits", type=Path)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--report", type=Path)
        if name != "train":
            command.add_argument("--model", type=Path, required=name == "tune")
        if name in ("train", "tune"):
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--epochs", type=int, default=3 if name == "train" else 1)
            command.add_argument("--batch-size", type=int, default=8)
        if name == "train":
            command.add_argument("--checkpoint")
            command.add_argument("--max-length", type=int, default=192)
        if name == "evaluate":
            command.add_argument("--rationale-method", choices=("shap", "lime"), default="shap")
            command.add_argument("--rationale-limit", type=int, default=0)
        if name == "tune":
            command.add_argument("--feedback", type=Path, default=Path("artifacts/reviews.sqlite3"))
            command.add_argument("--beta", type=float)
        if name == "predict":
            command.add_argument("text")
            command.add_argument("--explain", choices=("shap", "lime"))
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.command == "fetch-model":
        try:
            print(fetch_model(args.output).resolve())
        except (ValueError, OSError) as error:
            parser.exit(2, f"Error: {error}\n")
        return
    if args.command in ("predict", "evaluate") and args.model is None:
        if args.backend != "pretrained":
            parser.error("--model is required for baseline and transformer inference.")
        args.model = MODEL_PATH
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.command == "tune" and args.beta is not None and not 0 < args.beta < float("inf"):
        parser.error("--beta must be positive and finite.")
    if args.command == "evaluate" and args.rationale_limit < 0:
        parser.error("--rationale-limit must be nonnegative.")
    try:
        if args.command == "predict":
            result = _predict(args)
        else:
            records = load_records(args.data, source=args.source, splits_path=args.splits)
            splits = split_records(records, seed=args.seed)
            metadata = {
                "source": args.source,
                "data_file": args.data.name,
                "backend": args.backend,
                "seed": args.seed,
                "threshold": args.threshold,
                "split_sizes": {name: len(rows) for name, rows in splits.items()},
                "split_sha256": _signature(splits),
            }
            if args.command == "tune":
                result = {**metadata, **_tune(args, splits)}
            else:
                if args.command == "train":
                    model = _train(args, splits)
                    _write_json(_training_manifest(args.output), {"sha256": metadata["split_sha256"]})
                else:
                    model = load_model(args.model, args.backend, args.device)
                result = {**metadata, **_report(model, splits, args.threshold)}
                if args.command == "evaluate" and args.rationale_limit:
                    result["rationales"] = _rationale_audit(
                        model, splits["test"], args.rationale_method, args.rationale_limit, args.seed,
                    )
            if args.report:
                _write_json(args.report, result)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
