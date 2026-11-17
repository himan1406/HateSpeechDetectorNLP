"""Experimental meme classification with frozen image features and caption TF-IDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

from .data import Record, normalize_text
from .evaluation import evaluate
from .transformer import _torch_device


DEFAULT_VISION_MODEL = "google/vit-base-patch16-224-in21k"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, kw_only=True)
class MemeRecord(Record):
    image: Path


def load_manifest(manifest: str | Path) -> dict[str, list[MemeRecord]]:
    """Read a CSV with id, text, image, label and split columns."""
    manifest = Path(manifest)
    split_records: dict[str, list[MemeRecord]] = {name: [] for name in SPLITS}
    ids: set[str] = set()
    samples: dict[tuple[str, str], str] = {}
    image_hashes: dict[Path, str] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "text", "image", "label", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("Meme manifest needs id, text, image, label and split columns.")
        for line, row in enumerate(reader, start=2):
            record_id = (row["id"] or "").strip()
            caption = normalize_text(row["text"] or "")
            image_name = (row["image"] or "").strip()
            split = (row["split"] or "").strip().lower()
            label = (row["label"] or "").strip()
            if not record_id or not caption or not image_name or split not in SPLITS or label not in {"0", "1"}:
                raise ValueError(f"Invalid meme row {line}. Each row needs text, a binary label and an explicit split.")
            if record_id in ids:
                raise ValueError(f"Duplicate meme ID on row {line}: {record_id}")
            ids.add(record_id)
            image = (manifest.parent / image_name).resolve()
            if not image.is_file():
                raise ValueError(f"Meme image on row {line} does not exist: {image_name}")
            if image not in image_hashes:
                with image.open("rb") as file:
                    image_hashes[image] = hashlib.file_digest(file, "sha256").hexdigest()
            key = (caption.casefold(), image_hashes[image])
            if key in samples:
                raise ValueError(f"Duplicate caption-image pair on row {line} (already in {samples[key]}).")
            samples[key] = split
            split_records[split].append(MemeRecord(
                id=record_id, text=caption, image=image, label=int(label), split=split,
                language=(row.get("language") or "").strip() or "unknown",
                group=(row.get("group") or "").strip(),
            ))

    for name, records in split_records.items():
        if {record.label for record in records} != {0, 1}:
            raise ValueError(f"The {name} meme split needs both binary labels.")
    return split_records


class ImageEncoder:
    def __init__(self, checkpoint: str | Path, *, device: str = "auto", local: bool = False):
        from transformers import AutoImageProcessor, AutoModel

        self.device = _torch_device(device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False, local_files_only=local)
        self.model = AutoModel.from_pretrained(checkpoint, local_files_only=local).to(self.device).eval()

    def encode(self, paths: Sequence[Path], batch_size: int) -> np.ndarray:
        import torch
        from PIL import Image, ImageOps

        cached: dict[Path, np.ndarray] = {}
        unique = list(dict.fromkeys(paths))
        with torch.inference_mode():
            for start in range(0, len(unique), batch_size):
                batch = unique[start:start + batch_size]
                images = []
                for path in batch:
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                output = self.model(**inputs).last_hidden_state
                # ViT pools image content in its class token.
                features = normalize(output[:, 0].float().cpu().numpy(), norm="l2")
                cached.update(zip(batch, features))
        return np.stack([cached[path] for path in paths])

    def save(self, path: Path) -> None:
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)


def _fuse(captions, images: np.ndarray):
    return hstack((captions, csr_matrix(images)), format="csr")


def train_multimodal(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    vision_model: str = DEFAULT_VISION_MODEL,
    batch_size: int = 8,
    epochs: int = 30,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """Train fusion, caption-only and image-only models on the same training rows."""
    if batch_size < 1 or epochs < 1:
        raise ValueError("Batch size and epochs must be positive.")
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Choose a new multimodal checkpoint directory. Already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)
    encoder = ImageEncoder(vision_model, device=device)
    all_rows = [row for split in SPLITS for row in rows[split]]
    all_image_features = encoder.encode([row.image for row in all_rows], batch_size)
    images_by_split = np.split(all_image_features, np.cumsum([len(rows[split]) for split in SPLITS[:-1]]))

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), max_features=10_000, dtype=np.float32)
    vectorizer.fit(row.text for row in rows["train"])
    features = {}
    for split, images in zip(SPLITS, images_by_split):
        captions = vectorizer.transform(row.text for row in rows[split])
        features[split] = {"fusion": _fuse(captions, images), "text_only": captions, "image_only": images}

    classifiers = {}
    report = {"vision_model": str(vision_model), "seed": seed, "epochs": epochs, "branches": {}}
    labels = [row.label for row in rows["train"]]
    for name, matrix in features["train"].items():
        classifier = MLPClassifier(hidden_layer_sizes=(64,), max_iter=epochs, random_state=seed)
        classifier.fit(matrix, labels)
        classifiers[name] = classifier
        report["branches"][name] = {
            split: evaluate(rows[split], classifier.predict_proba(features[split][name])[:, 1], threshold=0.5)
            for split in ("validation", "test")
        }

    with TemporaryDirectory(dir=path.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        checkpoint.mkdir()
        encoder.save(checkpoint / "vision")
        joblib.dump({"vectorizer": vectorizer, "classifiers": classifiers}, checkpoint / "fusion.joblib")
        (checkpoint / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        checkpoint.rename(path)
    return report


def predict_meme(
    model_dir: str | Path, text: str, image: str | Path, *, device: str = "auto",
) -> dict[str, float]:
    """Score one image-caption pair using a trusted local model artifact."""
    caption = normalize_text(text)
    if not caption:
        raise ValueError("A meme caption is required.")
    path = Path(model_dir)
    artifact = joblib.load(path / "fusion.joblib")
    encoder = ImageEncoder(path / "vision", device=device, local=True)
    embeddings = encoder.encode([Path(image)], batch_size=1)
    captions = artifact["vectorizer"].transform([caption])
    classifier = artifact["classifiers"]["fusion"]
    probabilities = classifier.predict_proba(_fuse(captions, embeddings))[0]
    return {"non-hateful": float(probabilities[0]), "hateful": float(probabilities[1])}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train fusion and both single-modality comparisons")
    train.add_argument("--manifest", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto")
    predict = commands.add_parser("predict", help="Score a caption and local image")
    predict.add_argument("--model", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "train":
        report = train_multimodal(
            args.manifest, args.out, vision_model=args.vision_model,
            batch_size=args.batch_size, epochs=args.epochs, seed=args.seed, device=args.device,
        )
    else:
        report = predict_meme(args.model, args.text, args.image, device=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
