"""Multilingual supervised training and offline binary-label preference tuning."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Sequence

import numpy as np
from sklearn.metrics import f1_score

from .data import normalize_text

if TYPE_CHECKING:
    from .data import Record


DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"
LABELS = {0: "non-hateful", 1: "hateful"}
METADATA = "xai_hsd_training.json"


def _torch_device(device: str):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class TextTransformer:
    """Adapts a saved transformer to the project probability interface."""

    def __init__(self, model, tokenizer, *, max_length: int, batch_size: int, device: str):
        if max_length < 2 or batch_size < 1:
            raise ValueError("max_length must be at least 2 and batch_size must be positive.")
        if model.config.id2label != LABELS:
            raise ValueError("TextTransformer expects the project binary class order.")
        self.device = _torch_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def load(cls, model_dir: str | Path, *, device: str = "auto", batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / METADATA).read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        return cls(
            model, tokenizer, max_length=metadata["max_length"], batch_size=batch_size, device=device,
        )

    def _logits(self, texts: Sequence[str]):
        batch = self.tokenizer(
            [normalize_text(text) for text in texts], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return self.model(**batch.to(self.device)).logits

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.inference_mode():
            chunks = [
                self._logits(texts[start:start + self.batch_size]).float().softmax(dim=1).cpu().numpy()
                for start in range(0, len(texts), self.batch_size)
            ]
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.float32)

    def _save(self, directory: Path, metadata: dict) -> None:
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / METADATA).write_text(
            json.dumps({"max_length": self.max_length, **metadata}, indent=2) + "\n",
            encoding="utf-8",
        )


def _check_training(output_dir: str | Path, epochs: int, batch_size: int, learning_rate: float) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Choose a new checkpoint directory. Already exists: {path}")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def train_transformer(
    train_records: Sequence[Record],
    validation_records: Sequence[Record],
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 192,
    learning_rate: float = 2e-5,
    seed: int = 42,
    device: str = "auto",
) -> TextTransformer:
    """Fit a binary classification head and encoder, selecting by validation macro F1."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = _check_training(output_dir, epochs, batch_size, learning_rate)
    train_labels = np.array([row.label for row in train_records])
    validation_labels = np.array([row.label for row in validation_records])
    if set(train_labels) != {0, 1} or set(validation_labels) != {0, 1}:
        raise ValueError("Training and validation must each contain both binary labels.")
    train_texts = [row.text for row in train_records]
    validation_texts = [row.text for row in validation_records]
    train_keys = {normalize_text(text).casefold() for text in train_texts}
    if any(normalize_text(text).casefold() in train_keys for text in validation_texts):
        raise ValueError("Training and validation overlap after text normalization.")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=LABELS, label2id={name: key for key, name in LABELS.items()},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = TextTransformer(
        model, tokenizer, max_length=max_length, batch_size=batch_size, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_f1 = -1.0

    with TemporaryDirectory(dir=path.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        for epoch in range(1, epochs + 1):
            model.train()
            indices = rng.permutation(len(train_texts))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                labels = torch.as_tensor(train_labels[batch], dtype=torch.long, device=classifier.device)
                logits = classifier._logits([train_texts[index] for index in batch])
                loss = torch.nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            scores = classifier.predict_proba(validation_texts)[:, 1]
            validation_f1 = f1_score(validation_labels, scores >= 0.5, average="macro", zero_division=0)
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                classifier._save(checkpoint, {
                    "objective": "cross_entropy", "base_model": str(model_name),
                    "epoch": epoch, "validation_macro_f1": float(best_f1), "seed": seed,
                })
        checkpoint.rename(path)

    del optimizer, model, classifier
    return TextTransformer.load(path, device=device, batch_size=batch_size)


def reward_tune(
    reference_dir: str | Path,
    reviews: Sequence[tuple[str, int]],
    output_dir: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    seed: int = 42,
    device: str = "auto",
) -> TextTransformer:
    """Apply DPO to the reviewed class versus the other class for each post."""
    import torch

    path = _check_training(output_dir, epochs, batch_size, learning_rate)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be positive.")
    unique: dict[str, tuple[str, int]] = {}
    for text, label in reviews:
        normalized = normalize_text(text)
        if not normalized or label not in (0, 1):
            raise ValueError("Each review requires usable text and a binary label.")
        key = normalized.casefold()
        previous = unique.setdefault(key, (normalized, label))
        if previous[1] != label:
            raise ValueError("Resolve conflicting reviewed labels before reward tuning.")
    if not unique:
        raise ValueError("At least one eligible review is required for reward tuning.")
    texts, labels = zip(*unique.values())

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classifier = TextTransformer.load(reference_dir, device=device, batch_size=batch_size)
    # The log-probability ratio between the two labels equals their logit difference.
    # Store frozen reference ratios before the first optimizer step.
    reference = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            logits = classifier._logits(texts[start:start + batch_size])
            reference.append((logits[:, 1] - logits[:, 0]).detach())
    reference_log_odds = torch.cat(reference)
    preference_sign = torch.tensor(labels, device=classifier.device, dtype=torch.float32) * 2 - 1

    model = classifier.model
    # Keep dropout disabled for both policies. Gradients still flow through the active model.
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        indices = rng.permutation(len(texts))
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            batch_index = torch.as_tensor(batch, device=classifier.device)
            logits = classifier._logits([texts[index] for index in batch])
            log_odds = logits[:, 1] - logits[:, 0]
            preference_gain = preference_sign[batch_index] * (log_odds - reference_log_odds[batch_index])
            loss = -torch.nn.functional.logsigmoid(beta * preference_gain).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(batch)
        losses.append(total_loss / len(texts))

    with TemporaryDirectory(dir=path.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        classifier._save(checkpoint, {
            "objective": "binary_dpo", "reference": str(reference_dir), "review_count": len(texts),
            "epochs": epochs, "mean_loss_per_epoch": losses, "beta": beta, "seed": seed,
        })
        checkpoint.rename(path)
    return classifier
