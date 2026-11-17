"""Dataset adapters and leakage-controlled splits for binary hate speech detection."""

from __future__ import annotations

import csv
import html
import json
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_SPLITS = ("train", "validation", "test")
_SPLIT_NAMES = {"val": "validation", "valid": "validation", "dev": "validation"}


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    label: int
    language: str = "unknown"
    group: str = ""
    rationales: list[int] | None = None
    split: str | None = None


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving casing, emoji and punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _split_name(value: str | None) -> str | None:
    name = (value or "").strip().lower()
    if not name:
        return None
    name = _SPLIT_NAMES.get(name, name)
    if name not in _SPLITS:
        raise ValueError(f"Unknown split name {value!r}")
    return name


def _rationale_mask(encoded: str, size: int) -> list[int] | None:
    if not encoded.strip():
        return None
    mask = json.loads(encoded)
    if (
        not isinstance(mask, list)
        or len(mask) != size
        or any(type(item) is not int or item not in (0, 1) for item in mask)
    ):
        raise ValueError("Rationales must be a JSON binary mask aligned to whitespace tokens")
    return mask


def _binary_label(value: str) -> int:
    if value.strip() not in {"0", "1"}:
        raise ValueError("Canonical labels must be 0 (non-hateful) or 1 (hateful)")
    return int(value)


def _table_records(path: Path, source: str) -> list[Record]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        table = csv.DictReader(handle, delimiter=delimiter)
        fields = set(table.fieldnames or ())
        required = {"tweet", "class"} if source == "davidson" else {"text"}
        if source == "csv":
            required.add("label")
        if not required <= fields:
            raise ValueError(f"Missing required columns: {', '.join(sorted(required - fields))}")
        if source == "hasoc" and not {"label", "task_1", "task_2"} & fields:
            raise ValueError("HASOC requires a mapped label or task_1/task_2 columns")

        records = []
        for row in table:
            if None in row or any(row.get(name) is None for name in required):
                raise ValueError(f"Malformed table record ending at line {table.line_num}")
            try:
                record = _table_record(row, source, table.line_num)
            except ValueError as error:
                raise ValueError(f"Line {table.line_num}: {error}") from error
            if record is not None:
                records.append(record)
        return records


def _table_record(row: dict, source: str, line: int) -> Record | None:
    language = (row.get("language") or "").strip().lower()
    identifier = (row.get("id") or row.get("tweet_id") or row.get("text_id") or "").strip()
    if source == "davidson":
        label_code = row["class"].strip()
        if label_code not in {"0", "1", "2"}:
            raise ValueError("Davidson class must be 0, 1 or 2")
        label = int(label_code == "0")
        text = normalize_text(html.unescape(row["tweet"]))
        language = language or "en"
        identifier = identifier or (row.get("") or row.get("Unnamed: 0") or "").strip()
    else:
        label = _hasoc_label(row) if source == "hasoc" else _binary_label(row["label"])
        text = normalize_text(row["text"])
    if not text:
        return None
    return Record(
        id=identifier or f"row-{line}",
        text=text,
        label=label,
        language=language or "unknown",
        group=(row.get("group") or "").strip(),
        rationales=_rationale_mask(row.get("rationales") or "", len(text.split())),
        split=_split_name(row.get("split")),
    )


def _hasoc_label(row: dict) -> int:
    fine = (row.get("task_2") or "").strip().upper()
    coarse = (row.get("task_1") or "").strip().upper()
    mapped = (row.get("label") or "").strip().upper()
    code = mapped or fine
    if code == "HATE":
        return 1
    if code in {"NOT", "OFFN", "PRFN", "OFFENSIVE", "PROFANITY"}:
        return 0
    if mapped in {"0", "1"}:
        return int(mapped)
    if coarse == "NOT" and code in {"", "NONE", "NULL"}:
        return 0
    raise ValueError(
        "HASOC hate-only mapping requires HATE, NOT, OFFN or PRFN labels, "
        "or a binary label column. HOF combines hate and offensive content."
    )


def _published_splits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("HateXplain divisions must be a JSON object")
    splits = {}
    for key, identifiers in data.items():
        name = _split_name(key)
        if name is None or not isinstance(identifiers, list):
            raise ValueError("HateXplain divisions must map split names to ID lists")
        for identifier in map(str, identifiers):
            if identifier in splits:
                raise ValueError(f"Repeated ID {identifier!r} in HateXplain divisions")
            splits[identifier] = name
    return splits


def _hatexplain_records(path: Path, splits_path: Path | None) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("HateXplain dataset must be a JSON object keyed by post ID")
    published = _published_splits(splits_path) if splits_path is not None else None
    if published is not None and not published.keys() <= data.keys():
        raise ValueError("HateXplain divisions reference post IDs missing from this dataset")
    records = []
    for identifier, row in data.items():
        if published is not None and identifier not in published:
            continue
        try:
            record = _hatexplain_record(
                identifier, row, published[identifier] if published is not None else None
            )
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"Invalid HateXplain record {identifier!r}: {error}") from error
        if record is not None:
            records.append(record)
    return records


def _hatexplain_record(identifier: str, row: dict, split: str | None) -> Record | None:
    tokens = row["post_tokens"]
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("post_tokens must be a string list")
    token_parts = [normalize_text(token).split() for token in tokens]
    text = " ".join(part for parts in token_parts for part in parts)
    if not text:
        return None
    annotators = row["annotators"]
    votes = Counter(annotation["label"] for annotation in annotators)
    if not votes or not votes.keys() <= {"hatespeech", "offensive", "normal"}:
        raise ValueError("Invalid annotator labels")
    majority, count = votes.most_common(1)[0]
    if count <= len(annotators) / 2:
        return None

    groups = sorted(
        target.strip()
        for target in {
            target
            for annotation in annotators
            for target in annotation["target"]
            if target.strip().lower() not in {"", "none"}
        }
    )
    label = int(majority == "hatespeech")
    valid_masks = [
        mask for mask in row.get("rationales", [])
        if len(mask) == len(tokens) and all(item in (0, 1) for item in mask)
    ]
    rationale = (
        [max(marks) for marks, parts in zip(zip(*valid_masks), token_parts) for _ in parts]
        if label and valid_masks else None
    )
    return Record(identifier, text, label, "en", "|".join(groups), rationale, split)


def load_records(
    path: str | Path, source: str = "csv", splits_path: str | Path | None = None
) -> list[Record]:
    """Read a canonical table or a supported benchmark in its native format."""
    path = Path(path)
    if source == "hatexplain":
        sibling = path.with_name("post_id_divisions.json")
        divisions = Path(splits_path) if splits_path is not None else None
        if divisions is None and sibling.is_file():
            divisions = sibling
        records = _hatexplain_records(path, divisions)
    elif source in {"csv", "davidson", "hasoc"}:
        if splits_path is not None:
            raise ValueError("The divisions file is specific to HateXplain")
        records = _table_records(path, source)
    else:
        raise ValueError(f"Unknown dataset source {source!r}")
    if not records:
        raise ValueError("Dataset contains no usable records")
    return records


def split_records(records: Iterable[Record], seed: int = 42) -> dict[str, list[Record]]:
    """Preserve supplied splits or allocate 80/10/10 per class after deduplication.

    On existing splits, keep a duplicate in the most held-out split first.
    On unsplit corpora, omit exact duplicate text with conflicting labels.
    """
    records = list(records)
    if not records:
        raise ValueError("Dataset contains no records")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Dataset contains repeated record IDs")
    labeled_splits = [_split_name(record.split) for record in records]
    if any(labeled_splits) and not all(labeled_splits):
        raise ValueError("Split must be assigned for every record or left blank for every record")

    partitions: dict[str, list[Record]] = {key: [] for key in _SPLITS}
    buckets: dict[str, list[tuple[Record, str | None]]] = {}
    for record, split in zip(records, labeled_splits):
        key = normalize_text(record.text).casefold()
        if not key or record.label not in (0, 1):
            raise ValueError("Every record needs nonempty text and a binary label")
        buckets.setdefault(key, []).append((record, split))

    if all(labeled_splits):
        priority = {key: index for index, key in enumerate(_SPLITS)}
        for duplicates in buckets.values():
            record, split = max(duplicates, key=lambda row: priority[row[1]])
            partitions[split].append(record)
        if any(not part for part in partitions.values()):
            raise ValueError("All three supplied splits must contain usable, distinct posts")
        return partitions

    rng = random.Random(seed)
    by_label: dict[int, list[Record]] = {0: [], 1: []}
    for duplicates in buckets.values():
        record = duplicates[0][0]
        if all(item.label == record.label for item, _ in duplicates):
            by_label[record.label].append(record)
    for same_class in by_label.values():
        same_class.sort(key=lambda record: record.id)
        rng.shuffle(same_class)
        if len(same_class) < 3:
            raise ValueError("At least three distinct, consistently labeled posts per class are needed")
        held_out = max(1, round(len(same_class) * 0.1))
        partitions["test"].extend(same_class[:held_out])
        partitions["validation"].extend(same_class[held_out:2 * held_out])
        partitions["train"].extend(same_class[2 * held_out:])
    for part in partitions.values():
        rng.shuffle(part)
    return partitions
