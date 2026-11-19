"""Reproducible download of a small, split-preserving HateXplain fixture.

The full corpus is 19k posts. Training several models on all of it is slow on a
laptop, so this keeps a stratified sample of each official split. The fixture is
downloaded rather than committed: it is a few megabytes of harmful language.
"""

from __future__ import annotations

import collections
import csv
import json
import random
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/"
    "01d742279dac941981f53806154481c0e15ee686/Data/"
)
FILES = ("dataset.json", "post_id_divisions.json")
FIXTURE = Path("data/hatexplain-6k.csv")
SPLIT_TARGETS = {"train": 4000, "validation": 1000, "test": 1000}
FIXTURE_SEED = 20261119


def _download(name: str, directory: Path) -> Path:
    path = directory / name
    if path.is_file():
        return path
    request = urllib.request.Request(
        BASE_URL + name, headers={"User-Agent": "xai-hsd/0.1 (research fixture)"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def fetch_hatexplain(directory: str | Path = Path("data")) -> tuple[Path, Path]:
    """Download the corpus and its published split divisions, if absent."""
    directory = Path(directory)
    dataset = _download(FILES[0], directory)
    divisions = _download(FILES[1], directory)
    return dataset, divisions


def build_fixture(
    dataset: str | Path,
    divisions: str | Path,
    output: str | Path = FIXTURE,
    *,
    seed: int = FIXTURE_SEED,
    targets: dict[str, int] | None = None,
) -> Path:
    """Write a stratified sample that keeps the official train/val/test split."""
    from .data import load_records

    targets = targets or SPLIT_TARGETS
    records = load_records(dataset, source="hatexplain", splits_path=divisions)
    buckets: dict[tuple[str, int], list] = collections.defaultdict(list)
    for record in records:
        buckets[(record.split, record.label)].append(record)
    for rows in buckets.values():
        rows.sort(key=lambda record: record.id)

    rng = random.Random(seed)
    chosen = []
    for split, target in targets.items():
        total = sum(len(rows) for (name, _), rows in buckets.items() if name == split)
        if not total:
            raise ValueError(f"The dataset has no {split} records")
        for label in (0, 1):
            rows = buckets.get((split, label), [])
            size = min(len(rows), max(1, round(target * len(rows) / total)))
            chosen.extend(rng.sample(rows, size))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "label", "language", "group", "rationales", "split"])
        for record in sorted(chosen, key=lambda item: (item.split, item.id)):
            writer.writerow([
                record.id, record.text, record.label, record.language, record.group,
                json.dumps(record.rationales) if record.rationales else "", record.split,
            ])
    return output


def fetch_fixture(
    output: str | Path = FIXTURE,
    *,
    directory: str | Path = Path("data"),
    seed: int = FIXTURE_SEED,
    force: bool = False,
) -> Path:
    """Download the corpus if needed and build the fixture unless it exists."""
    output = Path(output)
    if output.is_file() and not force:
        return output
    dataset, divisions = fetch_hatexplain(directory)
    return build_fixture(dataset, divisions, output, seed=seed)
