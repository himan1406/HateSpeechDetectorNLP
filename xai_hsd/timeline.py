"""Shared plumbing for read-only platform timeline sources.

Each platform module (``reddit.py``, ``bluesky.py``) parses source records into
a small frozen dataclass with ``text``, ``created_utc``, ``id`` and a ``url``
property, then reuses :class:`CachedTimeline` for a TTL disk cache so a repeated
assessment does not hit the platform again.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REMOVED = {"[deleted]", "[removed]", "[unavailable]", ""}
DEFAULT_CACHE = Path("data/raw")


def default_transport(method, url, headers, data=None, timeout=20):
    body = urllib.parse.urlencode(data).encode("utf-8") if data else None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


def clean_text(value: str) -> str:
    text = " ".join((value or "").split())
    return "" if text in REMOVED else text


def iso_to_epoch(value: str) -> float:
    """Convert an ISO-8601 timestamp to a UTC epoch, returning 0 when unusable."""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class CachedTimeline:
    """Disk cache shared by the platform timeline clients."""

    source = ""
    cache_group = ""
    post_type = None

    def _cache_path(self, username: str, kind: str) -> Path | None:
        if self.cache_dir is None:
            return None
        folder = self.cache_dir / username.lower()
        if self.cache_group:
            folder = folder / self.cache_group
        return folder / f"{kind}.json"

    def _read_cache(self, path: Path | None, *, refresh: bool) -> list | None:
        if path is None or refresh or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if self._clock() - float(payload["fetched_at"]) > self.ttl_seconds:
                return None
            return [self.post_type(**item) for item in payload["posts"]]
        except (KeyError, TypeError, ValueError, OSError):
            return None

    def _write_cache(self, path: Path | None, posts: list) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"fetched_at": self._clock(), "posts": [asdict(post) for post in posts]},
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
