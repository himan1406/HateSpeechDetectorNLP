"""Read-only Bluesky timeline collection over the public AT Protocol API.

Bluesky serves ``app.bsky.feed.getAuthorFeed`` from its AppView at
``public.api.bsky.app`` without authentication, so no key, app password, or
login is needed. The feed returns the author's own posts and replies together
with reposts of other people's posts; reposts are dropped because the words are
not the author's. Pinned posts are excluded so the timeline stays chronological.
Only public data is read, and every response is cached.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .timeline import CachedTimeline, clean_text, default_transport, iso_to_epoch

PUBLIC_ROOT = "https://public.api.bsky.app/xrpc"
DEFAULT_CACHE = Path("data/raw/bluesky")
PAGE_LIMIT = 100
AGENT = "xai-hsd/0.1 (hate-speech research)"
PROFILE_URL = "https://bsky.app/profile/"
POST_URI_PATTERN = re.compile(r"/app\.bsky\.feed\.post/([A-Za-z0-9]+)$")
ACTOR_PATTERN = re.compile(
    r"^(?:did:[a-z0-9]+:[A-Za-z0-9._:%-]+"
    r"|[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)$"
)


class BlueskyError(RuntimeError):
    """Raised for unknown handles, refused feeds, or rejected requests."""


@dataclass(frozen=True)
class BlueskyPost:
    id: str
    kind: str
    text: str
    created_utc: float
    permalink: str = ""
    handle: str = ""

    @property
    def url(self) -> str:
        return PROFILE_URL + self.permalink if self.permalink else ""


def normalize_actor(value: str) -> str:
    """Accept a handle, ``@handle``, profile URL, or DID and return the actor."""
    name = (value or "").strip()
    if name.lower().startswith(PROFILE_URL):
        name = name[len(PROFILE_URL):]
    name = name.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    name = name.split("/", 1)[0].removeprefix("@")
    if not ACTOR_PATTERN.match(name):
        raise BlueskyError(
            "Enter a Bluesky handle such as 'name.bsky.social', or a did: DID."
        )
    return name


def _error_message(payload: bytes) -> str:
    try:
        body = json.loads(payload.decode("utf-8"))
    except (ValueError, AttributeError):
        return "no details"
    return str(body.get("message") or body.get("error") or "no details")


def post_from_feed_item(item: dict) -> BlueskyPost | None:
    """Convert one feed item into the author's own post, dropping reposts."""
    if not isinstance(item, dict) or item.get("reason"):
        return None
    post = item.get("post") or {}
    record = post.get("record") or {}
    if record.get("$type") != "app.bsky.feed.post":
        return None
    text = clean_text(record.get("text", ""))
    match = POST_URI_PATTERN.search(str(post.get("uri") or ""))
    if not text or not match:
        return None
    handle = str((post.get("author") or {}).get("handle") or "")
    return BlueskyPost(
        id=match.group(1),
        kind="reply" if record.get("reply") else "post",
        text=text,
        created_utc=iso_to_epoch(record.get("createdAt", "")),
        permalink=f"{handle}/post/{match.group(1)}" if handle else "",
        handle=handle,
    )


class BlueskyClient(CachedTimeline):
    """Keyless timeline client for the public Bluesky AppView.

    Pagination follows the ``cursor`` returned with each page. Replies count as
    the author's own text; reposts and image-only posts are skipped.
    """

    source = "bluesky public appview"
    post_type = BlueskyPost

    def __init__(
        self,
        *,
        timeout: int = 30,
        cache_dir: str | Path | None = DEFAULT_CACHE,
        ttl_seconds: int = 21_600,
        transport=None,
        clock=time.time,
        sleeper=time.sleep,
        user_agent: str = AGENT,
    ):
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds
        self._transport = transport or default_transport
        self._clock = clock
        self._sleeper = sleeper
        self.user_agent = user_agent

    def _request(self, path: str, params: dict, *, attempts: int = 4) -> dict:
        query = urllib.parse.urlencode(params)
        for attempt in range(attempts):
            status, _, payload = self._transport(
                "GET", f"{PUBLIC_ROOT}{path}?{query}",
                {"User-Agent": self.user_agent, "Accept": "application/json"},
                None, self.timeout,
            )
            if status == 200:
                try:
                    return json.loads(payload.decode("utf-8"))
                except ValueError as error:
                    raise BlueskyError("Bluesky returned malformed JSON.") from error
            detail = _error_message(payload)
            if status == 429 and attempt + 1 < attempts:
                self._sleeper(min(5.0 * (attempt + 1), 30.0))
                continue
            if status in (500, 502, 503, 504) and attempt + 1 < attempts:
                self._sleeper(min(2.0 * (attempt + 1), 10.0))
                continue
            if status == 429:
                raise BlueskyError("Bluesky is rate limiting this address; retry in a minute.")
            if status in (400, 404):
                raise BlueskyError(
                    f"Bluesky rejected the account lookup (HTTP {status}: {detail})."
                )
            if status in (401, 403):
                raise BlueskyError(
                    f"Bluesky refused the feed (HTTP {status}: {detail}). The account "
                    "may be blocked, deactivated, private, or taken down."
                )
            raise BlueskyError(f"The Bluesky request failed with HTTP {status}: {detail}")
        raise BlueskyError(f"Bluesky kept failing after {attempts} attempts.")

    def timeline(
        self,
        username: str,
        *,
        kinds: tuple[str, ...] = ("posts",),
        limit: int = 200,
        refresh: bool = False,
    ) -> list[BlueskyPost]:
        """Fetch an account's public posts and replies, newest first."""
        actor = normalize_actor(username)
        if limit < 1:
            raise ValueError("Limit must be positive")
        if tuple(kinds) != ("posts",):
            raise ValueError("Bluesky exposes one feed; use kinds=('posts',)")

        path = self._cache_path(actor, "posts")
        cached = self._read_cache(path, refresh=refresh)
        if cached is not None:
            return cached[:limit]

        posts: list[BlueskyPost] = []
        seen: set[str] = set()
        cursor: str | None = None
        while len(posts) < limit:
            params = {
                "actor": actor,
                "limit": min(PAGE_LIMIT, limit - len(posts)),
                "filter": "posts_with_replies",
                "includePins": "false",
            }
            if cursor:
                params["cursor"] = cursor
            body = self._request("/app.bsky.feed.getAuthorFeed", params)
            page = body.get("feed") or []
            for post in (post_from_feed_item(item) for item in page):
                if post and post.id not in seen:
                    seen.add(post.id)
                    posts.append(post)
            next_cursor = body.get("cursor")
            if not page or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        posts = posts[:limit]
        self._write_cache(path, posts)
        return posts
