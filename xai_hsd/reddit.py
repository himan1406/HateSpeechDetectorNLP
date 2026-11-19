"""Read-only Reddit timeline collection.

Two sources are available. ``RedditClient`` uses the official OAuth API with the
client-credentials flow of a registered "script" application (export
``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET``, ``REDDIT_USER_AGENT``). Reddit
stopped accepting new app registrations on 31 October 2026, so
``ArchiveClient`` is the default: it reads the same public submissions and
comments from the Arctic Shift archive, which needs no key and no registration.

The archive lags roughly 36 hours and is volunteer-run with no uptime guarantee.
Only public data is read, and every response is cached so a repeated assessment
does not hit the source again.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .timeline import CachedTimeline as _CachedTimeline
from .timeline import clean_text as _clean_text
from .timeline import default_transport as _default_transport

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_ROOT = "https://oauth.reddit.com"
ARCHIVE_ROOT = "https://arctic-shift.photon-reddit.com/api"
ARCHIVE_PAGE = 100
ARCHIVE_AGENT = "xai-hsd/0.1 (hate-speech research)"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,20}$")
DEFAULT_CACHE = Path("data/raw/reddit")


class RedditError(RuntimeError):
    """Raised for missing credentials, rejected requests, or blocked accounts."""


@dataclass(frozen=True)
class RedditPost:
    id: str
    kind: str
    text: str
    created_utc: float
    permalink: str
    subreddit: str = ""
    score: int | None = None

    @property
    def url(self) -> str:
        return f"https://www.reddit.com{self.permalink}" if self.permalink else ""


def post_from_listing(child: dict) -> RedditPost | None:
    """Convert one listing child into a post, dropping deleted or empty items."""
    kind = {"t3": "submission", "t1": "comment"}.get(child.get("kind"))
    data = child.get("data") or {}
    if kind is None or not data.get("id"):
        return None
    if kind == "submission":
        title = _clean_text(data.get("title", ""))
        body = _clean_text(data.get("selftext", ""))
        text = " ".join(part for part in (title, body) if part)
    else:
        text = _clean_text(data.get("body", ""))
    if not text:
        return None
    return RedditPost(
        id=str(data["id"]),
        kind=kind,
        text=text,
        created_utc=float(data.get("created_utc") or 0.0),
        permalink=str(data.get("permalink") or ""),
        subreddit=str(data.get("subreddit") or ""),
        score=int(data["score"]) if isinstance(data.get("score"), int) else None,
    )


def parse_listing(payload: dict) -> tuple[list[RedditPost], str | None]:
    """Return the posts in one listing page and the pagination cursor."""
    children = (payload.get("data") or {}).get("children") or []
    posts = [post for post in (post_from_listing(child) for child in children) if post]
    return posts, (payload.get("data") or {}).get("after")


def post_from_archive(item: dict, kind: str) -> RedditPost | None:
    """Convert one Arctic Shift record into a post, dropping empty items."""
    if not kind or not item.get("id"):
        return None
    if kind == "submission":
        title = _clean_text(item.get("title", ""))
        body = _clean_text(item.get("selftext", ""))
        text = " ".join(part for part in (title, body) if part)
    elif kind == "comment":
        text = _clean_text(item.get("body", ""))
    else:
        return None
    if not text:
        return None
    return RedditPost(
        id=str(item["id"]),
        kind=kind,
        text=text,
        created_utc=float(item.get("created_utc") or 0.0),
        permalink=str(item.get("permalink") or ""),
        subreddit=str(item.get("subreddit") or ""),
        score=int(item["score"]) if isinstance(item.get("score"), int) else None,
    )


class ArchiveClient(_CachedTimeline):
    """Read-only timeline client for the Arctic Shift archive (no credentials).

    Authenticating to Reddit is no longer possible for new projects, so this is
    the default source. Pagination walks backwards in time with ``before``, since
    the archive has no cursor; the archive is roughly 36 hours behind live.
    """

    source = "arctic-shift archive"
    cache_group = "archive"
    post_type = RedditPost

    def __init__(
        self,
        *,
        timeout: int = 40,
        cache_dir: str | Path | None = DEFAULT_CACHE,
        ttl_seconds: int = 21_600,
        transport=None,
        clock=time.time,
        sleeper=time.sleep,
        user_agent: str = ARCHIVE_AGENT,
    ):
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds
        self._transport = transport or _default_transport
        self._clock = clock
        self._sleeper = sleeper
        self.user_agent = user_agent

    def _request(self, path: str, params: dict, *, attempts: int = 4) -> dict:
        query = urllib.parse.urlencode(params)
        for attempt in range(attempts):
            status, _, payload = self._transport(
                "GET", f"{ARCHIVE_ROOT}{path}?{query}",
                {"User-Agent": self.user_agent}, None, self.timeout,
            )
            if status == 200:
                try:
                    body = json.loads(payload.decode("utf-8"))
                except ValueError as error:
                    raise RedditError("The archive returned malformed JSON.") from error
                if body.get("error"):
                    raise RedditError(f"The archive refused the query: {body['error']}")
                return body
            if status in (422, 429, 500, 502, 503, 504) and attempt + 1 < attempts:
                # 422 is the archive's documented "query timed out" overload reply.
                self._sleeper(min(4.0 * (attempt + 1), 30.0))
                continue
            if status in (422, 429):
                raise RedditError(
                    "The archive is overloaded (HTTP " f"{status}); retry in a minute."
                )
            raise RedditError(f"The archive request failed with HTTP {status}.")
        raise RedditError(f"The archive kept failing after {attempts} attempts.")

    def _page(self, path: str, params: dict, kind: str) -> list[RedditPost]:
        body = self._request(path, params)
        return [
            post
            for post in (post_from_archive(item, kind) for item in body.get("data") or [])
            if post
        ]

    def timeline(
        self,
        username: str,
        *,
        kinds: tuple[str, ...] = ("submitted", "comments"),
        limit: int = 200,
        refresh: bool = False,
    ) -> list[RedditPost]:
        """Fetch a user's public submissions and comments, newest first."""
        name = (username or "").strip().removeprefix("u/").removeprefix("/u/")
        if not USERNAME_PATTERN.match(name):
            raise RedditError("Enter a plain Reddit username, for example 'spez'.")
        if limit < 1:
            raise ValueError("Limit must be positive")

        collected: list[RedditPost] = []
        for kind in kinds:
            if kind not in ("submitted", "comments"):
                raise ValueError("Kinds must be 'submitted' or 'comments'")
            path = self._cache_path(name, kind)
            cached = self._read_cache(path, refresh=refresh)
            if cached is not None:
                collected.extend(cached[:limit])
                continue
            posts: list[RedditPost] = []
            seen: set[str] = set()
            cursor: float | None = None
            while len(posts) < limit:
                params = {
                    "author": name,
                    "limit": min(ARCHIVE_PAGE, limit - len(posts)),
                    "sort": "desc",
                }
                if cursor is not None:
                    params["before"] = int(cursor)
                page = self._page(
                    "/posts/search" if kind == "submitted" else "/comments/search",
                    params,
                    "submission" if kind == "submitted" else "comment",
                )
                fresh = [post for post in page if post.id not in seen]
                for post in fresh:
                    seen.add(post.id)
                    posts.append(post)
                if not fresh or len(page) < params["limit"]:
                    break
                oldest = min(post.created_utc for post in page)
                if cursor is not None and oldest >= cursor:
                    break
                cursor = oldest
            posts = posts[:limit]
            self._write_cache(path, posts)
            collected.extend(posts)
        return collected


class RedditClient(_CachedTimeline):
    """Minimal read-only client with token caching, rate limits, and disk cache."""

    source = "reddit oauth api"
    post_type = RedditPost

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        *,
        timeout: int = 20,
        cache_dir: str | Path | None = DEFAULT_CACHE,
        ttl_seconds: int = 21_600,
        transport=None,
        clock=time.time,
        sleeper=time.sleep,
    ):
        if not client_id or not client_secret:
            raise RedditError(
                "Reddit credentials are missing. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET for a script app from https://www.reddit.com/prefs/apps"
            )
        if not user_agent or len(user_agent) < 8:
            raise RedditError("Set a descriptive REDDIT_USER_AGENT, e.g. 'xai-hsd/0.1 by u/name'")
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds
        self._transport = transport or _default_transport
        self._clock = clock
        self._sleeper = sleeper
        self._token: str | None = None
        self._token_expires_at = 0.0

    @classmethod
    def from_env(cls, **options) -> "RedditClient":
        return cls(
            os.environ.get("REDDIT_CLIENT_ID", ""),
            os.environ.get("REDDIT_CLIENT_SECRET", ""),
            os.environ.get("REDDIT_USER_AGENT", "xai-hsd/0.1 (hate-speech research)"),
            **options,
        )

    def _access_token(self) -> str:
        if self._token and self._clock() < self._token_expires_at - 30:
            return self._token
        credentials = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        import base64

        headers = {
            "Authorization": "Basic " + base64.b64encode(credentials).decode("ascii"),
            "User-Agent": self.user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        status, _, payload = self._transport(
            "POST", TOKEN_URL, headers, {"grant_type": "client_credentials"}, self.timeout,
        )
        if status != 200:
            raise RedditError(f"Reddit rejected the credentials (HTTP {status}).")
        try:
            body = json.loads(payload.decode("utf-8"))
            self._token = body["access_token"]
            self._token_expires_at = self._clock() + float(body.get("expires_in", 3600))
        except (KeyError, ValueError) as error:
            raise RedditError("Reddit returned an unexpected token response.") from error
        return self._token

    def _request(self, path: str, params: dict, *, attempts: int = 3) -> dict:
        query = urllib.parse.urlencode({**params, "raw_json": 1})
        for attempt in range(attempts):
            headers = {
                "Authorization": f"bearer {self._access_token()}",
                "User-Agent": self.user_agent,
            }
            status, response_headers, payload = self._transport(
                "GET", f"{API_ROOT}{path}?{query}", headers, None, self.timeout,
            )
            remaining = response_headers.get("x-ratelimit-remaining")
            reset = response_headers.get("x-ratelimit-reset")
            if status == 200:
                if remaining is not None and float(remaining) <= 0 and reset:
                    self._sleeper(min(float(reset), 60.0))
                return json.loads(payload.decode("utf-8"))
            if status == 401:
                self._token = None
                continue
            if status == 429 and attempt + 1 < attempts:
                self._sleeper(min(float(reset or 5), 60.0))
                continue
            if status == 404:
                raise RedditError(f"Reddit has no account at {path} (HTTP 404).")
            if status in (403, 451):
                raise RedditError(
                    f"Reddit refused {path} (HTTP {status}); the account may be private, "
                    "suspended, or access is restricted."
                )
            raise RedditError(f"Reddit request failed with HTTP {status}.")
        raise RedditError(f"Reddit request kept failing after {attempts} attempts.")

    def timeline(
        self,
        username: str,
        *,
        kinds: tuple[str, str] = ("submitted", "comments"),
        limit: int = 200,
        refresh: bool = False,
    ) -> list[RedditPost]:
        """Fetch a user's public submissions and comments, newest first."""
        name = (username or "").strip().removeprefix("u/").removeprefix("/u/")
        if not USERNAME_PATTERN.match(name):
            raise RedditError("Enter a plain Reddit username, for example 'spez'.")
        if limit < 1:
            raise ValueError("Limit must be positive")

        collected: list[RedditPost] = []
        for kind in kinds:
            if kind not in ("submitted", "comments"):
                raise ValueError("Kinds must be 'submitted' or 'comments'")
            path = self._cache_path(name, kind)
            cached = self._read_cache(path, refresh=refresh)
            if cached is not None:
                collected.extend(cached[:limit])
                continue
            posts: list[RedditPost] = []
            cursor = None
            while len(posts) < limit:
                params = {"limit": min(100, limit - len(posts)), "t": "all", "sort": "new"}
                if cursor:
                    params["after"] = cursor
                page, cursor = parse_listing(self._request(f"/user/{name}/{kind}", params))
                posts.extend(page)
                if not cursor or not page:
                    break
            posts = posts[:limit]
            self._write_cache(path, posts)
            collected.extend(posts)
        return collected
