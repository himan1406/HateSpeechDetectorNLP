"""Tests for the keyless public Bluesky timeline source."""

import io
import json
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from xai_hsd.account import account_handle, assess_posts, describe
from xai_hsd.bluesky import (
    BlueskyClient,
    BlueskyError,
    normalize_actor,
    post_from_feed_item,
)


def feed_item(rkey, text, created="2026-01-02T03:04:05.000Z", handle="tester.bsky.social", reply=False):
    record = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": created,
    }
    if reply:
        record["reply"] = {"parent": {"uri": "at://did:plc:x/app.bsky.feed.post/parent"}}
    return {
        "post": {
            "uri": f"at://did:plc:abc123/app.bsky.feed.post/{rkey}",
            "cid": "cid",
            "author": {"did": "did:plc:abc123", "handle": handle},
            "record": record,
        },
    }


class FeedParsingTests(unittest.TestCase):
    def test_post_text_url_and_timestamp(self):
        post = post_from_feed_item(feed_item("3kabc", "A normal post"))
        self.assertEqual(post.id, "3kabc")
        self.assertEqual(post.kind, "post")
        self.assertEqual(post.text, "A normal post")
        self.assertEqual(post.url, "https://bsky.app/profile/tester.bsky.social/post/3kabc")
        self.assertAlmostEqual(post.created_utc, 1767323045.0)

    def test_replies_are_kept_as_the_authors_own_words(self):
        post = post_from_feed_item(feed_item("3krep", "a reply", reply=True))
        self.assertEqual(post.kind, "reply")

    def test_reposts_are_dropped(self):
        item = feed_item("3krepost", "someone else wrote this")
        item["reason"] = {"$type": "app.bsky.feed.defs#reasonRepost"}
        self.assertIsNone(post_from_feed_item(item))

    def test_empty_and_image_only_posts_are_dropped(self):
        self.assertIsNone(post_from_feed_item(feed_item("3k1", "   ")))
        self.assertIsNone(post_from_feed_item({"post": {"uri": "at://did:plc:x/app.bsky.feed.post/3k", "record": {}}}))
        self.assertIsNone(post_from_feed_item({"post": {"record": {"$type": "app.bsky.feed.post", "text": "x"}}}))

    def test_whitespace_is_collapsed(self):
        post = post_from_feed_item(feed_item("3k2", "  two\nlines\there "))
        self.assertEqual(post.text, "two lines here")

    def test_unusable_timestamp_is_zero(self):
        post = post_from_feed_item(feed_item("3k3", "text", created="not a date"))
        self.assertEqual(post.created_utc, 0.0)

    def test_malformed_items_are_ignored(self):
        self.assertIsNone(post_from_feed_item({}))
        self.assertIsNone(post_from_feed_item("not a dict"))


class ActorTests(unittest.TestCase):
    def test_accepts_handle_forms(self):
        for value in (
            "tester.bsky.social",
            "@tester.bsky.social",
            "https://bsky.app/profile/tester.bsky.social",
            "https://bsky.app/profile/tester.bsky.social/",
            "did:plc:abc123",
        ):
            self.assertEqual(
                normalize_actor(value),
                "did:plc:abc123" if value.startswith("did:") else "tester.bsky.social",
            )

    def test_rejects_obvious_non_handles(self):
        for value in ("", "   ", "not a handle!", "user_without_domain", "did:"):
            with self.assertRaises(BlueskyError):
                normalize_actor(value)


def page(items, cursor=None):
    return {"feed": items, "cursor": cursor}


class FakeFeed:
    """Serves canned pages keyed by cursor and records every request."""

    def __init__(self, pages, status=200, body=None):
        self.pages = pages
        self.status = status
        self.body = body
        self.requests = []

    def __call__(self, method, url, headers, data=None, timeout=20):
        parsed = urllib.parse.urlparse(url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.requests.append({"path": parsed.path, "params": params, "headers": headers})
        if self.status != 200:
            return self.status, {}, self.body or b'{"error": "BadRequest", "message": "nope"}'
        payload = self.pages.get(params.get("cursor"), page([]))
        return 200, {}, json.dumps(payload).encode("utf-8")


class ClientTests(unittest.TestCase):
    def client(self, transport, cache_dir=None):
        return BlueskyClient(transport=transport, cache_dir=cache_dir, sleeper=lambda _: None)

    def test_paginates_with_the_cursor_and_caps_the_page(self):
        pages = {
            None: page([feed_item(f"p{i}", f"post {i}") for i in range(100)], "cursor-1"),
            "cursor-1": page([feed_item(f"p{i}", f"post {i}") for i in range(100, 120)], "cursor-2"),
        }
        fake = FakeFeed(pages)
        posts = self.client(fake).timeline("tester.bsky.social", limit=120)
        self.assertEqual(len(posts), 120)
        self.assertEqual([posts[0].id, posts[-1].id], ["p0", "p119"])
        self.assertEqual(len(fake.requests), 2)
        self.assertNotIn("cursor", fake.requests[0]["params"])
        self.assertEqual(fake.requests[1]["params"]["cursor"], "cursor-1")
        self.assertEqual([call["params"]["limit"] for call in fake.requests], ["100", "20"])
        self.assertEqual(fake.requests[0]["params"]["actor"], "tester.bsky.social")
        self.assertEqual(fake.requests[0]["params"]["filter"], "posts_with_replies")
        self.assertEqual(fake.requests[0]["params"]["includePins"], "false")
        self.assertTrue(fake.requests[0]["path"].endswith("/app.bsky.feed.getAuthorFeed"))

    def test_a_page_of_reposts_is_skipped_without_stopping(self):
        pages = {
            None: page([feed_item("r1", "reposted")], "cursor-1"),
            "cursor-1": page([feed_item("p1", "own post")], None),
        }
        pages[None]["feed"][0]["reason"] = {"$type": "app.bsky.feed.defs#reasonRepost"}
        posts = self.client(FakeFeed(pages)).timeline("tester.bsky.social", limit=5)
        self.assertEqual([post.id for post in posts], ["p1"])

    def test_duplicate_ids_across_pages_are_not_repeated(self):
        pages = {
            None: page([feed_item("p1", "one")], "cursor-1"),
            "cursor-1": page([feed_item("p1", "one")], "cursor-2"),
            "cursor-2": page([feed_item("p2", "two")], None),
        }
        posts = self.client(FakeFeed(pages)).timeline("tester.bsky.social", limit=5)
        self.assertEqual([post.id for post in posts], ["p1", "p2"])

    def test_limit_is_respected(self):
        pages = {None: page([feed_item(f"p{i}", f"post {i}") for i in range(5)], None)}
        posts = self.client(FakeFeed(pages)).timeline("tester.bsky.social", limit=2)
        self.assertEqual(len(posts), 2)

    def test_cache_is_reused_and_narrowed(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeFeed({None: page([feed_item("p1", "cached text")], None)})
            client = self.client(fake, cache_dir=Path(directory))
            self.assertEqual(len(client.timeline("tester.bsky.social", limit=5)), 1)
            self.assertEqual(len(client.timeline("tester.bsky.social", limit=5)), 1)
            self.assertEqual(len(fake.requests), 1)
            self.assertTrue((Path(directory) / "tester.bsky.social" / "posts.json").is_file())

    def test_refresh_bypasses_the_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeFeed({None: page([feed_item("p1", "text")], None)})
            client = self.client(fake, cache_dir=Path(directory))
            client.timeline("tester.bsky.social", limit=5)
            client.timeline("tester.bsky.social", limit=5, refresh=True)
            self.assertEqual(len(fake.requests), 2)

    def test_rate_limit_is_retried(self):
        state = {"calls": 0}

        def flaky(method, url, headers, data=None, timeout=20):
            state["calls"] += 1
            if state["calls"] < 3:
                return 429, {}, b'{"error": "RateLimitExceeded"}'
            return 200, {}, json.dumps(page([feed_item("p1", "hello")], None)).encode()

        posts = self.client(flaky).timeline("tester.bsky.social", limit=5)
        self.assertEqual([post.id for post in posts], ["p1"])
        self.assertEqual(state["calls"], 3)

    def test_persistent_rate_limit_raises(self):
        with self.assertRaises(BlueskyError) as caught:
            self.client(lambda *a, **k: (429, {}, b"{}")).timeline("tester.bsky.social", limit=5)
        self.assertIn("rate limiting", str(caught.exception))

    def test_unknown_account_is_reported(self):
        body = b'{"error": "InvalidRequest", "message": "Error: actor must be a valid did or handle"}'
        with self.assertRaises(BlueskyError) as caught:
            self.client(FakeFeed({}, status=400, body=body)).timeline("tester.bsky.social", limit=5)
        self.assertIn("rejected the account lookup", str(caught.exception))
        self.assertIn("valid did or handle", str(caught.exception))

    def test_blocked_or_private_feed_is_explicit(self):
        with self.assertRaises(BlueskyError) as caught:
            self.client(FakeFeed({}, status=403, body=b'{"error": "BlockedActor"}')).timeline(
                "tester.bsky.social", limit=5,
            )
        self.assertIn("refused the feed", str(caught.exception))

    def test_server_errors_are_retried_then_reported(self):
        with self.assertRaises(BlueskyError) as caught:
            self.client(lambda *a, **k: (503, {}, b"")).timeline("tester.bsky.social", limit=5)
        self.assertIn("503", str(caught.exception))

    def test_sends_a_descriptive_user_agent(self):
        fake = FakeFeed({None: page([], None)})
        self.client(fake).timeline("tester.bsky.social", limit=3)
        self.assertIn("xai-hsd", fake.requests[0]["headers"]["User-Agent"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.client(FakeFeed({})).timeline("tester.bsky.social", limit=0)
        with self.assertRaises(ValueError):
            self.client(FakeFeed({})).timeline("tester.bsky.social", kinds=("comments",))


class AccountIntegrationTests(unittest.TestCase):
    def test_assessment_labels_the_platform_and_links_posts(self):
        posts = [
            post_from_feed_item(feed_item(f"p{i}", f"post number {i}"))
            for i in range(3)
        ]
        report = assess_posts(
            posts, lambda texts: [[0.1, 0.9] for _ in texts],
            platform="bluesky", account="tester.bsky.social",
            min_posts=1, min_words=1,
        )
        self.assertEqual(account_handle(report), "@tester.bsky.social")
        self.assertEqual(report["evidence"][0]["context"], "tester.bsky.social")
        self.assertTrue(
            report["evidence"][0]["url"].startswith("https://bsky.app/profile/")
        )
        self.assertIn("@tester.bsky.social", describe(report))


class CommandLineTests(unittest.TestCase):
    def test_account_platform_bluesky_runs_end_to_end(self):
        from xai_hsd import cli

        posts = [
            post_from_feed_item(feed_item(f"p{i}", f"post number {i}"))
            for i in range(3)
        ]

        class Model:
            def predict_proba(self, texts):
                return [[0.1, 0.9] for _ in texts]

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "bsky.account.json"
            argv = [
                "xai-hsd", "account", "--platform", "bluesky",
                "--account", "@tester.bsky.social", "--backend", "pretrained",
                "--model", "fake", "--min-posts", "2", "--min-words", "1",
                "--report", str(report_path),
            ]
            with mock.patch.object(cli, "load_model", return_value=Model()), \
                 mock.patch("xai_hsd.bluesky.BlueskyClient.timeline", return_value=posts), \
                 mock.patch.object(sys, "argv", argv), \
                 redirect_stdout(io.StringIO()):
                cli.main()
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["platform"], "bluesky")
        self.assertEqual(report["source"], "bluesky public appview")
        self.assertEqual(report["account"], "tester.bsky.social")
        self.assertEqual(report["counts"]["scored"], 3)


if __name__ == "__main__":
    unittest.main()
