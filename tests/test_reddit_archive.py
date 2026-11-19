"""Tests for the credential-free Arctic Shift timeline source."""

import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from xai_hsd.reddit import ArchiveClient, RedditError, post_from_archive


def record(kind, ident, created, text):
    item = {"id": ident, "created_utc": created, "subreddit": "test", "permalink": f"/r/test/{ident}"}
    if kind == "submission":
        item.update(title=text, selftext="")
    else:
        item.update(body=text)
    return item


class FakeArchive:
    """Serves canned pages keyed by the requested endpoint and cursor."""

    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def __call__(self, method, url, headers, data=None, timeout=20):
        parsed = urllib.parse.urlparse(url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        self.requests.append({"path": parsed.path, "params": params, "headers": headers})
        key = (parsed.path, params.get("before"))
        page = self.pages.get(key, [])
        return 200, {}, json.dumps({"data": page, "error": None}).encode("utf-8")


class ArchiveParseTests(unittest.TestCase):
    def test_submission_joins_title_and_selftext(self):
        post = post_from_archive(
            {"id": "a1", "title": "Title", "selftext": "Body", "created_utc": 5, "subreddit": "s"},
            "submission",
        )
        self.assertEqual(post.text, "Title Body")
        self.assertEqual(post.kind, "submission")
        self.assertEqual(post.created_utc, 5.0)

    def test_deleted_and_empty_items_are_dropped(self):
        self.assertIsNone(post_from_archive({"id": "x", "body": "[deleted]"}, "comment"))
        self.assertIsNone(post_from_archive({"id": "x", "body": "   "}, "comment"))
        self.assertIsNone(post_from_archive({"body": "no id"}, "comment"))
        self.assertIsNone(post_from_archive({"id": "x", "body": "hi"}, "unknown"))


class ArchiveClientTests(unittest.TestCase):
    def client(self, transport, cache_dir=None):
        return ArchiveClient(transport=transport, cache_dir=cache_dir, sleeper=lambda _: None)

    def test_paginates_backwards_with_before(self):
        # The archive caps a page at 100, so a 230-post limit needs three walks.
        pages = {
            ("/api/posts/search", None): [
                record("submission", f"p{i}", i, f"post {i}") for i in range(230, 130, -1)
            ],
            ("/api/posts/search", "131"): [
                record("submission", f"p{i}", i, f"post {i}") for i in range(130, 30, -1)
            ],
            ("/api/posts/search", "31"): [
                record("submission", f"p{i}", i, f"post {i}") for i in range(30, 0, -1)
            ],
        }
        fake = FakeArchive(pages)
        posts = self.client(fake).timeline("tester", kinds=("submitted",), limit=230)
        self.assertEqual(len(posts), 230)
        self.assertEqual([posts[0].id, posts[-1].id], ["p230", "p1"])
        calls = [call for call in fake.requests if call["params"].get("author") == "tester"]
        self.assertEqual(len(calls), 3)
        self.assertNotIn("before", calls[0]["params"])
        self.assertEqual(calls[1]["params"]["before"], "131")
        self.assertEqual(calls[2]["params"]["before"], "31")
        self.assertEqual([call["params"]["limit"] for call in calls], ["100", "100", "30"])
        self.assertEqual(calls[0]["params"]["sort"], "desc")

    def test_duplicate_ids_across_pages_are_not_repeated(self):
        pages = {
            ("/api/comments/search", None): [record("comment", "c2", 200, "two"), record("comment", "c1", 100, "one")],
            ("/api/comments/search", "100"): [record("comment", "c1", 100, "one")],
        }
        posts = self.client(FakeArchive(pages)).timeline("tester", kinds=("comments",), limit=10)
        self.assertEqual([post.id for post in posts], ["c2", "c1"])

    def test_limit_is_respected_per_kind(self):
        pages = {
            ("/api/posts/search", None): [
                record("submission", f"p{i}", 100 - i, f"post {i}") for i in range(5)
            ],
        }
        posts = self.client(FakeArchive(pages)).timeline("tester", kinds=("submitted",), limit=2)
        self.assertEqual(len(posts), 2)

    def test_transient_overload_is_retried(self):
        state = {"calls": 0}

        def flaky(method, url, headers, data=None, timeout=20):
            state["calls"] += 1
            if state["calls"] < 3:
                return 422, {}, b'{"data": null, "error": "Query timed out"}'
            return 200, {}, json.dumps({"data": [record("comment", "c1", 10, "hello there")]}).encode()

        posts = self.client(flaky).timeline("tester", kinds=("comments",), limit=5)
        self.assertEqual(len(posts), 1)
        self.assertEqual(state["calls"], 3)

    def test_persistent_overload_raises_a_readable_error(self):
        with self.assertRaises(RedditError) as caught:
            self.client(lambda *a, **k: (422, {}, b"{}")).timeline("tester", kinds=("comments",), limit=5)
        self.assertIn("overloaded", str(caught.exception))

    def test_server_error_is_reported(self):
        with self.assertRaises(RedditError) as caught:
            self.client(lambda *a, **k: (503, {}, b"")).timeline("tester", kinds=("comments",), limit=5)
        self.assertIn("503", str(caught.exception))

    def test_error_field_in_a_200_body_is_surfaced(self):
        body = json.dumps({"data": None, "error": "'permalink' is not a valid field"}).encode()
        with self.assertRaises(RedditError) as caught:
            self.client(lambda *a, **k: (200, {}, body)).timeline("tester", kinds=("comments",), limit=5)
        self.assertIn("not a valid field", str(caught.exception))

    def test_rejects_bad_usernames(self):
        with self.assertRaises(RedditError):
            self.client(FakeArchive({})).timeline("not a user!")
        self.assertEqual(self.client(FakeArchive({})).timeline("u/tester", kinds=("comments",), limit=3), [])

    def test_archive_cache_is_separate_and_reused(self):
        pages = {("/api/comments/search", None): [record("comment", "c1", 10, "cached text")]}
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeArchive(pages)
            client = self.client(fake, cache_dir=Path(directory))
            first = client.timeline("tester", kinds=("comments",), limit=5)
            second = client.timeline("tester", kinds=("comments",), limit=5)
            self.assertEqual([post.id for post in second], ["c1"])
            self.assertEqual(len(fake.requests), len(first) and 1)
            self.assertTrue((Path(directory) / "tester" / "archive" / "comments.json").is_file())

    def test_cached_timeline_respects_a_smaller_limit(self):
        pages = {
            ("/api/comments/search", None): [
                record("comment", f"c{i}", 100 - i, f"comment {i}") for i in range(5)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeArchive(pages)
            client = self.client(fake, cache_dir=Path(directory))
            client.timeline("tester", kinds=("comments",), limit=5)
            narrowed = client.timeline("tester", kinds=("comments",), limit=2)
            self.assertEqual(len(narrowed), 2)
            self.assertEqual(len(fake.requests), 1)

    def test_sends_a_descriptive_user_agent(self):
        fake = FakeArchive({})
        self.client(fake).timeline("tester", kinds=("comments",), limit=3)
        self.assertIn("xai-hsd", fake.requests[0]["headers"]["User-Agent"])


if __name__ == "__main__":
    unittest.main()
