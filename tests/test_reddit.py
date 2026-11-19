import json
import tempfile
import unittest
from pathlib import Path

from xai_hsd.reddit import RedditClient, RedditError, parse_listing, post_from_listing


def submission(identifier, title, selftext="", created=1.0):
    return {
        "kind": "t3",
        "data": {
            "id": identifier, "title": title, "selftext": selftext,
            "created_utc": created, "permalink": f"/r/test/{identifier}",
            "subreddit": "test", "score": 5,
        },
    }


class ListingParsingTests(unittest.TestCase):
    def test_submission_joins_title_and_body(self):
        post = post_from_listing(submission("a1", "Hello", "world"))
        self.assertEqual(post.text, "Hello world")
        self.assertEqual(post.kind, "submission")
        self.assertEqual(post.url, "https://www.reddit.com/r/test/a1")

    def test_comment_uses_body(self):
        child = {"kind": "t1", "data": {"id": "c1", "body": "a comment", "permalink": "/r/t/c1"}}
        self.assertEqual(post_from_listing(child).text, "a comment")

    def test_removed_and_empty_items_are_dropped(self):
        self.assertIsNone(post_from_listing(submission("a2", "[deleted]")))
        self.assertIsNone(post_from_listing(submission("a3", "[deleted]", "[removed]")))
        self.assertIsNone(post_from_listing({"kind": "t5", "data": {"id": "x"}}))
        self.assertIsNone(post_from_listing({}))
        # A removed body leaves the visible title as legitimate content.
        self.assertEqual(post_from_listing(submission("a4", "Title", "[removed]")).text, "Title")

    def test_parse_listing_returns_cursor(self):
        payload = {"data": {"children": [submission("a", "one")], "after": "t3_next"}}
        posts, cursor = parse_listing(payload)
        self.assertEqual(len(posts), 1)
        self.assertEqual(cursor, "t3_next")
        self.assertIsNone(parse_listing({"data": {"children": []}})[1])


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.status = 200
        self.after = None

    def __call__(self, method, url, headers, data=None, timeout=20):
        self.calls.append((method, url))
        if "access_token" in url:
            if self.status == 401:
                return 401, {}, b"{}"
            return 200, {}, json.dumps({"access_token": "token", "expires_in": 3600}).encode()
        if self.status != 200:
            return self.status, {}, b"{}"
        if "submitted" in url:
            children = [submission("a", "first post"), submission("b", "[deleted]")]
        elif "comments" in url:
            children = [{"kind": "t1", "data": {"id": "c", "body": "a comment", "permalink": "/r/t/c"}}]
        else:
            raise AssertionError(url)
        return 200, {}, json.dumps({"data": {"children": children, "after": self.after}}).encode()


class ClientTests(unittest.TestCase):
    def client(self, transport, directory):
        return RedditClient(
            "id", "secret", "xai-hsd/0.1 test", transport=transport, cache_dir=directory,
        )

    def test_requires_credentials(self):
        with self.assertRaises(RedditError):
            RedditClient("", "", "xai-hsd/0.1 test")
        with self.assertRaises(RedditError):
            RedditClient("id", "secret", "xai")

    def test_rejects_bad_usernames(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(FakeTransport(), directory)
            for name in ("", "ab", "has spaces", "u/" + "x" * 30):
                with self.assertRaises(RedditError):
                    client.timeline(name)

    def test_timeline_collects_both_kinds_and_caches(self):
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(transport, directory)
            posts = client.timeline("someone", limit=10)
            self.assertEqual([post.id for post in posts], ["a", "c"])
            self.assertEqual(sum(1 for method, _ in transport.calls if method == "POST"), 1)

            first_count = len(transport.calls)
            again = client.timeline("someone", limit=10)
            self.assertEqual([post.id for post in again], ["a", "c"])
            self.assertEqual(len(transport.calls), first_count)

            cache = Path(directory) / "someone" / "submitted.json"
            self.assertTrue(cache.is_file())
            self.assertIn("first post", cache.read_text(encoding="utf-8"))

    def test_refresh_bypasses_the_cache(self):
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(transport, directory)
            client.timeline("someone", limit=10)
            count = len(transport.calls)
            client.timeline("someone", limit=10, refresh=True)
            self.assertGreater(len(transport.calls), count)

    def test_pagination_follows_the_cursor(self):
        transport = FakeTransport()
        transport.after = "t3_next"
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(transport, directory)
            posts = client.timeline("someone", limit=10, kinds=("submitted",))
            self.assertGreater(len(posts), 0)
            self.assertTrue(any("after=t3_next" in url for _, url in transport.calls))

    def test_auth_and_lookup_errors_are_explicit(self):
        transport = FakeTransport()
        transport.status = 401
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RedditError):
                self.client(transport, directory).timeline("someone", limit=5)

        transport = FakeTransport()
        transport.status = 404
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RedditError) as caught:
                self.client(transport, directory).timeline("someone", limit=5)
            self.assertIn("no account", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
