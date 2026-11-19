import threading
import unittest
import urllib.parse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import xai_hsd.reddit as reddit_module
from xai_hsd.reddit import RedditClient, RedditError


class FakeReddit(BaseHTTPRequestHandler):
    """A tiny stand-in for the two Reddit endpoints the client uses."""

    requests = []
    deny_token = False
    expire_once = 0

    def log_message(self, *args):  # keep test output readable
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        FakeReddit.requests.append({
            "method": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "body": self.rfile.read(length).decode("utf-8"),
        })
        if FakeReddit.deny_token:
            self._send(401, {"message": "Unauthorized", "error": 401})
            return
        if self.headers.get("Authorization", "").startswith("Basic "):
            self._send(200, {"access_token": "tok-123", "expires_in": 3600})
        else:
            self._send(400, {"error": "missing basic auth"})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        FakeReddit.requests.append({
            "method": "GET",
            "path": parsed.path,
            "headers": dict(self.headers),
            "query": query,
        })
        if not self.headers.get("Authorization", "").startswith("bearer tok-123"):
            self._send(401, {"error": 401})
            return
        if FakeReddit.expire_once:
            FakeReddit.expire_once -= 1
            self._send(401, {"error": 401})
            return
        kind = parsed.path.rsplit("/", 1)[-1]
        if query.get("after"):
            self._send(200, {"data": {"after": None, "children": [{
                "kind": "t1",
                "data": {
                    "id": f"{kind}-2", "body": "a second page comment",
                    "created_utc": 5.0, "permalink": "/r/test/2",
                },
            }]}})
        else:
            self._send(200, {"data": {"after": "cursor-1", "children": [{
                "kind": "t3",
                "data": {
                    "id": f"{kind}-1", "title": "a first page title", "selftext": "",
                    "created_utc": 4.0, "permalink": "/r/test/1",
                    "subreddit": "test", "score": 3,
                },
            }]}})


class RedditHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeReddit)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        FakeReddit.requests.clear()
        FakeReddit.deny_token = False
        FakeReddit.expire_once = 0
        self.original = (reddit_module.TOKEN_URL, reddit_module.API_ROOT)
        reddit_module.TOKEN_URL = self.base + "/api/v1/access_token"
        reddit_module.API_ROOT = self.base

    def tearDown(self):
        reddit_module.TOKEN_URL, reddit_module.API_ROOT = self.original

    def client(self, **options):
        return RedditClient(
            "test-id", "test-secret", "xai-hsd/0.1 by u/tester",
            cache_dir=None, **options,
        )

    def test_token_request_is_a_client_credentials_post_with_basic_auth(self):
        self.client().timeline("spez", limit=2)
        token_calls = [call for call in FakeReddit.requests if call["method"] == "POST"]
        self.assertEqual(len(token_calls), 1)
        call = token_calls[0]
        self.assertEqual(call["body"], "grant_type=client_credentials")
        self.assertEqual(
            call["headers"].get("Content-Type"), "application/x-www-form-urlencoded"
        )
        import base64

        encoded = call["headers"]["Authorization"].split(" ", 1)[1]
        self.assertEqual(base64.b64decode(encoded).decode(), "test-id:test-secret")

    def test_listing_uses_bearer_token_and_follows_pagination(self):
        posts = self.client().timeline("spez", limit=4)
        listing_calls = [call for call in FakeReddit.requests if call["method"] == "GET"]
        self.assertEqual({call["path"] for call in listing_calls},
                         {"/user/spez/submitted", "/user/spez/comments"})
        for call in listing_calls:
            self.assertTrue(call["headers"]["Authorization"].startswith("bearer tok-123"))
            self.assertEqual(call["query"]["raw_json"], "1")
        # Each kind paginates: one page with a cursor, then a final page.
        self.assertEqual(len(listing_calls), 4)
        self.assertEqual(
            sum(1 for call in listing_calls if call["query"].get("after") == "cursor-1"), 2
        )
        self.assertEqual(len(posts), 4)
        self.assertEqual(len({post.id for post in posts}), 4)

    def test_disk_cache_prevents_a_second_fetch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            client = RedditClient(
                "test-id", "test-secret", "xai-hsd/0.1 by u/tester",
                cache_dir=directory,
            )
            first = client.timeline("spez", limit=4)
            FakeReddit.requests.clear()
            second = client.timeline("spez", limit=4)
        self.assertEqual([post.id for post in first], [post.id for post in second])
        self.assertEqual(FakeReddit.requests, [])

    def test_expired_token_is_refreshed_without_failing(self):
        FakeReddit.expire_once = 1
        posts = self.client().timeline("spez", limit=2)
        self.assertTrue(posts)
        token_calls = [c for c in FakeReddit.requests if c["method"] == "POST"]
        self.assertEqual(len(token_calls), 2)  # initial token, then a refresh

    def test_rejected_credentials_raise_a_readable_error(self):
        FakeReddit.deny_token = True
        with self.assertRaises(RedditError) as caught:
            self.client().timeline("spez", limit=2)
        self.assertIn("rejected the credentials", str(caught.exception))


class CredentialFileTests(unittest.TestCase):
    keys = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")

    def setUp(self):
        import os

        self.os = os
        self.saved = {key: os.environ.pop(key, None) for key in self.keys}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                self.os.environ.pop(key, None)
            else:
                self.os.environ[key] = value

    def test_file_values_are_loaded_into_the_environment(self):
        import json
        import tempfile
        from pathlib import Path

        from xai_hsd.cli import _load_credentials

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            path.write_text(json.dumps({
                "REDDIT_CLIENT_ID": "abc",
                "REDDIT_CLIENT_SECRET": "def",
                "REDDIT_USER_AGENT": "xai-hsd/0.1 by u/tester",
            }), encoding="utf-8")
            _load_credentials(path)
        self.assertEqual(self.os.environ["REDDIT_CLIENT_ID"], "abc")
        self.assertEqual(self.os.environ["REDDIT_CLIENT_SECRET"], "def")

    def test_a_non_object_file_is_rejected(self):
        import json
        import tempfile
        from pathlib import Path

        from xai_hsd.cli import _load_credentials

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_credentials(path)


if __name__ == "__main__":
    unittest.main()
