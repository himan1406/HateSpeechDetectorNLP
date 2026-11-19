import unittest
from dataclasses import dataclass

import numpy as np

from xai_hsd.account import assess_posts, dedupe_posts, wilson_interval


@dataclass(frozen=True)
class FakePost:
    id: str
    text: str
    created_utc: float = 0.0
    permalink: str = ""
    subreddit: str = "test"

    @property
    def url(self):
        return f"https://www.reddit.com{self.permalink}" if self.permalink else ""


def scorer(texts):
    """Deterministic stand-in: posts containing 'bad' score 0.9, others 0.1."""
    return np.array([[0.1, 0.9] if "bad" in text else [0.9, 0.1] for text in texts])


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        low, high = wilson_interval(10, 20)
        self.assertAlmostEqual(low, 0.2993, places=3)
        self.assertAlmostEqual(high, 0.7007, places=3)

    def test_zero_successes_has_a_bounded_upper_edge(self):
        low, high = wilson_interval(0, 20)
        self.assertAlmostEqual(low, 0.0, places=9)
        self.assertAlmostEqual(high, 0.1611, places=3)

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(5, 0)
        with self.assertRaises(ValueError):
            wilson_interval(5, 3)


class AssessmentTests(unittest.TestCase):
    def posts(self, count, bad_every=2):
        return [
            FakePost(f"p{index}", f"post number {index} " + ("bad thing" if index % bad_every == 0 else "fine thing"),
                     created_utc=float(index), permalink=f"/r/test/{index}")
            for index in range(count)
        ]

    def test_reports_a_share_with_an_interval(self):
        report = assess_posts(self.posts(10), scorer, min_posts=5, min_words=1)
        self.assertEqual(report["counts"]["scored"], 10)
        self.assertEqual(report["flagged_posts"], 5)
        self.assertAlmostEqual(report["flagged_share"], 0.5)
        self.assertLessEqual(report["interval"]["low"], 0.5)
        self.assertGreaterEqual(report["interval"]["high"], 0.5)
        self.assertEqual(report["signal"], "no_reference")

    def test_small_samples_are_refused(self):
        report = assess_posts(self.posts(4), scorer, min_posts=5, min_words=1)
        self.assertEqual(report["signal"], "insufficient_data")
        self.assertIsNone(report["flagged_share"])
        self.assertEqual(report["evidence"], [])

    def test_duplicates_and_short_posts_are_excluded(self):
        posts = [
            FakePost("a", "the same post text here", 3.0, "/r/t/a"),
            FakePost("b", "THE same   post text here", 2.0, "/r/t/b"),
            FakePost("c", "short", 1.0, "/r/t/c"),
            FakePost("d", "a different post about things", 0.0, "/r/t/d"),
        ]
        kept, removed = dedupe_posts(posts)
        self.assertEqual(removed, 1)
        self.assertEqual(kept[0].id, "a")  # newest copy wins
        report = assess_posts(posts, scorer, min_posts=1, min_words=3)
        self.assertEqual(report["counts"]["duplicates_removed"], 1)
        self.assertEqual(report["counts"]["excluded_short"], 1)
        self.assertEqual(report["counts"]["scored"], 2)

    def test_evidence_is_flagged_sorted_and_limited(self):
        report = assess_posts(
            self.posts(12), scorer, min_posts=5, min_words=1, evidence_limit=3,
        )
        self.assertEqual(len(report["evidence"]), 3)
        self.assertTrue(all(row["hate_score"] >= 0.5 for row in report["evidence"]))
        scores = [row["hate_score"] for row in report["evidence"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(report["evidence"][0]["url"].startswith("https://www.reddit.com/r/test/"))

    def test_reference_comparisons(self):
        above = assess_posts(self.posts(20, bad_every=1), scorer, min_posts=5, min_words=1,
                             reference_rate=0.1)
        self.assertEqual(above["signal"], "above_reference")
        below = assess_posts(self.posts(20, bad_every=1000), scorer, min_posts=5, min_words=1,
                             reference_rate=0.9)
        self.assertEqual(below["signal"], "below_reference")
        overlapping = assess_posts(self.posts(6), scorer, min_posts=5, min_words=1,
                                   reference_rate=0.5)
        self.assertEqual(overlapping["signal"], "indistinguishable_from_reference")

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            assess_posts([], scorer, threshold=1.5)
        with self.assertRaises(ValueError):
            assess_posts([], scorer, min_posts=0)
        with self.assertRaises(ValueError):
            assess_posts([], scorer, reference_rate=-0.1)


if __name__ == "__main__":
    unittest.main()
