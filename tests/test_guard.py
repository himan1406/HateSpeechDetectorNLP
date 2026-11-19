import unittest

from xai_hsd.data import Record
from xai_hsd.guard import MIN_MATCHED_FEATURES, MIN_WORDS, assess_evidence, matched_features
from xai_hsd.models import train_candidate


def corpus():
    rows = [
        ("i hate those people and they should leave this country now", 1),
        ("those people are vermin and a disease on this nation", 1),
        ("everyone deserves respect and a fair chance in this life", 0),
        ("this community is welcoming and kind to every newcomer here", 0),
    ]
    records = []
    for index, (text, label) in enumerate(rows * 3):
        records.append(Record(f"r{index}", text, label, "en", "", None, "train"))
    return records


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_candidate("lr-word", corpus())

    def test_short_post_cannot_get_a_verdict(self):
        evidence = assess_evidence(self.model, "have a nice day")
        self.assertFalse(evidence.sufficient)
        self.assertEqual(evidence.words, 4)
        self.assertIn("word", evidence.reason)

    def test_empty_post_is_rejected(self):
        evidence = assess_evidence(self.model, "   ")
        self.assertFalse(evidence.sufficient)
        self.assertIn("empty", evidence.reason)

    def test_known_post_of_normal_length_is_sufficient(self):
        evidence = assess_evidence(self.model, "those people are vermin and a disease on this nation")
        self.assertTrue(evidence.sufficient)
        self.assertIsNone(evidence.reason)
        self.assertGreaterEqual(evidence.matched_features, MIN_MATCHED_FEATURES)

    def test_long_unseen_gibberish_is_caught_by_vocabulary_coverage(self):
        text = "zzqx vbnm qqww ttyy ppll kkjj hhgg ffdd ssaa mmnn"
        evidence = assess_evidence(self.model, text)
        self.assertGreaterEqual(evidence.words, MIN_WORDS)
        self.assertEqual(evidence.matched_features, 0)
        self.assertFalse(evidence.sufficient)
        self.assertIn("vocabulary", evidence.reason)

    def test_ensemble_reports_the_weakest_member(self):
        ensemble = train_candidate("ensemble-lr-wordchar", corpus(), seed=2, ensemble_size=2)
        individual = [matched_features(member, "have a nice day") for member in ensemble.members]
        self.assertEqual(matched_features(ensemble, "have a nice day"), min(individual))

    def test_models_without_a_vectorizer_skip_the_feature_check(self):
        class Opaque:
            def predict_proba(self, texts):
                raise AssertionError("not called")

        self.assertIsNone(matched_features(Opaque(), "some text"))
        self.assertTrue(assess_evidence(Opaque(), "a post long enough to pass the word check").sufficient)

    def test_minimums_must_be_positive(self):
        with self.assertRaises(ValueError):
            assess_evidence(self.model, "any text at all here now", min_words=0)
        with self.assertRaises(ValueError):
            assess_evidence(self.model, "any text at all here now", min_matched_features=0)

    def test_thresholds_are_configurable(self):
        text = "those people are vermin and a disease on this nation"
        self.assertFalse(assess_evidence(self.model, text, min_words=20).sufficient)
        self.assertTrue(assess_evidence(self.model, text, min_words=5).sufficient)


if __name__ == "__main__":
    unittest.main()
