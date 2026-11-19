import unittest

from xai_hsd.augment import COUNTER_FRAMES, counter_speech, hard_negatives
from xai_hsd.data import Record


def records():
    return [
        Record("h1", "those people are vermin", 1, "en", "race", None, "train"),
        Record("h2", "that religion is a disease", 1, "en", "religion", None, "train"),
        Record("n1", "everyone is welcome here", 0, "en", "", None, "train"),
    ]


class CounterSpeechTests(unittest.TestCase):
    def test_frame_contains_the_quoted_post(self):
        text = counter_speech("those people are vermin", COUNTER_FRAMES[0])
        self.assertIn("those people are vermin", text)
        self.assertIn("quoting", text)

    def test_empty_post_is_rejected(self):
        with self.assertRaises(ValueError):
            counter_speech("   ")

    def test_generated_rows_are_non_hateful_and_train_only(self):
        generated = hard_negatives(records(), copies=1, seed=7)
        self.assertEqual(len(generated), 2)  # only the hateful posts are sources
        for row in generated:
            self.assertEqual(row.label, 0)
            self.assertEqual(row.split, "train")
            self.assertIn("counter", row.id)
            self.assertTrue(row.text)
        self.assertEqual({row.group for row in generated}, {"race", "religion"})

    def test_identifiers_are_unique_and_deterministic(self):
        first = hard_negatives(records(), copies=2, seed=7)
        second = hard_negatives(records(), copies=2, seed=7)
        self.assertEqual([row.id for row in first], [row.id for row in second])
        self.assertEqual([row.text for row in first], [row.text for row in second])
        self.assertEqual(len({row.id for row in first}), len(first))
        self.assertEqual(len(first), 4)

    def test_frames_vary_across_copies(self):
        generated = hard_negatives(records(), copies=2, seed=7)
        h1 = [row.text for row in generated if row.id.startswith("h1")]
        self.assertNotEqual(h1[0], h1[1])

    def test_input_is_not_modified(self):
        rows = records()
        before = [(row.id, row.text, row.label, row.split) for row in rows]
        hard_negatives(rows, copies=1)
        self.assertEqual([(row.id, row.text, row.label, row.split) for row in rows], before)

    def test_duplicates_are_dropped(self):
        rows = [Record("a", "same hateful text here", 1), Record("b", "same hateful text here", 1)]
        self.assertEqual(len(hard_negatives(rows, copies=1)), 1)

    def test_zero_copies_and_invalid_inputs(self):
        self.assertEqual(hard_negatives(records(), copies=0), [])
        with self.assertRaises(ValueError):
            hard_negatives(records(), copies=-1)
        with self.assertRaises(ValueError):
            hard_negatives(records(), copies=1, frames=())
        # Only hateful posts are sources, so a corpus with none yields nothing.
        self.assertEqual(hard_negatives([Record("n", "nice post", 0)], copies=1), [])


if __name__ == "__main__":
    unittest.main()
