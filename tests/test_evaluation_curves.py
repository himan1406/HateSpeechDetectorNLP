import unittest

from xai_hsd.data import Record
from xai_hsd.evaluation import evaluate, score_curves


class ScoreCurvesTests(unittest.TestCase):
    def test_perfect_separation(self):
        curves = score_curves([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertAlmostEqual(curves["roc"]["auc"], 1.0)
        self.assertAlmostEqual(curves["precision_recall"]["average_precision"], 1.0)
        row = next(item for item in curves["threshold_sweep"] if item["threshold"] == 0.5)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(row["recall"], 1.0)
        self.assertEqual(row["confusion_matrix"], [[2, 0], [0, 2]])

    def test_inverted_separation(self):
        curves = score_curves([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(curves["roc"]["auc"], 0.0)
        row = next(item for item in curves["threshold_sweep"] if item["threshold"] == 0.5)
        self.assertEqual(row["accuracy"], 0.0)
        self.assertEqual(row["recall"], 0.0)

    def test_single_class_curves_are_undefined(self):
        curves = score_curves([0, 0, 0], [0.1, 0.5, 0.9], thresholds=[0.5])
        self.assertIsNone(curves["roc"]["auc"])
        self.assertEqual(curves["roc"]["fpr"], [])
        self.assertIsNone(curves["precision_recall"]["average_precision"])
        self.assertAlmostEqual(curves["threshold_sweep"][0]["accuracy"], 1 / 3)

    def test_threshold_rule_matches_evaluate(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.6, 0.9]
        records = [Record(str(index), "post", label) for index, label in enumerate(labels)]
        report = evaluate(records, scores, threshold=0.5)["overall"]
        sweep_row = score_curves(labels, scores, thresholds=[0.5])["threshold_sweep"][0]
        for key in (
            "count", "positive_support", "negative_support", "accuracy", "precision",
            "recall", "f1", "macro_f1", "confusion_matrix", "false_positive_rate",
            "false_negative_rate",
        ):
            self.assertEqual(report[key], sweep_row[key])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            score_curves([0, 1], [0.5])
        with self.assertRaises(ValueError):
            score_curves([0, 2], [0.2, 0.8])
        with self.assertRaises(ValueError):
            score_curves([0, 1], [0.2, 1.5])
        with self.assertRaises(ValueError):
            score_curves([0, 1], [0.2, 0.8], thresholds=[1.5])


if __name__ == "__main__":
    unittest.main()
