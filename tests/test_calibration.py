import unittest

import numpy as np

from xai_hsd.calibration import (
    apply_calibrator,
    bootstrap_interval,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    metrics_at,
    reliability_curve,
    select_threshold,
)


class CalibrationTests(unittest.TestCase):
    def test_brier_and_ece_are_zero_for_perfect_probabilities(self):
        labels = [0, 0, 1, 1]
        scores = [0.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(brier_score(labels, scores), 0.0)
        self.assertAlmostEqual(expected_calibration_error(labels, scores), 0.0)

    def test_brier_penalises_confident_mistakes(self):
        self.assertAlmostEqual(brier_score([0, 1], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(brier_score([0, 1], [0.5, 0.5]), 0.25)

    def test_reliability_bins_sum_to_the_sample(self):
        curve = reliability_curve([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], bins=10)
        self.assertEqual(sum(bin["count"] for bin in curve), 4)
        self.assertEqual(len(curve), 10)

    def test_metrics_at_matches_hand_counts(self):
        row = metrics_at([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
        self.assertEqual((row["tp"], row["tn"], row["fp"], row["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(row["f1"], 0.5)
        self.assertAlmostEqual(row["macro_f1"], 0.5)

    def test_target_fpr_constraint_is_respected(self):
        labels = [0, 0, 0, 0, 1, 1]
        scores = [0.1, 0.2, 0.55, 0.7, 0.6, 0.9]
        choice = select_threshold(labels, scores, target_fpr=0.25)
        self.assertLessEqual(choice["achieved_fpr"], 0.25)
        self.assertEqual(choice["threshold"], choice["metrics"]["threshold"])
        # The smallest threshold inside the budget should keep both positives.
        self.assertAlmostEqual(choice["metrics"]["recall"], 1.0)

    def test_objective_selection_maximises_the_requested_metric(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.2, 0.3, 0.8, 0.4, 0.6, 0.9]
        choice = select_threshold(labels, scores, objective="macro_f1")
        swept = [metrics_at(labels, scores, threshold)["macro_f1"]
                 for threshold in np.unique(np.r_[0.0, scores, 1.0])]
        self.assertAlmostEqual(choice["value"], max(swept))

    def test_bootstrap_interval_is_ordered_and_seeded(self):
        labels = [0, 0, 0, 1, 1, 1] * 10
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10
        first = bootstrap_interval(labels, scores, n_resamples=200, seed=7)
        second = bootstrap_interval(labels, scores, n_resamples=200, seed=7)
        self.assertEqual(first, second)
        for metric in first["metrics"].values():
            self.assertLessEqual(metric["low"], metric["high"])
            self.assertTrue(0.0 <= metric["low"] <= 1.0)

    def test_isotonic_calibration_is_monotone(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 0]
        scores = [0.05, 0.2, 0.35, 0.55, 0.7, 0.8, 0.95, 0.15]
        calibrator = fit_calibrator(labels, scores, "isotonic")
        mapped = apply_calibrator(calibrator, sorted(scores))
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12))
        self.assertTrue(np.all((mapped >= 0) & (mapped <= 1)))

    def test_platt_calibration_returns_probabilities(self):
        labels = [0, 0, 1, 1, 0, 1]
        scores = [0.1, 0.3, 0.6, 0.9, 0.2, 0.8]
        calibrator = fit_calibrator(labels, scores, "platt")
        mapped = apply_calibrator(calibrator, scores)
        self.assertTrue(np.all((mapped >= 0) & (mapped <= 1)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            brier_score([0, 1], [0.5])
        with self.assertRaises(ValueError):
            brier_score([0, 2], [0.2, 0.8])
        with self.assertRaises(ValueError):
            select_threshold([0, 1], [0.2, 0.8], target_fpr=1.5)
        with self.assertRaises(ValueError):
            bootstrap_interval([0, 1], [0.2, 0.8], n_resamples=5)


if __name__ == "__main__":
    unittest.main()
