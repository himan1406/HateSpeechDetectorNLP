import tempfile
import unittest
from pathlib import Path

import numpy as np

from xai_hsd.baseline import Baseline, EnsembleBaseline
from xai_hsd.compare import compare_models, comparison_table
from xai_hsd.data import Record
from xai_hsd.models import CANDIDATES, train_candidate

HATEFUL = (
    "those people are vermin and should leave",
    "i hate that group they are animals",
    "disgusting people ruin everything here",
    "that religion is a disease on society",
)
CLEAN = (
    "people from that group are welcome here",
    "i support that community and their work",
    "the religion adds a lot to the city",
    "everyone should be treated with respect",
)


def synthetic(count=40, seed=11):
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        hateful = index % 2 == 1
        pool = HATEFUL if hateful else CLEAN
        text = f"{pool[int(rng.integers(0, len(pool)))]} number {index}"
        records.append(Record(f"r{index}", text, int(hateful)))
    return records


class CandidateTests(unittest.TestCase):
    def test_every_candidate_trains_and_scores(self):
        records = synthetic(40)
        texts = [record.text for record in records]
        for name in CANDIDATES:
            with self.subTest(model=name):
                model = train_candidate(name, records, seed=3)
                probabilities = np.asarray(model.predict_proba(texts))
                self.assertEqual(probabilities.shape, (len(texts), 2))
                self.assertTrue(np.all(np.isfinite(probabilities)))
                self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
                self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            train_candidate("does-not-exist", synthetic(10))

    def test_single_class_training_is_rejected(self):
        records = [Record(str(i), text, 0) for i, text in enumerate(CLEAN)]
        with self.assertRaises(ValueError):
            train_candidate("lr-word", records)

    def test_calibrated_models_stay_in_range(self):
        records = synthetic(40)
        model = train_candidate("lr-word", records)
        from xai_hsd.calibration import fit_calibrator

        raw = np.asarray(model.predict_proba([record.text for record in records]))[:, 1]
        model.calibrator = fit_calibrator([record.label for record in records], raw, "isotonic")
        probabilities = np.asarray(model.predict_proba([record.text for record in records]))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))


class EnsembleTests(unittest.TestCase):
    def test_ensemble_averages_its_members(self):
        records = synthetic(40)
        texts = [record.text for record in records]
        model = train_candidate("ensemble-lr-char", records, seed=3, ensemble_size=2)
        self.assertIsInstance(model, EnsembleBaseline)
        self.assertEqual(len(model.members), 2)
        combined = np.asarray(model.predict_proba(texts))
        manual = np.mean([member.predict_proba(texts) for member in model.members], axis=0)
        np.testing.assert_allclose(combined, manual)
        self.assertTrue(np.all((combined >= 0) & (combined <= 1)))

    def test_ensemble_members_differ(self):
        records = synthetic(40)
        model = train_candidate("ensemble-lr-char", records, seed=5, ensemble_size=2)
        first, second = (np.asarray(member.predict_proba([records[0].text]))[0, 1]
                         for member in model.members)
        self.assertNotAlmostEqual(first, second)

    def test_ensemble_survives_save_and_load(self):
        model = train_candidate("ensemble-lr-char", synthetic(40), seed=3, ensemble_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.joblib"
            model.save(path)
            loaded = Baseline.load(path)
        self.assertIsInstance(loaded, EnsembleBaseline)
        self.assertEqual(len(loaded.members), 2)

    def test_zero_members_are_rejected(self):
        with self.assertRaises(ValueError):
            train_candidate("ensemble-lr-char", synthetic(20), ensemble_size=0)


class ComparisonTests(unittest.TestCase):
    def splits(self):
        records = synthetic(90)
        return {
            "train": records[:60],
            "validation": records[60:75],
            "test": records[75:],
        }

    def test_ranks_candidates_and_reports_a_winner(self):
        report = compare_models(
            self.splits(), names=["lr-word", "nb-count"], n_resamples=100,
            include_pretrained=False,
        )
        self.assertEqual(len(report["models"]), 2)
        self.assertEqual([row["rank"] for row in report["models"]], [1, 2])
        best = report["best"]
        self.assertIn(best["name"], {"lr-word", "nb-count"})
        self.assertGreaterEqual(best["validation_macro_f1"], report["models"][1]["validation"]["metrics"]["macro_f1"])
        self.assertIn("saved_threshold", best)
        self.assertIn("test mF1", comparison_table(report))
        self.assertEqual(set(report["_models"]), {"lr-word", "nb-count"})

    def test_hard_negatives_enlarge_the_training_split(self):
        report = compare_models(
            self.splits(), names=["lr-word"], n_resamples=100,
            include_pretrained=False, hard_negatives=1,
        )
        block = report["hard_negatives"]
        self.assertEqual(block["copies_per_post"], 1)
        self.assertGreater(block["added"], 0)
        self.assertEqual(block["train_size_after"], block["train_size_before"] + block["added"])

    def test_cross_validation_ranks_by_fold_mean(self):
        report = compare_models(
            self.splits(), names=["lr-word"], n_resamples=100,
            include_pretrained=False, cv_folds=3,
        )
        cv = report["models"][0]["cross_validation"]
        self.assertEqual(len(cv["folds"]), 3)
        self.assertAlmostEqual(cv["mean_macro_f1"], float(np.mean([f["macro_f1"] for f in cv["folds"]])))
        self.assertEqual(report["ranking_metric"], "mean macro F1 over 3 training-split folds")
        self.assertAlmostEqual(report["best"]["ranking_metric"], cv["mean_macro_f1"])

    def test_missing_splits_are_rejected(self):
        splits = self.splits()
        splits["test"] = []
        with self.assertRaises(ValueError):
            compare_models(splits, names=["lr-word"], include_pretrained=False)


if __name__ == "__main__":
    unittest.main()
