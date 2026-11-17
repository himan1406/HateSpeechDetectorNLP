"""Classroom walkthrough of the original HateXplain training experiment.

The supervised baseline uses word and character TF-IDF with a linear SVM.
The model fitting code lives in xai_hsd/baseline.py and is shared with the
project CLI. This file shows the experiment in order, from labeled posts to
a saved model. Importing it leaves training idle.
"""

import json
from pathlib import Path

from xai_hsd.baseline import train_baseline
from xai_hsd.data import load_records, split_records
from xai_hsd.evaluation import evaluate

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.json"
MODEL_PATH = PROJECT_DIR / "artifacts" / "classroom-svm.joblib"
SEED = 42
HATE_THRESHOLD = 0.50


def main():
    # 1. Load annotated HateXplain posts. Majority hate maps to class 1.
    # Majority offensive or normal maps to class 0. Ties are discarded.
    records = load_records(DATA_PATH, source="hatexplain")

    # 2. Preserve the official split assignments and deduplicate posts.
    # Training uses the train split. Validation and test measure performance
    # on the two held-out splits using the same fixed decision threshold.
    splits = split_records(records, seed=SEED)

    # 3. Fit the original model on the training posts.
    # TF-IDF downweights common features. Word unigrams and bigrams capture
    # terms and short phrases. Character 2–5-grams pick up spelling fragments.
    # A class-weighted linear SVM learns a boundary in the combined feature space.
    # Cross-validation calibration maps the SVM margins to class probabilities.
    model = train_baseline(splits["train"], seed=SEED)

    # 4. Score unseen posts. Column 1 is the probability for the hate class.
    # Evaluate reports precision, recall, F1, and [[TN, FP], [FN, TP]].
    results = {}
    for split_name in ("validation", "test"):
        rows = splits[split_name]
        hate_scores = model.predict_proba([row.text for row in rows])[:, 1]
        results[split_name] = evaluate(rows, hate_scores, threshold=HATE_THRESHOLD)["overall"]
    print(json.dumps(results, indent=2))

    # 5. Save the fitted vectorizers and calibrated SVM in one artifact.
    model.save(MODEL_PATH)
    print(f"Saved classroom baseline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
