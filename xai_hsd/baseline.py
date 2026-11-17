"""Calibrated word and character TF-IDF linear SVM."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .data import normalize_text


@dataclass
class Baseline:
    features: FeatureUnion
    classifier: CalibratedClassifierCV
    adjustment: np.ndarray | None = None

    def predict_proba(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 2))
        matrix = self.features.transform([normalize_text(str(text)) for text in texts])
        probabilities = self.classifier.predict_proba(matrix)
        if self.adjustment is None:
            return probabilities
        reference = logit(np.clip(probabilities[:, 1], 1e-7, 1 - 1e-7))
        scores = expit(reference + matrix @ self.adjustment[:-1] + self.adjustment[-1])
        return np.column_stack((1 - scores, scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary, compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"Expected a trained XAI-HSD baseline at {path}")
        return model


def train_baseline(records, seed=42):
    texts = [normalize_text(record.text) for record in records]
    labels = np.array([record.label for record in records], dtype=int)
    folds = min(5, int(np.count_nonzero(labels == 0)), int(np.count_nonzero(labels == 1)))
    if folds < 2:
        raise ValueError("Baseline training requires at least two records per class.")
    features = FeatureUnion([
        ("words", TfidfVectorizer(
            ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b|[^\w\s]", sublinear_tf=True,
        )),
        ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    matrix = features.fit_transform(texts)
    classifier = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=seed), cv=folds, ensemble=False,
    )
    classifier.fit(matrix, labels)
    return Baseline(features, classifier)
