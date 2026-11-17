"""Download and run the pinned public TF-IDF/logistic regression pickle."""

import io
import pickle
import re
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("artifacts/model.pkl")
REVISION = "8523992895f5145361e49212af7f1d6c94b8f7a9"
MODEL_URL = f"https://huggingface.co/pajola/hate-speech-detection-lr-tfidf/resolve/{REVISION}/model.pkl"
DIGEST = "6ec756a704b58c022af51e9ec48904b781c57b9041fa468fb05f70ad99f57c71"
MODEL_SIZE = 3_840_436

# Exact dependencies observed in the pinned publisher artifact.
PICKLE_CLASSES = {
    ("sklearn.pipeline", "Pipeline"),
    ("sklearn.feature_extraction.text", "TfidfVectorizer"),
    ("sklearn.feature_extraction.text", "TfidfTransformer"),
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("scipy.sparse._csr", "csr_matrix"),
    ("numpy", "float64"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
}


class _PublisherUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in PICKLE_CLASSES:
            raise pickle.UnpicklingError(f"Unexpected model dependency: {module}.{name}")
        return super().find_class(module, name)


def _verify(payload):
    if len(payload) != MODEL_SIZE or sha256(payload).hexdigest() != DIGEST:
        raise ValueError("The downloaded pickle differs from the pinned publisher artifact.")


def fetch_model(output=MODEL_PATH):
    output = Path(output)
    if output.is_file():
        _verify(output.read_bytes())
        return output
    with urlopen(MODEL_URL, timeout=30) as response:
        payload = response.read(MODEL_SIZE + 1)
    _verify(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(payload)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _publisher_preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text.lower())
    text = re.sub(r"@\w+|#\w+|\brt\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


class PretrainedClassifier:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.hate_index = list(pipeline.classes_).index(0)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with Path(path).open("rb") as file:
            payload = file.read(MODEL_SIZE + 1)
        _verify(payload)
        pipeline = _PublisherUnpickler(io.BytesIO(payload)).load()
        if (
            type(pipeline) is not Pipeline
            or list(pipeline.named_steps) != ["tfidf", "classifier"]
            or type(pipeline.named_steps["tfidf"]) is not TfidfVectorizer
            or type(pipeline.steps[-1][1]) is not LogisticRegression
            or set(pipeline.classes_) != {0, 1, 2}
        ):
            raise ValueError("The publisher pipeline has an unexpected class mapping.")
        return cls(pipeline)

    def has_known_features(self, text):
        vectorizer = self.pipeline.named_steps["tfidf"]
        return bool(vectorizer.transform([_publisher_preprocess(text)]).nnz)

    def predict_proba(self, texts):
        cleaned = [_publisher_preprocess(text) for text in texts]
        if not cleaned:
            return np.empty((0, 2))
        hateful = self.pipeline.predict_proba(cleaned)[:, self.hate_index]
        return np.column_stack((1 - hateful, hateful))
