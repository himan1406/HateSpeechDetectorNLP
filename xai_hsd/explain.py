"""Local, position-level SHAP and LIME attributions for binary classifiers."""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np

_SHAP_RANDOM_LOCK = Lock()


@dataclass(frozen=True)
class Attribution:
    token: str
    index: int
    start: int
    end: int
    score: float


def explain(
    text: str,
    predict_proba: Callable[[list[str]], np.ndarray],
    method: str = "shap",
    label: int = 1,
    *,
    seed: int = 42,
) -> list[Attribution]:
    """Explain a class probability using deletion of individual token positions.

    The predictor returns two columns in label order (0, 1) and must handle an
    empty string. Offsets reference the input text. Positive scores support
    the selected class and negative scores oppose it.
    """
    if label not in (0, 1):
        raise ValueError("Explanation label must be 0 or 1.")
    method = method.lower()
    if method not in {"shap", "lime"}:
        raise ValueError("Explanation method must be 'shap' or 'lime'.")

    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        raise ValueError("An explanation requires at least one token.")

    # Keep each original separator so the full mask reproduces the input.
    previous_ends = [0, *(token.end() for token in tokens[:-1])]
    pieces = [
        (text[end:token.start()], token.group())
        for end, token in zip(previous_ends, tokens)
    ]

    def render(mask: np.ndarray) -> str:
        if mask.all():
            return text
        return "".join(
            prefix + (token if keep else "") for (prefix, token), keep in zip(pieces, mask)
        ).strip()

    def predict_masks(masks: np.ndarray) -> np.ndarray:
        posts = [render(mask) for mask in masks]
        probabilities = np.asarray(predict_proba(posts), dtype=float)
        if probabilities.shape != (len(posts), 2):
            raise ValueError("The predictor must return one row and two probabilities per post.")
        if not np.isfinite(probabilities).all():
            raise ValueError("The predictor must return finite probabilities.")
        return probabilities

    if method == "shap":
        scores = _shap_scores(predict_masks, len(tokens), label, seed)
    else:
        scores = _lime_scores(predict_masks, len(tokens), label, seed)

    scores[np.abs(scores) < 1e-12] = 0.0
    return [
        Attribution(token.group(), index, token.start(), token.end(), float(score))
        for index, (token, score) in enumerate(zip(tokens, scores))
    ]


def _shap_scores(predict_masks, token_count: int, label: int, seed: int) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP explanations require the 'shap' package.") from exc

    deleted = np.zeros((1, token_count), dtype=np.int8)
    original = np.ones_like(deleted)
    if token_count <= 8:
        explainer = shap.ExactExplainer(predict_masks, deleted)
        result = explainer(original, batch_size=32, silent=True)
    else:
        # SHAP's permutation explainer seeds NumPy's process-wide generator.
        with _SHAP_RANDOM_LOCK:
            random_state = np.random.get_state()
            try:
                explainer = shap.PermutationExplainer(predict_masks, deleted, seed=seed)
                result = explainer(
                    original, max_evals=5 * (2 * token_count + 1), batch_size=32, silent=True
                )
            finally:
                np.random.set_state(random_state)
    return result.values[0, :, label]


def _lime_scores(predict_masks, token_count: int, label: int, seed: int) -> np.ndarray:
    try:
        from lime.lime_base import LimeBase
    except ImportError as exc:
        raise ImportError("LIME explanations require the 'lime' package.") from exc

    rng = np.random.default_rng(seed)
    masks = np.ones((256, token_count), dtype=np.int8)
    # Sample a removal count uniformly, then choose positions with that count.
    for mask in masks[1:]:
        removed = rng.integers(1, token_count + 1)
        mask[rng.choice(token_count, size=removed, replace=False)] = 0

    probabilities = np.concatenate([
        predict_masks(masks[start:start + 32]) for start in range(0, len(masks), 32)
    ])
    # Cosine distance from an all-present binary vector, scaled as in LIME Text.
    distances = 100 * (1 - np.sqrt(masks.mean(axis=1)))

    def kernel(distances: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (distances / 25) ** 2)

    explainer = LimeBase(kernel, random_state=seed)
    _, weights, _, _ = explainer.explain_instance_with_data(
        masks, probabilities, distances, label, token_count, feature_selection="none"
    )
    scores = np.zeros(token_count, dtype=float)
    for index, weight in weights:
        scores[index] = weight
    return scores


def rationale_overlap(
    attributions: Sequence[Attribution], rationale: Sequence[int]
) -> dict[str, float] | None:
    """Compare up to k supportive positions with a binary human mask.

    Here k is the number of highlighted human positions. A mask with no
    human highlights is skipped. Explain the human label before calling this.
    """
    if len(attributions) != len(rationale) or any(value not in (0, 1) for value in rationale):
        raise ValueError("The rationale must have one binary value per attribution.")
    human = {i for i, value in enumerate(rationale) if value}
    if not human:
        return None

    ordered = sorted(attributions, key=lambda item: (-item.score, item.index))
    predicted = {item.index for item in ordered[:len(human)] if item.score > 0}
    common = len(human & predicted)
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(human)
    f1 = 2 * precision * recall / (precision + recall) if common else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": common / len(human | predicted)}
