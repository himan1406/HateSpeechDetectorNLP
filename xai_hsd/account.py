"""Account-level aggregation of per-post scores.

A hateful post is not a hateful person. This module therefore reports a measured
share of flagged posts with an interval and a minimum sample size, never a
binary verdict about an account.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

from .data import normalize_text

DISCLAIMER = (
    "Automated signal for human review. A flagged post is not proof of intent, "
    "and an account score is not a determination about the person posting. "
    "Quotes, counter-speech, reclaimed language, and news reporting can all be "
    "flagged by mistake."
)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial share. Stable for small samples."""
    if total < 1:
        raise ValueError("A Wilson interval needs at least one observation")
    if not 0 <= successes <= total:
        raise ValueError("Successes must be between zero and the total")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    share = successes / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    margin = z * sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def account_handle(report: dict) -> str:
    """Platform-appropriate account label, e.g. ``u/name`` or ``@name``."""
    prefix = "@" if report.get("platform") == "bluesky" else "u/"
    return f"{prefix}{report['account']}"


def _text_of(post) -> str:
    return normalize_text(str(getattr(post, "text", post)))


def dedupe_posts(posts: Iterable) -> tuple[list, int]:
    """Drop repeated text, keeping the newest copy. Returns (kept, removed)."""
    ordered = sorted(
        posts,
        key=lambda post: float(getattr(post, "created_utc", 0.0) or 0.0),
        reverse=True,
    )
    seen, kept = set(), []
    for post in ordered:
        text = _text_of(post)
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(post)
    return kept, len(ordered) - len(kept)


def assess_posts(
    posts: Sequence,
    predict_proba,
    *,
    threshold: float = 0.5,
    platform: str = "reddit",
    source: str = "",
    account: str = "",
    model: str = "",
    min_posts: int = 20,
    min_words: int = 3,
    reference_rate: float | None = None,
    confidence: float = 0.95,
    evidence_limit: int = 10,
    collected_at: str | None = None,
) -> dict:
    """Score an account's posts and report a share with an interval.

    ``reference_rate`` is the flagged share expected on a comparable baseline
    corpus. Without it the report states a measured share and makes no
    comparison. ``min_posts`` guards against judging an account on a handful of
    posts.
    """
    posts = list(posts)
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    if min_posts < 1 or min_words < 1:
        raise ValueError("min_posts and min_words must be positive")
    if reference_rate is not None and not 0 <= reference_rate <= 1:
        raise ValueError("Reference rate must be between 0 and 1")

    unique, duplicates = dedupe_posts(posts)
    eligible = [post for post in unique if len(_text_of(post).split()) >= min_words]
    texts = [_text_of(post) for post in eligible]
    short = len(unique) - len(eligible)

    report = {
        "platform": platform,
        "source": source,
        "account": account,
        "model": model,
        "threshold": float(threshold),
        "collected_at": collected_at,
        "counts": {
            "collected": len(posts),
            "unique": len(unique),
            "duplicates_removed": duplicates,
            "excluded_short": short,
            "scored": len(texts),
        },
        "minimum_posts": min_posts,
        "reference_rate": reference_rate,
        "disclaimer": DISCLAIMER,
    }

    if len(texts) < min_posts:
        report.update({
            "signal": "insufficient_data",
            "flagged_share": None,
            "interval": None,
            "scores": None,
            "evidence": [],
            "note": (
                f"Only {len(texts)} usable posts; at least {min_posts} are needed "
                "before reporting a share."
            ),
        })
        return report

    scores = np.asarray([float(row[1]) for row in predict_proba(texts)])
    if scores.size != len(texts) or not np.all(np.isfinite(scores)):
        raise ValueError("The predictor must return one finite hate score per post")
    flagged = scores >= threshold
    count = int(flagged.sum())
    share = count / len(texts)
    low, high = wilson_interval(count, len(texts), confidence)

    if reference_rate is None:
        signal = "no_reference"
    elif low > reference_rate:
        signal = "above_reference"
    elif high < reference_rate:
        signal = "below_reference"
    else:
        signal = "indistinguishable_from_reference"

    order = np.argsort(-scores)
    evidence = [
        {
            "id": getattr(eligible[index], "id", str(index)),
            "text": texts[index][:500],
            "hate_score": float(scores[index]),
            "url": getattr(eligible[index], "url", ""),
            "created_utc": getattr(eligible[index], "created_utc", None),
            "subreddit": getattr(eligible[index], "subreddit", ""),
            "context": (
                getattr(eligible[index], "subreddit", "")
                or getattr(eligible[index], "handle", "")
            ),
        }
        for index in order[:evidence_limit]
        if flagged[index]
    ]

    report.update({
        "signal": signal,
        "flagged_share": share,
        "flagged_posts": count,
        "interval": {"low": low, "high": high, "confidence": confidence},
        "scores": {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p90": float(np.quantile(scores, 0.9)),
        },
        "evidence": evidence,
        "note": (
            "Share of flagged posts with a Wilson interval. Compare it with a "
            "reference corpus before drawing any conclusion."
        ),
    })
    return report


def describe(report: dict) -> str:
    """One-paragraph plain-text summary for the command line."""
    counts = report["counts"]
    source = report.get("source")
    head = (
        f"{report['platform']} {account_handle(report)}"
        f"{' via ' + source if source else ''}: collected {counts['collected']}, "
        f"scored {counts['scored']} after {counts['duplicates_removed']} duplicates and "
        f"{counts['excluded_short']} very short posts."
    )
    if report["signal"] == "insufficient_data":
        return f"{head}\nInsufficient data: {report['note']}"
    interval = report["interval"]
    body = (
        f"\n{report['flagged_posts']} flagged at threshold {report['threshold']:.2f} "
        f"({report['flagged_share']:.1%}, {interval['confidence']:.0%} CI "
        f"{interval['low']:.1%}-{interval['high']:.1%}); mean score "
        f"{report['scores']['mean']:.3f}, signal: {report['signal']}."
    )
    return head + body + f"\n{report['disclaimer']}"
