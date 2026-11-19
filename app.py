"""Local moderator review interface for XAI-HSD."""

import argparse
import sqlite3
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st

from xai_hsd import ui
from xai_hsd.cli import LABELS, load_model
from xai_hsd.data import normalize_text
from xai_hsd.explain import explain
from xai_hsd.feedback import Review, save_review
from xai_hsd.guard import assess_evidence
from xai_hsd.pretrained import MODEL_PATH, PretrainedClassifier


def settings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--backend", choices=("pretrained", "baseline", "transformer"), default="pretrained")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feedback", type=Path, default=Path("artifacts/reviews.sqlite3"))
    return parser.parse_args()


@st.cache_resource
def cached_model(path, backend, device, version):
    return load_model(path, backend, device)


def highlighted_text(text, tokens):
    scale = max((abs(token.score) for token in tokens), default=0)
    parts = []
    cursor = 0
    for token in tokens:
        parts.append(escape(text[cursor:token.start]))
        if token.score and scale:
            strength = 0.12 + 0.44 * abs(token.score) / scale
            color = "255, 113, 90" if token.score > 0 else "83, 170, 255"
            title = f"{token.score:+.4f}"
            parts.append(
                f'<mark title="{title}" style="background-color:rgba({color},{strength:.3f});'
                f'color:inherit;border-radius:3px;">{escape(text[token.start:token.end])}</mark>'
            )
        else:
            parts.append(escape(text[token.start:token.end]))
        cursor = token.end
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def render_result(result, args):
    text, score, label = result["text"], result["score"], result["label"]
    threshold = result["threshold"]

    with st.container(border=True):
        if label is None:
            st.badge("Insufficient evidence", color="gray")
        elif label == 1:
            st.badge("Flagged as hateful", color="red")
        else:
            st.badge("Not flagged", color="green")
        st.caption(f"Hateful at or above {threshold:.0%} · {args.backend.title()} model")
        st.metric("Hate score", f"{score:.1%}")
        ui.score_bar(score, threshold)

        if label is None:
            st.warning(f"Insufficient evidence: {result['reason']}")
            ui.post_block(text)
            st.caption("The score is shown for transparency. No verdict, explanation, or review is offered.")
            return
        if not result.get("has_known_features", True):
            st.warning("The pretrained vocabulary has no matches for this post. The score comes from the model's class bias.")

    tokens = result.get("tokens")
    if tokens is not None:
        st.markdown(f"#### {result['method'].upper()} word attributions")
        ui.post_block(highlighted_text(text, tokens), html=True)
        strongest = sorted(tokens, key=lambda token: abs(token.score), reverse=True)[:10]
        ui.chips([(token.token, token.score) for token in strongest if token.score])
        st.caption(
            "Coral raises the displayed class score, blue lowers it, and darker highlights "
            "mark larger effects. Hover a word for the exact value."
        )
        with st.expander("All word attributions"):
            st.dataframe(
                [{"Word": token.token, "Position": token.index + 1, "Attribution": token.score}
                 for token in tokens],
                hide_index=True,
            )
    else:
        ui.post_block(text)
        st.warning(result["explanation_error"])

    st.markdown("#### Moderator review")
    if result.get("reviewed"):
        st.success("Review saved to the local feedback database.")
        return
    alternatives = ("Confirm prediction", f"Correct to {LABELS[1 - label]}")
    decision = st.segmented_control(
        "Decision", alternatives, default=alternatives[0],
        key="decision_" + result["id"], required=True,
    )
    st.caption("Saving records the normalized post and your decision in the local feedback database.")
    if st.button("Save review", type="primary", key="save_" + result["id"]):
        reviewed_label = label if decision == alternatives[0] else 1 - label
        review = Review(
            result["id"], text, result["model"], label, reviewed_label, score, threshold,
        )
        try:
            save_review(args.feedback, review)
        except (OSError, ValueError, sqlite3.Error) as error:
            st.error(f"Could not save review. {error}")
            return
        result["reviewed"] = True
        st.rerun()


def main():
    args = settings()
    ui.page("Post review", layout="centered")
    ui.header(
        "XAI-HSD · moderator workbench",
        "Post review",
        "Score one post, inspect the words behind the score, and record a decision.",
    )
    if not args.model.exists():
        st.error(f"The checkpoint at {args.model} is missing.")
        if args.backend == "pretrained":
            st.code("python -m xai_hsd.cli fetch-model")
        st.stop()

    with st.container(border=True):
        post = st.text_area(
            "Post text", height=140, max_chars=1500,
            placeholder="Paste the text of a public post…",
        )
        first, second = st.columns(2)
        method = first.segmented_control(
            "Explanation", ("shap", "lime"), default="shap",
            format_func=str.upper, required=True,
        )
        threshold = second.slider(
            "Hateful threshold", 0.01, 0.99, 0.50, 0.01, format="%.2f",
        )
        actions = st.columns(2)
        analyze = actions[0].button("Analyze post", type="primary", width="stretch")
        clear = actions[1].button("Clear", width="stretch")

    if clear:
        st.session_state.pop("result", None)
        st.rerun()

    if analyze:
        st.session_state.pop("result", None)
        text = normalize_text(post)
        if not text:
            st.warning("Enter a post to analyze.")
        else:
            try:
                config = args.model / "config.json" if args.model.is_dir() else args.model
                version = config.stat().st_mtime_ns
                model = cached_model(args.model, args.backend, args.device, version)
                score = float(model.predict_proba([text])[0, 1])
            except (OSError, ValueError, ImportError, RuntimeError) as error:
                st.error(f"Could not run classification. {error}")
            else:
                evidence = assess_evidence(model, text)
                result = {
                    "id": uuid4().hex,
                    "text": text,
                    "score": score,
                    "label": int(score >= threshold) if evidence.sufficient else None,
                    "method": method,
                    "threshold": threshold,
                    "reason": evidence.reason,
                    "model": f"{args.backend}:{args.model.name}@{version}",
                }
                if isinstance(model, PretrainedClassifier):
                    result["has_known_features"] = model.has_known_features(text)
                if evidence.sufficient:
                    with st.spinner("Computing word attributions…"):
                        try:
                            result["tokens"] = explain(text, model.predict_proba, method, result["label"])
                        except (ImportError, ValueError, RuntimeError) as error:
                            result["explanation_error"] = f"Word attributions are unavailable. {error}"
                st.session_state.result = result

    if "result" in st.session_state:
        render_result(st.session_state.result, args)
    elif not analyze:
        ui.empty(
            "Nothing analyzed yet",
            "Paste a post above, choose an explanation method, and select Analyze post.",
        )


if __name__ == "__main__":
    main()
