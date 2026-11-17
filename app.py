"""Local moderator review interface for XAI-HSD."""

import argparse
import sqlite3
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st

from xai_hsd.cli import LABELS, load_model
from xai_hsd.data import normalize_text
from xai_hsd.explain import explain
from xai_hsd.feedback import Review, save_review
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
    return '<div style="line-height:2;overflow-wrap:anywhere;">' + "".join(parts) + "</div>"


def render_prediction(result, args):
    text, score, label = result["text"], result["score"], result["label"]
    st.subheader("Prediction")
    first, second = st.columns(2)
    first.metric("Predicted label", LABELS[label])
    second.metric("Hate score", f"{score:.1%}")
    st.caption(f"Hateful at or above {result['threshold']:.0%} · {args.backend.title()} model")
    if not result.get("has_known_features", True):
        st.warning("The pretrained vocabulary has no matches for this post. The score comes from the model's class bias.")

    st.subheader(f"{result['method'].upper()} word attributions")
    tokens = result.get("tokens")
    if tokens is not None:
        st.markdown(highlighted_text(text, tokens), unsafe_allow_html=True)
        st.caption("Coral raises the displayed class score. Blue lowers it. Darker highlights indicate a larger effect.")
        most_influential = sorted(tokens, key=lambda token: abs(token.score), reverse=True)[:12]
        st.dataframe(
            [{"Word": token.token, "Position": token.index + 1, "Attribution": token.score}
             for token in most_influential],
            hide_index=True,
        )
    else:
        st.text(text)
        st.warning(result["explanation_error"])

    st.subheader("Moderator review")
    if result.get("reviewed"):
        st.success("Review saved.")
        return
    with st.form("review_" + result["id"]):
        decision = st.radio("Decision", ("Confirm prediction", "Correct to " + LABELS[1 - label]))
        st.caption("Saving records the normalized post and decision in the local feedback database.")
        submitted = st.form_submit_button("Save review")
    if submitted:
        reviewed_label = label if decision == "Confirm prediction" else 1 - label
        review = Review(
            result["id"], text, result["model"], label, reviewed_label, score, result["threshold"],
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
    st.set_page_config(page_title="Hate speech review", layout="centered")
    st.title("Hate speech review")
    st.caption("Classify a post, inspect the model’s word attributions, and record a moderator decision.")
    if not args.model.exists():
        st.error(f"The checkpoint at {args.model} is missing.")
        if args.backend == "pretrained":
            st.code("python -m xai_hsd.cli fetch-model")
        st.stop()

    with st.form("classify"):
        post = st.text_area("Social media post", height=135, max_chars=1500)
        options, limit = st.columns(2)
        method = options.selectbox("Explanation method", ("shap", "lime"), format_func=str.upper)
        threshold = limit.slider("Hateful threshold", 0.01, 0.99, 0.50, 0.01)
        analyze = st.form_submit_button("Analyze post", type="primary")

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
                result = {
                    "id": uuid4().hex,
                    "text": text,
                    "score": score,
                    "label": int(score >= threshold),
                    "method": method,
                    "threshold": threshold,
                    "model": f"{args.backend}:{args.model.name}@{version}",
                }
                if isinstance(model, PretrainedClassifier):
                    result["has_known_features"] = model.has_known_features(text)
                with st.spinner("Computing word attributions…"):
                    try:
                        result["tokens"] = explain(text, model.predict_proba, method, result["label"])
                    except (ImportError, ValueError, RuntimeError) as error:
                        result["explanation_error"] = f"Word attributions are unavailable. {error}"
                st.session_state.result = result

    if "result" in st.session_state:
        render_prediction(st.session_state.result, args)


if __name__ == "__main__":
    main()
