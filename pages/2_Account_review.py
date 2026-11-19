"""Account dossier: aggregate public posts into a measured share, never a verdict."""

import json
import os
from pathlib import Path

import streamlit as st

from xai_hsd import ui
from xai_hsd.account import DISCLAIMER, account_handle, assess_posts
from xai_hsd.bluesky import BlueskyClient, BlueskyError, normalize_actor
from xai_hsd.cli import load_model
from xai_hsd.reddit import DEFAULT_CACHE as REDDIT_CACHE
from xai_hsd.reddit import ArchiveClient, RedditClient, RedditError

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
MODELS = {
    "Trained best model": (ARTIFACTS / "best-model.joblib", "baseline"),
    "Publisher pickle": (ARTIFACTS / "model.pkl", "pretrained"),
}
SIGNALS = {
    "above_reference": ("Above reference", "orange"),
    "below_reference": ("Below reference", "blue"),
    "indistinguishable_from_reference": ("Within reference", "gray"),
    "insufficient_data": ("Insufficient data", "gray"),
    "no_reference": ("No reference", "gray"),
}


@st.cache_resource
def cached_model(path, backend, version):
    return load_model(Path(path), backend)


def active_model():
    available = {name: value for name, value in MODELS.items() if value[0].is_file()}
    if not available:
        st.error("No model artifact found. Run `xai-hsd compare --save-best artifacts/best-model.joblib`.")
        st.stop()
    st.sidebar.markdown("#### Model")
    name = st.sidebar.selectbox("Model", list(available), label_visibility="collapsed")
    path, backend = available[name]
    st.sidebar.caption(f"`{path.relative_to(ROOT)}` · {backend}")
    version = path.stat().st_mtime_ns
    return cached_model(str(path), backend, version), f"{backend}:{path.name}"


def assessment_settings():
    st.sidebar.markdown("#### Assessment")
    threshold = st.sidebar.slider("Hateful threshold", 0.05, 0.95, 0.50, 0.05, format="%.2f")
    min_posts = st.sidebar.number_input(
        "Minimum posts", 5, 100, 5, 5,
        help="Refuse to summarize a timeline shorter than this.",
    )
    reference = st.sidebar.number_input(
        "Reference rate (optional)", 0.0, 1.0, 0.0, 0.01,
        help="Expected flagged share on a comparable corpus. Zero disables the comparison.",
    )
    return threshold, int(min_posts), reference or None


def render(report):
    counts = report["counts"]
    handle = account_handle(report) if report["account"] else "Pasted posts"
    signal_label, signal_color = SIGNALS.get(
        report["signal"], (report["signal"].replace("_", " ").title(), "gray")
    )

    with st.container(border=True):
        heading, status = st.columns([3, 1], vertical_alignment="center")
        with heading:
            st.markdown(f"### {handle}")
            meta = [report.get("source") or report["platform"].title()]
            meta.append(f"{counts['collected']} collected · {counts['scored']} scored")
            if counts["duplicates_removed"]:
                meta.append(f"{counts['duplicates_removed']} duplicates removed")
            if counts["excluded_short"]:
                meta.append(f"{counts['excluded_short']} too short")
            st.caption(" · ".join(meta))
        status.badge(signal_label, color=signal_color)

        if report["signal"] == "insufficient_data":
            st.warning(report["note"])
            return

        interval = report["interval"]
        columns = st.columns(3)
        columns[0].metric("Flagged share", f"{report['flagged_share']:.1%}")
        columns[1].metric("Mean score", f"{report['scores']['mean']:.2f}")
        columns[2].metric("Threshold", f"{report['threshold']:.2f}")
        ui.interval_bar(
            report["flagged_share"], interval["low"], interval["high"],
            reference=report.get("reference_rate"),
        )
        if report["signal"] == "above_reference":
            st.caption("The interval lies entirely above the reference rate.")
        elif report["signal"] == "below_reference":
            st.caption("The interval lies entirely below the reference rate.")
        elif report["signal"] == "indistinguishable_from_reference":
            st.caption("The interval contains the reference rate, so this account does not stand out.")

    st.markdown("#### Highest-scoring flagged posts")
    if report["evidence"]:
        st.dataframe(
            [
                {
                    "Score": row["hate_score"],
                    "Context": row.get("context") or row.get("subreddit") or "",
                    "Post": row.get("url") or "",
                    "Text": row["text"],
                }
                for row in report["evidence"]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", help="Model score for this post", min_value=0.0,
                    max_value=1.0, format="%.2f",
                ),
                "Context": st.column_config.TextColumn("Context", width="small"),
                "Post": st.column_config.LinkColumn("Post", display_text="Open", width="small"),
                "Text": st.column_config.TextColumn("Post text", width="large"),
            },
        )
    else:
        st.caption("No post reached the threshold.")

    with st.expander("How to read this assessment"):
        st.write(report.get("note", ""))
        st.write(report.get("disclaimer") or DISCLAIMER)
        st.caption(f"Model: {report.get('model', 'unknown')}")


def collect_panel(model, model_label, threshold, min_posts, reference):
    with st.container(border=True):
        platform = st.segmented_control(
            "Platform", ("Reddit", "Bluesky"), default="Reddit", required=True,
        )
        if platform == "Bluesky":
            has_credentials = False
            st.caption(
                "Public AppView, no key or login. Only the author's own posts and "
                "replies are scored; reposts of other people's text are skipped."
            )
            label, placeholder = "Bluesky handle or DID", "name.bsky.social"
        else:
            has_credentials = bool(
                os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")
            )
            if has_credentials:
                st.caption("Live API using the Reddit credentials in your environment.")
            else:
                st.caption(
                    "Public Arctic Shift archive, no key. It runs about 36 hours behind "
                    "live Reddit and is volunteer-run."
                )
            label, placeholder = "Reddit username", "spez"

        account = st.text_input(label, placeholder=placeholder)
        limit = st.number_input("Posts to fetch", 20, 500, 200, 20)
        submitted = st.button("Collect and assess", type="primary", width="stretch")

    if not submitted:
        return
    try:
        if platform == "Bluesky":
            client = BlueskyClient()
            account = normalize_actor(account)
        else:
            client = (
                RedditClient.from_env(cache_dir=REDDIT_CACHE)
                if has_credentials
                else ArchiveClient(cache_dir=REDDIT_CACHE)
            )
        posts = client.timeline(account, limit=int(limit))
        report = assess_posts(
            posts, model.predict_proba, threshold=threshold, account=account,
            platform=platform.lower(), model=model_label, min_posts=min_posts,
            reference_rate=reference, source=client.source,
        )
    except (BlueskyError, RedditError, ValueError, OSError) as error:
        st.error(f"Collection failed. {error}")
        return
    st.session_state.account_report = report


def paste_panel(model, model_label, threshold, min_posts, reference):
    with st.container(border=True):
        raw = st.text_area(
            "One post per line", height=180,
            help="Test the aggregation without touching a platform API.",
        )
        submitted = st.button("Assess posts", type="primary", width="stretch")
    if not submitted:
        return
    texts = [line for line in raw.splitlines() if line.strip()]
    if not texts:
        st.warning("Paste at least one post.")
        return
    st.session_state.account_report = assess_posts(
        texts, model.predict_proba, threshold=threshold, platform="manual",
        model=model_label, min_posts=min_posts, min_words=1,
        reference_rate=reference,
    )


def saved_panel():
    paths = sorted(ARTIFACTS.glob("*.account.json"))
    if not paths:
        st.caption("No saved dossiers yet. Generate one with `xai-hsd account --account NAME`.")
        return
    with st.container(border=True):
        name = st.selectbox("Dossier file", [path.name for path in paths])
        if st.button("Open dossier", type="primary", width="stretch"):
            try:
                report = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                st.error(f"Could not read {name}. {error}")
            else:
                st.session_state.account_report = report


def main():
    ui.page("Account review", layout="wide")
    ui.header(
        "XAI-HSD · public timeline triage",
        "Account review",
        "Summarize a public timeline as a flagged share with an interval. "
        "This is triage evidence, not a finding about a person.",
    )
    model, model_label = active_model()
    threshold, min_posts, reference = assessment_settings()

    mode = st.pills(
        "Mode", ("Collect", "Paste", "Saved"), default="Collect",
        label_visibility="collapsed",
    ) or "Collect"
    if mode == "Collect":
        collect_panel(model, model_label, threshold, min_posts, reference)
    elif mode == "Paste":
        paste_panel(model, model_label, threshold, min_posts, reference)
    else:
        saved_panel()

    if "account_report" in st.session_state:
        render(st.session_state.account_report)


if __name__ == "__main__":
    main()
