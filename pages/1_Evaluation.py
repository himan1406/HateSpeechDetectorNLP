"""Evaluation dashboard for generated XAI-HSD metric reports."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from xai_hsd import ui
from xai_hsd.evaluation import score_curves

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
METRICS = (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("Macro F1", "macro_f1"),
    ("False-positive rate", "false_positive_rate"),
    ("False-negative rate", "false_negative_rate"),
)


@st.cache_data(show_spinner=False)
def load_json(path, mtime_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_upload(upload):
    return json.loads(upload.getvalue().decode("utf-8"))


def report_candidates():
    if not ARTIFACTS.is_dir():
        return []
    rows = [
        path for path in ARTIFACTS.glob("*.json")
        if not path.name.endswith(".splits.json") and not path.name.endswith(".predictions.json")
    ]
    rows.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    valid = []
    for path in rows:
        try:
            if is_report(load_json(str(path), path.stat().st_mtime_ns)):
                valid.append(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return valid


def prediction_candidates():
    if not ARTIFACTS.is_dir():
        return []
    rows = [path for path in ARTIFACTS.glob("*.predictions.json")]
    rows.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    valid = []
    for path in rows:
        try:
            if is_prediction_report(load_json(str(path), path.stat().st_mtime_ns)):
                valid.append(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return valid


def is_report(data):
    if not isinstance(data, dict):
        return False
    return any(
        isinstance(data.get(split), dict) and isinstance(data[split].get("overall"), dict)
        for split in ("validation", "test")
    )


def is_prediction_report(data):
    if not isinstance(data, dict) or not isinstance(data.get("predictions"), dict):
        return False
    return any(
        isinstance(data["predictions"].get(split), dict)
        and isinstance(data["predictions"][split].get("labels"), list)
        and isinstance(data["predictions"][split].get("scores"), list)
        for split in ("validation", "test")
    )


def pct(value):
    return "n/a" if value is None else f"{float(value):.1%}"


def metric_table(report):
    rows = []
    for split in ("validation", "test"):
        overall = report.get(split, {}).get("overall", {})
        if not isinstance(overall, dict):
            continue
        row = {
            "Split": split.title(),
            "N": overall.get("count"),
            "Hateful": overall.get("positive_support"),
            "Non-hateful": overall.get("negative_support"),
        }
        for label, key in METRICS:
            row[label] = pct(overall.get(key))
        calibration = report.get(split, {}).get("calibration") or {}
        row["Brier"] = "n/a" if calibration.get("brier") is None else f"{calibration['brier']:.3f}"
        row["ECE"] = "n/a" if calibration.get("ece") is None else f"{calibration['ece']:.3f}"
        interval = (report.get(split, {}).get("intervals") or {}).get("f1") or {}
        row["F1 95% CI"] = (
            f"{interval['low']:.1%}–{interval['high']:.1%}" if interval else "n/a"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def slice_table(block):
    rows = []
    for name, metrics in (block or {}).items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            "Slice": name,
            "N": metrics.get("count"),
            "Hateful": metrics.get("positive_support"),
            "Accuracy": pct(metrics.get("accuracy")),
            "Precision": pct(metrics.get("precision")),
            "Recall": pct(metrics.get("recall")),
            "F1": pct(metrics.get("f1")),
            "FPR": pct(metrics.get("false_positive_rate")),
            "FNR": pct(metrics.get("false_negative_rate")),
        })
    return pd.DataFrame(sorted(rows, key=lambda row: row["N"] or 0, reverse=True))


def metric_cards(report):
    split = st.segmented_control(
        "Split", ("test", "validation"), default="test",
        format_func=str.title, key="split", required=True,
    ) or "test"
    other_split = "validation" if split == "test" else "test"
    overall = report.get(split, {}).get("overall", {})
    comparison = report.get(other_split, {}).get("overall", {})
    columns = st.columns(4)
    for column, (label, key) in zip(columns, METRICS[:4]):
        value = overall.get(key)
        previous = comparison.get(key)
        delta = None
        if value is not None and previous is not None:
            delta = f"{float(value) - float(previous):+.1%}"
        column.metric(label, pct(value), delta=delta, delta_color="normal", border=True)
    st.caption(
        f"Showing {split} · compared with {other_split} · "
        f"count {overall.get('count', 0)} · threshold {report.get('threshold', 'n/a')}"
    )
    return split


def confusion_panel(overall):
    matrix = overall.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        st.info("No confusion matrix was found in this report.")
        return
    table = pd.DataFrame(
        {"Predicted non-hateful": matrix[0], "Predicted hateful": matrix[1]},
        index=["Gold non-hateful", "Gold hateful"],
    )
    st.dataframe(table)
    counts = pd.DataFrame(
        {"Count": [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]},
        index=["True negative", "False positive", "False negative", "True positive"],
    )
    st.bar_chart(counts)


def metadata_caption(report):
    sizes = report.get("split_sizes", {})
    caption = (
        f"Backend: {report.get('backend', 'unknown')} · "
        f"Source: {report.get('source', 'unknown')}/{report.get('data_file', 'unknown')} · "
        f"Seed: {report.get('seed', 'unknown')} · "
        f"Report threshold: {report.get('threshold', 'unknown')} · "
        f"Split sizes: {sizes if sizes else 'unknown'}"
    )
    st.caption(caption)
    signature = report.get("split_sha256")
    if signature:
        st.caption(f"Split signature: `{signature[:16]}…`")


def select_report():
    upload = st.sidebar.file_uploader("Upload metric report", type="json", key="report_upload")
    if upload is not None:
        try:
            data = load_upload(upload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            st.sidebar.error(f"Invalid uploaded report: {error}")
            st.stop()
        if not is_report(data):
            st.sidebar.error("The uploaded file does not contain validation/test metrics.")
            st.stop()
        return data, None
    candidates = report_candidates()
    if not candidates:
        return None, None
    selected = st.sidebar.selectbox("Metric report", [path.name for path in candidates])
    path = ARTIFACTS / selected
    data = load_json(str(path), path.stat().st_mtime_ns)
    if not is_report(data):
        st.error(f"{path.name} does not contain validation/test metrics.")
        st.stop()
    return data, path


def select_predictions(report_path):
    upload = st.sidebar.file_uploader(
        "Upload prediction scores", type="json", key="predictions_upload"
    )
    if upload is not None:
        try:
            data = load_upload(upload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            st.sidebar.error(f"Invalid uploaded scores: {error}")
            st.stop()
        if not is_prediction_report(data):
            st.sidebar.error("The uploaded file does not contain per-record labels and scores.")
            st.stop()
        return data
    candidates = prediction_candidates()
    names = ["None"] + [path.name for path in candidates]
    preferred = f"{report_path.stem}.predictions.json" if report_path else None
    index = names.index(preferred) if preferred in names else 0
    selected = st.sidebar.selectbox("Prediction scores", names, index=index)
    if selected == "None":
        return None
    path = ARTIFACTS / selected
    data = load_json(str(path), path.stat().st_mtime_ns)
    if not is_prediction_report(data):
        st.error(f"{path.name} does not contain per-record labels and scores.")
        st.stop()
    return data


def curve_chart(values, labels, y_name, x_name):
    frame = pd.DataFrame({y_name: labels}, index=pd.Index(values, name=x_name))
    st.line_chart(frame)


def curves_panel(report, predictions):
    if predictions is None:
        st.info(
            "Generate a companion scores file with `xai-hsd evaluate --predictions-report ...` "
            "to unlock ROC, precision-recall, and threshold curves."
        )
        return
    split = st.segmented_control(
        "Split", ("test", "validation"), default="test",
        format_func=str.title, key="curve_split", required=True,
    ) or "test"
    row = predictions["predictions"].get(split, {})
    labels = row.get("labels", [])
    scores = row.get("scores", [])
    if not labels or not scores:
        st.info(f"No prediction scores were saved for the {split} split.")
        return
    expected = report.get("split_sizes", {}).get(split)
    if expected is not None and expected != len(labels):
        st.warning(f"Scores contain {len(labels)} rows, but the report says {expected}.")
    try:
        curves = score_curves(labels, scores)
    except ValueError as error:
        st.error(f"Could not compute curves: {error}")
        return

    default_threshold = float(report.get("threshold", 0.5))
    default_threshold = min(max(default_threshold, 0.01), 0.99)
    threshold = st.slider("Operating threshold", 0.01, 0.99, default_threshold, 0.01)
    sweep = curves["threshold_sweep"]
    current = min(sweep, key=lambda item: abs(item["threshold"] - threshold))
    best = max(sweep, key=lambda item: (item["f1"], item["macro_f1"], item["accuracy"]))

    roc = curves["roc"]
    precision_recall = curves["precision_recall"]
    first, second, third, fourth = st.columns(4)
    first.metric("ROC AUC", "n/a" if roc["auc"] is None else f"{roc['auc']:.3f}", border=True)
    second.metric(
        "Average precision",
        "n/a" if precision_recall["average_precision"] is None
        else f"{precision_recall['average_precision']:.3f}",
        border=True,
    )
    third.metric(f"Accuracy @{threshold:.2f}", pct(current["accuracy"]), border=True)
    fourth.metric(f"Recall @{threshold:.2f}", pct(current["recall"]), border=True)

    st.caption(
        f"Positive class is hateful. `score >= threshold` is hateful. "
        f"Highest hate F1 is {best['f1']:.1%} at threshold {best['threshold']:.2f}."
    )

    st.subheader("ROC curve")
    if roc["auc"] is None:
        st.info("ROC is undefined when the split has only one class.")
    else:
        curve_chart(roc["fpr"], roc["tpr"], "True-positive rate", "False-positive rate")

    st.subheader("Precision-recall curve")
    if precision_recall["average_precision"] is None:
        st.info("Precision-recall is undefined when the split has no hateful examples.")
    else:
        curve_chart(
            precision_recall["recall"], precision_recall["precision"], "Precision", "Recall"
        )

    st.subheader("Threshold sweep")
    sweep_frame = pd.DataFrame(
        {
            "Accuracy": [item["accuracy"] for item in sweep],
            "Precision": [item["precision"] for item in sweep],
            "Recall": [item["recall"] for item in sweep],
            "F1": [item["f1"] for item in sweep],
            "Macro F1": [item["macro_f1"] for item in sweep],
        },
        index=pd.Index([item["threshold"] for item in sweep], name="Threshold"),
    )
    st.line_chart(sweep_frame)
    st.dataframe(
        pd.DataFrame(
            [{**{"Threshold": f"{item['threshold']:.2f}"}, **{
                label: pct(item.get(key)) for label, key in METRICS
            }} for item in sweep]
        ),
        hide_index=True,
    )


def main():
    ui.page("Evaluation", layout="wide")
    ui.header(
        "XAI-HSD · measurement",
        "Evaluation",
        "Accuracy, recall, confusion matrices, and threshold curves from generated reports.",
    )

    with st.sidebar:
        st.markdown("#### Report source")
        st.caption("Runs: `xai-hsd evaluate --report ... --predictions-report ...`")
    report, report_path = select_report()
    if report is None:
        st.info(
            "No metric reports found. Generate one in `artifacts/`, or upload a JSON report "
            "from the sidebar. Example: `python -m xai_hsd.cli evaluate --data data/demo.csv "
            "--report artifacts/demo.json --predictions-report artifacts/demo.predictions.json`"
        )
        st.stop()
    predictions = select_predictions(report_path)

    metadata_caption(report)
    overview, slices, curves, about = st.tabs(
        ["Overview", "Slices", "Curves", "Guide"]
    )
    with overview:
        split = metric_cards(report)
        st.subheader("Validation and test")
        st.dataframe(metric_table(report), hide_index=True)
        st.caption(
            "Brier and expected calibration error (ECE) measure whether an 80% score really "
            "means 80% of posts are hateful. Lower is better. Intervals are a percentile "
            "bootstrap over posts and exclude annotation noise."
        )
        st.subheader(f"Confusion matrix ({split})")
        confusion_panel(report.get(split, {}).get("overall", {}))
    with slices:
        slice_split = st.segmented_control(
            "Split", ("test", "validation"), default="test",
            format_func=str.title, key="slice_split", required=True,
        ) or "test"
        block = report.get(slice_split, {})
        st.subheader(f"By language ({slice_split})")
        languages = slice_table(block.get("by_language"))
        if languages.empty:
            st.info("This report has no language slices.")
        else:
            st.dataframe(languages, hide_index=True)
        st.subheader(f"By annotated target group ({slice_split})")
        groups = slice_table(block.get("by_group"))
        if groups.empty:
            st.info("This report has no target-group slices.")
        else:
            st.dataframe(groups, hide_index=True)
            st.caption(
                "A post can name several groups, so these slices overlap and do not sum "
                "to the total. Compare rates with their denominators; `n/a` means the "
                "slice has no examples of the class the rate is defined over."
            )
    with curves:
        curves_panel(report, predictions)
    with about:
        st.markdown(
            """
            - **Accuracy**: correct decisions divided by all labelled posts.
            - **Precision**: share of flagged posts that are truly hateful.
            - **Recall**: share of hateful posts that were flagged.
            - **F1**: harmonic mean of precision and recall for the hateful class.
            - **Macro F1**: unweighted mean of hateful and non-hateful F1.
            - **FPR/FNR**: false-positive and false-negative rates; `n/a` when a class has no support.
            - **ROC AUC / average precision**: ranking quality from saved per-post scores.
            - **Threshold sweep**: metrics from 0.00 to 1.00 using `score >= threshold`.
            """
        )
        st.caption(
            "Reports are produced by `python -m xai_hsd.cli evaluate`. "
            "Add `--predictions-report PATH` to save the scores needed for curves."
        )


if __name__ == "__main__":
    main()
