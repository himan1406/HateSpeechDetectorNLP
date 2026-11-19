# Explainable hate speech detection

A local NLP research prototype based on the project slides. It classifies a social media post, shows which words influenced the prediction with SHAP or LIME, and records a moderator's confirmation or correction.

## Run the moderator app

The default model is a downloaded scikit-learn pickle. Inference, explanations, and review storage run on this computer. The app does not fit a model when it starts or when a moderator saves a decision.

From the project root on Windows, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m xai_hsd.cli fetch-model
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the address printed by Streamlit. Enter a post, choose SHAP or LIME, then select **Analyze post**. Coral highlights support the displayed class and blue highlights oppose it. Under **Moderator review**, confirm or correct the classification and save the decision.

For inference from the command line:

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli predict "People of every religion are welcome here" --explain lime
```

The app stores the normalized post, original prediction, probability, threshold, model version, and review decision in `artifacts/reviews.sqlite3`. Confirmations record reward +1 and corrections record −1.

Single posts are guarded against confident nonsense. If a post has fewer than 8 words, or its text activates fewer than 3 features from the training vocabulary, `predict` returns `"label": "insufficient_evidence"` with `hate_score: null` and a reason instead of a verdict, and the app shows a warning rather than a label, explanation, or review form. The feature check is strict for word-level models and loose for character n-grams, so a short post is the case that is reliably caught.

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli predict "have a nice day" --backend baseline --model artifacts/best-model.joblib
```

## Evaluation dashboard

Start the app normally; an **Evaluation dashboard** page appears in the Streamlit sidebar. It reads JSON reports from `artifacts/`, or you can upload a report from the sidebar. Generate a report and its optional companion scores file with:

```powershell
python -m xai_hsd.cli evaluate --data data/demo.csv --report artifacts/demo.json --predictions-report artifacts/demo.predictions.json
```

The `--report` file contains accuracy, precision, recall, F1, macro F1, error rates, and confusion matrices. `--predictions-report` saves the per-post scores needed for ROC/AUC, precision-recall, and threshold curves. The dashboard compares validation and test results; `data/demo.csv` is only a synthetic workflow smoke test.

## Downloaded model

`artifacts/model.pkl` is the publisher's TF-IDF + logistic regression pipeline. The downloader and loader check its pinned SHA-256 and the loader restricts the pickle to its known NumPy, SciPy, and scikit-learn classes. Use the declared Python and scikit-learn versions for compatibility. [Publisher model card](https://huggingface.co/pajola/hate-speech-detection-lr-tfidf/tree/8523992895f5145361e49212af7f1d6c94b8f7a9)

The application uses the publisher's hate-speech probability as the hate score and adds the publisher's offensive and neither probabilities for the non-hateful score. The cutoff defaults to 0.50. The model card describes social-media tweets and an English-oriented preprocessing pipeline that removes emoji, punctuation, hashtags, and mentions. Hindi and Hinglish inputs can be entered, but this checkpoint's performance on them is unverified. Attribution scores reflect the model's response to token deletion and can be used alongside the original text during review.

## Retrain and compare models

The downloaded pickle is kept for reference, but it is no longer the best option. A small, split-preserving HateXplain slice and a comparison harness are built in:

```powershell
.\\.venv\\Scripts\\python.exe -m xai_hsd.cli fetch-data
.\\.venv\\Scripts\\python.exe -m xai_hsd.cli compare --data data/hatexplain-6k.csv --report artifacts/model-comparison.json --save-best artifacts/best-model.joblib
```

`fetch-data` downloads the public corpus and writes a 6,000-post fixture that keeps the official train/validation/test assignment (30.9% hateful in every split). `compare` trains each candidate, picks the operating threshold on validation, and reports the test split only at that threshold.

Three options extend the same command: `--hard-negatives 1` adds counter-speech copies of hateful training posts as non-hateful examples, `ensemble-lr-char` and `ensemble-lr-wordchar` are bootstrap-bagged three-member ensembles, and `--cv-folds 5` ranks models by a stratified 5-fold split of the training data. On this fixture none of them beat the single `lr-char` model: hard negatives cost recall (0.793 to 0.728) and every ensemble stayed inside its 95% interval at 6-10 times the fit cost. Details and numbers are in `docs/experiments.md`.

On the fixture:

| Model | Test macro F1 | Test precision | Test recall | Test ROC AUC | Test ECE |
| --- | --- | --- | --- | --- | --- |
| char 2-5 gram TF-IDF + logistic regression | 0.810 | 0.702 | 0.793 | 0.881 | 0.150 |
| word + char TF-IDF + logistic regression | 0.805 | 0.693 | 0.790 | 0.886 | 0.136 |
| word TF-IDF + logistic regression | 0.808 | 0.774 | 0.686 | 0.873 | 0.173 |
| word + char TF-IDF + calibrated SVM | 0.780 | 0.676 | 0.728 | 0.867 | 0.061 |
| word counts + Complement Naive Bayes | 0.768 | 0.702 | 0.647 | 0.835 | 0.147 |
| publisher pickle | 0.612 | 0.441 | 0.583 | 0.682 | 0.151 |

The winning model is fitted with isotonic calibration on the validation split, which cut test expected calibration error from 0.150 to 0.046. Because the saved model emits calibrated scores, use its `saved_threshold` (0.44 here), not the raw selection threshold, when running it:

```powershell
.\\.venv\\Scripts\\python.exe -m xai_hsd.cli evaluate --backend baseline --model artifacts/best-model.joblib --data data/hatexplain-6k.csv --threshold 0.44 --report artifacts/trained-lr-char.json --predictions-report artifacts/trained-lr-char.predictions.json --rationale-limit 30
.\\.venv\\Scripts\\python.exe -m streamlit run app.py -- --backend baseline --model artifacts/best-model.joblib
```

Evaluation reports now include Brier score, expected calibration error, and a bootstrap interval for F1, and the dashboard has a **Slices** tab for language and target-group error rates.

Use `--target-fpr 0.05` instead of the default objective to choose the most sensitive threshold that keeps the false-positive rate at or below a moderation budget.

## Account review

`xai-hsd account` scores a public Reddit or Bluesky timeline and reports a flagged share with a 95% Wilson interval. Neither source needs a key:

```powershell
.\\.venv\\Scripts\\python.exe -m xai_hsd.cli account --account someuser --backend baseline --model artifacts/best-model.joblib --threshold 0.44 --min-posts 20 --reference-rate 0.10 --report artifacts/someuser.account.json
```

Reddit stopped accepting new API applications on 31 October 2026, so the default source is the [Arctic Shift](https://arctic-shift.photon-reddit.com) archive of public submissions and comments. It is volunteer-run, has no uptime guarantee, and trails live Reddit by roughly 36 hours. If you already hold Reddit API credentials, `--source api` (or `--credentials .reddit-credentials.json`) uses the live OAuth API instead; `--source archive` forces the archive.

Responses are cached under `data/raw/reddit/` (archive reads land in an `archive/` subfolder) and only public submissions and comments are read. The command reports `insufficient_data` below `--min-posts`, never emits a verdict about a person, and its Streamlit page is **Account review**. Quotes, counter-speech, and reclaimed language can all be flagged, so the dossier is triage evidence for a human reviewer.

Bluesky needs no key or login either. `--platform bluesky` reads the public AT Protocol AppView (`app.bsky.feed.getAuthorFeed`) for an account's own posts and replies, skipping reposts because those words belong to someone else, along with empty and image-only posts:

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli account --platform bluesky --account bsky.app --backend baseline --model artifacts/best-model.joblib --threshold 0.44 --min-posts 20 --report artifacts/bsky.account.json
```

The handle can be plain, `@`-prefixed, a `https://bsky.app/profile/...` URL, or a DID. Reads are cached under `data/raw/bluesky/`. The same aggregation caveats apply: a flagged post is not a verdict about the person, and the **Account review** page collects from either platform.

## Research code

[LaTeX research paper](docs/research_paper.tex) contains six main sections with page breaks and 23 embedded references. Upload the single file to Overleaf and select pdfLaTeX. The author and affiliation follow the supplied presentation. The paper uses the recorded pretrained-model evaluation and identifies experiments whose performance remains unmeasured.

[Experiment guide](docs/experiments.md) covers dataset preparation, evaluation and rationale overlap, the optional SVM and multilingual transformer, offline preference tuning, and the image-plus-text model. The downloaded pickle remains frozen in the moderator app. Use the training paths in a separate compute environment when you have approved research data and moderator reviews.
