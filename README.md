# Explainable hate speech detection

A local NLP research prototype that classifies a social-media post, shows which words moved the prediction with SHAP or LIME, and records a moderator's confirmation or correction. It can also summarize a public Reddit or Bluesky timeline as a flagged share with a confidence interval, and explore evaluation reports in a dashboard.

Everything runs on your machine: inference, explanations, and review storage. There is no hosted service and no account to create.

- **Post review** — score one post, inspect the word attributions, save a moderator decision.
- **Account review** — collect a public Reddit or Bluesky timeline and report a measured flagged share. No API key.
- **Evaluation** — accuracy, slices, calibration, confusion matrices, and threshold curves from JSON reports.

## Requirements

- **Python 3.10–3.12.** Python 3.13+ is not supported because the project pins scikit-learn 1.4.2.
- Windows, macOS, or Linux.
- Internet access on first run, to download the model (and optionally the training data).
- About 2 GB of free disk space for the virtual environment and model files.

## Quickstart

```bash
git clone https://github.com/himan1406/HateSpeechDetectorNLP.git
cd HateSpeechDetectorNLP
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m xai_hsd.cli fetch-model
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m xai_hsd.cli fetch-model
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the address Streamlit prints (usually <http://localhost:8501>). `fetch-model` downloads the 3.7 MB pretrained pickle to `artifacts/model.pkl` and checks its pinned SHA-256. If the app says the checkpoint is missing, run that command again.

> The downloaded pickle is the publisher's baseline, not the strongest model in this repo. To use the trained model described below, start the app with:
>
> ```powershell
> .\.venv\Scripts\python.exe -m streamlit run app.py -- --backend baseline --model artifacts/best-model.joblib
> ```

## Using the app

### Post review

Enter a post, choose SHAP or LIME, then select **Analyze post**. Coral highlights support the displayed class, blue highlights oppose it, and darker marks mean a larger effect. Under **Moderator review**, confirm or correct the classification and save the decision.

The app stores the normalized post, original prediction, probability, threshold, model version, and decision in `artifacts/reviews.sqlite3`. Confirmations record reward +1 and corrections record −1.

Single posts are guarded against confident nonsense. If a post has fewer than 8 words, or activates fewer than 3 features from the training vocabulary, the classifier returns `"label": "insufficient_evidence"` with `hate_score: null` and a reason. The app then shows a warning instead of a label, explanation, or review form.

### Account review

Collect a public timeline or paste posts, then read the flagged share, its 95% Wilson interval, and the highest-scoring posts. A dossier is triage evidence for a human reviewer, never a verdict about a person. The interval is compared with an optional reference rate; without one, no comparison is made.

Reddit uses the public [Arctic Shift](https://arctic-shift.photon-reddit.com) archive by default because Reddit stopped accepting new API applications on 31 October 2026. The archive trails live Reddit by roughly 36 hours. Existing Reddit API credentials can be used instead through `--source api`.

Bluesky uses the keyless public AppView (`app.bsky.feed.getAuthorFeed`). Reposts are skipped because the text belongs to someone else; the author's own replies are kept.

### Evaluation

The page reads metric reports from `artifacts/` or a JSON file you upload. Generate one from the CLI:

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli evaluate --data data/demo.csv --report artifacts/demo.json --predictions-report artifacts/demo.predictions.json
```

`data/demo.csv` is a tiny synthetic smoke test, not a research result. `--predictions-report` saves per-post scores for ROC, precision-recall, and threshold curves.

## Command line

```powershell
# Classify and explain one post
.\.venv\Scripts\python.exe -m xai_hsd.cli predict "People of every religion are welcome in this community" --explain lime

# Assess Reddit and Bluesky timelines (no keys needed)
.\.venv\Scripts\python.exe -m xai_hsd.cli account --account someuser --min-posts 20
.\.venv\Scripts\python.exe -m xai_hsd.cli account --platform bluesky --account bsky.app --min-posts 20

# Write a dossier for the app's Saved tab
.\.venv\Scripts\python.exe -m xai_hsd.cli account --platform bluesky --account bsky.app --backend baseline --model artifacts/best-model.joblib --threshold 0.44 --min-posts 20 --report artifacts/bsky.account.json
```

On macOS/Linux, replace `.\\.venv\\Scripts\\python.exe` with `python` after activating the environment. Run any command with `--help` for its options.

Timelines are cached under `data/raw/reddit/` and `data/raw/bluesky/`; `--refresh` fetches again. Only public posts are read.

## Train the stronger model (optional)

The repository ships code to build a 6,000-post HateXplain fixture and compare candidate models. This downloads the public corpus and takes a few minutes:

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli fetch-data
.\.venv\Scripts\python.exe -m xai_hsd.cli compare --data data/hatexplain-6k.csv --report artifacts/model-comparison.json --save-best artifacts/best-model.joblib
```

`fetch-data` keeps the official train/validation/test assignment (30.9% hateful in every split). `compare` picks the operating threshold on validation and reports test results only at that threshold.

The winning model is a character 2–5 gram TF-IDF + logistic regression pipeline, fitted with isotonic calibration on the validation split. Calibration cut test expected calibration error from 0.150 to 0.046. Because the saved model emits calibrated scores, use its `saved_threshold` (0.44 here), not the raw selection threshold:

```powershell
.\.venv\Scripts\python.exe -m xai_hsd.cli evaluate --backend baseline --model artifacts/best-model.joblib --data data/hatexplain-6k.csv --threshold 0.44 --report artifacts/trained-lr-char.json --predictions-report artifacts/trained-lr-char.predictions.json --rationale-limit 30
.\.venv\Scripts\python.exe -m streamlit run app.py -- --backend baseline --model artifacts/best-model.joblib
```

Comparison-harness results on the fixture:

| Model | Test macro F1 | Test precision | Test recall | Test ROC AUC | Test ECE |
| --- | --- | --- | --- | --- | --- |
| char 2–5 gram TF-IDF + logistic regression | 0.810 | 0.702 | 0.793 | 0.881 | 0.150 |
| word + char TF-IDF + logistic regression | 0.805 | 0.693 | 0.790 | 0.886 | 0.136 |
| word TF-IDF + logistic regression | 0.808 | 0.774 | 0.686 | 0.873 | 0.173 |
| word + char TF-IDF + calibrated SVM | 0.780 | 0.676 | 0.728 | 0.867 | 0.061 |
| word counts + Complement Naive Bayes | 0.768 | 0.702 | 0.647 | 0.835 | 0.147 |
| publisher pickle | 0.612 | 0.441 | 0.583 | 0.682 | 0.151 |

Hard negatives, bootstrap ensembles, `--cv-folds 5`, and `--target-fpr` are also supported. On this fixture none beat the single `lr-char` model: hard negatives cost recall (0.793 to 0.728), and every ensemble stayed inside its 95% interval at 6–10 times the fit cost. Details are in [docs/experiments.md](docs/experiments.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

No test data or API keys are required. The HTTP tests use local fixture servers.

## Project layout

```
app.py                     Streamlit entry point (Post review)
pages/1_Evaluation.py      Evaluation dashboard
pages/2_Account_review.py  Reddit/Bluesky account dossiers
xai_hsd/                   Package: CLI, models, explanations, collectors, UI helpers
tests/                     unittest suite
docs/                      Experiment guide and LaTeX research paper
data/demo.csv              Synthetic smoke-test data
pyproject.toml             Dependencies and the xai-hsd console script
```

`artifacts/`, `data/raw/`, the virtual environment, and local secrets are gitignored.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named streamlit` or `xai_hsd` | Run `pip install -e .` with the environment's Python. |
| `The checkpoint at artifacts/model.pkl is missing` | Run `python -m xai_hsd.cli fetch-model`. |
| Python 3.13+ fails to install dependencies | Create the environment with Python 3.12. |
| A short or unusual post returns no verdict | The insufficient-evidence guard is working; add more text. |
| Reddit collection is slow or fails | Arctic Shift is volunteer-run. Retry, or use Reddit API credentials with `--source api`. |
| Bluesky returns 429 | The public AppView is rate-limiting your IP; wait a few minutes. |
| Stale timeline data | Delete `data/raw/<platform>/` or pass `--refresh`. |
| Reset moderator feedback | Delete `artifacts/reviews.sqlite3`. |
| Transformer, image, or multilingual extras | `pip install -e ".[models]"` (installs PyTorch and Transformers). |

## Research notes

The downloaded model is the publisher's TF-IDF + logistic regression pipeline. The loader restricts the pickle to known NumPy, SciPy, and scikit-learn classes and verifies its SHA-256. It was trained on English social-media text that had emoji, punctuation, hashtags, and mentions removed; Hindi and Hinglish inputs are accepted but their performance is unverified.

The [research paper](docs/research_paper.tex) is a single LaTeX file for pdfLaTeX/Overleaf. The [experiment guide](docs/experiments.md) covers dataset preparation, rationale overlap, the optional SVM and multilingual transformer, offline preference tuning, and the image-plus-text model.

This is a research prototype, not a moderation authority. Quotes, counter-speech, reclaimed language, and news reporting are routinely misclassified. Automated flags require human review.
