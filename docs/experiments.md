# Research experiments

The supplied slides define a phased research project. The current review app uses a third-party pretrained pickle. The source also provides a reproducible TF-IDF and linear SVM benchmark, a supervised multilingual transformer, offline binary-class preference tuning, and an experimental text-image classifier.

## Data and labels

The project's binary label is **1 for hateful** and **0 for non-hateful**. Map generic profanity or offensive content to 0 unless the source explicitly labels it hate. Benchmark datasets contain harmful language. Keep downloaded records in `data/`, which is ignored except for the small synthetic workflow fixture.

| Source | Format and mapping |
| --- | --- |
| `hatexplain` | Official `dataset.json` and adjacent `post_id_divisions.json`. Use majority class, skip ties, and align an aggregate hate rationale to whitespace token positions. |
| `davidson` | Official `labeled_data.csv`. Map Davidson class 0 to hate and classes 1 and 2 to non-hate. |
| `hasoc` | Use fine labels HATE, OFFN, PRFN, and NOT from a compatible edition. An undifferentiated HOF label is ambiguous for hate-only detection. |
| `csv` | CSV or TSV with required `text,label` and optional `id,language,group,rationales,split`. Use pipe-separated target groups and JSON for a binary rationale mask. |

Dataset sources: [HateXplain data at the pinned revision](https://github.com/hate-alert/HateXplain/tree/01d742279dac941981f53806154481c0e15ee686/Data), [Davidson dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/cb50f7e9eb855a99abc51423bf40df0e26d6fe30/data), [HASOC 2020 datasets](https://hasocfire.github.io/hasoc/2020/dataset.html).

The data loader applies Unicode normalization and preserves case, slang, and emoji. The default pickle applies its own additional preprocessing during inference. The split code deduplicates normalized posts and prioritizes official held-out assignments. If no explicit split is available, it makes a seeded, stratified 80/10/10 split.

## Evaluation with no training

```powershell
python -m xai_hsd.cli evaluate --source hatexplain --data data/dataset.json --report artifacts/pretrained-hatexplain.json --rationale-limit 30
```

Validation and test reports include accuracy, hate precision/recall/F1, macro F1, and a confusion matrix. Slices report counts and error rates by language and annotated target group when those values are present. An undefined rate appears as `null`. Compare rates with their denominators, especially for rare groups. The target group refers to who a post discusses.

Rationale evaluation compares the top supportive hate attributions with human highlights on annotated test posts. It reports mean precision, recall, F1, and IoU. SHAP uses exact or permutation estimates relative to a fully masked post. LIME fits a local weighted linear model on perturbed posts. [SHAP permutation explainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html) · [LIME base API](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_base)

The publisher does not specify the precise training split for the pickle. Evaluating it on a public corpus measures behavior on that corpus, with possible training overlap. Use an independently collected and labeled evaluation set for an out-of-sample claim. Validate Hindi and code-mixed language separately.

The downloaded model scored 62.8% accuracy, 42.4% hate precision, 56.9% hate recall, and 48.6% hate F1 on 1,923 deduplicated HateXplain test records at threshold 0.50. The model flagged 460 of 1,329 non-hateful posts. SHAP had mean token overlap F1 of 0.513 across a random sample of 30 hateful test posts with human rationales. The full local report is `artifacts/pretrained-hatexplain.json`.

### Evaluation dashboard

The Streamlit app includes an **Evaluation dashboard** page. It reads generated metric reports from `artifacts/` and compares validation and test accuracy, precision, recall, F1, macro F1, false-positive rate, false-negative rate, and confusion matrices. Add a companion scores file to enable ROC/AUC, precision-recall/average precision, and a threshold sweep from 0.00 to 1.00.

```powershell
python -m xai_hsd.cli evaluate --source hatexplain --data data/dataset.json --report artifacts/pretrained-hatexplain.json --predictions-report artifacts/pretrained-hatexplain.predictions.json
python -m streamlit run app.py
```

The main report stays compact; the `--predictions-report` file stores only labels and scores for each validation/test record. Keep the two files together so the dashboard can pair them automatically. Without a scores file, the dashboard still shows the metric tables and confusion matrices, and the Curves tab explains how to generate one. Reports can also be uploaded from the dashboard sidebar. For a local smoke test with synthetic data, replace the dataset arguments with `--data data/demo.csv`.

## Small-fixture model comparison

`xai-hsd fetch-data` downloads the pinned HateXplain corpus and writes `data/hatexplain-6k.csv`: 4,000 train, 1,000 validation, and 1,000 test posts drawn from the official divisions, stratified by label. It is a speed fixture for model selection, not a new benchmark, and the file stays out of version control because it is harmful text.

```powershell
python -m xai_hsd.cli fetch-data
python -m xai_hsd.cli compare --data data/hatexplain-6k.csv --report artifacts/model-comparison.json --save-best artifacts/best-model.joblib
```

The harness fits classical candidates on train, selects one threshold per model on validation (maximum macro F1 by default, or `--target-fpr` for a false-positive budget), and reports test only at that threshold. Reports include confusion counts, Brier score, expected calibration error, ROC AUC, and percentile bootstrap intervals for the headline metrics.

On the 6,000-post fixture, `char` TF-IDF with balanced logistic regression ranked first on validation macro F1 (0.824) and reached 0.810 test macro F1. The publisher pickle, evaluated under the same protocol, reached 0.612. The gap is mostly in-domain training data rather than architecture: every retrained linear model beat the pickle, and a Naive Bayes counts model still reached 0.768. Rationale overlap also improved, from the documented 0.513 mean F1 for the pickle to 0.666 for the retrained model on 8 sampled test posts.

The winner is calibrated with isotonic regression fitted on validation; that reduced test ECE from 0.150 to 0.046 and Brier score from 0.145 to 0.122. Isotonic output is a step function, so the raw selection threshold (0.49) maps to a calibrated threshold (0.44) that must be used with the saved model. `compare --save-best` writes both numbers into the report.

### Hard negatives, ensembles, and cross-validation

Three follow-up experiments are built into the harness:

- `--hard-negatives COPIES` wraps each unique hateful training post in a condemning counter-speech frame (quoting, reporting, moderating) and adds the result as a non-hateful training example. Generated rows are train-only, deduplicated by source post, and cycled through eight frames so one run never reuses a frame for the same post.
- `ensemble-lr-char` and `ensemble-lr-wordchar` average three logistic regressions fit on seeded bootstrap samples of the training split.
- `--cv-folds 5` additionally ranks models by the mean of a stratified 5-fold split of the *training* data. Each fold refits on its remaining rows, chooses a threshold on the untouched validation split, and scores only its held-out fold, so the test split is never used for selection.

| Model | Test macro F1, no hard negatives | Test macro F1, with hard negatives | Delta |
| --- | --- | --- | --- |
| lr-char | 0.810 | 0.796 | -0.013 |
| lr-wordchar | 0.805 | 0.785 | -0.020 |
| lr-word | 0.808 | 0.704 | -0.104 |
| svm-wordchar | 0.780 | 0.778 | -0.002 |
| nb-count | 0.768 | 0.650 | -0.118 |

Hard negatives did not help. Recall fell while precision barely moved (`lr-char`: precision 0.702 to 0.712, recall 0.793 to 0.728), which fits this corpus: many HateXplain hateful posts are themselves quoting someone, so teaching that a slur inside a quote is harmless removes a boundary that legitimately separates the classes. For `lr-char` the change sits inside the bootstrap interval ([0.783, 0.836] without, [0.768, 0.822] with), so the honest reading is "no improvement, plausibly a small loss" rather than "hard negatives are harmful". The flag stays available for corpora where quoting is the dominant source of false positives; the shipped model is still trained without it.

| Model | Test macro F1 | 95% CI | Test precision | Test recall | Fit time |
| --- | --- | --- | --- | --- | --- |
| ensemble-lr-wordchar | 0.811 | [0.784, 0.837] | 0.778 | 0.693 | 10.0s |
| lr-char | 0.810 | [0.783, 0.836] | 0.702 | 0.793 | 0.9s |
| lr-wordchar | 0.805 | [0.778, 0.830] | 0.693 | 0.790 | 3.6s |
| ensemble-lr-char | 0.802 | [0.773, 0.827] | 0.729 | 0.722 | 5.8s |
| svm-wordchar | 0.780 | [0.750, 0.807] | 0.676 | 0.728 | 5.7s |

Bagging moved test macro F1 by at most 0.006 in either direction, well inside the intervals, at 6-10 times the fit cost. It did change the operating point: ensembles trade recall for precision at the same selected threshold. Under 5-fold ranking in the hard-negative run, `lr-char` (0.787) and both ensembles (0.778 and 0.777) landed within 0.01 of each other, so the single model remains the default. Where fold ranking and test ranking disagree, the disagreement is the useful part: one 1,000-row validation split cannot separate models that differ by less than roughly 0.03 macro F1.

### Single-post evidence guard

A linear text model with no usable features falls back to its class prior, which prints as a confident number. `xai_hsd/guard.py` refuses that case: `assess_evidence` reports the word count and the number of fitted features the text activates, and the interface returns `insufficient_evidence` with no verdict when either is below a threshold (8 words, 3 features by default).

Measured on the fixture model, `"have a nice day"` produces 4 words and is refused, while a 12-word post built from training vocabulary passes. The feature count is a strong signal for word-level candidates and a weak one for character n-grams, which still match on unfamiliar strings; for character models the word count is what actually protects the interface, so a passing feature count is necessary rather than sufficient evidence.

Caveats for the write-up: one seed, one fixture, and hyperparameters that were not tuned. The bootstrap intervals cover sampling noise only, not annotator disagreement. Promote a claim to the full corpus (and ideally to a corpus the model was not trained on) before presenting it as a result.

### Account-level aggregation

`xai-hsd account` collects a public Reddit timeline and applies the calibrated model per post. It reports the share of flagged posts with a Wilson interval, suppresses the share below a minimum post count, removes duplicates and very short posts, and lists the highest-scoring posts with links as evidence. It does not produce a conclusion about an account: a hateful post is not a hateful person, and the returned `signal` values are relative to an optional `--reference-rate` that you must justify from your own corpus.

### Where the timeline comes from

Two sources implement the same interface, and `--source` selects one:

- **`archive`** (default when no credentials are set) reads the [Arctic Shift](https://arctic-shift.photon-reddit.com) archive over plain HTTP with no key and no registration. Pagination walks backwards with a `before` cursor, since the archive has no listing cursor. The archive caps a page at 100 records, trails live Reddit by roughly 36 hours, and is one volunteer-run server: a `422` is its documented "query timed out" reply and is retried with backoff, a persistent `429`/`422` surfaces as "the archive is overloaded", and there is no uptime guarantee.
- **`api`** uses the official OAuth API. Reddit stopped accepting new app registrations on 31 October 2026 and has announced the Data API closes in March 2027, so this only works for projects that already hold credentials. Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` from a **script** app at <https://www.reddit.com/prefs/apps>, or pass `--credentials PATH` with the same three keys in a JSON object (gitignored, never echoed). The id sits directly under the app name and the secret is labelled `secret`; the client-credentials grant reads public data and stores no Reddit password.

```powershell
python -m xai_hsd.cli account --account spez --backend baseline --model artifacts/best-model.joblib --threshold 0.44 --limit 25 --min-posts 5
```

Responses are cached under `data/raw/reddit/`, with archive reads in an `archive/` subfolder so the two sources never mix. Run with `--limit 25 --min-posts 5` first: the default `--min-posts 20` will refuse to report on a short timeline, and the refusal message is the intended behaviour, not an error. Archive coverage of brand-new accounts can be thin, which produces exactly that refusal.

### Bluesky timelines

`--platform bluesky` reads `app.bsky.feed.getAuthorFeed` from the public AppView at `public.api.bsky.app`, which needs no key, app password, or login. Each page returns at most 100 posts and the client follows the returned `cursor` backwards until the limit is met. Only the author's own words are scored: reposts are skipped because the text belongs to someone else, as are empty and image-only posts. Pinned posts are excluded so the timeline stays chronological, and replies are kept. Plain handles, `@handles`, `https://bsky.app/profile/...` URLs, and DIDs are accepted. Reads are cached under `data/raw/bluesky/`.

```powershell
python -m xai_hsd.cli account --platform bluesky --source public --account bsky.app --backend baseline --model artifacts/best-model.joblib --threshold 0.44 --limit 100 --min-posts 5
```

The public AppView rate limits by IP and refuses some accounts (blocked, deactivated, private, or taken down); both surface as explicit messages rather than an empty dossier. Posting and any private data still require an authenticated session, which this read-only path does not use.

## Optional model development

[Classroom training script](../train_baseline.py) walks through the original HateXplain SVM experiment using the [shared model implementation](../xai_hsd/baseline.py).

Use a separate training environment to reproduce experiments. Select the model and operating threshold with validation data and reserve the test set for the frozen comparison. The two text backends share the same dataset and metrics interfaces.

```powershell
python -m xai_hsd.cli train --data data/dataset.json --source hatexplain --output artifacts/svm.joblib --report artifacts/svm.json
python -m pip install -e ".[models]"
python -m xai_hsd.cli train --backend transformer --data data/dataset.json --source hatexplain --output artifacts/xlmr --device cuda --report artifacts/xlmr.json
```

The transformer fine-tunes XLM-RoBERTa for two labels and selects the checkpoint by validation macro F1. For multilingual claims, include labeled Hindi and Hinglish data in the experiment and report the language slices. [XLM-RoBERTa model card](https://huggingface.co/FacebookAI/xlm-roberta-base)

### Offline moderation feedback

Use the app with a trained checkpoint and record independently reviewed posts. Export the resulting SQLite database to the training environment and run tuning with the source dataset and split settings used to train the reference checkpoint.

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 -- --backend transformer --model artifacts/xlmr
python -m xai_hsd.cli tune --backend transformer --model artifacts/xlmr --output artifacts/xlmr-reviewed --data data/dataset.json --source hatexplain --feedback artifacts/reviews.sqlite3 --device cuda --report artifacts/feedback-comparison.json
```

The reviewed label is the preferred binary answer and the opposite label is the rejected answer. Transformer tuning uses the change in preferred-to-rejected log odds relative to a frozen reference, as in DPO. The SVM variant fits a regularized log-odds adjustment over its frozen TF-IDF features. The tuning command excludes normalized matches to validation and test examples, deduplicates repeated reviews, saves the candidate separately, and reports before/after metrics at the same threshold. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

### Image and caption experiments

Prepare a manifest with `id,text,image,label,split` and optional `language,group`. Use image paths relative to the manifest and assign train, validation, and test splits containing both labels. For datasets with benign confounders, keep related examples together where the dataset's official splits permit it.

```powershell
python -m xai_hsd.multimodal train --manifest data/memes.csv --out artifacts/meme.joblib
python -m xai_hsd.multimodal predict --model artifacts/meme.joblib --image data/example.png --text "Caption text"
```

The experimental model fuses a frozen Vision Transformer embedding with character TF-IDF caption features. Its evaluation compares the fused model with caption-only and image-only models. The manifest provides caption text. This route has its own CLI and metrics. [Hateful Memes benchmark](https://arxiv.org/abs/2005.04790)
