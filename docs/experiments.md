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
