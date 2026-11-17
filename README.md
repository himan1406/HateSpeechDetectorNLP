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

## Downloaded model

`artifacts/model.pkl` is the publisher's TF-IDF + logistic regression pipeline. The downloader and loader check its pinned SHA-256 and the loader restricts the pickle to its known NumPy, SciPy, and scikit-learn classes. Use the declared Python and scikit-learn versions for compatibility. [Publisher model card](https://huggingface.co/pajola/hate-speech-detection-lr-tfidf/tree/8523992895f5145361e49212af7f1d6c94b8f7a9)

The application uses the publisher's hate-speech probability as the hate score and adds the publisher's offensive and neither probabilities for the non-hateful score. The cutoff defaults to 0.50. The model card describes social-media tweets and an English-oriented preprocessing pipeline that removes emoji, punctuation, hashtags, and mentions. Hindi and Hinglish inputs can be entered, but this checkpoint's performance on them is unverified. Attribution scores reflect the model's response to token deletion and can be used alongside the original text during review.

## Research code

[LaTeX research paper](docs/research_paper.tex) contains six main sections with page breaks and 23 embedded references. Upload the single file to Overleaf and select pdfLaTeX. The author and affiliation follow the supplied presentation. The paper uses the recorded pretrained-model evaluation and identifies experiments whose performance remains unmeasured.

[Experiment guide](docs/experiments.md) covers dataset preparation, evaluation and rationale overlap, the optional SVM and multilingual transformer, offline preference tuning, and the image-plus-text model. The downloaded pickle remains frozen in the moderator app. Use the training paths in a separate compute environment when you have approved research data and moderator reviews.
