# Content-Aware-Sarcasm-Detection-Using-Fine-Tuned-Transformer-Models


A reproducible comparison of classical TF-IDF machine learning models against fine-tuned BERT and RoBERTa for sarcasm detection in news headlines, with SHAP interpretability, quantitative error analysis, and an explicit discussion of dataset bias.

An accompanying IEEE-style research paper is included in [`paper/`](paper/).

## Overview

Sarcasm expresses an intent that is the opposite of its literal meaning, which makes it a common failure mode for sentiment-analysis systems that rely on literal lexical polarity. This project implements and evaluates a complete, text-only sarcasm-detection pipeline on the [News Headlines Sarcasm Detection dataset](https://doi.org/10.1016/j.aiopen.2023.01.001) (Misra & Arora, *AI Open*, 2023) — 28,619 headlines drawn from *TheOnion* (sarcastic) and *HuffPost* (non-sarcastic).

Two model families are trained and evaluated under an identical, stratified 70/15/15 protocol:

- **Classical baselines** — Logistic Regression, Linear SVM, and Multinomial Naive Bayes on TF-IDF unigram/bigram features
- **Fine-tuned transformers** — `bert-base-uncased` and `roberta-base`

The project is presented as an empirical, comparative benchmark rather than a novel architecture or a state-of-the-art claim.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| RoBERTa | 93.709% | 0.92774 | 0.94097 | 0.93431 |
| BERT | 92.657% | 0.92277 | 0.92277 | 0.92277 |
| Logistic Regression | 84.004% | 0.82727 | 0.83866 | 0.83293 |
| Linear SVM | 83.980% | 0.83137 | 0.83178 | 0.83157 |
| Naive Bayes | 84.027% | 0.84794 | 0.80915 | 0.82809 |

Fine-tuned transformers outperform the strongest classical baseline by approximately 9–10 accuracy points and 10 F1 points on the same held-out 4,276-headline test set.

RoBERTa's confusion matrix: 2,094 true negatives, 149 false positives, 120 false negatives, 1,913 true positives. BERT's confusion matrix is exactly symmetric: 2,086 true negatives, 157 false positives, 157 false negatives, 1,876 true positives.

## Dataset and Label Bias

Sarcastic headlines come from *TheOnion* and non-sarcastic headlines come from *HuffPost*, so every label is inherited from the publishing outlet rather than assigned by an individual human sarcasm judgment. This creates a risk that a classifier partly learns source-specific lexical or stylistic cues rather than sarcasm itself.

| Statistic | Raw | After Cleaning |
|---|---|---|
| Total headlines | 28,619 | 28,503 |
| Non-sarcastic (label 0) | 14,985 | 14,951 |
| Sarcastic (label 1) | 13,634 | 13,552 |

Cleaning removes 0 null/empty records and 116 exact-duplicate headlines.

Sarcastic headlines skew toward generic, faux-official nouns (*man*, *report*, *area*, *nation*), while non-sarcastic headlines skew toward named political figures and reporting verbs (*trump*, *donald*, *says*) — a measurable stylistic difference between the two outlets. Sarcastic headlines are also slightly longer on average (mean 10.32 words vs. 9.82 words) with a heavier right tail.

Results in this project are reported as benchmark performance on this specific, source-labeled corpus, not as proof of general sarcasm understanding. See the paper's Discussion and Limitations sections for the full analysis.

## Methodology

1. Clean the raw dataset (28,619 → 28,503 headlines)
2. Stratified 70/15/15 split — 19,952 train / 4,275 validation / 4,276 test (seed = 42)
3. TF-IDF branch: unigrams + bigrams, `max_features=20000`, sublinear TF, smoothed IDF → Logistic Regression / Linear SVM / Multinomial Naive Bayes
4. Transformer branch: native subword tokenization, max length 64 → fine-tune `bert-base-uncased` / `roberta-base` (3 epochs, batch size 32, learning rate 2e-5, fp16, early stopping on validation F1)
5. Evaluate all models on the identical held-out test set
6. Interpret the best model (RoBERTa) with SHAP token attribution and a quantitative error taxonomy

Full equations and configuration details are in [`paper/main.tex`](paper/main.tex).

## Error Analysis

RoBERTa misclassifies 269 of 4,276 test headlines (6.29%) — 149 false positives and 120 false negatives.

All eight of the model's highest-confidence errors are sarcastic *TheOnion* headlines that read as plausible literal news in isolation (e.g. "those we lost in 2011", "it's shark week!"), indicating that a portion of this benchmark's difficulty comes from headlines whose sarcastic intent depends on context outside the text itself.

A SHAP token-attribution analysis on representative cases shows attribution concentrating on corporate-register buzzwords (e.g. "synergy", "eco-conscious") for correctly classified sarcastic headlines. A standalone, regenerated SHAP figure is intentionally not included unless produced from the actual fine-tuned checkpoint under a GPU environment — see the paper's Error Analysis section for the full explanation.

## Quickstart

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv && source .venv/bin/activate

pip install torch transformers datasets evaluate accelerate \
            scikit-learn pandas numpy matplotlib seaborn shap jupyter

jupyter notebook notebook/Final_Context_Sarcasm.ipynb
```

Regenerate the paper's figures:

```bash
cd paper/figures
python3 generate_figures.py
python3 generate_eda_figures.py
```

Compile the paper:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

See [`paper/README.md`](paper/README.md) for full build notes.

## Reproducibility

| | |
|---|---|
| Dataset | News Headlines Sarcasm Detection (Misra & Arora, 2023) |
| Split | Stratified 70/15/15 — 19,952 / 4,275 / 4,276 |
| Seed | 42 (Python `random`, NumPy, PyTorch) |
| TF-IDF | `max_features=20000`, `ngram_range=(1,2)`, `sublinear_tf=True`, smoothed IDF, L2-normalized |
| Transformers | `bert-base-uncased`, `roberta-base` — max length 64, batch 32, LR 2e-5, weight decay 0.01, 10% warm-up, fp16, early stopping (patience 2, val F1) |
| Hardware | Classical models: CPU. Transformers: single NVIDIA T4 GPU |

The source notebook contains two separate RoBERTa evaluations differing by about 0.68 accuracy points (93.709% vs. 93.031%) under an identical seed, most plausibly due to fp16/GPU non-determinism. Both runs are documented in the paper's Table VI rather than silently discarding the lower one.

## Limitations

- Scope is limited to two source publications (28,503 headlines); results may not transfer to other domains or languages without re-evaluation.
- Labels are source-level, not per-utterance human judgments (see Dataset and Label Bias above).
- No conversational, author, or multimodal context is used.
- SHAP interpretability evidence is limited to three qualitative case studies from the original notebook run.

Full discussion in `paper/main.tex`, Section XVI.

## Future Work

- Cross-dataset and cross-domain evaluation
- Article-body and source-identity context integration
- Multilingual and multimodal sarcasm detection
- Repeated-seed evaluation for proper confidence intervals
- A regenerated, corpus-wide SHAP attribution study

## Citation

```bibtex
@unpublished{lodh2026sarcasm,
  author = {Lodh, Darshita and Kumar, Nitish},
  title  = {Content-Aware Sarcasm Detection Using Fine-Tuned Transformer Models},
  note   = {KES Shroff College, Mumbai},
  year   = {2026}
}
```

Dataset citation:

```bibtex
@article{misra2023sarcasm,
  author  = {Misra, Rishabh and Arora, Prahal},
  title   = {Sarcasm detection using news headlines dataset},
  journal = {AI Open},
  volume  = {4},
  pages   = {13--18},
  year    = {2023},
  doi     = {10.1016/j.aiopen.2023.01.001}
}
```

## Authors

Darshita Lodh — KES Shroff College, Mumbai
Nitish Kumar — Assistant Professor, Dept. of IT, DS and AI, KES Shroff College, Mumbai

## License

Released under the [MIT License](LICENSE).
