# News Credibility MVP with LIAR

This repository is a compact machine learning MVP for binary credibility classification of short political/news-related claims from the LIAR dataset.

The project predicts credibility for short LIAR-style textual claims, not full articles. It does not perform real-time fact checking.

## Setup

```bash
uv sync
```

## Run the project

```bash
uv run jupyter nbconvert --to notebook --execute data_exploration.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
uv run pytest tests -v
```

## Dataset

This project uses the LIAR dataset. LIAR contains short political statements labeled by PolitiFact using six truthfulness classes. The MVP converts the dataset into a binary credibility task and drops `half-true` from the main experiment.

## Binary Label Mapping

| Original label | Binary label   | Numeric label |
| -------------- | -------------- | ------------: |
| `pants-fire`   | `not_credible` |             0 |
| `false`        | `not_credible` |             0 |
| `barely-true`  | `not_credible` |             0 |
| `mostly-true`  | `credible`     |             1 |
| `true`         | `credible`     |             1 |
| `half-true`    | dropped        |           n/a |

## Models

1. Dummy majority-class baseline.
2. TF-IDF + Logistic Regression.
3. Sentence-transformer embeddings + Logistic Regression.
4. Sentence embeddings + GBM.

## Final Outputs

- `data/processed/liar_binary.csv`
- `reports/metrics_summary.csv`
- `reports/tfidf_top_ngrams.csv`
- `reports/tfidf_example_explanations.csv`
- `reports/bertopic_topic_risk.csv`
- `reports/error_analysis.md`
- confusion matrices in `reports/figures/`
- saved model artifacts in `models/`, including the dummy baseline, TF-IDF model, sentence embedding logistic regression model, and sentence embedding GBM model

## Model Modules

The four supervised model definitions live in `src/models/`, one module per model. Each model class exposes `__init__`, `fit`, `predict`, `save`, and `load`.

## Limitations

1. LIAR contains short statements, not full articles.
2. Binary mapping loses nuance.
3. Dropping `half-true` makes the task cleaner but less complete.
4. Political-domain bias is likely.
5. The models classify claim credibility patterns; they do not retrieve evidence or fact check in real time.
6. BERTopic is used mainly for topic-risk interpretation, not as the primary classifier in the final notebook.

## Suggested Presentation Outline

1. Problem definition
2. Why the MVP uses LIAR
3. Binary label mapping
4. Dataset exploration
5. Model 1: Dummy baseline
6. Model 2: TF-IDF + Logistic Regression
7. Model 3: Sentence embeddings + Logistic Regression
8. Model 4: Sentence embeddings + GBM
9. Evaluation results
10. Explainability examples
11. Error analysis
12. Limitations
13. What would be improved with more time
