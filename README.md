# News Credibility Prediction MVP

This repository contains a compact machine learning MVP for binary credibility prediction. The project trains and evaluates models that classify text as either:

* `credible`
* `not_credible`

The prepared demo can be used on full-length text articles and returns a credibility prediction for the provided article text. The prediction is not expected to be perfectly accurate, and it should not be treated as a replacement for full evidence-based fact checking. It is a lightweight ML credibility classifier trained under time and budget constraints.

The project uses the LIAR dataset as the main training source. LIAR contains short political and news-related claims labeled by PolitiFact. A larger, more diverse custom dataset of full-length articles from reliable media, blogs, social media, and low-credibility websites would be ideal if additional time, budget, and data collection resources were available.

## Project Scope

The goal of this MVP is to demonstrate an end-to-end approach to text credibility classification:

1. Prepare and clean a labeled text dataset.
2. Convert multi-class credibility labels into a binary target.
3. Train several baseline and semantic ML models.
4. Compare model quality using standard classification metrics.
5. Save model artifacts and reports.
6. Provide a simple CLI demo for predicting credibility of a provided article file.

## Notebooks Showcasing the Results

| Notebook                                         | Description                                                                                                                               | Main outputs                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [data_exploration.ipynb](data_exploration.ipynb) | Downloads LIAR, prepares the binary dataset, performs basic EDA, and saves processed data.                                                | [data/processed/liar_binary.csv](data/processed/liar_binary.csv), [reports/figures/class_distribution.png](reports/figures/class_distribution.png), [reports/figures/split_distribution.png](reports/figures/split_distribution.png), [reports/figures/text_length_by_class.png](reports/figures/text_length_by_class.png), [reports/figures/top_subjects.png](reports/figures/top_subjects.png) |
| [analysis.ipynb](analysis.ipynb)                 | Trains and evaluates supervised credibility models. Saves model artifacts, metrics, confusion matrices, explanations, and error analysis. | [reports/metrics_summary.csv](reports/metrics_summary.csv), [reports/error_analysis.md](reports/error_analysis.md), [reports/tfidf_top_ngrams.csv](reports/tfidf_top_ngrams.csv), [reports/tfidf_example_explanations.csv](reports/tfidf_example_explanations.csv), [models/](models/)                                       |


## Setup

```bash
uv sync
```

Install necessary NLTK resources

```bash
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Run the Project

### Execute the tests:

```bash
uv run pytest tests -v
```

### Perform classification on an example article (that can be changed to your custom text) [example_article.md](example_article.md)

```bash
uv run python src/fact_check.py
```

## Dataset

The project uses the LIAR dataset because it is publicly available, already labeled, and suitable for quickly building a credibility-classification MVP.

LIAR contains short political statements labeled by PolitiFact using six truthfulness classes. For this project, the original labels are converted into a binary credibility task.

A production-ready solution would benefit from a larger dataset containing full-length articles from multiple source types, for example:

* reliable media outlets,
* blogs,
* social media posts,
* partisan websites,
* low-credibility domains,
* manually fact-checked long-form articles.

Such a dataset would improve real-world generalization, especially for full-length article credibility prediction.

## Binary Label Mapping

| Original label | Binary label                 | Numeric label |
| -------------- | ---------------------------- | ------------: |
| `pants-fire`   | `not_credible`               |             0 |
| `false`        | `not_credible`               |             0 |
| `barely-true`  | `not_credible`               |             0 |
| `mostly-true`  | `credible`                   |             1 |
| `true`         | `credible`                   |             1 |
| `half-true`    | dropped from main experiment |           n/a |

The ambiguous `half-true` class is dropped in the main experiment to create a cleaner binary classification problem.

## Prepared Dataset Summary

After processing and binary filtering:

| Stage                              |   Rows |
| ---------------------------------- | -----: |
| Original LIAR dataset              | 12,791 |
| Binary dataset without `half-true` |  9,210 |

Class distribution:

| Class          |  Rows |
| -------------- | ----: |
| `not_credible` | 5,080 |
| `credible`     | 4,130 |

Processed dataset path:

* [data/processed/liar_binary.csv](data/processed/liar_binary.csv)

## Models

The project compares four model variants:

1. Dummy majority-class baseline.
2. TF-IDF + Logistic Regression.
3. Sentence-transformer embeddings + Logistic Regression.
4. Sentence-transformer embeddings + Gradient Boosting Machine.

Model definitions are implemented in:

* [src/models/](src/models/)

Each model class exposes a common API:

* `fit`
* `predict`
* `save`
* `load`

Saved model artifacts:

* [models/dummy_baseline.joblib](models/dummy_baseline.joblib)
* [models/tfidf_logreg.joblib](models/tfidf_logreg.joblib)
* [models/transformer_or_embeddings.joblib](models/transformer_or_embeddings.joblib)
* [models/embedding_gbm.joblib](models/embedding_gbm.joblib)

## Main Results

Metrics are saved in:

* [reports/metrics_summary.csv](reports/metrics_summary.csv)

| Model                                     | Accuracy | Macro F1 | Main value                                                  |
| ----------------------------------------- | -------: | -------: | ----------------------------------------------------------- |
| Sentence embeddings + GBM                 |    0.627 |    0.624 | Best overall result; nonlinear model on semantic embeddings |
| Sentence embeddings + Logistic Regression |    0.617 |    0.616 | Semantic transformer-based baseline without fine-tuning     |
| TF-IDF + Logistic Regression              |    0.611 |    0.610 | Strong explainable supervised baseline                      |
| Dummy baseline                            |    0.551 |    0.355 | Minimum reference baseline                                  |

The best-performing model in this MVP is `Sentence embeddings + GBM`, which reaches approximately:

* accuracy: `0.627`
* macro F1: `0.624`

The improvement over the dummy baseline shows that the models learn useful credibility-related text patterns, although the task remains difficult and the results are not sufficient for high-stakes automatic fact checking.

## Confusion Matrices

Generated confusion matrices are available in:

* [reports/figures/confusion_matrix_dummy.png](reports/figures/confusion_matrix_dummy.png)
* [reports/figures/confusion_matrix_tfidf.png](reports/figures/confusion_matrix_tfidf.png)
* [reports/figures/confusion_matrix_embeddings.png](reports/figures/confusion_matrix_embeddings.png)
* [reports/figures/confusion_matrix_embeddings_gbm.png](reports/figures/confusion_matrix_embeddings_gbm.png)

## Explainability and Error Analysis

The project includes several result files intended to make the model behavior easier to inspect:

* [reports/tfidf_top_ngrams.csv](reports/tfidf_top_ngrams.csv)
* [reports/tfidf_example_explanations.csv](reports/tfidf_example_explanations.csv)
* [reports/bertopic_topic_risk.csv](reports/bertopic_topic_risk.csv)
* [reports/error_analysis.md](reports/error_analysis.md)

### Selected Error Analysis Findings

Manual review of model errors suggests that false positives often happen when not-credible claims contain:

* concrete numeric claims,
* institutional or policy-related wording,
* matter-of-fact phrasing that sounds plausible without external context.

False negatives often happen when credible claims contain:

* adversarial political framing,
* polarizing topics,
* campaign-like rhetoric,
* language patterns commonly associated with misleading claims.

This confirms that text-only credibility classification is limited. A stronger production system should combine text classification with external evidence retrieval, source metadata, publication history, and fact-checking signals.

## CLI Article Demo

A lightweight CLI demo is available in:

* [src/fact_check.py](src/fact_check.py)

It can classify a provided text article file:

```bash
uv run python src/fact_check.py \
  --model embedding_gbm \
  --model-path models/embedding_gbm.joblib \
  --article-path example_article.md
```

Supported model values:

```text
tfidf_logreg
embedding_logreg
embedding_gbm
```

Example output shape:

```json
{
  "model": "embedding_gbm",
  "model_path": "models/embedding_gbm.joblib",
  "article_path": "example_article.md",
  "prediction": 1,
  "prediction_label": "credible",
  "input_chars": 1234,
  "input_words": 210,
  "preprocessed_chars": 890,
  "preprocessed_words": 132,
  "scope_note": "This classifies LIAR-style short-claim credibility patterns; it does not perform evidence retrieval or real-time fact checking."
}
```

The demo accepts full-length text files and produces a credibility prediction. Because the training data is based on short LIAR-style claims, the prediction should be interpreted as an MVP-level signal rather than a fully reliable article-level fact-checking result.

## Tests

The repository includes tests for:

* processed data contract,
* model API consistency,
* model save/load behavior,
* report artifacts,
* CLI behavior.

Run tests with:

```bash
uv run pytest tests -v
```

## Limitations

This MVP is intentionally lightweight and has several important limitations:

1. The training data comes from LIAR, which contains short claims rather than full-length articles.
2. The dataset was selected due to time and budget constraints.
3. A bigger, custom article-level dataset would likely improve real-world performance.
4. Binary mapping loses nuance from the original six-class truthfulness scale.
5. Dropping `half-true` makes the task cleaner but less complete.
6. The model does not retrieve external evidence.
7. The model does not perform real-time fact checking.
8. Political-domain bias is likely because LIAR is focused on political claims.
9. Full-length article predictions are supported by the demo, but accuracy is limited by the training data and MVP scope.

## Next Steps

With additional resources, the project could be improved by:

1. Building a larger custom dataset of full-length articles.
2. Adding source-level metadata such as domain, author, publication date, and historical credibility.
3. Preserving richer article features such as numbers, links, quotes, named entities, and article structure.
4. Adding evidence retrieval from trusted sources.
5. Fine-tuning transformer models directly on the credibility task.
6. Calibrating prediction probabilities.
7. Adding confidence thresholds and an `uncertain` class.
8. Evaluating cross-domain generalization.
9. Deploying the model behind a small API or web demo.
10. Monitoring predictions and collecting feedback for iterative improvement.

## Summary

This repository demonstrates an end-to-end ML approach to credibility prediction under realistic time and budget constraints. It includes data preparation, exploratory analysis, multiple supervised models, evaluation reports, saved artifacts, tests, and a CLI demo capable of classifying full-length text articles.

The current solution should be treated as an MVP and a proof of concept. For production-grade article credibility prediction, the most important next step would be collecting a larger and more diverse full-article dataset.
