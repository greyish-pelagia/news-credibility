from pathlib import Path

import pandas as pd


REQUIRED_ARTIFACTS = [
    Path("data/processed/liar_binary.csv"),
    Path("models/dummy_baseline.joblib"),
    Path("models/tfidf_logreg.joblib"),
    Path("models/transformer_or_embeddings.joblib"),
    Path("models/embedding_gbm.joblib"),
    Path("models/bertopic_model"),
    Path("reports/metrics_summary.csv"),
    Path("reports/tfidf_top_ngrams.csv"),
    Path("reports/tfidf_example_explanations.csv"),
    Path("reports/bertopic_topic_risk.csv"),
    Path("reports/error_analysis.md"),
    Path("reports/figures/class_distribution.png"),
    Path("reports/figures/split_distribution.png"),
    Path("reports/figures/text_length_by_class.png"),
    Path("reports/figures/top_subjects.png"),
    Path("reports/figures/confusion_matrix_dummy.png"),
    Path("reports/figures/confusion_matrix_tfidf.png"),
    Path("reports/figures/confusion_matrix_embeddings.png"),
    Path("reports/figures/confusion_matrix_embeddings_gbm.png"),
]


def test_required_artifacts_exist():
    missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
    assert not missing, f"Missing artifacts: {missing}"


def test_metrics_summary_contains_expected_models_and_columns():
    metrics = pd.read_csv("reports/metrics_summary.csv")
    expected_columns = {
        "model",
        "accuracy",
        "precision_credible",
        "recall_credible",
        "f1_credible",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "main_value",
    }
    expected_models = {
        "Dummy baseline",
        "TF-IDF + Logistic Regression",
        "Sentence embeddings + Logistic Regression",
        "Sentence embeddings + GBM",
    }

    assert expected_columns.issubset(metrics.columns)
    assert expected_models.issubset(set(metrics["model"]))
