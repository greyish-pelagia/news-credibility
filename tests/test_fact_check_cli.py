from pathlib import Path

import pytest

from fact_check import (
    load_model,
    predict_article,
    preprocess_text,
    read_text_article,
)


def test_preprocess_text_matches_exploration_cleanup_shape():
    text = "Hello, WORLD!\nThis costs $123 and includes policy details."

    processed = preprocess_text(text)

    assert processed == processed.lower()
    assert "," not in processed
    assert "!" not in processed
    assert "123" not in processed
    assert "\n" not in processed
    assert "  " not in processed


def test_read_text_article_rejects_missing_path(tmp_path):
    with pytest.raises(ValueError, match="Article path does not exist"):
        read_text_article(tmp_path / "missing.md")


def test_read_text_article_rejects_non_text_suffix(tmp_path):
    article_path = tmp_path / "article.pdf"
    article_path.write_text("plain text with the wrong suffix", encoding="utf-8")

    with pytest.raises(ValueError, match="text-like extension"):
        read_text_article(article_path)


def test_read_text_article_rejects_binary_content(tmp_path):
    article_path = tmp_path / "article.md"
    article_path.write_bytes(b"not\x00text")

    with pytest.raises(ValueError, match="binary"):
        read_text_article(article_path)


def test_load_model_rejects_unknown_model_name(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown model"):
        load_model("unknown", model_path)


def test_load_model_rejects_missing_model_path(tmp_path):
    with pytest.raises(ValueError, match="Model path does not exist"):
        load_model("tfidf_logreg", tmp_path / "missing.joblib")


def test_predict_article_uses_preprocessed_text_and_loaded_model(tmp_path, monkeypatch):
    article_path = tmp_path / "article.md"
    article_path.write_text("A verified public claim.", encoding="utf-8")

    class FakeModel:
        def predict(self, rows):
            assert rows == [preprocess_text("A verified public claim.")]
            return [1]

    monkeypatch.setattr("fact_check.load_model", lambda model_name, model_path: FakeModel())

    result = predict_article("embedding_gbm", Path("model.joblib"), article_path)

    assert result["prediction"] == 1
    assert result["prediction_label"] == "credible"
    assert result["article_path"] == str(article_path)
