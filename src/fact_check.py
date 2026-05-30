from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

from models import (
    SentenceEmbeddingGBMModel,
    SentenceEmbeddingLogisticRegressionModel,
    TfidfLogisticRegressionModel,
)


MODEL_LOADERS = {
    "tfidf_logreg": TfidfLogisticRegressionModel.load,
    "embedding_logreg": SentenceEmbeddingLogisticRegressionModel.load,
    "embedding_gbm": SentenceEmbeddingGBMModel.load,
}
LABEL_NAMES = {
    0: "not_credible",
    1: "credible",
}
TEXT_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".text",
    ".rst",
}


def clean_text(text: str) -> str:
    """Mirror the text cleanup used in data_exploration.ipynb."""
    text = text.lower()
    text = re.sub(".*?¿", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("[0-9]+", "", text)
    text = re.sub(r" +", " ", text)
    return text


def remove_stop_words(text: str) -> str:
    tokens = _tokenize(text)
    stop_words = _load_stop_words()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(token for token in tokens)


def remove_spaces(text: str) -> str:
    return re.sub(r" +", " ", text).strip()


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stop_words(text)
    return remove_spaces(text)


def read_text_article(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Article path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Article path is not a file: {path}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(
            f"Article path must use a text-like extension {sorted(TEXT_SUFFIXES)}: {path}"
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read article path: {path}") from exc

    if b"\x00" in raw_bytes:
        raise ValueError(f"Article appears to be binary rather than text: {path}")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Article is not valid UTF-8 text: {path}") from exc

    if not text.strip():
        raise ValueError(f"Article is empty: {path}")
    return text


def load_model(model_name: str, model_path: Path):
    if model_name not in MODEL_LOADERS:
        valid = ", ".join(sorted(MODEL_LOADERS))
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    if not model_path.is_file():
        raise ValueError(f"Model path is not a file: {model_path}")

    loader = MODEL_LOADERS[model_name]
    try:
        return loader(model_path)
    except Exception as exc:
        raise ValueError(
            f"Could not load '{model_name}' model from {model_path}. "
            "Check that --model and --model-path refer to the same model type."
        ) from exc


def predict_article(model_name: str, model_path: Path, article_path: Path) -> dict[str, object]:
    article = read_text_article(article_path)
    processed_article = preprocess_text(article)
    if not processed_article:
        raise ValueError("Article has no usable text after preprocessing.")

    model = load_model(model_name, model_path)
    prediction = int(model.predict([processed_article])[0])

    return {
        "model": model_name,
        "model_path": str(model_path),
        "article_path": str(article_path),
        "prediction": prediction,
        "prediction_label": LABEL_NAMES.get(prediction, "unknown"),
        "input_chars": len(article),
        "input_words": len(article.split()),
        "preprocessed_chars": len(processed_article),
        "preprocessed_words": len(processed_article.split()),
        "scope_note": "This classifies LIAR-style short-claim credibility patterns; it does not perform evidence retrieval or real-time fact checking.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify an externally provided text article with a pre-trained news credibility model.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_LOADERS),
        default="embedding_gbm",
        help="Model family to load.",
    )
    parser.add_argument(
        "--model-path",
        default="models/embedding_gbm.joblib",
        help="Path to the saved model artifact.",
    )
    parser.add_argument(
        "--article-path",
        default="example_article.md",
        help="Path to the text article to classify.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = predict_article(
            model_name=args.model,
            model_path=Path(args.model_path),
            article_path=Path(args.article_path),
        )
    except ValueError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    print(json.dumps(result, indent=2))
    return 0


def _load_stop_words() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS)


def _tokenize(text: str) -> list[str]:
    try:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    except Exception:
        return text.split()


if __name__ == "__main__":
    sys.exit(main())
