from pathlib import Path
from typing import Self

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SentenceEmbeddingLogisticRegressionModel:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        classifier: Pipeline | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.classifier = classifier or Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        C=1.0,
                    ),
                ),
            ]
        )
        self._embedding_model: SentenceTransformer | None = None

    def fit(self, X, y) -> Self:
        X_emb = self.encode(X)
        self.classifier.fit(X_emb, y)
        return self

    def predict(self, X):
        X_emb = self.encode(X)
        return self.classifier.predict(X_emb)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "embedding_model_name": self.embedding_model_name,
                "batch_size": self.batch_size,
                "classifier": self.classifier,
                "notes": "Reload SentenceTransformer(embedding_model_name), encode statements, then call classifier.predict(...).",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        state = joblib.load(path)
        return cls(
            embedding_model_name=state["embedding_model_name"],
            batch_size=state["batch_size"],
            classifier=state["classifier"],
        )

    def encode(self, X):
        return self._encoder().encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _encoder(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
