from pathlib import Path
from typing import Self

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogisticRegressionModel:
    def __init__(
        self,
        max_features: int = 30_000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple[int, int] = (1, 2),
        class_weight: str | dict | None = "balanced",
        c: float = 1.0,
        max_iter: int = 1000,
        pipeline: Pipeline | None = None,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.class_weight = class_weight
        self.c = c
        self.max_iter = max_iter
        self.pipeline = pipeline or Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=max_features,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight=class_weight,
                        C=c,
                    ),
                ),
            ]
        )

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    def fit(self, X, y) -> Self:
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "max_features": self.max_features,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "ngram_range": self.ngram_range,
                "class_weight": self.class_weight,
                "c": self.c,
                "max_iter": self.max_iter,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        state = joblib.load(path)
        return cls(
            max_features=state["max_features"],
            min_df=state["min_df"],
            max_df=state["max_df"],
            ngram_range=state["ngram_range"],
            class_weight=state["class_weight"],
            c=state["c"],
            max_iter=state["max_iter"],
            pipeline=state["pipeline"],
        )
