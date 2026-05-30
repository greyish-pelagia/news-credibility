from pathlib import Path
from typing import Self

import joblib
from sklearn.dummy import DummyClassifier


class DummyBaselineModel:
    def __init__(self, strategy: str = "most_frequent") -> None:
        self.strategy = strategy
        self.model = DummyClassifier(strategy=strategy)

    def fit(self, X, y) -> Self:
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        joblib.dump({"strategy": self.strategy, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        state = joblib.load(path)
        instance = cls(strategy=state["strategy"])
        instance.model = state["model"]
        return instance
