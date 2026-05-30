import numpy as np

from models import (
    DummyBaselineModel,
    SentenceEmbeddingGBMModel,
    SentenceEmbeddingLogisticRegressionModel,
    TfidfLogisticRegressionModel,
)


def test_all_model_classes_expose_common_api():
    for model_class in [
        DummyBaselineModel,
        TfidfLogisticRegressionModel,
        SentenceEmbeddingLogisticRegressionModel,
        SentenceEmbeddingGBMModel,
    ]:
        model = model_class()
        for method_name in ["fit", "predict", "save", "load"]:
            assert hasattr(model, method_name)


def test_dummy_baseline_model_fit_predict_save_load(tmp_path):
    model = DummyBaselineModel()
    model.fit(["claim one", "claim two", "claim three"], [0, 0, 1])

    predictions = model.predict(["new claim"])
    assert predictions.tolist() == [0]

    path = tmp_path / "dummy.joblib"
    model.save(path)

    loaded = DummyBaselineModel.load(path)
    assert loaded.predict(["new claim"]).tolist() == [0]


def test_tfidf_logreg_model_fit_predict_save_load(tmp_path):
    X = [
        "tax cut creates jobs",
        "budget numbers are accurate",
        "fake rumor about fraud",
        "false attack ad claim",
        "verified public spending data",
        "made up conspiracy claim",
    ]
    y = np.array([1, 1, 0, 0, 1, 0])
    model = TfidfLogisticRegressionModel(min_df=1, max_features=100)
    model.fit(X, y)

    predictions = model.predict(["verified budget data", "fake fraud rumor"])
    assert set(predictions).issubset({0, 1})

    path = tmp_path / "tfidf.joblib"
    model.save(path)

    loaded = TfidfLogisticRegressionModel.load(path)
    loaded_predictions = loaded.predict(["verified budget data", "fake fraud rumor"])
    assert loaded_predictions.tolist() == predictions.tolist()

def test_embedding_logreg_model_fit_predict_save_load(tmp_path):
    X = [
        "tax cut creates jobs",
        "budget numbers are accurate",
        "fake rumor about fraud",
        "false attack ad claim",
        "verified public spending data",
        "made up conspiracy claim",
    ]
    y = np.array([1, 1, 0, 0, 1, 0])
    model = SentenceEmbeddingLogisticRegressionModel()
    model.fit(X, y)

    predictions = model.predict(["verified budget data", "fake fraud rumor"])
    assert set(predictions).issubset({0, 1})

    path = tmp_path / "embedding_logreg.joblib"
    model.save(path)

    loaded = SentenceEmbeddingLogisticRegressionModel.load(path)
    loaded_predictions = loaded.predict(["verified budget data", "fake fraud rumor"])
    assert loaded_predictions.tolist() == predictions.tolist()

def test_embedding_gbm_model_fit_predict_save_load(tmp_path):
    X = [
        "tax cut creates jobs",
        "budget numbers are accurate",
        "fake rumor about fraud",
        "false attack ad claim",
        "verified public spending data",
        "made up conspiracy claim",
    ]
    y = np.array([1, 1, 0, 0, 1, 0])
    model = SentenceEmbeddingGBMModel()
    model.fit(X, y)

    predictions = model.predict(["verified budget data", "fake fraud rumor"])
    assert set(predictions).issubset({0, 1})

    path = tmp_path / "embedding_gbm.joblib"
    model.save(path)

    loaded = SentenceEmbeddingGBMModel.load(path)
    loaded_predictions = loaded.predict(["verified budget data", "fake fraud rumor"])
    assert loaded_predictions.tolist() == predictions.tolist()