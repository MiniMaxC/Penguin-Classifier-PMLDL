import json

from fastapi.testclient import TestClient
import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from conftest import load_module
from penguins import EXAMPLE, FEATURES


def test_saved_model_metrics_scaling_and_mlflow(trained_model):
    directory, metadata = trained_model
    model = joblib.load(directory / "models/model.joblib")
    train = pd.read_csv(directory / "processed/train.csv")
    test = pd.read_csv(directory / "processed/test.csv")
    np.testing.assert_allclose(model.named_steps["scale"].mean_, train[FEATURES].mean())
    prediction = model.predict(test[FEATURES])
    metrics = json.loads((directory / "models/metrics.json").read_text())
    assert metrics["test_accuracy"] == accuracy_score(test.species, prediction)
    assert metrics["test_macro_f1"] == f1_score(test.species, prediction, average="macro")
    assert metrics["test_accuracy"] > 0.8
    run = mlflow.get_run(metadata["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics == metrics
    restored = mlflow.sklearn.load_model(f"runs:/{metadata['run_id']}/model")
    np.testing.assert_array_equal(restored.predict(test[FEATURES]), prediction)


def test_api_uses_saved_model_and_reports_run_id(trained_model):
    directory, metadata = trained_model
    api = load_module("penguin_api", "code/deployment/api/main.py")
    expected = joblib.load(directory / "models/model.joblib").predict(pd.DataFrame([EXAMPLE]))[0]
    with TestClient(api.create_app(directory / "models")) as client:
        assert client.get("/health").json() == {"status": "ok", "run_id": metadata["run_id"]}
        response = client.post("/predict", json=EXAMPLE)
        assert response.status_code == 200
        assert response.json() == {"species": expected, "run_id": metadata["run_id"]}


@pytest.mark.parametrize("value", [-1, 0, None, "not a number", "NaN", "Infinity"])
def test_api_rejects_invalid_measurements(value, trained_model):
    directory, _ = trained_model
    api = load_module("penguin_api", "code/deployment/api/main.py")
    with TestClient(api.create_app(directory / "models")) as client:
        assert client.post("/predict", json={**EXAMPLE, "body_mass_g": value}).status_code == 422


def test_api_rejects_missing_field(trained_model):
    directory, _ = trained_model
    api = load_module("penguin_api", "code/deployment/api/main.py")
    with TestClient(api.create_app(directory / "models")) as client:
        assert client.post("/predict", json={"body_mass_g": 3750}).status_code == 422


def test_api_does_not_start_without_model(tmp_path):
    api = load_module("penguin_api", "code/deployment/api/main.py")
    with pytest.raises(FileNotFoundError):
        with TestClient(api.create_app(tmp_path)):
            pass
