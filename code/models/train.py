"""Train one scaled logistic regression, log it to MLflow, and package it."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from penguins import FEATURES, ROOT, SEED, SPECIES, TARGET, write_json

LOGGER = logging.getLogger("train")


def train_model(processed_dir: Path, model_dir: Path, tracking_dir: Path) -> dict:
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    if set(train["row_id"]) & set(test["row_id"]):
        raise ValueError("Train and test row IDs overlap.")
    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    tracking_dir = tracking_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (tracking_dir / "mlflow.db").as_posix())
    experiment_name = "penguin-species"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=(tracking_dir / "artifacts").as_uri())
    mlflow.set_experiment(experiment_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="logistic-regression") as run:
        mlflow.log_params({"model": "LogisticRegression", "scaler": "StandardScaler",
                           "max_iter": 1000, "seed": SEED, "test_fraction": 0.2,
                           "features": ",".join(FEATURES), "train_rows": len(train), "test_rows": len(test)})
        estimator.fit(train[FEATURES], train[TARGET])
        predicted = estimator.predict(test[FEATURES])
        metrics = {"test_accuracy": float(accuracy_score(test[TARGET], predicted)),
                   "test_macro_f1": float(f1_score(test[TARGET], predicted, average="macro"))}
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(processed_dir / "cleaning_report.json"), artifact_path="data")
        mlflow.sklearn.log_model(estimator, name="model", input_example=train[FEATURES].head(3))
        joblib.dump(estimator, model_dir / "model.joblib")
        metadata = {"run_id": run.info.run_id, "features": FEATURES, "species": SPECIES,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "sklearn_version": sklearn.__version__,
                    "model_sha256": hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest()}
        write_json(model_dir / "metadata.json", metadata)
        write_json(model_dir / "metrics.json", metrics)
        mlflow.log_artifact(str(model_dir / "metadata.json"))
    # MLflow's database migrations may reconfigure logging; keep this summary visible.
    print(f"Training complete: run_id={metadata['run_id']} metrics={json.dumps(metrics)}", flush=True)
    return {**metadata, "metrics": metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--tracking-dir", type=Path, default=ROOT / ".runtime/mlflow")
    args = parser.parse_args()
    train_model(args.processed_dir, args.model_dir, args.tracking_dir)
