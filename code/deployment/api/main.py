"""HTTP boundary for the packaged model; no training occurs in the API."""

from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
import sklearn

from penguins import FEATURES, ROOT, SPECIES

Measurement = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bill_length_mm: Measurement
    bill_depth_mm: Measurement
    flipper_length_mm: Measurement
    body_mass_g: Measurement


class PredictionOutput(BaseModel):
    species: str
    run_id: str


def create_app(model_dir: Path | None = None) -> FastAPI:
    directory = model_dir or Path(os.environ.get("MODEL_DIR", ROOT / "models"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["features"] != FEATURES or metadata["sklearn_version"] != sklearn.__version__:
            raise RuntimeError("The model feature contract or scikit-learn version does not match the API.")
        if hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise RuntimeError("Model checksum does not match its metadata.")
        estimator = joblib.load(directory / "model.joblib")
        if set(estimator.classes_) != set(SPECIES):
            raise RuntimeError("Model class labels do not match the API.")
        app.state.estimator, app.state.run_id = estimator, metadata["run_id"]
        yield

    app = FastAPI(title="Penguin Species API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "run_id": app.state.run_id}

    @app.post("/predict", response_model=PredictionOutput)
    def predict(measurements: PredictionInput) -> PredictionOutput:
        frame = pd.DataFrame([measurements.model_dump()], columns=FEATURES)
        prediction = app.state.estimator.predict(frame)[0]
        return PredictionOutput(species=str(prediction), run_id=app.state.run_id)

    return app


app = create_app()
