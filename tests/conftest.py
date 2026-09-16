import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def prepare_module():
    return load_module("prepare", "code/datasets/prepare.py")


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, prepare_module):
    directory = tmp_path_factory.mktemp("trained")
    prepare_module.prepare_dataset(ROOT / "data/raw/penguins.csv", directory / "processed")
    trainer = load_module("train", "code/models/train.py")
    metadata = trainer.train_model(directory / "processed", directory / "models", directory / "tracking")
    return directory, metadata
