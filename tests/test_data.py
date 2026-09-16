import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from penguins import FEATURES


def test_preparation_is_reproducible_and_holdout_disjoint(tmp_path, prepare_module):
    a, b = tmp_path / "a", tmp_path / "b"
    report = prepare_module.prepare_dataset(ROOT / "data/raw/penguins.csv", a)
    prepare_module.prepare_dataset(ROOT / "data/raw/penguins.csv", b)
    for filename in ("train.csv", "test.csv", "cleaning_report.json"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()
    train, test = pd.read_csv(a / "train.csv"), pd.read_csv(a / "test.csv")
    assert not set(train.row_id) & set(test.row_id)
    assert set(train.species) == set(test.species) == {"Adelie", "Chinstrap", "Gentoo"}
    assert np.isfinite(train[FEATURES]).all().all()
    assert (train[FEATURES] > 0).all().all()
    assert report["missing_required_rows_removed"] == 2
    assert sum(report[key] for key in ["missing_required_rows_removed", "invalid_measurement_rows_removed",
                                      "duplicate_rows_removed", "training_outliers_removed", "train_rows", "test_rows"]) == 344


def test_missing_and_invalid_values_are_removed(tmp_path, prepare_module):
    data = pd.read_csv(ROOT / "data/raw/penguins.csv")
    additions = data.iloc[:4].copy()
    additions.loc[additions.index[0], FEATURES[0]] = -1
    additions.loc[additions.index[1], FEATURES[0]] = np.inf
    additions.loc[additions.index[2], FEATURES[0]] = np.nan
    path = tmp_path / "dirty.csv"
    pd.concat([data, additions], ignore_index=True).to_csv(path, index=False)
    report = prepare_module.prepare_dataset(path, tmp_path / "processed")
    assert report["invalid_measurement_rows_removed"] == 2
    assert report["missing_required_rows_removed"] == 4


def test_iqr_removes_extreme_training_outlier(prepare_module):
    frame = pd.DataFrame({feature: list(range(10, 30)) + [100000] for feature in FEATURES})
    clean, bounds = prepare_module.remove_training_outliers(frame)
    assert len(clean) == 20
    assert 20 not in clean.index
    assert bounds[FEATURES[0]]["upper"] < 100000


def test_missing_column_fails_clearly(tmp_path, prepare_module):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"species": ["Adelie"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_module.prepare_dataset(path, tmp_path / "processed")
