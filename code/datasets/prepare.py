"""Read the committed CSV, clean it, and produce an honest holdout split."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from penguins import FEATURES, ROOT, SEED, SPECIES, TARGET, write_json

LOGGER = logging.getLogger("prepare")


def remove_training_outliers(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit IQR fences on training features only; never use the holdout."""
    q1, q3 = train[FEATURES].quantile(0.25), train[FEATURES].quantile(0.75)
    spread = q3 - q1
    lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    outlier = ((train[FEATURES] < lower) | (train[FEATURES] > upper)).any(axis=1)
    bounds = {name: {"lower": float(lower[name]), "upper": float(upper[name])} for name in FEATURES}
    return train.loc[~outlier].copy(), bounds


def prepare_dataset(raw_path: Path, output_dir: Path) -> dict:
    raw = pd.read_csv(raw_path)
    required = FEATURES + [TARGET]
    absent = sorted(set(required) - set(raw.columns))
    if absent:
        raise ValueError(f"Raw CSV is missing required columns: {absent}")
    data = raw[required].copy()
    data.insert(0, "row_id", raw.index)
    for name in FEATURES:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    missing = data[required].isna().any(axis=1)
    missing_count = int(missing.sum())
    data = data.loc[~missing].copy()
    valid = (np.isfinite(data[FEATURES]) & (data[FEATURES] > 0)).all(axis=1)
    invalid_count = int((~valid).sum())
    data = data.loc[valid].copy()
    unknown = sorted(set(data[TARGET]) - set(SPECIES))
    if unknown:
        raise ValueError(f"Unexpected species labels: {unknown}")
    duplicate = data.duplicated(subset=required)
    duplicate_count = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    if set(data[TARGET]) != set(SPECIES) or data[TARGET].value_counts().min() < 2:
        raise ValueError("Need at least two valid rows for each of the three species.")
    train, test = train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=SEED)
    before_outliers = len(train)
    train, bounds = remove_training_outliers(train)
    if set(train[TARGET]) != set(SPECIES):
        raise ValueError("Outlier cleaning removed a training class; inspect the input data.")
    report = {
        "raw_rows": len(raw), "missing_required_rows_removed": missing_count,
        "invalid_measurement_rows_removed": invalid_count, "duplicate_rows_removed": duplicate_count,
        "training_rows_before_outliers": before_outliers,
        "training_outliers_removed": before_outliers - len(train),
        "train_rows": len(train), "test_rows": len(test), "random_seed": SEED,
        "test_fraction": 0.2, "outlier_method": "1.5 * IQR, fitted on and applied to training only",
        "outlier_bounds": bounds,
        "train_class_counts": train[TARGET].value_counts().sort_index().to_dict(),
        "test_class_counts": test[TARGET].value_counts().sort_index().to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    train.sort_values("row_id").to_csv(output_dir / "train.csv", index=False)
    test.sort_values("row_id").to_csv(output_dir / "test.csv", index=False)
    write_json(output_dir / "cleaning_report.json", report)
    LOGGER.info("Prepared %s training / %s testing rows; removed %s missing and %s training outliers",
                len(train), len(test), missing_count, before_outliers - len(train))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/penguins.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    prepare_dataset(args.raw, args.output_dir)
