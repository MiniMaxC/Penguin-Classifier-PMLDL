"""Small shared contract for preparation, training, and serving."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "species"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
SEED = 42
EXAMPLE = dict(zip(FEATURES, [39.1, 18.7, 181.0, 3750.0]))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
