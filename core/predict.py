from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class ModelPaths:
    freight_model: Path = MODELS_DIR / "predict_freight_model.pkl"
    flag_model: Path = MODELS_DIR / "predict_flag_invoice.pkl"
    flag_scaler: Path = MODELS_DIR / "scaler.pkl"


FLAG_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "freight_invoiced",
    "total_item_quantity",
    "total_item_dollars",
]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path.as_posix()}")


def load_joblib(path: Path):
    _require_file(path)
    return joblib.load(path)


def predict_freight(dollars: float, paths: ModelPaths) -> float:
    model = load_joblib(paths.freight_model)
    df = pd.DataFrame({"Dollars": [dollars]})
    y = model.predict(df)
    return float(y[0])


def predict_flag(features: dict, paths: ModelPaths) -> int:
    model = load_joblib(paths.flag_model)
    scaler = load_joblib(paths.flag_scaler)
    df = pd.DataFrame([features], columns=FLAG_FEATURES)
    x_scaled = scaler.transform(df)
    pred = model.predict(x_scaled)
    return int(pred[0])

