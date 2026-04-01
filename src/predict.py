from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import SklearnFeatureEngineer
from src.utils import MODELS_DIR, load_pickle


BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


class InferencePipeline:
    """Input -> preparation -> trained pipeline -> prediction."""

    def __init__(self, model_path: str | Path = BEST_MODEL_PATH) -> None:
        self.model = load_pickle(model_path)

    def _prepare_input(self, record: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(record, pd.DataFrame):
            X = record.copy()
        else:
            X = pd.DataFrame([record])

        if "label" in X.columns:
            X = X.drop(columns=["label"])

        return X

    def predict(self, record: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        X = self._prepare_input(record)

        pred = int(self.model.predict(X)[0])

        result = {
            "prediction": pred,
            "prediction_label": "near_failure" if pred == 1 else "healthy",
        }

        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0, 1])
            result["failure_probability"] = prob

        return result


if __name__ == "__main__":
    example_input = {
        "sensor_2": 641.82,
        "sensor_3": 1589.7,
        "sensor_4": 1400.6,
        "sensor_7": 554.36,
        "sensor_8": 2388.06,
        "sensor_9": 9046.19,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_17": 391,
        "sensor_20": 39.06,
        "sensor_21": 23.419,
    }

    pipeline = InferencePipeline()
    print(pipeline.predict(example_input))