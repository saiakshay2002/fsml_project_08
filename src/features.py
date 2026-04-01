from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-6


class FeatureEngineer:
    """Create additional predictive features from the processed sensor data.

    Added transformations:
    1. sensor_11_12_gap:
       Captures divergence between two related sensor measurements.
       Sudden mismatch between correlated subsystems can indicate degradation.

    2. sensor_20_21_ratio:
       Converts two absolute measurements into a relative stress indicator.
       Ratios often capture change more robustly than raw values.

    3. sensor_15_squared:
       Emphasizes extreme values in sensor_15. Non-linear growth can help
       tree-based and linear models capture risk near failure.
    """

    def __init__(self) -> None:
        self.created_features: list[str] = []

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        self.created_features = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if {"sensor_11", "sensor_12"}.issubset(X.columns):
            X["sensor_11_12_gap"] = X["sensor_11"] - X["sensor_12"]
            self._track("sensor_11_12_gap")

        if {"sensor_20", "sensor_21"}.issubset(X.columns):
            X["sensor_20_21_ratio"] = X["sensor_20"] / (np.abs(X["sensor_21"]) + EPSILON)
            self._track("sensor_20_21_ratio")

        if "sensor_15" in X.columns:
            X["sensor_15_squared"] = X["sensor_15"] ** 2
            self._track("sensor_15_squared")

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _track(self, feature_name: str) -> None:
        if feature_name not in self.created_features:
            self.created_features.append(feature_name)
            
class SklearnFeatureEngineer:
    """sklearn-compatible wrapper around FeatureEngineer."""

    def __init__(self) -> None:
        self.engineer = FeatureEngineer()

    def fit(self, X: pd.DataFrame, y=None) -> "SklearnFeatureEngineer":
        self.engineer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.engineer.transform(X)
