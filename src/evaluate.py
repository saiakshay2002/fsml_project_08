from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import ensure_dir


def evaluate_classifier(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    cm = metrics["confusion_matrix"]
    return (
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1-score: {metrics['f1']:.4f}\n"
        f"ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}\n"
        f"Confusion Matrix: {cm}\n"
        f"Classification Report:\n{metrics['classification_report']}"
    )


def save_evaluation_report(results: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        for model_name, sections in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 80 + "\n")
            for split_name, metrics in sections.items():
                f.write(f"[{split_name.upper()}]\n")
                f.write(format_metrics(metrics))
                f.write("\n\n")
            f.write("=" * 80 + "\n\n")
