"""
eval_utils.py
General evaluation utilities for deep learning models.
"""

import numpy as np
from typing import Dict


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    General accuracy for classification.
    Supports binary and multiclass classification.

    y_pred:
        - Binary: (m,) or (1, m) with {0,1}
        - Multiclass: (m,) class indices
    y_true:
        Same shape as y_pred

    Returns:
        Accuracy in percentage (0–100)
    """
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)

    if y_true.size == 0:
        return 0.0

    return np.mean(y_pred == y_true) * 100.0


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error (for regression models).
    """
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    return np.mean((y_pred - y_true) ** 2)


def evaluate(results: Dict) -> Dict:
    """
    General evaluation dispatcher.
    Works with any model that returns a results dictionary.

    Expected keys in results:
        - y_pred_train, y_true_train (optional)
        - y_pred_test, y_true_test (optional)
        - task: 'binary', 'multiclass', or 'regression'

    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    task = results.get("task", "binary")

    if task in ("binary", "multiclass"):
        if "y_pred_train" in results:
            metrics["train_accuracy"] = accuracy(
                results["y_pred_train"],
                results["y_true_train"],
            )
        if "y_pred_test" in results:
            metrics["test_accuracy"] = accuracy(
                results["y_pred_test"],
                results["y_true_test"],
            )

    elif task == "regression":
        if "y_pred_train" in results:
            metrics["train_mse"] = mse(
                results["y_pred_train"],
                results["y_true_train"],
            )
        if "y_pred_test" in results:
            metrics["test_mse"] = mse(
                results["y_pred_test"],
                results["y_true_test"],
            )

    return metrics


def print_report(results: Dict):
    """
    Print a clean, general report.
    """
    print("=== Model Evaluation Report ===")

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")
