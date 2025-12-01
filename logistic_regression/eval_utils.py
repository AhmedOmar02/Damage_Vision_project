"""
eval_utils.py
Simple evaluation helpers.
"""
import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute percentage accuracy (0-100).
    y_pred and y_true should both be shape (1, m)
    """
    if y_true.size == 0:
        return 0.0
    return 100.0 - np.mean(np.abs(y_pred - y_true)) * 100.0


def print_report(result_dict: dict):
    """
    Print a compact report from model() dictionary
    """
    train_acc = result_dict.get("train_accuracy", None)
    test_acc = result_dict.get("test_accuracy", None)
    if train_acc is not None:
        print(f"Train accuracy: {train_acc:.2f}%")
    if test_acc is not None:
        print(f"Test accuracy : {test_acc:.2f}%")
    costs = result_dict.get("costs", None)
    if costs is not None:
        print(f"Costs recorded (every 100 iterations): {len(costs)} values")
