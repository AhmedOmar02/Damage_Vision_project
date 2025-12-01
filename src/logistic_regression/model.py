"""
model.py
Vectorized logistic regression (binary) training utilities.
"""

import numpy as np
from typing import Tuple, Dict


def initialize_with_zeros(dim: int) -> Tuple[np.ndarray, float]:
    """
    Initialize weights vector w of shape (dim, 1) and bias b=0
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid
    """
    return 1.0 / (1.0 + np.exp(-z))


def propagate(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Forward and backward propagation for logistic regression.

    X: shape (n_x, m)
    Y: shape (1, m)
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # shape (1, m)
    ''' even though Sigmoid never outputs exactly 0 or 1 but keep + 1e-12 just to be sure '''
    cost = - (1.0 / m) * np.sum(Y * np.log(A + 1e-12) + (1 - Y) * np.log(1 - A + 1e-12))

    dw = (1.0 / m) * np.dot(X, (A - Y).T)  # shape (n_x, 1)
    db = (1.0 / m) * np.sum(A - Y)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray, num_iterations: int, learning_rate: float, print_cost: bool = False):
    """
    Optimize parameters by gradient descent
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Iteration {i} - cost: {cost:.6f}")
    params = {"w": w, "b": b}
    grads_last = {"dw": dw, "db": db}
    return params, grads_last, costs


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """
    Predict binary labels for X using learned w and b
    X: shape (n_x, m)
    Returns: Y_prediction (1, m)
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    #astype(int) converts boolean values to integers
    Y_prediction = (A > 0.5).astype(int)
    return Y_prediction


def model(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
          num_iterations: int = 2000, learning_rate: float = 0.005, print_cost: bool = True) -> dict:
    """
    Build and train the logistic regression model.

    X_train: shape (n_x, m_train)
    Y_train: shape (1, m_train)
    X_test: shape (n_x, m_test)
    Y_test: shape (1, m_test)
    """
    n_x = X_train.shape[0]
    w, b = initialize_with_zeros(n_x)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test = predict(w, b, X_test)

    train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_test,
         "Y_prediction_train": Y_pred_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_accuracy": train_acc,
         "test_accuracy": test_acc}
    return d
