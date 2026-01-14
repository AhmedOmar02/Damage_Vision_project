"""
visualization.py
General visualization helpers for deep learning models.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_costs(costs, step=100):
    """
    Plot training cost curve.

    Args:
        costs: list of loss values
        step: iterations between recorded costs
    """
    if not costs:
        print("No costs to plot.")
        return

    plt.figure()
    plt.plot(np.arange(len(costs)) * step, costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost / Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()


def show_images(X, titles=None, ncols=5):
    """
    Display a grid of images.
    Supports grayscale and RGB images.

    X shape:
        - (m, h, w) grayscale
        - (m, h, w, 3) RGB
    """
    m = X.shape[0]
    ncols = min(ncols, m)
    nrows = int(np.ceil(m / ncols))

    plt.figure(figsize=(ncols * 2, nrows * 2))

    for i in range(m):
        plt.subplot(nrows, ncols, i + 1)

        if X.ndim == 3:
            plt.imshow(X[i], cmap="gray")
        else:
            plt.imshow(np.clip(X[i], 0, 1))

        if titles is not None:
            plt.title(str(titles[i]))
        plt.axis("off")

    plt.tight_layout()
    plt.show()
