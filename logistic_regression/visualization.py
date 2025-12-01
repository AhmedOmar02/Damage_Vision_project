"""
visualization.py
Small plotting helpers using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_costs(costs, learning_rate):
    """
    Plot costs list.
    """
    if not costs:
        print("No costs to plot.")
        return
    plt.figure()
    plt.plot(np.arange(len(costs)) * 100, costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Learning rate = {learning_rate}")
    plt.grid(True)
    plt.show()


def show_image_grid(X, titles=None, ncols=5):
    """
    Show a grid of RGB images. X is shape (m, h, w, 3) in [0,1] float range.
    """
    m = X.shape[0]
    ncols = min(ncols, m)
    nrows = int(np.ceil(m / ncols))
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for i in range(m):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(np.clip(X[i], 0, 1))
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
