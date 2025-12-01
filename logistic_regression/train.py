"""
train.py
Main script that ties everything together.

Usage:
    python -m logistic_regression.train
or
    python logistic_regression/train.py
"""
import os
import time
import pickle

import numpy as np
from xml_utils import parse_cvat_xml_all_labels, label_Y_binary
from data_loader import load_and_resize_images, build_label_array
from model import model
from eval_utils import print_report
from visualization import plot_costs

# --- Config ---
TRAIN_XML = "../EIDSeg_Dataset/data/train/train.xml"
TEST_XML = "../EIDSeg_Dataset/data/test/test.xml"
TRAIN_IMAGES = "../EIDSeg_Dataset/data/train/images/default"
TEST_IMAGES = "../EIDSeg_Dataset/data/test/images/default"
SAVED_MODEL = "saved_model.pkl"

IMAGE_SIZE = (64, 64)  # width, height
NUM_ITER = 2000
LR = 0.001

def main():
    start_time = time.time()
    # Parse XMLs
    labels_train_raw = parse_cvat_xml_all_labels(TRAIN_XML)
    labels_test_raw = parse_cvat_xml_all_labels(TEST_XML)

    # Convert textual labels to integer labels if needed
    Y_train_map = label_Y_binary(labels_train_raw)
    Y_test_map = label_Y_binary(labels_test_raw)

    # Load images
    X_train_org, ordered_filenames_train = load_and_resize_images(TRAIN_IMAGES, size=IMAGE_SIZE)
    X_test_org, ordered_filenames_test = load_and_resize_images(TEST_IMAGES, size=IMAGE_SIZE)

    # Build Y arrays aligned with filenames
    Y_train_org = build_label_array(ordered_filenames_train, Y_train_map)
    Y_test_org = build_label_array(ordered_filenames_test, Y_test_map)

    # Flatten and transpose to shape (n_x, m)
    train_set_x_flatten = X_train_org.reshape(X_train_org.shape[0], -1).T  # (n_x, m_train)
    test_set_x_flatten = X_test_org.reshape(X_test_org.shape[0], -1).T  # (n_x, m_test)

    print("train_set_x shape:", train_set_x_flatten.shape)
    print("train_set_y shape:", Y_train_org.shape)
    print("test_set_x shape:", test_set_x_flatten.shape)
    print("test_set_y shape:", Y_test_org.shape)

    # Train model
    results = model(train_set_x_flatten, Y_train_org, test_set_x_flatten, Y_test_org,
                    num_iterations=NUM_ITER, learning_rate=LR, print_cost=True)

    # Report
    print_report(results)
    plot_costs(results.get("costs", []), LR)

    # Save model (weights + metadata)
    to_save = {
        "w": results["w"],
        "b": results["b"],
        "learning_rate": results["learning_rate"],
        "num_iterations": results["num_iterations"],
        "train_accuracy": results["train_accuracy"],
        "test_accuracy": results["test_accuracy"],
    }
    with open(SAVED_MODEL, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Saved model to {SAVED_MODEL}")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed - minutes * 60
    print(f"Time taken: {minutes} minutes and {seconds:.2f} seconds")


if __name__ == "__main__":
    main()
