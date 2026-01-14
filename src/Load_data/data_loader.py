import os
from typing import Tuple, List
import cv2
import numpy as np


def load_and_resize_images(folder: str, size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, List[str]]:
    """
    Load all images from folder, resize them to `size` and return an array
    with shape (m, height, width, channels) and an ordered list of filenames.

    Args:
        folder: directory containing image files.
        size: target (width, height) to resize images.

    Returns:
        X: numpy array shaped (m, h, w, c)
        ordered_filenames: list of filenames in the loaded order
    """
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    
    directory_contents = os.listdir(folder)
    
    image_files = []
    for filename in directory_contents:
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_files.append(filename)
    
    # Sort alphabetically for consistent ordering
    file_list = sorted(image_files)
    images = []
    ordered = []
    for fname in file_list:
        full = os.path.join(folder, fname)
        img = cv2.imread(full, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {full}, skipping")
            continue
        # OpenCV uses (width, height) in resize param
        img = cv2.resize(img, size)
        # Convert BGR -> RGB for consistency with matplotlib if visualizing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)  # normalize to [0,1]
        ordered.append(fname)
    if not images:
        X = np.zeros((0, size[1], size[0], 3), dtype=np.float32)
    else:
        X = np.stack(images, axis=0)
    print("Final X shape:", X.shape)
    return X, ordered


def build_label_array(ordered_filenames: List[str], labels_dict: dict, default_value: int = 0):
    """
    Build a label array aligned with ordered_filenames.

    Returns:
        Y: numpy array shaped (1, m)
    """
    Y = []
    for fname in ordered_filenames:
        # match exact or without extension if labels dict used base names
        if fname in labels_dict:
            Y.append(labels_dict[fname])
        else:
            # try without extension
            base = os.path.splitext(fname)[0]
            if base in labels_dict:
                Y.append(labels_dict[base])
            else:
                print(f"Warning: No label found for {fname}, assigning {default_value}")
                Y.append(default_value)
    return np.array(Y).reshape(1, -1)





#for 
