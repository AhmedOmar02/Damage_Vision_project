"""
Dataset class for image classification.
Loads images on-demand from a CSV/TSV file that has columns:
    image_path  |  damage_severity
"""

import os
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile

# Allow loading of truncated images (common with real-world datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(data.Dataset):
    """
    Loads images one-by-one from a label file.

    Label file format (TSV or CSV):
        image_path          damage_severity
        images/img001.jpg   0
        images/img002.jpg   1
        ...

    Args:
        file_path (str):  Path to the label CSV/TSV file.
        root_dir  (str):  Root directory that contains the images.
        sep       (str):  Column separator. Use '\\t' for TSV, ',' for CSV.
        transform:        torchvision transforms to apply to each image.
    """

    def __init__(self, file_path: str, root_dir: str, sep: str = "\t", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(file_path, sep=sep)
        df["damage_severity"] = df["damage_severity"].astype(str)

        self.samples = list(zip(df["image_path"].tolist(), df["damage_severity"].tolist()))

        # Build class list and mapping  {class_name -> int_index}
        self.classes = sorted(set(df["damage_severity"]), key=int)
        self.class_to_idx = {cls: int(cls) for cls in self.classes}

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        with open(os.path.join(self.root_dir, img_path), "rb") as f:
            img = Image.open(f)
            if img.mode != "RGB":
                img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, int(label)
