import xml.etree.ElementTree as ET
from typing import Dict, List
import numpy as np
    
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








#xml utils
def parse_cvat_xml_all_labels(path: str) -> Dict[str, List[str]]:
    """Parse CVAT XML and return all labels per image."""
    labels = {}
    
    tree = ET.parse(path)
    root = tree.getroot()
    
    for image in root.findall(".//image"):
        filename = image.get("name")
        if not filename:
            continue
            
        # Find ALL polygons in this image
        image_labels = []
        for polygon in image.findall("polygon"):
            label = polygon.get("label")
            if label:
                image_labels.append(label)
        
        labels[filename] = image_labels if image_labels else ["no_label"]
    
    return labels



def label_Y_binary(labels_per_image: dict) -> dict:
    """
    Binary classification: Destroyed vs Not Destroyed
    Destroyed = contains ANY "D_Building" or "Debris"
    """
    DESTROYED = {"D_Building", "Debris"}
    
    result = {}
    for filename, label_list in labels_per_image.items():
        # Check ALL labels, not just first
        is_destroyed = any(label in DESTROYED for label in label_list)
        result[filename] = 1 if is_destroyed else 0
    
    return result


def polygon_area(coords: List[Tuple[float, float]]) -> float:
    """Compute polygon area using the shoelace formula.
    coords: list of (x, y) tuples in vertex order (clockwise or ccw).
    """
    if len(coords) < 3:
        return 0.0
    x = np.array([p[0] for p in coords], dtype=float)
    y = np.array([p[1] for p in coords], dtype=float)
    # use roll(-1) to get x_i * y_{i+1}
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def parse_destroyed_with_size_check(path: str, min_coverage: float = 0.05) -> Dict[str, int]:
    """
    Return {basename(filename): 0/1} where 1 means the image contains at least
    one polygon labeled as destroyed and that polygon covers >= min_coverage of image area.

    - path: xml annotation file path
    - min_coverage: fraction of image area (0..1), e.g. 0.05 -> 5%
    """
    DESTROYED_LABELS = {"D_Building", "Debris"}
    result: Dict[str, int] = {}

    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse XML {path}: {e}")
    except FileNotFoundError:
        raise RuntimeError(f"XML file not found: {path}")

    root = tree.getroot()

    for image in root.findall(".//image"):
        filename = image.get("name")
        if not filename:
            continue
        # normalize to basename so it matches files in your images folder
        filename_key = os.path.basename(filename)

        # get image size (some annotations may store width/height as attributes)
        try:
            width = int(image.get("width", 0))
            height = int(image.get("height", 0))
        except ValueError:
            width = 0
            height = 0
        image_area = float(width * height)

        # if size missing, try to read size from a nested tag or skip
        if image_area == 0:
            # fallback: mark as not destroyed (or optionally skip)
            result[filename_key] = 0
            continue

        is_destroyed = False

        for polygon in image.findall("polygon"):
            label = polygon.get("label")
            points = polygon.get("points")

            if not label or not points:
                continue
            if label not in DESTROYED_LABELS:
                continue

            # Flexible parsing of points:
            # common formats: "x1,y1;x2,y2;..." or "x1,y1 x2,y2 ..." or "x1,y1;x2,y2;"
            pts_str = points.strip()
            if ";" in pts_str:
                raw_pts = pts_str.split(";")
            else:
                raw_pts = pts_str.split()  # split on whitespace

            coords = []
            for p in raw_pts:
                p = p.strip()
                if not p:
                    continue
                # support "x,y" or "x,y," etc.
                if "," not in p:
                    # unexpected format
                    coords = []
                    break
                try:
                    x_str, y_str = p.split(",")[:2]
                    coords.append((float(x_str), float(y_str)))
                except Exception:
                    coords = []
                    break

            if not coords:
                continue

            poly_area = polygon_area(coords)
            coverage = poly_area / image_area

            if coverage >= min_coverage:
                is_destroyed = True
                break

        result[filename_key] = int(is_destroyed)

    return result