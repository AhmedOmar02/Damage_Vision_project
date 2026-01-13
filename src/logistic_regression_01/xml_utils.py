import xml.etree.ElementTree as ET
from typing import Dict, List
import numpy as np
    

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


def polygon_area(coords):
    """Compute polygon area using the Shoelace formula."""
    if len(coords) < 3:
        return 0.0

    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])

    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def parse_destroyed_with_size_check(path: str, min_coverage: float = 0.05) -> Dict[str, int]:
    """
    Binary classification:
    Destroyed (1) if ANY polygon with destroyed label covers > min_coverage of image.
    Otherwise Not Destroyed (0).
    """
    
    DESTROYED_LABELS = {"D_Building", "Debris"}
    result = {}

    tree = ET.parse(path)
    root = tree.getroot()

    for image in root.findall(".//image"):
        filename = image.get("name")
        if not filename:
            continue

        # get image size
        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        image_area = width * height

        if image_area == 0:
            result[filename] = 0
            continue

        is_destroyed = False

        for polygon in image.findall("polygon"):
            label = polygon.get("label")
            points = polygon.get("points")

            if not label or not points:
                continue

            if label not in DESTROYED_LABELS:
                continue

            # parse polygon points
            try:
                coords = [tuple(map(float, p.split(',')))
                          for p in points.split(';') if p]
            except:
                continue

            # compute real polygon area
            poly_area = polygon_area(coords)

            coverage = poly_area / image_area

            if coverage >= min_coverage:
                is_destroyed = True
                break

        result[filename] = int(is_destroyed)

    return result
