"""
ResNet50 model for image classification.
Extracted and adapted from the Earthquake Infrastructure Damage repo.
"""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """
    Build a ResNet50 model with a custom classification head.

    Args:
        num_classes:      Number of output classes.
        pretrained:       If True, loads ImageNet pretrained weights.
        freeze_backbone:  If True, freezes all layers except the final FC layer
                          (feature extraction mode). If False, fine-tunes the
                          entire network.

    Returns:
        model (nn.Module): Ready-to-use ResNet50 model.

    Image requirements:
        - Resize images to 256×256, then center/random-crop to 224×224.
        - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Optionally freeze backbone (keep only classifier trainable)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    num_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Linear(num_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    NUM_CLASSES = 4  # change to your number of classes
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False)
    print(model)

    # Forward pass with a dummy batch
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"\nOutput shape: {out.shape}")  # expected: (2, NUM_CLASSES)
