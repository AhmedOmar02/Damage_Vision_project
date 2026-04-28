"""
Train / evaluate ResNet50 for image classification.

Example usage:
    python train.py \
        --train  data/labels/train.tsv \
        --val    data/labels/val.tsv \
        --test   data/labels/test.tsv \
        --data-dir data/images/ \
        --num-classes 4 \
        --epochs 50 \
        --lr 1e-5 \
        --batch-size 32
"""

import os
import copy
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm

from model import build_resnet50
from dataset import ImageDataset


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="ResNet50 image classifier")
parser.add_argument("--train",       required=True,  help="Path to train label file")
parser.add_argument("--val",         required=True,  help="Path to val label file")
parser.add_argument("--test",        default=None,   help="Path to test label file (optional)")
parser.add_argument("--data-dir",    required=True,  help="Root directory of images")
parser.add_argument("--sep",         default="\t",   help="CSV/TSV separator (default: tab)")
parser.add_argument("--num-classes", default=4,      type=int)
parser.add_argument("--epochs",      default=50,     type=int)
parser.add_argument("--lr",          default=1e-5,   type=float)
parser.add_argument("--weight-decay",default=0.0,    type=float)
parser.add_argument("--batch-size",  default=32,     type=int)
parser.add_argument("--freeze",      action="store_true",
                    help="Freeze backbone; train only the FC head")
parser.add_argument("--rand-augment",action="store_true",
                    help="Apply RandAugment to training images")
parser.add_argument("--rand-n",      default=2,      type=int)
parser.add_argument("--rand-m",      default=9,      type=int)
parser.add_argument("--out-dir",     default="output",help="Directory for checkpoints & logs")
parser.add_argument("--seed",        default=42,     type=int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_transforms(rand_augment: bool, rand_n: int, rand_m: int):
    """Return train and val transform pipelines for ResNet50 (224×224 input)."""
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_tfs = [
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ]
    if rand_augment:
        train_tfs.insert(0, T.RandAugment(rand_n, rand_m))

    val_tfs = [
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        normalize,
    ]

    return T.Compose(train_tfs), T.Compose(val_tfs)


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  → Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, loaders, criterion, optimizer, scheduler, device,
                num_epochs, out_dir, class_names):
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_path = os.path.join(out_dir, "checkpoints", "best.pth")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}  {'─' * 40}")

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in tqdm(loaders[phase], desc=f"  {phase.upper()}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc  = running_correct / len(loaders[phase].dataset)
            lr_now     = optimizer.param_groups[0]["lr"]

            print(f"  [{phase:5s}]  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  lr={lr_now:.2e}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    save_checkpoint(
                        {"epoch": epoch, "state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "class_names": class_names},
                        best_path,
                    )
                scheduler.step(epoch_loss)

    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, class_names, device, split_name="Test"):
    model.eval()
    preds_, labels_ = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"  Evaluating {split_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_.extend(preds.cpu().numpy())
            labels_.extend(labels.numpy())

    print(f"\n{'─'*50}")
    print(f"{split_name} results:")
    print(f"  Accuracy : {metrics.accuracy_score(labels_, preds_):.4f}")
    print(metrics.classification_report(labels_, preds_, target_names=class_names, digits=3))
    print(metrics.confusion_matrix(labels_, preds_))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Transforms ────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(args.rand_augment, args.rand_n, args.rand_m)

    # ── Datasets & loaders ────────────────────────────────────────────────
    datasets = {
        "train": ImageDataset(args.train, args.data_dir, sep=args.sep, transform=train_tf),
        "val":   ImageDataset(args.val,   args.data_dir, sep=args.sep, transform=val_tf),
    }
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size,
                          shuffle=(split == "train"), num_workers=4)
        for split, ds in datasets.items()
    }
    class_names = datasets["train"].classes
    print(f"Classes: {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_resnet50(
        num_classes=args.num_classes,
        pretrained=True,
        freeze_backbone=args.freeze,
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Optimizer / scheduler / loss ──────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    criterion = nn.CrossEntropyLoss()

    # ── Train ─────────────────────────────────────────────────────────────
    model = train_model(
        model, loaders, criterion, optimizer, scheduler,
        device=device, num_epochs=args.epochs,
        out_dir=args.out_dir, class_names=class_names,
    )

    # ── Evaluate on val ───────────────────────────────────────────────────
    evaluate(model, loaders["val"], class_names, device, split_name="Val")

    # ── Evaluate on test (optional) ───────────────────────────────────────
    if args.test:
        test_ds = ImageDataset(args.test, args.data_dir, sep=args.sep, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        evaluate(model, test_loader, class_names, device, split_name="Test")


if __name__ == "__main__":
    main()
