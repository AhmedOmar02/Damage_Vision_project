#!/usr/bin/env python3
"""
train_1hidden_pytorch.py

One-hidden-layer image classifier in PyTorch (works on GPU if available).

Usage examples:
# Using default options (assumes ./data/train and ./data/val folders with class subfolders)
python train_1hidden_pytorch.py --data_dir ./data --epochs 30 --batch_size 64

# If you only have a train folder and want to split automatically:
python train_1hidden_pytorch.py --data_dir ./data --auto_split --split_ratio 0.2

Author: ChatGPT (GPT-5 Thinking mini)
"""

import os
import argparse
from pathlib import Path
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------
def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_from_logits(logits, targets):
    _, preds = torch.max(logits, 1)
    correct = (preds == targets).sum().item()
    return correct, preds

def confusion_matrix_from_preds(true_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm

# ---------------------------
# Model: simple MLP (1 hidden layer)
# ---------------------------
class OneHiddenNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # initialization (He for relu)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x: (N, C, H, W) -> flatten
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

# ---------------------------
# Training and evaluation
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct, _ = accuracy_from_logits(logits, targets)
        running_correct += correct
        running_total += images.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            correct, preds = accuracy_from_logits(logits, targets)
            running_correct += correct
            running_total += images.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    cm = confusion_matrix_from_preds(all_targets, all_preds, num_classes)
    return epoch_loss, epoch_acc, cm

# ---------------------------
# Main
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train 1-hidden-layer NN with PyTorch (GPU-ready).")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path with 'train' and 'val' subfolders or a single folder with classes.")
    parser.add_argument("--train_dir", type=str, default=None, help="Optional: explicit train folder (overrides data_dir).")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional: explicit val folder.")
    parser.add_argument("--image_size", type=int, nargs=2, default=(64,64), help="Resize to this H W. Example: --image_size 64 64")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--auto_split", action="store_true", help="If only a single dataset folder exists, automatically split into train/val.")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Validation ratio when using --auto_split.")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU even if CUDA is available.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.force_cpu)
    print("Device:", device)

    # Prepare transforms
    H, W = args.image_size
    train_transforms = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load dataset: prefer train/val explicit folders; else data_dir with class subfolders
    if args.train_dir and args.val_dir:
        train_root = args.train_dir
        val_root = args.val_dir
        train_dataset = datasets.ImageFolder(train_root, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transforms)
    else:
        # use data_dir. Expect either data_dir/train & data_dir/val OR data_dir with class subfolders
        data_dir = Path(args.data_dir)
        train_folder = data_dir / "train"
        val_folder = data_dir / "val"
        if train_folder.exists() and val_folder.exists():
            print("Loading from data_dir/train and data_dir/val")
            train_dataset = datasets.ImageFolder(str(train_folder), transform=train_transforms)
            val_dataset = datasets.ImageFolder(str(val_folder), transform=val_transforms)
        else:
            # Use the single data_dir as ImageFolder and optionally split
            print("Loading from single folder and splitting (if requested)...")
            full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transforms)
            num_total = len(full_dataset)
            if args.auto_split:
                val_count = int(num_total * args.split_ratio)
                train_count = num_total - val_count
                train_dataset, val_dataset = random_split(full_dataset, [train_count, val_count],
                                                          generator=torch.Generator().manual_seed(args.seed))
                # random_split returns Subset and uses the transform from the original dataset (train_transforms).
                # Replace val transform to val_transforms:
                # Build a new dataset object for val with same samples but val transforms:
                # quick workaround: convert val_dataset to a dataset wrapper that applies val_transforms
                # We'll build a simple SubsetWithTransform wrapper:
                class SubsetWithTransform(torch.utils.data.Dataset):
                    def __init__(self, subset, transform):
                        self.subset = subset
                        self.transform = transform
                    def __len__(self):
                        return len(self.subset)
                    def __getitem__(self, idx):
                        img, label = self.subset[idx]
                        # subset already returned a transformed image; to correct this we'd need raw PIL.
                        # Instead, rebuild from original samples indexes:
                        real_idx = self.subset.indices[idx]
                        path, label = full_dataset.samples[real_idx]
                        from PIL import Image
                        img = Image.open(path).convert("RGB")
                        img = self.transform(img)
                        return img, label
                val_dataset = SubsetWithTransform(val_dataset, val_transforms)
                # For train_dataset keep original (with train_transforms)
            else:
                # No split requested; treat full_dataset as train, and require a val folder
                train_dataset = full_dataset
                val_dataset = None
                print("No validation dataset provided. Use --auto_split or provide a separate val folder.")
    # If val_dataset is None here, we'll skip validation during training.
    if 'train_dataset' not in locals():
        # If above created train_dataset earlier via splitting it's present; safeguard
        train_dataset = train_dataset

    # Build loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if 'val_dataset' in locals() and val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    # Classes
    if isinstance(train_dataset, datasets.ImageFolder):
        class_names = train_dataset.classes
    else:
        # If it's a Subset or similar, try to use the underlying samples
        try:
            class_names = train_dataset.dataset.classes
        except:
            # fallback
            class_names = [str(i) for i in range(2)]
    num_classes = len(class_names)
    print("Number of classes:", num_classes, "Labels:", class_names)

    # Model config
    channels = 3
    input_dim = channels * H * W
    model = OneHiddenNet(input_dim=input_dim, hidden_dim=args.hidden_dim,
                         num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)
    print("Model parameter count:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    start_epoch = 1

    # Optionally resume
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
        else:
            print("Resume path not found:", args.resume)

    # Training loop
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc, cm = (None, None, None)
        if val_loader is not None:
            val_loss, val_acc, cm = validate(model, val_loader, criterion, device, num_classes)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(val_loss)

        epoch_time = time.time() - t0
        if val_loader is not None:
            print(f"Epoch {epoch}/{args.epochs}  time:{epoch_time:.1f}s  train_loss:{train_loss:.4f} train_acc:{train_acc:.4f}  val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")
        else:
            print(f"Epoch {epoch}/{args.epochs}  time:{epoch_time:.1f}s  train_loss:{train_loss:.4f} train_acc:{train_acc:.4f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(ckpt, os.path.join(args.save_dir, "checkpoint.pth"))

        # Save best model
        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

    # Final evaluation & confusion matrix
    if val_loader is not None:
        _, final_val_acc, cm = validate(model, val_loader, criterion, device, num_classes)
        print("Final validation accuracy:", final_val_acc)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
    else:
        print("No validation performed (no val_loader).")

    # Save training history plot
    try:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.plot(history['train_loss'], label='train_loss')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
        plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(history['train_acc'], label='train_acc')
        if history['val_acc']:
            plt.plot(history['val_acc'], label='val_acc')
        plt.legend()
        plt.title('Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "training_history.png"))
        print("Saved training_history.png")
    except Exception as e:
        print("Could not save plot:", e)

if __name__ == "__main__":
    main()
