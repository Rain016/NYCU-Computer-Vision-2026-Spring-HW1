"""
NYCU Visual Recognition HW1 - Improved Training Script v3
ResNet101 + Two-stage training + Image size 320
AdamW + Label Smoothing + Mixup/CutMix
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt  # [NEW] for plotting
from sklearn.metrics import confusion_matrix  # [NEW] for confusion matrix
import seaborn as sns  # [NEW] for confusion matrix heatmap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────

DATA_DIR = "/mnt/wdc4tb/nycu-2/DL/hw1/data"
SAVE_DIR = "/mnt/wdc4tb/nycu-2/DL/hw1/checkpoints"
NUM_CLASSES = 100
BATCH_SIZE = 32        # smaller batch for larger image size
NUM_EPOCHS = 40
WARMUP_EPOCHS = 0      # freeze backbone for first N epochs
LR_FC = 1e-3           # lr for fc layer
LR_BACKBONE = 1e-4     # lr for backbone during fine-tune
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 0.8
CUTMIX_PROB = 0.5
IMAGE_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ─────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """Dataset for test images (no labels)."""

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(test_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(
            os.path.join(self.test_dir, fname)
        ).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


# ── Transforms ──────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Mixup / CutMix ──────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """Apply Mixup augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    _, _, h, w = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Model ───────────────────────────────────────────────────────────────

def build_model(num_classes):
    """Build ResNet101 with pretrained weights."""
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model):
    """Freeze all layers except fc."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


# ── Train / Eval ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        if np.random.rand() < CUTMIX_PROB:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)
        else:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss, acc, preds, labels."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return (
        total_loss / total,
        total_correct / total,
        all_preds,
        all_labels,
    )


# ── Plot Training Curves ───────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses,
                         train_accs, val_accs, save_dir):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


# ── Plot Confusion Matrix ──────────────────────────────────────────────

def plot_confusion_matrix(all_preds, all_labels, num_classes, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(
        all_labels, all_preds, labels=list(
            range(num_classes)))
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, ax=ax,
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
        linewidths=0.3,
        cbar=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Validation Set)", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


# ── Inference ───────────────────────────────────────────────────────────

def inference(model, test_dir, device, class_to_idx):
    """Run inference on test set and save prediction.csv."""
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}

    test_dataset = TestDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    model.eval()
    results = []

    with torch.no_grad():
        for imgs, fnames in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().tolist()
            for fname, pred in zip(fnames, preds):
                results.append(
                    (fname.replace(".jpg", ""), idx_to_class[pred])
                )

    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(results)

    print(f"Saved prediction.csv to {csv_path}")


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HW1 Improved v3")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--infer_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Data ──
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ──
    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best_ckpt = os.path.join(SAVE_DIR, "best_model.pth")

    if args.infer_only:
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        inference(model, os.path.join(DATA_DIR, "test"),
                  DEVICE, train_dataset.class_to_idx)
        return

    best_val_acc = 0.0

    # lists to record history for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_preds, best_labels = [], []  # for confusion matrix

    for epoch in range(1, args.epochs + 1):

        # ── Stage 1: freeze backbone, only train fc ──
        if epoch == 1:
            print("Stage 1: Freezing backbone, training fc only...")
            freeze_backbone(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FC, weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=WARMUP_EPOCHS
            )

        # ── Stage 2: unfreeze all, fine-tune with layerwise LR ──
        if epoch == WARMUP_EPOCHS + 1:
            print("Stage 2: Unfreezing all layers, fine-tuning...")
            unfreeze_all(model)
            optimizer = optim.AdamW([
                {"params": model.fc.parameters(), "lr": LR_FC},
                {"params": [p for n, p in model.named_parameters()
                            if "fc" not in n], "lr": LR_BACKBONE},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step()

        # record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), best_ckpt)
            print(f"  >> Saved best model (val acc: {best_val_acc:.4f})")

    print(f"\nTraining done. Best Val Acc: {best_val_acc:.4f}")

    # plot training curves and confusion matrix
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, SAVE_DIR
    )
    plot_confusion_matrix(best_preds, best_labels, NUM_CLASSES, SAVE_DIR)

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    inference(model, os.path.join(DATA_DIR, "test"),
              DEVICE, train_dataset.class_to_idx)


if __name__ == "__main__":
    main()
