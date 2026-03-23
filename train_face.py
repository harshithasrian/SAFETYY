# ============================================================
# SafeGuard AI — Facial Expression Model Training
# Dataset: FER2013 (train/ + test/ folders)
# Model:   MobileNetV2 pretrained → fine-tuned
# Output:  face_emotion.pth in project root
# ============================================================

import os
import sys
import time
import numpy as np

"""
Train facial emotion model weights for SafeGuardAI.

Expected dataset structure (FER-style):
  <dataset_root>/
    train/<class_name>/*.jpg
    test/<class_name>/*.jpg

Run:
  python train_face.py --dataset \"D:\\path\\to\\fer_dataset\" --epochs 20

Output:
  face_emotion.pth (saved into SafeGuardAI/ project root by default)
"""

import argparse

# ── Defaults ─────────────────────────────────────────────────
DEFAULT_EPOCHS     = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 1e-3
DEFAULT_IMG_SIZE   = 48

# FER2013 → SafeGuard label mapping
LABEL_MAP = {
    "angry":    "anger",
    "disgust":  "anger",      # merge disgust into anger
    "fear":     "fear",
    "happy":    "happiness",
    "neutral":  "neutral",
    "sad":      "sadness",
    "surprise": "excitement",
}

SAFEGUARD_LABELS = ["anger", "fear", "happiness", "excitement", "sadness", "neutral"]

# ── Imports ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    print(f"PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install torch torchvision pillow")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []

        for fer_class in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, fer_class)
            if not os.path.isdir(class_dir):
                continue
            sg_label = LABEL_MAP.get(fer_class.lower())
            if sg_label is None:
                continue
            label_idx = SAFEGUARD_LABELS.index(sg_label)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        print(f"  Loaded {len(self.samples)} samples from {root_dir}")
        # Print per-class counts
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        for i, name in enumerate(SAFEGUARD_LABELS):
            print(f"    {name:12s}: {counts.get(i, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # FER2013 is grayscale but MobileNetV2 needs RGB
        if self.transform:
            img = self.transform(img)
        return img, label


def make_transforms(img_size: int):
    """Build train/test torchvision transforms for a given image size."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform


# ── Model ─────────────────────────────────────────────────────
def build_model(num_classes=6):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ── Training ──────────────────────────────────────────────────
def train(train_dir: str, test_dir: str, output_path: str,
          epochs: int, batch_size: int, lr: float, img_size: int):
    print("\n-- Loading datasets --------------------------------")
    train_transform, test_transform = make_transforms(img_size)
    print("Train:")
    train_ds = FERDataset(train_dir, train_transform)
    print("Test:")
    test_ds  = FERDataset(test_dir,  test_transform)

    # Class weights to handle imbalance (disgust merged into anger)
    from collections import Counter
    counts = Counter(lbl for _, lbl in train_ds.samples)
    total  = len(train_ds)
    weights = torch.FloatTensor([
        total / (len(SAFEGUARD_LABELS) * max(counts.get(i, 1), 1))
        for i in range(len(SAFEGUARD_LABELS))
    ]).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n-- Building model (MobileNetV2) ---------------------")
    model = build_model(num_classes=len(SAFEGUARD_LABELS)).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    print(f"\n-- Training {epochs} epochs -------------------------------")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds   = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Loss: {train_loss/train_total:.4f} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}% | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  [OK] Saved best model ({val_acc:.2f}%) -> {output_path}")

    print(f"\n-- Training complete --------------------------------")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {output_path}")

    # Per-class accuracy on test set
    print(f"\n-- Per-class accuracy -------------------------------")
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    class_correct = [0] * len(SAFEGUARD_LABELS)
    class_total   = [0] * len(SAFEGUARD_LABELS)
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            for label, pred in zip(labels, preds):
                class_total[label]   += 1
                class_correct[label] += (pred == label).item()

    for i, name in enumerate(SAFEGUARD_LABELS):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"  {name:12s}: {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SafeGuardAI facial emotion model (FER-style dataset).")
    ap.add_argument("--dataset", required=True, help="Dataset root containing train/ and test/")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_emotion.pth"),
                    help="Output .pth path (default: SafeGuardAI/face_emotion.pth)")
    args = ap.parse_args()

    dataset_root = args.dataset
    train_dir = os.path.join(dataset_root, "train")
    test_dir  = os.path.join(dataset_root, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print("ERROR: dataset must contain 'train' and 'test' folders.")
        print("Got:", dataset_root)
        sys.exit(2)

    img_size = int(args.img_size)
    train(train_dir, test_dir, args.out, args.epochs, args.batch_size, args.lr, img_size)
