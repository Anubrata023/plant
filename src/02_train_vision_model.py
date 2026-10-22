import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =====================================================================
# CONFIGURATION (Change these settings to experiment!)
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "unified_index.csv")
CLASSES_JSON = os.path.join(PROJECT_ROOT, "data", "class_names.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model architecture: "efficientnet_b0" or "mobilenet_v3_large"
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 3e-4

# Speed helper: Caps max images per class so training finishes quickly.
# Set to None if you want to train on the entire un-capped dataset with full GPU.
MAX_IMAGES_PER_CLASS = 400

# =====================================================================
# BLOCK 1: CUSTOM PYTORCH DATASET
# =====================================================================
class RiceWheatDataset(Dataset):
    """
    Custom PyTorch Dataset that loads images on-the-fly from unified_index.csv.
    This prevents loading thousands of images into RAM at once.
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["class_id"])

        # Open image and ensure standard 3-channel RGB format
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black canvas in case a file is corrupted
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================================
# BLOCK 2: IMAGE AUGMENTATION & PREPROCESSING
# =====================================================================
def get_transforms():
    """
    Data Augmentation pipeline.
    Training uses random rotations, crops, flips, and color jitter to make the
    model resilient to field lighting and angle variations.
    Validation only resizes and normalizes to measure true model performance.
    """
    # Standard ImageNet normalization parameters
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 24, IMAGE_SIZE + 24)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform

# =====================================================================
# BLOCK 3: TRANSFER LEARNING MODEL BUILDER
# =====================================================================
def build_model(model_name, num_classes):
    """
    Instantiates a pretrained CNN backbone from torchvision and replaces
    its final classification layer with a new Linear layer sized for our classes.
    """
    print(f"\n[MODEL] Building pretrained '{model_name}' for {num_classes} target classes...")
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model

# =====================================================================
# BLOCK 4: CLASS WEIGHTS & TRAINING / VALIDATION LOOP
# =====================================================================
def compute_class_weights(train_df, num_classes, device):
    """
    Computes smoothed inverse class weights so smaller classes (e.g. rare diseases)
    receive proportional importance during loss calculation, preventing majority class bias.
    """
    counts = train_df["class_id"].value_counts().to_dict()
    total = len(train_df)
    weights = []
    for cid in range(num_classes):
        cnt = max(counts.get(cid, 1), 1)
        w = np.sqrt(total / (num_classes * cnt))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_and_evaluate():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found! Run 'python data/01_unify_datasets.py' first.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Detect hardware accelerator (NVIDIA CUDA GPU, Apple MPS, or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[HARDWARE] Training on device: {device}")
    if device.type == "cuda":
        print(f"           GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(CSV_PATH)

    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Optional cap per class to prevent oversized classes from dominating training time
    if MAX_IMAGES_PER_CLASS is not None:
        df = pd.concat(
            [
                grp.sample(n=min(len(grp), MAX_IMAGES_PER_CLASS), random_state=42)
                for _, grp in df.groupby("class_id")
            ],
            ignore_index=True,
        )
        print(f"[BALANCING] Capped dataset at max {MAX_IMAGES_PER_CLASS} images/class -> {len(df):,} total images")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print(f"[DATA] Train images: {len(train_df):,} | Val images: {len(val_df):,}")

    train_tf, val_tf = get_transforms()

    train_loader = DataLoader(
        RiceWheatDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        RiceWheatDataset(val_df, val_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model(MODEL_NAME, num_classes).to(device)
    class_weights = compute_class_weights(train_df, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    history = []
    best_model_path = os.path.join(MODELS_DIR, f"best_{MODEL_NAME}.pth")

    for epoch in range(1, EPOCHS + 1):
        start_t = time.time()

        # --- 1. TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = (train_correct / max(train_total, 1)) * 100

        # --- 2. VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val] "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = (val_correct / max(val_total, 1)) * 100
        elapsed = time.time() - start_t

        print(
            f"-> Epoch {epoch}: Train Acc: {train_acc:.2f}% (Loss: {avg_train_loss:.4f}) | "
            f"Val Acc: {val_acc:.2f}% (Loss: {avg_val_loss:.4f}) | Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        })

        # Save model weights whenever Validation Accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": MODEL_NAME,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
            }, best_model_path)
            print(f"   [SAVED NEW BEST MODEL] {best_model_path} (Val Acc: {best_val_acc:.2f}%)")

    # Save training metrics for Technical Report and evaluation charts
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{MODEL_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"\n[COMPLETE] Best Validation Accuracy: {best_val_acc:.2f}% | Saved to {best_model_path}")

if __name__ == "__main__":
    train_and_evaluate()
