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
# CONFIGURATION FOR METHOD 3: FULL DATA + LABEL SMOOTHING + MIXUP
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "unified_index.csv")
CLASSES_JSON = os.path.join(PROJECT_ROOT, "data", "class_names.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 3e-4

# Method 3 Upgrades:
# 1. Uncap images per class (Set to None for 100% full dataset, or 800-1200 for rapid training)
MAX_IMAGES_PER_CLASS = None

# 2. Label Smoothing: Prevents overconfidence on visually borderline rust/blight spots
LABEL_SMOOTHING = 0.1

# 3. MixUp Augmentation: Blends pairs of images & targets to improve generalization
USE_MIXUP = True
MIXUP_ALPHA = 0.2  # Beta distribution parameter (0.2 favors subtle blends)

# System & Acceleration Settings
USE_AMP = True
NUM_WORKERS = 4

# =====================================================================
# BLOCK 1: CUSTOM DATASET
# =====================================================================
class RiceWheatDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["class_id"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================================
# BLOCK 2: DATA TRANSFORMS
# =====================================================================
def get_transforms():
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
# BLOCK 3: MIXUP AUGMENTATION FUNCTION
# =====================================================================
def apply_mixup(images, labels, alpha=0.2):
    """
    Applies MixUp data augmentation:
    Blends two random images in the batch by a factor of lambda ~ Beta(alpha, alpha)
    and returns the mixed image and both sets of labels with their blend weight.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    perm = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]
    labels_a, labels_b = labels, labels[perm]
    return mixed_images, labels_a, labels_b, lam

# =====================================================================
# BLOCK 4: MODEL BUILDER & CLASS WEIGHTS
# =====================================================================
def build_model(model_name, num_classes):
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
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def compute_class_weights(train_df, num_classes, device):
    counts = train_df["class_id"].value_counts().to_dict()
    total = len(train_df)
    weights = []
    for cid in range(num_classes):
        cnt = max(counts.get(cid, 1), 1)
        w = np.sqrt(total / (num_classes * cnt))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_safe_device():
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"[INFO] CUDA detected but kernels are pending for this architecture ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

# =====================================================================
# BLOCK 5: METHOD 3 TRAINING LOOP
# =====================================================================
def train_method3():
    if not os.path.exists(CSV_PATH) or not os.path.exists(CLASSES_JSON):
        print(f"[ERROR] Unified index or class names missing. Run 01_unify_datasets.py first!")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    device = get_safe_device()
    is_cuda = (device.type == "cuda")

    print("=" * 70)
    print("      METHOD 3 TRAINING: FULL-DATA + LABEL SMOOTHING + MIXUP")
    print("=" * 70)
    print(f"[HARDWARE] Training on: {device}")
    if is_cuda:
        print(f"           GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(CSV_PATH)
    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # 1. Dataset Sizing
    if MAX_IMAGES_PER_CLASS is not None:
        df = pd.concat(
            [grp.sample(n=min(len(grp), MAX_IMAGES_PER_CLASS), random_state=42) for _, grp in df.groupby("class_id")],
            ignore_index=True
        )
        print(f"[DATA] Capped at max {MAX_IMAGES_PER_CLASS} images/class -> {len(df):,} total images")
    else:
        print(f"[DATA] Full Uncapped Dataset Active: {len(df):,} total field images!")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    print(f"[SPLIT] Train: {len(train_df):,} images | Val: {len(val_df):,} images | Classes: {num_classes}")

    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(
        RiceWheatDataset(train_df, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        RiceWheatDataset(val_df, val_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = build_model(MODEL_NAME, num_classes).to(device)
    class_weights = compute_class_weights(train_df, num_classes, device)

    # 2. Criterion with Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    print(f"[UPGRADE 1] Label Smoothing active: epsilon = {LABEL_SMOOTHING}")
    if USE_MIXUP:
        print(f"[UPGRADE 2] MixUp Augmentation active: alpha = {MIXUP_ALPHA}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and is_cuda))

    best_val_acc = 0.0
    history = []
    best_model_path = os.path.join(MODELS_DIR, f"best_{MODEL_NAME}_method3.pth")

    for epoch in range(1, EPOCHS + 1):
        start_t = time.time()

        # --- 1. TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [Train]"):
            images = images.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            optimizer.zero_grad(set_to_none=True)

            # Apply MixUp during training
            if USE_MIXUP:
                images_mixed, labels_a, labels_b, lam = apply_mixup(images, labels, alpha=MIXUP_ALPHA)
                with torch.amp.autocast("cuda", enabled=(USE_AMP and is_cuda)):
                    outputs = model(images_mixed)
                    loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                with torch.amp.autocast("cuda", enabled=(USE_AMP and is_cuda)):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            if USE_AMP and is_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            # For accuracy reporting on mixed batches, compare with dominant label
            target_dominant = labels if (not USE_MIXUP or lam >= 0.5) else labels_b
            train_correct += (preds == target_dominant).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = (train_correct / max(train_total, 1)) * 100

        # --- 2. VALIDATION PHASE (No MixUp, clean evaluation) ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [Val]  "):
                images = images.to(device, non_blocking=is_cuda)
                labels = labels.to(device, non_blocking=is_cuda)

                with torch.amp.autocast("cuda", enabled=(USE_AMP and is_cuda)):
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
            f"-> Epoch {epoch:2d}: Train Acc: {train_acc:.2f}% (Loss: {avg_train_loss:.4f}) | "
            f"Val Acc: {val_acc:.2f}% (Loss: {avg_val_loss:.4f}) | Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": MODEL_NAME,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "method": "method_3_mixup_labelsmoothing",
            }, best_model_path)
            print(f"   🏆 [SAVED NEW BEST MODEL] {best_model_path} (Val Acc: {best_val_acc:.2f}%)")

    metrics_path = os.path.join(MODELS_DIR, "metrics_method3.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 70)
    print(f"METHOD 3 COMPLETE! Best Val Acc: {best_val_acc:.2f}% | Saved to {best_model_path}")
    print("=" * 70)

if __name__ == "__main__":
    train_method3()

