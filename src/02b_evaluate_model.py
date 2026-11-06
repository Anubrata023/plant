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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "unified_index.csv")
CLASSES_JSON = os.path.join(PROJECT_ROOT, "data", "class_names.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_efficientnet_b0.pth")
REPORT_JSON = os.path.join(PROJECT_ROOT, "models", "evaluation_report.json")

IMAGE_SIZE = 224
BATCH_SIZE = 64

class EvalDataset(Dataset):
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

def get_val_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

def build_model(model_name, num_classes):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
    return model

def get_safe_device():
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"[INFO] CUDA device detected ({torch.cuda.get_device_name(0)}), but kernels for its architecture are not in current torch wheel ({e}). Falling back to CPU for evaluation.")
            return torch.device("cpu")
    return torch.device("cpu")

def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(CSV_PATH) or not os.path.exists(CLASSES_JSON):
        print("[ERROR] CSV index or class names JSON missing!")
        return

    device = get_safe_device()
    print(f"\n[EVALUATION] Running model audit on device: {device}")
    if device.type == "cuda":
        print(f"             GPU: {torch.cuda.get_device_name(0)}")

    with open(CLASSES_JSON, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    id_to_name = {int(k): v for k, v in class_names.items()}
    num_classes = len(id_to_name)

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "efficientnet_b0")
    print(f"[MODEL] Loading '{model_name}' trained on {ckpt.get('num_classes', num_classes)} classes...")

    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load validation set
    df = pd.read_csv(CSV_PATH)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    print(f"[DATA] Evaluating on {len(val_df):,} validation images across {num_classes} classes...\n")

    val_loader = DataLoader(
        EvalDataset(val_df, get_val_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_preds = []
    all_targets = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    overall_acc = (all_preds == all_targets).mean() * 100

    # Per-Class Precision, Recall, F1
    report_rows = []
    f1_list = []
    weights_list = []

    for cid in range(num_classes):
        cname = id_to_name.get(cid, f"Class {cid}")
        tp = cm[cid, cid]
        fp = cm[:, cid].sum() - tp
        fn = cm[cid, :].sum() - tp
        support = cm[cid, :].sum()

        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        f1_list.append(f1)
        weights_list.append(support)

        report_rows.append({
            "Class ID": cid,
            "Class Name": cname,
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1-Score": round(f1, 2),
            "Support": int(support),
        })

    macro_f1 = np.mean(f1_list)
    weighted_f1 = np.average(f1_list, weights=weights_list)

    # Format Per-Class Table
    rep_df = pd.DataFrame(report_rows)
    print("=" * 86)
    print("               PER-CLASS DIAGNOSTIC AUDIT (PRECISION, RECALL, F1)")
    print("=" * 86)
    print(f"{'ID':<3} | {'Class Name':<35} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Support'}")
    print("-" * 86)
    for _, row in rep_df.iterrows():
        print(f"{row['Class ID']:<3} | {row['Class Name']:<35} | {row['Precision']:>7.2f}% | {row['Recall']:>6.2f}% | {row['F1-Score']:>6.2f}% | {row['Support']:>5}")
    print("-" * 86)
    print(f"Overall Accuracy: {overall_acc:.2f}% | Macro F1: {macro_f1:.2f}% | Weighted F1: {weighted_f1:.2f}%")
    print(f"Evaluation speed: {len(val_df) / elapsed:.1f} images/sec (Total: {elapsed:.2f}s)")
    print("=" * 86)

    # Top Confused Pairs
    confused_pairs = []
    for true_id in range(num_classes):
        for pred_id in range(num_classes):
            if true_id != pred_id and cm[true_id, pred_id] > 0:
                confused_pairs.append({
                    "true_class": id_to_name[true_id],
                    "predicted_class": id_to_name[pred_id],
                    "errors": int(cm[true_id, pred_id]),
                })

    confused_pairs.sort(key=lambda x: x["errors"], reverse=True)

    print("\n" + "=" * 86)
    print("                      TOP 10 MOST CONFUSED CLASS PAIRS")
    print("=" * 86)
    for idx, cp in enumerate(confused_pairs[:10], 1):
        print(f"  {idx:2d}. True: {cp['true_class']:<32} --> Predicted: {cp['predicted_class']:<32} ({cp['errors']} errors)")

    # Cross-Domain Audit (Crop vs Pest confusion)
    cross_confusions = []
    for cp in confused_pairs:
        t_crop = cp["true_class"].split(" - ")[0].strip()
        p_crop = cp["predicted_class"].split(" - ")[0].strip()
        if t_crop != p_crop:
            cross_confusions.append(cp)

    print("\n" + "=" * 86)
    print(f"               CROSS-DOMAIN ERRORS (e.g. Rice confused with Wheat or Pest)")
    print("=" * 86)
    if cross_confusions:
        for idx, cc in enumerate(cross_confusions[:8], 1):
            print(f"  {idx:2d}. True: {cc['true_class']:<32} --> Predicted: {cc['predicted_class']:<32} ({cc['errors']} errors)")
        total_cross_errors = sum(c["errors"] for c in cross_confusions)
        print(f"\n  Total cross-domain errors: {total_cross_errors} out of {int(cm.sum() - np.diag(cm).sum())} total errors")
    else:
        print("  Excellent! Zero cross-domain errors (No Rice vs Wheat vs Pest cross-contamination).")

    # Save summary report to JSON
    summary_report = {
        "overall_accuracy": round(overall_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "per_class": report_rows,
        "top_confusions": confused_pairs[:15],
        "cross_domain_errors": cross_confusions,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\n[SAVED] Diagnostic report saved to: {REPORT_JSON}\n")

if __name__ == "__main__":
    run_evaluation()
