# 🌱 Rice, Wheat & Agricultural Pest Detection System

An enterprise-grade, high-performance deep learning pipeline specialized for unified **Rice**, **Wheat**, and **Agricultural Pest** identification and diagnosis.

---

## 📌 Project Overview

This repository focuses on cereal crops (**Rice** and **Wheat**) alongside major **Agricultural Pests** into a single, balanced training pipeline:
- **Total Images Indexed**: 18,996
- **Total Unique Classes**: 22 (10 Disease/Healthy categories + 12 Agricultural Pest categories)
- **Train / Validation Split**: Stratified 80% Train (15,186 images) / 20% Validation (3,810 images)
- **Unified Pipeline Script**: `data/01_unify_datasets.py` executes in ~3.5 seconds.
- **Paired Health Baseline**: Both Rice and Wheat have paired Healthy baseline classes (`Rice - Healthy` and `Wheat - Healthy`) to prevent false positives in diagnosis.
- **Focus Mode**: `ONLY_RICE_WHEAT_PESTS = True` (set in `data/01_unify_datasets.py`) isolates target datasets while allowing expansion if needed.

---

## 📊 Dataset Integration Matrix

| Dataset Source | Images Active | Target Categories | Structure |
| :--- | :--- | :--- | :--- |
| **AgroPest-12** | 13,140 | 12 Crop Pest classes (Weevils, Ants, Caterpillars, etc.) | YOLO Bounding Box & Class labels |
| **Wheat Diseases** | 3,228 | 4 Classes: Crown & Root Rot, Leaf Rust, Loose Smut, Healthy | Folder-per-class |
| **RiceLeafsDisease** | 2,628 | 6 Classes: Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Narrow Brown Spot, Healthy | Folder-per-class |
| *Other Crops (PlantVillage, Cassava, etc.)* | *Excluded* | *Excluded by focus filter (`ONLY_RICE_WHEAT_PESTS = True`)* | Pre-configured in raw/ |

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Clone this repository and create a Python 3.11+ virtual environment:

```powershell
git clone https://github.com/Anubrata023/plant.git
cd plant

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

#### For GPU Acceleration (NVIDIA CUDA):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Dataset Unification

To re-index and verify all raw datasets inside `data/raw/`:

```powershell
python data/01_unify_datasets.py
```

This generates:
- `data/unified_index.csv`: Full index of all 18,996 samples with filepaths, class names, categories (`disease` / `pest`), sources, and stratified splits.
- `data/class_names.json`: Master dictionary mapping 22 numeric class IDs to class names.

---

## 📁 Repository Structure

```
plant/
│
├── data/
│   ├── 01_unify_datasets.py     # Unified indexer & stratified splitter for Rice, Wheat & Pests
│   ├── class_names.json         # Master dictionary of 22 target classes
│   ├── unified_index.csv        # Stratified dataset index (18,996 rows)
│   └── raw/
│       ├── agropest_12/         # AgroPest-12 YOLO images and labels
│       ├── rice_diseases/       # RiceLeafsDisease dataset
│       └── wheat_diseases/      # Wheat leaf disease dataset
│
├── reference_repo/              # Reference web interface & MobileNet baseline
│   ├── Crop_disease.py          # Flask backend inference service
│   ├── index.html               # Responsive diagnostic dashboard
│   ├── script.js                # Frontend client logic
│   └── style.css                # Interface styling
│
├── .gitignore                   # Excludes raw data, virtualenv, and temp files
├── requirements.txt             # Python project dependencies
└── README.md                    # Project documentation
```

---

## 🏷️ Supported Classes (22 Total: 10 Cereal Disease/Healthy + 12 Agricultural Pests)

Every supported crop is strictly paired with a `Healthy` baseline class:

### 🌾 Rice (6 Classes)
- `Rice - Bacterial Leaf Blight`
- `Rice - Brown Spot`
- `Rice - Healthy`
- `Rice - Leaf Blast`
- `Rice - Leaf Scald`
- `Rice - Narrow Brown Spot`

### 🌾 Wheat (4 Classes)
- `Wheat - Crown and Root Rot`
- `Wheat - Healthy`
- `Wheat - Leaf Rust`
- `Wheat - Loose Smut`

### 🐛 Agricultural Pests (12 Classes)
- `Pest - Ants`
- `Pest - Bees`
- `Pest - Beetles`
- `Pest - Caterpillars`
- `Pest - Earthworms`
- `Pest - Earwigs`
- `Pest - Grasshoppers`
- `Pest - Moths`
- `Pest - Slugs`
- `Pest - Snails`
- `Pest - Wasps`
- `Pest - Weevils`

