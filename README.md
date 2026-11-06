# 🌱 Practical In-Field Rice, Wheat & Pest Detection System

An enterprise-grade deep learning pipeline specialized exclusively for practical, real-world field conditions across **Rice**, **Wheat**, and **Agricultural Pests**.

---

## 📌 Project Overview

This repository builds an authentic, field-ready vision pipeline for cereal crops (**Rice** and **Wheat**) alongside major **Agricultural Pests**, purposefully cleaned to eliminate studio and laboratory artifacts:
- **Total Field Images Indexed**: 16,487
- **Total Practical Classes**: 19 (7 In-Field Crop Categories + 12 Agricultural Pest categories)
- **Train / Validation Split**: Stratified 80% Train (13,183 images) / 20% Validation (3,304 images)
- **Real-Case Practical Filter**: `FILTER_WHITE_BACKGROUND = True` automatically purges laboratory samples shot on white paper sheets or plain studio backgrounds.
- **Focus Mode**: `ONLY_RICE_WHEAT_PESTS = True` strictly restricts processing to Rice, Wheat, and Agricultural Pests.

---

## 📊 Dataset Integration Matrix (Field Conditions Only)

| Dataset Source | Images Active | Studio/White Filtered | Target Categories | Structure |
| :--- | :--- | :--- | :--- | :--- |
| **AgroPest-12** | 12,770 | 370 purged | 12 Crop Pest classes (Weevils, Ants, Caterpillars, etc.) | Real-world crop & field imagery |
| **Wheat Diseases** | 3,064 | 164 purged | 4 Classes: Crown & Root Rot, Leaf Rust, Loose Smut, Healthy | Natural field photos & soil conditions |
| **RiceLeafsDisease** | 653 | 1,975 purged | 3 Classes: Brown Spot, Leaf Blast, Healthy | Authenticated in-field foliage photos |
| *Other Datasets (PlantVillage, etc.)* | *Excluded* | *Excluded* | Laboratory studio images / non-target crops | Pre-configured in raw/ |

> **Note on Rice Disease Purification**: In the raw `RiceLeafsDisease` dataset, *Bacterial Leaf Blight*, *Leaf Scald*, and *Narrow Brown Spot* were captured in a laboratory on plain white paper sheets (0% in-field representation). To guarantee practical real-world accuracy without lab background shortcuts, those white-paper samples are purged, retaining only verified in-field cases.

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

### 2. Dataset Unification & Background Purification

To re-index and purify datasets inside `data/raw/`:

```powershell
python data/01_unify_datasets.py
```

This generates:
- `data/unified_index.csv`: Clean index of 16,487 authentic in-field samples with filepaths, class names, categories (`disease` / `pest`), sources, and stratified splits.
- `data/class_names.json`: Master dictionary mapping 19 numeric class IDs to class names.

---

## 📁 Repository Structure

```
plant/
│
├── data/
│   ├── 01_unify_datasets.py     # Fast threaded indexer & white-background purifier
│   ├── class_names.json         # Master dictionary of 19 practical classes
│   ├── unified_index.csv        # Stratified dataset index (16,487 field rows)
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
├── src/
│   └── 02_train_vision_model.py # PyTorch training pipeline with auto class-weighting
├── .gitignore                   # Excludes raw data, virtualenv, and temp files
├── requirements.txt             # Python project dependencies
└── README.md                    # Project documentation
```

---

## 🏷️ Supported Classes (19 Total: 7 Cereal Disease/Healthy + 12 Agricultural Pests)

Every supported crop maintains a paired `Healthy` baseline class:

### 🌾 Rice (3 Field Classes)
- `Rice - Brown Spot`
- `Rice - Healthy`
- `Rice - Leaf Blast`

### 🌾 Wheat (4 Field Classes)
- `Wheat - Crown and Root Rot`
- `Wheat - Healthy`
- `Wheat - Leaf Rust`
- `Wheat - Loose Smut`

### 🐛 Agricultural Pests (12 Field Classes)
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

