# 🌱 Intelligent Plant Disease & Pest Detection System

An enterprise-grade, multi-dataset deep learning pipeline for unified crop disease diagnosis and agricultural pest identification.

---

## 📌 Project Overview

This repository unifies multiple leading agricultural vision datasets into a single, high-performance training pipeline:
- **Total Images Indexed**: 120,670
- **Total Unique Classes**: 58 (46 Plant Disease/Healthy categories + 12 Agricultural Pest categories)
- **Train / Validation Split**: Stratified 80% Train (96,512 images) / 20% Validation (24,158 images)
- **Unified Pipeline Script**: `data/01_unify_datasets.py` executes in ~4 seconds.
- **Paired Health Baseline**: Configured with `EXCLUDE_INCOMPLETE_CROPS = True` to guarantee that every single supported crop has both disease and healthy baseline classes.

---

## 📊 Dataset Integration Matrix

| Dataset Source | Images Active | Primary Focus | Structure |
| :--- | :--- | :--- | :--- |
| **PlantVillage** | 45,507 | 30 Paired Disease & Healthy classes | Folder-per-class |
| **Disease-Dataset (entishar)** | 35,000 | In-field disease variations | Harmonized `Condition(Crop)` |
| **Cassava Dataset v2** | 24,395 | 5 Cassava virus & blight diseases + Healthy | CSV Annotation (`train.csv`) |
| **AgroPest-12** | 13,140 | 12 Crop pest classes (Weevils, Ants, Caterpillars, etc.) | YOLO Bounding Box & Class labels |
| **RiceLeafsDisease** | 2,628 | Bacterial Leaf Blight, Blast, Brown Spot, Scald + Healthy | Field conditions |
| *TeaLeafNet* | *Excluded* | *Excluded by default (no healthy baseline available)* | Geotagged high-res images |

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
- `data/unified_index.csv`: Full index of all 120,670 samples with filepaths, class names, categories (`disease` / `pest`), sources, and stratified splits.
- `data/class_names.json`: Deterministic 58-class mapping dictionary.

---

## 📁 Repository Structure

```
plant/
│
├── data/
│   ├── 01_unify_datasets.py     # Unified multi-dataset indexer & stratified splitter
│   ├── class_names.json         # Master dictionary of 58 paired classes
│   ├── unified_index.csv        # Stratified dataset index (120k rows)
│   └── raw/                     # Raw datasets (excluded from git)
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

## 🏷️ Supported Classes (12 Paired Crops + 12 Agricultural Pests)

Every supported crop is strictly paired with a `Healthy` baseline class to prevent false positives during diagnosis:

- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Cassava**: Bacterial Blight (CBB), Brown Streak Disease (CBSD), Green Mottle (CGM), Mosaic Disease (CMD), Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn (Maize)**: Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
- **Orange**: Huanglongbing (Citrus Greening), Healthy
- **Peach**: Bacterial Spot, Healthy
- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Rice**: Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Narrow Brown Spot, Healthy
- **Strawberry**: Leaf Scorch, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus, Healthy
- **Agricultural Pests**: Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, Weevils
