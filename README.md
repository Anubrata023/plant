# 🌱 Intelligent Plant Disease & Pest Detection System

An enterprise-grade, multi-dataset deep learning pipeline for unified crop disease diagnosis and agricultural pest identification.

---

## 📌 Project Overview

This repository unifies multiple leading agricultural vision datasets into a single, high-performance training pipeline:
- **Total Images Indexed**: 137,468
- **Total Unique Classes**: 66 (52 Plant Disease/Healthy categories + 14 Agricultural Pest categories)
- **Train / Validation Split**: Stratified 80% Train (109,949 images) / 20% Validation (27,519 images)
- **Unified Pipeline Script**: `data/01_unify_datasets.py` executes in ~4 seconds.

---

## 📊 Dataset Integration Matrix

| Dataset Source | Images Indexed | Primary Focus | Structure |
| :--- | :--- | :--- | :--- |
| **PlantVillage** | 54,305 | 38 Disease & Healthy classes | Folder-per-class |
| **Disease-Dataset (entishar)** | 38,000 | In-field disease variations | Harmonized `Condition(Crop)` |
| **Cassava Dataset v2** | 24,395 | 5 Cassava virus & blight diseases | CSV Annotation (`train.csv`) |
| **AgroPest-12** | 13,140 | 12 Crop pest classes (Weevils, Ants, Caterpillars, etc.) | YOLO Bounding Box & Class labels |
| **TeaLeafNet** | 5,000 | Blister Blight, Gray Blight, Red Rust, Red Spider Mite | Geotagged high-res images |
| **RiceLeafsDisease** | 2,628 | Bacterial Leaf Blight, Blast, Brown Spot, Scald | Field conditions |

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
- `data/unified_index.csv`: Full index of all 137,468 samples with filepaths, class names, categories (`disease` / `pest`), sources, and stratified splits.
- `data/class_names.json`: Deterministic 66-class mapping dictionary.

---

## 📁 Repository Structure

```
plant/
│
├── data/
│   ├── 01_unify_datasets.py     # Unified multi-dataset indexer & stratified splitter
│   ├── class_names.json         # Master dictionary of 66 classes
│   ├── unified_index.csv        # Stratified dataset index (137k rows)
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

## 🏷️ Supported Classes (Summary)

- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry**: Healthy
- **Cassava**: Bacterial Blight (CBB), Brown Streak Disease (CBSD), Green Mottle (CGM), Mosaic Disease (CMD), Healthy
- **Cherry**: Powdery Mildew, Healthy
- **Corn (Maize)**: Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
- **Orange**: Huanglongbing (Citrus Greening), Healthy
- **Peach**: Bacterial Spot, Healthy
- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Raspberry**: Healthy
- **Rice**: Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Narrow Brown Spot, Healthy
- **Soybean**: Healthy
- **Squash**: Powdery Mildew
- **Strawberry**: Leaf Scorch, Healthy
- **Tea**: Blister Blight (BB), Gray Blight (GL), Red Rust (RR), Red Spider Mite (RSM)
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus, Healthy
- **Agricultural Pests**: Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, Weevils
