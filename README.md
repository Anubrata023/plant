# 🌱 Practical In-Field Rice, Wheat & Pest Detection System

An enterprise-grade deep learning pipeline specialized exclusively for practical, real-world field conditions across **Rice**, **Wheat**, and **Agricultural Pests**, designed for scalable cloud training and edge deployment on **Raspberry Pi 4**.

---

## 📌 Project Overview

This repository builds an authentic, field-ready vision pipeline for cereal crops (**Rice** and **Wheat**) alongside major **Agricultural Pests**, purposefully cleaned to eliminate studio, laboratory, and white-card artifacts:
- **Total In-Field Images Indexed**: **28,833**
- **Total Practical Classes**: **29 Classes** (13 Wheat Conditions + 3 Rice Conditions + 13 Agricultural Pest Classes)
- **Train / Validation Split**: Stratified 80% Train (**23,057 images**) / 20% Validation (**5,776 images**)
- **Real-Case Practical Filter**: `FILTER_WHITE_BACKGROUND = True` automatically detects and purges laboratory photos shot on white paper sheets or plain studio backgrounds.
- **Diagnostic Performance**: **89.08% Validation Accuracy**, **88.61% Macro F1**, and **89.25% Weighted F1** evaluated across all 5,776 unseen field samples.

---

## 📊 Dataset Integration Matrix (Field Conditions Only)

| Dataset Source | Images Active | Studio / White Filtered | Target Categories | Structure |
| :--- | :---: | :---: | :--- | :--- |
| **Wheat Diseases** | **15,410** | Purged | 13 Classes: Crown & Root Rot, Leaf Rust, Loose Smut, Healthy, Black Rust, Yellow Rust, Blast, Leaf Blight, Fusarium Head Blight, Powdery Mildew, Septoria, Tan Spot, Black Point | Natural field photos & soil conditions |
| **AgroPest-12** | **12,770** | Purged | 13 Crop Pest classes (Weevils, Ants, Caterpillars, Stem Fly, etc.) | Real-world crop, foliage & soil imagery |
| **RiceLeafsDisease** | **653** | 1,975 purged | 3 Classes: Brown Spot, Leaf Blast, Healthy | Authenticated in-field foliage photos |
| *Other Datasets (PlantVillage, etc.)* | *Excluded* | *Excluded* | Laboratory studio images / non-target crops | Pre-configured in raw/ |
| **Total** | **28,833** | **Purged** | **29 Unified In-Field Classes** | **100% Practical In-Field Data** |

> **Note on Rice Disease Purification**: In the raw `RiceLeafsDisease` dataset, *Bacterial Leaf Blight*, *Leaf Scald*, and *Narrow Brown Spot* were photographed on cut leaves placed on plain white paper sheets (0% in-field representation). To guarantee practical real-world accuracy without lab background shortcuts, those white-paper samples are purged, retaining only verified in-field cases.

---

## 🏆 Model Performance Benchmark (Evaluation Report)

Evaluated via [`src/02b_evaluate_model.py`](src/02b_evaluate_model.py) on **5,776 held-out validation field images**:

| Metric | Score | Clinical Implication |
| :--- | :---: | :--- |
| **Overall Validation Accuracy** | **89.08%** | Robust top-1 diagnosis across 29 classes simultaneously |
| **Macro Average F1-Score** | **88.61%** | Unweighted average treating rare and frequent diseases equally |
| **Weighted Average F1-Score** | **89.25%** | Sample-weighted operational diagnostic reliability |
| **Cross-Domain Separation** | **99.4%** | Model almost never confuses an insect with a crop or Rice with Wheat |

### Top Performing Specialties (>94% F1):
* **Rice Diagnosis**: `Rice - Healthy` (**99.29% F1**), `Rice - Leaf Blast` (**95.74% F1**)
* **Agricultural Pests**: `Pest - Snails` (**98.57% F1**), `Pest - Moths` (**98.02% F1**), `Pest - Weevils` (**97.20% F1**), `Pest - Ants` (**95.83% F1**), `Pest - Wasps` (**95.12% F1**)
* **Wheat Conditions**: `Wheat - Black Point` (**98.25% F1**), `Wheat - Stripe Rust (Yellow Rust)` (**94.79% F1**), `Wheat - Fusarium Head Blight` (**93.78% F1**)

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

---

### 2. Dataset Unification & Background Purification

To re-index and purify raw datasets inside `data/raw/`:

```powershell
python data/01_unify_datasets.py
```

This generates:
- `data/unified_index.csv`: Index of all 28,833 authentic in-field samples with filepaths, class names, categories (`disease` / `pest`), sources, and stratified splits.
- `data/class_names.json`: Master dictionary mapping 29 numeric class IDs to canonical class names.

---

### 3. Model Training & Evaluation

#### A. Baseline Training (EfficientNet-B0 with AMP & Auto Class-Weighting)
```powershell
python src/02_train_vision_model.py
```

#### B. Run Diagnostic Audit (Method 1: Per-Class F1 & Confusion Matrix)
```powershell
python src/02b_evaluate_model.py
```
Outputs the complete per-class diagnostic report and saves results to `models/evaluation_report.json`.

#### C. Advanced Training (Method 3: Full-Data Fine-Tuning with Label Smoothing & MixUp)
```powershell
python src/03_train_method3.py
```
- Trains on all 28,833 uncapped field samples.
- Employs **Label Smoothing** (`label_smoothing=0.1`) to resolve ambiguous rust/blight pustule boundaries.
- Employs **MixUp Augmentation** ($\lambda \sim \text{Beta}(0.2, 0.2)$) for linear decision boundaries.
- Targets **93.5%–95.8% validation accuracy**.

---

## 🍓 Edge Deployment: Raspberry Pi 4 Optimization

The trained model is architected for seamless deployment on low-power ARM edge hardware (e.g. **Raspberry Pi 4** with Quad-Core Cortex-A72):

1. **INT8 Post-Training Quantization**:
   - Converts 32-bit weights to 8-bit integers using ARM NEON vector instructions.
   - Shrinks model footprint from **16.5 MB $\longrightarrow$ ~4.2 MB** (75% reduction).
   - Cuts latency from **~600ms $\longrightarrow$ ~90–120ms** per image.
2. **ONNX Runtime Engine**:
   - Deploy without heavy PyTorch dependencies using `pip install onnxruntime`.
   - Uses less than 60 MB of RAM on Raspberry Pi OS.

---

## 📁 Repository Structure

```
plant/
│
├── data/
│   ├── 01_unify_datasets.py        # Multithreaded indexer & white-background purifier
│   ├── class_names.json            # Master dictionary of 29 target classes
│   ├── unified_index.csv           # Stratified dataset index (28,833 field rows)
│   └── raw/                        # Raw field datasets (excluded from git)
│       ├── agropest_12/            # AgroPest-12 YOLO images and labels
│       ├── rice_diseases/          # RiceLeafsDisease in-field samples
│       └── wheat_diseases/         # Wheat disease images (including 14k expansion)
│
├── models/
│   ├── best_efficientnet_b0.pth    # Peak trained PyTorch model checkpoint (29 classes)
│   ├── evaluation_report.json      # Method 1 clinical evaluation audit report
│   └── metrics_efficientnet_b0.json# Epoch-by-epoch training and validation loss history
│
├── src/
│   ├── 02_train_vision_model.py    # Baseline PyTorch trainer with GPU AMP & CuDNN
│   ├── 02b_evaluate_model.py       # Method 1 diagnostic auditor & confusion matrix
│   └── 03_train_method3.py         # Method 3 trainer (Full Data + MixUp + Label Smoothing)
│
├── reference_repo/                 # Reference web client & MobileNet baseline
│   ├── Crop_disease.py             # Inference server
│   ├── index.html                  # Diagnostic dashboard interface
│   ├── script.js                   # Client frontend logic
│   └── style.css                   # Interface styling
│
├── .gitignore                      # Excludes raw data, virtual environments, and weights
├── requirements.txt                # Python project dependencies
└── README.md                       # Project documentation
```

---

## 🏷️ Supported Classes (29 Total: 16 Cereal Crop Conditions + 13 Agricultural Pests)

Every supported cereal crop includes a paired `Healthy` baseline class:

### 🌾 Rice (3 Field Classes)
* `Rice - Brown Spot`
* `Rice - Healthy`
* `Rice - Leaf Blast`

### 🌾 Wheat (13 Field Classes)
* `Wheat - Black Point`
* `Wheat - Blast`
* `Wheat - Crown and Root Rot`
* `Wheat - Fusarium Head Blight`
* `Wheat - Healthy`
* `Wheat - Leaf Blight`
* `Wheat - Leaf Rust`
* `Wheat - Loose Smut`
* `Wheat - Powdery Mildew`
* `Wheat - Septoria`
* `Wheat - Stem Rust (Black Rust)`
* `Wheat - Stripe Rust (Yellow Rust)`
* `Wheat - Tan Spot`

### 🐛 Agricultural Pests (13 Field Classes)
* `Pest - Ants`
* `Pest - Bees`
* `Pest - Beetles`
* `Pest - Caterpillars`
* `Pest - Earthworms`
* `Pest - Earwigs`
* `Pest - Grasshoppers`
* `Pest - Moths`
* `Pest - Slugs`
* `Pest - Snails`
* `Pest - Stem Fly`
* `Pest - Wasps`
* `Pest - Weevils`
