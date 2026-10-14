import os
import re
import json
import random
import time
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUTPUT_CSV = os.path.join("data", "unified_index.csv")
CLASSES_JSON = os.path.join("data", "class_names.json")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 1. Cassava Disease Mapping (ammarali32/cassava-datasetv2 & competition train.csv)
CASSAVA_MAP = {
    "0": "Cassava - Bacterial Blight (CBB)",
    "1": "Cassava - Brown Streak Disease (CBSD)",
    "2": "Cassava - Green Mottle (CGM)",
    "3": "Cassava - Mosaic Disease (CMD)",
    "4": "Cassava - Healthy",
}

# 2. Tea Disease Mapping (TeaLeafNet & Tea Leaf datasets)
TEA_MAP = {
    "bb": "Tea - Blister Blight (BB)",
    "gl": "Tea - Gray Blight (GL)",
    "rr": "Tea - Red Rust (RR)",
    "rsm": "Tea - Red Spider Mite (RSM)",
}

# 3. Rice Disease Mapping (RiceLeafsDisease)
RICE_MAP = {
    "bacterial_leaf_blight": "Rice - Bacterial Leaf Blight",
    "brown_spot": "Rice - Brown Spot",
    "healthy": "Rice - Healthy",
    "leaf_blast": "Rice - Leaf Blast",
    "leaf_scald": "Rice - Leaf Scald",
    "narrow_brown_spot": "Rice - Narrow Brown Spot",
}

# 4. Canonical PlantVillage & Disease-Dataset Mapping
CANONICAL_DE_MAP = {
    "Apple_scab(Apple)": "Apple - Apple Scab",
    "Black_rot(Apple)": "Apple - Black Rot",
    "Cedar_apple_rust(Apple)": "Apple - Cedar Apple Rust",
    "healthy(Apple)": "Apple - Healthy",
    "healthy(Blueberry)": "Blueberry - Healthy",
    "healthy(Cherry)": "Cherry - Healthy",
    "Powdery_mildew(Cherry)": "Cherry - Powdery Mildew",
    "Cercospora_leaf_spot(Corn)": "Corn (maize) - Cercospora Leaf Spot Gray Leaf Spot",
    "Common_rust(Corn)": "Corn (maize) - Common Rust",
    "Northern_Leaf_Blight(Corn)": "Corn (maize) - Northern Leaf Blight",
    "healthy(Corn)": "Corn (maize) - Healthy",
    "Black_rot(Grape)": "Grape - Black Rot",
    "Esca(Grape)": "Grape - Esca (Black Measles)",
    "Leaf_blight(Grape)": "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "healthy(Grape)": "Grape - Healthy",
    "Haunglongbing(Orange)": "Orange - Huanglongbing (Citrus Greening)",
    "healthy(Orange)": "Orange - Healthy",
    "Bacterial_spot(Peach)": "Peach - Bacterial Spot",
    "healthy(Peach)": "Peach - Healthy",
    "Bacterial_spot(Pepper)": "Pepper Bell - Bacterial Spot",
    "healthy(Pepper)": "Pepper Bell - Healthy",
    "Early_blight(Potato)": "Potato - Early Blight",
    "Late_blight(Potato)": "Potato - Late Blight",
    "healthy(Potato)": "Potato - Healthy",
    "healthy(Raspberry)": "Raspberry - Healthy",
    "healthy(Soybean)": "Soybean - Healthy",
    "healthy(Strawberry)": "Strawberry - Healthy",
    "Leaf_scorch(Strawberry)": "Strawberry - Leaf Scorch",
    "Bacterial_spot(Tomato)": "Tomato - Bacterial Spot",
    "Early_blight(Tomato)": "Tomato - Early Blight",
    "Late_blight(Tomato)": "Tomato - Late Blight",
    "Leaf_Mold(Tomato)": "Tomato - Leaf Mold",
    "Septoria_leaf_spot(Tomato)": "Tomato - Septoria Leaf Spot",
    "Spider_mites(Tomato)": "Tomato - Spider Mites Two-spotted Spider Mite",
    "Target_Spot(Tomato)": "Tomato - Target Spot",
    "Tomato_mosaic_virus(Tomato)": "Tomato - Tomato Mosaic Virus",
    "Tomato_Yellow_Leaf_Curl_Virus(Tomato)": "Tomato - Tomato Yellow Leaf Curl Virus",
    "healthy(Tomato)": "Tomato - Healthy",
}

CANONICAL_PV_MAP = {
    "Apple___Apple_scab": "Apple - Apple Scab",
    "Apple___Black_rot": "Apple - Black Rot",
    "Apple___Cedar_apple_rust": "Apple - Cedar Apple Rust",
    "Apple___healthy": "Apple - Healthy",
    "Blueberry___healthy": "Blueberry - Healthy",
    "Cherry_(including_sour)___healthy": "Cherry - Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry - Powdery Mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn (maize) - Cercospora Leaf Spot Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn (maize) - Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn (maize) - Northern Leaf Blight",
    "Corn_(maize)___healthy": "Corn (maize) - Healthy",
    "Grape___Black_rot": "Grape - Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape - Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape___healthy": "Grape - Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange - Huanglongbing (Citrus Greening)",
    "Peach___Bacterial_spot": "Peach - Bacterial Spot",
    "Peach___healthy": "Peach - Healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper Bell - Bacterial Spot",
    "Pepper,_bell___healthy": "Pepper Bell - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Raspberry___healthy": "Raspberry - Healthy",
    "Soybean___healthy": "Soybean - Healthy",
    "Squash___Powdery_mildew": "Squash - Powdery Mildew",
    "Strawberry___Leaf_scorch": "Strawberry - Leaf Scorch",
    "Strawberry___healthy": "Strawberry - Healthy",
    "Tomato___Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato___Early_blight": "Tomato - Early Blight",
    "Tomato___Late_blight": "Tomato - Late Blight",
    "Tomato___Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato - Spider Mites Two-spotted Spider Mite",
    "Tomato___Target_Spot": "Tomato - Target Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato - Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato - Tomato Yellow Leaf Curl Virus",
    "Tomato___healthy": "Tomato - Healthy",
}

# Generic container folders that are NOT disease/pest names
IGNORE_FOLDERS = {
    "train", "val", "valid", "validation", "test", "images", "labels",
    "raw", "color", "segmented", "grayscale", "plantvillage", "plantvillage dataset",
    "new plant diseases dataset(augmented)", "new plant diseases dataset",
    "ip102_v1.1", "classification", "detection", "data", "dataset", "test_images", "backup",
    "tea_leaf_850_notresized_backup"
}

def load_ip102_class_map(dataset_path):
    """Finds classes.txt inside IP102 so numbered folders get real insect names."""
    ip102_map = {}
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower() == "classes.txt":
                txt_path = os.path.join(root, f)
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as fp:
                    for idx, line in enumerate(fp):
                        clean_line = re.sub(r"^\d+\s+", "", line.strip())
                        if clean_line:
                            ip102_map[str(idx)] = f"Pest - {clean_line.title()}"
                            ip102_map[str(idx + 1)] = f"Pest - {clean_line.title()}"
                return ip102_map
    return ip102_map

def parse_yolo_yaml_names(dataset_path):
    """Reads class names from data.yaml for YOLO datasets like AgroPest-12 and FieldPlant."""
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".yaml", ".yml")):
                yaml_path = os.path.join(root, f)
                with open(yaml_path, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                list_match = re.search(r"names\s*:\s*\[([^\]]+)\]", content)
                if list_match:
                    raw_names = [n.strip().strip("'\"") for n in list_match.group(1).split(",") if n.strip()]
                    return {str(i): n for i, n in enumerate(raw_names)}
                lines = content.splitlines()
                names_dict = {}
                in_names = False
                idx = 0
                for line in lines:
                    if re.match(r"^\s*names\s*:", line):
                        in_names = True
                        continue
                    if in_names:
                        m_kv = re.match(r"^\s*(\d+)\s*:\s*(.+)$", line)
                        m_list = re.match(r"^\s*-\s*(.+)$", line)
                        if m_kv:
                            names_dict[m_kv.group(1)] = m_kv.group(2).strip().strip("'\"")
                        elif m_list:
                            names_dict[str(idx)] = m_list.group(1).strip().strip("'\"")
                            idx += 1
                        elif line.strip() and not line.startswith(" "):
                            break
                if names_dict:
                    return names_dict
    return {}

def scan_yolo_dataset(dataset_path, dataset_name, yolo_names):
    """Pairs images/xxx.jpg with labels/xxx.txt to extract the class for YOLO datasets."""
    records = []
    category = "pest" if "pest" in dataset_name.lower() else "disease"
    prefix = "Pest" if category == "pest" else "Crop"

    for root, _, files in os.walk(dataset_path):
        if os.path.basename(root).lower() != "images":
            continue
        labels_dir = os.path.join(os.path.dirname(root), "labels")
        if not os.path.exists(labels_dir):
            continue

        abs_root = os.path.abspath(root)
        for img_file in files:
            if not img_file.lower().endswith(VALID_EXTENSIONS):
                continue
            base_name = os.path.splitext(img_file)[0]
            label_file = os.path.join(labels_dir, base_name + ".txt")
            if not os.path.exists(label_file):
                continue

            with open(label_file, "r", encoding="utf-8", errors="ignore") as lf:
                first_line = lf.readline().strip()
                if not first_line:
                    continue
                class_num = first_line.split()[0]
                raw_label = yolo_names.get(class_num, f"{dataset_name}_Class_{class_num}")
                records.append({
                    "image_path": os.path.join(abs_root, img_file),
                    "label_name": f"{prefix} - {raw_label}",
                    "category": category,
                    "source": dataset_name,
                })
    return records

def scan_cassava_dataset(dataset_path, dataset_name):
    """Indexes Cassava images using train.csv metadata instead of folder names."""
    records = []
    train_csv = None
    train_dir = None
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower() == "train.csv":
                train_csv = os.path.join(root, f)
        for d in dirs:
            if d.lower() == "train_images":
                train_dir = os.path.join(root, d)
        if train_csv and train_dir:
            break

    if not train_csv or not train_dir:
        return records

    abs_train_dir = os.path.abspath(train_dir)
    existing_images = set(os.listdir(abs_train_dir))
    df_csv = pd.read_csv(train_csv)
    for _, row in df_csv.iterrows():
        img_id = str(row["image_id"]).strip()
        label_code = str(row["label"]).strip()
        if img_id in existing_images:
            label_name = CASSAVA_MAP.get(label_code, f"Cassava - Class {label_code}")
            records.append({
                "image_path": os.path.join(abs_train_dir, img_id),
                "label_name": label_name,
                "category": "disease",
                "source": dataset_name,
            })
    return records

def clean_label_name(folder_name, source_dataset, ip102_map):
    """Converts folder names or numeric IDs from any dataset into uniform labels."""
    name = folder_name.strip()
    s_lower = source_dataset.lower()

    if "ip102" in s_lower and name.isdigit():
        return ip102_map.get(str(int(name)), f"Pest - IP102 Class {int(name):03d}")

    if s_lower == "disease_extra":
        if name in CANONICAL_DE_MAP:
            return CANONICAL_DE_MAP[name]
        m = re.match(r"^(.+?)\s*\(\s*(.+?)\s*\)$", name)
        if m:
            cond, crop = m.group(1).replace("_", " ").title(), m.group(2).replace("_", " ").title()
            return f"{crop} - {cond}"

    if s_lower == "plantvillage":
        if name in CANONICAL_PV_MAP:
            return CANONICAL_PV_MAP[name]

    if s_lower == "tea_diseases":
        name_lower = name.lower()
        if name_lower in TEA_MAP:
            return TEA_MAP[name_lower]

    if s_lower == "rice_diseases":
        name_lower = name.lower()
        if name_lower in RICE_MAP:
            return RICE_MAP[name_lower]

    cleaned = name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def scan_all_datasets():
    t0 = time.time()
    records = []

    if not os.path.exists(RAW_DIR):
        print(f"[ERROR] {RAW_DIR} not found! Please check your data/raw folder.", flush=True)
        return

    for dataset_name in sorted(os.listdir(RAW_DIR)):
        dataset_path = os.path.join(RAW_DIR, dataset_name)
        if not os.path.isdir(dataset_path) or dataset_name == "poulami_crop_repo":
            continue

        print(f"Scanning '{dataset_name}' ...", flush=True)

        # 1. Cassava with CSV structure
        if "cassava" in dataset_name.lower():
            cassava_recs = scan_cassava_dataset(dataset_path, dataset_name)
            if cassava_recs:
                records.extend(cassava_recs)
                print(f"   -> [CSV Format] Found {len(cassava_recs):,} labeled images in '{dataset_name}'", flush=True)
                continue

        # 2. YOLO structure (AgroPest-12, FieldPlant)
        yolo_names = parse_yolo_yaml_names(dataset_path)
        if yolo_names:
            yolo_records = scan_yolo_dataset(dataset_path, dataset_name, yolo_names)
            if yolo_records:
                records.extend(yolo_records)
                print(f"   -> [YOLO Format] Found {len(yolo_records):,} labeled images in '{dataset_name}'", flush=True)
                continue

        # 3. Standard Folder-per-class structure
        ip102_map = load_ip102_class_map(dataset_path) if "ip102" in dataset_name.lower() else {}
        count_for_ds = 0

        for root, dirs, files in os.walk(dataset_path):
            root_lower = root.lower()
            # Skip duplicate grayscale/segmented or backup copies
            if any(skip in root_lower for skip in ["grayscale", "segmented", "annotations", "backup", "tea_leaf_3400"]):
                continue

            img_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
            if not img_files:
                continue

            parent_folder = os.path.basename(root)
            if parent_folder.lower() in IGNORE_FOLDERS:
                continue

            label = clean_label_name(parent_folder, dataset_name, ip102_map)
            
            # Determine category (pest vs disease)
            is_pest = any(p in label.lower() or p in dataset_name.lower() for p in ["pest", "spider mite", "mite", "ip102"])
            category = "pest" if is_pest else "disease"

            abs_root = os.path.abspath(root)
            for img_file in img_files:
                records.append({
                    "image_path": os.path.join(abs_root, img_file),
                    "label_name": label,
                    "category": category,
                    "source": dataset_name,
                })
                count_for_ds += 1

        if count_for_ds > 0:
            print(f"   -> [Folder Format] Found {count_for_ds:,} images in '{dataset_name}'", flush=True)
        else:
            # Check if there are unextracted/partial archive files
            zips = [f for f in os.listdir(dataset_path) if f.lower().endswith((".zip", ".kaggle-partial"))]
            if zips:
                print(f"   -> [Notice] '{dataset_name}' contains unextracted/partial archive files ({', '.join(zips[:2])}). Skipped.", flush=True)
            else:
                print(f"   -> [Notice] No valid images found in '{dataset_name}'.", flush=True)

    if not records:
        print("[WARNING] No images found! Check if your datasets are unzipped inside data/raw/.", flush=True)
        return

    df = pd.DataFrame(records)

    # Filter out accidental tiny folders (< 10 images)
    class_counts = df["label_name"].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df["label_name"].isin(valid_classes)].copy()

    # Create master deterministic numeric IDs (0, 1, 2...)
    unique_labels = sorted(df["label_name"].unique())
    label_to_id = {name: idx for idx, name in enumerate(unique_labels)}
    id_to_label = {idx: name for idx, name in enumerate(unique_labels)}
    df["class_id"] = df["label_name"].map(label_to_id)

    # Stratified 80% Train / 20% Validation split per class
    rng = random.Random(42)
    splits = {}
    for label, group in df.groupby("label_name"):
        idx_list = group.index.tolist()
        rng.shuffle(idx_list)
        n_train = max(1, int(len(idx_list) * 0.8))
        train_set = set(idx_list[:n_train])
        for i in idx_list:
            splits[i] = "train" if i in train_set else "val"
    df["split"] = df.index.map(splits)

    df.to_csv(OUTPUT_CSV, index=False)
    with open(CLASSES_JSON, "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=4)

    total_time = time.time() - t0
    disease_classes = df[df['category'] == 'disease']['label_name'].nunique()
    pest_classes = df[df['category'] == 'pest']['label_name'].nunique()

    print("\n" + "=" * 55, flush=True)
    print("         UNIFIED DATASET SUMMARY REPORT", flush=True)
    print("=" * 55, flush=True)
    print(f"Total Images Indexed : {len(df):,}", flush=True)
    print(f"Total Unique Classes : {len(unique_labels)} ({disease_classes} Disease, {pest_classes} Pest)", flush=True)
    print(f"Train Set Images     : {(df['split']=='train').sum():,} ({(df['split']=='train').mean()*100:.1f}%)", flush=True)
    print(f"Val Set Images       : {(df['split']=='val').sum():,} ({(df['split']=='val').mean()*100:.1f}%)", flush=True)
    print(f"Processing Time      : {total_time:.2f} seconds", flush=True)
    print("\nImages Contributed by Each Dataset:", flush=True)
    print(df["source"].value_counts().to_string(), flush=True)
    print(f"\n[SAVED] {OUTPUT_CSV}", flush=True)
    print(f"[SAVED] {CLASSES_JSON}", flush=True)
    print("=" * 55, flush=True)

if __name__ == "__main__":
    scan_all_datasets()
