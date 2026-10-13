# Member A — Kaggle Notebook: Surgical Tool Tip Detection
### Endoscope Auto-Camera Project | Perception Workstream
### Dataset: m2cai16-tool-locations (Stanford AI Lab)

> **How to use this file:** 
> Copy and paste each cell below into your Kaggle Notebook in order.
> Every cell is self-contained, robust, and commented with **what** it does and **why**.
> 
> **Prerequisite:** Upload `m2cai16-tool-locations.zip` via Kaggle's "+ Add Input" -> "Upload a Dataset" button.

---

## PART 0 — Environment Setup

### Cell 0.1 — Install & verify libraries
> **Why:** We need `ultralytics` for YOLOv8-nano. Make sure **GPU T4** is selected in Kaggle settings (right panel -> Session options -> Accelerator).

```python
# Cell 0.1 — Install packages
!pip install -q ultralytics==8.2.0

import torch, subprocess
print("PyTorch Version:", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name    :", torch.cuda.get_device_name(0))
else:
    print("⚠️ WARNING: GPU not detected. Select 'GPU T4 x1' in the right panel under Session options!")
```

---

### Cell 0.2 — Imports and folder structure
> **Why:** Sets up imports, seeds for reproducibility, and directory structure inside `/kaggle/working/` so results and weights persist cleanly.

```python
# Cell 0.2 — Imports & directories setup
import os, sys, time, shutil, random, json, warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

ROOT        = Path("/kaggle/working")
DATA_DIR    = ROOT / "data"
YOLO_DIR    = DATA_DIR / "yolo_dataset"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

for d in [DATA_DIR, YOLO_DIR, CKPT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Directory tree created at /kaggle/working/")
```

---

## PART 1 — Dataset Discovery & Parsing (m2cai16-tool-locations)

### Cell 1.1 — Locate uploaded dataset
> **Why:** Checks `/kaggle/input/` to automatically find `JPEGImages` and `Annotations` folders from your uploaded `m2cai16-tool-locations` dataset.

```python
# Cell 1.1 — Auto-detect m2cai16 dataset folders
input_path = Path("/kaggle/input")

jpeg_dirs = list(input_path.rglob("JPEGImages"))
xml_dirs  = list(input_path.rglob("Annotations"))

if not jpeg_dirs or not xml_dirs:
    # Check if dataset was uploaded as a zip file inside input
    zip_files = list(input_path.rglob("*.zip"))
    if zip_files:
        print(f"📦 Found zip file: {zip_files[0]}. Extracting...")
        import zipfile
        with zipfile.ZipFile(zip_files[0], 'r') as zf:
            zf.extractall("/kaggle/working/data/m2cai16_raw")
        jpeg_dirs = list(Path("/kaggle/working/data/m2cai16_raw").rglob("JPEGImages"))
        xml_dirs  = list(Path("/kaggle/working/data/m2cai16_raw").rglob("Annotations"))

assert len(jpeg_dirs) > 0 and len(xml_dirs) > 0, (
    "❌ Could not find JPEGImages or Annotations folder! "
    "Please make sure your m2cai16 dataset is uploaded via '+ Add Input'."
)

JPEG_DIR = jpeg_dirs[0]
XML_DIR  = xml_dirs[0]

all_images = sorted(list(JPEG_DIR.glob("*.jpg")) + list(JPEG_DIR.glob("*.png")))
all_xmls   = sorted(list(XML_DIR.glob("*.xml")))

print(f"✅ Found dataset!")
print(f"   Images folder     : {JPEG_DIR} ({len(all_images)} frames)")
print(f"   Annotations folder: {XML_DIR} ({len(all_xmls)} XML files)")
```

---

### Cell 1.2 — Parse XML Annotations & inspect classes
> **Why:** Reads the Pascal VOC XML files to see bounding boxes, dimensions, and tool classes.

```python
# Cell 1.2 — Parse XML labels
CLASSES = [
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "SpecimenBag"
]

def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append({
            "name": name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "width": w, "height": h
        })
    return boxes

sample_xml = all_xmls[0]
sample_boxes = parse_voc_xml(sample_xml)
print(f"Sample XML ({sample_xml.name}) objects:")
for b in sample_boxes:
    print(f" - Tool: {b['name']} at [{b['xmin']:.0f}, {b['ymin']:.0f}, {b['xmax']:.0f}, {b['ymax']:.0f}]")
```

---

### Cell 1.3 — Train / Val / Test Split & YOLO Dataset Builder
> **Why:** Creates YOLO format labels (`.txt` files) and splits the dataset into `train` (70%), `val` (15%), and `test` (15%). Saves `dataset.yaml`.

```python
# Cell 1.3 — Build YOLO dataset structure
for split in ["train", "val", "test"]:
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# Pair up XML and images
valid_pairs = []
for xf in all_xmls:
    stem = xf.stem
    img_candidates = list(JPEG_DIR.glob(f"{stem}.*"))
    if img_candidates:
        valid_pairs.append((img_candidates[0], xf))

random.seed(42)
random.shuffle(valid_pairs)

n_total = len(valid_pairs)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

split_data = {
    "train": valid_pairs[:n_train],
    "val":   valid_pairs[n_train:n_train+n_val],
    "test":  valid_pairs[n_train+n_val:]
}

for split_name, pairs in split_data.items():
    for img_path, xml_path in pairs:
        boxes = parse_voc_xml(xml_path)
        if not boxes:
            continue
        
        yolo_lines = []
        for b in boxes:
            # Map tool name to class ID (treat all as class 0 'tool' or multi-class)
            # For camera centring, single class 'tool' (class 0) gives the most stable detection
            cls_id = 0
            
            w, h = b["width"], b["height"]
            xc = ((b["xmin"] + b["xmax"]) / 2.0) / w
            yc = ((b["ymin"] + b["ymax"]) / 2.0) / h
            bw = (b["xmax"] - b["xmin"]) / w
            bh = (b["ymax"] - b["ymin"]) / h
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        
        # Copy image and write label file
        dst_img = YOLO_DIR / "images" / split_name / img_path.name
        dst_lbl = YOLO_DIR / "labels" / split_name / f"{img_path.stem}.txt"
        shutil.copy(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(yolo_lines))

print(f"✅ Converted and split {n_total} annotated frames:")
for s in ["train", "val", "test"]:
    n_imgs = len(list((YOLO_DIR / 'images' / s).glob('*')))
    print(f"   {s:<6}: {n_imgs} frames")

# Write YOLO dataset.yaml
yaml_text = f"""path: {YOLO_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['surgical_tool']
"""
with open(YOLO_DIR / "dataset.yaml", "w") as f:
    f.write(yaml_text)

print(f"✅ Created {YOLO_DIR / 'dataset.yaml'}")
```

---

### Cell 1.4 — Visualise 4 Sample Ground-Truth Bounding Boxes
> **Why:** Sanity check to confirm bounding boxes align accurately with laparoscopic tools.

```python
# Cell 1.4 — Inspect annotated samples
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sample_pairs = random.sample(split_data["train"], 4)

for idx, (img_path, xml_path) in enumerate(sample_pairs):
    ax = axes[idx // 2, idx % 2]
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    boxes = parse_voc_xml(xml_path)
    for b in boxes:
        x1, y1, x2, y2 = int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Approximate tool tip: center-bottom of bounding box
        tip_x, tip_y = int((x1 + x2) / 2), int(y2)
        cv2.circle(img_rgb, (tip_x, tip_y), 8, (255, 0, 0), -1)
        cv2.putText(img_rgb, b["name"], (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    ax.imshow(img_rgb)
    ax.set_title(f"{img_path.name} ({len(boxes)} tool(s))")
    ax.axis("off")

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "01_sample_ground_truth.png"), dpi=150)
plt.show()
print("✅ Saved visual check to results/01_sample_ground_truth.png")
```

---

## PART 2 — Baseline Detector: K-Means Clustering

### Cell 2.1 — Fit K-Means on Sampled Training Pixels
> **Why:** Unsupervised baseline. Clusters pixels by colour in HSV. Metallic tools have low saturation compared to liver/abdominal tissue.

```python
# Cell 2.1 — Train K-Means (k=3)
K = 3
train_imgs = list((YOLO_DIR / "images" / "train").glob("*"))[:300]
print(f"Sampling pixels from {len(train_imgs)} training frames...")

sampled_hsv = []
for p in train_imgs:
    im = cv2.imread(str(p))
    if im is None: continue
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    idx = np.random.choice(len(hsv), 1000, replace=False)
    sampled_hsv.append(hsv[idx])

sampled_hsv = np.vstack(sampled_hsv)

kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=5000, n_init=5)
kmeans.fit(sampled_hsv)

# Instrument cluster = lowest average saturation
cluster_sat = kmeans.cluster_centers_[:, 1]
instrument_cluster = int(np.argmin(cluster_sat))

print(f"K-Means Centers (HSV):\n{kmeans.cluster_centers_.round(1)}")
print(f"→ Instrument cluster: {instrument_cluster} (Lowest Saturation: {cluster_sat[instrument_cluster]:.1f})")

joblib.dump({"kmeans": kmeans, "instrument_cluster": instrument_cluster},
            CKPT_DIR / "kmeans_model.pkl")
print("✅ Saved: checkpoints/kmeans_model.pkl")
```

---

### Cell 2.2 — Tip Extraction & Evaluate K-Means on Test Set
> **Why:** Tests K-Means baseline on the test set. Calculates median tip error in pixels.

```python
# Cell 2.2 — Evaluate K-Means
def detect_tip_kmeans(img_bgr, km, cluster_id):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    preds = km.predict(hsv).reshape(img_bgr.shape[:2])
    mask = ((preds == cluster_id) * 255).astype(np.uint8)
    
    # Clean noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        return None, None, 0.0
    
    # Largest component
    largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    ys, xs = np.where(labels == largest_idx)
    if len(xs) == 0:
        return None, None, 0.0
    
    # Tip = bottom-most point of the detected component (tools usually enter from top/side)
    bottom_idx = np.argmax(ys)
    return float(xs[bottom_idx]), float(ys[bottom_idx]), 0.6

test_imgs = sorted(list((YOLO_DIR / "images" / "test").glob("*")))[:150]
km_errors = []

for imp in test_imgs:
    im = cv2.imread(str(imp))
    lbl_file = YOLO_DIR / "labels" / "test" / f"{imp.stem}.txt"
    if not lbl_file.exists() or im is None: continue
    
    # Ground truth tip from label
    with open(lbl_file) as lf:
        lines = lf.readlines()
    if not lines: continue
    parts = list(map(float, lines[0].split()))
    _, xc, yc, bw, bh = parts
    h, w = im.shape[:2]
    gt_tip_x = xc * w
    gt_tip_y = (yc + bh / 2.0) * h  # bottom of bounding box
    
    u, v, conf = detect_tip_kmeans(im, kmeans, instrument_cluster)
    if u is not None:
        err = np.sqrt((u - gt_tip_x)**2 + (v - gt_tip_y)**2)
        km_errors.append(err)

km_results = {
    "median_tip_err_px": float(np.median(km_errors)) if km_errors else 999.0,
    "p95_tip_err_px":    float(np.percentile(km_errors, 95)) if km_errors else 999.0
}
print("K-Means Test Error:")
print(f"  Median tip error: {km_results['median_tip_err_px']:.1f} px")
print(f"  95th-pct error  : {km_results['p95_tip_err_px']:.1f} px")

with open(RESULTS_DIR / "kmeans_results.json", "w") as f:
    json.dump(km_results, f, indent=2)
```

---

## PART 3 — Supervised Baseline: Random Forest Pixel Classifier

### Cell 3.1 — Extract Features & Train Random Forest
> **Why:** Trains an ensemble of 100 trees on colour (HSV, Lab) and edge gradients to classify tool vs background pixels.

```python
# Cell 3.1 — Feature extraction & RF training
def extract_pixel_features(img_bgr, sample_size=300):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 255.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 255.0
    grad = np.sqrt(gx**2 + gy**2)
    
    feats = np.dstack([hsv, lab, grad.reshape(img_bgr.shape[0], img_bgr.shape[1], 1)])
    feats_flat = feats.reshape(-1, 7)
    return feats_flat

print("Sampling balanced pixel features for Random Forest...")
X_rf, y_rf = [], []

for imp in train_imgs[:150]:
    im = cv2.imread(str(imp))
    lbl_file = YOLO_DIR / "labels" / "train" / f"{imp.stem}.txt"
    if im is None or not lbl_file.exists(): continue
    
    h, w = im.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(lbl_file) as lf:
        for line in lf:
            _, xc, yc, bw, bh = map(float, line.split())
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
            
    feats = extract_pixel_features(im)
    labels = mask.flatten()
    
    tool_idx = np.where(labels == 1)[0]
    bg_idx   = np.where(labels == 0)[0]
    
    n_sample = min(150, len(tool_idx), len(bg_idx))
    if n_sample > 10:
        sel = np.concatenate([
            np.random.choice(tool_idx, n_sample, replace=False),
            np.random.choice(bg_idx, n_sample, replace=False)
        ])
        X_rf.append(feats[sel])
        y_rf.append(labels[sel])

X_rf = np.vstack(X_rf)
y_rf = np.concatenate(y_rf)
print(f"Total training pixels: {len(X_rf):,} (50% tool, 50% background)")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf_model.fit(X_rf, y_rf)
print("✅ Random Forest training finished!")

joblib.dump(rf_model, CKPT_DIR / "rf_model.pkl")
print("✅ Saved: checkpoints/rf_model.pkl")
```

---

### Cell 3.2 — Evaluate Random Forest on Test Set
> **Why:** Tests Random Forest tip accuracy on the test set.

```python
# Cell 3.2 — Evaluate Random Forest
rf_errors = []

for imp in test_imgs[:150]:
    im = cv2.imread(str(imp))
    lbl_file = YOLO_DIR / "labels" / "test" / f"{imp.stem}.txt"
    if not lbl_file.exists() or im is None: continue
    
    # Ground truth tip
    with open(lbl_file) as lf:
        lines = lf.readlines()
    if not lines: continue
    _, xc, yc, bw, bh = map(float, lines[0].split())
    h, w = im.shape[:2]
    gt_tip_x = xc * w
    gt_tip_y = (yc + bh / 2.0) * h
    
    # Predict on downscaled image for speed
    im_small = cv2.resize(im, (w // 4, h // 4))
    feats = extract_pixel_features(im_small)
    preds = rf_model.predict(feats).reshape(im_small.shape[:2])
    
    ys, xs = np.where(preds == 1)
    if len(xs) > 0:
        bottom_idx = np.argmax(ys)
        pred_x = float(xs[bottom_idx] * 4)
        pred_y = float(ys[bottom_idx] * 4)
        err = np.sqrt((pred_x - gt_tip_x)**2 + (pred_y - gt_tip_y)**2)
        rf_errors.append(err)

rf_results = {
    "median_tip_err_px": float(np.median(rf_errors)) if rf_errors else 999.0,
    "p95_tip_err_px":    float(np.percentile(rf_errors, 95)) if rf_errors else 999.0
}
print("Random Forest Test Error:")
print(f"  Median tip error: {rf_results['median_tip_err_px']:.1f} px")
print(f"  95th-pct error  : {rf_results['p95_tip_err_px']:.1f} px")

with open(RESULTS_DIR / "rf_results.json", "w") as f:
    json.dump(rf_results, f, indent=2)
```

---

## PART 4 — Deep Learning: YOLOv8-nano Fine-Tuning

### Cell 4.1 — Train YOLOv8-nano
> **Why:** Fine-tunes pretrained YOLOv8-nano on surgical tools. Checkpoints are automatically saved, and early stopping prevents overfitting. Runs in ~20–35 minutes on T4 GPU.

```python
# Cell 4.1 — Train YOLOv8-nano (Fixed: project name + PyTorch 2.6)
import os, shutil
from pathlib import Path
import torch

# 1. PyTorch 2.6 fix — delegate directly to raw torch.serialization.load (immune to recursion)
import torch.serialization
_real_load = torch.serialization.load
def safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_load(*args, **kwargs)
torch.load = safe_torch_load

# 2. Neutralize third-party callbacks (Ray Tune, WandB, MLflow)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Provide dummy _get_session to satisfy Ray Tune's broken on_fit_epoch_end check
try:
    import ray.train._internal.session as ray_session
    ray_session._get_session = lambda: None
except Exception:
    pass

import ultralytics.engine.trainer as trainer_module
import ultralytics.utils.callbacks as cb_module
trainer_module.callbacks.add_integration_callbacks = lambda instance: None
cb_module.add_integration_callbacks = lambda instance: None

from ultralytics import YOLO

print("🚀 Starting YOLOv8-nano training...")
model = YOLO("yolov8n.pt")  # load pretrained weights

results = model.train(
    data        = str(YOLO_DIR / "dataset.yaml"),
    epochs      = 50,
    imgsz       = 640,
    batch       = 16,
    lr0         = 0.001,
    patience    = 15,          # early stopping
    save_period = 10,          # checkpoint every 10 epochs
    project     = "yolo_runs", # simple name without path slashes
    name        = "m2cai16_detector",
    exist_ok    = True,
    device      = 0 if torch.cuda.is_available() else "cpu",
    verbose     = True,
    plots       = True
)

# 3. Copy best weights to root for easy download and Cell 4.2 evaluation
target_pt = ROOT / "BEST_yolov8n_m2cai16.pt"
found_best = None

# Check the results directory first, then fallback locations
if hasattr(results, 'save_dir') and (Path(results.save_dir) / "weights" / "best.pt").exists():
    found_best = Path(results.save_dir) / "weights" / "best.pt"
elif Path("yolo_runs/m2cai16_detector/weights/best.pt").exists():
    found_best = Path("yolo_runs/m2cai16_detector/weights/best.pt")
elif (CKPT_DIR / "yolo_runs" / "m2cai16_detector" / "weights" / "best.pt").exists():
    found_best = CKPT_DIR / "yolo_runs" / "m2cai16_detector" / "weights" / "best.pt"

if found_best and found_best.exists():
    shutil.copy(str(found_best), str(target_pt))
    print(f"\n✅ Training Complete! Model exported to: {target_pt}")
else:
    # Search working directory for any saved best.pt
    candidates = list(Path(".").rglob("best.pt"))
    if candidates:
        shutil.copy(str(candidates[0]), str(target_pt))
        print(f"\n✅ Found best.pt at {candidates[0]}, copied to: {target_pt}")
    else:
        print("\n⚠️ Note: best.pt will appear once training completes.")
```

---

### Cell 4.2 — Evaluate YOLOv8-nano on Test Set
> **Why:** Computes standard mAP@50 and calculates tip-tracking accuracy to compare against K-Means and Random Forest.

```python
# Cell 4.2 — Evaluate YOLOv8
best_model = YOLO(str(target_pt))

# Run validation on test split
metrics = best_model.val(data=str(YOLO_DIR / "dataset.yaml"), split="test")

yolo_errors = []
for imp in test_imgs:
    im = cv2.imread(str(imp))
    lbl_file = YOLO_DIR / "labels" / "test" / f"{imp.stem}.txt"
    if im is None or not lbl_file.exists(): continue
    
    with open(lbl_file) as lf:
        lines = lf.readlines()
    if not lines: continue
    _, xc, yc, bw, bh = map(float, lines[0].split())
    h, w = im.shape[:2]
    gt_tip_x = xc * w
    gt_tip_y = (yc + bh / 2.0) * h
    
    preds = best_model(im, verbose=False)[0]
    if len(preds.boxes) > 0:
        box = preds.boxes.xyxy.cpu().numpy()[0]
        pred_tip_x = (box[0] + box[2]) / 2.0
        pred_tip_y = box[3]  # bottom center of predicted tool box
        err = np.sqrt((pred_tip_x - gt_tip_x)**2 + (pred_tip_y - gt_tip_y)**2)
        yolo_errors.append(err)

yolo_results = {
    "mAP50":             float(metrics.box.map50),
    "mAP50_95":          float(metrics.box.map),
    "median_tip_err_px": float(np.median(yolo_errors)) if yolo_errors else 999.0,
    "p95_tip_err_px":    float(np.percentile(yolo_errors, 95)) if yolo_errors else 999.0
}

print("\nYOLOv8 Test Results:")
print(f"  mAP@50          : {yolo_results['mAP50']:.3f}")
print(f"  Median tip error: {yolo_results['median_tip_err_px']:.1f} px")
print(f"  95th-pct error  : {yolo_results['p95_tip_err_px']:.1f} px")

with open(RESULTS_DIR / "yolo_results.json", "w") as f:
    json.dump(yolo_results, f, indent=2)
```

---

## PART 5 — Lost-Tool Classifier (FSM Helper)

### Cell 5.1 — Train Decision Tree Classifier
> **Why:** A small, transparent decision tree that predicts tracking status: `TRACKING` (0), `UNCERTAIN` (1), or `LOST` (2). Member B's Finite State Machine uses this to trigger zoom-out recovery.

```python
# Cell 5.1 — Lost-tool classifier
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib

lt_X, lt_y = [], []

for imp in test_imgs:
    im = cv2.imread(str(imp))
    lbl_file = YOLO_DIR / "labels" / "test" / f"{imp.stem}.txt"
    if im is None or not lbl_file.exists(): continue
    
    with open(lbl_file) as lf:
        lines = lf.readlines()
    if not lines: continue
    _, xc, yc, bw, bh = map(float, lines[0].split())
    h, w = im.shape[:2]
    gt_tip_x = xc * w
    gt_tip_y = (yc + bh / 2.0) * h
    
    preds = best_model(im, verbose=False)[0]
    if len(preds.boxes) == 0:
        # Tool completely missed
        features = [0.0, 0.0, 0.5, 0.5]  # [conf, box_area, norm_x, norm_y]
        label = 2  # LOST
    else:
        confs = preds.boxes.conf.cpu().numpy()
        boxes = preds.boxes.xyxy.cpu().numpy()
        best_idx = np.argmax(confs)
        b = boxes[best_idx]
        
        box_area = float((b[2] - b[0]) * (b[3] - b[1]) / (w * h))
        pred_x = float((b[0] + b[2]) / (2 * w))
        pred_y = float(b[3] / h)
        conf_val = float(confs[best_idx])
        
        pred_tip_px = ((b[0] + b[2]) / 2.0, b[3])
        err = np.sqrt((pred_tip_px[0] - gt_tip_x)**2 + (pred_tip_px[1] - gt_tip_y)**2)
        
        features = [conf_val, box_area, pred_x, pred_y]
        if err > 75.0 or conf_val < 0.35:
            label = 2  # LOST
        elif err > 35.0 or conf_val < 0.60:
            label = 1  # UNCERTAIN
        else:
            label = 0  # TRACKING
            
    lt_X.append(features)
    lt_y.append(label)

lt_X = np.array(lt_X)
lt_y = np.array(lt_y)

dt = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
dt.fit(lt_X, lt_y)
print("✅ Lost-tool Decision Tree trained!")
print("\nDecision Tree Rules:")
print(export_text(dt, feature_names=["conf", "box_area", "norm_x", "norm_y"]))

joblib.dump(dt, CKPT_DIR / "lost_tool_classifier.pkl")
print("✅ Saved: checkpoints/lost_tool_classifier.pkl")
```

---

## PART 6 — Model Comparison Table & Visual Grid

### Cell 6.1 — Comparison Table
> **Why:** The main table for your project report showing the progression from baseline to deep learning.

```python
# Cell 6.1 — Print final comparison table
import json

# Fallback to saved json files if variables are not in memory
if 'km_results' not in globals() and (RESULTS_DIR / 'kmeans_results.json').exists():
    with open(RESULTS_DIR / 'kmeans_results.json') as f: km_results = json.load(f)
if 'rf_results' not in globals() and (RESULTS_DIR / 'rf_results.json').exists():
    with open(RESULTS_DIR / 'rf_results.json') as f: rf_results = json.load(f)
if 'yolo_results' not in globals() and (RESULTS_DIR / 'yolo_results.json').exists():
    with open(RESULTS_DIR / 'yolo_results.json') as f: yolo_results = json.load(f)

print("\n" + "=" * 65)
print(f"{'Detector':<20} {'Median Tip Error (px)':>25} {'95th-pct Error':>15}")
print("=" * 65)
print(f"{'K-Means':<20} {km_results.get('median_tip_err_px', 0):>25.1f} {km_results.get('p95_tip_err_px', 0):>15.1f}")
print(f"{'Random Forest':<20} {rf_results.get('median_tip_err_px', 0):>25.1f} {rf_results.get('p95_tip_err_px', 0):>15.1f}")
print(f"{'YOLOv8-nano':<20} {yolo_results.get('median_tip_err_px', 0):>25.1f} {yolo_results.get('p95_tip_err_px', 0):>15.1f}")
print("=" * 65)
print(f"YOLOv8-nano mAP@50: {yolo_results.get('mAP50', 0):.3f}")
```

---

### Cell 6.2 — 4-Panel Visual Demonstration
> **Why:** Visual comparison screenshot for slides.

```python
# Cell 6.2 — Visual comparison grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
test_samples = random.sample(test_imgs, 4)

for idx, imp in enumerate(test_samples):
    ax = axes[idx // 2, idx % 2]
    im = cv2.imread(str(imp))
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    preds = best_model(im, verbose=False)[0]
    for box in preds.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tip_x, tip_y = int((x1 + x2) / 2), int(y2)
        cv2.circle(im_rgb, (tip_x, tip_y), 8, (255, 0, 0), -1)
        cv2.putText(im_rgb, "Tool Tip", (tip_x - 30, tip_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    ax.imshow(im_rgb)
    ax.set_title(f"{imp.name} - YOLOv8 Detections")
    ax.axis("off")

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "06_yolo_detections.png"), dpi=150)
plt.show()
print("✅ Saved comparison grid to results/06_yolo_detections.png")
```

---

## PART 7 — Package Weights & Handoff to Team

### Cell 7.1 — Package into ZIP for Download
> **Why:** Bundles your trained models into `member_a_outputs.zip`. Download this file (File menu -> Download Output) to run `demo.py` on your computer!

```python
# Cell 7.1 — Create zip package
import zipfile

zip_path = ROOT / "member_a_outputs.zip"
with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
    for model_file in [
        ROOT / "BEST_yolov8n_m2cai16.pt",
        CKPT_DIR / "rf_model.pkl",
        CKPT_DIR / "kmeans_model.pkl",
        CKPT_DIR / "lost_tool_classifier.pkl"
    ]:
        if model_file.exists():
            zf.write(str(model_file), model_file.name)
            
    for res_file in RESULTS_DIR.glob("*"):
        zf.write(str(res_file), f"results/{res_file.name}")

size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"🎉 All Done! Download package created: {zip_path.name} ({size_mb:.1f} MB)")
print("👉 Go to the Kaggle File menu (or right panel Output section) and click Download on 'member_a_outputs.zip'!")
```

---

### Cell 7.2 — Python code for Member C's `detector.py`
> **Why:** Code to paste into your local repo's `src/detector.py`.

```python
# Cell 7.2 — Print interface
print('''
# Paste this into src/detector.py in your project folder:

from pathlib import Path
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path("models/BEST_yolov8n_m2cai16.pt")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model

def detect_tip(frame_bgr: np.ndarray) -> tuple:
    """
    Detects surgical tool tip in viewport frame.
    Returns: (u, v, confidence)
    """
    model = get_model()
    results = model(frame_bgr, verbose=False)[0]
    if len(results.boxes) == 0:
        return None, None, 0.0
    
    confs = results.boxes.conf.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    best_idx = np.argmax(confs)
    box = boxes[best_idx]
    
    tip_u = float((box[0] + box[2]) / 2.0)
    tip_v = float(box[3])  # bottom center of detected tool
    conf  = float(confs[best_idx])
    return tip_u, tip_v, conf
''')
```
