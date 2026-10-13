# Autonomous Endoscopic Camera Assistant (Robotic Vision & ML Cockpit)
## Comprehensive Technical Documentation & Project Inventory

---

## 1. Executive Summary & Clinical Motivation

In minimally invasive abdominal surgery (laparoscopy), an assistant surgeon is tasked with holding and aiming the endoscope camera for hours. This introduces critical clinical bottlenecks:
- **Assistant Fatigue & Hand Tremor:** Subtle physiological hand vibrations cause erratic camera movement and optical blur.
- **Miscommunication & Lag:** Verbal commands ("pan left", "zoom in") cause a 1.5–3.0 second cognitive lag between the chief surgeon's intent and camera movement.
- **Visual Disorientation & Instrument Loss:** High-speed surgical movements, electrocautery smoke, specular lens glare, and instrument retraction cause the camera to lose track of the surgical site.

### The Solution
This project implements an **Autonomous Endoscopic Camera Assistant (Idea 13)** that continuously tracks surgical instruments, maintains optimal framing, damps physiological tremors, and autonomously searches for lost tools. 

Crucially, **this is not just a fine-tuned neural network**. It combines a **5-tier Machine Learning and Estimation Stack**:
1. **Deep Learning Perception:** Fine-tuned YOLOv8-nano for real-time instrument localization.
2. **Classical Perception Baselines:** Unsupervised K-Means and Random Forest for quantitative empirical comparison.
3. **Machine Learning State Machine:** A Decision Tree classifier for tracking confidence and failure recovery.
4. **Bayesian Filtering:** Multi-instance 2D Kalman Filters providing occlusion dead-reckoning.
5. **Online Adaptive Regressor:** Dynamic PID gain scheduling for physiological tremor damping.

---

## 2. Workspace Inventory: What is in this Project Right Now

The workspace contains trained model weights, evaluation datasets, video streams, snapshot artifacts, and an interactive simulation cockpit:

```
c:\Users\sudee\Desktop\ML Project\
│
├── 🧠 TRAINED ML MODELS & WEIGHTS
│   ├── member_a_outputs/
│   │   ├── BEST_yolov8n_m2cai16.pt       # Fine-tuned YOLOv8-nano weights (6.2 MB, 95.9% mAP@50)
│   │   ├── lost_tool_classifier.pkl     # Trained Decision Tree for state recovery (conf, area, pos)
│   │   ├── kmeans_model.pkl              # Unsupervised baseline clustering model
│   │   ├── rf_model.pkl                  # 100-Tree Random Forest baseline (40.7 MB)
│   │   └── results/                      # JSON validation metrics & ground truth evaluation plots
│   ├── best.pt / last.pt                 # Training checkpoint mirrors
│   └── yolov8n.pt                        # Base pretrained COCO weights
│
├── 🎮 UNIVERSAL SIMULATION ENGINE
│   ├── demo.py                           # Real-time dual-view simulator & ML telemetry cockpit
│   ├── sample_surgery.mp4                # 148-frame sequential surgical video stream (596x334 @ 25 FPS)
│   └── demo_data/frames/                 # Sequential JPEG surgical frames (v01_052975 to v01_057675)
│
├── 🎬 RECORDED DEMO VIDEOS & EVIDENCE
│   ├── stress_test_demo.mp4              # 8.5 MB recorded showcase of all 4 adverse stress stages
│   └── demo_output.mp4                   # 25.4 MB baseline multi-tool tracking session
│
├── 📸 HIGH-RESOLUTION EVALUATION ARTIFACTS (results/)
│   ├── stage1_baseline.png               # Baseline nominal multi-tool tracking
│   ├── stage2_smoke_kalman.png           # Electrocautery smoke + Kalman dead-reckoning coasting
│   ├── stage3_glare_adaptive.png         # Specular tissue glare + Adaptive tremor filter damping
│   ├── stage4a_zoom_out_search.png       # Tool leaves field -> Decision Tree LOST -> Wide FOV search
│   └── stage4b_reacquired_focus.png      # Tool re-enters -> Focus Lock reticle & tight zoom-in
│
└── 📄 DESIGN PLANS & EXPERIMENT LOGS
    ├── PROJECT_OVERVIEW.md               # [This File] Complete project reference documentation
    ├── endoscope_auto_camera_plan.md     # Initial 3-member team architectural plan
    ├── endoscope_auto_camera_plan_v2_integrated_ml.md # Advanced ML integration specification
    ├── member_a_kaggle_notebook.md       # Reproducible Kaggle training script for perception ladder
    └── m2cai16-tool-locations.zip        # Stanford m2cai16 surgical tool benchmark dataset (145 MB)
```

---

## 3. The 5-Tier Machine Learning Architecture

When evaluators ask: *"Where did you use Machine Learning other than fine-tuning YOLO?"*, this architecture provides the answer:

```
                       RAW LAPAROSCOPIC VIDEO FRAME
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
[1. DEEP PERCEPTION]                              [PERCEPTION BASELINES]
YOLOv8-nano (Fine-Tuned)                          • K-Means (Unsupervised)
• 95.9% mAP@50                                    • Random Forest (100 Trees)
• 11.0 px Median Tip Error                        (Proven inferior on test set)
• 2.5 ms GPU Inference
           │
           │ (Bounding boxes, confidence, tip coordinates)
           ▼
[2. BAYESIAN KALMAN FILTERING] (ML-3: SurgicalKalmanFilter)
• Multi-instance 4-state constant velocity model: [x, y, vx, vy]
• Eliminates high-frequency surgical jitter
• Dead-Reckoning Coasting: Predicts trajectory when tool is occluded by smoke
           │
           │ (Smoothed trajectory, tool velocities, active spread)
           ▼
[3. MULTI-TOOL COORDINATOR]
• CENTROID: Encompassing zoom framing for all active instruments
• MOTION_PRIORITY: Focuses on actively cutting/moving instrument over retractors
• SINGLE_PRIMARY: Locks strictly onto highest-confidence instrument
           │
           │ (Target position, bounding box area, normalized coordinates)
           ▼
[4. STATE MACHINE CLASSIFIER] (ML-2: DecisionTreeClassifier)
• Input features: [conf, box_area, norm_x, norm_y]
• Classifies operational state:
    - Label 0: TRACKING (Nominal surgical guidance)
    - Label 1: UNCERTAIN (Borderline occlusion)
    - Label 2: LOST / OUT OF FIELD (Triggers recovery protocol)
           │
           ▼
[5. ONLINE ADAPTIVE REGRESSOR] (ML-4: AdaptiveMotionEstimator)
• Rolling 15-frame velocity distribution analysis
• Classifies motion regime: FAST TRACKING vs. TREMOR DAMPING vs. STABLE
• Dynamically tunes virtual camera PID gains and deadband
           │
           ▼
[6. VIRTUAL CAMERA VIEWPORT]
• Smooth exponential zoom & PID panning
• Search & Auto-Focus Recovery Cycle
```

---

## 4. Perception Ladder Benchmark (Quantitative Proof)

We evaluated three machine learning paradigms on the Stanford **m2cai16-tool-locations** dataset (cholecystectomy procedures):

| Model | Architecture / Technique | Median Tip Error | 95th Percentile Error | Inference Speed | Clinical Feasibility |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **K-Means Baseline** | Unsupervised spatial clustering on high-contrast pixels | **278.9 px** | 595.8 px | 8.2 ms | ❌ **Failed:** Confuses liver/blood tissue with metallic tools |
| **Random Forest Baseline** | 100 Trees on 7 handcrafted features (LAB color, Sobel edges, texture) | **379.9 px** | 522.6 px | 34.0 ms | ❌ **Failed:** High variance, cannot generalize across surgical lighting |
| **Fine-Tuned YOLOv8-nano** | Deep CNN with Feature Pyramid Network & anchor-free tip head | **11.0 px** | **228.2 px** | **2.5 ms** | ✅ **Optimal:** 95.9% mAP@50, sub-centimeter tip precision, 400 FPS capable |

### Key Insight for Project Viva
Classical pixel-level ML fails because endoscopic lighting is highly non-uniform and tissue is moist and reflective. Deep hierarchical representations are mandatory for robust tool detection, while **classical ML is optimally utilized at the decision, state-machine, and filtering layers**.

---

## 5. Universal Simulation Cockpit (`demo.py`)

The simulator provides a split-screen surgical workstation:
- **Left Panel (Global Laparoscopic View):** Displays the raw 596×334 wide-field laparoscope view, bounding boxes, tool trajectory trails, multi-tool target crosses, and the moving virtual camera viewport.
- **Right Panel (Autonomous Surgeon's View):** Shows the stabilized, centered, and zoomed surgical field that the surgeon would see on their surgical monitor, complete with reticle crosshairs.
- **Top Telemetry HUD:** Displays real-time operational state (`TRACKING`, `SEARCH (ZOOM OUT)`, `RE-ACQUIRED (FOCUSING)`), centering error in pixels, multi-tool policy, current zoom magnification, playback speed, active stress mode, and FPS.
- **Bottom Live ML Methods Inspector:** A real-time data visualizer showing exact numerical inputs and outputs for **YOLOv8**, the **Decision Tree**, the **Kalman Filter**, and the **Adaptive Regressor** at every frame.

### Multi-Tool Coordination Policies
Pressing `m` cycles through three surgical framing policies:
1. **`CENTROID` (Default):** Calculates the geometric center of all instruments and automatically expands zoom if instruments spread far apart.
2. **`MOTION_PRIORITY`:** Computes the instantaneous velocity of each instrument via Kalman state vectors. Weights camera focus toward the active cutting instrument rather than stationary retractors.
3. **`SINGLE_PRIMARY`:** Locks focus onto the instrument with the highest detection confidence.

---

## 6. Feature 1: Adverse Conditions Stress-Test & Search-and-Focus Mode

In real surgery, cameras fail due to visual artifacts. Feature 1 stress-tests the system under these conditions and proves how ML prevents failure:

### A. Electrocautery Smoke Occlusion (`STAGE 2/4`)
- **Clinical Event:** Burning tissue releases thick white/grey smoke plumes, occluding the laparoscope lens.
- **Visual Impact:** YOLO confidence drops or fails to detect tools.
- **ML Safeguard:** The **Surgical Kalman Filter** automatically enters **Bayesian Dead-Reckoning Coasting** (`predict()` without measurement update). It holds the instrument's last valid trajectory smoothly through the smoke plume, preventing camera disorientation.

### B. Specular Tissue Glare (`STAGE 3/4`)
- **Clinical Event:** High-intensity cold laparoscope lighting reflects off moist liver tissue, overexposing pixels.
- **Visual Impact:** Rapid false-positive bounding box jitter.
- **ML Safeguard:** The **Adaptive Motion Regressor** detects high-frequency velocity variance, automatically **widening the deadband (`13px`)** and reducing proportional gain (`Kp = 0.040`) to suppress camera shaking.

### C. The Search-and-Focus Recovery Cycle (`STAGE 4A & 4B`)
- **Stage 4A (Tool Out of Field):** When the surgeon withdraws the instrument from the trocar, visual detections drop to 0. The **Decision Tree** evaluates `[conf=0, area=0]` and flags `LOST`. The camera automatically executes **exponential zoom-out** to `0.98` crop (`1.0x` wide-angle) and centers on the cavity, maximizing field-of-view to spot incoming tools.
- **Stage 4B (Tool Re-acquisition & Auto-Focus):** The moment the tool re-enters the wide-angle view, YOLO detects it (`conf >= 50%`). The Decision Tree immediately triggers `RE-ACQUIRED (FOCUSING)`. The camera rapidly re-centers (`Kp = 0.20`), **zooms in tight to `1.85x` magnification**, and crosshairs glow bright emerald green with `TARGET ACQUIRED: FOCUS LOCKED`.

---

## 7. How to Run the Project

### 1. Live Interactive Cockpit
Run on any video, image sequence, or webcam:
```powershell
# Run with default sequential surgical video at 0.5x clear slow-motion
python demo.py --source sample_surgery.mp4 --speed 0.5

# Run on live webcam
python demo.py --source 0

# Run on image folder
python demo.py --source demo_data/frames
```

### 2. Automated 4-Phase Stress-Test Showcase
Runs the automated scenario through all 4 stages:
```powershell
python demo.py --source sample_surgery.mp4 --stress auto --speed 0.5
```

### 3. Headless Video Generation
Generates video files without requiring HighGUI windows:
```powershell
python demo.py --stress auto --headless --output stress_test_demo.mp4 --max-frames 148
```

### Keyboard Controls Reference
| Key | Function | Demonstration Purpose |
| :---: | :--- | :--- |
| `SPACE` | **Pause / Resume** | Freeze frame to inspect the live ML telemetry inspector. |
| `o` | **Toggle Electrocautery Smoke** | Demonstrates **Kalman dead-reckoning** during occlusion. |
| `g` | **Toggle Specular Glare** | Demonstrates **Adaptive tremor damping** under reflection. |
| `x` | **Toggle Tool Out of Field** | Demonstrates **Zoom-out cavity search** & **auto-focus lock**. |
| `m` | **Cycle Multi-Tool Policy** | Switches between `CENTROID`, `MOTION_PRIORITY`, and `SINGLE_PRIMARY`. |
| `s` / `f` | **Slower / Faster Playback** | Slows motion down to `0.15x` for fine inspection. |
| `+` / `-` | **Manual Zoom In / Out** | Manually adjust field-of-view magnification. |
| `r` | **Reset Viewport** | Returns camera to optical center. |
| `q` / `ESC`| **Quit & Save Video** | Cleanly exits and finalizes recorded MP4. |

---

## 8. Viva & Evaluation Oral Defense Cheat Sheet

### Q1: "Is this project just fine-tuning a YOLO model?"
> **Answer:** No. Fine-tuning YOLO is only Tier 1 of a 5-tier system. While YOLO localizes the tool tip, YOLO alone cannot hold a camera: raw detections produce severe high-frequency jitter, fail when obscured by smoke, and cannot decide when a tool has left the field. We integrated:
> 1. A **Decision Tree State Machine** trained to classify operational state (`TRACKING`, `UNCERTAIN`, `LOST`).
> 2. **Multi-Instance 2D Kalman Filters** providing Bayesian dead-reckoning coasting during occlusion.
> 3. An **Online Adaptive Motion Regressor** dynamically damping physiological hand tremor.
> 4. Multi-tool coordination policies (`CENTROID` and `MOTION_PRIORITY`).

### Q2: "Why didn't K-Means or Random Forest work for perception?"
> **Answer:** In laparoscopy, tissue is non-rigid, moist, and covered in blood, bile, and specular reflections. K-Means failed (median tip error 278.9 px) because metallic instruments and specular highlights share high-intensity pixel clusters. Random Forest with handcrafted color/edge features failed (median error 379.9 px) because lighting changes drastically during camera movement. Deep CNN features (YOLOv8-nano) achieved 11.0 px error and 95.9% mAP@50 by learning invariant hierarchical representations.

### Q3: "What happens when the tool goes out of view?"
> **Answer:** Detections drop to 0. Our Decision Tree evaluates `[conf=0, box_area=0]` and transitions to `LOST`. The virtual camera immediately initiates an autonomous search routine: it smoothly expands its zoom to full wide-angle (`1.0x`) and re-centers on the abdominal cavity. As soon as the instrument re-enters from any trocar port, YOLO detects it in the wide field, the Decision Tree transitions to `RE-ACQUIRED (FOCUSING)`, and the camera snaps back to center and zooms in tight (`1.85x`) to lock focus on the tool.
