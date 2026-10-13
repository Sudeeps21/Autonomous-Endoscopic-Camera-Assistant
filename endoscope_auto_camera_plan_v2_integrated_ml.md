# Endoscope Auto-Camera v2: A Learned Perception–Prediction–Control Stack

**Change from v1:** every ML component now sits **inside** the control loop and changes what the camera does. Nothing is there only to be compared against a baseline. The system has six learned modules, and each one consumes the output of another.

**Team:** 3 members, working in parallel | **Timeline:** 3 days

---

## 1. System Overview

A robotic holder keeps a laparoscope pointed at the active instrument tip. The scope is gripped outside the body and pivots about the trocar (remote centre of motion): pan, tilt and zoom (insertion depth). Nothing is attached to the instrument; everything comes from the image.

**Evaluation setting (to avoid the sim-to-real gap):** a *virtual viewport* over real laparoscopic video. The "camera" is a crop window moved and resized by the controller. The full video frame acts as the fixed world, so tip positions in **full-frame coordinates** are independent of where the crop window is. A PyBullet arm with an RCM constraint, driven by the same command topic, demonstrates the kinematics and the ROS 2 loop.

---

## 2. Architecture

```
                         ┌───────────────────────────────────────────────┐
  Video frame ─────────► │ M1  Tip detector (fine-tuned YOLO / heatmap)  │
  (crop window)          │     outputs: tip (u,v), confidence, bbox      │
                         └───────────────┬───────────────────────────────┘
                                         │ features
                         ┌───────────────▼───────────────────────────────┐
                         │ M2  Learned noise model (gradient boosting)   │
                         │     predicts expected tip error → R_t         │
                         └───────────────┬───────────────────────────────┘
                                         │ (u,v), R_t
                         ┌───────────────▼───────────────────────────────┐
                         │ Kalman filter with learned, per-frame R_t     │
                         │ outputs: smoothed tip, velocity, innovation   │
                         └──────┬─────────────────────┬──────────────────┘
                                │ track history       │ track window
                 ┌──────────────▼───────┐   ┌─────────▼─────────────────┐
                 │ M3  Motion forecaster │   │ M5  Motion-mode clustering │
                 │  (GRU/TCN) → next 0.5 s│  │  (k-means / GMM)           │
                 │  tip positions         │  │  → mode: fine / transit    │
                 └──────┬────────────────┘   └─────────┬─────────────────┘
                        │ forecast                     │ mode
                 ┌──────▼───────────────┐              │
                 │ M4  Lost-tool predictor│            │
                 │  (random forest / GBM) │            │
                 │  P(lost in next 0.5 s) │            │
                 └──────┬────────────────┘             │
                        │ p_lost                       │
                 ┌──────▼──────────────────────────────▼───────────────┐
                 │ M6  Learned controller (MLP, imitation of MPC expert)│
                 │ inputs: error, velocity, forecast, σ, mode, p_lost   │
                 │ output: pan/tilt/zoom acceleration                   │
                 └──────┬───────────────────────────────────────────────┘
                        │
                 ┌──────▼───────────────────────────────────────────────┐
                 │ Safety shield: jerk/velocity limits, PID fallback,    │
                 │ recovery FSM (TRACKING→COASTING→SEARCH→HOLD)          │
                 └──────┬───────────────────────────────────────────────┘
                        ▼
              Camera arm / virtual viewport  ──► new frame ─► M1
```

### 2.1 What each learned module does, and who uses its output

| Module | ML method | Training data | What it outputs | Who uses it |
|---|---|---|---|---|
| **M1 Tip detector** | Fine-tuned YOLOv8-nano (or small U-Net/heatmap head for the tip); augmentation for glare, smoke, blur; **self-training** on unlabelled Cholec80 frames (pseudo-labels with confidence filtering, then retrain) | m2cai16-tool-locations, CholecSeg8k, pseudo-labelled video | Tip pixel, confidence, bbox | M2, Kalman filter |
| **M2 Learned noise model** | Gradient-boosted regressor predicting expected tip error (px) from confidence, bbox size, blur score, tip-to-border distance, tool count | Detector outputs vs ground truth on validation videos | Per-frame measurement noise `R_t` | Kalman filter (it trusts good detections more and bad ones less) |
| **M3 Motion forecaster** | GRU or temporal CNN: last ~0.6 s of tip track → next ~0.5 s of tip positions (Huber loss, offsets relative to current position) | Tip tracks extracted from Cholec80 video by M1, plus synthetic minimum-jerk trajectories | Forecast trajectory with spread | M4, M6 (camera moves toward where the tool *will be*) |
| **M4 Lost-tool predictor** | Random forest or gradient boosting classifier (class-weighted); optionally a single decision tree for an interpretable plot | Windows from the same tracks, labelled "lost within the next 0.5 s" | `p_lost` | M6 (starts zooming out *before* the tool leaves), FSM |
| **M5 Motion-mode clustering** | K-means (k chosen by silhouette) or Gaussian mixture on window features: speed mean/variance, acceleration RMS, path tortuosity, bbox size | Same tracks | Mode label (e.g. fine dissection, transit, idle) | M6 (mode is an input; also selects the PID fallback gains) |
| **M6 Learned controller** | MLP trained by **imitation learning** from an offline MPC "expert" that sees the true future; 1 to 2 rounds of DAgger in simulation so it learns to recover from its own mistakes | Simulated closed-loop rollouts with noisy observations | Pan/tilt/zoom acceleration | Safety shield, then the arm |

This is a chain, not a list. The detector's uncertainty (M2) changes the filter; the filter's track trains the forecaster (M3), the clusterer (M5) and the lost-tool predictor (M4); and all three feed the controller (M6), which is itself learned. Removing any one of them changes the camera's behaviour.

### 2.2 Why these choices

- **M2 (learned noise):** a fixed Kalman noise value is wrong whenever the image is bad. Making it data-driven is a clean, cheap way to propagate detector uncertainty into control.
- **M3 and M4 (prediction):** a camera that reacts to the current error always lags. Forecasting lets it lead, and `p_lost` lets it act before a loss happens, which is where "lost tool" behaviour normally fails.
- **M5 (clustering):** a single gain set cannot be right both for slow fine dissection (needs a calm, wide dead-band) and for fast transit (needs a quick follow). Unsupervised modes give a free switching signal without hand labels.
- **M6 (imitation, not RL):** the expert uses privileged information (true future tip) and an optimisation solver that is too slow for the loop. The student is a small MLP that runs in microseconds on noisy observations. This is a real learned controller without the instability and tuning cost of RL.

### 2.3 ROS 2 interface contract (freeze on Day 1)

| Topic | Type | Rate | Producer → Consumer |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 25–30 Hz | Viewport → M1 |
| `/tool/tip` | `geometry_msgs/PointStamped` (full-frame px) | 25–30 Hz | M1 → filter |
| `/tool/meas_noise` | `std_msgs/Float32` (R_t, px²) | 25–30 Hz | M2 → filter |
| `/tool/track` | custom msg: filtered pos, vel, innovation | 25–30 Hz | Filter → M3/M4/M5/M6 |
| `/tool/forecast` | `std_msgs/Float32MultiArray` (K future points + spread) | 25–30 Hz | M3 → M4/M6 |
| `/tool/p_lost` | `std_msgs/Float32` | 25–30 Hz | M4 → M6/FSM |
| `/tracker/mode` | `std_msgs/Int8` | 1–5 Hz | M5 → M6 |
| `/camera/cmd_acc` | `geometry_msgs/Accel` (pan, tilt, zoom) | 25–30 Hz | Shield → viewport/arm |
| `/tracker/state` | `std_msgs/String` | on change | FSM → all |

---

## 3. Critical Analysis

### 3.1 Strengths

- Six learned modules covering supervised, unsupervised, semi-supervised and imitation learning, all wired into one loop.
- A clear story: perception uncertainty → state estimate → prediction → learned decision.
- Anticipatory behaviour (`p_lost`, forecasting) is more interesting than purely reactive tracking.
- Real surgical video is used for evaluation, avoiding a fake-looking simulator.

### 3.2 Weaknesses and honest risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Training data for M3/M4/M5 comes from M1's own detections** | Errors in the detector are baked into the "ground truth" tracks (circularity) | Keep only high-confidence segments for training, validate against the human-labelled frames of m2cai16, and state this limitation explicitly |
| **Little labelled tip data** | Small datasets overfit | Transfer learning, augmentation, split by video, pseudo-label self-training with a confidence threshold |
| **Camera-motion confound** | Source videos are already recorded from a moving camera | Use full-frame coordinates and treat the source camera as slowly varying; state the assumption |
| **Forecaster may not beat constant-velocity Kalman** | Surgical motion is irregular, data is limited | Train on real tracks plus synthetic trajectories; if gains are small, the controller still works because the forecast is only one input (M6 sees uncertainty). Report the result either way |
| **Imitation student cannot exceed its expert** | Upper bound is the MPC expert | Value comes from speed, and robustness to noisy observations. DAgger reduces compounding error |
| **Learned controller safety** | An unbounded network driving an arm is unacceptable | Safety shield: hard velocity/acceleration/jerk limits, workspace and RCM limits, automatic PID fallback if outputs saturate or state is HOLD |
| **Rare event labels (lost tool)** | Severe class imbalance | Class weights, PR-AUC instead of accuracy, tune the threshold for early warning |
| **Latency** | Chain of 4 modules in 33 ms | Small models, batch-free inference, ONNX export, measure per-module latency, run M2–M5 on CPU (they are tiny) |
| **Integration complexity** | Six modules plus ROS 2 in 3 days | Priority tiers (Section 4), mocks per member, and a PID path that always works |
| **Validation limits** | Replayed video does not react to camera motion; no surgeon; no clinical validity | State clearly: research prototype, not clinical |

### 3.3 Risk score

- **Technical risk: 6/10** (up from 4/10 in v1 because of more modules).
- Cut to ~5/10 by the tiering below, which guarantees a working system at the end of Day 2.

---

## 4. Priority Tiers (What Is Guaranteed vs Stretch)

| Tier | Contents | Rule |
|---|---|---|
| **P0 (must ship)** | M1 detector + Kalman filter + M2 learned noise + PID controller + safety shield + metrics | Done by end of Day 1–2; this alone is already more than "finetune YOLO" |
| **P1 (committed)** | M3 forecaster, M4 lost-tool predictor, M5 clustering with gain scheduling | Done by mid Day 2; each is small and integrates into P0 |
| **P2 (stretch)** | M6 imitation controller, M1 self-training | Start Day 2; ship only if it beats PID fallback in the closed loop, otherwise report honestly as an experiment |

---

## 5. Datasets

> Check licence, registration and availability before relying on any of these.

| Dataset | Used for |
|---|---|
| **m2cai16-tool-locations** (bounding boxes, 7 tools, ~2.5–3k frames, cholecystectomy) | Fine-tune M1; ground truth for validating tip error and for training M2 |
| **Cholec80** (80 videos, tool presence labels) | Long, continuous video: virtual viewport, track extraction for M3/M4/M5, unlabelled data for self-training |
| **CholecSeg8k** (~8k frames from Cholec80 with pixel masks) | Masks to derive better tip positions, optional segmentation head for M1 |
| **EndoVis 2017 instrument segmentation** (da Vinci, optional) | Extra masks; different instrument appearance, so use only if time allows |
| **Synthetic trajectories** (generated by Member B) | Early development of M3/M4/M5/M6 before real tracks exist; augmentation later |
| **Fallback** | Self-recorded tabletop clips with a metallic stick, ~200 frames annotated in CVAT/Roboflow, clearly labelled as a proxy |

---

## 6. Three-Day Build Plan

### Roles

| Member | Owns |
|---|---|
| **A: Perception** | M1 detector, M2 noise model, track extraction pipeline, self-training |
| **B: Prediction and Learning** | Synthetic trajectories, M3 forecaster, M4 lost-tool predictor, M5 clustering, M6 imitation controller |
| **C: Systems and Control** | Virtual viewport, PyBullet arm, ROS 2 nodes, Kalman filter, PID, safety shield, FSM, metrics, demo |

**How they stay independent:** A works offline on dataset images and videos. B trains on synthetic trajectories until A's real tracks arrive. C uses a dummy detector (ground-truth tip or colour blob) and dummy forecaster until real nodes are ready.

### Day 1: Build each part in isolation

**Member A**
- Download and organise data; split by video, never by frame.
- Fine-tune YOLOv8-nano on m2cai16-tool-locations (start training early; use a free Colab/Kaggle GPU).
- Define the tip-extraction rule and write `detect_tip(image) → (u, v, conf, bbox)`.
- Write the track-extraction script: run the detector over Cholec80 clips, output per-frame tip tracks in full-frame coordinates.
- *Deliverable:* trained detector, tip function, first batch of tracks (even a few clips).

**Member B**
- Write a synthetic trajectory generator: smooth minimum-jerk segments, pauses, fast transits, noise, dropouts, tool exits.
- Build the shared dataset code: windowing (input history, future target, "lost-in-next-0.5 s" label).
- Train a first **M3** (GRU/TCN) and **M4** (random forest/GBM) on synthetic data; implement **M5** (k-means with silhouette-chosen k) on window features.
- Write the **MPC expert** (cvxpy or scipy) that takes the true future tip trajectory and produces optimal pan/tilt/zoom accelerations.
- *Deliverable:* training pipelines that run end to end on synthetic data, plus the expert controller.

**Member C**
- Virtual viewport (crop and resize window driven by acceleration commands, with border clamping).
- ROS 2 workspace, message definitions and launch file following the contract.
- Kalman filter (constant velocity) with an input for per-frame `R_t`.
- PID with dead-band, plus jerk/velocity limits (the safety shield).
- Metrics logger writing CSV.
- *Deliverable:* ROS 2 closed loop running with a ground-truth tip and the PID controller.

**End-of-day sync (30 min):** confirm message formats, run the dummy loop.

### Day 2: Real data, integration, learned control

**Member A**
- Evaluate the detector (mAP, tip error); wrap it as a ROS 2 node.
- Build **M2**: collect (features → tip error) pairs on validation videos, train the gradient-boosted regressor, publish `R_t`.
- Hand the real tracks to Member B by **midday**.
- *Stretch (P2):* **self-training**: pseudo-label unlabelled Cholec80 frames above a confidence threshold, retrain, and check tip error on held-out labelled frames.
- *Deliverable:* perception node publishing tip and `R_t`; real tracks.

**Member B**
- Retrain **M3, M4, M5** on real tracks mixed with synthetic ones; export small models (ONNX or TorchScript) and wrap each as a ROS 2 node.
- Hook **M5** into the PID: one gain set per mode.
- Train **M6** by behaviour cloning on rollouts from the MPC expert (with noisy observations and the same inputs the live system provides), then 1 to 2 DAgger rounds in the simulator.
- *Deliverable:* forecaster, lost-tool predictor and clustering nodes; first M6 checkpoint.

**Member C**
- Integrate A's node, then B's nodes, into the loop one at a time, verifying that each is still running faster than the frame rate (measure per-module and end-to-end latency).
- Implement the **recovery FSM** using `p_lost` for pre-emptive zoom-out; add automatic fallback to PID.
- Add scripted failure scenarios (occlusion, tool leaving frame, glare overlay) for repeatable tests.
- Connect the same command topic to the PyBullet arm.
- *Deliverable:* full P0 + P1 pipeline running closed loop; M6 swappable with PID by a launch argument.

**End-of-day sync (45 min):** full run; list the top three failure modes.

### Day 3: Fix, validate, present

**Member A**
- Fix the top perception failures found in the integrated runs (augmentation, thresholds, temporal smoothing).
- Produce qualitative figures of success and failure cases, and the calibration plot for M2 (predicted vs actual tip error).
- Write the perception section.

**Member B**
- Final models; freeze before the held-out run. Produce plots for the forecaster (predicted vs actual paths), the lost-tool predictor (PR curve, early-warning lead time) and the clusters (mode visualisation, silhouette scores).
- Finish M6: decide on ship or experiment by closed-loop results versus the PID fallback.
- Write the learning section.

**Member C**
- Run the final evaluation on held-out clips: centring error, jerk, loss events, response lag, reacquisition time.
- Run **leave-one-out ablations** (disable M2, M3, M4, M5 or M6 in turn) to show each module's effect on the camera's behaviour.
- Record the side-by-side demo video (full frame with crop window, and the surgeon's view); assemble slides; rehearse a 3-minute demo script; keep a backup recording.

**Final 2 hours (all):** bug fixing, rehearsal, limitations slide.

---

## 7. Evaluation (Short)

| Metric | What it shows |
|---|---|
| Centring error (mean, 95th percentile) | Core task performance |
| Jerk / acceleration RMS | Smoothness |
| Loss-of-tool events per minute | Robustness |
| Response lag | Reactivity |
| Reacquisition time and success rate | Recovery behaviour |
| Tip error (px), mAP | Detector quality |
| Calibration of M2 (predicted vs actual error) | Whether the noise model is trustworthy |
| PR-AUC and lead time for M4 | Whether early warning works |
| Latency per module and end to end | Real-time feasibility |

The leave-one-out ablation serves as the check that every module earns its place. Any module that does not help should be reported honestly and kept optional.

---

## 8. Deliverables Checklist

- [ ] Fine-tuned detector and tip extraction
- [ ] M2 learned noise model feeding the Kalman filter
- [ ] M3 forecaster, M4 lost-tool predictor and M5 clustering running as live ROS 2 nodes
- [ ] M6 imitation controller with PID fallback (or an honest experiment write-up)
- [ ] Recovery FSM driven by `p_lost`
- [ ] Safety shield (limits, fallback)
- [ ] Held-out evaluation and leave-one-out ablation
- [ ] Demo video, slides and limitations section

## 9. Tech Stack

Python, PyTorch, Ultralytics YOLO, scikit-learn (random forest, gradient boosting, k-means, GMM), cvxpy or SciPy (MPC expert), OpenCV, NumPy, Matplotlib, ROS 2 (Humble or Jazzy), PyBullet, ONNX Runtime, CVAT/Roboflow for any extra annotation.

---

*This is an academic prototype with replayed video and no clinical validation.*
