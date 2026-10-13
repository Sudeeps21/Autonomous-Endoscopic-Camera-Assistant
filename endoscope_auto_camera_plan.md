# Idea 13: Endoscope Auto-Camera (Learned Tool Tracking)

**Project type:** Machine Learning + Computer Vision (vision-based control)
**Team size:** 3 members, working in parallel
**Timeline:** 3 days
**Risk score:** 3/10 — no ROS, no simulator install, runs on any laptop with Python

---

## 1. Problem Statement and Motivation

In laparoscopic (keyhole) surgery the surgeon operates with both hands on instruments and watches a monitor. The image comes from a **laparoscope** held by an **assistant**, who must continuously reposition it so the working instruments stay in view.

Problems with a human camera holder:

- Hand tremor and fatigue cause drifting or shaky images.
- Verbal commands ("left… up… zoom in") add lag and miscommunication.
- It occupies a trained person for hours.

**Proposed solution (demo version):** a pure-Python program that takes a real laparoscopic video, detects the instrument tip in each frame, and **moves a virtual crop window** (the "camera") so the tip stays centred — no robot, no ROS, no simulator required.

---

## 2. How the Demo Works

### 2.1 Virtual Viewport (the "simulated camera")

```
Full-resolution surgical video frame  (e.g. 1280 × 720)
        ┌──────────────────────────────────┐
        │                                  │
        │    ┌─────────────┐               │
        │    │  VIEWPORT   │  ← the crop   │
        │    │  (640×360)  │    window     │
        │    └─────────────┘               │
        │                                  │
        └──────────────────────────────────┘
```

- The **viewport** is a crop window over the full frame.
- The controller moves this window (pan left/right, tilt up/down) and resizes it (zoom in/out).
- The "surgeon's view" is just the content inside the window — resize it to a fixed display size.
- This simulates a camera arm **using real tissue and real instruments**, so the detector sees realistic images.
- No robot needed; the whole system is a Python script.

### 2.2 Control Loop

```
Video frame → Tool-tip detector → Kalman filter → PID controller → Move viewport
     ↑            (pixel x, y)      (smooth,          (pan/tilt       ↓
     └─────────── next frame ←───── predict)          velocities)  new crop
```

1. **Detect** the instrument tip in the current viewport → pixel `(u, v)`.
2. **Compute error:** `e = (u, v) − (centre of viewport)`.
3. **Filter** with a Kalman filter (smooths noise, predicts through missed detections).
4. **Control** with PID + dead-band (don't move if tip is already near centre).
5. **Shift the viewport** by the commanded pan/tilt pixels.
6. **Repeat** frame-by-frame.

---

## 3. Machine Learning Techniques

### 3.1 Perception ladder (three detectors, one comparison table)

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **K-means (k = 2–3) on HSV pixels** | Segment tool (low-saturation grey/metal) vs tissue (red/pink) | No labels; fast; easy to explain | Breaks with glare, blood, smoke |
| **Random forest pixel classifier** | Per-pixel features (HSV, gradient, Gabor texture, position); trained on mask labels | More robust than k-means; no GPU | Needs ~200 labelled frames |
| **YOLOv8-nano (bounding box)** | Fine-tuned on surgical tool dataset | Accurate; real-time on CPU | Needs training data; slower to set up |

Run all three on the same test clips; report mAP/IoU and tip-localisation error. This comparison **is** the ML contribution.

**Tip extraction from mask (k-means / RF):** take the largest connected component; the tip is the extremal point along the tool shaft direction (PCA on the component → project to get the far end).

### 3.2 Lost-tool classifier (genuine ML helper)

- **Input features:** detector confidence, bounding-box area, tip velocity (from Kalman), Kalman innovation magnitude.
- **Model:** decision tree or random forest (sklearn, fits in seconds).
- **Output:** `TRACKING / UNCERTAIN / LOST`.
- Used to trigger the recovery FSM (see below).

### 3.3 State estimation — Kalman filter

- **State:** `[u, v, du, dv]` (tip position + velocity in viewport pixels).
- **Constant-velocity motion model.**
- Smooths detector noise; predicts through 1–3 missed frames; provides velocity for feed-forward.
- Optional: compare against a simple LSTM/GRU; the Kalman filter will likely win on small data — a valid result.

### 3.4 Controller

- **PID with dead-band:** do not move if the tip is within a central zone (avoids jitter).
- **Velocity and acceleration limits:** clamp the viewport shift per frame to stop the view lurching.
- **Min-jerk smoothing (optional):** low-pass filter on commands.

### 3.5 Recovery FSM

```
TRACKING ──(LOST for N frames)──► COASTING
COASTING: follow Kalman prediction, slow pan
   ├─ tip re-detected ──► TRACKING
   └─ timeout ──► SEARCH
SEARCH: zoom out (expand viewport)
   ├─ tip detected ──► REACQUIRE ──► TRACKING
   └─ timeout ──► HOLD (freeze + flag)
```

Measure: **time-to-reacquire** and **recovery success rate** on scripted failure clips.

---

## 4. Evaluation Metrics

| Metric | Definition |
|---|---|
| **Centring error** | Mean + 95th-percentile pixel distance between tool tip and viewport centre |
| **Jerk / smoothness** | RMS of the 3rd derivative of viewport position (less = smoother) |
| **Loss-of-tool events** | Count of frames where the tip leaves the viewport, per minute |
| **Response lag** | Cross-correlation delay between tool motion onset and viewport motion |
| **Reacquisition time** | Seconds from loss to tip re-centred |
| **Detector quality** | mAP@0.5 or IoU; tip-localisation error in pixels; inference time ms/frame |
| **Ablation table** | k-means+PID vs RF+Kalman+PID vs YOLO+Kalman+PID+FSM |

---

## 5. Critical Analysis

### 5.1 Strengths

- Clear clinical motivation; easily understood demo (play the video side-by-side).
- Covers perception, estimation, control, and ML comparison — all course areas.
- Vision-only: works with any instrument.
- Self-contained Python: runs on any laptop; no install headaches.
- Clean ablation comparisons are straightforward.

### 5.2 Weaknesses and risks

| Risk | Mitigation |
|---|---|
| Video replay: tool does not react to camera | Acceptable for evaluation; state it as a scope limitation |
| Occlusion, smoke, blood | Kalman coasting + recovery FSM; report failure cases |
| Multiple tools in frame | Fix a policy (centroid of two tips, or follow the most recently moving one) |
| Overfitting / poor generalisation | Cross-video split (train and test on different patients/videos) |
| Evaluation validity | Explicit: evaluates tracking on replayed video, not clinical performance |

---

## 6. Dataset Options

> Always check each dataset's licence before use.

### Option A: m2cai16-tool-locations *(recommended for detector training)*
- Laparoscopic cholecystectomy frames with **bounding-box annotations** for 7 tool types.
- ~2.5k–3k annotated frames.
- Directly trains YOLOv8; box centre approximates the tip.

### Option B: CholecSeg8k (from Cholec80)
- ~8,000 frames with **pixel-level segmentation** (grasper, L-hook, etc.).
- Great for training the k-means baseline and random forest classifier.

### Option C: Cholec80 video clips
- 80 cholecystectomy videos; tool presence labels.
- Use as **source video for the virtual viewport** and tracking evaluation.

### Fallback (no download access)
- Record a short clip with a **webcam + metallic pen/stick on a table** (clearly labelled as a proxy).
- Annotate ~200 frames in Roboflow (free tier) or CVAT.
- Works surprisingly well for demonstrating the tracking loop.

---

## 7. Project Structure

```
endoscope_tracker/
├── data/               # video clips and annotation files
├── models/             # saved model weights (.pt, .pkl)
├── src/
│   ├── detector.py     # k-means, RF, YOLO detectors (same interface)
│   ├── kalman.py       # Kalman filter
│   ├── controller.py   # PID + dead-band + limits
│   ├── fsm.py          # recovery finite-state machine
│   ├── viewport.py     # virtual viewport (crop/zoom logic)
│   ├── metrics.py      # centring error, jerk, lag, loss events
│   └── demo.py         # main entry point (plays the video)
├── notebooks/
│   ├── 01_detector_training.ipynb
│   ├── 02_kalman_tuning.ipynb
│   └── 03_evaluation.ipynb
├── results/            # CSV logs, plots, demo video
└── requirements.txt
```

**`requirements.txt`** (everything installable with `pip`):
```
opencv-python
numpy
scipy
scikit-learn
ultralytics
torch
matplotlib
```

---

## 8. 3-Day Build Plan (3 Members, Independent Workstreams)

### Roles and interfaces (agree in the first hour)

| Member | Workstream | Owns |
|---|---|---|
| **Member A** | Perception (ML) | Datasets, k-means / RF / YOLO, tip extraction, lost-tool classifier |
| **Member B** | Estimation and Control | Kalman filter, PID controller, recovery FSM, synthetic test harness |
| **Member C** | Viewport, Integration and Evaluation | `viewport.py`, `demo.py`, metrics logger, demo video, report |

**Shared interface (plain Python function calls — no message bus, no ROS):**

```python
# detector.py  — Member A delivers this
def detect_tip(frame_bgr: np.ndarray) -> tuple[float, float, float]:
    """Returns (u_pixels, v_pixels, confidence 0-1) in the viewport frame."""

# controller.py  — Member B delivers this
def compute_cmd(tip_uv, viewport_state) -> tuple[float, float, float]:
    """Returns (dpan_px, dtilt_px, dzoom_factor) for this frame."""

# viewport.py  — Member C delivers this
def step_viewport(frame_full, cmd) -> np.ndarray:
    """Applies command to viewport state; returns the cropped+resized view."""
```

Mocks that allow independence:
- **A** works offline on still frames — just calls `detect_tip(img)` in a loop.
- **B** tests with a **synthetic moving point** (sinusoid + noise + dropout windows).
- **C** tests with a **colour-blob dummy detector** and a **constant-velocity fake controller**.

---

### Day 1 — Build each component in isolation

**Member A: Perception baseline**
- Download and organise dataset; make a train/val/test split **by video** (not by frame).
- Implement **k-means** (HSV/Lab, k=2–3) + tip extraction from connected components.
- Train **random forest** pixel classifier; compare IoU against k-means.
- Start YOLOv8-nano fine-tuning (let it train overnight).
- *Deliverable:* `detect_tip(image) → (u, v, conf)` + baseline comparison table.

**Member B: Estimation and control**
- Write a **trajectory simulator** (smooth sinusoidal tool path, Gaussian noise, dropout windows).
- Implement **Kalman filter** (constant-velocity); tune Q and R on synthetic data.
- Implement **PID with dead-band** + pan/tilt limits; plot centring error and jerk.
- *Deliverable:* `kalman.py` + `controller.py` + synthetic test plots.

**Member C: Viewport and plumbing**
- Build `viewport.py` — crop window, pan/tilt shift, zoom resize — driven by a `(dpan, dtilt, dzoom)` command.
- Build `demo.py` — reads video, calls detector → controller → viewport in a loop, displays side-by-side (`full frame with crop rectangle | surgeon's view`).
- Build `metrics.py` — logs centring error, jerk, viewport state to CSV each frame.
- *Deliverable:* `demo.py` running end-to-end with the dummy detector and dummy controller.

**End of Day 1 sync (30 min):** plug real detector into demo; check the loop runs at ≥15 FPS.

---

### Day 2 — Integrate and add intelligence

**Member A**
- Finish YOLO training; evaluate mAP, tip error, inference ms/frame.
- Integrate the lost-tool **random forest / decision tree classifier**.
- *Deliverable:* perception comparison table (k-means vs RF vs YOLO); `detector.py` with a `method=` switch.

**Member B**
- Integrate Kalman filter into the live demo loop.
- Implement the **recovery FSM** (`fsm.py`); wire its state output to the controller.
- Add optional min-jerk smoothing or velocity ramping.
- *Deliverable:* control node with FSM demonstrated on synthetic dropouts + first live test.

**Member C**
- Run first full closed-loop tests on 3–5 video clips; collect CSV metrics.
- Create **scripted failure scenarios**: artificial occlusion mask painted over frames, glare circle injection, tool forced out of frame.
- Start editing the demo video (screen-record `demo.py`).
- *Deliverable:* integrated system + first metrics spreadsheet.

**End of Day 2 sync (45 min):** run integrated demo; list top 3 failure modes; freeze interfaces.

---

### Day 3 — Tune, evaluate, present

**Member A**
- Fix top perception failures (augmentation: blur, brightness jitter, horizontal flip).
- Qualitative figure: 6-panel grid of success and failure cases with tip prediction overlaid.
- Write the perception section of the report.

**Member B**
- Final gain / dead-band / FSM threshold tuning on a **tuning clip set**; freeze before evaluating on **held-out set**.
- Run the ablation: PID only → +Kalman → +FSM; record metric deltas.
- Write the control and recovery section.

**Member C**
- Run full evaluation on held-out clips; generate plots (centring error CDF, jerk histogram, loss-event timeline).
- Finalise and narrate the demo video (screen-record + voice-over).
- Assemble report and slides; prepare a 3-minute demo script.

**Final 2 hours (all):** rehearsal, bug fixes, save a backup demo video in case of live failure.

---

## 9. Deliverables Checklist

- [ ] `demo.py` runs end-to-end on a video file, no extra setup beyond `pip install`
- [ ] Detector comparison table: k-means vs RF vs YOLO (IoU, tip error, ms/frame)
- [ ] Ablation table: baseline vs +Kalman vs +recovery FSM
- [ ] Metrics: centring error, jerk, loss events, lag, reacquisition time
- [ ] Recovery behaviour demonstrated on scripted failure clips
- [ ] Demo video (screen-recorded, side-by-side view)
- [ ] Slides and report with limitations section

---

## 10. Tech Stack

| Component | Library |
|---|---|
| Video I/O | `opencv-python` |
| ML models | `scikit-learn` (k-means, RF, DT), `ultralytics` (YOLOv8) |
| State estimation | `numpy` / `scipy` (Kalman) |
| Visualisation | `matplotlib`, `cv2.imshow` |
| Data annotation | Roboflow (free) or CVAT |
| Experiment logs | CSV + Matplotlib (no heavy tooling) |

No ROS. No simulator. No GPU required (CPU-only PyTorch works for YOLOv8-nano at real-time on most laptops).

---

## 11. Demo Script (3 minutes)

1. **(30 s)** Show the problem: play the raw video, point out the tool drifting around.
2. **(30 s)** k-means baseline: show the segmented mask + viewport struggling.
3. **(60 s)** Full system (YOLO + Kalman + PID + FSM): play the demo video, highlight smooth centring.
4. **(30 s)** Scripted failure: tool leaves the viewport → system zooms out → reacquires.
5. **(30 s)** Show the metrics table and ablation results.

---

## 12. Possible Extensions (if time remains)

- Zoom control based on tip-to-tip distance (two instruments).
- Horizon / roll stabilisation.
- Two-tool midpoint tracking.
- K-means clustering of motion patterns to auto-switch controller gains.
- Domain-randomisation experiments (brightness, hue shift) to quantify robustness.

---

*Disclaimer: this is an academic prototype. It is not validated for clinical use.*

---

## 1. Problem Statement and Motivation

In laparoscopic (keyhole) surgery the surgeon operates with both hands on instruments and watches a monitor. The image comes from a **laparoscope** (a rigid endoscope with a camera) held by an **assistant**, who must continuously reposition it so the working instruments stay in view.

Problems with a human camera holder:

- Hand tremor and fatigue cause drifting or shaky images.
- Verbal commands ("left... up... zoom in") add lag and miscommunication.
- It occupies a trained person for hours.

**Proposed solution:** a robotic camera holder that watches the video, detects the instrument tip, and moves the laparoscope smoothly so the tip stays near the image centre, with no human intervention.

---

## 2. How It Works, and How It Attaches (Important Clarification)

### 2.1 What the system is NOT

The system is **not attached to the surgical instrument**. There are no sensors, markers or magnets on the tool. All information comes from the **camera image itself**. This is what makes the idea attractive: it works with any standard instrument.

### 2.2 What the system physically is

```
            Robot arm (holder)
                  |
          grips the laparoscope
          camera head / shaft  (outside the body)
                  |
 ---------- abdominal wall ---------   <- trocar acts as a pivot point
                  |                       (Remote Center of Motion, RCM)
          laparoscope shaft inside body
                  |
             camera lens  ---->  sees  ---->  instrument tips + tissue
```

- The **laparoscope passes through its own trocar** (a small port in the abdominal wall). The robot arm clamps the **scope's external end**, exactly where a human assistant would hold it.
- The trocar acts as a fixed **pivot (Remote Center of Motion, RCM)**. The scope can only: **pan** (rotate left/right about the pivot), **tilt** (up/down about the pivot), **insert/retract** (zoom in/out), and optionally **roll** (rotate the image).
- Real examples of such holders: AESOP, ViKY, FreeHand. Some use head-tracking, voice or pedals; ours uses **vision**.

### 2.3 The control loop (visual servoing)

```
 Camera frame ──> Tool-tip detector ──> Filter / predictor ──> Controller ──> Arm motion
      ^            (pixel x, y, conf)    (Kalman: smooth,       (PID / MPC:    (pan, tilt,
      |                                   predict, handle        smooth,        zoom
      └──────────────── new image <───── missing frames)         limited)       velocities)
```

1. **Perceive:** detect the instrument tip in pixels `(u, v)`.
2. **Compute error:** `e = (u, v) - (u_centre, v_centre)`.
3. **Map pixels to joint motion:** the *image Jacobian* converts a pixel error into pan/tilt velocities (for small angles, roughly `pan_rate ∝ e_u / f` and `tilt_rate ∝ e_v / f`, where `f` is focal length in pixels).
4. **Move smoothly:** velocity, acceleration and jerk limits stop the view lurching.
5. **Repeat** at 20 to 30 Hz.

### 2.4 Which tool to follow?

Surgery usually involves two or more instruments. Define a policy early:

- Follow the **centroid between the two tips**, or
- Follow the **most recently moving / active tool**, or
- Follow a **designated tool** (for example, the dissector).

For a 3-day project, choose the **single dominant tool** (or the midpoint of two) and state this as a scope limitation.

---

## 3. Machine Learning Techniques

### 3.1 Perception (detecting the tool)

| Approach | Role | Pros | Cons |
|---|---|---|---|
| **K-means clustering** | Unsupervised colour/pixel segmentation (tool is grey/metallic with low saturation; tissue is red/pink) | No labels needed; fast baseline; easy to explain | Fails with specular highlights, blood, smoke, shadows |
| **Decision tree** | Classify pixels as tool vs background using colour/texture features | Interpretable; very fast | Overfits; noisy masks |
| **Random forest** | Same as above, ensemble | More robust than a single tree; no GPU; was used in classical surgical-tool segmentation research | Hand-crafted features limit accuracy |
| **CNN detector (YOLOv8-nano / SSD / Faster R-CNN)** | Bounding box of tools | Accurate; real-time on a modest GPU | Needs labelled data and training time |
| **Keypoint / heatmap model (small U-Net or HRNet-lite)** | Predict the tool tip directly | Gives the exact point to servo on | More annotation effort; use segmentation masks to derive tips if no keypoints exist |

**Direct answer to your question:** Yes, k-means, decision trees and random forests are all usable. They are best positioned as a **classical baseline** against a **deep learning model**, which gives you a comparison table. Do not make them the main detector, because endoscopic video has glare, blood and smoke that defeat colour-based methods. Recommended "ladder":

1. **K-means (k = 2 or 3) on HSV/Lab pixels** to get a rough tool mask. Tip is the extremal point of the largest connected component.
2. **Random forest** on per-pixel features (HSV, local gradient, texture, position) trained on dataset masks. This gives a much cleaner mask.
3. **YOLOv8-nano (or segmentation head)** as the main model.

Report mAP/IoU and tip-localisation error for all three.

### 3.2 Where else classical ML helps (useful, not forced)

- **Random forest / gradient boosting for "tool lost" classification:** input = detector confidence, bounding-box size, recent velocity, filter innovation; output = tracking / uncertain / lost. Small, interpretable, genuinely useful.
- **K-means on trajectories:** cluster recorded tool-tip motion patterns (for example, "slow dissection" vs "fast transit") to switch controller gains. Optional extension.

### 3.3 State estimation and prediction

- **Kalman filter** (constant-velocity model) on the tip position. It smooths detector noise, predicts through brief dropouts, and gives a velocity estimate for feed-forward.
- **Optional:** small LSTM/GRU to predict the next 0.5 s of motion. Compare against the Kalman filter; the Kalman filter will likely win on small data, which is itself a legitimate finding.

### 3.4 Control

- **PID with dead-band** (do not move if the tool is already within a central zone). This is the strongest simple baseline and avoids jitter.
- **Minimum-jerk or S-curve velocity profile / low-pass filter** on commands for smoothness.
- **MPC (optional):** minimise centring error + control effort + jerk subject to velocity/acceleration limits and the RCM workspace. Only attempt if time remains.
- **RL (optional, not recommended in 3 days):** hard to tune, hard to validate safely. Mention as future work.

### 3.5 "Lost-tool" recovery (the key differentiator)

Implement as a finite-state machine:

```
TRACKING ──(confidence < τ for N frames)──> COASTING
COASTING: follow Kalman prediction, slow down
   ├─ tool re-detected ──> TRACKING
   └─ timeout ──> SEARCH
SEARCH: zoom out (retract), widen field of view
   ├─ tool detected ──> REACQUIRE (re-centre gently) ──> TRACKING
   └─ timeout ──> HOLD (stop and flag "tool lost" to the surgeon)
```

Measure: **time-to-reacquire**, **recovery success rate**, and **loss events per minute**.

---

## 4. Evaluation Metrics

| Metric | Definition |
|---|---|
| **Centring error** | Mean and 95th-percentile pixel distance (or % of image diagonal) between tool tip and image centre |
| **Jerk / smoothness** | RMS of the third derivative of camera position, or mean absolute acceleration |
| **Loss-of-tool events** | Count of frames or episodes where the tip is outside the field of view, per minute |
| **Response lag** | Delay between tool motion and camera response (cross-correlation or step-response time) |
| **Reacquisition time** | Seconds from loss to tool re-centred |
| **Detector quality** | mAP@0.5 or IoU; tip-localisation error in pixels; inference time (ms) |
| **Ablation table** | Baseline (k-means + PID) vs RF + Kalman + PID vs YOLO + Kalman + PID (+ recovery FSM) |

---

## 5. Critical Analysis

### 5.1 Strengths

- Clear clinical motivation and an easily understood demo.
- Combines perception, estimation and control, covering several course areas.
- Vision-only: works with any instrument, no hardware changes.
- Clean ablations are possible (which component improves which metric).

### 5.2 Weaknesses and risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Sim-to-real gap** | Gazebo/PyBullet rendering looks nothing like tissue, so a detector trained on real frames will fail in sim | Use **real video with a virtual viewport** (Section 6) instead of rendering tissue |
| **Perception and control tuned together** | Coupled debugging is slow | Fix interfaces early; test each side with mocks |
| **Occlusion, smoke, blood, glare** | Detector dropouts | Kalman coasting + recovery FSM; report failure cases honestly |
| **Multiple tools** | Ambiguous target | State the policy and scope |
| **Latency** | Detector + ROS 2 + actuation delays can destabilise the loop | Measure end-to-end latency; use a smaller model; predict ahead with Kalman |
| **Safety** | A real robot near tissue must be bounded | Hard limits on speed, workspace and RCM; emergency hold; clearly state this is **research only, not clinical** |
| **Small labelled datasets** | Overfitting; poor generalisation across procedures | Augmentation (blur, brightness, glare), cross-video split (train and test on different videos) |
| **Camera-motion coupling** | Moving the camera changes the whole image, which confuses motion models | Track in the image frame, but keep the Kalman model in terms of the world/virtual frame if possible |
| **Evaluation validity** | No real surgeon-in-the-loop | Be explicit: this evaluates tracking in replayed video, not clinical performance |

### 5.3 Risk score justification

- **Original 5/10** is reasonable: the perception model is easy to obtain via public datasets, but integration and tuning are the time sinks.
- **Revised 4/10** if you use the virtual viewport (no rendering gap, fewer simulator headaches) and cap scope to one tracked tool.
- It would rise to **6 to 7/10** if you attempt RL, multi-tool policies, or a physical robot.

---

## 6. Recommended Simulation Strategy (Reduces Risk)

**Virtual viewport on real surgical video.**
Take a real laparoscopic video (high-resolution if possible). The "camera" sees only a **crop window** (for example, 640×480) of the full frame. The controller moves the window (pan/tilt in pixels) and resizes it (zoom). This simulates a camera arm using **real tissue and real instruments**, so the detector sees realistic images.

Caveat: tool motion is replayed from the video and does not react to the camera, which is acceptable for evaluating tracking.

**Hybrid for the "robotics hook":**

- Use the virtual viewport as the main evaluation.
- Add a **PyBullet or Gazebo arm model** with an RCM constraint (2 revolute joints + 1 prismatic) driven by the same ROS 2 command topic, to demonstrate kinematics and the ROS 2 loop. A coloured moving tool target in sim is enough here.
- PyBullet is faster to set up than Gazebo for a 3-day sprint; pick **one**.

---

## 7. Dataset Options

> Always check each dataset's licence, registration requirement and download availability before relying on it, since access terms can change.

### Option A: m2cai16-tool-locations (recommended for the detector)
- **Content:** laparoscopic cholecystectomy frames with **bounding-box annotations** for 7 instrument types.
- **Size:** roughly 2.5k to 3k annotated frames (verify exact count on download).
- **Why:** directly trains a YOLO-style detector; box centre or tip heuristic gives the target point.
- **Limitation:** boxes, not tip keypoints; tip must be approximated (for example, extreme point of the box along the shaft direction, or refined via segmentation).

### Option B: EndoVis 2017 Robotic Instrument Segmentation (MICCAI challenge)
- **Content:** da Vinci robotic surgery video frames with **pixel-level instrument masks** (including part labels in some versions).
- **Size:** about 8 training sequences of 225 frames, plus test sequences.
- **Why:** masks let you train the random forest / U-Net and **derive tip positions** from the mask geometry; higher-quality images.
- **Limitation:** robotic instruments look different from manual laparoscopic tools; small dataset; registration required.

### Option C: Cholec80 + CholecSeg8k
- **Cholec80:** 80 cholecystectomy videos with **tool presence labels** (7 tools) and phase labels. Excellent as **source video for the virtual viewport** and for the tool-lost classifier.
- **CholecSeg8k:** about 8,000 frames from Cholec80 with **pixel-level segmentation** (including grasper and L-hook electrocautery classes). Good for the k-means / random-forest / U-Net segmentation ladder.
- **Why:** long continuous videos for tracking evaluation plus segmentation masks for training.
- **Limitation:** Cholec80 has presence labels only (no locations); needs access request; large download.

### Suggested combination
- **Train detector:** m2cai16-tool-locations (A) and/or CholecSeg8k (C).
- **Evaluate tracking/control:** continuous clips from Cholec80 (C) with the virtual viewport.
- **Fallback if downloads are slow:** record a short clip yourself with a webcam and a coloured or metallic stick on a tabletop (clearly label as a proxy), and use a small manually annotated set (~200 frames, labelled in Roboflow or CVAT).

---

## 8. 3-Day Build Plan (3 Members, Independent Workstreams)

### Roles and interfaces (agree in the first hour)

| Member | Workstream | Owns |
|---|---|---|
| **Member A** | Perception (ML) | Datasets, k-means/RF/YOLO models, tip extraction, lost-tool classifier |
| **Member B** | Estimation and Control | Kalman filter, PID/min-jerk controller, recovery FSM, simulated tool trajectories |
| **Member C** | Simulation, ROS 2 and Evaluation | Virtual viewport, PyBullet/Gazebo arm, ROS 2 nodes, metrics, demo, report |

**Interface contract (freeze on Day 1):**

| Topic | Type | Rate | Producer to Consumer |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 30 Hz | C to A |
| `/tool/tip` | `geometry_msgs/PointStamped` (pixels) + a confidence field (custom msg or `Float32` on `/tool/confidence`) | 30 Hz | A to B |
| `/camera/cmd_vel` | `geometry_msgs/Twist` (pan, tilt, zoom rates) | 30 Hz | B to C |
| `/tracker/state` | `std_msgs/String` (TRACKING / COASTING / SEARCH / HOLD) | event | B to all |

**Mocks that allow independence:**
- **A** works offline on dataset images (no ROS needed until Day 2).
- **B** tests with a **synthetic moving point** (sinusoid, random walk, with dropouts and noise), no real detector needed.
- **C** tests with a **dummy detector** (colour-blob or ground-truth tip from labels) and a **fake controller** (constant velocity).

---

### Day 1: Build each component in isolation

**Member A: Perception baseline**
- Download and organise datasets; fix a train/val/test split **by video**, not by frame.
- Implement the **k-means** tool mask baseline and tip extraction.
- Train the **random forest** pixel classifier; compare mask IoU against k-means.
- Start YOLOv8-nano training (leave running overnight if possible).
- *Deliverable:* `detect_tip(image) -> (u, v, conf)` function plus a baseline results table.

**Member B: Estimation and control core**
- Write a **trajectory simulator** (smooth tool motion, noise, dropouts, occlusion windows).
- Implement the **Kalman filter** (constant velocity); tune Q and R.
- Implement **PID with dead-band** + velocity/acceleration/jerk limits; implement the pixel-error-to-pan/tilt mapping.
- Plot centring error and jerk on synthetic data.
- *Deliverable:* controller library plus synthetic-test plots.

**Member C: Simulation and plumbing**
- Build the **virtual viewport** (crop and resize window over video, driven by `cmd_vel`).
- Set up the ROS 2 workspace, nodes, topics and launch file following the interface contract.
- Build the PyBullet (or Gazebo) arm with an RCM constraint and a moving coloured target.
- Write the **metrics logger** (centring error, jerk, lag, loss events) writing to CSV.
- *Deliverable:* running ROS 2 loop with dummy detector and dummy controller.

**End of Day 1 sync (30 min):** verify message formats match; run a hello-world loop.

---

### Day 2: Integrate and add intelligence

**Member A**
- Finish YOLO training and evaluate (mAP, tip error, FPS). Export (ONNX/TorchScript) for faster inference.
- Wrap the best detector as a **ROS 2 node** publishing `/tool/tip` and confidence.
- Train the **lost-tool classifier** (random forest or decision tree on confidence, box size, velocity, Kalman innovation).
- *Deliverable:* perception node and model comparison table (k-means vs RF vs YOLO).

**Member B**
- Wrap the Kalman filter + controller as a ROS 2 node.
- Implement the **recovery FSM** (TRACKING, COASTING, SEARCH, REACQUIRE, HOLD) with parameters for thresholds and timeouts.
- Add optional **MPC** (or min-jerk smoothing) if time permits.
- Unit-test the FSM against synthetic dropouts.
- *Deliverable:* control node plus recovery behaviour demonstrated on synthetic data.

**Member C**
- Integrate A and B nodes into the virtual viewport pipeline; measure **end-to-end latency**.
- Run first full closed-loop tests on 3 to 5 video clips.
- Create **scripted failure scenarios** (artificial occlusion, tool leaving frame, glare injection) for repeatable recovery tests.
- Connect the same command topic to the PyBullet/Gazebo arm.
- *Deliverable:* integrated system running closed-loop, with first metrics.

**End of Day 2 sync (45 min):** integrated run; list top 3 failure modes.

---

### Day 3: Tune, evaluate, present

**Member A**
- Fix the top perception failure modes (augmentation, confidence thresholds, temporal smoothing).
- Produce a qualitative figure of success and failure cases.
- Write the perception section of the report.

**Member B**
- Final tuning of gains, dead-band, and FSM thresholds on a **tuning clip set**; freeze before evaluating on a **held-out set**.
- Run the ablation: PID only vs Kalman + PID vs Kalman + PID + recovery.
- Write the control and recovery section.

**Member C**
- Run the full evaluation on the held-out clips and generate tables and plots (centring error, jerk, loss events, lag, reacquisition time).
- Record the **demo video**: side-by-side (full frame with crop window, and the "surgeon's view").
- Assemble the report and slides; prepare a 3-minute demo script.

**Final 2 hours (all):** rehearsal, bug fixing, backup demo video in case of live failure.

---

## 9. Deliverables Checklist

- [ ] Working ROS 2 closed loop (virtual viewport, plus arm model if time allows)
- [ ] Detector comparison table: k-means vs random forest vs YOLO (mAP/IoU, tip error, FPS)
- [ ] Ablation table: baseline vs +Kalman vs +recovery
- [ ] Metrics: centring error, jerk, loss events, lag, reacquisition time
- [ ] Recovery behaviour demonstrated and measured on scripted failures
- [ ] Demo video and slides
- [ ] Honest limitations section (sim-to-real, no clinical validation, single-tool scope)

---

## 10. Suggested Tech Stack

- **Language/tools:** Python, PyTorch, Ultralytics YOLO (or torchvision), scikit-learn (k-means, decision tree, random forest), OpenCV, NumPy/SciPy, Matplotlib
- **Robotics:** ROS 2 (Humble or Jazzy), PyBullet (or Gazebo), `cv_bridge`
- **Annotation (if needed):** CVAT or Roboflow
- **Experiment tracking:** a simple CSV log plus Matplotlib (avoid heavy tooling in 3 days)

---

## 11. Possible Extensions (if time remains)

- Zoom control based on tool-tip spread (zoom in during fine dissection, zoom out during transit).
- Horizon/roll stabilisation.
- Two-tool midpoint tracking.
- Learning gain schedules via k-means clustering of motion patterns.
- Domain-randomisation experiments to quantify sim-to-real sensitivity.

---

*Disclaimer: this is an academic prototype. It is not validated for clinical use.*
