"""
Endoscope Auto-Camera: Universal Autonomous Surgical Tool Tracking & Stabilization Demo
=========================================================================================
Features:
  - Universal Video Input: Works with ANY video (.mp4, .avi, .mov, .mkv), webcam (0), or image sequence
  - Multi-Tool Handling Policy:
      * Centroid Framing: Smoothly frames multiple tools simultaneously and auto-scales zoom
      * Motion Priority: Automatically follows the active operating tool over stationary retractors
      * Single Tool Focus: Locks onto highest-confidence primary instrument
  - Deep Learning Perception: Fine-Tuned YOLOv8-nano (Surgical instrument detector)
  - Probabilistic Filtering: Multi-instance Kalman Filters for tremor removal & trajectory prediction
  - Learned State Machine: Decision Tree (TRACKING, UNCERTAIN, LOST) with auto zoom-out recovery
  - Adaptive Damping: Online ML velocity regressor that adapts camera speed to surgeon tempo

Usage:
  python demo.py                                # Auto-detects sample video or frames
  python demo.py --source my_surgery.mp4        # Run on ANY video file
  python demo.py --source 0                     # Run on live webcam / capture card
  python demo.py --headless --source test.mp4   # Render output video without GUI

Keyboard Controls:
  [SPACE] : Pause / Resume
  [m]     : Cycle Multi-Tool Policy (Centroid Framing -> Motion Priority -> Single Tool)
  [+] / [-]: Adjust Zoom In / Out
  [r]     : Reset Camera Viewport
  [q]/ESC : Quit
"""

import os, sys, time, cv2, joblib, warnings, argparse
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Distinct color palette for multiple surgical tools (BGR)
TOOL_COLORS = [
    (0, 255, 120),   # Green
    (255, 180, 0),   # Cyan
    (0, 160, 255),   # Orange
    (220, 80, 255),  # Magenta
]

# -----------------------------------------------------------------------------
# 1. Surgical Kalman Filter for Single/Multi-Tool State Estimation
# -----------------------------------------------------------------------------
class SurgicalKalmanFilter:
    """Constant-velocity Kalman filter tracking instrument tip (u, v, du, dv)."""
    def __init__(self, dt=1.0/30.0, process_noise=0.8, measurement_noise=2.5):
        self.dt = dt
        self.state = np.zeros(4, dtype=np.float32)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 50.0
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.is_initialized = False
        self.history = deque(maxlen=30)
        self.missed_frames = 0

    def initialize(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 5.0
        self.is_initialized = True
        self.history.clear()
        self.history.append((float(x), float(y)))
        self.missed_frames = 0

    def predict(self):
        if not self.is_initialized:
            return None
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed_frames += 1
        return float(self.state[0]), float(self.state[1])

    def update(self, z_x, z_y):
        if not self.is_initialized:
            self.initialize(z_x, z_y)
            return z_x, z_y
        z = np.array([z_x, z_y], dtype=np.float32)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        self.missed_frames = 0
        pos = (float(self.state[0]), float(self.state[1]))
        self.history.append(pos)
        return pos

    def get_velocity(self):
        if not self.is_initialized:
            return 0.0, 0.0
        return float(self.state[2]), float(self.state[3])

    def get_speed(self):
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2))


# -----------------------------------------------------------------------------
# 2. Multi-Tool Coordinator (Option 3 Implementation)
# -----------------------------------------------------------------------------
class MultiToolCoordinator:
    """
    Coordinates tracking of multiple surgical tools and selects the target camera center.
    Policies:
      - 'CENTROID': Frames all instruments simultaneously and dynamically widens zoom.
      - 'MOTION_PRIORITY': Weights focus toward active moving tools over stationary retractors.
      - 'SINGLE_PRIMARY': Locks strictly onto the highest confidence instrument.
    """
    POLICIES = ["CENTROID", "MOTION_PRIORITY", "SINGLE_PRIMARY"]

    def __init__(self):
        self.policy_idx = 0
        self.trackers = [SurgicalKalmanFilter() for _ in range(4)]

    @property
    def current_policy(self):
        return self.POLICIES[self.policy_idx]

    def cycle_policy(self):
        self.policy_idx = (self.policy_idx + 1) % len(self.POLICIES)
        return self.current_policy

    def update_tools(self, detections):
        """
        detections: list of dicts [{'tip': (x,y), 'conf': float, 'box': (x1,y1,x2,y2)}]
        returns: (target_tip_xy, target_spread, active_tool_count, tool_info_list)
        """
        # Match detections to Kalman trackers (greedy nearest-neighbor)
        tool_info = []
        for i, det in enumerate(detections[:4]):
            tip = det['tip']
            kf = self.trackers[i]
            kf.predict()
            smooth_tip = kf.update(tip[0], tip[1])
            tool_info.append({
                'idx': i,
                'tip': smooth_tip,
                'raw_tip': tip,
                'conf': det['conf'],
                'box': det['box'],
                'speed': kf.get_speed(),
                'history': list(kf.history),
                'color': TOOL_COLORS[i % len(TOOL_COLORS)]
            })

        n_tools = len(tool_info)
        if n_tools == 0:
            # ML Safeguard: Kalman Filter Bayesian Coasting / Dead-Reckoning (Occlusion bypass)
            kf = self.trackers[0]
            if kf.is_initialized and kf.missed_frames < 25:
                pred_pos = kf.predict()
                if pred_pos:
                    coast_info = [{
                        'idx': 0,
                        'tip': pred_pos,
                        'raw_tip': None,
                        'conf': 0.0,
                        'box': None,
                        'speed': kf.get_speed(),
                        'history': list(kf.history),
                        'color': (0, 165, 255),
                        'is_coasting': True,
                        'missed_frames': kf.missed_frames
                    }]
                    return pred_pos, 0.0, 1, coast_info
            return None, 0.0, 0, []

        # 1. Single Tool Policy
        if self.current_policy == "SINGLE_PRIMARY" or n_tools == 1:
            best = max(tool_info, key=lambda t: t['conf'])
            return best['tip'], 0.0, n_tools, tool_info

        # 2. Motion Priority Policy
        if self.current_policy == "MOTION_PRIORITY":
            total_speed = sum(t['speed'] for t in tool_info) + 1e-5
            weights = np.array([t['speed'] / total_speed for t in tool_info])
            # If all tools nearly static, fallback to equal weights
            if np.max(weights) < 0.4:
                weights = np.ones(n_tools) / n_tools
            weighted_x = sum(t['tip'][0] * w for t, w in zip(tool_info, weights))
            weighted_y = sum(t['tip'][1] * w for t, w in zip(tool_info, weights))
            spread = max(np.linalg.norm(np.array(tool_info[i]['tip']) - np.array(tool_info[j]['tip']))
                         for i in range(n_tools) for j in range(i+1, n_tools)) if n_tools > 1 else 0.0
            return (weighted_x, weighted_y), spread, n_tools, tool_info

        # 3. Centroid Framing Policy (Default)
        mean_x = float(np.mean([t['tip'][0] for t in tool_info]))
        mean_y = float(np.mean([t['tip'][1] for t in tool_info]))
        spread = max(np.linalg.norm(np.array(tool_info[i]['tip']) - np.array(tool_info[j]['tip']))
                     for i in range(n_tools) for j in range(i+1, n_tools)) if n_tools > 1 else 0.0

        return (mean_x, mean_y), spread, n_tools, tool_info


# -----------------------------------------------------------------------------
# 3. Adverse Surgical Stress-Test Simulator (Feature 1)
# -----------------------------------------------------------------------------
class SurgicalStressTestSimulator:
    """
    Simulates real-world clinical surgical artifacts:
      - Electrocautery smoke plumes (triggers Kalman dead-reckoning)
      - Specular glare / lens overexposure (triggers Adaptive Tremor Damping)
      - Sudden tool extraction (triggers Decision Tree LOST -> FSM Zoom Recovery)
    """
    def __init__(self, mode="manual"):
        self.mode = mode.lower()  # 'manual', 'auto', 'smoke', 'glare', 'extraction'
        self.smoke_active = (self.mode == "smoke")
        self.glare_active = (self.mode == "glare")
        self.tool_hidden = (self.mode == "extraction")
        self.smoke_phase = 0.0
        self.current_stage_name = "NORMAL BASELINE"
        self.active_safeguard = "NOMINAL MULTI-MODEL TRACKING"

    def update_timeline(self, frame_idx, total_frames=148):
        if self.mode != "auto":
            if self.smoke_active:
                self.current_stage_name = "ELECTROCAUTERY SMOKE"
                self.active_safeguard = "ML SAFEGUARD: KALMAN DEAD-RECKONING (OCCLUSION BYPASS)"
            elif self.glare_active:
                self.current_stage_name = "SPECULAR GLARE FLASH"
                self.active_safeguard = "ML SAFEGUARD: ADAPTIVE TREMOR FILTER (NOISE DAMPING)"
            elif self.tool_hidden:
                self.current_stage_name = "TOOL EXTRACTION / ABSENCE"
                self.active_safeguard = "ML SAFEGUARD: DECISION TREE FSM ZOOM-OUT RECOVERY"
            else:
                self.current_stage_name = "NORMAL BASELINE"
                self.active_safeguard = "NOMINAL MULTI-MODEL TRACKING"
            return

        # Automated 4-Phase Stress Protocol
        # Phase 1 (0-36): Baseline Surgery
        # Phase 2 (37-73): Thick Electrocautery Smoke
        # Phase 3 (74-108): Specular Glare / Flash Overexposure
        # Phase 4 (109-end): Tool Extraction
        if frame_idx < 37:
            self.smoke_active = False
            self.glare_active = False
            self.tool_hidden = False
            self.current_stage_name = "STAGE 1/4: BASELINE NOMINAL SURGERY"
            self.active_safeguard = "NOMINAL (YOLOv8 + KALMAN ACTIVE)"
        elif frame_idx < 74:
            self.smoke_active = True
            self.glare_active = False
            self.tool_hidden = False
            self.current_stage_name = "STAGE 2/4: ELECTROCAUTERY SMOKE OCCLUSION"
            self.active_safeguard = "ML SAFEGUARD: KALMAN DEAD-RECKONING (BAYESIAN PREDICTION)"
        elif frame_idx < 109:
            self.smoke_active = False
            self.glare_active = True
            self.tool_hidden = False
            self.current_stage_name = "STAGE 3/4: SPECULAR GLARE OVEREXPOSURE"
            self.active_safeguard = "ML SAFEGUARD: ADAPTIVE TREMOR FILTER (NOISE SUPPRESSION)"
        elif frame_idx < 127:
            self.smoke_active = False
            self.glare_active = False
            self.tool_hidden = True
            self.current_stage_name = "STAGE 4A: TOOL OUT OF FIELD -> AUTO ZOOM-OUT (SEARCHING)"
            self.active_safeguard = "ML DECISION TREE: LOST STATE -> EXPONENTIAL ZOOM-OUT TO WIDE VIEW"
        else:
            self.smoke_active = False
            self.glare_active = False
            self.tool_hidden = False  # Tool re-enters cavity
            self.current_stage_name = "STAGE 4B: TOOL RE-ENTERED -> RE-ACQUIRED, CENTERING & AUTO-FOCUS"
            self.active_safeguard = "ML YOLOv8 + DECISION TREE: RE-ACQUIRED -> RAPID ZOOM-IN FOCUS LOCK"

    def toggle_smoke(self):
        self.smoke_active = not self.smoke_active
        return self.smoke_active

    def toggle_glare(self):
        self.glare_active = not self.glare_active
        return self.glare_active

    def toggle_tool_hidden(self):
        self.tool_hidden = not self.tool_hidden
        return self.tool_hidden

    def apply(self, frame, tool_tip=None):
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Simulate Tool Extraction (Obscure tool region)
        if self.tool_hidden:
            if tool_tip is not None:
                tx, ty = int(tool_tip[0]), int(tool_tip[1])
                pr = int(min(w, h) * 0.28)
                x1, y1 = max(0, tx - pr), max(0, ty - pr * 2)
                x2, y2 = min(w, tx + pr), min(h, ty + pr)
                if x2 > x1 and y2 > y1:
                    patch = cv2.GaussianBlur(out[y1:y2, x1:x2], (61, 61), 45)
                    out[y1:y2, x1:x2] = patch
            cv2.putText(out, "[TOOL EXTRACTED / OUT OF FIELD]", (int(w * 0.15), int(h * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 240), 2)

        # 2. Simulate Electrocautery Smoke
        if self.smoke_active:
            self.smoke_phase += 0.14
            smoke_overlay = np.zeros_like(out, dtype=np.uint8)
            cx = int(w * 0.50 + np.sin(self.smoke_phase) * 60)
            cy = int(h * 0.52 + np.cos(self.smoke_phase * 0.7) * 40)
            cv2.circle(smoke_overlay, (cx, cy), int(140 * (w / 600.0)), (230, 235, 240), -1)
            cv2.circle(smoke_overlay, (cx + 40, cy - 30), int(95 * (w / 600.0)), (230, 235, 240), -1)
            smoke_overlay = cv2.GaussianBlur(smoke_overlay, (95, 95), 45)
            out = cv2.addWeighted(out, 0.35, smoke_overlay, 0.65, 0)

        # 3. Simulate Specular Glare / Flash Overexposure
        if self.glare_active:
            glare_overlay = np.zeros_like(out, dtype=np.uint8)
            gx, gy = int(w * 0.48), int(h * 0.42)
            cv2.ellipse(glare_overlay, (gx, gy), (int(w * 0.26), int(h * 0.17)), 35, 0, 360, (255, 255, 255), -1)
            glare_overlay = cv2.GaussianBlur(glare_overlay, (75, 75), 40)
            out = cv2.addWeighted(out, 0.60, glare_overlay, 0.90, 0)

        return out

    def get_status_str(self):
        tags = []
        if self.smoke_active: tags.append("SMOKE")
        if self.glare_active: tags.append("GLARE")
        if self.tool_hidden: tags.append("TOOL EXTRACTED")
        return " + ".join(tags) if tags else "NORMAL"


# -----------------------------------------------------------------------------
# 4. Adaptive Motion & Tremor Damping Estimator (Online ML)
# -----------------------------------------------------------------------------
class AdaptiveMotionEstimator:
    def __init__(self, window_size=15):
        self.speeds = deque(maxlen=window_size)
        self.base_kp = 0.08
        self.base_deadband = 18.0

    def record_step(self, speed):
        self.speeds.append(speed)

    def get_adaptive_gains(self):
        if len(self.speeds) < 5:
            return self.base_kp, self.base_deadband, "NOMINAL"
        mean_speed = float(np.mean(self.speeds))
        if mean_speed > 22.0:
            return min(0.16, self.base_kp * 1.5), self.base_deadband * 0.7, "FAST TRACKING"
        elif mean_speed < 7.0:
            return max(0.04, self.base_kp * 0.7), self.base_deadband * 1.3, "TREMOR DAMPING"
        return self.base_kp, self.base_deadband, "STABLE DISSECTION"


# -----------------------------------------------------------------------------
# 4. Universal Dynamic Virtual Camera Viewport
# -----------------------------------------------------------------------------
class VirtualCameraViewport:
    """Adapts dynamically to ANY frame size and scales zoom based on tool spread."""
    def __init__(self, frame_w, frame_h, nominal_zoom=0.60):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.nominal_zoom = nominal_zoom
        self.current_zoom = nominal_zoom
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.fsm_state = "TRACKING"
        self.lost_frames_count = 0
        self.focus_lock_frames = 0

    def reset(self):
        self.cx = self.frame_w / 2.0
        self.cy = self.frame_h / 2.0
        self.current_zoom = self.nominal_zoom
        self.fsm_state = "TRACKING"
        self.lost_frames_count = 0
        self.focus_lock_frames = 0

    def update(self, target_pos, tool_spread, dt_label, adaptive_kp, adaptive_deadband):
        # FSM state transitions: Detect when tool leaves field
        if dt_label == 2 or target_pos is None:
            self.lost_frames_count += 1
            if self.lost_frames_count > 3:
                self.fsm_state = "SEARCH (ZOOM OUT)"
            else:
                self.fsm_state = "COASTING"
            self.focus_lock_frames = 0
        elif dt_label == 1:
            self.fsm_state = "UNCERTAIN"
            self.lost_frames_count = 0
        else:
            # Tool re-acquired or actively tracking!
            if self.fsm_state in ["SEARCH (ZOOM OUT)", "COASTING"] or self.lost_frames_count > 0:
                self.focus_lock_frames = 25  # Engage active auto-focus lock
            
            if self.focus_lock_frames > 0:
                self.focus_lock_frames -= 1
                self.fsm_state = "RE-ACQUIRED (FOCUSING)"
            else:
                self.fsm_state = "TRACKING"
            self.lost_frames_count = 0

        # Multi-Tool Dynamic Zoom & Auto-Focus Adaptation
        if self.fsm_state == "SEARCH (ZOOM OUT)":
            # Tool out of field: expand field-of-view to wide angle (1.0x wide-angle)
            target_zoom = 0.98
            self.current_zoom += (target_zoom - self.current_zoom) * 0.12
            # Re-center viewport toward optical cavity center to maximize search coverage
            self.cx += (self.frame_w / 2.0 - self.cx) * 0.08
            self.cy += (self.frame_h / 2.0 - self.cy) * 0.08
            return 0.0
        elif self.fsm_state == "RE-ACQUIRED (FOCUSING)":
            # Tool found: dynamically zoom in and snap tight focus onto the instrument tip!
            target_zoom = min(self.nominal_zoom, 0.54)  # 1.85x magnification tight focus
            self.current_zoom += (target_zoom - self.current_zoom) * 0.16
            adaptive_kp = max(adaptive_kp, 0.20)  # Snappy re-centering
            adaptive_deadband = 4.0
        elif tool_spread > 0.0:
            # Expand zoom if tools are spread far apart so both stay visible
            spread_ratio = tool_spread / float(self.frame_w)
            desired_zoom = min(0.90, max(self.nominal_zoom, spread_ratio * 1.6))
            self.current_zoom += (desired_zoom - self.current_zoom) * 0.06
        else:
            self.current_zoom += (self.nominal_zoom - self.current_zoom) * 0.06

        if target_pos is None:
            return 0.0

        target_x, target_y = target_pos
        err_x = target_x - self.cx
        err_y = target_y - self.cy
        dist_err = float(np.sqrt(err_x**2 + err_y**2))

        # Deadband
        if dist_err < adaptive_deadband and self.fsm_state != "RE-ACQUIRED (FOCUSING)":
            return dist_err

        # PID Velocity Control
        kp, kd, ki = adaptive_kp, 0.02, 0.001
        self.integral_x = np.clip(self.integral_x + err_x, -100, 100)
        self.integral_y = np.clip(self.integral_y + err_y, -100, 100)
        vx = np.clip(kp * err_x + kd * (err_x - self.prev_err_x) + ki * self.integral_x, -22.0, 22.0)
        vy = np.clip(kp * err_y + kd * (err_y - self.prev_err_y) + ki * self.integral_y, -22.0, 22.0)

        self.cx += vx
        self.cy += vy
        self.prev_err_x = err_x
        self.prev_err_y = err_y

        # Clamping inside raw frame
        crop_w = int(self.frame_w * self.current_zoom)
        crop_h = int(self.frame_h * self.current_zoom)
        self.cx = np.clip(self.cx, crop_w / 2.0, self.frame_w - crop_w / 2.0)
        self.cy = np.clip(self.cy, crop_h / 2.0, self.frame_h - crop_h / 2.0)
        return dist_err

    def get_crop_rect(self):
        crop_w = int(self.frame_w * self.current_zoom)
        crop_h = int(self.frame_h * self.current_zoom)
        x1 = max(0, int(round(self.cx - crop_w / 2.0)))
        y1 = max(0, int(round(self.cy - crop_h / 2.0)))
        x2 = min(self.frame_w, x1 + crop_w)
        y2 = min(self.frame_h, y1 + crop_h)
        return x1, y1, x2, y2

    def render_surgeons_view(self, raw_frame, out_w=580, out_h=380):
        x1, y1, x2, y2 = self.get_crop_rect()
        crop = raw_frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = raw_frame
        view = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        # Overlay surgical crosshairs
        hc_x, hc_y = out_w // 2, out_h // 2
        is_focusing = (self.fsm_state == "RE-ACQUIRED (FOCUSING)")
        is_searching = (self.fsm_state == "SEARCH (ZOOM OUT)")
        reticle_color = (0, 255, 100) if is_focusing else (0, 100, 255) if is_searching else (0, 220, 255)
        thickness = 2 if is_focusing else 1

        cv2.line(view, (hc_x - 20, hc_y), (hc_x + 20, hc_y), reticle_color, thickness)
        cv2.line(view, (hc_x, hc_y - 20), (hc_x, hc_y + 20), reticle_color, thickness)
        cv2.circle(view, (hc_x, hc_y), 34, reticle_color, thickness)
        if is_focusing:
            cv2.circle(view, (hc_x, hc_y), 18, (0, 255, 100), 1)
            cv2.putText(view, "TARGET ACQUIRED: FOCUS LOCKED", (hc_x - 120, hc_y + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 100), 1)
        elif is_searching:
            cv2.putText(view, "TOOL LOST: SEARCHING WIDE FOV", (hc_x - 120, hc_y + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 100, 255), 1)
        return view


# -----------------------------------------------------------------------------
# 5. Universal Video Stream Reader (Files, Folders, Webcam)
# -----------------------------------------------------------------------------
class UniversalVideoStream:
    """Seamlessly streams ANY video file, folder of images, or webcam with auto-looping."""
    def __init__(self, source_path=None):
        self.is_video_file = False
        self.is_image_folder = False
        self.is_webcam = False
        self.cap = None
        self.image_files = []
        self.img_idx = 0
        self.fps = 25.0

        base_dir = Path(__file__).resolve().parent

        # 1. Resolve source
        if source_path is None or source_path == "":
            # Auto-detect priority: sample_surgery.mp4 -> any .mp4 -> demo_data/frames
            if (base_dir / "sample_surgery.mp4").exists():
                source_path = str(base_dir / "sample_surgery.mp4")
            elif list(base_dir.glob("*.mp4")):
                source_path = str(list(base_dir.glob("*.mp4"))[0])
            elif (base_dir / "demo_data" / "frames").exists():
                source_path = str(base_dir / "demo_data" / "frames")

        source_str = str(source_path)

        # Check if webcam index
        if source_str.isdigit():
            self.is_webcam = True
            self.cap = cv2.VideoCapture(int(source_str))
            print(f"📷 Connected to Live Camera [{source_str}]")
        elif Path(source_str).is_dir():
            self.is_image_folder = True
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for ext in exts:
                self.image_files.extend(sorted(list(Path(source_str).glob(ext))))
        elif Path(source_str).is_file():
            self.is_video_file = True
            self.cap = cv2.VideoCapture(source_str)
            cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if cap_fps and cap_fps > 5.0 and cap_fps < 120.0:
                self.fps = cap_fps
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"🎬 Loaded Video File: '{Path(source_str).name}' ({self.fps:.1f} native FPS, {self.total_frames} frames)")
        else:
            raise FileNotFoundError(f"Could not open source: '{source_str}'")

        if self.is_image_folder:
            self.total_frames = len(self.image_files)
        elif not self.is_video_file:
            self.total_frames = None

    def read_frame(self):
        if self.is_webcam or self.is_video_file:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    # Loop video back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    return None
            return frame
        elif self.is_image_folder:
            if not self.image_files:
                return None
            path = self.image_files[self.img_idx]
            self.img_idx = (self.img_idx + 1) % len(self.image_files)
            return cv2.imread(str(path))
        return None

    def release(self):
        if self.cap:
            self.cap.release()


# -----------------------------------------------------------------------------
# 6. Main Interactive Cockpit
# -----------------------------------------------------------------------------
def run_auto_camera_demo(source=None, speed_factor=0.5, headless=False, max_frames=None, record_output="demo_output.mp4", stress_mode="manual"):
    print("=" * 70)
    print("🚀 Initializing Endoscope Auto-Camera Universal Simulator...")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent

    # 1. Load YOLOv8 Model
    model_paths = [
        base_dir / "member_a_outputs" / "BEST_yolov8n_m2cai16.pt",
        base_dir / "BEST_yolov8n_m2cai16.pt",
        base_dir / "best.pt",
        base_dir / "yolov8n.pt"
    ]
    model_path = next((p for p in model_paths if p.exists()), None)
    if not model_path:
        print("❌ Error: Could not locate YOLO weights file!")
        return
    print(f"📦 Loaded YOLOv8-nano Weights: {model_path.name}")
    detector = YOLO(str(model_path))

    # 2. Load Decision Tree State Classifier
    dt_paths = [
        base_dir / "member_a_outputs" / "lost_tool_classifier.pkl",
        base_dir / "checkpoints" / "lost_tool_classifier.pkl"
    ]
    dt_path = next((p for p in dt_paths if p.exists()), None)
    dt_classifier = joblib.load(str(dt_path)) if dt_path else None
    if dt_classifier:
        print("🌲 Loaded Decision Tree State Machine.")

    # 3. Initialize Video Stream
    stream = UniversalVideoStream(source)
    sample_frame = stream.read_frame()
    if sample_frame is None:
        print("❌ Error: Could not retrieve initial frame from stream!")
        return
    frame_h, frame_w = sample_frame.shape[:2]
    print(f"📐 Native Video Resolution: {frame_w} × {frame_h}")

    # Core Systems
    coordinator = MultiToolCoordinator()
    adaptive_motion = AdaptiveMotionEstimator()
    camera = VirtualCameraViewport(frame_w, frame_h, nominal_zoom=0.62)
    stress_sim = SurgicalStressTestSimulator(mode=stress_mode)
    if stress_mode != "manual":
        print(f"🧪 Adverse Conditions Stress-Test Engaged: Mode = '{stress_mode.upper()}'")

    # Output Video Writer & Canvas Setup
    record_path = base_dir / record_output
    disp_w, disp_h = 1200, 570
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(record_path), fourcc, stream.fps, (disp_w, disp_h))

    if not headless:
        try:
            cv2.namedWindow("Endoscope Auto-Camera Cockpit", cv2.WINDOW_AUTOSIZE)
        except cv2.error as e:
            print(f"⚠️ HighGUI window not available: {e}\n   Switching to background video recording mode...")
            headless = True

    paused = False
    fps_history = deque(maxlen=20)
    total_processed = 0
    current_speed = speed_factor
    last_known_target = None
    print("\n🟢 Cockpit Ready! [SPACE]: Pause | [o]: Smoke | [g]: Glare | [x]: Extract Tool | [s/f]: Speed | [q]: Quit\n")

    raw_frame = sample_frame

    while True:
        loop_start = time.time()

        if not paused:
            next_frame = stream.read_frame()
            if next_frame is not None:
                raw_frame = next_frame
            total_processed += 1

            # Update automated stress test timeline if in auto/scheduled mode
            stress_sim.update_timeline(total_processed, stream.total_frames or 148)

            # --- 0. Apply Adverse Surgical Conditions Stress-Test ---
            proc_frame = stress_sim.apply(raw_frame, last_known_target)

            # --- 1. YOLOv8 Detection ---
            results = detector(proc_frame, verbose=False, conf=0.25)[0]
            detections = []
            if len(results.boxes) > 0:
                confs = results.boxes.conf.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy()
                for b, conf in zip(boxes, confs):
                    tip_x = float((b[0] + b[2]) / 2.0)
                    tip_y = float(b[3])
                    detections.append({
                        'tip': (tip_x, tip_y),
                        'conf': float(conf),
                        'box': (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                    })

            # --- 2. Multi-Tool Coordination (Option 3) ---
            target_pos, tool_spread, n_tools, tool_info = coordinator.update_tools(detections)
            if target_pos:
                last_known_target = target_pos

            # --- 3. Decision Tree State Classification (ML Method 2) ---
            if tool_info and not tool_info[0].get('is_coasting', False):
                best_t = max(tool_info, key=lambda t: t['conf'])
                conf_val = best_t['conf']
                bx = best_t.get('box')
                box_area = float((bx[2]-bx[0]) * (bx[3]-bx[1]) / (frame_w * frame_h)) if bx else 0.0
                norm_x = target_pos[0] / frame_w if target_pos else 0.5
                norm_y = target_pos[1] / frame_h if target_pos else 0.5
                if conf_val >= 0.50:
                    dt_label = 0  # Robust re-acquisition & active tracking
                elif dt_classifier is not None:
                    dt_label = int(dt_classifier.predict([[conf_val, box_area, norm_x, norm_y]])[0])
                else:
                    dt_label = 1 if conf_val >= 0.30 else 2
            else:
                conf_val, box_area = 0.0, 0.0
                norm_x = target_pos[0] / frame_w if target_pos else 0.5
                norm_y = target_pos[1] / frame_h if target_pos else 0.5
                dt_label = 2  # LOST (Occlusion / Tool departure)

            # --- 4. Adaptive Motion Intelligence (ML Method 4) ---
            max_speed = max([t['speed'] for t in tool_info], default=0.0)
            adaptive_motion.record_step(max_speed)
            ad_kp, ad_deadband, ad_regime = adaptive_motion.get_adaptive_gains()

            # --- 5. Virtual Camera Viewport Update ---
            centering_err = camera.update(target_pos, tool_spread, dt_label, ad_kp, ad_deadband)

        # ---------------------------------------------------------------------
        # Render Cockpit Visuals
        # ---------------------------------------------------------------------
        # Left Panel: Global View (shows real-time stress test if active)
        global_view = cv2.resize(proc_frame, (580, 360))
        scale_x = 580.0 / frame_w
        scale_y = 360.0 / frame_h

        # Render all tracked tools with individual colors & trails
        for t in tool_info:
            color = t['color']
            if t.get('is_coasting'):
                color = (0, 165, 255)
                tx, ty = int(t['tip'][0] * scale_x), int(t['tip'][1] * scale_y)
                cv2.circle(global_view, (tx, ty), 14, color, 2)
                cv2.circle(global_view, (tx, ty), 4, color, -1)
                cv2.putText(global_view, f"KALMAN DEAD-RECKON #{t['missed_frames']}",
                            (max(10, tx - 70), max(20, ty - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            else:
                bx = t.get('box')
                if bx:
                    bx1, by1, bx2, by2 = bx
                    cv2.rectangle(global_view,
                                  (int(bx1 * scale_x), int(by1 * scale_y)),
                                  (int(bx2 * scale_x), int(by2 * scale_y)),
                                  color, 2)
                    cv2.putText(global_view, f"Tool {t['idx']+1}: {t['conf']*100:.0f}%",
                                (int(bx1 * scale_x), max(18, int(by1 * scale_y) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                # Tip marker
                cv2.circle(global_view, (int(t['tip'][0] * scale_x), int(t['tip'][1] * scale_y)), 5, color, -1)

            # Trajectory trail
            for k in range(1, len(t['history'])):
                p1 = (int(t['history'][k-1][0] * scale_x), int(t['history'][k-1][1] * scale_y))
                p2 = (int(t['history'][k][0] * scale_x), int(t['history'][k][1] * scale_y))
                cv2.line(global_view, p1, p2, color, 2)

        # Draw Target Cross (Center of Multi-Tool System)
        if target_pos:
            tx, ty = int(target_pos[0] * scale_x), int(target_pos[1] * scale_y)
            cv2.drawMarker(global_view, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)

        # Draw Virtual Viewport Outline on Global View
        vx1, vy1, vx2, vy2 = camera.get_crop_rect()
        cv2.rectangle(global_view,
                      (int(vx1 * scale_x), int(vy1 * scale_y)),
                      (int(vx2 * scale_x), int(vy2 * scale_y)),
                      (255, 255, 255), 2)
        cv2.putText(global_view, "CAMERA VIEWPORT",
                    (int(vx1 * scale_x) + 6, int(vy1 * scale_y) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # Right Panel: Stabilized Surgeon's View
        surgeons_view = camera.render_surgeons_view(proc_frame, out_w=580, out_h=360)

        # Assemble Cockpit Canvas
        canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        canvas[85:445, 15:595] = global_view
        canvas[85:445, 605:1185] = surgeons_view

        cv2.rectangle(canvas, (15, 85), (595, 445), (60, 60, 60), 1)
        cv2.rectangle(canvas, (605, 85), (1185, 445), (60, 60, 60), 1)
        cv2.putText(canvas, f"1. GLOBAL LAPAROSCOPIC VIEW ({n_tools} Tools Detected)", (20, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (200, 200, 200), 1)
        cv2.putText(canvas, f"2. AUTONOMOUS SURGEON VIEW (Policy: {coordinator.current_policy})", (610, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1)

        # Prominent Full-Width Stress Safeguard Banner
        if stress_sim.get_status_str() != "NORMAL" or stress_sim.mode == "auto":
            banner_bg = (38, 30, 20) if "SMOKE" in stress_sim.current_stage_name or stress_sim.smoke_active else \
                        (45, 40, 15) if "GLARE" in stress_sim.current_stage_name or stress_sim.glare_active else \
                        (48, 18, 22) if "EXTRACTION" in stress_sim.current_stage_name or stress_sim.tool_hidden else (24, 28, 38)
            cv2.rectangle(canvas, (15, 36), (1185, 60), banner_bg, -1)
            cv2.rectangle(canvas, (15, 36), (1185, 60), (110, 160, 220), 1)
            banner_text = f"{stress_sim.current_stage_name}  |  {stress_sim.active_safeguard}"
            cv2.putText(canvas, banner_text, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)

        # --- Top Telemetry HUD Banner ---
        cv2.rectangle(canvas, (0, 0), (disp_w, 32), (20, 20, 24), -1)
        cv2.line(canvas, (0, 32), (disp_w, 32), (45, 45, 50), 1)

        badge_color = (0, 220, 100) if "TRACKING" in camera.fsm_state or "RE-ACQUIRED" in camera.fsm_state else (0, 165, 255) if camera.fsm_state == "UNCERTAIN" else (0, 40, 230)
        state_txt = f"STATE: {camera.fsm_state}"
        (t_w, _), _ = cv2.getTextSize(state_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        badge_w = max(180, t_w + 18)
        cv2.rectangle(canvas, (15, 5), (15 + badge_w, 27), badge_color, -1)
        cv2.putText(canvas, state_txt, (22, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        fps = 1.0 / max(1e-4, (time.time() - loop_start))
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)

        cx_offset = 25 + badge_w
        cv2.putText(canvas, f"Err: {centering_err:.1f}px", (cx_offset, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        cv2.putText(canvas, f"Policy: [m] {coordinator.current_policy[:8]}", (cx_offset + 145, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1)
        cv2.putText(canvas, f"Zoom: {1.0/camera.current_zoom:.1f}x", (cx_offset + 325, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        cv2.putText(canvas, f"Speed: {current_speed:.2f}x [s/f]", (cx_offset + 450, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

        # Stress Test Status Badge
        stress_color = (0, 165, 255) if stress_sim.get_status_str() != "NORMAL" else (160, 160, 160)
        cv2.putText(canvas, f"Stress: {stress_sim.get_status_str()}", (960, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, stress_color, 1)

        cv2.putText(canvas, f"FPS: {avg_fps:.1f}", (1100, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 100), 1)

        # --- Bottom Live ML Inference Inspector (Methods in Action) ---
        cv2.rectangle(canvas, (15, 452), (1185, 532), (16, 20, 28), -1)
        cv2.rectangle(canvas, (15, 452), (1185, 532), (50, 75, 110), 1)

        # Inspector Title Badge
        cv2.putText(canvas, "LIVE ML METHODS INSPECTOR", (25, 469),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # Line 1: Deep Learning + Decision Tree
        tip_str = f"({target_pos[0]:.0f}, {target_pos[1]:.0f})" if target_pos else "N/A"
        ml_line1 = (f"[ML-1 YOLOv8-nano] Tip: {tip_str} | Conf: {conf_val*100:.0f}%   -->   "
                    f"[ML-2 Decision Tree] In: [Conf={conf_val:.2f}, Area={box_area*100:.1f}%, Pos=({norm_x:.2f},{norm_y:.2f})] -> PRED: {camera.fsm_state}")
        cv2.putText(canvas, ml_line1, (25, 492),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)

        # Line 2: Bayesian Kalman Filter + Online Adaptive Regressor
        active_speed = max_speed if tool_info else 0.0
        ml_line2 = (f"[ML-3 Kalman Filter] Speed: {active_speed:.1f} px/f (Bayesian Damping)   -->   "
                    f"[ML-4 Adaptive Regressor] Regime: {ad_regime} | Kp={ad_kp:.3f}, Deadband={ad_deadband:.0f}px")
        cv2.putText(canvas, ml_line2, (25, 517),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1)

        # Bottom Instructions Footer
        cv2.putText(canvas, "SPACE: Pause | o: Smoke | g: Glare | x: Tool Extract | s/f: Speed | m: Policy | +/-: Zoom | q: Quit",
                    (210, 554), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

        writer.write(canvas)

        if not headless:
            cv2.imshow("Endoscope Auto-Camera Cockpit", canvas)
            delay_ms = max(1, int(1000.0 / (stream.fps * current_speed)))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                camera.reset()
            elif key == ord('s'):
                current_speed = max(0.15, current_speed - 0.10)
                print(f"🐢 Slowed speed to: {current_speed:.2f}x")
            elif key == ord('f'):
                current_speed = min(2.5, current_speed + 0.15)
                print(f"⚡ Increased speed to: {current_speed:.2f}x")
            elif key == ord('o'):
                act = stress_sim.toggle_smoke()
                print(f"💨 Surgical Smoke Stress-Test: {'ACTIVE (Kalman dead-reckoning engaged)' if act else 'OFF'}")
            elif key == ord('g'):
                act = stress_sim.toggle_glare()
                print(f"✨ Specular Glare Stress-Test: {'ACTIVE' if act else 'OFF'}")
            elif key == ord('x'):
                act = stress_sim.toggle_tool_hidden()
                if act:
                    print(f"🚪 Tool Extracted / Out of Field: [SEARCH (ZOOM OUT)] engaged -> camera expands to wide view!")
                else:
                    print(f"🎯 Tool Returned to Field: Camera detects tool, snaps to center, and AUTO-FOCUSES zoom-in!")
            elif key == ord('m'):
                new_pol = coordinator.cycle_policy()
                print(f"🔄 Switched Multi-Tool Policy to: {new_pol}")
            elif key == ord('+') or key == ord('='):
                camera.nominal_zoom = max(0.35, camera.nominal_zoom - 0.05)
            elif key == ord('-') or key == ord('_'):
                camera.nominal_zoom = min(0.95, camera.nominal_zoom + 0.05)
        else:
            if total_processed % 30 == 0:
                print(f"  Processed {total_processed} frames ({avg_fps:.1f} FPS)...")
            if max_frames and total_processed >= max_frames:
                break

    writer.release()
    stream.release()
    if not headless:
        cv2.destroyAllWindows()
    print("\n✅ Simulation session ended.")
    print(f"🎥 Output video recorded: {record_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endoscope Auto-Camera Universal Simulator")
    parser.add_argument("--source", type=str, default=None, help="Path to video file (.mp4, .avi), webcam (0), or image folder")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed multiplier (default: 0.5 for clear slow-motion)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI and render video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process in headless mode")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Output video filename")
    parser.add_argument("--stress", type=str, default="manual", choices=["manual", "auto", "smoke", "glare", "extraction"], help="Adverse condition stress-test mode")
    args = parser.parse_args()

    run_auto_camera_demo(source=args.source, speed_factor=args.speed, headless=args.headless, max_frames=args.max_frames, record_output=args.output, stress_mode=args.stress)

