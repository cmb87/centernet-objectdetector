import os
import sys
import numpy as np
import cv2
import colorsys
import tensorflow as tf
import hydra
from omegaconf import DictConfig
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import Model

# Ensure dependencies can be loaded properly
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(root_dir)

from src.data.labels import decode


class SeekerState:
    SEARCHING = "SEARCHING"
    LOCKING = "LOCKING"
    LOCKED = "LOCKED"
    COASTING = "COASTING"


class SeekerTrack:
    """
    State tracking of a single target under the seeker's gaze.
    State vector: [x, y, u, v, w, h, w_dot, h_dot]^T
    """
    def __init__(self, z, class_id):
        self.track_id = np.random.randint(100, 999)  # Assign a unique 3-digit transponder ID
        self.class_id = class_id
        
        # State: [x, y, u, v, w, h, w_dot, h_dot]^T
        self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3], 0.0, 0.0], dtype=np.float32)
        
        # Transition F
        self.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process Noise Q (tightened for smooth predictions)
        std_pos = 0.5
        std_vel = 0.1
        std_sz = 0.1
        std_sz_vel = 0.01
        self.Q = np.diag([
            std_pos**2, std_pos**2,
            std_vel**2, std_vel**2,
            std_sz**2, std_sz**2,
            std_sz_vel**2, std_sz_vel**2
        ]).astype(np.float32)
        
        # Measurement Noise R (higher to ignore high-frequency detection jitter)
        std_m_pos = 8.0
        std_m_sz = 15.0
        self.R = np.diag([
            std_m_pos**2, std_m_pos**2,
            std_m_sz**2, std_m_sz**2
        ]).astype(np.float32)
        
        # Initial Covariance P
        self.P = np.diag([
            std_m_pos**2, std_m_pos**2,
            10.0**2, 10.0**2,
            std_m_sz**2, std_m_sz**2,
            10.0**2, 10.0**2
        ]).astype(np.float32)
        
        self.history = []

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            inv_S = np.linalg.inv(S)
            K = np.dot(np.dot(self.P, self.H.T), inv_S)
            self.x = self.x + np.dot(K, y)
            I = np.eye(len(self.x))
            self.P = np.dot(I - np.dot(K, self.H), self.P)
        except np.linalg.LinAlgError:
            pass
            
        self.history.append((int(self.x[0]), int(self.x[1])))
        if len(self.history) > 30:
            self.history.pop(0)
            
    def mahalanobis_distance(self, z):
        z_pred = np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        y = z - z_pred
        try:
            inv_S = np.linalg.inv(S)
            d2 = np.dot(np.dot(y.T, inv_S), y)
        except np.linalg.LinAlgError:
            d2 = 1e5
        return d2

    def get_box(self):
        return [self.x[0], self.x[1], self.x[4], self.x[5]]


class SeekerTracker:
    """
    Manages the Seeker State Machine, lock-on logic, and Nearest Neighbor association.
    """
    def __init__(self, max_age=15, n_init=3, center_weight=0.005):
        self.max_age = max_age
        self.n_init = n_init
        self.center_weight = center_weight
        
        self.state = SeekerState.SEARCHING
        self.track = None
        self.hits = 0
        self.misses = 0

    def update(self, detections, img_center):
        """
        detections: list of dicts with keys 'z', 'class_id', 'score'
        img_center: (cx, cy) of the optical boresight
        """
        if self.state == SeekerState.SEARCHING:
            # 1. Search Mode: Pick the target closest to the boresight center
            if len(detections) > 0:
                best_idx = -1
                best_score = -1e5
                for idx, det in enumerate(detections):
                    z = det['z']
                    dist = np.sqrt((z[0] - img_center[0])**2 + (z[1] - img_center[1])**2)
                    # Boresight score prioritizes high confidence and close centering
                    score = det['score'] - self.center_weight * dist
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx != -1:
                    det = detections[best_idx]
                    self.track = SeekerTrack(det['z'], det['class_id'])
                    self.state = SeekerState.LOCKING
                    self.hits = 1
                    self.misses = 0
            return None

        # Predict forward
        self.track.predict()

        # 2. Nearest Neighbor Association using Mahalanobis distance
        best_det_idx = -1
        min_d2 = 9.49  # Chi-Squared 95% threshold for 4 degrees of freedom
        
        for idx, det in enumerate(detections):
            # Only match if classes are identical
            if det['class_id'] == self.track.class_id:
                d2 = self.track.mahalanobis_distance(det['z'])
                if d2 < min_d2:
                    min_d2 = d2
                    best_det_idx = idx

        # 3. State Transitions and Actions
        if best_det_idx != -1:
            # Match Found!
            self.track.update(detections[best_det_idx]['z'])
            
            if self.state == SeekerState.LOCKING:
                self.hits += 1
                if self.hits >= self.n_init:
                    self.state = SeekerState.LOCKED
                    self.misses = 0
            elif self.state in [SeekerState.LOCKED, SeekerState.COASTING]:
                self.state = SeekerState.LOCKED
                self.misses = 0
        else:
            # Match NOT Found (Target missing)
            if self.state == SeekerState.LOCKING:
                # Discard lock candidate instantly if not tracked during lock phase
                self.state = SeekerState.SEARCHING
                self.track = None
            elif self.state == SeekerState.LOCKED:
                self.state = SeekerState.COASTING
                self.misses = 1
            elif self.state == SeekerState.COASTING:
                self.misses += 1
                if self.misses > self.max_age:
                    # Lock fully lost, return to searching
                    self.state = SeekerState.SEARCHING
                    self.track = None

        return self.track


def center_crop_square(frame):
    """
    Crops the frame to a centered square.
    """
    h, w, _ = frame.shape
    if w == h:
        return frame

    if w > h:
        offset = (w - h) // 2
        cropped = frame[:, offset:offset + h]
    else:
        offset = (h - w) // 2
        cropped = frame[offset:offset + w, :]

    return cropped


def draw_seeker_hud(img, state, track, misses, max_age):
    """
    Draws an advanced, high-tech Seeker Heads-Up Display (HUD) reticle overlay.
    """
    ih, iw, _ = img.shape
    cx, cy = iw // 2, ih // 2
    
    # 1. Color Theme based on Seeker State
    if state == SeekerState.SEARCHING:
        color = (0, 165, 255)  # Orange (Scanning)
        status_text = "MODE: [ACQUIRING SCAN]"
    elif state == SeekerState.LOCKING:
        color = (0, 255, 255)  # Yellow (Locking)
        status_text = f"MODE: [LOCK ACQUISITION - VERIFYING HITS]"
    elif state == SeekerState.LOCKED:
        color = (0, 255, 0)  # Lime Green (Solid Lock)
        status_text = "MODE: [TARGET ACQUIRED - ACTIVE LOCK]"
    else:  # COASTING
        color = (0, 0, 255)  # Red (Coasting / Target Lost)
        status_text = f"MODE: [SIGNAL LOSS - COASTING ({misses}/{max_age})]"

    # 2. Draw Boresight Reticle (Center circular crosshair)
    # Outer ring
    cv2.circle(img, (cx, cy), 40, color, 1, cv2.LINE_AA)
    # Inner center dot
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)
    # Boresight ticks
    cv2.line(img, (cx - 55, cy), (cx - 45, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 45, cy), (cx + 55, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 55), (cx, cy - 45), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 45), (cx, cy + 55), color, 1, cv2.LINE_AA)

    # 3. Draw Locked Target brackets and telemetry
    if track is not None:
        b = track.get_box()
        tx, ty = int(b[0]), int(b[1])
        tw, th = int(b[2]), int(b[3])
        x1, y1 = tx - tw // 2, ty - th // 2
        x2, y2 = tx + tw // 2, ty + th // 2
        
        # Draw Boresight Vector line (Error command output)
        cv2.line(img, (cx, cy), (tx, ty), color, 1, cv2.LINE_AA)
        
        # Draw bracket corners (no full box for high-tech aesthetic)
        bracket_len = min(15, tw // 3, th // 3)
        # Top-Left Corner
        cv2.line(img, (x1, y1), (x1 + bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Top-Right Corner
        cv2.line(img, (x2, y1), (x2 - bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Left Corner
        cv2.line(img, (x1, y2), (x1 + bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Right Corner
        cv2.line(img, (x2, y2), (x2 - bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - bracket_len), color, 2, cv2.LINE_AA)
        
        # Boresight tracking errors (displacement coordinates)
        dx = tx - cx
        dy = ty - cy
        
        # Draw target details near target box
        lbl = f"ID: TR-{track.track_id} [C{track.class_id}]"
        cv2.putText(img, lbl, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        
        # Telemetry Display Overlay (Top Left corner)
        cv2.putText(img, "=== SEEKER TELEMETRY ===", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(img, status_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"ACQUIRED TARGET: TR-{track.track_id}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"COORDINATES:     X={tx} px | Y={ty} px", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"DIMENSIONS:      W={tw} px | H={th} px", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"BORESIGHT ERROR: DX={dx:+d} px | DY={dy:+d} px", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        
    else:
        # Telemetry Display Overlay when searching
        cv2.putText(img, "=== SEEKER TELEMETRY ===", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(img, status_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        cv2.putText(img, "SIGNAL LOCK:     NO TARGET LOCK-ON", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def non_max_suppression(detections, iou_threshold=0.5, min_center_dist=10.0):
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep_detections = []
    for det in detections:
        box = det['z']
        duplicate = False
        for keep_det in keep_detections:
            keep_box = keep_det['z']
            dist = np.sqrt((box[0] - keep_box[0])**2 + (box[1] - keep_box[1])**2)
            if dist < min_center_dist:
                duplicate = True
                break
            b1_x1, b1_y1 = box[0] - 0.5 * box[2], box[1] - 0.5 * box[3]
            b1_x2, b1_y2 = box[0] + 0.5 * box[2], box[1] + 0.5 * box[3]
            b2_x1, b2_y1 = keep_box[0] - 0.5 * keep_box[2], keep_box[1] - 0.5 * keep_box[3]
            b2_x2, b2_y2 = keep_box[0] + 0.5 * keep_box[2], keep_box[1] + 0.5 * keep_box[3]
            
            ix1, iy1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
            ix2, iy2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            intersection = iw * ih
            union = (box[2] * box[3]) + (keep_box[2] * keep_box[3]) - intersection
            if union > 0 and (intersection / union) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep_detections.append(det)
    return keep_detections


# CLI overrides extracted before Hydra initialization
CLI_INPUT_SOURCE = None
CLI_WEIGHTS_PATH = None
CLI_OUTPUT_PATH = None
CLI_GUI = False
CLI_MAX_FRAMES = 0
CLI_VERSION = None

def parse_cli_args():
    global CLI_INPUT_SOURCE, CLI_WEIGHTS_PATH, CLI_OUTPUT_PATH, CLI_GUI, CLI_MAX_FRAMES, CLI_VERSION
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--weights", "-w") and i + 1 < len(sys.argv):
            CLI_WEIGHTS_PATH = os.path.expanduser(sys.argv[i + 1])
            i += 2
        elif arg in ("--version", "-v") and i + 1 < len(sys.argv):
            CLI_VERSION = sys.argv[i + 1]
            i += 2
        elif arg in ("--output", "-o") and i + 1 < len(sys.argv):
            CLI_OUTPUT_PATH = os.path.expanduser(sys.argv[i + 1])
            i += 2
        elif arg == "--max-frames" and i + 1 < len(sys.argv):
            CLI_MAX_FRAMES = int(sys.argv[i + 1])
            i += 2
        elif arg == "--gui":
            CLI_GUI = True
            i += 1
        elif not arg.startswith("-") and "=" not in arg:
            # Positional argument (video or image input path)
            CLI_INPUT_SOURCE = os.path.expanduser(arg)
            i += 1
        else:
            new_argv.append(arg)
            i += 1
    sys.argv = new_argv


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    ih, iw, ic = cfg.model.input_size
    nc = cfg.dataset.class_num
    conf_thres = cfg.get("conf_thres", 0.15)
    skip_frames = cfg.get("skip_frames", 1)

    print(f"Seeker Configuration: {cfg.model.name}")
    print(f"Target Resolution:    {iw}x{ih}")
    print(f"Boresight Skip Frame: {skip_frames}")

    # Build backbone
    if cfg.model.name == "shufflenet":
        from src.backends.shufflenet import Shuffle_Net
        backbone = Shuffle_Net(
            start_channels=cfg.model.start_channels, 
            groups=cfg.model.groups, 
            input_shape=(ih, iw, ic), 
            nf=cfg.model.num_features
        )
    elif cfg.model.name == "mobilenet_pretrained":
        from src.backends.mobilenetv3 import MobileNetV3_Small_CenterNet
        backbone = MobileNetV3_Small_CenterNet(
            input_shape=(ih, iw, ic), 
            num_features=cfg.model.num_features
        )
    else:
        raise ValueError(f"Unknown backbone: {cfg.model.name}")

    xhead1 = Conv2D(nc, (1, 1), padding="same", use_bias=True, activation="sigmoid", name="head1_heatmap")(backbone.output)
    xhead2 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head2_box_size")(backbone.output)
    xhead3 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head3_offset")(backbone.output)

    yhead = Concatenate(axis=-1, name="head_final")([xhead1, xhead2, xhead3])
    model = Model(inputs=backbone.inputs, outputs=yhead, name=f"CenterNet_{cfg.model.name}")

    # weights path
    candidate_dir = f"./runs/{cfg.task.task}/{cfg.model.name}_{cfg.name}"
    default_weights_path = ""
    if os.path.isdir(candidate_dir):
        if CLI_VERSION:
            v_clean = str(CLI_VERSION).lower().replace("version_", "").replace("v", "")
            target_v = f"version_{v_clean}"
            v_candidate = os.path.join(candidate_dir, target_v, "checkpoints", "best_model.weights.h5")
            if os.path.exists(v_candidate):
                default_weights_path = v_candidate
            else:
                v_final = os.path.join(candidate_dir, target_v, f"weights_{cfg.model.name}_final.weights.h5")
                if os.path.exists(v_final):
                    default_weights_path = v_final
        else:
            versions = [v for v in os.listdir(candidate_dir) if v.startswith("version_")]
            if versions:
                latest_version = max(versions, key=lambda v: int(v.split("_")[1]) if v.split("_")[1].isdigit() else -1)
                candidate_path = os.path.join(candidate_dir, latest_version, "checkpoints", "best_model.weights.h5")
                if os.path.exists(candidate_path):
                    default_weights_path = candidate_path

    print("\n" + "="*60)
    print("SEEKER CONTROLLER SYSTEM STARTUP")
    print("="*60)

    if CLI_WEIGHTS_PATH:
        v_clean = CLI_WEIGHTS_PATH.lower().replace("version_", "").replace("v", "")
        shorthand_candidate = os.path.join(candidate_dir, f"version_{v_clean}", "checkpoints", "best_model.weights.h5")
        if v_clean.isdigit() and os.path.exists(shorthand_candidate):
            weights_path = shorthand_candidate
        else:
            weights_path = CLI_WEIGHTS_PATH
    elif (CLI_INPUT_SOURCE or CLI_VERSION) and default_weights_path:
        weights_path = default_weights_path
        print(f"Using model weights: {weights_path}")
    else:
        weights_path = input(f"Weights file path [{default_weights_path}]: ").strip()
        if not weights_path:
            weights_path = default_weights_path

    if not weights_path or not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' does not exist.")
        return

    model.load_weights(weights_path)
    print("Loaded weights successfully.")

    if CLI_INPUT_SOURCE:
        input_source = CLI_INPUT_SOURCE
        print(f"Processing input source: {input_source}")
    else:
        input_source = input("\nEnter image or video path to run seeker on: ").strip()
        input_source = os.path.expanduser(input_source)

    if not os.path.exists(input_source):
        print(f"Error: '{input_source}' does not exist.")
        return

    is_video = input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'))

    # Instantiate Seeker Tracker
    # Max coasting age is 15 frames, requires 3 validation frames to lock.
    seeker = SeekerTracker(max_age=15, n_init=1 if not is_video else 3)
    img_center = (iw // 2, ih // 2)

    if not is_video:
        # Image Mode (Search and lock first target)
        print("Running seeker in single-image acquisition mode...")
        img_bgr = cv2.imread(input_source)
        img_cropped = center_crop_square(img_bgr)
        img_resized = cv2.resize(img_cropped, (iw, ih))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        raw_output = model.predict(img_input)
        score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
        score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

        detections = []
        for s, c, b, w in zip(score, classes, bc, wh):
            if s > conf_thres:
                detections.append({
                    'z': np.array([b[0], b[1], w[0], w[1]], dtype=np.float32),
                    'class_id': int(c),
                    'score': float(s)
                })

        detections = non_max_suppression(detections, iou_threshold=0.5, min_center_dist=10.0)
        
        # Update twice to simulate instant lock validation for image display
        seeker.update(detections, img_center)
        locked_track = seeker.update(detections, img_center)

        draw_seeker_hud(img_resized, seeker.state, locked_track, seeker.misses, seeker.max_age)

        output_path = "seeker_hud_output.jpg"
        cv2.imwrite(output_path, img_resized)
        print(f"Seeker output saved to: {output_path}")

    else:
        # Video Mode (Full real-time tracking with prediction-only coasting/skipping)
        print("Running seeker on video feed...")
        cap = cv2.VideoCapture(input_source)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        out_filename = CLI_OUTPUT_PATH if CLI_OUTPUT_PATH else "seeker_hud_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_filename, fourcc, fps, (iw, ih))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_cropped = center_crop_square(frame)
            frame_resized = cv2.resize(frame_cropped, (iw, ih))

            run_detection = (frame_idx % skip_frames == 0) or (seeker.state == SeekerState.SEARCHING)

            if run_detection:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_input = np.expand_dims(frame_rgb, axis=0).astype(np.float32) / 255.0

                raw_output = model.predict(img_input, verbose=0)
                score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
                score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

                detections = []
                for s, c, b, w in zip(score, classes, bc, wh):
                    if s > conf_thres:
                        detections.append({
                            'z': np.array([b[0], b[1], w[0], w[1]], dtype=np.float32),
                            'class_id': int(c),
                            'score': float(s)
                        })

                detections = non_max_suppression(detections, iou_threshold=0.5, min_center_dist=10.0)
                locked_track = seeker.update(detections, img_center)
            else:
                # Coast / Predict only when skipping frames (insanely fast!)
                if seeker.track is not None:
                    seeker.track.predict()
                    # Check for signal lost timeouts on skipped frames
                    if seeker.state == SeekerState.COASTING:
                        seeker.misses += 1
                        if seeker.misses > seeker.max_age:
                            seeker.state = SeekerState.SEARCHING
                            seeker.track = None
                    elif seeker.state == SeekerState.LOCKED:
                        # On skip frames, increment age but don't drop lock unless missing
                        pass
                locked_track = seeker.track

            # Render Military HUD
            draw_seeker_hud(frame_resized, seeker.state, locked_track, seeker.misses, seeker.max_age)

            if CLI_GUI:
                try:
                    cv2.imshow("Seeker HUD", frame_resized)
                    if cv2.waitKey(1) == 27:
                        break
                except cv2.error:
                    pass

            out_writer.write(frame_resized)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

            if CLI_MAX_FRAMES > 0 and frame_idx >= CLI_MAX_FRAMES:
                break

        cap.release()
        out_writer.release()
        if CLI_GUI:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"Seeker tracking complete! Saved output video to: {out_filename}")


if __name__ == "__main__":
    parse_cli_args()
    main()
