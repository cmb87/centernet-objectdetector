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


class Track:
    """
    Represents an individual tracked object using a Kalman Filter.
    State vector: [x, y, u, v, w, h, w_dot, h_dot]^T
      - x, y: Bounding box center coordinates
      - u, v: Center coordinates velocities
      - w, h: Bounding box size (width and height)
      - w_dot, h_dot: Bounding box size rates of change
    """
    _id_count = 0

    def __init__(self, z, class_id, score):
        Track._id_count += 1
        self.track_id = Track._id_count
        self.class_id = class_id
        self.score = score
        
        # State vector: [x, y, u, v, w, h, w_dot, h_dot]^T
        self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3], 0.0, 0.0], dtype=np.float32)
        
        # State transition matrix F
        self.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],  # x = x + u
            [0, 1, 0, 1, 0, 0, 0, 0],  # y = y + v
            [0, 0, 1, 0, 0, 0, 0, 0],  # u = u
            [0, 0, 0, 1, 0, 0, 0, 0],  # v = v
            [0, 0, 0, 0, 1, 0, 1, 0],  # w = w + w_dot
            [0, 0, 0, 0, 0, 1, 0, 1],  # h = h + h_dot
            [0, 0, 0, 0, 0, 0, 1, 0],  # w_dot = w_dot
            [0, 0, 0, 0, 0, 0, 0, 1]   # h_dot = h_dot
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # We measure center x
            [0, 1, 0, 0, 0, 0, 0, 0],  # We measure center y
            [0, 0, 0, 0, 1, 0, 0, 0],  # We measure width w
            [0, 0, 0, 0, 0, 1, 0, 0]   # We measure height h
        ], dtype=np.float32)
        
        # Process noise covariance matrix Q
        # Lower process noise tells the filter that the true state transitions are very smooth.
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
        
        # Measurement noise covariance matrix R
        # Higher measurement noise tells the filter that raw bounding box detections are noisy,
        # prompting the Kalman Filter to heavily damp high-frequency frame-to-frame fluctuations.
        std_m_pos = 8.0
        std_m_sz = 15.0
        self.R = np.diag([
            std_m_pos**2, std_m_pos**2,
            std_m_sz**2, std_m_sz**2
        ]).astype(np.float32)
        
        # Initial covariance matrix P
        self.P = np.diag([
            std_m_pos**2, std_m_pos**2,
            10.0**2, 10.0**2,
            std_m_sz**2, std_m_sz**2,
            10.0**2, 10.0**2
        ]).astype(np.float32)
        
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.is_confirmed = False
        
        # Color configuration for premium visualization
        self.color = self._generate_color()
        
        # Track trail history (stores center x, y coordinates)
        self.history = [(int(z[0]), int(z[1]))]

    def _generate_color(self):
        # Golden ratio conjugate to spread hues evenly
        hue = (self.track_id * 137.5) % 360
        rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.85)
        # Convert RGB to BGR for OpenCV
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        self.age += 1
        
    def update(self, z, score=None):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            inv_S = np.linalg.inv(S)
            K = np.dot(np.dot(self.P, self.H.T), inv_S)
            self.x = self.x + np.dot(K, y)
            I = np.eye(len(self.x))
            self.P = np.dot(I - np.dot(K, self.H), self.P)
        except np.linalg.LinAlgError:
            pass  # Skips covariance update if singular to remain stable
            
        self.time_since_update = 0
        self.hits += 1
        if score is not None:
            self.score = score
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
        """
        Returns estimated center x, center y, width, height.
        """
        return [self.x[0], self.x[1], self.x[4], self.x[5]]


def sinkhorn_association(tracks, detections, max_iter=30, epsilon=1.0, eta=-4.75):
    """
    Associates tracks with detections using Sinkhorn's algorithm with dustbins.
    
    tracks: list of Track objects
    detections: list of dicts {'z': np.array([x, y, w, h]), 'class_id': int, 'score': float}
    """
    M = len(tracks)
    N = len(detections)
    
    if M == 0:
        return [], [], list(range(N))
    if N == 0:
        return [], list(range(M)), []
        
    # Construct score matrix S of size (M+1, N+1)
    S = np.zeros((M + 1, N + 1), dtype=np.float32)
    
    # Fill in S_ij for 0 <= i < M and 0 <= j < N
    for i in range(M):
        for j in range(N):
            # Class-based pruning: if classes don't match, penalize heavily
            if tracks[i].class_id != detections[j]['class_id']:
                S[i, j] = -50.0  # Safe penalty avoiding absolute float32 underflow
            else:
                d2 = tracks[i].mahalanobis_distance(detections[j]['z'])
                # Gating: if Mahalanobis distance is too large, reject match by penalizing
                if d2 > 9.49:  # Chi-squared 95% threshold for 4 degrees of freedom
                    S[i, j] = -50.0
                else:
                    S[i, j] = -0.5 * d2
                    
    # Fill in dustbin scores
    for i in range(M):
        S[i, N] = eta
    for j in range(N):
        S[M, j] = eta
    S[M, N] = 0.0
    
    # Compute similarity kernel K = exp(S / epsilon)
    K = np.exp(S / epsilon)
    
    # Target marginal distributions
    r = np.ones(M + 1, dtype=np.float32)
    r[-1] = N
    c = np.ones(N + 1, dtype=np.float32)
    c[-1] = M
    
    # Sinkhorn-Knopp iterations to scale matrix to meet marginal requirements
    u = np.ones(M + 1, dtype=np.float32)
    v = np.ones(N + 1, dtype=np.float32)
    
    for _ in range(max_iter):
        denom_v = np.dot(K.T, u)
        denom_v = np.where(denom_v == 0.0, 1e-12, denom_v)
        v = c / denom_v
        
        denom_u = np.dot(K, v)
        denom_u = np.where(denom_u == 0.0, 1e-12, denom_u)
        u = r / denom_u
        
    P = u[:, None] * K * v[None, :]  # Double stochastic-like joint probability matrix
    
    matches = []
    unmatched_tracks = list(range(M))
    unmatched_detections = list(range(N))
    
    # Retrieve associations based on mutual maximums and dustbin comparisons
    for i in range(M):
        row_max_idx = np.argmax(P[i, :])
        if row_max_idx < N:  # Did not match dustbin column
            col_max_idx = np.argmax(P[:, row_max_idx])
            if col_max_idx == i:  # Mutual argmax
                # Must be more likely than dustbins and exceed a confidence threshold
                if (P[i, row_max_idx] > P[i, N] and 
                    P[i, row_max_idx] > P[M, row_max_idx] and 
                    P[i, row_max_idx] > 0.5):
                    
                    matches.append((i, row_max_idx))
                    if i in unmatched_tracks:
                        unmatched_tracks.remove(i)
                    if row_max_idx in unmatched_detections:
                        unmatched_detections.remove(row_max_idx)
                        
    return matches, unmatched_tracks, unmatched_detections


def non_max_suppression(detections, iou_threshold=0.5, min_center_dist=10.0):
    """
    Applies Non-Maximum Suppression (NMS) to detections based on IoU and center distance.
    """
    if len(detections) == 0:
        return []
        
    # Sort detections by score descending
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep_detections = []
    for det in detections:
        box = det['z']  # [center_x, center_y, w, h]
        
        duplicate = False
        for keep_det in keep_detections:
            keep_box = keep_det['z']
            
            # Check center distance
            dist = np.sqrt((box[0] - keep_box[0])**2 + (box[1] - keep_box[1])**2)
            if dist < min_center_dist:
                duplicate = True
                break
                
            # Check IoU
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


class Tracker:
    """
    Manages active Kalman Filter tracks, performs associations, and handles lifecycle.
    """
    def __init__(self, max_age=15, n_init=3):
        self.max_age = max_age
        self.n_init = n_init
        self.tracks = []

    def update(self, detections):
        """
        detections: list of dicts with keys 'z', 'class_id', 'score'
        """
        # 1. Predict all tracks
        for track in self.tracks:
            track.predict()

        # 2. Match existing tracks with detections using Sinkhorn
        matches, unmatched_tracks, unmatched_detections = sinkhorn_association(
            self.tracks, detections, max_iter=30, epsilon=1.0, eta=-4.75
        )

        # 3. Update matched tracks
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det['z'], det['score'])
            if not self.tracks[t_idx].is_confirmed and self.tracks[t_idx].hits >= self.n_init:
                self.tracks[t_idx].is_confirmed = True

        # 4. Initialize new tentative tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(det['z'], det['class_id'], det['score'])
            if self.n_init <= 1:
                new_track.is_confirmed = True
            self.tracks.append(new_track)

        # 5. Purge dead tracks
        # Tentative tracks that miss even one update are immediately deleted.
        # Confirmed tracks are deleted if they miss more than max_age updates.
        self.tracks = [
            t for t in self.tracks
            if (t.is_confirmed and t.time_since_update <= self.max_age) or 
               (not t.is_confirmed and t.time_since_update == 0)
        ]

        # 6. Remove duplicate tracks (to handle multi-box tracking anomalies)
        self.remove_duplicate_tracks()

        # Return only confirmed tracks to be displayed
        return [t for t in self.tracks if t.is_confirmed]

    def predict_only(self):
        """
        Advances the state projection of all active tracks without checking detections.
        Useful for multi-rate / skipped frame tracking on edge devices.
        """
        for track in self.tracks:
            track.predict()
            track.history.append((int(track.x[0]), int(track.x[1])))
            if len(track.history) > 30:
                track.history.pop(0)

        # Purge tracks that exceed max_age during skipped frames.
        # Tentative tracks are kept during skipped frames so they can be associated when detections resume.
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # Return only confirmed tracks to be displayed
        return [t for t in self.tracks if t.is_confirmed]


    def remove_duplicate_tracks(self, iou_threshold=0.7, min_center_dist=15.0):
        """
        Removes duplicate/overlapping tracks to prevent multiple boxes tracking the same target.
        """
        if len(self.tracks) < 2:
            return

        # Sort tracks so we prioritize keeping confirmed, older, and more stable tracks
        sorted_indices = sorted(
            range(len(self.tracks)),
            key=lambda k: (self.tracks[k].is_confirmed, self.tracks[k].hits, self.tracks[k].age),
            reverse=True
        )

        keep_indices = []
        for idx in sorted_indices:
            track = self.tracks[idx]
            box = track.get_box()  # [center_x, center_y, w, h]

            duplicate = False
            for keep_idx in keep_indices:
                keep_track = self.tracks[keep_idx]
                keep_box = keep_track.get_box()

                # Check center Euclidean distance
                dist = np.sqrt((box[0] - keep_box[0])**2 + (box[1] - keep_box[1])**2)
                if dist < min_center_dist:
                    duplicate = True
                    break

                # Check IoU
                if self._calculate_iou(box, keep_box) > iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                keep_indices.append(idx)

        self.tracks = [self.tracks[idx] for idx in keep_indices]

    def _calculate_iou(self, box1, box2):
        """
        Calculates IoU of two boxes in [center_x, center_y, w, h] format.
        """
        b1_x1 = box1[0] - 0.5 * box1[2]
        b1_y1 = box1[1] - 0.5 * box1[3]
        b1_x2 = box1[0] + 0.5 * box1[2]
        b1_y2 = box1[1] + 0.5 * box1[3]

        b2_x1 = box2[0] - 0.5 * box2[2]
        b2_y1 = box2[1] - 0.5 * box2[3]
        b2_x2 = box2[0] + 0.5 * box2[2]
        b2_y2 = box2[1] + 0.5 * box2[3]

        ix1 = max(b1_x1, b2_x1)
        iy1 = max(b1_y1, b2_y1)
        ix2 = min(b1_x2, b2_x2)
        iy2 = min(b1_y2, b2_y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union



def center_crop_square(frame):
    """
    Crops the frame to a centered square (removes sides if width > height).
    """
    h, w, _ = frame.shape
    if w == h:
        return frame  # already square

    if w > h:
        offset = (w - h) // 2
        cropped = frame[:, offset:offset + h]
    else:
        offset = (h - w) // 2
        cropped = frame[offset:offset + w, :]

    return cropped


def draw_premium_track(img, track):
    """
    Draws a track bounding box, a filled background label, and its historic trail.
    """
    b = track.get_box()
    x1 = int(b[0] - 0.5 * b[2])
    y1 = int(b[1] - 0.5 * b[3])
    x2 = int(b[0] + 0.5 * b[2])
    y2 = int(b[1] + 0.5 * b[3])
    
    color = track.color
    
    # 1. Draw historical path trail
    for idx in range(1, len(track.history)):
        pt1 = track.history[idx - 1]
        pt2 = track.history[idx]
        thickness = int(1 + (idx / len(track.history)) * 3)
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        
    # 2. Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # 3. Draw premium text label with solid color background
    label = f"ID {track.track_id} | C{track.class_id} | {track.score*100:.0f}%"
    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    
    # Background rectangle for text
    cv2.rectangle(img, (x1, y1 - h_text - 6), (x1 + w_text + 4, y1), color, -1)
    # White overlay text
    cv2.putText(img, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


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
    # 1. Setup model dimensions and path configurations
    ih, iw, ic = cfg.model.input_size
    nc = cfg.dataset.class_num
    conf_thres = cfg.get("conf_thres", 0.15)

    print(f"Model Configuration: {cfg.model.name}")
    print(f"Input Size: {ih}x{iw}x{ic}")
    print(f"Number of classes: {nc}")
    print(f"Confidence Threshold: {conf_thres}")

    # 2. Instantiate Selected Backbone Model
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
        raise ValueError(f"Unknown backbone model name: {cfg.model.name}")

    # 3. Build CenterNet multi-task output heads
    xhead1 = Conv2D(nc, (1, 1), padding="same", use_bias=True, activation="sigmoid", name="head1_heatmap")(backbone.output)
    xhead2 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head2_box_size")(backbone.output)
    xhead3 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head3_offset")(backbone.output)

    yhead = Concatenate(axis=-1, name="head_final")([xhead1, xhead2, xhead3])
    model = Model(inputs=backbone.inputs, outputs=yhead, name=f"CenterNet_{cfg.model.name}")

    # 4. Prompt user for checkpoint file path
    print("\n" + "=" * 60)
    print("Please place your trained .weights.h5 file path here.")
    print("Example: ./runs/detect/shufflenet_run/version_0/checkpoints/best_model.weights.h5")
    print("=" * 60)
    
    # Locate best candidate weight file dynamically if it exists
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

    # Load weights
    model.load_weights(weights_path)
    print(f"Loaded weights successfully from: {weights_path}")

    # 5. Get input image or video path
    if CLI_INPUT_SOURCE:
        input_source = CLI_INPUT_SOURCE
        print(f"Processing input source: {input_source}")
    else:
        input_source = input("\nEnter image or video path to run inference on: ").strip()
        input_source = os.path.expanduser(input_source)

    if not os.path.exists(input_source):
        print(f"Error: Input source '{input_source}' does not exist.")
        return

    # Determine if input is video or image
    is_video = input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'))

    # Initialize tracker
    # For image, n_init=1 since there is only a single frame to observe
    # For video, n_init=3 ensures temporal consistency and filters out noise
    tracker = Tracker(max_age=15, n_init=1 if not is_video else 3)

    if not is_video:
        # Image Inference
        print("Running filtered inference on image...")
        img_bgr = cv2.imread(input_source)
        img_cropped = center_crop_square(img_bgr)
        img_resized = cv2.resize(img_cropped, (iw, ih))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        # Predict
        raw_output = model.predict(img_input)
        score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
        score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

        # Gather detections
        detections = []
        for s, c, b, w in zip(score, classes, bc, wh):
            if s > conf_thres:
                detections.append({
                    'z': np.array([b[0], b[1], w[0], w[1]], dtype=np.float32),
                    'class_id': int(c),
                    'score': float(s)
                })

        # Apply Non-Maximum Suppression to detections
        detections = non_max_suppression(detections, iou_threshold=0.5, min_center_dist=100.0)

        # Track and associate
        active_tracks = tracker.update(detections)

        # Draw active tracks
        for track in active_tracks:
            draw_premium_track(img_resized, track)

        output_path = CLI_OUTPUT_PATH if CLI_OUTPUT_PATH else "inference_filtered_output.jpg"
        cv2.imwrite(output_path, img_resized)
        print(f"Filtered inference complete! Saved output visualization to: {output_path}")

    else:
        # Video Inference
        print("Running filtered inference on video...")
        cap = cv2.VideoCapture(input_source)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        out_filename = CLI_OUTPUT_PATH if CLI_OUTPUT_PATH else "inference_filtered_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_filename, fourcc, fps, (iw, ih))

        skip_frames = cfg.get("skip_frames", 1)
        print(f"Tracking configured with skip_frames={skip_frames} (detection once every {skip_frames} frames)")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_cropped = center_crop_square(frame)
            frame_resized = cv2.resize(frame_cropped, (iw, ih))

            # Multi-rate: run expensive neural network detection only every N frames
            run_detection = (frame_idx % skip_frames == 0)

            if run_detection:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_input = np.expand_dims(frame_rgb, axis=0).astype(np.float32) / 255.0

                raw_output = model.predict(img_input, verbose=0)
                score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
                score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

                # Gather detections
                detections = []
                for s, c, b, w in zip(score, classes, bc, wh):
                    if s > conf_thres:
                        detections.append({
                            'z': np.array([b[0], b[1], w[0], w[1]], dtype=np.float32),
                            'class_id': int(c),
                            'score': float(s)
                        })

                # Apply Non-Maximum Suppression to detections
                detections = non_max_suppression(detections, iou_threshold=0.5, min_center_dist=10.0)

                # Track and associate (performs predict, matching, and measurement update)
                active_tracks = tracker.update(detections)
            else:
                # Prediction-only tracking (extremely lightweight, perfect for real-time edge devices)
                active_tracks = tracker.predict_only()

            # Draw active tracks
            for track in active_tracks:
                draw_premium_track(frame_resized, track)

            if CLI_GUI:
                try:
                    cv2.imshow("Filtered Tracker", frame_resized)
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
        print(f"Filtered inference complete! Saved output video to: {out_filename}")


if __name__ == "__main__":
    parse_cli_args()
    main()
