import os
import sys
import numpy as np
import cv2
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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Setup model dimensions and path configurations
    ih, iw, ic = cfg.model.input_size
    nc = cfg.dataset.class_num

    print(f"Model Configuration: {cfg.model.name}")
    print(f"Input Size: {ih}x{iw}x{ic}")
    print(f"Number of classes: {nc}")

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
        versions = [v for v in os.listdir(candidate_dir) if v.startswith("version_")]
        if versions:
            latest_version = max(versions, key=lambda v: int(v.split("_")[1]) if v.split("_")[1].isdigit() else -1)
            candidate_path = os.path.join(candidate_dir, latest_version, "checkpoints", "best_model.weights.h5")
            if os.path.exists(candidate_path):
                default_weights_path = candidate_path

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
    input_source = input("\nEnter image or video path to run inference on: ").strip()
    if not os.path.exists(input_source):
        print(f"Error: Input source '{input_source}' does not exist.")
        return

    # Determine if input is video or image
    is_video = input_source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'))

    if not is_video:
        # Image Inference
        print("Running inference on image...")
        img_bgr = cv2.imread(input_source)
        img_cropped = center_crop_square(img_bgr)
        img_resized = cv2.resize(img_cropped, (iw, ih))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        # Predict
        raw_output = model.predict(img_input)
        score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
        score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

        # Draw detections
        for s, c, b, w in zip(score, classes, bc, wh):
            if s > 0.15: # Confidence threshold
                x1 = int(b[0] - 0.5 * w[0])
                y1 = int(b[1] - 0.5 * w[1])
                x2 = int(b[0] + 0.5 * w[0])
                y2 = int(b[1] + 0.5 * w[1])
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_resized, f"C{c} {s*100:.1f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        output_path = "inference_output.jpg"
        cv2.imwrite(output_path, img_resized)
        print(f"Inference complete! Saved output visualization to: {output_path}")

    else:
        # Video Inference
        print("Running inference on video...")
        cap = cv2.VideoCapture(input_source)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter("inference_output.mp4", fourcc, fps, (iw, ih))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_cropped = center_crop_square(frame)
            frame_resized = cv2.resize(frame_cropped, (iw, ih))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(frame_rgb, axis=0).astype(np.float32) / 255.0

            raw_output = model.predict(img_input, verbose=0)
            score, classes, bc, wh = decode(raw_output, iw=iw, ih=ih, K=50)
            score, classes, bc, wh = score.numpy()[0], classes.numpy()[0], bc.numpy()[0], wh.numpy()[0]

            for s, c, b, w in zip(score, classes, bc, wh):
                if s > 0.15:
                    x1 = int(b[0] - 0.5 * w[0])
                    y1 = int(b[1] - 0.5 * w[1])
                    x2 = int(b[0] + 0.5 * w[0])
                    y2 = int(b[1] + 0.5 * w[1])
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_resized, f"C{c} {s*100:.1f}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            out_writer.write(frame_resized)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        out_writer.release()
        print("Inference complete! Saved output video to: inference_output.mp4")


if __name__ == "__main__":
    main()
