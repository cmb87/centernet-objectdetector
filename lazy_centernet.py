import os
import sys

# --- GPU Bootstrapping for pip-installed NVIDIA/CUDA packages ---
# If python is run directly without sourcing activate first, LD_LIBRARY_PATH is not set.
# We dynamically locate the venv's nvidia libraries and re-execute python to load them.
venv_path = sys.prefix
nvidia_dir = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
if os.path.isdir(nvidia_dir):
    cuda_paths = []
    for pkg in os.listdir(nvidia_dir):
        pkg_lib = os.path.join(nvidia_dir, pkg, "lib")
        if os.path.isdir(pkg_lib):
            cuda_paths.append(pkg_lib)
    if cuda_paths:
        old_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if not all(p in old_ld_path for p in cuda_paths):
            new_ld_path = ":".join(cuda_paths) + (f":{old_ld_path}" if old_ld_path else "")
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
            print(f"[GPU Bootstrap] Configured library paths and restarting script...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

from datetime import datetime

# Enable GPU Memory growth and visibility before loading TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Ensure the 'src' directory is in python search path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(root_dir)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate
from tensorflow.keras import Model

import hydra
from omegaconf import DictConfig

from src.data.datapipe import Datapipe
from src.losses import centerNetLoss, cls_loss, reg_loss, wh_loss, offset_loss
from src.callbacks import DrawImageCallback

def get_next_version_dir(base_dir: str) -> str:
    """
    Scans the base directory and returns the path to the next available 'version_X' directory.
    If 'version_0' and 'version_1' exist, returns path to 'version_2'.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_versions = []
    for d in os.listdir(base_dir):
        if d.startswith("version_") and os.path.isdir(os.path.join(base_dir, d)):
            try:
                v_num = int(d.split("_")[1])
                existing_versions.append(v_num)
            except (IndexError, ValueError):
                pass
    next_v = max(existing_versions) + 1 if existing_versions else 0
    version_dir = os.path.join(base_dir, f"version_{next_v}")
    os.makedirs(version_dir, exist_ok=True)
    return version_dir

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Setup CUDA device visibility and memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_id = int(cfg.task.gpu_id)
            if gpu_id >= 0 and gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                print(f"Using GPU device: {gpu_id} (Memory growth enabled)")
            elif gpu_id < 0:
                tf.config.set_visible_devices([], 'GPU')
                print("Running on CPU (GPU disabled)")
            else:
                print(f"Requested GPU ID {gpu_id} is out of range (found {len(gpus)} GPUs). Using all available GPUs.")
        except Exception as e:
            print(f"Error configuring GPU devices: {e}")
    else:
        print("No GPU devices detected by TensorFlow. Running on CPU.")

    # 2. Extract dimensions and parameters
    ih, iw, ic = cfg.model.input_size
    ny, nx = ih // 4, iw // 4
    nc = cfg.dataset.class_num

    print(f"Input Resolution: {ih}x{iw}x{ic}")
    print(f"Heatmap Resolution: {ny}x{nx} with {nc} classes")

    # 3. Initialize Datapipes
    pipe = Datapipe()
    print("Loading training dataset pipeline...")
    g = pipe(
        cfg.dataset.csv_train, 
        nx, ny, nc, iw, ih, ic, 
        batchSize=cfg.task.batch_size,
        shuffle_buffer_size=1000,
        augment=cfg.task.augmentations.enabled,
        transform_cfg=cfg.task.augmentations
    )
    print("Loading testing dataset pipeline...")
    gt = pipe(
        cfg.dataset.csv_test, 
        nx, ny, nc, iw, ih, ic, 
        augment=False, 
        batchSize=cfg.task.batch_size,
        shuffle_buffer_size=1000
    )

    # 4. Instantiate Selected Backbone Model
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

    # 5. Build CenterNet multi-task output heads on top of backbone feature map
    # Heatmap head (sigmoid activation)
    xhead1 = Conv2D(nc, (1, 1), padding="same", use_bias=True, activation="sigmoid", name="head1_heatmap")(backbone.output)
    # Box size (width, height) log-regressed head
    xhead2 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head2_box_size")(backbone.output)
    # Sub-pixel local coordinate offset correction head
    xhead3 = Conv2D(2, (1, 1), padding="same", use_bias=True, name="head3_offset")(backbone.output)

    # Concatenate head outputs into a unified multi-task target tensor [B, ny, nx, nc + 4]
    yhead = Concatenate(axis=-1, name="head_final")([xhead1, xhead2, xhead3])
    model = Model(inputs=backbone.inputs, outputs=yhead, name=f"CenterNet_{cfg.model.name}")

    print(model.summary(line_length=120))

    # 6. Create Callbacks
    now = datetime.now()
    timestamp = str(now)[:19].replace(' ', '_').replace(':', '').replace('-', '')
    base_log_dir = f"./runs/{cfg.task.task}/{cfg.model.name}_{cfg.name}"
    log_dir = get_next_version_dir(base_log_dir)
    print(f"Log directory resolved to: {log_dir}")

    tfbcb = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, "tblogs"),
        write_graph=True,
        write_images=True,
        update_freq='batch'
    )

    mcpcb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, "checkpoints", "best_model.weights.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto'
    )

    rlrcb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        verbose=1,
        min_delta=0.0001
    )

    dricb = DrawImageCallback(logdir=os.path.join(log_dir, "tblogs"), tfdataset=gt, writerName="imagerVal")
    drtcb = DrawImageCallback(logdir=os.path.join(log_dir, "tblogs"), tfdataset=g, writerName="imagerTrain")
    term = tf.keras.callbacks.TerminateOnNaN()

    # 7. Compile and Fit CenterNet Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.task.learning_rate)
    model.compile(
        loss=centerNetLoss,
        optimizer=optimizer,
        metrics=[cls_loss, reg_loss, wh_loss, offset_loss]
    )

    initial_epoch = 0
    resume_path = cfg.task.get("resume", None)
    if resume_path:
        if os.path.exists(resume_path):
            print(f"\n🔄 [Resume] Resuming training: loading weights from {resume_path}")
            model.load_weights(resume_path)
            init_epoch_val = cfg.task.get("initial_epoch", 0)
            if init_epoch_val:
                initial_epoch = int(init_epoch_val)
                print(f"🔄 [Resume] Setting initial epoch counter to: {initial_epoch}")
        else:
            print(f"\n⚠️ [Resume] Warning: Specified resume weights path '{resume_path}' does not exist! Starting training from scratch.")

    print("\nBeginning CenterNet Model Training fits...")
    model.fit(
        g, 
        epochs=cfg.task.epochs,
        initial_epoch=initial_epoch,
        callbacks=[tfbcb, mcpcb, rlrcb, dricb, drtcb, term],
        validation_data=gt,
        steps_per_epoch=max(1, cfg.dataset.num_train_samples // cfg.task.batch_size),
        validation_steps=max(1, cfg.dataset.num_test_samples // cfg.task.batch_size)
    )

    # Save final model weights
    final_weights_path = os.path.join(log_dir, f"weights_{cfg.model.name}_final.weights.h5")
    model.save_weights(final_weights_path)
    print(f"Training fully completed! Saved final weights to {final_weights_path}")

if __name__ == "__main__":
    main()
