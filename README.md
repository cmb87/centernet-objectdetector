# 🛰️ CenterNet Object Detector for Mobile & Edge Deployment

An anchor-free, lightweight CenterNet object detector designed and optimized specifically for real-time aerial drone tracking on mobile devices and edge NPUs.

---

## ✨ Highlights & New Upgrades

We have executed a comprehensive technical overhaul of this repository, resolving critical mathematical and structural bugs, implementing high-performance pretrained transfer learning, and introducing a modern dynamic Hydra configuration system that mirrors your YOLO development pipeline!

### 📊 1. Unified Hydra Configuration Setup
You can now run, manage, and scale your CenterNet experiments exactly like your YOLO repository!
* Config path: `config/config.yaml`
* Tasks: `config/task/train.yaml`
* Backbones: `config/model/shufflenet.yaml` and `config/model/mobilenet_pretrained.yaml`
* Datasets: `config/dataset/thermal.yaml`

---

## 🛠️ Usage & Command Reference

### 🚀 Train with ShuffleNet Backbone
To train your lightweight ShuffleNet model on the thermal drone dataset:
```bash
/home/cpeeren/projects/01_ml/venvTF/bin/python lazy_centernet.py model=shufflenet task.gpu_id=0 task.batch_size=16
```

### 🚀 Train with Pretrained MobileNetV3 Backbone
To leverage Transfer Learning with pretrained ImageNet weights for highly accurate sub-pixel convergence:
```bash
/home/cpeeren/projects/01_ml/venvTF/bin/python lazy_centernet.py model=mobilenet_pretrained task.gpu_id=0 task.batch_size=16
```

### 📦 Dataset Options
You can train CenterNet on either the original **thermal** dataset or the new unified, multi-source **merged drone** dataset!
* **Thermal Dataset (Default)**: `dataset=thermal`
* **Merged Drone Dataset (New!)**: `dataset=merged`
  * We built a high-speed, zero-dependency parser script `src/data/prepare_merged_dataset.py` that automatically converts your YOLO dataset structure into CenterNet CSV coordinates.
  * To run training on the merged drone dataset:
    ```bash
    /home/cpeeren/projects/01_ml/venvTF/bin/python lazy_centernet.py dataset=merged model=mobilenet_pretrained
    ```

### 🎛️ Dynamic Overrides (Examples)
* **Custom Learning Rate**: `task.learning_rate=5e-5`
* **Custom Target Epochs**: `task.epochs=500`
* **Custom GPU Select**: `task.gpu_id="1"`
* **Custom Batch Size**: `task.batch_size=8`

---

## 🐞 Codebase Audit: Bugs Fixed

We successfully identified and resolved four deep, silent bugs in the repository:

### 1. Loss Normalization Discrepancy (`src/losses.py`)
* **Bug**: The heatmap loss normalization factor `N` was being silently overwritten by the local bounding box offset mask sum. This mathematically distorted the scale of the heatmap loss.
* **Fix**: Assigned separate, distinct normalizers `N_hm` (for keypoints) and `N_box` (for coordinates), stabilizing overall loss descent.

### 2. Discarded Regularization Pathway (`src/backends/shufflenet.py`)
* **Bug**: In the `x4` skip connection fusion layer of ShuffleNet, a `Dropout(rate=0.25)` layer was initialized, but its output was immediately ignored because the original `x4` tensor was passed directly to the `ChannelAttentionLayer` instead of the regularized tensor.
* **Fix**: Chained the output correctly to restore the regularization pathway and combat overfitting.

### 3. Hardcoded Shape Constraint (`src/data/datapipe.py`)
* **Bug**: The dataset format helper forced the target dimension to `y.set_shape([None, None, None, 5])`, which permanently broke training if attempting to run multi-class scenarios.
* **Fix**: Replaced the hardcoded constant with a dynamic shape query `y.set_shape([None, None, None, self.nc + 4])`.

### 4. Swapped Heatmap Visualizer Tags (`src/callbacks.py`)
* **Bug**: In the TensorBoard image logging callback, the image inputs for predicted and ground-truth heatmaps were completely inverted, meaning predicted heatmaps were logged under the `"hmTrue"` tag and vice-versa.
* **Fix**: Swapped the logged inputs so that predictions and ground truths are accurately named inside your TensorBoard dashboard.

---

## 📂 Project Directory Structure

```
├── config/                     # Hydra Configuration Directory
│   ├── config.yaml             # Main configuration entry point
│   ├── dataset/
│   │   └── thermal.yaml        # Dataset-specific annotations & statistics
│   ├── model/
│   │   ├── mobilenet_pretrained.yaml
│   │   └── shufflenet.yaml
│   └── task/
│       └── train.yaml          # Hyperparameters and learning schedules
│
├── src/                        # Model & pipeline source code
│   ├── backends/
│   │   ├── mobilenetv3.py      # New Pretrained MobileNetV3-Small Backbone
│   │   └── shufflenet.py       # Refactored ShuffleNet-V1 Backbone (Fixed!)
│   ├── data/
│   │   └── datapipe.py         # TensorFlow Data Input Pipeline (Fixed!)
│   ├── callbacks.py            # TensorBoard & Visualization Callbacks (Fixed!)
│   ├── losses.py               # CenterNet Multi-task loss functions (Fixed!)
│   └── metrics.py              # Performance evaluation metrics
│
└── lazy_centernet.py           # Unified entry point for CenterNet training
```
