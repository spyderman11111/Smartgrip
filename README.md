# gripanything — ROS 2 Perception–Control Pipeline for UR5e Wrist Camera

`gripanything` is a ROS 2 + Python perception–control pipeline for a UR5e wrist-camera setup.
It supports:

- **Text-prompted detection** (GroundingDINO) to obtain target candidates.
- **Segmentation** (SAM2) to extract object masks and object-only images.
- **Vision-to-robot geometry** utilities to convert detections into target points and hover poses in `base_link`.
- **Motion execution** via IK and joint trajectory publishing.
- **Active multi-view capture** along a polygon or circular scan path.
- **Offline reconstruction** (VGGT) and **point-cloud post-processing** to estimate object center and geometry in `base_link`.

---

## Requirements

- Ubuntu 20.04 / 22.04
- ROS 2 (Humble recommended)
- Python **3.10** (venv; **do not use conda**)
- CUDA-capable GPU recommended (for DINO / SAM2 / LightGlue)

---

## Quickstart

### Create Python venv (Python 3.10)

```bash
python3.10 -m venv ~/Smartgrip/py310 --system-site-packages
source ~/Smartgrip/py310/bin/activate
python -m pip install -U pip setuptools wheel
```

### Initialize git submodules

```bash
git submodule update --init --recursive
```

### Build ROS 2 workspace

```bash
cd ~/Smartgrip/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Installation

### Grounded-SAM-2 (GroundingDINO + SAM2)

```bash
cd Grounded-SAM-2
pip install torch torchvision torchaudio
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install transformers
```

Download checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
cd gdino_checkpoints
bash download_ckpts.sh
```

(Optional) add PYTHONPATH:

```bash
echo 'export PYTHONPATH=$PYTHONPATH:~/Smartgrip/Grounded-SAM-2' >> ~/.bashrc
source ~/.bashrc
```

---

### LightGlue

```bash
cd LightGlue
pip install -e .
```

If AMP causes errors, edit `lightglue/lightglue.py`:

```python
try:
    AMP_CUSTOM_FWD_F32 = torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
except AttributeError:
    AMP_CUSTOM_FWD_F32 = torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
```

---

### VGGT

```bash
cd vggt
pip install -r requirements.txt
pip install trimesh pycolmap
```

---

## Repository Layout

```text
gripanything/
├── gripanything
│   ├── core/
│   │   ├── config.py
│   │   ├── detect_with_dino.py
│   │   ├── segment_with_sam2.py
│   │   ├── dino_sam_lg.py
│   │   ├── LightGlueMatcher.py
│   │   ├── vision_geom.py
│   │   ├── tf_ops.py
│   │   ├── ik_and_traj.py
│   │   ├── polygon_path.py
│   │   ├── vggtreconstruction.py
│   │   ├── point_processing.py
│   │   └── folder_dino_sam2_rgbmask.py
│   ├── utils/
│   │   ├── extract_frames.py
│   │   ├── goto_point_from_object_json.py
│   │   ├── publish_object_points_tf.py
│   │   ├── tool_to_camera_tf_publisher.py
│   │   ├── ur5e_charuco_handeye_ros2.py
│   │   └── vs.py
│   ├── output/
│   │   ├── ur5image/
│   │   ├── ur5camerajointstates.json
│   │   └── offline_output/
│   │       ├── points.ply
│   │       ├── cameras.json
│   │       ├── cameras_lines.ply
│   │       ├── points_no_table.ply
│   │       ├── main_cluster_clean.ply
│   │       └── object_in_base_link.json
│   ├── seeanything.py
│   └── seeanything_debug.py
├── package.xml
├── resource/
├── setup.cfg
└── setup.py
```

---

## Main Pipeline (`seeanything.py`)

Online stage:

1. Move robot to INIT pose.
2. Stage-1 coarse detection (DINO) → target `C1` in `base_link`.
3. Stage-2 fine detection → refined target `C2` and hover pose.
4. Generate polygon/circular scan path.
5. Capture loop: move → dwell → save image → log joint state and camera pose.

Offline stage:

6. VGGT reconstruction → `points.ply`, `cameras.json`.
7. Point-cloud post-processing and alignment → `object_in_base_link.json`.
8. Return robot to INIT and exit.

---

## Outputs

- `output/ur5image/pose_k_image.png`: captured scan images.
- `output/ur5camerajointstates.json`: joint states and camera poses.
- `output/offline_output/points.ply`: reconstructed point cloud.
- `output/offline_output/object_in_base_link.json`: final object estimate.

---

## Notes

- Importable modules belong in `gripanything/core/`.
- Utility scripts belong in `gripanything/utils/`.
- Outputs stay under `gripanything/output/` for reproducibility.

## Aria Eye-Gaze Mask (Project Aria) — Setup and Run

This project uses an additional script to generate **Aria eye-gaze conditioned masks** (Aria RGB + EyeTrack → gaze ROI → SAM2 mask).
The output masks are saved to `gripanything/output/ariaimage/` and can be used for downstream verification/matching.

### 1) Activate the Aria Python environment

```bash
source /home/MA_SmartGrip/Smartgrip/aria/bin/activate
cd /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples
```

### 2) Required source patch (Torch checkpoint loading)

You must modify the following file to avoid checkpoint loading issues:

- File:
  `/home/MA_SmartGrip/Smartgrip/projectaria_eyetracking/projectaria_eyetracking/inference/model/model_utils.py`
- Line ~203:
  Change to:

```python
model_buffer = torch.load(chkpt_path, map_location=map_location, weights_only=False)
```

### 3) Run Aria gaze→mask script

The script below is responsible for running **Aria streaming** and saving **eye-gaze masks**:

- Script:
  `/home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/stream_rgb_eye_sam2.py`

All outputs will be written into:

- Output directory:
  `/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage`

#### USB mode

```bash
python /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/stream_rgb_eye_sam2.py \
  --interface usb \
  --calib-vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --output-dir /home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage
```

#### WiFi mode

```bash
python /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/stream_rgb_eye_sam2.py \
  --interface wifi \
  --device-ip 192.168.0.102 \
  --calib-vrs /home/MA_SmartGrip/Smartgrip/projectaria_client_sdk_samples/Gaze_tracking_attempts_1.vrs \
  --update_iptables \
  --output-dir /home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything/gripanything/output/ariaimage
```

> Notes:
> - `--calib-vrs` points to a recorded calibration VRS file used by the eye-tracking inference.
> - In WiFi mode, `--update_iptables` is used to update system rules for streaming.
> - Make sure the output directory exists and is writable.

## UR5 Wrist Camera (Basler / pylon_ros2_camera_wrapper)

Start the wrist camera node with `pylon_ros2_camera_wrapper`.
This should be launched from the ROS 2 workspace, and **do not** activate the `py310` environment for this step.

```bash
cd ~/Smartgrip/ros2_ws
source install/setup.bash
ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py
