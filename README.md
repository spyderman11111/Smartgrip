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

### 1) Create Python venv (Python 3.10)

```bash
python3.10 -m venv ~/Smartgrip/py310 --system-site-packages
source ~/Smartgrip/py310/bin/activate
python -m pip install -U pip setuptools wheel
2) Initialize git submodules
git submodule update --init --recursive
3) Build ROS 2 workspace
From your ROS 2 workspace root:

cd ~/Smartgrip/ros2_ws
colcon build --symlink-install
source install/setup.bash
Installation Details
A) Install Grounded-SAM-2 (GroundingDINO + SAM2)
cd Grounded-SAM-2
pip install torch torchvision torchaudio
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install transformers
Download checkpoints
cd checkpoints
bash download_ckpts.sh

cd gdino_checkpoints
bash download_ckpts.sh
(Optional) Add PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:~/Smartgrip/Grounded-SAM-2' >> ~/.bashrc
source ~/.bashrc
B) Install LightGlue
cd LightGlue
pip install -e .
Patch for AMP compatibility (if needed)
Edit lightglue/lightglue.py:

try:
    AMP_CUSTOM_FWD_F32 = torch.amp.custom_fwd(
        cast_inputs=torch.float32, device_type="cuda"
    )
except AttributeError:
    AMP_CUSTOM_FWD_F32 = torch.cuda.amp.custom_fwd(
        cast_inputs=torch.float32
    )
C) Install VGGT
cd vggt
pip install -r requirements.txt
pip install trimesh pycolmap
VGGT is used in the offline reconstruction stage after multi-view capture.

Repository Layout
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
Main Pipeline (seeanything.py)
seeanything.py implements an online perception–control routine followed by offline reconstruction.

Online stage (ROS 2 runtime)
Move robot to INIT pose.

Stage-1 detection (coarse)
Capture wrist image, run GroundingDINO, convert detection to target point C1 in base_link,
update XY position and move.

Stage-2 detection (fine)
Detect again from closer view, compute refined target C2, move to a hover pose above C2.

Active scan path generation
Generate a polygon or circular path around the target center.

Capture loop
At each waypoint: move, dwell, save image, and log joint state and base-to-camera pose.

Offline stage
VGGT reconstruction (core/vggtreconstruction.py)
Input: output/ur5image/*.png
Output: output/offline_output/points.ply, cameras.json, optional cameras_lines.ply.

Point-cloud post-processing (core/point_processing.py)
Produces cleaned object cluster, removes table plane, and aligns the object to base_link.

Robot returns to INIT pose and exits.

Outputs
All experiment artifacts are stored under gripanything/output/:

ur5image/pose_k_image.png
Images captured at each scan waypoint.

ur5camerajointstates.json
Per-image joint states and base-to-camera poses at capture time.

offline_output/points.ply
Reconstructed point cloud (VGGT world).

offline_output/object_in_base_link.json
Final object center and geometry in base_link.

Core Modules to Edit
core/config.py: topics, frames, thresholds, scan parameters.

core/vision_geom.py: image-to-3D geometry and frame transforms.

core/ik_and_traj.py: IK calls and trajectory publishing.

core/polygon_path.py: active scan trajectory generation.

core/point_processing.py: point-cloud filtering and object center estimation.

Utilities
utils/tool_to_camera_tf_publisher.py: publish static camera TF.

utils/ur5e_charuco_handeye_ros2.py: hand–eye calibration.

utils/publish_object_points_tf.py: publish object frames for RViz.

utils/goto_point_from_object_json.py: move robot to final object pose.

utils/vs.py: point-cloud visualization.

Troubleshooting
Check camera topic
ros2 topic list | grep image
ros2 topic echo -n 1 /<camera_topic>/image_raw
Check TF chain
ros2 run tf2_ros tf2_echo base_link tool0
ros2 run tf2_ros tf2_echo tool0 camera_optical
Verify outputs after a run
ls -lh gripanything/output/ur5image
ls -lh gripanything/output/offline_output
Notes on Naming and Imports
Reusable modules should live under gripanything/core/.

One-off scripts belong in gripanything/utils/.

Outputs remain under gripanything/output/ to keep runs reproducible.