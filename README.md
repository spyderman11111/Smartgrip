# gripanything — Repository Overview & Pipeline README

This repository contains a ROS 2 + Python perception–control pipeline for a UR5e wrist camera setup. It supports:

* **Text-prompted detection** (GroundingDINO) to obtain target location candidates.
* **Geometry utilities** to convert detections into robot-frame target points and hover poses.
* **Motion execution** through IK + joint trajectory publishing.
* **Multi-view image capture** along a polygon/circular scan path.
* **Offline reconstruction** (VGGT) and **point-cloud post-processing** to estimate the object pose/center in `base_link`.

# Setting Up the Environment

```bash
python3.10 -m venv ~/Smartgrip/py310 --system-site-packages
source ~/Smartgrip/py310/bin/activate
```

You can create a new environment at will, I am using python version **3.10**, don't use conda!

---

## Git Submodule Setup


```bash
git submodule update --init --recursive
```
---

# Installation Instructions

## Install Grounded-SAM-2

```bash
cd Grounded-SAM-2
pip3 install torch torchvision torchaudio
#check
# echo $CUDA_HOME
# export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
# echo $CUDA_HOME
# echo 'export CUDA_HOME=$(dirname $(dirname $(which nvcc)))' >> ~/.bashrc
# echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc

pip install -e .
```

### Download checkpoints

```bash
cd checkpoints
bash download_ckpts.sh

cd gdino_checkpoints
bash download_ckpts.sh
```

### Install

```bash
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install transformers

echo 'export PYTHONPATH=$PYTHONPATH:~/Smartgrip/Grounded-SAM-2' >> ~/.bashrc
source ~/.bashrc
```

## install Lightglue
```bash
cd LightGlue
python -m pip install -e .
```
need to change code in **lightglue.py** line 24

    try:
        AMP_CUSTOM_FWD_F32 = torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    except AttributeError:
        AMP_CUSTOM_FWD_F32 = torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)

## install VGGT

```bash
cd vggt
pip install -r requirements.txt
pip install trimesh
pip install pycolmap
```

colcon build --symlink-install \
  --cmake-args -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  --cmake-force-configure


---

## Directory Tree 

```text
/home/MA_SmartGrip/Smartgrip/ros2_ws/src/gripanything
├── gripanything
│   ├── core
│   │   ├── config.py
│   │   ├── detect_with_dino.py
│   │   ├── dino_sam_lg.py
│   │   ├── extract_frames.py
│   │   ├── folder_dino_sam2_rgbmask.py
│   │   ├── ik_and_traj.py
│   │   ├── LightGlueMatcher.py
│   │   ├── point_processing.py
│   │   ├── polygon_path.py
│   │   ├── segment_with_sam2.py
│   │   ├── tf_ops.py
│   │   ├── vggtreconstruction.py
│   │   └── vision_geom.py
│   ├── output
│   │   ├── image_jointstates.json
│   │   ├── postprocess_output
│   │   │   ├── main_cluster_clean.ply
│   │   │   ├── object_in_base_link.json
│   │   │   └── points_no_table.ply
│   │   ├── ur5image
│   │   │   ├── pose_1_image.png
│   │   │   ├── ...
│   │   │   └── pose_6_image.png
│   │   └── vggt_output
│   │       ├── cameras.json
│   │       ├── cameras_lines.ply
│   │       └── points.ply
│   ├── seeanything.py
│   ├── seeanything_debug.py
│   └── utils
│       ├── goto_point_from_object_json.py
│       ├── publish_object_points_tf.py
│       ├── tool_to_camera_tf_publisher.py
│       └── vs.py
├── package.xml
├── resource
│   └── gripanything
├── setup.cfg
└── setup.py
```

---

## `gripanything/` Top-level Scripts

### `seeanything.py` (Main ROS 2 node)

The main online pipeline script:

* Moves UR5 to an initial pose.
* Runs a **two-stage (coarse→fine)** detection to estimate the target center in `base_link`.
* Moves to a hover pose above the target.
* Executes an **active viewpoint scan** (polygon/circle) around the target.
* At each viewpoint: saves an image and logs robot joint state + camera pose into a JSON file.

Output artifacts are typically written under:

* `gripanything/output/ur5image/` (images)
* `gripanything/output/image_jointstates.json` (per-image robot state + camera pose)

### `seeanything_debug.py`

A debug-friendly variant of the main node, usually used to:

* Test perception modules without full motion,
* Verify topic/TF availability,
* Run a shorter or simplified motion/capture routine.

---

## `gripanything/core/` Modules

### `core/config.py`

Configuration loading utilities:

* Defines how parameters are read (ROS parameters and/or local defaults).
* Central place to manage topics, frame names, thresholds, motion times, scan settings, etc.

### `core/detect_with_dino.py`

GroundingDINO wrapper:

* Loads the DINO model and runs open-vocabulary detection from a text prompt.
* Returns candidate boxes, scores, and associated metadata.

### `core/segment_with_sam2.py`

SAM2 wrapper:

* Performs segmentation given an input image and prompt signals (often box prompts).
* Produces masks used for downstream filtering or visualization.

### `core/dino_sam_lg.py`

Integration module (DINO + SAM2 + LightGlue):

* A higher-level orchestration file that typically:

  * Detects with DINO,
  * Segments with SAM2,
  * Optionally runs feature matching with LightGlue to validate target identity or cross-view consistency.

### `core/LightGlueMatcher.py`

LightGlue-based feature matcher:

* Provides feature extraction + matching interface (e.g., SuperPoint + LightGlue).
* Used for cross-view association or match scoring.

### `core/vision_geom.py`

Vision-to-geometry conversion utilities (online geometry step):

* Takes image-space detections and TF transforms and produces:

  * Target point `C` in a robot frame (often `base_link`),
  * A hover pose above that point,
  * TF frames for visualization in RViz (e.g., object center frame, circle center frame).
* This is where camera-ray geometry, frame conversion, and bias compensation often live.

### `core/tf_ops.py`

TF / geometry helpers:

* Convert ROS TF messages into rotation/translation (`R`, `p`),
* Quaternion to rotation, common fixed transforms (e.g., camera_link ↔ camera_optical),
* Small math utilities used across the pipeline.

### `core/ik_and_traj.py`

Motion primitives and execution:

* `IKClient`: requests IK solutions (typically via MoveIt service).
* `TrajectoryPublisher`: publishes joint trajectories to the UR controller.
* `MotionContext`: reads `/joint_states`, checks stationarity, builds IK seeds, and handles safety checks (e.g., jump detection).

### `core/polygon_path.py`

Active scan path generator:

* Generates a sequence of poses on a circle/polygon around a center in a chosen plane.
* Supports direction, number of vertices, radius, and orientation mode.

### `core/point_processing.py`

Point cloud post-processing utilities:

* Filters, clustering, centroid computation, bounding boxes, etc.
* Often used by your offline postprocess step (table removal, main cluster extraction, robust center).

### `core/extract_frames.py`

Frame extraction helper:

* Extracts frames from videos into image files for offline experiments.

### `core/folder_dino_sam2_rgbmask.py`

Folder-based batch inference:

* Runs DINO + SAM2 over a directory of images.
* Saves per-image masks/overlays and possibly JSON logs.

### `core/vggtreconstruction.py`

VGGT reconstruction script/module:

* Reconstructs a point cloud from the captured multi-view images in `output/ur5image`.
* Produces:

  * `output/vggt_output/points.ply`
  * `output/vggt_output/cameras.json`
  * (optional) `output/vggt_output/cameras_lines.ply` for visualization

---

## `gripanything/utils/` Tools

### `utils/tool_to_camera_tf_publisher.py`

Publishes a static TF:

* Typically `tool0` → `camera_link` or `camera_optical`.
* Used for RViz visualization and to ensure consistent TF availability.

### `utils/publish_object_points_tf.py`

Publishes TF frames for object results:

* Reads object outputs (e.g., center/corners) and publishes them as TF frames
  for RViz visualization or downstream motion scripts.

### `utils/goto_point_from_object_json.py`

Simple motion utility:

* Loads `object_in_base_link.json` (or similar) and commands the robot to move to:

  * object center,
  * hover above center,
  * or a selected corner/point.

### `utils/vs.py`

Miscellaneous helper / scratch utility:

* Often used for quick tests, visualization, or ad-hoc experiments.

---

## `gripanything/output/` Artifacts

This folder contains run outputs (examples from your snapshot):

* `output/ur5image/pose_k_image.png`
  Images captured at each scan waypoint.

* `output/image_jointstates.json`
  Per-image joint positions and camera pose (in `base_link`) at capture time.

* `output/vggt_output/`
  VGGT reconstruction outputs:

  * `points.ply` (VGGT world)
  * `cameras.json` (camera pose/intrinsics per frame)
  * `cameras_lines.ply` (optional Open3D frustums)

* `output/postprocess_output/`
  Post-processing and alignment outputs:

  * `points_no_table.ply` (table removed)
  * `main_cluster_clean.ply` (main object cluster)
  * `object_in_base_link.json` (final object center + OBB corners in `base_link`)

---

## Main Pipeline: `seeanything.py`

Below is the conceptual pipeline implemented by `seeanything.py`:

1. **Move to INIT pose**

   * Publish a joint trajectory to a predefined initial configuration.
   * Wait until `/joint_states` indicates the robot is stationary.

2. **Stage-1 detection (coarse)**

   * Capture an image from the wrist camera topic.
   * Run GroundingDINO once to get a coarse target hypothesis.
   * Convert the detection to a target point `C1` in `base_link` (via `vision_geom.py` + TF).
   * Create a pose that updates **XY to `C1`** but keeps current tool **Z**.
   * Solve IK and move.

3. **Wait until stationary**

   * Ensure the robot has settled before the fine stage.

4. **Stage-2 detection (fine)**

   * Run detection again from the new viewpoint.
   * Convert to refined target `C2` in `base_link`.
   * Build a **hover pose** with Z = `C2.z + hover_above` (tool Z-down).
   * Solve IK and move to hover.

5. **Generate active scan path**

   * Read current tool yaw in the XY plane.
   * Generate an N-vertex polygon/circular path around the target center.
   * Optionally trim by `sweep_deg` (< 360°) to avoid joint limits.

6. **Capture loop over vertices**

   * For each vertex:

     * IK solve and move.
     * Wait a dwell time.
     * Save image to `output/ur5image/pose_k_image.png`.
     * Log joint state + base→camera pose to `output/image_jointstates.json`.
   * If an IK solution is flagged as an abnormal jump, skip that vertex (no capture).

7. **Return to INIT and exit**

   * Send the robot back to the initial pose and shutdown the node.

---

## Offline Pipeline (Typical Follow-up After `seeanything.py`)

After the scan completes, a common offline workflow is:

1. **VGGT reconstruction**

* Input: `output/ur5image/*.png`
* Script: `core/vggtreconstruction.py`
* Output: `output/vggt_output/points.ply`, `cameras.json`, ...

2. **Point-cloud post-process + alignment**

* Input:

  * `output/vggt_output/points.ply`
  * `output/vggt_output/cameras.json`
  * `output/image_jointstates.json`
* Output:

  * `output/postprocess_output/object_in_base_link.json`
  * plus intermediate PLYs for debugging/visualization

This offline stage yields the final object estimate in the robot base frame (`base_link`), suitable for grasping or evaluation.

---

## Notes on Naming & Imports

* Anything meant to be *imported by the main pipeline* should live under `gripanything/core/` and expose clear functions/classes.
* One-off scripts that are not part of the main pipeline should live under `gripanything/utils/`.
* Outputs should remain under `gripanything/output/` to keep runs reproducible and easy to archive.

---


