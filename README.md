# Setting Up the Environment

```bash
conda activate smartgrip
```

You can create a new conda environment at will, I am using python version 3.11

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
echo $CUDA_HOME
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
## vision_part package Instructions

### **extract_frames** 
method, implemented in extract_frames.py, supports two modes: video file input (e.g., .mp4) and webcam stream input (e.g., '0'). It uses OpenCV to extract every N-th frame and saves them to a specified directory using the original frame index in the filename.

### **GroundingDinoPredictor**

This class wraps a **GroundingDINO** model for **zero-shot object detection** using natural language prompts.

---

**Functionality:**

- Uses `transformers.AutoModelForZeroShotObjectDetection` and `AutoProcessor`.
- Takes an input RGB image and a **text prompt** (e.g., `"red box"`).
- Outputs:
  - `boxes`: Tensor of detected bounding boxes `[x0, y0, x1, y1]`
  - `labels`: List of matched class strings

---

**Key Parameters:**

- `model_id`: Default is `"IDEA-Research/grounding-dino-tiny"`
- `device`: `"cuda"` or `"cpu"`
- `box_threshold`: Filter out boxes with low visual confidence
- `text_threshold`: Filter out boxes with low textual alignment

---

**Example Usage:**

```python
predictor = GroundingDinoPredictor()
boxes, labels = predictor.predict(image, "red box")
```

### **SAM2ImagePredictorWrapper**

The `SAM2ImagePredictorWrapper` class wraps Meta AI's **SAM2** segmentation model to perform fine-grained mask prediction on RGB images, either using full images or cropped regions.

**Functionality:**

1. **Full Image Masking** (`run_inference`)
   - Loads a full RGB image and a target bounding box (x1, y1, x2, y2).
   - Predicts a segmentation mask using the SAM2 model.
   - Saves grayscale binary mask as `*_mask_gray.png` in the specified directory.
   - Returns a dictionary containing:
     - `mask_array` (`np.uint8`): Binary mask of shape (H, W)
     - `mask_score` (`float`): IoU confidence score
     - `low_res_mask` (`np.ndarray`): Raw low-resolution logits (256×256)

2. **Cropped Region Contour Masking** (`run_on_crop`)
   - Accepts a cropped RGB `PIL.Image` (e.g., extracted via a bounding box).
   - Uses full image region as a dummy box to apply SAM2.
   - Applies post-processing:
     - Thresholding (`mask < 0.2`)
     - Contour filtering by area (`> 700`)
     - Draws filtered contours on the original crop.
   - Saves result as an RGB image with drawn contours.

**Initialization Parameters:**

- `model_id` (str): Pretrained SAM2 model ID (e.g., `"facebook/sam2.1-hiera-large"`)
- `device` (str): `"cuda"` or `"cpu"`
- `mask_threshold` (float): Binarization threshold for masks
- `max_hole_area` (float): Max allowed hole size in masks
- `max_sprinkle_area` (float): Max allowed sprinkle noise in masks
- `multimask_output` (bool): Whether to output multiple masks per input
- `return_logits` (bool): Whether to return raw logits (used for low-resolution masks)

### **LightGlueMatcher**

The `LightGlueMatcher` class provides a modular wrapper around LightGlue and SuperPoint for image matching tasks. It allows configurable transformer architecture and inference settings and supports feature extraction and matching with minimal code.

**Functionality:**

1. **Feature Extraction**  
   - Loads and normalizes an image using `load_image`.
   - Uses `SuperPoint` to extract keypoints and descriptors.
   - Returns a feature dictionary including:
     - `keypoints` (Tensor): shape `[1, N, 2]`  
     - `descriptors` (Tensor): shape `[1, N, D]`  
     - `image_size` (Tensor): shape `[1, 2]` (width, height)

2. **Feature Matching**  
   - Accepts two sets of keypoints, descriptors, and image sizes.
   - Feeds them to `LightGlue` for matching.
   - Returns a dictionary including:
     - `matches`: matched keypoint indices per image
     - `scores`: confidence scores per match

3. **Optional Compilation**  
   - `matcher.compile("reduce-overhead")` speeds up inference via Torch 2.0 graph compilation.

**Initialization Parameters:**

- `feature_type` (str): Only `'superpoint'` is currently supported.
- `device` (str): `'cuda'` or `'cpu'`
- `descriptor_dim`, `n_layers`, `num_heads`: Transformer config
- `flash`, `mp`: Optimization flags (e.g., FlashAttention, mixed precision)
- `depth_confidence`, `width_confidence`, `filter_threshold`: Matching quality thresholds
- `weights`: Path to custom pretrained weights

### **vggtreconstruction**

This script runs **VGGT-based monocular 3D reconstruction** and exports a **COLMAP-compatible sparse model**.

---

**Pipeline Overview:**

1. **Load Config & Model**
   - Uses pretrained VGGT-1B weights (auto-download).
   - Reads images from `scene_dir/images`.

2. **VGGT Inference**
   - Predicts camera extrinsics, intrinsics, depth, and confidence.
   - Supports `float16` / `bfloat16` based on GPU capability.

3. **Point Cloud Generation**
   - Converts depth to 3D point cloud.
   - Applies confidence threshold and random sampling.

4. **COLMAP Structure Creation**
   - Builds `pycolmap.Reconstruction` with optional shared camera.
   - Rescales intrinsics to match original image resolution.

5. **Export**
   - Saves:
     - `sparse/`: COLMAP binary model + `points.ply`
     - `sparse_txt/`: TXT model via `colmap model_converter`

---

## **main.py - Aria & UR5e Multi-View Object Matching Pipeline**

This script performs **multi-view object detection, segmentation, feature extraction, and feature matching** between an **ARIA** glasses image and three selected **UR5e** arm camera frames. It integrates **GroundingDINO**, **SAM2**, and **LightGlue** into a complete matching pipeline.

---

### **Pipeline Overview**

1. **Frame Selection**  
   Function: `select_frame_paths(aria_dir, ur5e_dir)`  
   - Randomly selects **1 frame** from ARIA camera folder  
   - Selects **3 UR5e frames** (first, middle, last) for coverage  
   → Input to the downstream detection pipeline

2. **Model Initialization**  
   - Loads GroundingDINO (object detection), SAM2 (segmentation), and LightGlue (feature matching).
   - Device set automatically (`cuda` or `cpu`)  
   → Prepares all models for zero-shot visual-language inference.

3. **Object Detection + Segmentation + Feature Extraction**  
   Function: `extract_features(...)`  
   - Runs GroundingDINO with a text prompt (e.g., `"ball"`) to detect objects.  
   - Crops the detected bounding box → feeds into SAM2 to generate segmentation mask.  
   - Extracts keypoints + descriptors from the masked region using LightGlue.  
   → For each input image (ARIA or UR5e), returns a feature dictionary.

4. **Feature Matching**  
   Function: `match_and_report(...)`  
   - For each UR5e frame:
     - Matches its features with ARIA features.
     - Calculates number of matches and average score.
     - Saves match visualization (using `draw_matches`) to output directory.
   → Provides quantitative and qualitative comparison between views.

5. **Result Export**  
   - All match scores and image paths saved to a JSON file:  
     `outputs/match_confidence_result.json`  
   - Contains:
     - match count
     - average confidence score
     - image and mask paths
     - path to match visualization image

---

### **Step Relationships**

| Step | Input | Output | Used In |
|------|-------|--------|---------|
| 1. select_frame_paths | image folders | ARIA frame + 3 UR5e frames | → Step 3 |
| 2. model init | model names | loaded models | → Step 3–4 |
| 3. extract_features | image path, prompt | features (keypoints, descriptors, image size) | → Step 4 |
| 4. match_and_report | two feature sets | match count + scores + visual | → Step 5 |
| 5. save result | match data dict | JSON summary | — |

---

### **Example Output (JSON structure)**

```json
{
  "First": {
    "match_count": 34,
    "score_mean": 0.72,
    "aria_image": "...",
    "ur5e_image": "...",
    "match_vis": "outputs/aria_vs_first_matches.png"
  },
}
```
# ROS2 command

```bash
source install/setup.bash

ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.11 launch_rviz:=true
#Do not use vscode from app installer! Just install vscode from deb package.

ros2 launch pylon_ros2_camera_wrapper pylon_ros2_camera.launch.py

ros2 launch pylon_ros2_camera_wrapper my_blaze.launch.py
```