# Setting Up the Environment

```bash
conda activate smartgrip
```

---

# Git Submodule Setup


```bash
git submodule add <REPO_URL>

git submodule update --init --recursive
```
---

# Installation Instructions

## Install Grounded-SAM-2

```bash
cd Grounded-SAM-2
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
```
## install VGGT

```bash
cd vggt
pip install -r requirements.txt
```
## install Lightglue
```bash
cd LightGlue
python -m pip install -e .
```

## vision_part package Instructions

The **extract_frames** method, implemented in extract_frames.py, supports two modes: video file input (e.g., .mp4) and webcam stream input (e.g., '0'). It uses OpenCV to extract every N-th frame and saves them to a specified directory using the original frame index in the filename.

The **GroundingDinoPredictor** class performs zero-shot object detection using text prompts and returns bounding boxes and class labels from input images.

## Output Description

### 1. Saved Frames

Extracted frames are saved to the directory specified by output_dir (e.g., ../outputs).

Filenames follow the pattern:

frame_<frame_index>.jpg

### 2.  Console Output Format

Each saved frame will output detection info like:

[Frame] frame_00220.jpg | Detected 1 ball:
  - ball: x1=816, y1=544, x2=1076, y2=803

x1, y1: top-left corner of the bounding box (in pixels)

x2, y2: bottom-right corner of the bounding box (in pixels)