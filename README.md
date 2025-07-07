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

## vision_part package structure

The **extract_frames** method extracts and saves every N-th frame from a video file or webcam stream to a specified output directory.

The **GroundingDinoPredictor** class performs zero-shot object detection using text prompts and returns bounding boxes and class labels from input images.

## Output Description