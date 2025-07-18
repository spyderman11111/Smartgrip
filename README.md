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

**extract_frames** method, implemented in extract_frames.py, supports two modes: video file input (e.g., .mp4) and webcam stream input (e.g., '0'). It uses OpenCV to extract every N-th frame and saves them to a specified directory using the original frame index in the filename.

**GroundingDinoPredictor** class performs zero-shot object detection using text prompts and returns bounding boxes and class labels from input images.

**SAM2ImagePredictorWrapper** This class wraps Meta AI's SAM2 model to perform segmentation on a specified region of an input image defined by a bounding box.

**Function**

    Loads an RGB image and a bounding box (x1, y1, x2, y2)

    Applies the SAM2 model to predict a fine-grained mask for the specified region

    Saves results including color mask, grayscale mask, and overlay image

**Output**

Returns a dictionary with:

    mask_array (np.ndarray): Binary mask of shape (H, W)

    mask_score (float): IoU score of the predicted mask

    low_res_mask (np.ndarray): Raw low-resolution logits, shape (256, 256)

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