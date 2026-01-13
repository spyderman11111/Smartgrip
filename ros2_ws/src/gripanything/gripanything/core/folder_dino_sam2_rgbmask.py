#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from PIL import Image

from detect_with_dino import GroundingDinoPredictor
from segment_with_sam2 import SAM2ImagePredictorWrapper


def list_images(input_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [os.path.join(input_dir, f)
             for f in sorted(os.listdir(input_dir))
             if f.lower().endswith(exts)]
    return files


def build_rgb_mask(image_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """仅保留掩码内的原始 RGB，其他像素全黑。"""
    out = np.zeros_like(image_np, dtype=np.uint8)
    if mask_bool is not None and mask_bool.any():
        out[mask_bool] = image_np[mask_bool]
    return out


def clip_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 <= x1: x2 = min(w - 1.0, x1 + 1.0)
    if y2 <= y1: y2 = min(h - 1.0, y1 + 1.0)
    return int(x1), int(y1), int(x2), int(y2)


def try_delete(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def process_one_image(image_path, prompt, dino, sam2,
                      output_dir, select_method="best_score",
                      box_threshold=0.25, text_threshold=0.25,
                      save_black_if_empty=False, delete_extras=True):
    """对单张图片执行 DINO→SAM2，仅保存 *_rgbmask.png；返回状态字典。"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    image_np = np.array(image)

    # --- DINO 检测 ---
    boxes, labels, scores = dino.predict(
        image=image,
        text_prompts=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if boxes is None or len(boxes) == 0:
        if save_black_if_empty:
            out = np.zeros_like(image_np, dtype=np.uint8)
            out_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(image_path))[0] + "_rgbmask.png",
            )
            cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            return {"image": image_path, "status": "empty_black", "rgbmask_path": out_path}
        return {"image": image_path, "status": "no_detection", "rgbmask_path": None}

    boxes_np  = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
    scores_np = scores.cpu().numpy() if (scores is not None and hasattr(scores, "cpu")) else (
        np.asarray(scores) if scores is not None else None
    )

    # 选择框：默认分数最高；若无分数则按面积最大
    if scores_np is not None and select_method == "best_score":
        sel = int(np.argmax(scores_np))
    else:
        areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
        sel = int(np.argmax(areas))

    x1, y1, x2, y2 = clip_xyxy(boxes_np[sel], w, h)

    # --- SAM2 分割 ---
    result = sam2.run_inference(
        image_path=image_path,
        box=(x1, y1, x2, y2),
        save_dir=output_dir,
    )
    mask = result.get("mask_array", None)

    # 删掉 SAM2 封装生成的其它文件，只保留我们自建的 *_rgbmask.png
    if delete_extras:
        try_delete(result.get("mask_gray_path"))
        try_delete(result.get("mask_overlay_path"))
        try_delete(result.get("mask_rgba_path"))

    if mask is None or not np.any(mask):
        if save_black_if_empty:
            out = np.zeros_like(image_np, dtype=np.uint8)
            out_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(image_path))[0] + "_rgbmask.png",
            )
            cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            return {"image": image_path, "status": "empty_black", "rgbmask_path": out_path}
        return {"image": image_path, "status": "mask_empty", "rgbmask_path": None}

    # --- 构建并保存纯 RGB 掩码 ---
    rgb_mask = build_rgb_mask(image_np, mask.astype(bool))
    out_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(image_path))[0] + "_rgbmask.png",
    )
    cv2.imwrite(out_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))
    return {"image": image_path, "status": "ok", "rgbmask_path": out_path}


if __name__ == "__main__":
    # ======== 按需修改这里 ========
    INPUT_DIR  = "/home/MA_SmartGrip/Smartgrip/ur5_image_input"   
    OUTPUT_DIR = "/home/MA_SmartGrip/Smartgrip/ur5_image_input"
    PROMPT     = "yellow cube ."                   # 文本提示

    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    DINO_ID    = "IDEA-Research/grounding-dino-tiny"
    BOX_THRESHOLD  = 0.25
    TEXT_THRESHOLD = 0.25

    SAM2_ID    = "facebook/sam2.1-hiera-large"
    MASK_THRESHOLD      = 0.30
    MAX_HOLE_AREA       = 100.0
    MAX_SPRINKLE_AREA   = 50.0

    SELECT_METHOD = "best_score"   # or "largest"
    SAVE_BLACK_IF_EMPTY = False    # True: 未检出/空掩码也导出黑图；False: 跳过
    DELETE_EXTRAS = True           # 删除 SAM2 生成的灰度/叠加/RGBA文件
    # =============================

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Init] Loading GroundingDINO ({DINO_ID}) on {DEVICE}")
    dino = GroundingDinoPredictor(model_id=DINO_ID, device=DEVICE)

    print(f"[Init] Loading SAM2 ({SAM2_ID}) on {DEVICE}")
    sam2 = SAM2ImagePredictorWrapper(
        model_id=SAM2_ID,
        device=DEVICE,
        mask_threshold=MASK_THRESHOLD,
        max_hole_area=MAX_HOLE_AREA,
        max_sprinkle_area=MAX_SPRINKLE_AREA,
        multimask_output=False,
        return_logits=False,
    )

    files = list_images(INPUT_DIR)
    if not files:
        print(f"[Warn] {INPUT_DIR} 下未找到图片")
        raise SystemExit

    print(f"[Info] 待处理 {len(files)} 张，输出至：{OUTPUT_DIR}")
    ok_cnt = 0
    for i, img_path in enumerate(files, 1):
        info = process_one_image(
            image_path=img_path,
            prompt=PROMPT,
            dino=dino,
            sam2=sam2,
            output_dir=OUTPUT_DIR,
            select_method=SELECT_METHOD,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            save_black_if_empty=SAVE_BLACK_IF_EMPTY,
            delete_extras=DELETE_EXTRAS,
        )
        status = info["status"]
        outp = info["rgbmask_path"]
        if status == "ok":
            ok_cnt += 1
            print(f"[OK] {i}/{len(files)} -> {outp}")
        elif status == "no_detection":
            print(f"[NoDet] {i}/{len(files)} {img_path}")
        elif status == "mask_empty":
            print(f"[EmptyMask] {i}/{len(files)} {img_path}")
        elif status == "empty_black":
            print(f"[Empty->Black] {i}/{len(files)} -> {outp}")
        else:
            print(f"[Skip] {i}/{len(files)} {img_path} | {status}")

    print(f"[Done] 成功输出 {ok_cnt}/{len(files)} 张 RGB 掩码")
