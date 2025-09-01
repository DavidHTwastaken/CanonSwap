# coding: utf-8


import cv2
import numpy as np
from PIL import Image
import os

def add_image_watermark(image, watermark_path, position="bottom_right", opacity=0.2, scale=0.3):

    if image is None or not os.path.exists(watermark_path):
        return image

    # 读取水印图片
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        return image

    h, w = image.shape[:2]

    wm_h, wm_w = watermark.shape[:2]
    new_wm_w = int(w * scale)
    new_wm_h = int(wm_h * new_wm_w / wm_w)

    watermark_resized = cv2.resize(watermark, (new_wm_w, new_wm_h))

    margin = 20
    if position == "bottom_right":
        x = w - new_wm_w - margin
        y = h - new_wm_h - margin
    elif position == "bottom_left":
        x = margin
        y = h - new_wm_h - margin
    elif position == "top_right":
        x = w - new_wm_w - margin
        y = margin
    elif position == "top_left":
        x = margin
        y = margin
    else:
        x = w - new_wm_w - margin
        y = h - new_wm_h - margin

    x = max(0, min(x, w - new_wm_w))
    y = max(0, min(y, h - new_wm_h))

    result = image.copy()

    if watermark_resized.shape[2] == 4:
        wm_rgb = watermark_resized[:, :, :3]
        wm_alpha = watermark_resized[:, :, 3] / 255.0 * opacity

        for c in range(3):
            result[y:y+new_wm_h, x:x+new_wm_w, c] = (
                result[y:y+new_wm_h, x:x+new_wm_w, c] * (1 - wm_alpha) +
                wm_rgb[:, :, c] * wm_alpha
            )
    else:
        wm_rgb = watermark_resized
        alpha = opacity

        result[y:y+new_wm_h, x:x+new_wm_w] = (
            result[y:y+new_wm_h, x:x+new_wm_w] * (1 - alpha) +
            wm_rgb * alpha
        )

    return result.astype(np.uint8)

def add_watermark_to_frame_list(frame_list, watermark_path, opacity=0.2):
    if not frame_list or not os.path.exists(watermark_path):
        return frame_list

    watermarked_frames = []
    for frame in frame_list:
        watermarked_frame = add_image_watermark(frame, watermark_path, opacity=opacity)
        watermarked_frames.append(watermarked_frame)

    return watermarked_frames
