import cv2
import numpy as np
import os

FRAME_DIR = "frames"
RUNTIME_DIR = "runtime"

def resize_cover(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    x = (nw - target_w) // 2
    y = (nh - target_h) // 2
    return resized[y:y+target_h, x:x+target_w]

def compose_preview(frame_id: str, shot_index: int):
    frame_path = os.path.join(FRAME_DIR, f"{frame_id}.png")
    mask_path  = os.path.join(FRAME_DIR, f"{frame_id}_mask.png")

    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    canvas = frame.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for i in range(shot_index):
        shot_path = os.path.join(RUNTIME_DIR, f"shot_{i+1}.jpg")
        if not os.path.exists(shot_path):
            continue

        shot = cv2.imread(shot_path)

        x, y, w, h = cv2.boundingRect(contours[i])
        fitted = resize_cover(shot, w, h)

        roi = canvas[y:y+h, x:x+w]

        alpha = mask[y:y+h, x:x+w] / 255.0
        alpha = alpha[..., None]

        roi[:] = (fitted * alpha + roi * (1 - alpha)).astype(np.uint8)

    preview_path = os.path.join(RUNTIME_DIR, "preview.jpg")
    cv2.imwrite(preview_path, canvas)
    return preview_path
