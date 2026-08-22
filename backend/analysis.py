"""
analysis.py — Core pallor/erythema colour analysis.

This is a server-side (OpenCV + NumPy) port of the same colour-science logic
validated in the frontend prototype (index.html), so behaviour is consistent
whether the browser or the backend does the computation.

ROI (region of interest) matches the on-screen capture guide:
top 30%, left 38%, width 24%, height 24% of the frame.
"""

import cv2
import numpy as np

ROI_TOP = 0.30
ROI_LEFT = 0.38
ROI_W = 0.24
ROI_H = 0.24


def extract_roi(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x0 = int(w * ROI_LEFT)
    y0 = int(h * ROI_TOP)
    x1 = x0 + int(w * ROI_W)
    y1 = y0 + int(h * ROI_H)
    x1, y1 = min(x1, w), min(y1, h)
    return image_bgr[y0:y1, x0:x1]


def compute_features(image_bytes: bytes) -> dict:
    """Decode an image and compute pallor/erythema features from the ROI."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    roi = extract_roi(img_bgr)
    if roi.size == 0:
        raise ValueError("ROI is empty — image too small")

    # Mean RGB over the ROI
    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]

    # Mean HSV (OpenCV: H 0-179, S/V 0-255) — normalise S,V to 0..1
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    mean_sat = float(roi_hsv[:, 1].mean() / 255.0)
    mean_val = float(roi_hsv[:, 2].mean() / 255.0)

    # Same erythema/pallor formula as the frontend prototype
    erythema_index = ((r - (g + b) / 2) / 255.0) * (0.5 + mean_sat)
    pallor_score = max(0.0, min(1.0, 0.5 - erythema_index))

    return {
        "avg_r": float(r),
        "avg_g": float(g),
        "avg_b": float(b),
        "saturation": mean_sat,
        "value": mean_val,
        "erythema_index": float(erythema_index),
        "pallor_score": float(pallor_score),
    }


RULE_THRESHOLDS = [
    (0.32, "Normal", "Conjunctiva colour is within the expected healthy range. No elevated anaemia risk detected in this screen."),
    (0.45, "Mild Risk", "Slight pallor detected. Re-screen in better lighting; if it persists, a routine blood test is a good idea."),
    (0.60, "Moderate Risk", "Noticeable pallor detected. We recommend confirming with a clinical blood test soon."),
    (1.01, "Severe Risk", "Significant pallor detected. Please get a blood test at a health facility as soon as possible."),
]


def rule_based_band(pallor_score: float) -> dict:
    for threshold, label, guidance in RULE_THRESHOLDS:
        if pallor_score < threshold:
            return {"band": label, "guidance": guidance, "method": "rule_based"}
    return {"band": "Severe Risk", "guidance": RULE_THRESHOLDS[-1][2], "method": "rule_based"}
