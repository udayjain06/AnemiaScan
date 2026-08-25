"""
analysis.py -- AnemiaScan core analysis module (v2).

Two classification paths:
  1. ML (primary): Loads model_rf.pkl (best of RF/GB/Ensemble trained in v2)
     Uses 37 features: RGB + HSV + LAB + LBP texture + colour ratios
  2. Regression (secondary): Loads model_reg.pkl to predict exact Hgb value,
     then thresholds to Anemic/Mild/Normal
  3. Rule-based (fallback): Original pallor-score thresholds when no model found

Smart input detection:
  - RGBA images composited onto white (transparent background fix)
  - Pre-cropped conjunctiva (>30% white px) vs full eye photo (ROI crop)
"""

import os
import pickle
import logging
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

logger = logging.getLogger(__name__)

# ---- ROI for full-eye photos -------------------------------------------------
ROI_TOP  = 0.30
ROI_LEFT = 0.38
ROI_W    = 0.24
ROI_H    = 0.24

# ---- Labels & guidance -------------------------------------------------------
CLASS_LABELS = ["Anemic", "Mild", "Normal"]

BAND_GUIDANCE = {
    "Normal": "Conjunctiva colour is within the expected healthy range. No elevated anaemia risk detected in this screen.",
    "Mild":   "Slight pallor detected. Re-screen in better lighting; if it persists, a routine blood test is a good idea.",
    "Anemic": "Noticeable pallor detected. Haemoglobin is likely below 11 g/dL. We recommend confirming with a clinical blood test soon.",
}

def hgb_to_label(hgb: float) -> str:
    if hgb < 11.0: return "Anemic"
    elif hgb < 12.0: return "Mild"
    else: return "Normal"

# ---- Model registry ----------------------------------------------------------

_CLASSIFIER_BUNDLE: dict | None = None
_REGRESSION_BUNDLE: dict | None = None
_MODELS_LOADED = False


def _find(filename: str) -> str | None:
    for p in [Path(__file__).parent / filename, Path(filename)]:
        if p.is_file(): return str(p)
    return None


def load_model(clf_path: str | None = None, reg_path: str | None = None) -> bool:
    global _CLASSIFIER_BUNDLE, _REGRESSION_BUNDLE, _MODELS_LOADED
    if _MODELS_LOADED:
        return _CLASSIFIER_BUNDLE is not None
    _MODELS_LOADED = True

    # Classifier
    clf_file = clf_path or _find("model_rf.pkl")
    if clf_file:
        try:
            with open(clf_file, "rb") as f:
                _CLASSIFIER_BUNDLE = pickle.load(f)
            logger.info("Loaded classifier: %s (CV=%.3f test=%.3f feats=%s)",
                        _CLASSIFIER_BUNDLE.get("model_name","?"),
                        _CLASSIFIER_BUNDLE.get("cv_mean", 0),
                        _CLASSIFIER_BUNDLE.get("test_accuracy", 0),
                        _CLASSIFIER_BUNDLE.get("n_features", "?"))
        except Exception as e:
            logger.error("Classifier load failed: %s", e)

    # Regression
    reg_file = reg_path or _find("model_reg.pkl")
    if reg_file:
        try:
            with open(reg_file, "rb") as f:
                _REGRESSION_BUNDLE = pickle.load(f)
            logger.info("Loaded regression model: %s (MAE=%.2f g/dL)",
                        _REGRESSION_BUNDLE.get("model_name","?"),
                        _REGRESSION_BUNDLE.get("test_mae", 0))
        except Exception as e:
            logger.error("Regression load failed: %s", e)

    if not _CLASSIFIER_BUNDLE:
        logger.warning("No classifier found — using rule-based fallback.")
    return _CLASSIFIER_BUNDLE is not None


def is_ml_available() -> bool:
    return _CLASSIFIER_BUNDLE is not None

def is_regression_available() -> bool:
    return _REGRESSION_BUNDLE is not None

def classifier_info() -> str:
    if _CLASSIFIER_BUNDLE:
        name = _CLASSIFIER_BUNDLE.get("model_name", "unknown")
        cv   = _CLASSIFIER_BUNDLE.get("cv_mean", 0)
        return f"{name}_v2 (CV={cv:.1%})"
    return "rule_based_v0"


# ---- Image decoding (RGBA-safe) ----------------------------------------------

def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError("Could not decode image")
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3:4].astype(np.float64) / 255.0
        bgr   = img_raw[:, :, :3].astype(np.float64)
        return (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img_raw.ndim == 2:
        return cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    return img_raw


def extract_roi(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x0 = int(w * ROI_LEFT); y0 = int(h * ROI_TOP)
    x1 = min(x0 + int(w * ROI_W), w)
    y1 = min(y0 + int(h * ROI_H), h)
    return image_bgr[y0:y1, x0:x1]


# ---- 37-feature extraction (matches train_model.py v2) ----------------------

def extract_colour_features(roi_bgr: np.ndarray, mask_white: bool = True) -> np.ndarray:
    """
    Extract 37-feature vector from a BGR image region.

    Features:
        [0:6]   RGB mean + std
        [6:12]  HSV mean + std
        [12:15] Colour ratios r/g, r/b, g/b
        [15:17] Erythema index, pallor score
        [17]    Tissue area fraction
        [18:27] LAB mean + std (L, a*, b*)
        [27:37] LBP texture histogram (10 bins)
    """
    if mask_white:
        tissue_mask = ~np.all(roi_bgr > 240, axis=2)
    else:
        tissue_mask = np.ones(roi_bgr.shape[:2], dtype=bool)

    tissue_count = int(tissue_mask.sum())
    if tissue_count < 50:
        raise ValueError("Insufficient tissue pixels — image may be overexposed or too small.")

    tissue_bgr = roi_bgr[tissue_mask].astype(np.float64)
    b_px, g_px, r_px = tissue_bgr[:,0], tissue_bgr[:,1], tissue_bgr[:,2]

    # RGB
    mean_r, mean_g, mean_b = r_px.mean(), g_px.mean(), b_px.mean()
    std_r,  std_g,  std_b  = r_px.std(),  g_px.std(),  b_px.std()

    # HSV
    roi_hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    t_hsv     = roi_hsv[tissue_mask].astype(np.float64)
    h_px      = t_hsv[:,0] * (360.0 / 180.0)
    s_px      = t_hsv[:,1] / 255.0
    v_px      = t_hsv[:,2] / 255.0
    mean_h, mean_s, mean_v = h_px.mean(), s_px.mean(), v_px.mean()
    std_h,  std_s,  std_v  = h_px.std(),  s_px.std(),  v_px.std()

    # Ratios + derived
    eps = 1e-6
    rg_ratio       = mean_r / (mean_g + eps)
    rb_ratio       = mean_r / (mean_b + eps)
    gb_ratio       = mean_g / (mean_b + eps)
    erythema_index = ((mean_r - (mean_g + mean_b) / 2.0) / 255.0) * (0.5 + mean_s)
    pallor_score   = max(0.0, min(1.0, 0.5 - erythema_index))
    area_fraction  = tissue_count / max(roi_bgr.shape[0] * roi_bgr.shape[1], 1)

    # LAB
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    t_lab   = roi_lab[tissue_mask].astype(np.float64)
    L_px    = t_lab[:,0] / 255.0
    a_px    = (t_lab[:,1] - 128.0) / 127.0
    b_px2   = (t_lab[:,2] - 128.0) / 127.0
    mean_L, mean_a, mean_b2 = L_px.mean(), a_px.mean(), b_px2.mean()
    std_L,  std_a,  std_b2  = L_px.std(),  a_px.std(),  b_px2.std()

    # LBP texture
    if HAS_SKIMAGE:
        grey = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        lbp  = local_binary_pattern(grey, P=8, R=1, method="uniform")
        lbp_tissue = lbp[tissue_mask]
        hist, _ = np.histogram(lbp_tissue, bins=10, range=(0, 10), density=True)
        lbp_feats = hist.astype(np.float32)
    else:
        lbp_feats = np.zeros(10, dtype=np.float32)

    base = np.array([
        mean_r, mean_g, mean_b, std_r, std_g, std_b,
        mean_h, mean_s, mean_v, std_h, std_s, std_v,
        rg_ratio, rb_ratio, gb_ratio,
        erythema_index, pallor_score, area_fraction,
        mean_L, mean_a, mean_b2, std_L, std_a, std_b2,
        std_L, std_a, std_b2,   # indices 24-26 (std duplicates, overwritten below)
    ], dtype=np.float32)
    base[24] = float(std_L)
    base[25] = float(std_a)
    base[26] = float(std_b2)

    return np.concatenate([base, lbp_feats])  # 37 total


# ---- Public API --------------------------------------------------------------

def compute_features(image_bytes: bytes) -> dict:
    """
    Decode image bytes, auto-detect pre-cropped vs full-eye, extract 37 features.
    Returns a dict with public feature keys + internal _colour_features.
    """
    img_bgr = _decode_to_bgr(image_bytes)

    white_frac   = float(np.all(img_bgr > 240, axis=2).mean())
    is_precropped = white_frac > 0.30

    if is_precropped:
        roi       = img_bgr
        mask_white = True
    else:
        roi = extract_roi(img_bgr)
        if roi.size == 0:
            raise ValueError("ROI is empty — image too small")
        mask_white = False

    feats = extract_colour_features(roi, mask_white=mask_white)

    return {
        "avg_r":           float(feats[0]),
        "avg_g":           float(feats[1]),
        "avg_b":           float(feats[2]),
        "saturation":      float(feats[7]),
        "value":           float(feats[8]),
        "erythema_index":  float(feats[15]),
        "pallor_score":    float(feats[16]),
        "lab_a":           float(feats[19]),   # redness in LAB space
        "lab_L":           float(feats[18]),   # lightness
        "_colour_features": feats,
    }


def ml_classify(features: dict) -> dict:
    """Run ML classifier (best available: Ensemble/GB/RF) on extracted features."""
    if not is_ml_available():
        return rule_based_band(features["pallor_score"])

    bundle   = _CLASSIFIER_BUNDLE
    pipeline = bundle["pipeline"]
    feat_vec = features.get("_colour_features")

    # If model was trained with 18 features (old v1) and we now have 37, pad/trim
    expected_n = bundle.get("n_features", 18)
    if feat_vec is not None and len(feat_vec) != expected_n:
        if len(feat_vec) > expected_n:
            feat_vec = feat_vec[:expected_n]
        else:
            feat_vec = np.pad(feat_vec, (0, expected_n - len(feat_vec)))

    if feat_vec is None:
        return rule_based_band(features["pallor_score"])

    X = feat_vec.reshape(1, -1)
    try:
        label   = pipeline.predict(X)[0]
        proba   = pipeline.predict_proba(X)[0]
        conf    = float(proba.max())
        classes = pipeline.classes_
        proba_d = {str(c): round(float(p), 3) for c, p in zip(classes, proba)}
    except Exception as e:
        logger.error("ML inference failed: %s", e)
        return rule_based_band(features["pallor_score"])

    # Also get regression Hgb estimate if available
    hgb_estimate = None
    if is_regression_available():
        try:
            hgb_estimate = round(float(_REGRESSION_BUNDLE["pipeline"].predict(X)[0]), 1)
        except Exception:
            pass

    return {
        "band":          label,
        "guidance":      BAND_GUIDANCE.get(label, ""),
        "method":        bundle.get("model_name", "random_forest") + "_v2",
        "confidence":    round(conf, 3),
        "probabilities": proba_d,
        "hgb_estimate":  hgb_estimate,  # predicted Hgb g/dL (None if no regression model)
    }


# ---- Rule-based fallback (3-class) ------------------------------------------

RULE_THRESHOLDS = [
    (0.38, "Normal", BAND_GUIDANCE["Normal"]),
    (0.47, "Mild",   BAND_GUIDANCE["Mild"]),
    (1.01, "Anemic", BAND_GUIDANCE["Anemic"]),
]

def rule_based_band(pallor_score: float) -> dict:
    for threshold, label, guidance in RULE_THRESHOLDS:
        if pallor_score < threshold:
            return {"band": label, "guidance": guidance,
                    "method": "rule_based", "confidence": None, "hgb_estimate": None}
    return {"band": "Anemic", "guidance": BAND_GUIDANCE["Anemic"],
            "method": "rule_based", "confidence": None, "hgb_estimate": None}
