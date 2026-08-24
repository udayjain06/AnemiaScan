"""
train_model.py -- Train a Random Forest anemia classifier from the Kaggle
palpebral conjunctiva dataset (India + Italy cohorts).

Usage:
    python train_model.py --dataset "D:/anemia_dataset/dataset anemia"

Outputs:
    backend/model_rf.pkl       -- the trained scikit-learn pipeline
    backend/model_report.txt   -- cross-validation + test-set metrics

3-class labelling (WHO-based, adults):
    Anemic : Hgb < 11.0 g/dL   (merges Severe + Moderate for enough samples)
    Mild   : 11.0 <= Hgb < 12.0 g/dL
    Normal : Hgb >= 12.0 g/dL
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---- Constants ---------------------------------------------------------------

CLASS_LABELS = ["Anemic", "Mild", "Normal"]

# Map label string -> integer used internally
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}

# Hgb thresholds for 3-class labelling (g/dL)
def hgb_to_label(hgb: float) -> str:
    if hgb < 11.0:
        return "Anemic"     # covers Severe (<8) + Moderate (8-10.9)
    elif hgb < 12.0:
        return "Mild"
    else:
        return "Normal"


# ─── Image feature extraction ────────────────────────────────────────────────

def extract_features(image_path: str) -> np.ndarray | None:
    """
    Extract 18 colour features from a pre-segmented palpebral conjunctiva image.

    The images have a white background (pixel value > 240 in all channels).
    We mask out white pixels so we only analyse conjunctiva tissue.

    Features (18 total):
        RGB: mean_r, mean_g, mean_b, std_r, std_g, std_b
        HSV: mean_h, mean_s, mean_v, std_h, std_s, std_v
        Ratios: r/(g+1), r/(b+1), g/(b+1)
        Derived: erythema_index, pallor_score, rg_ratio
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        return None

    # Composite RGBA images onto a white background so transparent regions
    # become white and are excluded by the tissue mask below.
    # Images with only 3 channels (BGR) are used as-is.
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha = img_raw[:, :, 3:4].astype(np.float64) / 255.0
        bgr = img_raw[:, :, :3].astype(np.float64)
        img_bgr = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        img_bgr = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)

    # Build non-white mask — keep pixels where not all channels are near-white
    white_mask = np.all(img_bgr > 240, axis=2)  # True = white bg
    tissue_mask = ~white_mask

    if tissue_mask.sum() < 50:  # too few pixels
        return None

    tissue_bgr = img_bgr[tissue_mask].astype(np.float64)  # shape (N, 3)
    b_px, g_px, r_px = tissue_bgr[:, 0], tissue_bgr[:, 1], tissue_bgr[:, 2]

    # RGB stats
    mean_r, mean_g, mean_b = r_px.mean(), g_px.mean(), b_px.mean()
    std_r, std_g, std_b = r_px.std(), g_px.std(), b_px.std()

    # HSV stats
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    tissue_hsv = img_hsv[tissue_mask].astype(np.float64)
    h_px = tissue_hsv[:, 0] * (360.0 / 180.0)   # OpenCV H is 0-179 → normalise to 0-360
    s_px = tissue_hsv[:, 1] / 255.0              # 0-1
    v_px = tissue_hsv[:, 2] / 255.0              # 0-1

    mean_h, mean_s, mean_v = h_px.mean(), s_px.mean(), v_px.mean()
    std_h, std_s, std_v = h_px.std(), s_px.std(), v_px.std()

    # Colour ratios and derived features
    eps = 1e-6
    rg_ratio = mean_r / (mean_g + eps)
    rb_ratio = mean_r / (mean_b + eps)
    gb_ratio = mean_g / (mean_b + eps)

    # Same erythema formula as the original analysis.py (but on actual tissue)
    erythema_index = ((mean_r - (mean_g + mean_b) / 2.0) / 255.0) * (0.5 + mean_s)
    pallor_score = max(0.0, min(1.0, 0.5 - erythema_index))

    # Tissue area fraction
    area_fraction = tissue_mask.sum() / (img_bgr.shape[0] * img_bgr.shape[1])

    features = np.array([
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v,
        rg_ratio, rb_ratio, gb_ratio,
        erythema_index, pallor_score,
        area_fraction,
    ], dtype=np.float32)

    return features


FEATURE_NAMES = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "mean_h", "mean_s", "mean_v",
    "std_h", "std_s", "std_v",
    "rg_ratio", "rb_ratio", "gb_ratio",
    "erythema_index", "pallor_score",
    "area_fraction",
]


# ─── Dataset loading ─────────────────────────────────────────────────────────

def load_cohort(cohort_dir: str, excel_name: str) -> tuple[list, list]:
    """Load one cohort (India or Italy) and return (feature_rows, labels)."""
    excel_path = os.path.join(cohort_dir, excel_name)
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # Build patient_id → hgb map
    hgb_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid, hgb_raw = row[0], row[1]
        if pid is None or hgb_raw is None:
            continue
        try:
            hgb = float(str(hgb_raw).replace(",", "."))
            hgb_map[int(pid)] = hgb
        except (ValueError, TypeError):
            continue

    X, y = [], []
    skipped = []

    for pid, hgb in hgb_map.items():
        patient_dir = os.path.join(cohort_dir, str(pid))
        if not os.path.isdir(patient_dir):
            skipped.append((pid, "no folder"))
            continue

        # Find palpebral image — prefer exact *_palpebral.png over *_papebral.png (typo in dataset)
        palpebral_file = None
        for fname in os.listdir(patient_dir):
            lower = fname.lower()
            # Must end with _palpebral.png or _papebral.png but NOT contain "forniceal"
            if lower.endswith(".png") and "forniceal" not in lower and (
                "palpebral" in lower or "papebral" in lower
            ):
                # Prefer the one without "(1)" suffix
                if palpebral_file is None or "(" not in fname:
                    palpebral_file = fname

        if palpebral_file is None:
            skipped.append((pid, "no palpebral image"))
            continue

        img_path = os.path.join(patient_dir, palpebral_file)
        feats = extract_features(img_path)
        if feats is None:
            skipped.append((pid, "feature extraction failed"))
            continue

        label = hgb_to_label(hgb)
        X.append(feats)
        y.append(label)

    if skipped:
        print(f"  [Cohort {excel_name}] Skipped {len(skipped)} patients: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    return X, y


def load_dataset(dataset_root: str) -> tuple[np.ndarray, np.ndarray]:
    """Load India + Italy cohorts and combine."""
    india_dir = os.path.join(dataset_root, "India")
    italy_dir = os.path.join(dataset_root, "Italy")

    print("Loading India cohort...")
    X_in, y_in = load_cohort(india_dir, "India.xlsx")
    print(f"  Loaded {len(X_in)} samples from India")

    print("Loading Italy cohort...")
    X_it, y_it = load_cohort(italy_dir, "Italy.xlsx")
    print(f"  Loaded {len(X_it)} samples from Italy")

    X = np.array(X_in + X_it, dtype=np.float32)
    y = np.array(y_in + y_it)

    # Print class distribution
    from collections import Counter
    dist = Counter(y)
    print(f"\nClass distribution (N={len(y)}):")
    for lbl in CLASS_LABELS:
        print(f"  {lbl:10s}: {dist.get(lbl, 0)}")

    return X, y


# ─── Training ────────────────────────────────────────────────────────────────

def train(dataset_root: str, output_dir: str) -> None:
    print(f"\n{'='*60}")
    print("AnemiaScan — Random Forest Classifier Training")
    print(f"{'='*60}\n")

    # Load data
    X, y = load_dataset(dataset_root)
    if len(X) < 10:
        print("ERROR: Too few samples to train. Check dataset path.")
        sys.exit(1)

    # Train / test split (stratified 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Build scikit-learn pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",   # handles class imbalance (few Severe)
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Stratified k-fold cross-validation on training data
    print("\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full training set
    print("\nFitting final model on full training set...")
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    test_acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_acc:.3f}")

    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)

    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print("          " + "  ".join(f"{l:10s}" for l in CLASS_LABELS))
    for lbl, row in zip(CLASS_LABELS, cm):
        print(f"{lbl:10s}" + "  ".join(f"{v:10d}" for v in row))

    # Feature importances (top 10)
    importances = pipeline.named_steps["clf"].feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\nTop-10 feature importances:")
    for name, imp in ranked[:10]:
        print(f"  {name:20s}: {imp:.4f}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "model_rf.pkl")
    model_bundle = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LABELS,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(test_acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=5)
    print(f"\n[OK] Model saved -> {model_path}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("AnemiaScan Random Forest — Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset_root}\n")
        f.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")
        f.write(f"CV accuracy (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
        f.write(f"Test accuracy: {test_acc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        header = "          " + "  ".join(f"{l:10s}" for l in CLASS_LABELS)
        f.write(header + "\n")
        for lbl, row in zip(CLASS_LABELS, cm):
            f.write(f"{lbl:10s}" + "  ".join(f"{v:10d}" for v in row) + "\n")
        f.write("\nFeature Importances:\n")
        for name, imp in ranked:
            f.write(f"  {name:20s}: {imp:.4f}\n")
    print(f"[OK] Report saved -> {report_path}")
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnemiaScan RF classifier")
    parser.add_argument(
        "--dataset",
        default=r"D:\anemia_dataset\dataset anemia",
        help="Path to the extracted Kaggle dataset root (contains India/ and Italy/ subdirs)",
    )
    parser.add_argument(
        "--output",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for model_rf.pkl and model_report.txt (default: this script's dir)",
    )
    args = parser.parse_args()
    train(args.dataset, args.output)
