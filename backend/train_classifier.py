"""
train_classifier.py — Trains a real scikit-learn classifier for AnemiaScan.

IMPORTANT / HONEST DISCLOSURE:
No public conjunctiva-image dataset was reachable from this build environment
today (IEEE DataPort / Kaggle-hosted sets require external auth/downloads not
available here). To move from pure if/else thresholds to an actual trained
model *today*, this script generates a synthetic calibration dataset built
from the same colour-science relationship validated in analysis.py (pallor
score correlates with erythema index, saturation, value), plus randomised
noise to avoid the model trivially re-deriving the thresholds.

This is a v0 classifier for architecture/pipeline demonstration — swapping in
a model trained on a real public conjunctiva dataset (e.g. Eyes-Defy-Anemia,
CP-AnemiC) is the first item in our roadmap to final submission. This
disclosure is intentional: the hackathon rulebook requires honest submission
documents, and a real-trained-on-real-data claim would be false.
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)
N_SAMPLES = 2000

BANDS = ["Normal", "Mild Risk", "Moderate Risk", "Severe Risk"]


def synth_dataset(n=N_SAMPLES):
    """Generate a synthetic, labelled feature set for v0 training."""
    # Ground-truth "true pallor" drawn across the full range
    true_pallor = RNG.uniform(0.0, 1.0, n)

    # Observed features are noisy functions of true_pallor, mimicking how
    # lighting/skin-tone/camera variance would perturb real measurements.
    erythema_index = (0.5 - true_pallor) + RNG.normal(0, 0.04, n)
    saturation = np.clip(0.55 - 0.3 * true_pallor + RNG.normal(0, 0.05, n), 0, 1)
    value = np.clip(0.6 + RNG.normal(0, 0.08, n), 0, 1)
    avg_r = np.clip(180 - 60 * true_pallor + RNG.normal(0, 8, n), 0, 255)
    avg_g = np.clip(110 + 20 * true_pallor + RNG.normal(0, 8, n), 0, 255)
    avg_b = np.clip(105 + 20 * true_pallor + RNG.normal(0, 8, n), 0, 255)

    pallor_score = np.clip(0.5 - erythema_index, 0, 1)

    # Labels from the *true* underlying pallor (not the noisy observed score)
    # with thresholds jittered per-sample to simulate inter-rater/clinical
    # boundary uncertainty rather than a hard cutoff.
    jitter = RNG.normal(0, 0.03, n)
    t1 = 0.32 + jitter
    t2 = 0.45 + jitter
    t3 = 0.60 + jitter
    labels = np.select(
        [true_pallor < t1, true_pallor < t2, true_pallor < t3],
        [0, 1, 2],
        default=3,
    )

    X = np.column_stack([avg_r, avg_g, avg_b, saturation, value, erythema_index, pallor_score])
    y = labels
    return X, y


def main():
    X, y = synth_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy (synthetic v0 data): {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=BANDS))

    joblib.dump({"model": clf, "bands": BANDS}, "classifier.pkl")
    print("Saved classifier.pkl")


if __name__ == "__main__":
    main()
