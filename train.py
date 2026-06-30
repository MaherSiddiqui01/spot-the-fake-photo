"""
train.py - trains a tiny Logistic Regression classifier on hand-crafted
features (see features.py) to separate REAL photos vs SCREEN recaptures.

Usage:
    python train.py --real_dir real/ --screen_dir screen/ --out model.joblib

Expects:
    real/   folder of ~50 real-world photos (label 0)
    screen/ folder of ~50 photos-of-a-screen / printout (label 1)

Outputs:
    model.joblib   - contains {scaler, model} ready for predict.py
    Prints cross-validated accuracy so you have an HONEST number to report.
"""

import argparse
import glob
import os
import time

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from features import extract_features, FEATURE_NAMES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_folder(folder, label):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    X, y, used_paths = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [skip, unreadable] {p}")
            continue
        feats = extract_features(img)
        X.append(feats)
        y.append(label)
        used_paths.append(p)
    return X, y, used_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="real")
    ap.add_argument("--screen_dir", default="screen")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    print(f"Loading real photos from {args.real_dir} ...")
    X_real, y_real, p_real = load_folder(args.real_dir, 0)
    print(f"  {len(X_real)} real images loaded")

    print(f"Loading screen photos from {args.screen_dir} ...")
    X_screen, y_screen, p_screen = load_folder(args.screen_dir, 1)
    print(f"  {len(X_screen)} screen images loaded")

    X = np.array(X_real + X_screen)
    y = np.array(y_real + y_screen)
    paths = p_real + p_screen

    if len(X) < 10:
        raise SystemExit("Not enough images. Need real/ and screen/ folders populated.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 5-fold cross validation for an honest accuracy estimate (small dataset)
    n_splits = min(5, min(np.bincount(y)))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=4.0, gamma="scale", probability=True, class_weight="balanced"
        ),
    }

    best_name, best_acc, best_pred = None, -1, None
    for name, model in candidates.items():
        y_pred_cv = cross_val_predict(model, Xs, y, cv=skf)
        acc_cv = accuracy_score(y, y_pred_cv)
        print(f"  [{name}] {n_splits}-fold CV accuracy: {acc_cv*100:.1f}%")
        if acc_cv > best_acc:
            best_name, best_acc, best_pred = name, acc_cv, y_pred_cv

    print(f"\n>>> Best model: {best_name} ({best_acc*100:.1f}% CV accuracy) <<<")
    clf = candidates[best_name]
    y_pred = best_pred
    acc = best_acc
    print(f"\n=== {n_splits}-fold cross-validated accuracy: {acc*100:.1f}% ===")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, target_names=["real", "screen"]))

    # show misclassified files -> useful for the "what I'd improve" note
    wrong = [(paths[i], y[i], y_pred[i]) for i in range(len(y)) if y[i] != y_pred[i]]
    if wrong:
        print("Misclassified:")
        for p, t, pr in wrong:
            print(f"  {p}  true={t} pred={pr}")

    # fit final model on ALL data for shipping
    clf.fit(Xs, y)

    # feature importance (helps explain "how you did it") - only for linear/tree models
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        print("\nFeature weights (positive -> pushes toward SCREEN):")
        for name, c in sorted(zip(FEATURE_NAMES, coefs), key=lambda t: -abs(t[1])):
            print(f"  {name:22s} {c:+.3f}")
    elif hasattr(clf, "feature_importances_"):
        print("\nFeature importances:")
        for name, c in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1]):
            print(f"  {name:22s} {c:.3f}")

    joblib.dump({"scaler": scaler, "model": clf, "cv_accuracy": acc}, args.out)
    print(f"\nSaved model -> {args.out}")

    # quick latency check
    sample_img = cv2.imread(paths[0])
    t0 = time.time()
    for _ in range(20):
        f = extract_features(sample_img)
        scaler.transform([f])
        clf.predict_proba(scaler.transform([f]))
    dt = (time.time() - t0) / 20 * 1000
    print(f"Approx latency per image: {dt:.1f} ms (this machine)")


if __name__ == "__main__":
    main()
