#!/usr/bin/env python3
"""
predict.py - one-line predictor.

Usage:
    python predict.py some_image.jpg
    -> prints a single float 0..1   (0 = real photo, 1 = photo of a screen)

Loads model.joblib (trained by train.py) and scores a new image using the
same hand-crafted features (FFT/moire energy, sharpness, color cast, edge
density, LBP texture - see features.py for the why).
"""

import sys
import time

import cv2
import joblib

from features import extract_features

MODEL_PATH = "model.joblib"


def predict(image_path, bundle=None):
    if bundle is None:
        bundle = joblib.load(MODEL_PATH)
    scaler, clf = bundle["scaler"], bundle["model"]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    feats = extract_features(img)
    feats_s = scaler.transform([feats])
    # class 1 = "screen" probability
    score = clf.predict_proba(feats_s)[0][1]
    return float(score)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    t0 = time.time()
    score = predict(sys.argv[1], bundle)
    dt_ms = (time.time() - t0) * 1000

    print(f"{score:.4f}")
    sys.stderr.write(f"(latency: {dt_ms:.1f} ms)\n")
