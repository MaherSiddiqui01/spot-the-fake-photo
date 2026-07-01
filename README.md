# Spot the Fake Photo — Submission Note

## What I built
Classifies an image as a real photo (0) or photo of a screen (1) using 14 hand-crafted image features — FFT moire energy, Laplacian sharpness, color cast, overexposed pixel ratio, edge density, and LBP texture — fed into a Random Forest classifier. No deep learning, no GPU.

- **Accuracy:** 98.0% (5-fold CV, 646 photos: 414 real / 232 screen)
- **Latency:** ~150ms/image, laptop CPU
- **Cost:** $0 on-device · ~$0.0006 per 1,000 images on cloud

## What I'd improve
- Add features tuned to OLED/LCD refresh banding and print/paper texture to catch the 11 missed screen photos
- Collect more variety: curved screens, screen protectors, different lighting
- Replace Random Forest with MobileNetV3-small (<1MB int8) for better generalization to unseen screen types while staying phone-sized

## Installation

```
pip install opencv-python numpy scikit-learn scikit-image joblib
```

Place `model.joblib`, `predict.py`, and `features.py` in the same folder, then:

```
python predict.py some_image.jpg
```

Returns a score 0–1 (0 = real, 1 = screen).
