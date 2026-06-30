"""
features.py - extract hand-crafted features that distinguish a REAL photo
from a PHOTO OF A SCREEN (recapture).

Why these features work:
- Moire / FFT high-freq energy: screens have a regular pixel/subpixel grid.
  Photographing that grid with another camera creates aliasing -> moire
  patterns -> extra energy in the high-frequency band of the 2D FFT that
  real-world textures don't produce in the same structured way.
- Laplacian variance (sharpness): recaptures go through an extra optical
  path (screen -> air -> camera lens) so they are usually slightly softer
  / have more uniform blur than a real scene with varied depth.
- Color cast / saturation stats: screens (esp. phone/laptop) often shift
  white balance, have a blue-ish or warm cast, and compressed/clipped
  color range compared to real scenes.
- Edge density & local contrast: screen bezels, reflections and refresh
  banding create distinct edge statistics.
- Local Binary Pattern (LBP) texture histogram: captures the fine repetitive
  texture of a pixel grid vs natural micro-texture.
- Specular highlight / overexposed pixel ratio: glare off glass screens.

All features are cheap (just numpy/opencv ops on a downsized image), so
total feature extraction is a few ms per image -> fits the "fast, runs on
phone" requirement.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def _fft_highfreq_energy(gray):
    """Ratio of energy in high spatial frequencies vs total energy.
    Moire patterns from screen pixel grids show up as energy concentrated
    in a ring/band of frequencies, not just noise -> use ring energy."""
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)

    low_mask = dist < max_r * 0.15
    mid_mask = (dist >= max_r * 0.15) & (dist < max_r * 0.5)
    high_mask = dist >= max_r * 0.5

    total = mag.sum() + 1e-6
    low_e = mag[low_mask].sum() / total
    mid_e = mag[mid_mask].sum() / total
    high_e = mag[high_mask].sum() / total

    # peak-to-mean ratio in the mid/high band catches sharp moire spikes
    band = mag[mid_mask | high_mask]
    peak_ratio = (band.max() / (band.mean() + 1e-6)) if band.size else 0.0

    return low_e, mid_e, high_e, peak_ratio


def _laplacian_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _color_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    # blue cast common in screen photos (LED backlight)
    blue_cast = (b.mean() - r.mean())
    sat_mean, sat_std = sat.mean(), sat.std()
    val_mean, val_std = val.mean(), val.std()
    # overexposed/clipped pixel ratio (glare off glass)
    overexposed_ratio = float((val > 250).mean())
    return blue_cast, sat_mean, sat_std, val_mean, val_std, overexposed_ratio


def _edge_density(gray):
    edges = cv2.Canny(gray, 80, 160)
    return edges.mean() / 255.0


def _lbp_uniformity(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    # uniformity = how peaked the histogram is (repetitive grid -> peaky)
    return float(hist.max()), float(-(hist * np.log(hist + 1e-9)).sum())  # max bin, entropy


def extract_features(img_bgr):
    """img_bgr: numpy array (H,W,3) BGR (as read by cv2.imread).
    Returns a fixed-length 1D float feature vector."""
    img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    low_e, mid_e, high_e, peak_ratio = _fft_highfreq_energy(gray)
    sharp = _laplacian_sharpness(gray)
    blue_cast, sat_mean, sat_std, val_mean, val_std, overexp = _color_stats(img_bgr)
    edge_d = _edge_density(gray)
    lbp_max, lbp_entropy = _lbp_uniformity(gray)

    feats = np.array([
        low_e, mid_e, high_e, peak_ratio,
        sharp,
        blue_cast, sat_mean, sat_std, val_mean, val_std, overexp,
        edge_d,
        lbp_max, lbp_entropy,
    ], dtype=np.float32)
    return feats


FEATURE_NAMES = [
    "fft_low_e", "fft_mid_e", "fft_high_e", "fft_peak_ratio",
    "laplacian_sharpness",
    "blue_cast", "sat_mean", "sat_std", "val_mean", "val_std", "overexposed_ratio",
    "edge_density",
    "lbp_max_bin", "lbp_entropy",
]