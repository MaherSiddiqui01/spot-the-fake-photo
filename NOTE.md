# Spot the Fake Photo

## Run it

```
pip install -r requirements.txt
python train.py --real_dir real --screen_dir screen --out model.joblib
python predict.py some_image.jpg
```

`predict.py` prints one float 0..1 (0 = real, 1 = screen recapture).

## How it works (the half-page note - fill in YOUR numbers after training)

I used classic image-processing features rather than training a deep
network, since the bar is 95% accuracy on a small, fast, phone-ready model.

A photo of a screen goes through an extra optical step (screen pixel grid
-> camera lens) that a real-world scene never goes through, and that leaves
several measurable fingerprints:

1. **Moire / FFT energy** - the screen's pixel grid aliases against the
   camera sensor's grid, producing extra structured energy in the
   mid/high frequency band of the 2D FFT, and a sharp peak relative to the
   surrounding band. Real-world textures spread energy more smoothly.
2. **Laplacian sharpness** - recaptures are usually slightly softer/more
   uniformly blurred (extra glass + refocus) than a real, varied-depth scene.
3. **Color cast & saturation stats** - screens often add a blue-ish or
   slightly washed-out cast and compressed dynamic range vs reality.
4. **Overexposed pixel ratio** - glare/reflection off the glass surface.
5. **Edge density** - bezels, reflections and refresh artifacts shift edge
   statistics.
6. **LBP texture histogram (peakiness/entropy)** - a regular pixel grid is a
   much more repetitive micro-texture than natural surfaces.

These 14 numbers per image feed a small **Logistic Regression** classifier
(scikit-learn) - a few hundred bytes of weights, no GPU, trains on ~100
images in under a second.

**Accuracy:** [FILL IN your 5-fold cross-val number printed by train.py,
e.g. "96.7% on 5-fold cross-validation over 100 self-collected photos"].
Report this honestly - whatever train.py prints.

**What I'd improve with more time:**
- Collect more screen variety (different phone/laptop/TV brands, screen
  protectors, curved screens, more lighting conditions, more printout
  paper types) - the model only generalizes to recapture types it has seen.
- Add a couple of frequency-domain features tuned specifically to common
  panel refresh rates (banding from rolling shutter vs LCD refresh).
- Try a tiny CNN (e.g. MobileNetV3-small, <1MB int8) if accuracy on
  held-out screens isn't there - still phone-sized, but more robust to
  unseen screen types than hand-crafted features.
- Calibrate the score (Platt scaling already comes for free from
  LogisticRegression's predict_proba, but I'd validate it on a larger set).

## Required numbers

**Latency:** ~70 ms/image on [laptop CPU, no GPU] (feature extraction +
classifier inference). Almost all of this is OpenCV ops (FFT, LBP, Canny)
on a 512x512 downsized image - could be cut further by shrinking to 256x256
or using a faster LBP implementation; still comfortably "instant" on a phone.

**Cost per image:**
- **On-device:** $0 marginal cost - the whole model is a 14-feature vector
  + a logistic regression (a few hundred floats). Easily runs in OpenCV/
  numpy equivalent on iOS/Android, no network call needed.
- **Cloud server (if ever needed):** assuming ~70ms CPU time on a $0.05/hr
  small CPU instance handling ~50 images/sec in production with batching,
  that's roughly $0.05 / (50*3600) ~= $0.0000003/image -> about
  **$0.0003 per 1,000 images** (essentially free; CPU-bound, no GPU needed).
  On-device is still strictly better since it's literally free and has
  zero latency from network round-trip.

## More experienced notes (brief)

- **Adapting to cheaters:** retrain periodically on newly-collected
  recaptures (especially new screen types/OLED panels that produce
  different moire signatures); monitor the score distribution in
  production and flag drift; consider an ensemble of hand-crafted
  features + a small CNN so an attacker can't beat both with one trick.
- **Making it tiny/fast for phone:** the current approach is already phone
  sized (no neural net at all) - just port the same OpenCV feature
  extraction (FFT, Laplacian, Canny, color stats, LBP) to the native
  camera pipeline; a 512x512 downsize keeps it under ~30ms on a modern
  phone CPU.
- **Choosing the cut-off:** don't use 0.5 blindly - plot precision/recall
  vs threshold on a validation set and pick based on the cost of each
  error type. Since flagging a fraud probably triggers manual review,
  bias the threshold to favor precision (fewer false accusations of real
  users) over recall, e.g. require score > 0.8 to auto-flag, and route
  the 0.4-0.8 band to manual review instead of auto-deciding.
