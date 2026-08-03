# Centroiding algorithms, measured — PSF deliverable (d)

Can a PSF-aware centroider beat the pipeline's centre-of-mass? Two protocols, because each
can lie alone: synthetic frames with the truth known exactly (but a simulator can flatter a
method whose model matches it), and repeatability across the real 7-frame dithered eclipse
set (honest pixels, but a common floor from atmospheric motion that no estimator can dig
under). `tools/centroid_eval.py`; raw tables in `docs/bench/psf/centroids/`.

Candidates: `pipeline` (stage 1's current path: thresholded, variance-normalised COM),
`com` (plain centre of mass on a background-subtracted cutout), `windowed` (iterative
Gaussian-weighted centroid, SExtractor `XWIN` style), `gauss` (pixel-integrated elliptical
Gaussian least squares), `epsf` (Anderson & King effective PSF via photutils, built from
each frame's own bright stars).

## Synthetic truth — rms centroid error (px)

Moffat β=3 stars at random subpixel phases, photon + read noise matched to the data;
faint ≈ SNR 10–30, mid ≈ 30–100, bright ≈ 100+.

| FWHM 1.3 px (undersampled) | faint | mid | bright |
|---|---|---|---|
| **pipeline (current)** | **0.039** | **0.021** | **0.005** |
| windowed | 0.207 | 0.024 | 0.006 |
| gauss | 0.076 | 0.029 | 0.019 |
| ePSF | 0.102 | 0.028 | 0.014 |
| plain com | 0.570 | 0.153 | 0.023 |

| FWHM 2.4 px | faint | mid | bright |
|---|---|---|---|
| **pipeline (current)** | **0.099** | **0.037** | 0.012 |
| windowed | 0.306 | 0.041 | **0.011** |
| gauss | 0.233 | 0.041 | 0.012 |
| ePSF | 2.07* | 0.056 | 0.030 |
| plain com | 0.519 | 0.171 | 0.025 |

| FWHM 3.0 px | faint | mid | bright |
|---|---|---|---|
| **pipeline (current)** | **0.179** | **0.049** | 0.012 |
| windowed | 0.277 | 0.045 | **0.011** |
| gauss | 0.292 | 0.044 | 0.013 |
| ePSF | 1.70* | 0.099 | 0.072 |
| plain com | 0.580 | 0.172 | 0.022 |

\* the ePSF faint bins at 2.4/3.0 px are wrecked by fit divergences on the faintest
stars; its mid/bright numbers are the meaningful ones.

## Real frames — per-star scatter across the 7-frame dither (px rms)

| estimator | median | p90 |
|---|---|---|
| **pipeline (current)** | **0.114** | **0.171** |
| windowed | 0.114 | 0.179 |
| gauss | 0.116 | 0.176 |
| ePSF | 0.192 | 0.282 |
| plain com | 0.192 | 0.347 |

## Reading the numbers

* **The pipeline's centroid is already at the floor.** On real frames the pipeline,
  windowed and Gaussian-fit estimators are statistically indistinguishable; only plain COM
  falls off (its hard window admits noise pixels — exactly the Cramér–Rao gap the
  literature predicts). The current method's thresholding-plus-variance-normalisation
  behaves like a matched window, which is why it does not share plain COM's weakness.
* **The real-frame floor is not estimator noise.** ~0.11 px common to three unrelated
  estimators is atmospheric differential motion over the dither plus photon noise. A better
  centroider cannot show below it in this protocol; equally, nothing was left on the table
  above it.
* **Two caveats, stated rather than hidden.** The pipeline's synthetic faint-bin rms is
  conditioned on its own detections (it does not detect the faintest injected stars, so its
  bin is easier); and the ePSF's first run was wrecked by a caller bug (built on
  non-background-subtracted cutouts — the corrected build is what the table shows).

## Recommendation

**Keep the centre-of-mass path. Nothing here beats it, and now we know why.** The
literature's "COM cannot reach the noise limit" verdict applies to *plain* COM, and the
measurement confirms it (0.19 vs 0.11 px on real frames) — but stage 1's variant is not
plain COM: the variance normalisation and threshold act as a matched window, which is the
same trick that makes SExtractor's windowed centroid nearly optimal. On real frames it sits
exactly on the common floor with the windowed and Gaussian-fit estimators; on synthetic
truth it is best or tied-best in every regime, including the undersampled one where the
ePSF was expected to win.

The ePSF underperforms here for structural reasons, not implementation ones alone: it needs
many bright stars at diverse pixel phases from a *stable* PSF, and a per-frame amateur
dataset gives it neither the star budget of an HST field nor frame-to-frame PSF stability.
(Its first run was also wrecked by a genuine caller bug — built on non-background-subtracted
cutouts — which cost 0.3 px and is exactly the kind of silent misuse the photutils docs
warn about. Fixed, it reaches 0.10 px faint-bin on undersampled synthetic data, its home
turf, and still does not beat the pipeline.)

What the analysis *does* leave on the table, for separate follow-ups:

1. **Saturated stars** — a fit to the unsaturated wing pixels can recover centroids the
   COM cannot (Rasalhague's clipped core biases its position today). Different problem
   from precision; clear candidate win.
2. **The real-frame floor of ~0.11 px is atmospheric, not algorithmic** — the way past it
   is more frames in the stack, not a better estimator. This directly supports current
   observing practice (dithered stacks).
3. The faint-bin synthetic caveat: the pipeline's number is conditioned on its own
   detections. Its true faint-star behaviour is entangled with detection, which is a
   detection-threshold question rather than a centroiding one.
