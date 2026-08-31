# Field presets: the two standard stage-1 configurations

**Written 2026-08-31**, from the settings actually used in the reductions of record
(`refraction/zenith12/`, `cal_pileo_step2/`, `step3_s0_v4/`, `matrix_bruns2017/`) rather
than from intent. Every value below is one a run log can be checked against.

Two configurations cover every field this project shoots. The split is not stylistic: a
zenith field has thousands of stars on a flat sky, and an eclipse-day field has tens of
stars on a steep bright gradient beside a saturated object. The same detector settings
cannot serve both.

## The two standards

| option | **zenith / night calibration** | **eclipse day** (science *and* L/R calibration) |
|---|---|---|
| `sensitive_mode_stack` | True | True |
| `centroid_gaussian_subtract` | **False** | **True** |
| `centroid_gaussian_thresh` | 5.0 σ | **4.0 σ** |
| `min_area` | 4 px | **2 px** |
| `sigma_subtract` | 3.0 | **0.0** |
| `background_subtraction_mode` | annular | **Gaussian** (see below) |
| `centroid_refine_window` | True | True |
| `centroid_window_sigma` | 2.0 px | 2.0 px |
| `delete_saturated_blob` | True (inert: no blob present) | **False** — forbidden disk instead |
| `remove_edgy_centroids` | True | True |
| `img_edge_distance` | 5 px | 5 px |
| `reject_saturated_stars` (F16) | True | True |
| `saturation_fraction` | 0.95 | 0.95 |

Why each difference exists:

* **`centroid_gaussian_subtract`** is the *per-frame* sensitive detector; it feeds the
  alignment. On a zenith field with 1500–3500 stars the plain detector already gives the
  alignment far more than it needs, and the sensitive path costs time for nothing — which
  is what the classic UI's label means by "do not use for zenith or fields with >> 100
  stars". Near the Sun or Moon there may be only tens of stars per frame and they sit on
  a gradient, so the alignment needs it.
* **Threshold 5.0 → 4.0 σ and `min_area` 4 → 2 px**: the eclipse field is
  detection-limited (Leon's faintest admitted star is G 10.3; Bruns' union bottoms out at
  G 10.4), so the thresholds are opened as far as the two-filter fake-star doctrine
  allows — the catalogue match at stage 2 is the rigorous second filter.
* **`sigma_subtract` 3.0 → 0.0**: a soft-threshold subtraction that suppresses marginal
  detections. At zenith it keeps a huge star list manageable; on the eclipse field every
  marginal detection is potentially the one star that matters.
* **`delete_saturated_blob`**: see "the blob and the forbidden disk" below. It is *not*
  dormant in the program — only in this project's eclipse workflow.

## Moon fields use the eclipse-day standard

Decided 2026-08-31. The detection problem is the same shape (a bright saturated extended
object with a scattered-light halo, stars wanted close in), but the decisive reason is
different: the Portland full-moon field's job in the four-dataset matrix is to return
**L = 0** and thereby validate the *eclipse-day configuration*. Reduced with zenith
settings it would validate a configuration we never use on eclipse data, and the null
would prove nothing about the thing under test.

## The centroid estimator, and what it is worth

Three estimators are reachable, and **they are not interchangeable at this project's
precision**:

| estimator | reached by | what it does |
|---|---|---|
| `tetra-simple moments` | both sensitive flags off | `simple_get_centroids`: own hardcoded 25 px uniform background, **global** 2 σ threshold, moments over a 5–100 px footprint. `background_subtraction_mode` never applies to it. |
| `footprint moments` | sensitive on, `centroid_refine_window=False` | flux-weighted moment over each detected footprint of the background-subtracted image |
| `windowed` | sensitive on, `centroid_refine_window=True` | fixed-Gaussian-window iterated centroid (`windowed_centroid`), removing the moment's brightness dependence on an asymmetric PSF |

Measured stakes (`docs/MATRIX_2026.md`): on the Bruns 2017 data the windowed and moment
conventions differ by ~30 ppm of plate scale, **brightness-dependent**, which is ~0.2 ″
of L — larger than the statistical error of that measurement. Bruns' own §2.3 compared
the same two families (Astrometrica's Gaussian fit against MaxIm DL's moment) and
measured them differing by 0.039 px on an SNR-13 star and 0.003 px on an SNR-48 star:
the same brightness-dependent effect, in his own data, an order of magnitude apart
between a faint star and a bright one.

**`footprint moments` + `annular` is the shape of MaxIm DL's method** — a moment over an
inner region with a ring background — and therefore the convention family Bruns'
published 1.752 ″ was measured in. That combination is available today and is exactly the
current config default (`centroid_refine_window` defaults False).

Both are now recorded: stage 1 writes `centroid estimator` and `centroid_window_sigma`
into `results.txt`, and stage 2 carries `centroid estimator` into
`distortion_results.txt`, so a fitted plate scale states the convention it was measured
in. Archives written before this say `unknown (pre-v1.4.0 archive)`.

## Where each control lives

| option | classic UI | app window | notes |
|---|---|---|---|
| `centroid_gaussian_subtract` | "Sensitive stacking mode (use if close to sun or moon…)" | — | ticking it forces `sensitive_mode_stack` on |
| `sensitive_mode_stack` | "Use sensitive mode on stacked result" | "sensitive" checkbox | the stacked image uses `centroid_gaussian_subtract OR sensitive_mode_stack` |
| `centroid_gaussian_thresh` | "sigma_thresh [sensitive-mode]" | — | |
| `min_area` | "min_area (pixels) [sensitive-mode]" | — | |
| `sigma_subtract` | "sigma_subtract" | — | |
| `background_subtraction_mode` | "background subtraction mode" (Gaussian / annular) | — | affects **positions**, not just detection, whenever windowing is on: `windowed_centroid` runs on the background-subtracted image |
| `centroid_refine_window`, `centroid_window_sigma` | **not exposed** | **not exposed** | config file or CLI `--set` only — see the gap below |
| `delete_saturated_blob` + 3 sub-controls | "Remove big bright object (blob)" etc. | — | |
| `remove_edgy_centroids` | "Remove centroids near edges" | — | |
| `img_edge_distance` | — | — | config only; applies unconditionally |

**Known gap**: the estimator switch is applied by every reduction but shown by neither
interface, which is the exact failure the project rule "an interface should only apply
settings it can show" exists to prevent (`CLAUDE.md`). It should become a named selector
in the classic UI's sensitive-mode group ("centroid estimator: windowed / footprint
moments", sigma enabled only for windowed).

## Background subtraction, and the coronal subtraction that precedes it

They operate at different scales and compose:

* **In-pipeline** (`background_subtraction_mode`, inside detection, per image, never
  saved): `Gaussian` is a 17 px-kernel Gaussian blur subtracted from the image (σ ≈ 2.9
  px); `annular` is a 17 px box mean minus a 3 px box mean, i.e. a ring background that
  excludes the star's core. Both are **small-scale** high-passes — 17 px is ~35 ″ on
  Bruns' plate scale.
* **This project's coronal subtraction** (`tools/step3_s0_v4.py`,
  `tools/matrix_bruns/b17_s0.py`): a σ = 10 px Gaussian blur of the **tier-mean** image,
  subtracted from every frame with a +2000 ADU pedestal, written out as new FITS *before*
  stage 1. It models the **corona** — Bruns' own 2017 method — and everything downstream
  sees flattened data.

They work together: the coronal subtraction removes the large-scale gradient, then the
in-pipeline mode estimates what is left locally. Measured cost of the preprocessing:
≤ 0.014 ″ of centroid shift on every star tested, inner-annulus stars included
(`docs/MATRIX_2026.md`).

**Which in-pipeline mode is not a free choice — it is worth ~19 ppm of plate scale.**
Measured on the Bruns 2017 calibration fields, `annular` versus `Gaussian` moves the
fitted scale by 19.1 ppm (frame-identical stacks, everything else pinned), against 1.6
ppm for the windowed-versus-moment estimator. The mechanism is the same for both — how
much of an asymmetric off-axis PSF's wings a measurement sees — but the 17 px ring
reaches further into the coma than the Gaussian kernel does, so its bias grows with field
radius and reads as a scale error.

For **eclipse-day fields the standard is therefore `Gaussian`**, with `footprint moments`
as the estimator: that combination reproduces Bruns 2018 end to end
(L = 1.720 ± 0.069 ″ against his 1.752 ± 0.060), gives the lowest calibration residuals
of any configuration tried, and satisfies the project's founding design criterion.
`annular` remains the zenith standard, where the PSF is small, the field is flat and the
star counts are in the thousands — but the same A/B has not been run there, so that is
inherited practice rather than a measured choice.

## The Sun/Moon mask

**In the program since 2026-08-31** (ROADMAP F26). `delete_saturated_blob` is the on/off
switch — "Mask the Sun/Moon" in both interfaces — and `eclipse_mask_mode` chooses the
shape:

* **`disk`** (default): a circle centred on the saturated core. Centre = centroid of the
  eroded saturated component; radius = the 90th percentile of 36 azimuthal sector maxima,
  measured at full resolution, plus `eclipse_disk_margin_px` (10). The detection mask is
  that disk plus `centroid_gap_blob`, OR-ed with any saturated pixel the disk misses, so
  nothing clipped reaches detection.
* **`blob`**: the pre-v1.4.0 convex hull dilated by `blob_radius_extra`. Kept only to
  reproduce an older reduction — the hull followed streamers into lobes.

**Turning the mask off is a legitimate mode, not just a diagnostic**: it gives a plain
stack of the eclipse field aligned on the stars, which is often what you want to look at.

Known limit: the disk centre carries 4–11 px of streamer bias (measured against the
ephemeris on Bruns' three tiers), comparable to the 10 px margin, so a star within ~15 px
of the painted edge deserves a per-frame check.

The analysis tools (`tools/step3_s0_v4.py`, `tools/matrix_bruns/b17_s0.py`) still carry
their own copy of this preprocessing, with the radius rule
`max(1.25 R⊙, 99th-percentile saturation radius + 10 px)` and the Sun centre taken from
the ephemeris rather than from saturation. They and the pipeline agree on the Bruns
frames to within ~25 px of radius; the tools' numbers are the ones behind
`docs/MATRIX_2026.md`.

## Coronal subtraction in the pipeline

`coronal_subtract` (off by default) subtracts a σ = `coronal_subtract_sigma_px` (10 px)
blurred copy of each frame and adds `coronal_pedestal_adu` (2000). The stacked FITS is
then the flattened image — the same view Bruns published and Richard Berry reproduced,
with the coronal structure enhanced and the field stars visible against it.

It differs from the tools' version in one way, stated so the numbers are comparable: the
tools build the blur model from the **tier mean**, this builds it per frame. The mean
carries √N less noise into the subtraction, but needs every frame in hand before any can
be preprocessed. At σ = 10 px the blur already averages ~1200 pixels, so the per-frame
model adds ~3 % of a single frame's noise against ~0.4 % for a 45-frame mean.

## Edge filtering: keep it on

`remove_edgy_centroids` is **not** just an edge trim. For any detection more than 16 px
from a frame edge it runs a gradient-anomaly test that rejects readout and bloom
artifacts; within 16 px it passes everything that is at least 3 px from the edge.
Separately, `filter_very_edgy_centroids` always drops detections within
`img_edge_distance` (5 px) of an edge.

It has been suspected of deleting Leon's below-Sun G 7.71 anchor. It did not, twice
measured: the anchor is in the stage-1 centroid list of both tiers (0.80 px and 0.22 px
from expected), it sits 13 px from the bottom edge — inside the band the filter passes
by construction — and the run's log contains **zero** "deleting edgy centroid" messages.
The anchor was lost at stage 2 (rough-linear association dead below py ≈ 3850, and a
2.4–2.9 ″ physical displacement against a 2.0 ″ tolerance), which is what the two-pass
rematch fixed.

Open safety item, specified but **not implemented**: never drop a detection matching a
catalogue star brighter than **mag 8**. The ordering makes it a two-stage change — the
edge filters run before the plate solve, so stage 1 cannot know a magnitude — so the
design is: stage 1 *tags* rather than deletes (a `flag_edge_dropped` column, solve input
unchanged), and stage 2 admits a tagged detection only when it matches a catalogue star
brighter than the threshold, on the pattern F16 already uses. It is results-changing and
must not land mid-matrix, where every cell's comparability depends on identical detection
behaviour.
