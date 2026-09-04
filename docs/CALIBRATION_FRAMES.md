# Do the darks and flats earn their place?

**Date:** 2026-08-26. The question recurs at every campaign, and the honest answer differs by
field type. This records what is **measured**, what is merely **observed**, and what is
**untested** — because those have been getting conflated.

## The short version

| field type | verdict | basis |
|---|---|---|
| flat-darks | **not needed** | measured, `LEON_2026-08-11.md` §2.4 |
| zenith calibration | not needed | one controlled A/B; everything else observed only |
| L/R eclipse-day calibration (2026 CAL_piLeo) | **not needed** | measured, 4-arm ladder + paired test |
| L/R eclipse-day calibration (2017) | **unresolved at 30 ppm** | one candidate for an unexplained offset |
| the eclipse field itself | **untested, and there is a specific risk** | see below |
| Mexico 2024 eclipse field — flat | **unresolved, and the cell's largest sensitivity** | 2.4–3.3 mas by injection on the pixels, but the flat-fielded re-stack's centroids differ from the un-flatted one's by ~250 mas per star (the dark's: 37 mas) and **L moves −0.29 ″ on identical stars** (−0.16 ″ with the dark). Mechanism not identified — per-frame coronal model or the detection, not the pixel response. Dark + flat has the best residual and is the arm of record; ±0.2–0.4 ″ carried until the mechanism is closed |
| Mexico 2024 eclipse field — dark | **needed, for hot pixels only** | measured: 11 % of stacked detections are hot pixels, the dark removes every one, and it moves L by +0.04 ″ on identical stars — protection, not a shift |

## Measured: CAL_piLeo, the 2026 L calibration

Four arms over the same 17 pre-C3 frames (9 × 1 s + 8 × 2 s), identical in everything but
calibration, using the library at `G:\MEE_output\Library 1`:

| arm | stars | rms (″) | plate scale | se (ppm) |
|---|---|---|---|---|
| no calibration | 72 | 1.0752 | 2.205450 | 52.4 |
| master dark | 66 | 1.0498 | 2.205457 | 61.8 |
| dark + flat | 66 | 1.0647 | 2.205455 | 63.9 |
| dark + flat, 08-12 refs only | 66 | 1.0661 | 2.205440 | 64.1 |

**Plate scale moves 2.3 ppm between no calibration and dark+flat.** Against the ~4–6 ppm the
transfer needs (F&L eq. 23, `STAGE3_THEORY.md` §4), that is inside the noise.

The aggregate rms *appears* to favour the dark (1.0498 against 1.0752), but that compares
different star samples. Paired on the **69 stars common to all three arms**:

| comparison | median error | mean change | Wilcoxon p |
|---|---|---|---|
| dark vs none | 0.6780″ → 0.6716″ | +0.0127″ | 0.81 |
| dark+flat vs none | 0.6780″ → 0.6385″ | +0.0178″ | 0.99 |

rms over the common set: **none 1.0351″, dark 1.0499″, dark+flat 1.0642″**. Calibration is
marginally *worse* on every pairing, though at p ≈ 0.8–1.0 the honest reading is **no
effect**, not harm.

**Why this is physically expected.** §2.2 measured dark current at +10 °C as unresolvable —
the G0 median rises 0.042 ADU across the whole 0.1–2.0 s ladder — so the master dark is a
bias plus a defect map, and *subtracting a constant pedestal cannot move a centroid*. The
flat's structure is 1.278 % PRNU varying smoothly across 6248 px, essentially flat over a
star's few-pixel footprint. In a daylight-sky-limited regime at 3400 ADU background there is
nothing here for calibration to fix.

## Observed, not tested: zenith calibrations

One controlled A/B exists. Bruns 2017 field `EC07`, with and without the master dark
(`DARK_STACK1709081434.5583646.fit`, 208 hot pixels excluded):

    no dark    1143 stars   rms 0.0546″   plate scale 2.087903
    with dark  1144 stars   rms 0.0551″   plate scale 2.087903

Identical to six decimals. Everything else is *observed* rather than tested: the Leon zenith
archives record `darks:[], flats:[]`, and the Portland, Carrell, Leakey and London reductions
were all run uncalibrated — reaching 1.3 % cubic scatter and 1.1–1.4 ppm plate-scale standard
error. Excellent results without calibration, but nobody ran the comparison. **Do not quote
this as "measured".**

## Unresolved: the 2017 L/R calibration

Re-reducing Bruns' L and R eclipse-day fields **without** darks reproduces the 2024 reduction
(which used them) to 2 % on rms, and the day−night transfer to 14 ppm. But a **+30 ppm**
plate-scale offset between the two remains unexplained. Three candidates were tested and
eliminated — reference set (−0.4 ppm), F15's windowed centroid (+1.1 ppm), tolerance (<1 %) —
leaving two untested: the refraction fix of §11.1, and **the calibration frames themselves**.

So darks cannot be declared irrelevant for the L/R fields. They are one of two live
candidates for a 30 ppm effect, which is not negligible against a ~5 ppm target.

## Untested, with a specific risk: the eclipse field

The physical argument above — constant pedestal, smooth PRNU — covers *position* errors. It
does not cover **hot pixels**, and the eclipse field is where they are most dangerous:

- §16 sets `distortion_fit_tol = 999` deliberately, so **nothing rejects a bad centroid**:
  the residual is the signal.
- F16 rejects saturated stars; it does not test for hot pixels.
- Dark-free hot-pixel identification needs ≥3 px of dither and **correctly declines below
  it**. Bruns' `EC08` dithered 2.8 px and kept its hot pixels for exactly this reason.

A hot pixel promoted to a "star" therefore reaches the deflection fit unchallenged. That is
an argument for taking darks on the eclipse field even though they are immaterial everywhere
else, and it is cheap insurance rather than a measured need.

**The test, when SCI_ladder is reduced:** run it both ways and compare the star list, not
just the plate scale. The failure mode here is a spurious detection entering the fit, which a
plate-scale comparison would not reveal.

## Measured: Mexico 2024 (Station 1) — the first cell where the answer is *yes*

**Date:** 2026-09-03. Douglas asked whether calibration frames, immaterial for Bruns 2017 and
Leon 2026, would be needed for Mexico 2024. They are — but for the one reason this document
predicted, and not for either of the usual ones. Tools: `tools/matrix_station1/s1_darks_flats.py`
and `s1_hotpixel_risk.py`; masters in `station1_record/darks_flats/`.

The set is complete and lives on `G:\Mexico April 2024\Station-1-Eclipse-Data`: bias,
`dark-250ms`, `dark-300ms`, `dark-400ms`, `darkflats`, `flat`, 40 frames each. The eclipse
tiers are 0.25 / 0.3 / 0.4 / 0.3 s, so every tier has an exactly matching dark. Two things
about the set matter: it was shot **19:00–19:26 UTC, 50–75 minutes after totality**, and at
**CCD-TEMP 25–27 °C against the eclipse frames' −10 °C**.

### Dark current: still a non-issue, even 35 °C warm

The 400 ms master's median excess over bias is **0.00 ADU** at 25 °C. At this exposure the
master is a bias plus a defect map, exactly as at −20 °C and +10 °C, and the pedestal-plus-rate
argument of `calibration.py` covers it. The temperature mismatch does **not** disqualify these
darks. It would matter for a long-exposure campaign; it does not matter at 0.4 s.

### The flat: 2–3 mas, measured by injection

22.8 % vignetting from centre to corner, which sounds like a lot. The smooth-field estimate
σ²·d ln F/dx is the wrong tool for pixel-scale PRNU, so this was measured by **injection**: a
Gaussian PSF of the field's measured FWHM 3.74 px placed at 4000 random sub-pixel positions,
multiplied by the real master flat there, and centroided both ways.

| estimator | centroid shift from the flat |
|---|---|
| footprint moment | rms **2.4 mas**, median 1.9, 99th 5.3, max 12 |
| windowed, σ 2.0 px | rms **3.3 mas**, median 2.7, 99th 7.2, max 14 |

A quintic in position removes almost none of it (2.4 → 2.4 mas), so it is not the vignetting;
it is pixel-scale PRNU, random per star position, entering the per-star scatter rather than
biasing L. Against Station 1's per-star residual of 47–56 mas at G ≤ 12 that is **0.4 % in
quadrature**. Consistent in spirit with the flat-dark measurement above. **The flat is not
needed.**

### And the flat did not change across the refocus — so it cancels anyway

*Added 2026-09-03, on Douglas finding the zenith session's own flats at
`I:\Mexico 2024\Station 1 Zenith\Flats\2024-04-08_06_28_18Z`.* The verdict above rested on
each side's flat-fielding being small. It left the harder question open: the calibration
fields were shot at 05:32–06:15 UTC and the eclipse at 18:12, with a daytime refocus between
them worth −600 ppm of plate scale, and dust shadows change size with focus. A flat that
*changed* between the two sides is a systematic that does not cancel in the transfer, however
small each side's own flat-fielding effect is.

Both flats now exist, and both are properly exposed — 36 % and 33 % of full well:

| | when | exposure | gain / offset | temp |
|---|---|---|---|---|
| zenith session | 06:28:19 UTC | 2 s | 100 / 1 | −1.5 °C |
| post-eclipse | 19:21:27 UTC | 2 s | 0 / 50 | +27.6 °C |

Normalised each by its own median, their **vignetting profiles agree to 0.2 % at every
radius**, including the corners where the response is down to 0.75:

| r (px) | 0–500 | 1000–1500 | 2000–2500 | 3000–3500 | 4000–4500 | 5000–5600 |
|---|---|---|---|---|---|---|
| zenith / post-eclipse | 0.9991 | 0.9981 | 0.9998 | 1.0016 | 1.0014 | 0.9978 |

The full ratio image has an rms about unity of 1.1 %, almost all of it pixel noise; smoothed
to 64 px blocks it is 0.95 %, spanning 0.89–1.04, so a little localised structure did move.
Injecting a PSF through that *difference* — which is precisely the flat-fielding error that
fails to cancel between the two sides — gives **3.2 mas rms for the footprint moment and
5.1 mas windowed**, maxima of 9 and 15 mas. That is the same order as flat-fielding either
side on its own (2.4 and 3.3 mas), and an eighth of the per-star residual.

**So the flat verdict is strengthened rather than threatened: the response was stable across
the refocus, and the transfer cancels what little there is.** Skip the flat for Station 1.

> **A trap worth recording.** A flat has no separate dark-flat at gain 100 offset 1, so a
> scalar pedestal has to be assumed. Do **not** estimate it from a low percentile of the flat
> itself: the 2nd percentile of a vignetted flat is the *corner* (16.8 kADU of a 23.7 kADU
> flat), and subtracting it manufactures a 60 % fall-off that is not there. The first run of
> `s1_flat_stability.py` did exactly that and reported a spurious 13 % change in response
> between the two sessions. The tool now assumes 500 ADU and reports the sensitivity to that
> choice (0 to 1000 ADU moves the smoothed ratio's span from 0.92–1.05 to 0.86–1.03, leaving
> the conclusion intact).

### The hot pixels: the flagged risk is real here, and the dark is the only defence

This document's outstanding item was that a hot pixel promoted to a star reaches the
deflection fit unchallenged, and that dark-free identification "correctly declines" below
`hotpixels.MIN_DITHER_PX` = 3 px. Station 1 falls exactly into that hole.

**The dither is sub-pixel — and the gate did not notice.** Tracked stars in two opposite
2048 px corner windows (x 400 y 4000, and x 7000 y 400) put the frame-to-frame motion within
±0.25 px across all 123 frames of the 0.4 s tier: sixty seconds of sub-pixel tracking. The
stacker's own aligner agrees for most of them — the recorded shifts sit a median of 0.45 px
from their own median, with 90 of 123 frames inside 1 px — but places about five frames
(89, 90, 91, 106, 107) **3–4 px out in y**, which both star windows say did not move. Those
frames matched 18–25 stars against a typical 30–31, with roughly double the alignment
residual: degraded frames, mis-solved.

That matters here because `hotpixels.dither_span` is the largest **pairwise** offset between
any two frames, so those few bad shifts carry it to **5.39 px**, past the 3 px
`MIN_DITHER_PX`. The dark-free search therefore **ran**, and reported *"0 hot pixel(s)
identified from the dither (5.4 px) out of 4 bright candidates"*. On a field whose real
dither is sub-pixel that is precisely the regime the gate exists to prevent — a hot pixel
and a star are nearly indistinguishable — so **the search had no leverage and its verdict is
not evidence of absence.** Four candidates from a 61-megapixel frame says the same thing.

So the practical conclusion is unchanged, by a different route: there is no trustworthy
hot-pixel rejection on this field without a master dark. And because the real dither is
sub-pixel, hot pixels **do not smear**: they land within a pixel of the same stacked position
in every frame.

> **Correction.** The first version of this section said the dither was zero, measured by
> phase correlation, and that the dark-free search "cannot run, and correctly declines". Both
> halves were wrong. The phase correlation was locking onto the fixed pattern — vignetting
> times a bright corona is high-contrast and fixed to the detector — so it returned zero
> however the field moved; and the search did run, because a handful of bad shifts opened the
> gate. The tracked-star measurement replaces it. The robustness of `dither_span` against a
> few mis-solved frames, and the aligner's silent 3–4 px errors on degraded ones, are filed
> as separate work.

**And the hot pixels are really there at −10 °C.** At the 93 positions the 25 °C master calls
hot (> 200 ADU over bias), the eclipse frames themselves sit **+41.5 ADU** above a local
background — 3.3 σ of a single frame, with 44 % of them above 5 σ against 5.4 % at random
positions. They are at about **1/30** the amplitude the warm dark predicts, which is the
temperature difference doing its work. In a 123-frame stack, with the dither too small to
move them off their pixel, the noise falls as √123 while a fixed excess does not, so a 41 ADU
pixel arrives at roughly **37 σ** on the
stacked image, far above the 4 σ detection threshold.

**Why the temperature mismatch does not matter for this use.** The pipeline does not scale the
dark's hot pixels: `stacker_implementation.py` finds them once and they are *"excluded from
the stack rather than subtracted"*. It needs to know **which** pixels, not how much. A 25 °C
dark identifies the same defective sites as a −10 °C one and identifies them more easily.

### The star-list test, as this document asked for it

*"Run it both ways and compare the star list, not just the plate scale."* Done, on the
0.4 s tier re-stack (`s1_hotpixel_risk.py`), by asking how many detections land on a pixel
the master dark flags, under the shifts the stacker actually used:

| | positions | on a dark-flagged hot pixel | expected at random |
|---|---|---|---|
| all stacked detections | 292 | **33 (11.3 %)** | 0.0 % |
| matched to a catalogue star | 182 | 2 (1.1 %) | 0.0 % |
| the science set, G ≤ 12, non-outlier | 174 | **0** | 0.0 % |

The moments stack gives the same to within one detection (34 of 293, 2, 0).

Three things follow. **The contamination is real and large**: 11 % of what the stacker
detects on this field is a hot pixel, against a random expectation of essentially zero — and
the dark-free search, running on the same frames, reported *none*, which is the clearest
possible demonstration that it had no leverage at this dither. **It does propagate**: two of
them matched a catalogue star and would have entered a reduction with a fainter limiting
magnitude. **On this tier it did not reach the fit**: the G ≤ 12 cut and the outlier
rejection removed both before the science set, so the 2024 result is not contaminated by
this route.

That last line is a reprieve, not a guarantee. Nothing in the pipeline is designed to stop a
hot pixel reaching stage 3 — that is this document's original point — and the other three
tiers are shorter exposures reaching fainter, where the margin is thinner. Take the darks.

### Verdict for cell 2

| | verdict | basis |
|---|---|---|
| flat | **not needed** | measured, 2.4–3.3 mas by injection against a 47–56 mas residual; and the flat did not change across the refocus (0.2 % on the vignetting profile), so it cancels in the transfer too |
| dark, as a pedestal | not needed | measured, 0.00 ADU median excess at 0.4 s |
| **dark, as a hot-pixel map** | **needed** | measured: 11 % of the stack's detections land on a dark-flagged pixel, and the dark-free search found none of them |

**Use `--dark` on every eclipse tier, matched by exposure** (250 / 300 / 400 ms are all
present), and expect the library build to warn about the 35 °C setpoint difference — that
warning is correct and, for hot-pixel masking at 0.4 s, can be accepted deliberately. The
flat may be skipped. The test this document asked for — *compare the star list, not the
plate scale* — is `tools/matrix_station1/s1_hotpixel_risk.py`.

**Do not rely on the dark-free path on a well-tracked field.** It is built for a dithered
sequence and says so; the failure here is that it was not asked to decline. Until
`dither_span` is made robust, check the reported dither against the shift distribution
before believing a "0 hot pixels" result.

## What this does not say

None of the above bears on **flats for photometry**, on **hot-pixel maps for stage 1
alignment**, or on campaigns at higher sensor temperature where dark current is measurable.
The measurements here are all at −20 °C (2017), +10 °C (2026) or +25 °C
(2024, against −10 °C lights) with sub-2 s to 10 s exposures. The 2024 pair is the one
case where the temperatures were badly mismatched, and it still came out fine, because
at 0.4 s a master dark is a bias plus a defect map and neither part is temperature
sensitive enough to matter.
