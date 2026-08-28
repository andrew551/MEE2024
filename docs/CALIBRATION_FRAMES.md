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

## What this does not say

None of the above bears on **flats for photometry**, on **hot-pixel maps for stage 1
alignment**, or on campaigns at higher sensor temperature where dark current is measurable.
The measurements here are all at −20 °C (2017) or +10 °C (2026) with sub-2 s to 10 s
exposures.
