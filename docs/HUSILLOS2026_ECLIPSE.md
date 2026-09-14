# Husillos 2026, the eclipse captures: which frames belong, and whether stars are there

**Date:** 2026-09-10. Matrix cell 4. Measured on `G:\Joe Izen Spain 2026\2026-08-12` with
`v1.4.0-dev`. Companion to [`HUSILLOS2026_ZENITH.md`](HUSILLOS2026_ZENITH.md), which has the site
card, the naming rule and the zenith work.

Two captures, both 315 ms, both full frame, both offset 200, shot 0.323 s apart with no slew:

| | `SunJoe_20260812_182845/20_28_45.ser` | `Sn2_Joe_20260812_182942/20_29_43.ser` |
|---|---|---|
| slots | 180 | 103 |
| gain | **125** | **0** |
| sidecar start → end | 18:28:45.594 → **18:29:42.367** | **18:29:42.690** → 18:30:15.178 |
| trailer | **all zeros** (no per-frame times) | per-frame timestamps, 0.3152 s median |
| what it is | the C2 decay and the Sun | the eclipse star field |

Tools: `tools/husillos2026/hu_eclipse_frames.py` (the frame scan), `hu_eclipse_stars.py` (stage 1
and the cross-capture check), `hu_hotpixels.py` (the mask, §3c), `hu_eclipse_match.py` (a
catalogue matcher that **does not work yet and refuses to report** — §4).

**Headline: both eclipse fields now plate-solve, at both gains.** With a hot-pixel mask built
from the night captures, the gain-0 science field solves on **55 stars** and the gain-125 Sun
capture on **68**, and the two agree to **2.7 ″ in position, 0.062° in roll and 115 ppm in
scale** — separate captures, different gains, solved independently. §3b.

---

## 1. Totality starts at frame 46–47 of the Sun capture

Read off a 600-row band through the Sun on every frame. The sky falls **7.4 %/frame** through the
C2 decay and **0.18 %/frame** after it, and the break is sharp:

| frame | UTC (header + 0.3154 s cadence) | sky ADU | saturated px | px over sky+1000 |
|---|---|---|---|---|
| 40 | 18:28:58.56 | 5055 | 377 911 | 4 030 263 |
| 44 | 18:28:59.82 | 3813 | 274 477 | 2 644 742 |
| 46 | 18:29:00.46 | 3584 | 261 795 | 2 170 544 |
| **47** | **18:29:00.77** | 3519 | 259 558 | 2 074 781 |
| 48 | 18:29:01.09 | 3474 | 258 482 | 2 045 356 |
| 50 | 18:29:01.72 | 3411 | 257 424 | 2 027 267 |

**The site card's tabulated C2 is 18:29:00.5**, which lands between frames 46 and 47 — and that is
exactly where the decay flattens. So **totality begins at frame 46–47**, and Douglas' estimate of
"approximately frame 50" is right and conservative: by frame 50 the sky is already on the slow
0.18 %/frame limb, and the four frames cost nothing.

The cadence is assumed, not measured, because **this file's trailer is all zeros** — the timestamps
are written when a capture completes and this one did not. The sidecar's `Duration=56.773 s` over
180 slots gives 0.3154 s/frame, and Sn2's own trailer measures 0.3152 s, so the assumption is good
to 0.1 % and the estimate of C2's frame is good to well under one frame.

**Frames 172–179 are all zeros — eight blank slots**, not the five an earlier note guessed. The
last frame carrying signal is 171.

---

## 2. Sn2's first two frames belong to the Sun capture — confirmed three ways

Douglas, 2026-09-10: *"I think the first two frames actually belong to the first file (frame
buffer problem with SharpCap, as was seen with the Leon data)."* **They do.** This is testable to a
certainty most such claims are not, because **the two captures ran at different gains**, so a
carried-over frame is not merely "anomalous" — it is a gain-125 frame sitting in a gain-0 file.

**(a) The timing makes it possible.** The Sun capture ends at 18:29:42.367 and Sn2 starts at
18:29:42.690 — **0.323 s, one frame interval**, with no slew between them. That is exactly the
situation in which a ring buffer hands the first frames of a new capture the last frames of the
old.

**(b) Every statistic matches the Sun capture's tail and not Sn2's body:**

| | sky ADU | band median | saturated px | px over sky+1000 |
|---|---|---|---|---|
| Sun frame 170 | 2612 | 2976 | 247 415 | 1 832 763 |
| Sun frame 171 *(last with signal)* | 2610 | 2974 | 247 403 | 1 832 240 |
| **— file boundary —** | | | | |
| **Sn2 frame 0** | **2595** | **2957** | **246 999** | **1 825 439** |
| **Sn2 frame 1** | **2593** | **2955** | **246 815** | **1 825 680** |
| Sn2 frame 2 | 1830 | 1914 | 96 892 | 676 211 |
| Sn2 frames 5–102 | 1826 ± 3 | — | 95 124–97 084 | — |

The Sun capture's sky is falling 2.00 ADU/frame over its last twenty frames; the linear
extrapolation to the next two is 2608 and 2606 against the observed 2595 and 2593 — 13 ADU low,
which is six or seven frames' worth of decay and exactly the gap the eight blank slots represent.

**(c) The gain signature settles it.** Against Sn2's own body, frames 0–1 sit at a **level ratio of
×1.42 but a NOISE ratio of ×4.30**:

* if both were at the same gain, noise scales as √signal, so a ×1.42 level ratio permits at most
  **×1.19** — and less, since most of the level is bias;
* a **+125 gain step multiplies ADU per electron by 10^(125/200) = ×4.217**.

**The observed ×4.30 matches the gain step to 2 % and is 3.6× too large for any same-gain
brightness change.** Frames 0 and 1 were exposed at gain 125.

**So the usable frames are Sn2 2–102 (101 frames) and Sun 46–171 (126 frames).**

---

## 3. Stars: real sources are detected, and with a hot-pixel mask the field solves

Stage 1 at cell 2's eclipse settings (`tools/matrix_station1/s1_eclipse_corona.py`: disk occulter,
per-frame coronal subtraction at σ 10 px on a 2000 ADU pedestal, Gaussian-subtracted sensitive
detection, windowed centroids on an annular background) — with no darks and no flats, which
Husillos does not have:

| stack | frames | centroids | plate-solve | beyond 4 R☉ | per sq deg |
|---|---|---|---|---|---|
| `sn2_trimmed` (2–102) | 101 | 304 | **no** | 264 | 13.6 |
| `sn2_masked` (occulter grown past 4 R☉) | 101 | 4061 | **no** | 223 | 11.5 |
| `sun_totality` (46–171) | 126 | 216 | **no** | 194 | 10.0 |
| **`sn2_darkall`** (hot-pixel mask, §3b) | 101 | **115** | **yes, 55 stars** | — | — |
| **`sun_dark`** (hot-pixel mask, §3b) | 126 | **97** | **yes, 68 stars** | — | — |

**The inner field is not stars.** Of 1715 sources of ≥ 4 px above 12 σ in the trimmed stack,
**1697 lie inside 4 R☉**, at 330–910 per square degree against the **73 per square degree** the
Gaia G < 13 catalogue actually holds there. That is the residue the σ = 10 px coronal subtraction
leaves on the streamers, and it is 99 % of what the solver is being asked to identify the field
from. Growing the occulter past 4 R☉ removes it and raises the yield 13-fold — and still does not
solve.

**But the outer detections are real, and this is catalogue-free.** The two captures point at the
same sky 0.3 s apart at **different gains** and are stacked separately against different masters,
so a noise excursion cannot repeat between them:

* **170 pairs vote for a single offset of (−1.2, −1.0) px**, against a 99th percentile of 1 and a
  chance level of 0.0;
* **173 of Sn2's 264 outer sources (66 %) have a counterpart in the Sun capture within 3 px.**

The remaining alternative — a hot pixel, fixed to the detector, which would also repeat — is
excluded separately: only **7 of the 1715 eclipse sources sit within 2 px of one of the 2527
sources in the zenith stack**, against 3 expected by chance. Whatever these are, they are not the
sensor.

Their density is consistent too. 13.6 per square degree against the catalogue's 73 is 19 %, and
the brightest 19 % of that catalogue runs to about **G 11** — which is what an airmass-6.4 field
(the Sun was at **8.6° altitude**) at 0.315 s and gain 0 should reach. The stack's noise is
photon-predicted to the digit (1.97 ADU measured, 1.97 predicted from 19.3 ADU per frame over 101
frames), so the stacking is not the problem.

**So: yes, stars are being located — of order 200 real sources beyond 4 R☉ in each capture.**

### 3b. And with a hot-pixel mask, the field SOLVES

Douglas, 2026-09-10: *"Joe did not take any darks but he could do that now. In the meantime, is it
possible to create a hot pixel mask using the zenith field that we have?"* — `hu_hotpixels.py`
(§3c) builds one from the night data. Applied to the same 101 frames at the same settings:

**Both eclipse fields solve, at both gains:**

| stack | gain | mask | frames | centroids | solve | RA | Dec | roll | ″/px | time |
|---|---|---|---|---|---|---|---|---|---|---|
| `sn2_trimmed` | 0 | none | 101 | 304 | **no** | — | — | — | — | 15.2 s |
| `sn2_masked` | 0 | none, occulter past 4 R☉ | 101 | 4061 | **no** | — | — | — | — | — |
| `sn2_dark` | 0 | g0 (389 px) | 101 | 114 | **yes**, 20 stars | 142.3141 | +14.9256 | 326.148 | 2.20410 | 1.1 s |
| **`sn2_darkall`** | **0** | **all (2587 px)** | 101 | 115 | **yes, 55 stars** | **142.3141** | **+14.9256** | **326.148** | **2.20410** | 0.9 s |
| `sun_totality` | 125 | none | 126 | 216 | **no** | — | — | — | — | 17.6 s |
| **`sun_dark`** | **125** | **all (2587 px)** | 126 | 97 | **yes, 68 stars** | **142.3149** | **+14.9255** | **326.210** | **2.20384** | 1.7 s |

The mask removed 190 of the science field's 304 centroids, against the 191 predicted to be
sitting on a flagged pixel, and 119 of the Sun capture's 216, against 119 predicted. **A cleaner
list, not a bigger one, was exactly what the solver needed**, as §5 guessed — growing the
occulter raised the yield thirteenfold and still failed.

**The two fields agree with each other.** They are separate captures at *different gains*, 57 s
apart, stacked and solved independently:

| | |
|---|---|
| field centres | **2.7 ″ apart** |
| roll | **0.062° apart** |
| plate scale | **115 ppm apart** |
| distance from the Sun at mid-capture | 0.200° and 0.202° |

And each is right against things that were not inputs to the solve: the geometry (frame centre at
(4788, 3194), Sun at ~(5175, 2975)) predicts 0.273° from the Sun; the **zenith field's roll is
325.902°**, so the camera was not rotated between the two nights; and the zenith field's own blind
solve gives 2.20466 ″/px, 254 ppm from the science field's.

The deeper mask changed the science field's answer not at all — RA, Dec, roll and scale identical
to four decimals — while raising the matched-star count from 20 to 55. That is what a better mask
should do.

**Both Husillos eclipse fields are now solved astrometric fields**, which is what stage 2 needs to
begin.

### 3c. The hot-pixel mask, built without a dark

**Not from the zenith capture itself.** `2026-08-13/zenith/00_00_21.ser` dithers **1.21 px** over
its 50 frames, and `hotpixels.MIN_DITHER_PX` is 3 px — below that a hot pixel and a star are
indistinguishable by the persistence test and every star would be flagged. Stage 1 declined on it
in as many words.

**But from its siblings, yes.** Four captures share the zenith's settings exactly — gain 0,
offset 220, 1.0 s, same camera, same night, same sensor temperature — and dither far more, because
the mount was drifting or re-pointing:

| capture | frames tracked | dither |
|---|---|---|
| `2026-08-12/cal 8 deg/22_53_15` | 72 | **61.9 px** |
| `2026-08-12/cal 8 deg/22_56_41` | 100 | **25.9 px** |
| `2026-08-13/zenith/00_00_21` | 50 | 1.2 px — unusable |

A hot pixel is fixed to the **detector**, a star to the **sky**, so 26–62 px of dither separates
them cleanly. `hu_hotpixels.py` calls `mee2024.hotpixels.persistence_mask` — the project's own
implementation — rather than carrying a second copy, and supplies the two things it needs:

* **the shifts in the pipeline's convention.** Measured against a stage-1 run whose shifts are
  recorded: `shifts_px = (−dy, −dx)`, axis order (y, x), correlations −0.998 and −0.994. A sign
  error here would silently flag the *stars* instead — the one failure mode that looks like
  success.
* **a candidate pre-filter set to the criterion's own level.** `candidate_sigmas` is lowered from
  20 to 5, which cannot loosen the answer: it limits how many pixels are examined on frame 0,
  while the criterion is `MIN_DETECTOR_PERSISTENCE = 5.0` applied to the **weakest** of all
  frames. It found 70 128 candidates against 363, and 389 flagged against 332.

**389 hot pixels, 0.0006 % of the sensor**, of which **270 (69 %) are flagged independently by
both captures** — and those two captures point at different sky, so agreement can only be the
detector. The output is a synthetic master dark: **zero everywhere, flagged pixels at 1000 ADU**.
The pipeline then flags exactly those through its ordinary `--dark` path (`dark_mask` cuts at
median + max(10 ADU, 20 σ), and a zero dark has median 0 and σ 0) while **subtracting nothing** —
which is what makes it safe on frames whose bias it does not share. Stage 1 reports it as
*"389 hot pixel(s) found in the master dark (0.0006 % of the frame); excluded from the stack
rather than subtracted"*.

What it explains:

| stack | centroids | on a hot pixel | of the ≤ 2 px ones |
|---|---|---|---|
| **`sn2_trimmed`** | 304 | **191 (63 %)** | **119 of 146 (82 %)** |
| `sun_totality` | 216 | 119 (55 %) | 0 of 19 |
| `with_f0` (zenith) | 3211 | 277 (8.6 %) | — |

**A mask belongs to a gain, and the gain-125 captures see six times deeper.** Hot pixels are the
same silicon defects at any gain, but how far each stands above the noise is not: gain 125 has
**1.38 e- of read noise against gain 0's 4.73**, so pixels that miss the 5 σ criterion at gain 0
clear it easily at gain 125. Built separately from the `Capture` set (gain 125, 0.315 s, dither
4.6–30.6 px):

| family | captures | flagged | agreed by all captures |
|---|---|---|---|
| **g0** (gain 0, 1.0 s) | 2 | **389** | 270 (69 %) |
| **g125** (gain 125, 0.315 s) | 3 | **2484** | 1952 (79 %) |
| union | — | **2587** | — |

**286 of the gain-0 mask's 389 are also flagged at gain 125, against 0.0 expected by chance** —
the two families are the same silicon, and the deeper one simply sees more of it. 2484 is also
close to the ~2300 the dither experiment of `HUSILLOS2026_ZENITH.md` §6 implied matter in a deep
stack, which is an independent arrival at the same number.

Use the **union**. Both eclipse fields above were solved with it.

Real darks remain worth asking Joe for — they are minutes with the cap on, they need no dither,
and they reach the mildly-hot pixels this method cannot. **1.0 s / gain 0 / offset 220 / 0 °C** and
**0.315 s / gain 0 / offset 200**.

---

## 3d. What the two fields actually see, and how much they share

Douglas, 2026-09-10: *"So the gain zero exposures saw 55 stars and the gain 125 exposures saw 68?
What was the overlap?"*

**First, what 55 and 68 are not.** They are the **plate solver's verification counts** — how many
catalogue stars it lined up well enough to accept the solution, printed as *"MATCH ACCEPTED
(nstars matched = 55)"*. The solver stops once it is convinced. The science star list is what
stage 2 matches afterwards, and that is a different number.

**And the first attempt at it was wrong, because refraction was off.** Every diagnostic fit in
this cell has run with corrections off, which was right at the zenith and is badly wrong here: the
Sun was at **8.6° altitude**, z = 81.4°, where R = k·tan z is **384 ″** and its *second* derivative
across the field is 2k·sec²z·tan z = 33 200 ″/rad² — about **19 ″ of quadratic distortion over a
3.9° half-field**. A linear fit absorbs the shear; nothing absorbs that.

| field | gain | corrections | gate | stars | rms | plate scale |
|---|---|---|---|---|---|---|
| Sn2 | 0 | off | 2.0 ″ | 17 | 0.4152 ″ | 2.218138 |
| Sn2 | 0 | **refraction on** | 2.0 ″ | **71** | 0.6266 ″ | 2.202937 |
| Sn2 | 0 | **refraction on** | 0.5 ″ | 36 | **0.2895 ″** | 2.202962 |
| Sun capture | 125 | off | 2.0 ″ | 32 | 0.5093 ″ | 2.212959 |
| Sun capture | 125 | **refraction on** | 2.0 ″ | **84** | 0.7157 ″ | 2.202552 |
| Sun capture | 125 | **refraction on** | 0.5 ″ | 35 | **0.2526 ″** | 2.202489 |

Refraction quadruples the matched-star count and pulls the two fields' plate scales from 2300 ppm
apart to **385 ppm apart**. The residual at the tight gate, **0.25–0.29 ″**, is better than Leon's
CAL_piLeo (0.53 ″).

*The weather is assumed*: 926.5 hPa is the standard atmosphere at 743 m, with 25 °C and 35 %
humidity as ordinary August evening values. **This is a sensitivity, not a measurement** — the
record reduction needs the real conditions. Everything below is at gate 2.0 ″ with refraction on.

**The overlap, on Gaia source ids** (strings, never floats — `tests/test_star_id_handling.py`):

| | stars |
|---|---|
| gain 0 only | 8 |
| **both** | **63** — 89 % of the gain-0 list, 75 % of the gain-125 list |
| gain 125 only | 21 |
| **union** | **92** |

The 63 shared stars run G 4.93–10.00, median 8.70. The 21 the gain-125 capture sees alone are
fainter (median G 9.57), which is what its lower read noise (1.38 e- against 4.73) and 25 % more
frames should buy.

**An independent check that does not use the catalogue at all:** cross-matching the two *stage-1
centroid lists* in pixel space gives **71 common detections** at a single offset of (−1.9, −1.7) px
— 62 % of the gain-0 list, 73 % of the gain-125 one. So the detections were always in agreement;
it was the *matching* that refraction was breaking.

### The annotated masters

`hu_eclipse_overlap.py charts` draws them in the form cells 1–3 already use — arcsinh-stretched
master, yellow circles on the matched stars, the 2 R☉ circle dashed in cyan, the count in the
legend. Both the stretch and the sky-to-sensor affine come from `tools/record_charts.py`
(`arcsinh_stretch`, `SkyFrame`), never a private copy.

**The Sun is placed by the astrometry, not by image morphology.** A first attempt hunted for the
occulted disk in the stack and put the circle 400 px off, because the occulter's fill value
(5276.8 ADU here) is not the darkest thing in a coronal-subtracted frame. `SkyFrame.from_stars`
fits the affine on the matched stars and the Sun's apparent place at mid-capture goes through it
forwards — so the circle is a *check* on the solution, not a decoration. The two fields place it
at **(5042, 3382)** and **(5045, 3384) px** independently: 3 px apart.

They are written to `husillos2026/eclipse/charts/`, **not** to `RECORD/` — the fit behind them
assumes the weather, and `RECORD/` is for finished record charts.

---

## 3e. A first Method 2

Douglas, 2026-09-10: *"Let's do a quick Method 2 calculation with what we have so far."* — with
three corrections to a first attempt, each of which changed the answer.

**The pathway.** *"For Method 2, you need to take the cubic and higher coefficients from the
zenith field and apply them to the eclipse field data."* `distortion_fixed_coefficients` names the
highest order left **free** and freezes everything above it (`distortion_polynomial.py`,
`order_free = mapping[...]`), so that is **`quadratic`**. The first attempt used `constant`,
which also freezes the linear and quadratic — wrong for a field 3.5 hours and 73° of altitude
from its reference, where the low orders have moved. It cost 0.18 ″ of residual:

| eclipse rung | residual, gain-0 block |
|---|---|
| `constant` (linear and quadratic frozen too) | 0.8767 ″ |
| **`quadratic`** (cubic and higher frozen, as asked) | **0.6947 ″** |

**Two science blocks, not one.** The same field was shot twice within a minute at two gains, and
both plate-solve (§3b). An earlier draft of this document said Husillos "has one science block and
no second tier"; that was wrong.

**No radial crop.** `analysis_window.WINDOWS['husillos2026']` is registered and cited, but its
outer bound is cell 2's inherited 10 R☉ which has never been tested on cell 4's data. Enforcing it
would drop stars on a borrowed number, so it is recorded and not applied.

**Method 2 only.** An earlier draft also reported a "Method 1". That was not asked for, and it was
not Method 1: Method 1 imports a plate scale from a *calibration field*, and Husillos has none
reduced, so what it actually reported was Method 2's own fitted scale treated as known. It is
withdrawn.

### The conventions, read back from the runs' own records

Douglas, 2026-09-10, asked which coronal subtraction and which centroiding convention were used.
Both blocks' stage-1 zips store identical options:

| | |
|---|---|
| **coronal subtraction** | **yes, Gaussian** — `coronal subtraction? True`, blur **σ = 10.0 px**, pedestal **2000 ADU**. Bruns' method: blur heavily, subtract, restore a pedestal. |
| occulter | `eclipse mask mode: disk`, `eclipse_disk_margin_px: 10`; the saturated blob removed at 95 %, `blob_radius_extra` 200 px, `centroid_gap_blob` 100 px |
| **centroid estimator** | **windowed**, `centroid_window_sigma 2.0` |
| **background** | **annular** |
| detection | Gaussian-subtracted, threshold 4.0 σ, `min_area` 2, `sigma_subtract` 0.0, sensitive stacking on |
| calibration | the synthetic hot-pixel dark only (§3c); **no flat** |

That pair — windowed centroids on an annular background — is **cell 2's record convention**
(`tools/matrix_station1/s1_eclipse_corona.py:66-75`, copied deliberately when Douglas asked for
"similar eclipse settings used for Station 1 Mexico"), and it is *not* the `eclipse` field
preset's Gaussian background with footprint moments. The two are not cosmetic: Leon measured the
same 2 × 2 grid in L itself and its four cells span **1.60–2.12 ″**, the background axis worth
+0.14 to +0.30 ″ and the estimator axis −0.22 to −0.38 ″ (`docs/STEP3_2026.md`, the convention
grid). Leon fixed windowed on a measured aberration of its own optic, so cell 4 inherits the
convention rather than the justification, and no such grid has been run here. The run's
`field preset` reads `custom` because the settings were passed explicitly rather than by name.

### The fits

Reference: the one zenith field, free **quintic**, gate 0.5 ″, refraction on — 2635 stars, rms
0.1594 ″, ps 2.2059136 ″/px. Both blocks then at `quadratic`, gates 20 ″ then 3 ″, scale free:

| block | gain | frames | stars | rms | plate scale |
|---|---|---|---|---|---|
| Sn2 | 0 | 101 | 73 | 0.6947 ″ | 2.2029009 |
| Sun capture | 125 | 126 | 84 | 0.8019 ″ | 2.2027459 |

The two scales agree to **70 ppm**. Rung read back from each run's own results, as CLAUDE.md
requires: *fixed distortion order: **quadratic***, *plate scale source: **fitted on this field***.

### The deflection

| block | stars | deflected-position rms | **L (Method 2)** | plate scale |
|---|---|---|---|---|
| gain 125 | 84 | **0.780 ″** | **2.215 ± 0.433 ″** | 2.202649 ± 12 ppm |
| gain 0 | 75 | 8.241 ″ | 2.396 ± 5.379 ″ | 2.202963 ± 138 ppm |

**The gain-125 block measures something: L = 2.215 ± 0.433 ″, with GR's 1.75 ″ at 1.07 σ.**

**The gain-0 block is wrecked by exactly two stars**, and the cause is a pipeline behaviour worth
recording. Its two worst stars carry deflections of **40.7 ″ and 23.1 ″ at 5.41 and 9.08 R☉** —
radial residuals of 7.5 ″ and 2.5 ″, far outside the 3 ″ gate stage 2 fitted at. They are there
because **stage 3 admitted 75 stars where stage 2 fitted 73**: `mee2024/eclipse_analysis.py`
never reads `flag_is_outlier`, so it re-includes stars the distortion fit itself rejected. Those
two carry the whole error bar — the block's deflection rms is 5.473 ″ with them and **0.914 ″
without**. No star has been removed by hand here; the fact is reported instead.

### 3f. The two-witness rule, and what it exposes

Douglas, 2026-09-10: *"Let's use the two witness rule used for Leon 2026 data analysis. That
should get rid of the two outliers which are nonsensical."* Adopted matrix-wide on 2026-09-02
(`docs/MATRIX_2026.md`): **admit only stars seen in both tiers.** The reason is that a
single-witness star cannot be arbitrated — its one detection has nothing to contradict it — and
on Leon it cost six stars, ±0.04 ″ of statistical error and −0.06 ″ of L. Husillos' two
witnesses are the same field at two gains, 57 s apart.

It is applied as a **filter on the stage-2 output**, not as a re-implemented fit: stage 3 reads
`CATALOGUE_MATCHED_ERRORS.csv` out of the distortion zip, so a copy with the single-witness rows
removed runs through stage 3's own arithmetic unchanged (`hu_step3.py witness`).

**It does exactly what he predicted.** 64 of the stars are seen in both blocks; the rule drops 11
from the gain-0 list and 20 from the gain-125 one — **and the two nonsensical stars are two of the
eleven.**

| block | stars | worst \|deflection\| | deflected-position rms | **L (Method 2)** |
|---|---|---|---|---|
| gain 0, all matched | 75 | 40.66 ″ | 8.241 ″ | 2.396 ± 5.379 ″ |
| **gain 0, two-witness** | **64** | **1.24 ″** | **0.636 ″** | **1.596 ± 0.507 ″** |
| gain 125, all matched | 84 | — | 0.780 ″ | 2.215 ± 0.433 ″ |
| **gain 125, two-witness** | **64** | 1.40 ″ | 0.731 ″ | **2.782 ± 0.540 ″** |

The gain-0 block's error bar falls by a factor of **10.6**, and it did so without anyone naming a
star: the rule removed the 40.7 ″ and 23.1 ″ stars because nothing corroborated them, which is
what it is for.

**And now the two blocks disagree, which is the more useful result.** On identical stars they read
**1.596 ± 0.507 ″ and 2.782 ± 0.540 ″**, 1.186 ″ apart. Two candidate explanations, both testable:

*The plate scale — tested and rejected.* The blocks' fitted scales differ by 87 ppm, and Method 2
fits L and the scale together, so this is the obvious suspect. It fails on sign. Husillos' own
lever is **h = 1/mean(1/r²) = 34.2 R☉²** on these 64 stars (mean radius 7.15 R☉, reaching
12.05), so `δL = δS · h · R☉` gives **0.0324 ″ of L per ppm** — and 87 ppm predicts **+2.83 ″**
where the observed difference is **−1.19 ″**: wrong sign, 2.4× too large.

*Per-star noise — which fits.* The two blocks measure the same 64 stars, so their deflections can
be correlated directly. They correlate at only **r = 0.484**, with a per-star difference rms of
**0.500 ″** against each block's own scatter of 0.454 and 0.523 ″ — i.e. each block carries
**±0.35 ″ of independent per-star noise**, as large as the deflection signal it is trying to
measure. At that correlation the expected σ of the difference of the two L values is 0.533 ″, so
**1.186 ″ is 2.2 σ.** A real tension, and a statistical one.

The plate scale is not the cause but it *is* an amplifier: with the scale held, the blocks'
fits differ by 0.633 ″ (0.821 against 1.454); freeing it — Method 2 — pushes both up and widens
the gap to 1.186 ″.

**Two consequences for the cell.** First, sharing stars does **not** make these blocks one
measurement: their noise is largely independent, so this is two noisy measurements that disagree
at 2.2 σ, not one measurement quoted twice. Second, **h = 34.2 R☉² makes Husillos the most
scale-sensitive field in the matrix** — against Leon's 19.8 and Bruns' 8.2 — because its stars
sit far out (mean 7.15 R☉) where the deflection is small but a scale error is not. Its 0.0324
″/ppm is naive-h, not measured by injection as Leon's and Station 2's were, so treat it as the
right order and not the final figure. Either way it doubles the case for CalibS.

## 3g. The León union: combining the two gain blocks per star

Douglas, 2026-09-10: *"The two exposures at the same [field] in Leon 2026 are conceptually
equivalent to the two different gains at the same exposure of Husillos. Let's try this method."*
He is right about the equivalence, and the method transfers. `tools/husillos2026/hu_union.py`.

León's 0.6 s and 1.2 s tiers were never stacked together — they were reduced separately and
combined **at the star level** (`tools/step3_s2_union.py`): per block, displacement =
observation − catalogue with the block's **median displacement subtracted** (which kills the
per-block pointing constant so the blocks can be mixed); per star, the **median across blocks**;
then a **cross-block consistency vet** dropping any star whose blocks disagree by more than 3×
the field's cross-block MAD. The union rides **one host block's model**, so the output is one
consistent geometry.

### One step León did not need — and the reason is the clock, not the altitude

*(Correction, Douglas 2026-09-10: an earlier version of this section said León observed at 40°
altitude. That number was invented. León was at* **+9.9°** *at C2 —
`I:\Leon location and weather data\actual leon site.JPG` — essentially the same as Husillos'
8.6°. The two stations watched the same eclipse from 130 km apart; of course their altitudes
match.)*

The real difference is the **separation in time**. León's two deep tiers are **1 s** apart
(`step3_s2_union.MIDT`: 0.6 s at 18:28:33, 1.2 s at 18:28:32), so its catalogue frame cannot
move between them. Husillos' blocks are **39 s** apart, and at 8.6° dR/dz is about 46 ″ per
degree. Measured here rather than assumed: **the two blocks' catalogue positions differ by
0.60 ″ with a 0.54 ″ spread about that** — differential refraction over the 39 s, and
emphatically not a constant. So absolute positions must not be averaged. Displacements are,
because the same refraction that moves the catalogue moves the observation and it cancels.

For a star only one block saw, its displacement still has to be carried into the host's frame.
That transfer is a **quadratic in field position** fitted on the shared stars, and its residual
is the justification for using it:

| model of the block-to-block frame difference | residual |
|---|---|
| constant (i.e. ignore the structure) | 0.724 ″ |
| linear | 0.083 ″ |
| **quadratic** | **0.005 ″** |

Five milliarcseconds. The difference is smooth and deterministic, exactly as differential
refraction should be — which is also a quiet check that the refraction correction is doing
something real at this altitude.

### The result

The vet removed **one** shared star unaided (G 9.61 at px 7390,4425, cross-block spread 2.384 ″
against a 0.547 ″ field median), which the two-witness rule alone would have kept. 63 stars.

| set | N | h (R☉²) | rms | **L (Method 2) ± stat** | plate scale |
|---|---|---|---|---|---|
| gain 0 alone, two-witness | 64 | 34.2 | 0.636 ″ | 1.596 ± 0.507 ″ | 2.202848 |
| gain 125 alone, two-witness | 64 | 34.3 | 0.731 ″ | 2.782 ± 0.540 ″ | 2.202656 |
| **UNION, two-witness** | **63** | **34.2** | **0.550 ″** | **2.129 ± 0.430 ″** | 2.202674 |
| union, every star | 94 | 27.4 | 7.383 ″ | 2.133 ± 3.863 ″ | 2.202789 |

**The union is better than either block alone on every axis**: the residual rms falls to 0.550 ″
from 0.636 and 0.731, and the statistical error to ±0.430 ″ from ±0.507 and ±0.540. GR's 1.751 ″
sits at **0.88 σ**.

The improvement is smaller than the √2 that fully independent noise would give (±0.507 →
±0.36), and that is the r = 0.484 of §3f showing up again from the other side: about half the
per-star scatter is common to both blocks and no amount of averaging removes it.

### Two things the union settles

**Single-witness stars buy nothing — the same finding León made.** The 31 single-witness stars
add 50 % more stars and change L by 0.004 ″ while multiplying the error bar by **9×** (±0.430 →
±3.863). León's master-versus-union test said the same in different words: its eleven extra
single-witness stars bought ±0.61 against ±0.60. The two-witness rule is not a cleanup applied
to the union; it is the union's admission rule.

**The plate scale does not drive L here.** The union adopts its host's plate solution whole and
averages only the displacements, so hosting is a real choice and it was tested rather than
assumed (`HU_UNION_HOST=gain0`). Hosting on gain 0 instead moves the fitted scale by 70 ppm —
and **L not at all: 2.129 ± 0.430 ″ either way, rms 0.550 ″ both.** At §3f's naive leverage
70 ppm would be 2.3 ″ of L. It is worth 0.000. That is the clean statement of something §3f
only half-said: the 0.0324 ″/ppm lever describes **imposing** a wrong scale, which is Method 1's
exposure; with the scale free, the data sets it and L is left alone.

So the two blocks never disagreed about the deflection. They disagreed by per-star noise, and
averaging it is what the union is for.

## 3h. The vertical nuisance: measured, and not applied

Douglas, 2026-09-10: *"The Leon 2026 analysis also used a vertical nuisance filter. Are we able
to do that here or do we need to do further analysis first?"* Further analysis first — and it
has now been done. `tools/husillos2026/hu_vertical.py`. **The answer is that we could, and that
on this data it would remove nothing.**

León's estimator adds a degree-2 polynomial surface in field position to the **vertical
component only** (`step3_s2_union.design`, the `v{i}{j}` columns); it is the difference between
"L base" and "L v-deg2" in every León table. Two things had to be true for León first, and
neither transfers by assumption.

### 1. Where is the vertical on this sensor?

León applies the surface along the **sensor y axis**, which is only the vertical because León
measured it: its −y axis sits **3.6°** from the local vertical.

| | León | **Husillos** |
|---|---|---|
| field-centre altitude at mid-time | +9.9° (C2) | **+9.10°** (Sun itself 8.72°) |
| sensor −y from the local vertical | 3.6° | **−14.9°** |

So the surface **cannot** be applied along sensor y here; it would have to be fitted in a
rotated frame. That is a small, well-defined piece of work, not a blocker.

### 2. Is anything vertically polarised?

Yes — and this is the part worth keeping. On the 63 two-witness union stars:

| | León union | **Husillos union** |
|---|---|---|
| vertical rms | 0.898 ″ | **0.506 ″** |
| horizontal rms | 0.363 ″ | **0.280 ″** |
| **V/H** | **2.5** | **1.80** |

Husillos is polarised along the local vertical at 1.8×, against León's 2.5 and the 2.4 León's
night maps measured on other fields on other nights. **Two stations, 130 km apart, at the same
altitude on the same afternoon, with different telescopes and different mounts, both find the
displacement field polarised along the vertical.** That is an independent confirmation of the
atmospheric term León's budget is built on, and it is the most valuable thing in this section.

Note what it costs to look in the wrong frame: **on the raw sensor axes Husillos reads y/x =
0.98**, perfectly isotropic. The polarisation is invisible until the 14.9° rotation is applied,
which is the same fact as §1 seen from the other side.

### 3. But no smooth surface reproduces

The reason not to apply the filter. A k-parameter least-squares fit on n points removes
√(k/n) of the rms from *white noise alone*, so "the rms went down" is not evidence of
structure; and the test that cannot be fooled is held-out stars — the same star-split
cross-validation that settled the zenith field's polynomial order (§`HUSILLOS2026_ZENITH.md`).

| degree | params | vertical rms after | pure-noise expectation | **held-out gain** |
|---|---|---|---|---|
| 1 | 3 | 0.5050 ″ | 0.4935 ″ | **−5.7 %** |
| 2 (León's) | 6 | 0.5035 ″ | 0.4810 ″ | **−30.3 %** |
| 3 | 10 | 0.3953 ″ | 0.4638 ″ | **−11.7 %** |

Degrees 1 and 2 remove **less than white noise would**. Degree 3 looks impressive in-sample
(0.506 → 0.395, well past its noise expectation) and **fails on held-out stars by 11.7 %** —
textbook overfitting, caught by the one test that catches it.

So Husillos' vertical excess is entirely **patchy**, with no smooth low-order component at all.
León described its own residue as "patchy rather than smooth" too, but at León a deg-2 surface
still took 0.898 → 0.763 ″; here it takes 0.506 → 0.504 ″.

**Conclusion, and it is provisional — see §3i.** On this evidence the vertical nuisance is not
applied to cell 4 and the record's L stands at the unfiltered fit. But the evidence above is
the *eclipse field's own residuals*, and that is not how León decided the question.

## 3i. — and §3h asked it of the wrong data. Husillos has horizon fields.

Douglas, 2026-09-10: *"For the Leon site, we took horizon data and I believe from this we
defined the vertical nuisance. Is that true? We do have horizon data for Husillos as well."*

**True on both counts, and it invalidates §3h's decision** — though not its measurements.

### How León actually defined it

Not on its eclipse field. On **horizon night fields**, in two separate steps
(`docs/STEP3_2026.md` §S1, `tools/step3_atmosphere_maps.py`):

* **the direction** came from the night maps — *"M3 measured the wavefield vertically
  polarised at V/H ≈ 2.3"* — measured across the nine **horizon field-windows** (H1 = the
  eclipse alt/az, H2 = +2°, H3 = the calibration sightline) at alt 8.5–12.4° over three
  nights, alongside the twelve zenith fields;
* **the degree, and the vertical-only form**, came from a **null test on those same night
  fields**, where L is known to be zero:

| variant | N1 null | N2 null | N3 null | **worst** |
|---|---|---|---|---|
| base (no nuisance) | −0.77 | −0.04 | +0.67 | 0.77 ″ |
| **vertical deg-2** | −0.19 | +0.32 | −0.19 | **0.32 ″** |
| vector deg-2 | −0.75 | −0.15 | +0.57 | 0.75 ″ |
| vector deg-3 | −0.49 | +0.83 | +0.63 | 0.83 ″ |

Vertical-only beat both vector variants **on real atmospheres** — M3's polarisation
measurement vindicating itself in the estimator's own behaviour — and the deg-2 cut the
atmospheric inheritance **2.4× on all three nights**, which is where León's ±0.33 ″
atmosphere term comes from. (León's formal gate still *failed*: 0.32 ″ is ~6× its floor, so
the filter is applied and the residual systematic quoted rather than claimed away.)

So the object León scored the nuisance against was a field with **no deflection in it** and
hundreds of stars. §3h scored it against the eclipse field's 63 residuals, which contain the
signal being measured and offer no known-zero truth. **That is the wrong test, and its
conclusion is withdrawn pending the right one.** What survives from §3h is its two
measurements: the 14.9° sensor-to-vertical angle, and V/H = 1.80.

### Husillos has the equivalent data — on the eclipse night, at both eclipse gains

`tools/husillos2026/hu_horizon.py`, read from the SER headers and the SharpCap settings files:

| window | captures | UTC | gains | frames | what it is |
|---|---|---|---|---|---|
| `cal 8 deg` | 3 | 20:53–20:59 | 0, 0, 125 | 100 each | **the eclipse altitude** (Sun 8.72° at 18:29:20) |
| `10 deg` | 7 | 21:31–21:44 | 0 and 125 | 100, 100, 100, 14, 51, 50, 50 | ~~~+2° — León's H2 analogue~~ **three pointings**: 10.0°, 5.7° and 15.0° by plate solve (§3s) |

93.6 GB, all 1.000 s at offset 220 — the same settings as the zenith field, which is why the
`cal 8 deg` captures were already the source of the hot-pixel mask (§3c). They sit 2.4–3.2
hours after totality **on the same night**, and they come at **both eclipse gains**, so they
can be unioned exactly as the eclipse blocks are.

`tools/husillos2026/hu_horizon_reduce.py` stacks them and fits them against the same zenith
quintic reference at the same rung, refraction on. Stage 1 uses the **zenith star-field
preset** — there is no Sun in these frames, so no occulter, no coronal subtraction and no
saturated blob — and the synthetic hot-pixel dark is applied.

**What this unlocks, in order:** the polarisation measured on a null field instead of on the
science field; a Husillos null test that can gate the nuisance the way León gated it; and —
the larger prize — **an atmosphere term for cell 4 at all**, which §3e listed as simply
missing (*"no atmospheric data — Joe took none, so the ±0.11–0.33 ″ every other cell carries
from zenith nulls has no counterpart"*). That statement was wrong. He took the fields that
matter most, and they were sitting on the drive under `cal 8 deg` and `10 deg`.

## 3j. CalibS reduced — cell 4 gets an importable plate scale, and Method 1 at last

Douglas, 2026-09-11: start the calibration field taken after the two eclipse blocks.
`G:\Joe Izen Spain 2026\2026-08-12\CalibS_Joe_20260812_183018\20_30_18.ser`, 145 frames,
9576×6388, 315.0 ms, **gain 0**, offset 200, 17.7 GB, sensor −0.2 °C.
`tools/husillos2026/hu_calibs.py`.

### Where it sits in time — *after* the eclipse blocks, not before

| against | | CalibS is |
|---|---|---|
| eclipse gain 125 mid (18:29:20) | | **+71 s after** |
| eclipse gain 0 mid (18:29:59) | | **+32 s after** |
| C2 (18:29:00) | | +91 s after |
| C3 (18:30:44) | | 13 s **before** |

The capture runs 18:30:18 → 18:31:04, beginning 19 s after the gain-0 block ends. It is the
last thing shot inside totality. (An earlier draft of this document said "106 s before the
eclipse blocks' mid-times" — that was the gap to the *Sun capture's start*, quoted against
the wrong reference and with the sign reversed. Douglas caught it.)

### The C3 boundary, measured rather than divided

C3 falls 25.87 s into a 45.7 s capture — frame 82.0 by the clock. But **CalibS is not pointed
at the Sun**: no pixel saturates in any of the 145 frames (band max 2399–10417 ADU against a
65535 full scale), so there is no disk, no beads and no photosphere. It is a genuine **offset
calibration field**, which is what the ladder's middle rung is supposed to be.

So C3 shows itself in the **sky level** instead, and unmistakably:

| frames | band sky | behaviour |
|---|---|---|
| 57 → 82 | 1852 → 1913 ADU | creeping at ~2.4 ADU/frame — the last of totality |
| **83 onward** | +7, +11, +13, +17 … per frame | **accelerating every frame** |
| 140 | 8002 ADU | 4.4× its starting value |

The knee is at **frame 82–83** against the clock's 82.0. Two independent routes, one answer.
**Frames 1–81 are stacked**, mid-time **18:30:31 UTC** from their own SER timestamps.

### The fit, and why it matters

| | |
|---|---|
| stars used | 26 (from 29 centroids) |
| rms | 0.4818 ″ |
| **plate scale** | **2.2029895 ″/px, ±25.2 ppm** |
| pointing | RA 149.2185, Dec 7.5042, roll 326.2099 |
| at 18:30:31 | **alt 8.82°, az 272.00°** (the Sun: 8.51°, 282.28°) |
| separation from the Sun | **10.17°** |

| field | plate scale | CalibS is |
|---|---|---|
| eclipse gain 0 | 2.2029009 | **+40 ppm** |
| eclipse gain 125 | 2.2027459 | +111 ppm |
| night zenith | 2.2059136 | **−1326 ppm** |

**That is the whole point.** The night zenith is 1326 ppm away and unusable as an import; CalibS
is 40 ppm from the gain-0 block. At 0.0324 ″/ppm that is 1.3 ″ of L against 43 ″. **Cell 4 has
a same-day, same-altitude importable scale for the first time.**

Two supporting checks came out right. The camera was **not rotated** all night — roll 326.148
(eclipse), 326.210 (CalibS), 326.266 (a night horizon field), three independent solves inside
0.12°. And the geometry is a proper calibration sightline: same air, 0.3° apart in altitude.

### ⚠ The detection settings were wrong, and this fit is provisional

Douglas, 2026-09-11: *"I'm surprised there were fewer stars detected at gain zero and 315 ms
than the corresponding exposure with the Sun. Were the same sensitivity settings used?"*
**They were not.** Read back from the runs' own records:

| stage-1 option | **CalibS** | eclipse gain 0 | eclipse gain 125 |
|---|---|---|---|
| `field preset` | **zenith** | custom | custom |
| sigma threshold detection | **5.0** | 4.0 | 4.0 |
| `min_area` | **4** | 2 | 2 |
| `sigma_subtract` | **3.0** | 0.0 | 0.0 |
| Gaussian-subtracted detection | **False** | True | True |
| **centroids** | **29** | **115** | **97** |

Every difference runs in the strict direction for CalibS. The original reasoning was half right
— this field has no Sun in it, so it needs no occulter and no coronal subtraction — but it then
took the zenith preset's **detection thresholds** as well, which nothing required. Two things
were changed at once and only one was justified.

It is not tidiness. CalibS' ±25.2 ppm is the **binding term in the whole cell**, and
`mee2024/field_presets.py` requires that an imported scale be centroided the way the science
field was: the *estimator* does match (windowed 2.0 px, annular — verified), but detection
decides which stars are admitted and how bright they are, and this optic's centroid bias is
brightness-dependent. A re-run under the eclipse settings is in progress; if the scale itself
moves, §3k and the drift analysis below move with it.

## 3k. Method 1, at the ladder's third rung

`tools/husillos2026/hu_method1.py`. The pathway, which is **León's for CAL_piLeo, rung for
rung** (`docs/STEP3_2026.md` ladder table; `tools/step3_master_vs_union.py:111`):

| step | option | frozen from |
|---|---|---|
| night zenith | free quintic | — |
| CalibS | `distortion_fixed_coefficients=quadratic`, free scale | cubic+ from the zenith |
| eclipse blocks | **`constant`, `distortion_free_scale=False`** | linear+ from CalibS, **scale imported** |

The chain was verified rather than assumed: **15 of 15 cubic-and-above terms in CalibS are
bit-identical to the zenith's**, and all 6 constant/linear/quadratic terms were refitted. So
CalibS' stored model is already the composite object — same-day low order, night high order.

**The scale import is not a separate switch.** `distortion_fitter.py:538` grants it only when
`constant` **and** `free_scale=False` hold together, so `constant` is mandatory for Method 1
here. Both blocks read back `fixed distortion order: constant` and *`plate scale source:
imported from the reference files`*, carrying CalibS' 2.2029895 exactly; the tool refuses to
continue otherwise.

| pathway | N | rms | **L ± stat** | plate scale | GR at |
|---|---|---|---|---|---|
| **Method 1** (imported) | 63 | 0.594 ″ | **2.840 ± 0.884 ″** | 2.202989 *imported* | 1.23 σ |
| Method 2, on the Method 1 residuals | 63 | 0.594 ″ | 2.062 ± 0.528 ″ | 2.203042 fitted | 0.59 σ |
| **Method 2** (fitted straight against the zenith, scale free) | 63 | 0.550 ″ | **2.129 ± 0.430 ″** | 2.202674 fitted | 0.88 σ |

*(An earlier version of this table called those two rows "Method 2, CalibS rung" and "Method 2,
zenith rung". **Both terms were my coinage**, appear in no specification, and the second is
actively misleading: the ladder's rungs are the zenith reference, the daytime L/R calibration
and the eclipse field, so "the zenith rung" would name the first of them — where what is meant
is the eclipse field SKIPPING the middle rung, as Station 1 does. Douglas, 2026-09-11.)*

Only the second of those is Method 2 for this cell. The first is Method 2's estimator run on
residuals whose stage 2 had already **imported** CalibS' scale and frozen its linear and
quadratic — useful for isolating what the estimator alone does, not a pathway.

*(2026-09-12, §3s: the second row's pathway — quadratic free against the zenith — passes only
f = 0.863 of a 1/r deflection through to stage 3, measured by injection; the first row's rung
passes all of it. Read the 0.067 ″ agreement between them with that in mind.)*

### The gap between the methods is the scale, exactly

Method 1 and Method 2 sit on the **same 63 stars and the same residuals**, differing only in
whether the scale is imported or fitted:

| | |
|---|---|
| the gap | **0.778 ″** |
| the scale difference | **+24.1 ppm** |
| × the measured lever (0.0324 ″/ppm) | **0.779 ″** |

They agree to better than a thousandth of an arcsecond. The lever computed from this field's own
geometry (h = 34.24 R☉²) is confirmed by the fit's behaviour — and it means **the two methods
are not independent evidence**, but one measurement at two plate scales. The difference also
sits **inside CalibS' own ±25.2 ppm** (±0.82 ″), so they are consistent.

**Method 2 remains the more robust number**, and it is stable across rungs: 2.129 ± 0.430
against the zenith and 2.062 ± 0.528 against CalibS — 0.067 ″ apart on completely different
frozen coefficients. Method 1 is real but weaker (±0.884 ″), and always was going to be: the
`constant` rung also freezes the blocks' linear and quadratic from a 26-star fit, which costs
0.18 and 0.23 ″ of stage-2 residual and widens the per-block spread to 2.50 ″ (1.650 and 4.153)
against Method 2's 1.19 ″.

**That per-block spread is not outliers.** The two nonsensical stars were in the gain-0 block
and the two-witness rule removed them long before; the worst Method 1 deflection is 1.69 ″, and
dropping the eight worst of 64 moves L only 4.153 → 3.636. It is the imported scale: stage 3's
own Method 2 says gain 0 wants +14.1 ppm above the import and gain 125 wants +35.0 ppm, which at
the lever are +0.46 and +1.13 ″ — exactly the M1−M2 gaps in each block.

## 3l. The plate scale rises through totality — suggestive, not yet measured

Douglas, 2026-09-11, asked whether the block closer in time to CalibS gave the better scale fit,
implying a temperature-driven drift. **It does, and the ordering is monotonic:**

| field | mid UTC | gain | stars | plate scale | ±ppm | vs earliest |
|---|---|---|---|---|---|---|
| eclipse gain 125 | 18:29:20 | 125 | 84 | 2.2027459 | 21.0 | 0 |
| eclipse gain 0 | 18:29:59 | 0 | 73 | 2.2029009 | 19.0 | **+70.3 ppm** |
| CalibS | 18:30:31 | 0 | 26 | 2.2029895 | 25.2 | **+110.6 ppm** |

All three on one rung (`quadratic`, zenith reference, free scale), so directly comparable. The
sign matches the thermal reading the record already established: a cold tube reads a **larger**
scale, so warming lengthens the focal length and shrinks it; totality removes the heating and the
scale comes back up.

| cell | night → day | across totality | ppm/s |
|---|---|---|---|
| Bruns 2017 | −466 ppm | **−45.1** (R8 → L) | −0.358 |
| Mexico 2024 Station 2 | −725 ppm | **+33.7** (right → left) | +0.157 |
| **Husillos 2026** | **−1326 ppm** | **+110.6** | **+1.558** |

**But it is not established, for three reasons.** The rate is **10× Station 2's** — 2.3 K/min of
sustained cooling at the tube's −40 ppm/K, against Station 2's 0.24 K/min, on the same class of
telescope. The only pair at fixed gain is gain 0 → CalibS, **+40.2 ± 31.6 ppm, 1.3 σ**. And
three variables move together across those three fields: **time, gain** (125 vs 0) **and
pointing** (CalibS is 10.17° away, where a 0.3° altitude difference interacts strongly with a
refraction correction driven by *assumed* weather).

Cell 4 is also structurally weaker than Bruns here: **CalibS is one field on one side.** There is
no R to pair with it, so Bruns' mean-of-two and his half-split error bound are both unavailable.
The record prices exactly this — Bruns' bracketed reference is 10.3 ppm, León's one-sided CAL is
25 ppm, and CalibS' ±25.2 ppm is one-sided in the same sense.

**The test that settles it** (`tools/husillos2026/hu_halves.py`, set up and part-run): split
**both** eclipse blocks in half and refit. The two halves of one block share a gain and a
pointing and differ only in time, so all three confounds go at once, and two blocks give two
independent estimates of the same rate.

| half | gain | frames | mid UTC | separation |
|---|---|---|---|---|
| `g125_A` | 125 | 46–108 | 18:29:09.9 | **19.9 s** |
| `g125_B` | 125 | 109–171 | 18:29:29.8 | |
| `g0_A` | 0 | 2–52 | 18:29:51.2 | **15.8 s** |
| `g0_B` | 0 | 53–102 | 18:30:06.9 | |

At +1.558 ppm/s the halves should differ by 31 and 25 ppm, same sign in both. Comparing the
halves' *absolute* fitted scales is underpowered by construction (~±30 ppm each against a 31 ppm
signal), so the tool also measures the **differential on shared stars**: the catalogue positions,
the frozen cubic-and-above, the refraction model and the pointing are identical between halves
and cancel exactly, leaving only centroid noise. The slope of radial displacement against radius
*is* the fractional scale change.

**A timing trap found on the way.** The Sun capture's per-frame timestamp trailer is present at
the correct size (1440 bytes for 180 frames) but was **never written** — `read_timestamps`
returns None for exactly this, "the space is there and unwritten (an aborted capture)", the same
capture SharpCap never finished delivering. Sn2's trailer *is* written. The fallback reads
`StartCapture` and `ActualFrameRate` from each capture's own settings file, and the two run at
3.1705 and 3.1704 fps — not the rate the 315 ms exposure implies. On a test that is entirely
about time, this would have corrupted the answer silently.

## 3m. The `Capture` folder is not zenith fields — it is the sensor

`tools/husillos2026/hu_capture.py`. Four captures under `2026-08-12/Capture`, but SharpCap
folder names are local time and the headers are UTC: these are **22:54–23:01 UTC on 11 August**,
the night *before* the eclipse, at **offset 50** where everything else in the dataset is 200 or
220.

| capture | start (UTC) | frames | exposure | gain | sensor |
|---|---|---|---|---|---|
| `00_54_32` | 11 Aug 22:54:32 | 100 | 1.000 s | 125 | **33.7 °C** |
| `00_57_43` | 11 Aug 22:57:43 | 100 | 315 ms | 125 | **32.0 °C** |
| `00_59_36` | 11 Aug 22:59:36 | 100 | 315 ms | 125 | — |
| `01_01_09` | 11 Aug 23:01:09 | 100 | 315 ms | 0 | — |

315 ms at gains 125 and 0 is the eclipse acquisition's own setting, which reads like a rehearsal.
**But all four are empty of sky.** All plate solves failed — 18, 6 and 3 centroids, and
`01_01_09` could not even match frame 0 to frame 1.

| capture | spikes > 20σ | **on the hot-pixel mask** | resolved stars / 5 Mpx |
|---|---|---|---|
| `00_54_32` | 161 | **161 (100 %)** | 1 |
| `00_57_43` | 146 | 137 (93.8 %) | 2 |
| zenith | 120 | 54 (45 %) | **68** |

`00_54_32` is 1.0 s at gain 125 — a *deeper* configuration than the zenith's 1.0 s at gain 0 —
and returns 1 star against 68. Focus is fine (FWHM 1.36–1.60 px, same as the zenith's), the field
is flat to 0.12–0.18 % so there is no cloud structure, and **offset 50 is not clipping anything**
(zero pixels at the minimum, a clean ~400 ADU pedestal). What remains is a hot uncooled sensor at
32–34 °C against the zenith's 25.6 °C and `cal 8 deg`'s 2.8 °C, and no sky.

**Most likely these are dark or cap-on frames**, which sits awkwardly against Joe's report that he
took no darks — worth asking him rather than asserting. Either way: the pointing is undetermined
and will stay so, `00_54_32`'s stage-1 failure is explained, and they are **not** the second
star-rich pointing the distortion-order transfer test wants.

**Two measurement traps recorded here**, both of which caught me first. A 1-px matched filter
"finding ~2600 sources" in one of these frames was finding the sensor, and so was my own count of
736 peaks above 10 σ. Requiring a source to have **neighbours** — a star has a PSF, a hot pixel
does not — collapses those to 1. Before that filter, FWHM measured 1.13 px on *every* capture
including the zenith, which is a one-pixel source and not a PSF.

## 3n. The mount was still settling through CalibS — and the AM5's settling time

Douglas, 2026-09-11, reading `TWOD_RESIDUALS20260911025309.png`: *"this suggests to me
something was wrong with the tracking; perhaps at the beginning of the SER files the mount was
still slewing."* Correct, and it is CalibS. From stage 1's own per-frame alignment record:

| field | frames | **total drift** | max step |
|---|---|---|---|
| **CalibS** | 81 | **23.0 px** | 2.5 px |
| eclipse gain 0 | 101 | 1.0 px | 0.8 px |
| eclipse gain 125 | 126 | 3.2 px | 0.7 px |
| zenith | 50 | 1.2 px | 0.4 px |

### The totality timeline (`tools/husillos2026/hu_timeline.py`)

**C2 18:29:00.5 → C3 18:30:44.2, 103.7 s.**

| block | start UTC | end UTC | dur | frames | gain | used |
|---|---|---|---|---|---|---|
| 1 Sun | 18:28:45.593 | 18:29:42.366 | 56.8 s | 180 | 125 | 46–171 |
| 2 Sn2 | 18:29:42.689 | 18:30:15.177 | 32.5 s | 103 | 0 | 2–102 |
| 3 CalibS | 18:30:18.333 | 18:31:04.048 | 45.7 s | 145 | 0 | 1–81 |

* **1 → 2: 0.323 s = 1.02 frame intervals.** No real gap — one frame period, which is why
  SharpCap's buffer handed Sn2 the last two frames of the Sun capture (§2).
* **2 → 3: 3.156 s = 10.01 frame intervals.** A real gap, and it contains the slew.

**The slew is entirely inside the gap.** It is 10.17° = 36 612 ″; block 3 moves only 49.6 ″
across its whole capture, **738× smaller**. So the mount finished slewing before block 3's first
frame, and what block 3's alignment records is the **settling tail**. Douglas, 2026-09-11: the
AM5 slews at about 6 °/s, so the slew itself took ~1.7 s plus ramps — leaving roughly 0.6–1.5 s
of wait, not the 3.2 s an earlier draft of this section claimed by assuming the slew filled the
gap. (That was an upper bound on duration quoted as a measurement.)

### The settling time

Fitted on the **rate**, not the displacement — block 3 starts part-way through the settle so its
zero is arbitrary, and the rate does not care where the clock started:

> **τ = 9.2 s** after a 10.2° slew (15 windows, r = −0.90). **27 s to 5 %**, 42 s to 1 %.

*(2026-09-13, on Douglas revisiting the settling question.)* The same 81-frame displacement
admits a second description, and the record should carry both. Fitted as **exponential plus a
steady drift** — the settle riding on tracking error — it gives **A = 33.9 ± 1.6 ″, τ = 4.6 ±
0.3 s, drift 0.67 ± 0.07 ″/s (40 ″/min)**, residual 1.25 ″ rms (frame-to-frame image motion at
8.8°). The two fits agree on what matters and differ on what a 25 s capture cannot settle:

| | pure exponential (the record) | exponential + drift |
|---|---|---|
| velocity at frame 1 | 5.5 ″/s | 8.1 ″/s |
| smear inside the first 0.315 s exposure | 0.8 px | 1.15 px |
| velocity at frame 20 (6.3 s) | 3.8 ″/s | 2.5 ″/s |
| velocity at 20 s | 0.6 ″/s | 0.76 ″/s (0.67 of it drift) |
| what the frames-21–81 stack still carries of the settle | 15 ″ | 8.5 of its 20 ″ |

**`calibs_settling.png`** (`tools/husillos2026/hu_settle_chart.py`, in RECORD) draws the 81
points with the pure exponential and its residuals. Fitted on the *displacement* rather than
the rate, it returns **τ = 7.3 ± 0.2 s, A = 50.1 ″** (residual 1.58 ″ rms); the
exponential-plus-drift alternative (4.6 s, 1.25 ″ rms) was drawn on this chart through its
sixth revision and is left out from the seventh (Douglas, 2026-09-14: it is not reconcilable
with the by-axis chart, which fits a pure exponential per axis, and the "drift" is now known
to be the RA tail). It survives in `hu_settle_models.py` and in the tables below. So the
record's rate-based 9.2 s is the upper end of three estimators that all say the same thing at
the scale that matters. **The total and the by-axis charts reconcile exactly**: projected onto
the drift's own direction the total is 0.992 × RA + 0.125 × |Dec|, so the 0.9 s Dec settle
contributes 0.8 ″ of the 49.6 ″ and a single exponential through the total returns the RA
axis's constant, 7.3 against 7.4 s. The residual panel
shows what neither model has: a **+3 to +5 ″ overshoot at 4–5.5 s** (frames 13–18) that
decays within two seconds — a single damped oscillation on top of the exponential creep, the
signature of a strain-wave drive coming to rest.

A 10.17° slew at 6 °/s moves 6810 ″ in one exposure; frame 1 moved ~2.5 ″. **The mount was
not slewing at frame 1 under either model; it was settling.**

**`23_44_06`, the same treatment, decides between the two models** (`h10_g125d_settling.png`,
`hu_settle_chart.py --capture h10_g125d`). It is the first capture after the 14.3° slew from
pointing B to pointing C, almost all in declination, and the slew can be dated from
`23_42_43`'s own frames: frame 47 (opened 21:43:44.6) aligned to frame 0, frame 48
(21:43:45.9) did not, so the slew began at ~21:43:45.6; `23_44_06` frame 1 opened at
21:44:07.0, 21.4 s later, and the slew itself took ~2.5–3 s. **The capture begins 18–19 s after
the mount stopped, and shows nothing left to settle**: 49 frames over 63 s move 2.87 ″ in a
straight line, **2.5 ″/min**, residual 0.36 ″ rms — the AM5's zenith tracking rate to the
decimal (§HUSILLOS2026_ZENITH: 2.48 ″/min). Neither exponential model converges on it; there
is no curvature to fit. Against the two CalibS fits carried forward to 18.5 s after a slew
*(the chart drew both predictions through its third revision and from the fourth draws the
pure exponential's only, to match the CalibS chart — Douglas, 2026-09-14; the table keeps
both)*:

| CalibS model | predicts for `23_44_06` | observed |
|---|---|---|
| pure exponential, τ = 7.3 s | 4.0 ″ of settle still to come | < 0.4 ″ |
| exponential + drift, τ = 4.6 s, 40 ″/min | 0.6 ″ of settle + **42 ″ of drift** over 63 s | **2.9 ″** |

The 40 ″/min steady drift is refuted outright — the same mount, twenty minutes later, tracks
at 2.5 ″/min — so that term in the CalibS fit was a slow settling component, not tracking. And
the pure exponential's τ = 7.3 s over-predicts the residue too: by 18 s this slew had settled
to better than 0.4 ″. The two slews differ in size (14.3° against 10.2°) and axis (Dec against
a mixed slew from the Sun), and a strain-wave mount need not settle alike on both.

**Do the two charts agree? At face value no; split by axis, yes** (Douglas, 2026-09-13). Taken
as one settling curve they disagree by a factor of two or more: CalibS is still moving at
~36 ″/min 20–25 s after its slew and would reach the 2.5 ″/min floor only at 37–45 s on either
of its fits, while `23_44_06` is *at* the floor 18–19 s after its slew. Resolving each
capture's drift vector into RA and Dec on the sky, through the affine of its own matched
stars, shows where the disagreement lives:

| | CalibS drift | `23_44_06` drift | zenith drift |
|---|---|---|---|
| total | 49.6 ″ in 25.5 s | 2.9 ″ in 64.5 s | 2.6 ″ in 50 s |
| RA·cos δ component | **+49.2 ″** | −1.5 ″ | −0.8 ″ |
| Dec component | −6.2 ″ | +2.4 ″ | +2.5 ″ |
| direction | **7° from the RA axis** | 58° (mixed) | 72° (mostly Dec) |

*(Corrected 2026-09-14. The stacker stores `shifts_px` as **(row, column) = (y, x)**, and every
RA/Dec split in this section had read it as (x, y). The error was caught by calibrating the
split on the three untracked captures, whose pointing is known to have moved purely in RA at
the sidereal rate: read as (y, x) they resolve to RA +1692 / +1711 / +1765 ″ and Dec −2 / +4 /
−16 ″; read as (x, y) they acquire a spurious 22° Dec component. The swapped values are struck
through below where they were quoted; the total-drift charts are unaffected, since a projection
onto the drift's own direction does not depend on the order.)*

The AM5's steady tracking drift is ~2.5 ″/min, mostly **in Dec** with ≤ 1.4 ″/min in RA, at the zenith and at
15° alike. The CalibS slew moved the two axes almost equally — RA +7.12°, Dec −7.40°, the Sun
at RA 142.106° Dec +14.907° to CalibS at 149.227° +7.503° (an earlier draft of this paragraph
said "mostly the RA axis", reading the *drift* direction as the *slew* direction; corrected
2026-09-14) — and both axes settled; splitting its 81
frames per axis (`hu_settle_chart.py` prints it):

| CalibS, per sky axis | total | τ (pure exp.) | measured rate, first 3 s | **measured rate, last 5 s (20–25 s)** |
|---|---|---|---|---|
| RA·cos δ | **+49.2 ″** ~~43.2~~ | **7.4 ± 0.2 s** ~~7.7~~ | 5.31 ″/s | **0.49 ″/s = 29 ″/min** |
| Dec | **−6.2 ″** ~~24.5~~ | **0.9 ± 0.3 s** ~~6.0~~ | −1.77 ″/s | **0.15 ″/s = 9 ″/min** (noise: the settle is over by ~3 s) |

**`calibs_settling_by_axis.png`** (`hu_settle_chart.py --capture calibs --by-axis`, in RECORD)
draws the two components on one time axis, each with its own pure-exponential fit and the
2.5 ″/min tracking floor as the slope a settled axis would show.

**`h10_g125d_settling_by_axis.png`** is `23_44_06` drawn the same way (Douglas, 2026-09-14).
Neither component has an exponential to fit — the tool falls back to a line when the
1 σ error on τ exceeds τ — and the lines are the floor itself:

| `23_44_06`, per sky axis | total over 63 s | linear rate | residual rms |
|---|---|---|---|
| RA·cos δ | −1.5 ″ ~~−0.5~~ | −1.4 ″/min | — |
| Dec | +2.4 ″ ~~−2.8~~ | **+2.3 ″/min** | — |

*(Corrected 2026-09-14 with the (row, column) fix; the drift is 58° from the RA axis here, at
the floor on both axes.)*

Dec runs down the −2.5 ″/min floor line from frame 1; RA is flat to a slow ±1 ″ wander over
the minute, which is the RA drive's periodic error at a level nothing here needs to resolve.
Put beside the CalibS chart it is the same two axes in their finished state: Dec at the floor,
RA at rest — where 18 s earlier, after a slew that moved the RA axis, RA had not been.

**The Dec axis was at the tracking floor by 20 s in CalibS too** — 3 ″/min against the mount's
2.5 ″/min Dec drift — exactly as `23_44_06` shows for its Dec-only slew. What was still moving
at 25 s in CalibS was the **RA axis**, at 29 ″/min ~~31~~, and `23_44_06` never exercised the RA axis
because its slew was in declination. So the two captures agree on everything they both
measure, and CalibS alone measures the one thing that matters for a slew from the Sun to a
calibration field along the ecliptic: **the RA drive settles more slowly than Dec** — its
pure-exponential τ is eight times longer (7.4 against 0.9 s ~~7.7 against 6.0~~), and at 20–25 s it still carried a 29 ″/min
creep that either is a slow second component of its settle or is the tracking re-engaging
with a transient rate error; a 25 s record cannot tell those apart, and nothing else in the
campaign followed an RA slew. The "exponential + drift" fit on the combined displacement was
picking up this RA creep and calling it tracking; the Dec fit's "drift" term (28 ″/min) is
contradicted by its own last five seconds (3 ″/min) and is a fitting artefact.

**For 2027 this splits the rule in two.** After a Dec slew, 20 s is enough. After an RA slew of
~10°, allow the full 30 s and expect a small residual RA creep beyond it; if the design allows,
put the calibration field at the Sun's declination so the slew is RA-only and its settle can
be timed, or at the Sun's RA so it is Dec-only and fast.

### Is the RA/Dec difference the axis, or only the arrival velocity?

Douglas, 2026-09-14: the slew must brake both axes; if RA arrived faster it has more to shed;
an exponential is what damping gives; so is the difference a property of the axes at all, or
just of their initial velocities? `tools/husillos2026/hu_settle_models.py`.

Three things answer it, in order of weight.

**The premise does not hold: the axes travelled the same distance.** Resolving the slew from
the Sun's position at 18:30:15 to CalibS' solved centre, the RA axis moved **+7.12°** and the
Dec axis **−7.40°**. If the AM5 drives both axes on one rate profile, as it appears to, they
arrived together at the same speed. Whatever difference there is between the two settles was
not put there by the slew's kinematics.

**Under linear damping τ cannot depend on the arrival velocity anyway.** For a first-order
linear relaxation — a torsional spring against a viscous or back-EMF damper, which is what an
exponential *is* — τ is the system's own constant and the arrival velocity sets only the
amplitude, A = v₀ τ. So "the same system, different v₀" predicts one τ and two A's, and that is
a testable joint fit:

| joint exponential fit, both axes | A_RA | A_Dec | τ | RSS |
|---|---|---|---|---|
| one shared τ | 43.5 ″ | 24.9 ″ | 7.29 s | 435.4 |
| separate τ | 44.2 ″ | 23.7 ″ | **7.70 / 6.02 s** | 402.5 |

*(Corrected 2026-09-14 for the (row, column) fix; the swapped-axis values were 7.70 / 6.02 s,
F = 12.9, p = 0.0004.)* Shared τ 7.29 s (RSS 435.3) against separate **7.38 / 0.95 s** (RSS
361.2): F(1, 158) = 32.4, **p = 6 × 10⁻⁸** — the data demand two time constants, and not by a
little. With the 3.5–6.5 s overshoot frames excluded, since no model has the overshoot, it is
8.24 against 0.95 s, F = 31.9, p = 9 × 10⁻⁸. The axes differ in τ by a factor of eight, and
equal travel means that is not the velocity's doing.

**A velocity-dependent effective τ needs a nonlinear law, and the nonlinear laws fit worse.**
Coulomb friction (constant deceleration to a hard stop) would make the faster axis take
longer, which is the sign of the observation; quadratic drag would make it take less. Fitted
per axis:

| law | RA rms | Dec rms | what it says |
|---|---|---|---|
| exponential (linear damping) | **1.53 ″** | 1.45 ″ | τ 7.4 / 0.9 s |
| Coulomb friction | 2.62 ″ | 1.46 ″ | RA would stop dead at 17.5 s — the tail says no |
| quadratic drag | 1.57 ″ | 1.38 ″ | half-life 1.7 s on RA |

*(Corrected 2026-09-14, (row, column) fix.)* On RA, where there are 49 ″ and 25 s of settle to
test against, the exponential is the best of the three and friction is clearly the worst,
because the settle has a tail and friction stops dead. On Dec the three are indistinguishable,
because a 6 ″ settle that is over in three seconds does not discriminate anything. A
two-time-constant exponential adds nothing either axis can constrain in 25 s — the second
term degenerates into a straight line on both (on RA it is the 29 ″/min creep again).

**So the difference is in the axes, not in the velocities**, and the settle's own numbers say
how *(corrected 2026-09-14; the swapped-axis draft read 5.7 / 3.9 ″/s and claimed the axes
settled in opposite senses, which was the swap talking)*: with equal travel, the exponential's
implied arrival velocity A/τ is **6.8 ″/s on RA and 5.5 ″/s on Dec** — nearly the same, as
equal travel on one rate profile predicts — and yet RA shed its motion with **τ = 7.4 s over
49 ″** while Dec shed its in **τ = 0.9 s over 6 ″**. Same arrival speed, an eightfold
difference in how long it took to stop: that is an axis property and nothing else. **Both axes
kept creeping in the direction they had been slewing** — the shift record's sign was calibrated
on the untracked captures, whose pointing is known to move to increasing RA, and it reads as the
motion of the pointing; RA continued +49 ″ along its +7.12° slew, Dec continued −6 ″ along its
−7.40° slew. Neither sprang back. The RA
axis carries the Dec assembly and the OTA; the Dec axis carries the OTA alone — and this AM5
was run **without a counterweight**, as strain-wave mounts commonly are with light loads
(Douglas, 2026-09-14; an earlier draft of this sentence put one on the RA axis), so the RA
axis also carries whatever gravity torque the unbalanced load puts on it. Different loads on
the same strain-wave drive give different compliance, different damping and a different τ,
which is what is measured. The electromagnetic-braking picture is
right as far as it goes — the exponential is the mark of linear damping and nothing frictional
fits — but the braking is the controller's business during the slew; what these 25 s show is
the mechanical relaxation after it, and that is a property of each axis.

The chart titles that called this slew "mostly in RA" were reading the drift direction (30°
from the RA axis) as the slew direction, and are corrected.

**What the swapped "Dec" actually was, and how firm the late RA creep is** (Douglas,
2026-09-14: *"Does this mean that your previously measured time constant in Dec of 6.0 s was
completely wrong?"* Yes.) Expressed in the true basis, the swapped components were
old "RA" = 0.92 × RA + 0.38 × Dec and **old "Dec" = 0.38 × RA − 0.92 × Dec**: the old "Dec"
(24.5 ″, τ 6.0 s) was two-fifths of the RA settle with the real 6 ″ Dec settle folded in, which
is why it carried an RA-like time constant. The old "RA" was nine-tenths right, which is why
its τ barely moved (7.7 → 7.4 s). The 6.0 s was not a poor measurement of Dec; it was a
measurement of something else. The late-time rates in the by-axis tables are straight-line
slopes over the final 5 s of points (16 frames) with no exponential involved, so they say
whether an axis is still moving at the end of the record whatever model one believes. With
their formal errors: **RA +29 ± 4 ″/min, Dec +9 ± 15 ″/min.** RA is still moving at 25 s at
seven times its error and twelve times the floor — and faster than the exponential's own late
velocity of 13 ″/min, which the residual panel shows as the RA points rising off the fit after
20 s: the RA tail is longer than a single exponential. Dec is consistent with zero and with the
floor.

**What the frame-21 cut is, and what the stacker does with the rest** (Douglas, 2026-09-14:
*"This seems to be implying that stacking is ok after this time and that the MEE program can
accommodate the drift after this point."*). Two different things happen to a settling capture
and the cut addresses only one of them. *Within* each 0.315 s exposure the stars are trailed by
the mount's velocity: 0.79 px per exposure over frames 0–10, 0.54 over 10–20, 0.27 over 20–30,
≤ 0.21 thereafter, against a 1.6 px PSF (`hu_calibs.py`). Nothing removes that trail; frame 21
is where it falls below about a fifth of the PSF, a judgement, not a measured threshold.
*Between* exposures the field moves, and that the stacker does accommodate: `_align_frames`
fits each frame's shift against frame 0 and `add_img_to_stack` applies it with `np.roll`, a
**whole-pixel** shift, so the between-frame drift is absorbed to ±0.5 px per frame and the
sub-pixel remainders average out as a small broadening rather than a bias. So yes: after the
cut, stacking is sound and the remaining 19 ″ of RA creep across frames 21–81 is taken out by
the alignment, not tolerated. The test that it worked is the settled re-reduction above: the
scale moved +9.5 ppm and the error bar did not, which is what removing a trailing bias and
losing a quarter of the depth should look like. The calibrated (row, column) order of the
alignment record is the same order `add_img_to_stack` rolls by.

**The tracking floor is a magnitude, and its sign carries nothing** (Douglas, 2026-09-14: does
the 2.5 ″/min apply only to Dec, and does the floor line's positive slope against the negative
Dec settle matter?). Measured with tracking on and nothing settling, the residual drift is
2.3–3 ″/min in Dec and 0.8–1.4 ″/min in RA (zenith: RA −0.8 ″, Dec +2.5 ″ over 49 s;
`23_44_06`: RA −1.5 ″, Dec +2.4 ″ over 63 s), so "2.5 ″/min" is the Dec-dominated total and
the RA floor is smaller. The Dec drift is polar-misalignment drift, whose sign depends on the
hour angle and on which way the polar axis is off, so it is not a fixed sign; the by-axis
charts draw the floor as ±2.5 ″/min for that reason, and an axis is "settled" when its slope
lies within that band. The CalibS Dec settle running to −6 ″ while the floor line rises has no
consequence for anything computed: the settle is a transient whose sign is the slew's, the
floor drift is a steady term whose sign is the polar alignment's, and both are pure
translations of the field that the stacker's alignment removes and stage 2's constant term
absorbs. Where the sign would matter — a drift that reversed mid-capture, or a settle and a
drift that partly cancelled so that "at rest" was read too early — nothing in these records
shows it.

**Balance, and an experiment for 2027** (Douglas, 2026-09-14). The Dec axis of a strain-wave
mount is essentially balanced; the RA axis, run without a counterweight, carries a
gravitational torque at every pointing and must also turn at the sidereal rate, so it is never
at rest. In a strain-wave drive the flexspline is a torsional spring, wound up by
torque ÷ stiffness under a steady load. A slew changes the RA axis's gravity torque — the
load's hour angle moved 7° — so the equilibrium wind-up changes and the relaxation to it is the
settle; the balanced Dec axis has almost no torque change to relax, which is what 49 ″ against
6 ″ looks like. So the hypothesis that an RA counterweight (an option on the AM5) would bring
the RA settle toward Dec's is physically motivated. It should shrink the *amplitude*, which is
what a 0.3 ″ tolerance cares about; whether it shortens τ, a property of stiffness and damping,
is less certain, and the added inertia could lengthen it. It is a two-minute experiment on any
night: the same ~10° slew with and without the counterweight, read from the alignment record
or from per-frame solves, and separately an RA-only and a Dec-only slew to isolate the axis
from the load direction. **Recommended for 2027, before the design fixes the calibration-field
geometry** — if RA can be made to settle like Dec, the calibration field can sit anywhere. The 40 ″/min steady term is
suspect: the AM5 tracks at 2.5 ″/min overhead (§HUSILLOS2026_ZENITH) and refraction adds ~8
″/min at 8.8°, so ~10 ″/min is what a settled mount should show here, and the excess is more
plausibly a slower settling component that a 25 s record cannot separate from drift. The
recommendation is the same under both: **wait ~30 s after a ~10° slew.** No other capture in
the campaign is a clean second measurement — `23_44_06` began ~20 s after a 9° slew and its
first frames move at 0.2 ″/s, consistent with both models, and the 14-frame `23_40_27` that
sits right after the slew to pointing B turns out to be untracked (8.07 px/frame, sidereal),
which dates the tracking switch-on to the 17 s between 21:40:45 and 21:41:02 (§3w) and makes
it useless for settling.

Two plots carry the settling and they are not the same. `s1_calibs_ecl_settled`'s
`TWOD_RESIDUALS20260911160647.png` is the frames-21–81 stack and shows only the **tail**: 8.6 px
(19 ″) over 19 s. The full settle is in `s1_calibs_ecl`'s `TWOD_RESIDUALS20260911025309.png`,
frames 1–81: 22.5 px (49.6 ″) over 25 s, the curve visibly decelerating. Against the León AVX
zenith captures (`f16_zenith_test`, 30 × 4 s): 1.7–16.6 px over 128 s, i.e. 2.6–17 ″/min
steady, varying capture to capture with the worm phase. The settled-tail plot looks quieter
than an AVX plot only because it spans 19 s against 128; per unit time the AM5's settling tail
(60 ″/min) is four times the AVX's periodic error, and the AM5's settled tracking (2.5 ″/min)
is six times better.

Peak rate 19.4 ″/s, **470×** the AM5's ordinary 2.48 ″/min tracking drift. What it costs a
0.315 s exposure against a ~1.6 px PSF:

| stack index | 0–10 | 10–20 | 20–30 | 30–80 |
|---|---|---|---|---|
| smear per exposure | **0.79 px** | **0.54 px** | 0.27 px | ≤ 0.21 px |

So the first ~20 frames are measurably trailed, and it bites twice: `add_img_to_stack` aligns
**every** frame against `files[0]`, so the whole stack was referenced to the most disturbed
frame in it. That is Douglas' standing rule from 2026-09-09 — *"better to use a good frame in
the middle of the series than the first one at the beginning"* — failing on the one field whose
plate scale is the cell's binding term. **For 2027: after a ~10° slew, wait ~30 s.**

### CalibS re-reduced on the settled frames

| CalibS variant | stars | rms | plate scale | ±ppm |
|---|---|---|---|---|
| zenith preset, 1–81 | 26 | 0.4818 ″ | 2.2029895 | 25.2 |
| eclipse detection, 1–81 | 86 | 0.6613 ″ | 2.2030097 | 18.7 |
| **eclipse detection, 21–81 (settled)** | **88** | 0.6647 ″ | **2.2030306** | 19.0 |

Dropping the trailed frames **does not improve the error bar** (±18.7 → ±19.0) despite losing a
quarter of the integration — fewer frames costs depth, cleaner stars recovers it, and it finds
two *more* stars. But **the scale moved +9.5 ppm**, which is what mattered: the trailing was
biasing it. Cumulatively CalibS has walked **2.2029895 → 2.2030097 → 2.2030306, +18.6 ppm =
0.60 ″ of L**, under two corrections that were each clearly right, while its formal error bar
says ±19 ppm. **Treat ±19 ppm as a floor, not the budget.**

## 3o. The plate-scale drift hypothesis fails its own test

§3l proposed that the monotonic rise across the three fields was a cooling tube. The test that
removes all three confounds at once — split **both** eclipse blocks in half, so each pair shares
a gain and a pointing and differs only in time (`tools/husillos2026/hu_halves.py`):

| half | gain | frames | mid UTC | separation |
|---|---|---|---|---|
| `g125_A` / `g125_B` | 125 | 46–108 / 109–171 | 18:29:09.9 / 18:29:29.8 | 19.9 s |
| `g0_A` / `g0_B` | 0 | 2–52 / 53–102 | 18:29:51.2 / 18:30:06.9 | 15.8 s |

**The differential on shared stars** (the sensitive form: catalogue positions, frozen
cubic-and-above, refraction model and pointing are identical between halves and cancel exactly,
leaving only centroid noise):

| pair | shared | dt | **Δscale** | **ppm/s** | residual |
|---|---|---|---|---|---|
| gain 125 | 34 | 19.9 s | **−7.9 ± 15.8 ppm** | **−0.397 ± 0.793** | 0.246 px |
| gain 0 | 27 | 15.8 s | **−79.7 ± 21.0 ppm** | **−5.056 ± 1.334** | 0.276 px |

**Neither supports the predicted +1.558 ppm/s.** Gain 125 is consistent with zero and 2.5 σ
below it; gain 0 is 4.9 σ below and *negative*. The absolute comparison agrees more weakly
(−0.970 and +3.182 ppm/s, opposite signs, neither significant). **So the three-field rise is not
a within-block time drift**, and there is no case for extrapolating CalibS' scale back 71 s.

**The two halves disagree with each other at 3.0 σ**, which is its own open question: the gain-0
block shows a real −79.7 ± 21.0 ppm step that gain 125 does not. Candidates: the gain-0 halves
are the shallowest fits here (30 and 54 stars) so an unmodelled low-order term could leak into
the scale; 27 shared stars may be too few for a well-conditioned 4-parameter similarity; or
something genuinely changed during the Sn2 block.

**Two estimator bugs**, both of which produced confident nonsense first. (1) **No intercept**:
each half is aligned against *its own* first frame, so the two pixel grids differ by an
arbitrary whole-pixel translation; regressing radial displacement on radius without one fed that
into the slope and returned −618 ± 500 ppm. Fixed by fitting a full similarity — translation,
rotation, scale — and taking the scale term. (2) **No clipping**: even then the gain-0 pair
fitted at 4.8 px rms against gain 125's 0.29, a few bad centroids dominating a 4-parameter fit
on 27 stars. Three passes at 3 σ bring both to ~0.25 px, and the clip count is reported so a
heavy cut cannot pass unnoticed.

## 3p. The record charts

`tools/husillos2026/hu_field_charts.py` and `hu_covariance.py`, drawn through
`tools/record_charts.py` and published through `hu_record.publish()`.

**The two blocks' displacement fields**, same 64 two-witness stars, one arrow scale, in
**alt/az** as León's `record_field.png` is — the frame the atmosphere is polarised in:

| block | rms vector | radial mean ± sd | tangential mean ± sd | **V/H** |
|---|---|---|---|---|
| gain 0 | 0.650 ″ | **+0.073 ± 0.455 ″** | −0.005 ± 0.459 ″ | **1.43** |
| gain 125 | 0.770 ″ | **+0.153 ± 0.519 ″** | −0.006 ± 0.548 ″ | **1.89** |

The tangential mean is zero to 0.006 ″ in both — the null the geometry demands, and it passing
says neither block is grossly broken. Gain 125's radial mean is 2.1× gain 0's, which is the L
difference seen directly rather than through a fit. **And gain 125's excess is vertically
polarised** (V/H 1.89 against 1.43), so the block reading high in L is the one carrying more
vertical structure — which points at the atmosphere rather than at photometry.

**`record_covariance.png`** puts L against the plate scale, both ellipses on one base (ppm from
CalibS' import), and **`record_covariance_method2.png`** is cell 2's single-ellipse variant on
an absolute scale axis — the honest chart for the number of record, since Method 2 has no
imported scale to measure a ppm difference from.

The tilt is the cell in one picture: **correlation −0.90 (Method 1) and −0.78 (Method 2)**. And
**Method 2's fitted scale sits 162.0 ± 10.3 ppm below CalibS' import** — **8 σ** on CalibS' own
error bar. Two same-day fields 10° apart at the same altitude disagreeing that hard about the
scale is the thing Method 1 rests on.

Note the gap between the two ellipses is **not** the lever times the scale difference: −162 ppm
× 0.0324 would be −5.25 ″ against an observed **+1.16 ″**, opposite sign and five times the
size. Within one run that relation held to a thousandth of an arcsec; across these two it fails
because they are different distortion models, not merely different scale choices.

**Four chart faults corrected, all mine.** Method 2 was first drawn from the Method 1 run
because `Method 1 & 2` prints both adjacently — so it carried an **imported** scale, which is
not Method 2 (Douglas: *"Method 2 should not use an imported platescale"*). The Newton line was
removed. The text box was covering Method 2's ellipse entirely — drawn, then hidden, which is
worse than omitting it — so it is now cut to the two methods' own lines and **placed by a tested
overlap check** against both ellipses' bounding boxes; it rejected upper-left, which is where it
would have gone by eye. And `record_covariance.png` was **overwritten** once, from a run whose
parser read L = 64 ″/px because `np.float64(` contains the digits 64; that revision is not
recoverable, and `hu_record.publish()` now supersedes rather than overwrites.

## 3q. The block disagreement is the GAIN, and the mechanism is saturation — not the clock

Douglas, 2026-09-11: *"We did not see such a problem in the Leon 2026 exposure tiers where the
gain was held constant and the exposure was changed. Is changing the gain while holding the
exposure constant fundamentally different in terms of the corona subtraction?"*
`tools/husillos2026/hu_gain_or_time.py`.

**Yes, and the data says so directly.** But the reason is not the subtraction itself.

### First, why León could not have seen it either way

León's tiers were **interleaved**: the folder timestamps put 0.1 s at 18:28:13–45, 0.3 s at
18:28:21–41, 0.6 s at 18:28:26–37 and 1.2 s at 18:28:29 — all inside one 32 s window. Every
tier saw the same atmosphere at the same moment. Husillos' blocks are **sequential, 39 s
apart**. So *"León did not see it"* is compatible with either explanation, and the question
has to be settled on Husillos' own data.

### The test that separates gain from time

The four half-blocks (§3o) form pairs at the same gain ~20 s apart and across the gain
boundary ~21 s apart. For each, the star-to-star displacement is fitted **about the Sun** with
translation + rotation + scale, then with a 1/r term added, and split into inner and outer
halves by solar radius. A pure scale is the same in both halves by definition; a Sun-centred
structure is not.

| pair | kind | dt | 1/r term | inner (ppm) | outer (ppm) | inner − outer |
|---|---|---|---|---|---|---|
| g125_A → g125_B | same gain 125 | 19.9 s | +0.81 ± 0.68 ″ | +38 ± 34 | −19 ± 14 | 1.5 σ |
| g0_A → g0_B | same gain 0 | 15.7 s | −1.25 ± 1.08 ″ | −114 ± 34 | −64 ± 29 | 1.1 σ |
| **g125_B → g0_A** | **cross gain** | **21.4 s** | **−3.18 ± 1.07 ″** | **−162 ± 31** | −43 ± 18 | **3.3 σ** |
| g125_A → g0_B | cross gain | 57.0 s | −1.96 ± 1.20 ″ | −259 ± 46 | −81 ± 22 | **3.5 σ** |

**The Sun-centred structure appears in both cross-gain pairs and in neither same-gain pair.**
The decisive contrast is the top and third rows: **19.9 s at one gain shows nothing; 21.4 s
across the gain boundary shows it at 3.3 σ.** Same time separation, opposite result. It is
the gain.

On the two full blocks it is monotonic in radius — a scale is uniform, this is not:

| solar radius | scale difference (gain 0 − gain 125) |
|---|---|
| 2.3–6.0 R☉ | **−227 ± 35 ppm** |
| 6.0–8.2 R☉ | −135 ± 30 ppm |
| 8.2–12.1 R☉ | −60 ± 22 ppm |

And it is **not** brightness-dependent: the brighter half of the stars gives −99 ± 20 ppm,
the fainter half −90 ± 25. So saturation *of the stars* and this optic's brightness-dependent
centroid bias are ruled out — which was the leading suspect and is dead.

### Why gain at fixed exposure is fundamentally different from exposure at fixed gain

The coronal subtraction itself is **linear**: a Gaussian blur and a subtraction. The two blocks
collected the *same photons* — same exposure, same sky, same corona — and differ only by a
×4.217 in ADU per electron. A linear operation on a scaled image gives a scaled result, so for
the subtraction alone changing the gain is not merely equivalent to changing the exposure, it
is *cleaner*: no change in photon statistics, trailing, or time-averaging.

**What is not linear is saturation, and gain changes its physics where exposure does not.**

* At **gain 0** the ADC's 65 535 ADU corresponds to roughly the sensor's full well, so
  saturation is **sensor** saturation — soft, with a nonlinear approach and charge blooming
  around the core.
* At **gain 125** the ADC clips at 65 535 / 4.217 ≈ 15.5 k e⁻-equivalent, far below full well,
  so saturation is **ADC clipping** — hard, with the sensor itself still linear right up to
  the edge.
* Changing the exposure at fixed gain keeps the **same** mechanism at every tier; only the
  intensity at which it engages moves.

The footprint is measured, not inferred. Stage 1 finds the saturated coronal core at
**592 px at gain 0 and 706 px at gain 125** — the occulter is 114 px larger at the high gain
(1.38 against 1.64 R☉), the 95 % threshold is catching a soft-bloomed edge in one block and
a hard clip in the other, and the Gaussian near the mask edge is estimated over different
regions. The 2000 ADU pedestal is 2000 e⁻ at gain 0 and 474 e⁻ at gain 125. Everything
gain-dependent in the chain is **Sun-centred and radially concentrated**, which is exactly the
shape the difference has. León's tiers had different saturation radii too (the 1.2 s tier's
811 px disk), but in **one** physical regime, and the union machinery managed that.

### ⚠ The mechanism above is withdrawn; the measurement stands

Douglas approved the discriminating experiment (grow gain 0's occulter to 716 px) and reading
the code before running it showed it **cannot discriminate anything**, and that the
saturation story cannot carry the effect to where it is measured:

* in `disk` mode the occulter **modifies no pixel** — it is a detection gate — and
  `blob_radius_extra` is `blob` mode's parameter, unused here; a gate cannot move a star
  outside it, and every two-witness star is at ≥ 2.35 R☉ (1017 px), beyond both gates;
* the masked coronal blur's edge effect dies ~3σ = 30 px past the saturated core — at most
  ~740 px, still inside 1017;
* the stack is accumulated in **float64** and centroided there, so the 2000 ADU pedestal never
  clips anything the centroids see (the unsigned-integer clipping is only the file on disk);
* the raw frames say where saturation actually is: at gain 125, **27 % of pixels at 1.3–1.7 R☉
  and 0.0 % beyond 1.7**; at gain 0, 0.1 % at 1.3–1.7 and nothing beyond. Both blocks are
  fully linear at every two-witness star;
* detection footprints are **identical** in the two blocks at every radius (median 3.5–6 px
  in both), so the threshold-on-a-gradient idea is out too.

So *nothing gain-dependent in the pipeline reaches 2.35–12 R☉*, and the sentence "everything
gain-dependent in the chain is Sun-centred and radially concentrated, which is exactly the
shape the difference has" was true of the pipeline and irrelevant to the stars. **What
survives is the data**: the Sun-centred structure appears across the gain boundary and not
within a gain. What carries it is open. The one thing left that is Sun-centred, 1/r-shaped,
and can change between two sequential captures is the **seeing** — a wider PSF on a steep
gradient pulls a centroid Sunward by an amount that grows as the width squared — and a seeing
*step* coincident with the boundary would defeat a test built on ~20 s pairs.
`tools/husillos2026/hu_seeing.py` measured the star width per frame across both captures on
the same stars, and **there is no step**:

| | FWHM (half-maximum footprint) |
|---|---|
| gain 125, whole block | 2.106 px, IQR 0.30 |
| gain 0, whole block | 2.106 px, IQR 0.30 |
| last 20 frames of gain 125 → first 20 of gain 0 | 2.106 ± 0.083 → 1.954 ± 0.077 px, **1.3 σ** |

And the 1.3 σ hint runs the *wrong* way: a wider PSF pulls a centroid Sunward, but gain 125's
inner stars sit *further* from the Sun. A back-of-envelope kills it independently — at 2.35 R☉
the coronal gradient is ~1.5 ADU/px at gain 0 and a windowed centroid's pull is ~gσ²/S ≈
0.01–0.06 px, thirty times short of the 0.4 px measured. **Seeing is not the carrier.**

### ⚠ And the half-block table was over-read

Inner minus outer, from §3q's own table: same gain 125, **+57 ± 37** (nothing); cross the
boundary, **−119 ± 36**; same gain 0, **−50 ± 45** — *the same sign as the boundary, at 1.1 σ*.
"It follows the gain" was read from the first two rows. All three together are equally
consistent with something **absent in the first 20 s, starting around 18:29:30–50 and growing
through the gain-0 block** — an onset in time that happens to straddle the boundary. With
27–34 shared stars per pair the within-gain-0 term cannot decide it either way.

What can: **per-frame positions**. A windowed, background-subtracted centroid is scale-invariant
and the two blocks collected the same photons, so if the inner-minus-outer displacement
*steps* at frame 171 → 2 the step is in the electronics, and if it *ramps* through the boundary
without noticing it the carrier is time — atmosphere, the Moon crossing the corona (~20 ″ in
39 s), the sky brightening toward C3. `tools/husillos2026/hu_radial_vs_time.py`.

### The answer: it is neither a step nor a ramp — it moves on a 10–30 s timescale

Fifteen-frame sub-stacks (a single 315 ms frame reaches only the brightest handful of stars;
15 summed reach G ≈ 10 and 19–25 two-witness stars per bin), windowed centroids, a similarity
removed about the Sun, then the radial residual of the inner stars (2.3–5 R☉, 11 of them) minus
the outer (> 7 R☉, 32), relative to the gain-0 stack:

| block | frames | mid UTC | inner − outer (px) |
|---|---|---|---|
| gain 125 | 46–60 | 18:29:02 | +0.17 ± 0.22 |
| gain 125 | 61–75 | 18:29:07 | +0.32 ± 0.16 |
| gain 125 | 76–90 | 18:29:12 | +0.11 ± 0.14 |
| gain 125 | 91–105 | 18:29:17 | +0.34 ± 0.15 |
| gain 125 | 106–120 | 18:29:21 | +0.27 ± 0.28 |
| gain 125 | 121–135 | 18:29:26 | **+0.53 ± 0.20** |
| gain 125 | 136–150 | 18:29:31 | **+0.63 ± 0.19** |
| gain 125 | 151–165 | 18:29:35 | +0.42 ± 0.19 |
| gain 0 | 2–16 | 18:29:46 | −0.10 ± 0.15 |
| gain 0 | 17–31 | 18:29:50 | +0.07 ± 0.17 |
| gain 0 | 32–46 | 18:29:55 | +0.11 ± 0.09 |
| gain 0 | 47–61 | 18:30:00 | +0.11 ± 0.05 |
| gain 0 | 62–76 | 18:30:04 | −0.03 ± 0.09 |
| gain 0 | 77–91 | 18:30:09 | −0.16 ± 0.18 |
| gain 0 | 92–102 | 18:30:13 | **−0.52 ± 0.15** (11 frames, 18 stars) |

Whole blocks, inverse-variance: gain 125 **+0.333 ± 0.063 px**, gain 0 **+0.037 ± 0.034 px**,
difference **+0.296 ± 0.071 px (4.2 σ)** — the block disagreement, confirmed on scale-invariant
centroids of identical photons. But look at the shape:

* **within gain 125 it rises**, +0.38 px across the block (first four bins mean +0.24, last
  four +0.46);
* **at the boundary it drops** — +0.42 → −0.10, about 2.2 σ;
* **within gain 0 it is flat for 25 s, then falls** −0.5 px in the last 3.5 s before the
  capture ends (a thin bin: 11 frames, 4 inner stars).

The whole-block difference is real (4.2 σ) and the within-block bins move by as much as the
blocks differ. *The paragraphs that first stood here read that as "the atmosphere sampled at
two moments, on a 10–30 s timescale". That reading is withdrawn below — the sub-stacks share
the fault that undoes the whole thread.*

## 3r. The Sun-centred thread was differential refraction, projected — withdrawn in full

Two checks on the per-star residuals (`tools/husillos2026/hu_field_shape.py`,
`hu_block_diff.py`) end it.

**Per bin, nothing reproduces.** Fitting each 15-frame sub-stack's residual field with a
Sun-centred 1/r term and, separately, a generic quadratic, then scoring on held-out stars: the
1/r term scores **negative in 14 of 15 bins** (−0.4 to −4.4 %), the quadratic overfits (down to
−93 %). The per-bin fields are noise. But the quadratic's **linear** coefficients are a clean
step — stable at (−0.3, +0.45) through every gain-125 bin, ~0 through every gain-0 bin. A
stable **anisotropic scale and shear** between the blocks is what **differential refraction
changing between two epochs** produces at z = 81.4°, and an isotropic similarity cannot absorb
it.

**Then the two blocks compared properly**, 64 shared stars, later minus earlier:

| comparison | model | scale | anisotropy | shear | 1/r term | held-out |
|---|---|---|---|---|---|---|
| **raw pixels** (what §3q used) | similarity | −101 ± 16 ppm | — | — | — | — |
| raw pixels | **full affine** | x −42, y **−302** | **+260 ppm** | **−82, −76 ppm** | −0.40 ± 0.31 px (1.3 σ) | **−0.5 %** |
| **refraction-corrected** (what stage 3 fits) | similarity | **−6 ± 10 ppm** | — | — | — | — |
| refraction-corrected | full affine | x −12, y +4 | −16 ppm | 0, +8 ppm | −1.41 ± 0.68 ″ (2.1 σ) | **−0.3 %** |

Read across. The raw-pixel difference is **a 260 ppm vertical compression and an 80 ppm
shear** — the field set 0.11° in the 39 s between the block mid-times, and at 8.6° altitude
the differential refraction across a ±2.9° field changes by a few hundred ppm; the shear is the
sensor's 14.9° tilt from the vertical (§3h). Only the *isotropic* half of that was being
removed. What remained, projected radially about a Sun near the field centre through **11
inner stars lopsided in azimuth**, is a pattern that looks Sun-centred and monotonic in radius.
Once the affine removes it, no 1/r term reproduces on held-out stars. And on the
refraction-corrected displacements — the ones stage 3 uses — the blocks agree to **6 ppm in
scale and ~10 ppm in every affine term**, with a residual rms of 0.53 ″ that is exactly §3f's
per-star noise.

**So, withdrawn:** the −227/−135/−60 ppm "Sun-centred profile" (§3q); "the structure follows
the gain, 3.3 σ across the boundary" (§3q); the saturation mechanism (already withdrawn);
and "it is the atmosphere on a 10–30 s timescale" (above). Three analyses in a row were built
on one wrong nuisance model — a similarity on raw pixels at 8.6° — and each inherited the
artefact. The sub-stack table's within-block trends are not established either: its per-bin
fields fail held-out, its linear terms are the refraction ramp, and its last bin is thin.

**What stands** is §3f as first written: the two blocks differ by **per-star noise** — 0.53 ″
rms per star, correlation 0.484 — with no smooth structure between them, and their 1.19 ″ L
difference is that noise through the L–scale degeneracy (0.63 ″ with the scale held, 1.19 ″
free). That noise *is* atmospheric — two independent realisations of the wavefield 39 s apart
— which is where the León lesson lands, more modestly than claimed: **interleaved tiers share
one realisation of the atmosphere and sequential blocks do not**, so interleaving raises the
cross-tier correlation and tightens the union; it does not remove a Sun-centred systematic,
because there is none. The gain is a bystander.

**Method 1's gain-125 value is still bad**, and the reason is now narrower: on the shared
stars the blocks agree to 6 ppm, so the 70 ppm between their *stage-2* scales (2.2029009
against 2.2027459) comes from the non-shared stars and the free low-order fit, not from the
data the union uses; `constant` then converts whatever the import disagrees with into L.

*(2026-09-12, §3t: "the free low-order fit" is the **linear** terms, not the quadratic. The
rung ladder run against CalibS shows the 70 ppm survives freezing the quadratic — 72.8 ppm at
`linear` against 70.9 ppm at `quadratic` — and collapses to 20.9 ppm only when the linear
terms are frozen too. That is the same object §3r's affine found: an anisotropic linear map
between the block epochs, not a curvature difference.)*

**A trap for the record.** *At 8.6° altitude, any comparison of two epochs on raw pixel
positions must remove a full affine or correct refraction first.* A similarity leaves a
few-hundred-ppm anisotropy in, and eleven lopsided stars will make it look like anything.
This cost three sections and most of a day.

### What it does to the numbers

The two-witness union averages a gain-125 block that carries a Sun-centred systematic with a
gain-0 block that carries a different (smaller) one. The **−3.37 ± 0.92 ″** 1/r term between
the full blocks is that systematic, and 1.19 ″ of it is the difference in their fitted L.
Under **Method 1** only the constant is free, so the whole of it has nowhere to go but L —
which is why gain 125 reads 4.6 ″ there while gain 0 reads a sensible 2.1 ″.

*(Correction, Douglas 2026-09-11. An earlier draft said "gain 0 shares CalibS' saturation
regime". That is wrong: **CalibS does not saturate at all** — its brightest pixel is 10 417
of 65 535 — so it is in no regime to share. The plate scale is a property of the optics, and
CalibS measures it without a corona at either gain. The coronal blocks' fitted scales are that
optical scale plus a Sun-centred contamination from the saturated corona; gain changes the
contamination, not the optics. Gain 0 sits closer to CalibS simply because its contamination
is smaller. A CalibS shot at gain 125 would measure the same clean scale and would not bring
the gain-125 coronal block any closer.)*

**Still open, and separate from the gain question:** the gain-0 halves show a real *uniform*
scale step of −79.7 ± 21.0 ppm over 15.7 s (3.8 σ) that the gain-125 halves do not, with no
radial structure. That is not the corona and is not explained here.

~~**The discriminating experiment**: re-stack gain 0 with its occulter grown to 716 px~~ —
**void**, see above: the occulter is a gate that touches no pixel and no two-witness star is
inside it, so growing it cannot move the stars the effect is measured on. Replaced by the
per-frame seeing test.

### What this is and is not

* The two blocks' separate values (1.596 and 2.782 ″) **are superseded by the union** and should
  not be quoted alone; their 2.2 σ spread is per-star noise, which the union averages.
* **The weather is assumed** — 926.5 hPa is the standard atmosphere at 743 m, with 25 °C and 35 %
  as ordinary August values. At z = 81.4° the plate scale carries ~63 ppm per 1 % of the
  refraction constant, so this is the single largest unquantified term.
* **One zenith field, not seventeen** — no reference field-to-field term at all.
* ~~**No atmospheric data** — Joe took none~~ — **wrong** (§3i). Ten horizon captures at the
  eclipse altitude, on the eclipse night, at both eclipse gains. ~~Not yet reduced, so the term
  is still absent from the budget~~ — **reduced 2026-09-12 (§3s): ±0.25 ″, provisional, with no
  structure resolved above photon noise.**
* **No darks and no flats.**
* **The outer radial bound is inherited**, and is not applied.

So cell 4 stands at **L = 2.129 ± 0.430 (stat) ″**, the two-gain union under the two-witness
rule, GR at 0.88 σ — a real fit with an incomplete budget, not yet a matrix entry. What would
close it: **CalibS**, for a same-day same-altitude *imported* scale — the night zenith's is
1365 ppm from the eclipse field's and cannot be imported at all, and at 0.0324 ″/ppm this field
punishes an imported scale error harder than any other in the matrix; the **real weather**; and
more zenith fields for a reference term.

---

## 3s. The horizon fields reduced — three pointings, a first atmosphere term, and a pathway that under-reads the deflection by 14 %

Douglas, 2026-09-12: *"Reduce the horizon fields so we can get an atmosphere term."* Three
things came out, in the order they were found. Tools: `tools/husillos2026/hu_horizon_reduce.py`
(modes `h10`, `deep`, `deep10`), `hu_atmosphere.py` (the nulls), `hu_absorption.py` and
`hu_inject.py` (the third finding).

### The `10 deg` window is three pointings, not one field

§3i's table and the `h10` comment in `hu_horizon_reduce.py` described the window as one
tracked field sweeping 10.4° → 8.5° — one plate solve, propagated over seven captures. Solving
each capture on its own (zenith star-field preset, refraction on, each capture's own
`MidCapture` time):

| capture (local name) | UTC | gain | frames | stars | rms | plate scale | RA / Dec | alt / az | pointing |
|---|---|---|---|---|---|---|---|---|---|
| `23_31_59` | 21:31:59 | 0 | 100 | 71 | 1.189 ″ | 2.2058838 | 175.72 / +29.60 | 10.03° / 300.9° | **A** |
| `23_34_38` | 21:34:38 | 125 | 100 | 135 | 0.737 ″ | 2.2059863 | 176.38 / +29.60 | 10.01° / 300.9° | **A**, re-pointed +0.66° in RA to hold 10° |
| `23_37_17` | 21:37:17 | 125 | 100 | — | — | — | 35 centroids, 19 matched | — | no solve |
| `23_40_27` | 21:40:27 | 0 | 14 | — | — | — | not stacked | — | — |
| `23_41_01` | 21:41:02 | 0 | 51 | 22 | 0.703 ″ | 2.2089558 (**+1350 ppm**) | 177.87 / +23.28 | 5.73° / 296.2° | **B** |
| `23_42_43` | 21:42:43 | 125 | 50 | 74 (frames 1–47) | 1.575 ″ | 2.2102915 (**+2000 ppm**) | — | 5.45° | **B**; frame 48 will not align to frame 0 — the slew to C is inside this capture |
| `23_44_06` | 21:44:06 | 125 | 50 | **685** | **0.326 ″** | 2.2058177 | 178.01 / +37.54 | 14.97° / 307.4° | **C** |

Two captures at 10.0° that are one field (0.57° apart on the sky); one pointing at 5.7° whose
two solves are 1350–2000 ppm off in scale — at 5.5° the assumed weather's refraction error is
that large, and they are excluded; and one capture at 15.0° that is the best field of the night.
The "~+2°, León's H2 analogue" row in §3i is struck. The only consecutive same-field pair in the
window is `23_31_59` → `23_34_38`, 2 min 39 s apart at 10.0°. (The wrong description lived in
the tool's comment and in conversation; the record's §3n settling analysis did not use it.)

### Two null constructions, because the cell has two pathways

The matrix's construction (`docs/STEP3_CHARTS_AND_SETTINGS.md` §2) is: refit each night field
**constant-only against the previous field of the same night**, impose the eclipse Sun's frame
position, apply the science cuts, fit L; true L is zero, the rms over fields is the term. Its
rule — *match the null's construction to the science design, or it charges the wrong thing* —
means cell 4 needs two:

* **field-to-zenith**, the analogue of Method 2 of record: each horizon field fitted at the
  eclipse blocks' own rung (quadratic free, cubic and above frozen from the night zenith, scale
  free), the Sun imposed, L fitted. This is what the pathway of record manufactures on a field
  with no deflection — model transfer from 85° to 9°, refraction with the assumed weather, and
  the atmosphere — and it works on fields at different pointings;
* **consecutive pairs**, the matrix's standard and the analogue of Method 1: constant-only
  against the neighbouring capture, both ways round, **same field only** — the tool checks the
  solved centres and discards pairs more than 2° apart. A host with fewer than 40 stars is
  not trusted.

Cuts are the registered window (G ≤ 13, R > 2 R☉, no outer crop); the Sun is at (5043, 3386)
px; "L scale" is Method 2's freedoms with no nuisance (the cell's estimator), León's base and
v-deg2 beside it; the floor is a bootstrap on the per-star residuals; the 63-star column is a
subsample at the union's star count.

### Results — zenith star-field preset (`s2_`)

Field-to-zenith:

| field | alt | gain | N | rms | L base | **L scale** | L v-deg2 | floor | 63-star | |
|---|---|---|---|---|---|---|---|---|---|---|
| `23_31_59` | 10.03° | 0 | 66 | 0.843 ″ | +0.037 | **+0.112** | +0.120 | 0.758 | 0.211 | |
| `23_34_38` | 10.01° | 125 | 122 | 0.528 ″ | −0.161 | **−0.321** | −0.371 | 0.370 | 0.460 | |
| `23_42_43` | 5.45° | 125 | 72 | 1.133 ″ | +0.349 | +0.795 | +0.603 | 0.741 | 0.896 | outside 7.5–12°, not averaged |
| `23_44_06` | 14.97° | 125 | **653** | **0.231 ″** | +0.074 | **+0.137** | +0.091 | **0.071** | 0.247 | outside 7.5–12°, not averaged |

The two fields inside the band: **rms 0.241 ″** on the cell's estimator (base 0.117, v-deg2
0.276), against a **0.597 ″ floor** from their own per-star noise — the null is *below* the
noise of these shallow fields (1 s at gain 0 through 5.6 air masses gives 71 stars at 1.19 ″).
The one field deep enough to resolve structure above its noise is the 15° one: **+0.137 ″
against a 0.071 ″ floor**, 1.9 σ, with 653 stars — a first look at what the pathway
manufactures on a well-measured low field.

Consecutive pairs (constant-only, same field):

| field | host | way | N | rms | L base | L scale | L v-deg2 | floor |
|---|---|---|---|---|---|---|---|---|
| `23_34_38` | `23_31_59` | forward | 69 | 1.122 ″ | −0.245 | +2.064 | −3.567 | 1.116 |
| `23_31_59` | `23_34_38` | reversed | 48 | 1.381 ″ | +0.507 | −1.254 | +1.761 | 1.247 |

Noise, both ways: the gain-0 field can neither host nor be hosted at 71 stars, and the
constant-only refit inherits the host's model noise on top of its own. The pairs at 5.7° were
discarded by the tool (host 22 stars; different fields).

### Deep detection (`s2d_`) — and the eclipse altitude itself, at last

Re-stacked with the eclipse blocks' own detection settings (`hu_horizon_reduce.py deep`,
`deep10`), the twilight window **solves**, and the dark-sky fields roughly triple their star
counts. The sentence that first stood here — *"the `cal 8 deg` window has not yet produced a
solve at all"* — is superseded.

`cal 8 deg` turns out to be the best null geometry in the dataset: **one tracked field, two
captures 2 min 33 s apart, straddling the eclipse altitude**, which is also the matrix's own
cadence (León's zenith pairs are 2 min 34 s apart).

| capture | UTC mid | gain | stars | rms | RA / Dec | alt / az |
|---|---|---|---|---|---|---|
| `22_56_41` | 20:57:46.5 | 0 | 37 | 1.056 ″ | 185.9908 / +7.8545 | 9.01° / 272.30° |
| `22_59_14` | 21:00:19.6 | 125 | 53 | 1.099 ″ | 185.9918 / +7.8534 | 8.54° / 272.73° |
| `22_53_15` | — | 0 | — | — | 260 centroids, no solve | — |

Field-to-zenith, deep, every capture that solves:

| field | alt | gain | N | rms | L base | **L scale** | L v-deg2 | floor | 63-star | |
|---|---|---|---|---|---|---|---|---|---|---|
| `22_56_41` | 9.01° | 0 | 34 | 0.742 ″ | −0.531 | −1.540 | −0.728 | 1.242 | 1.540 | 37 stars: excluded |
| `22_59_14` | 8.54° | 125 | 47 | 0.790 ″ | +0.390 | +1.881 | +0.528 | 1.220 | 1.881 | |
| `23_31_59` | 10.03° | 0 | **174** | 0.884 ″ | +0.134 | **+0.235** | +0.200 | 0.457 | 0.645 | |
| `23_34_38` | 10.01° | 125 | **314** | 0.776 ″ | −0.140 | **−0.261** | −0.237 | 0.291 | 0.438 | |
| `23_37_17` | 5.70° | 125 | 69 | 0.993 ″ | +0.031 | +0.500 | +0.047 | 1.170 | 0.607 | pointing B |
| `23_41_01` | 5.73° | 0 | 61 | 0.721 ″ | −0.566 | −1.124 | −0.807 | 0.682 | 1.124 | pointing B |
| `23_42_43` | 5.45° | 125 | 128 | 0.780 ″ | −0.427 | −0.586 | −0.483 | 0.424 | 0.738 | pointing B |
| `23_44_06` | 14.97° | 125 | **935** | **0.321 ″** | +0.074 | **+0.142** | +0.092 | **0.078** | 0.335 | pointing C |

Consecutive pairs, deep: `22_56_41` against `22_59_14`'s model (35 stars) gives −1.803 ″
against a 1.242 ″ floor; the 10° pair gives **+0.058 ″** forward (145 stars, floor 0.649) and
+0.215 ″ reversed (84 stars). Pointing B's four refits run from −2.235 to +1.069 ″ with floors
of the same size, and are excluded on altitude.

**Two reproducibility faults were fixed while building this table**, both of which had already
produced numbers. The bootstrap floor drew from one random stream shared by every field, so
adding a capture to the run changed the floors already reported for the others — the same
145-star pair read 0.641 ″ and 0.773 ″ on identical input. Each field now seeds from its own
tag through `zlib.crc32` (Python salts `hash()` per process, so that would not have fixed it).
And the floor used 60 draws, whose ~9 % standard error on a standard deviation could not
support the 0.407-against-0.522 swing it was showing; it is now 300. **The floor decides
whether a null is structure or photon noise, so it has to be quieter than the thing it
judges.** The L values themselves never moved — they are a deterministic least squares.

### The complete pointing map, and where the refraction model gives way

With deep detection every capture in both windows solves except `22_53_15`. The `10 deg`
folder is **three pointings, and the middle one holds three captures, not one** — the earlier
table above could not place `23_37_17` because it did not solve at the zenith preset.

| pointing | declination | altitude | captures |
|---|---|---|---|
| **A** | +29.60° | 10.0° | `23_31_59`, `23_34_38` (re-pointed 0.66° in RA to hold 10°) |
| **B** | +23.28° | 5.5–5.7° | `23_37_17`, `23_41_01`, `23_42_43` (again re-pointed ~0.88° in RA) |
| **C** | +37.55° | 15.0° | `23_44_06` |

Every deep solve, plate scale against the night zenith's 2.2059136 ″/px:

| capture | alt | gain | stars | rms | **scale vs the zenith** |
|---|---|---|---|---|---|
| `23_44_06` | 14.97° | 125 | **972** | **0.451 ″** | −43 ppm |
| `23_34_38` | 10.01° | 125 | 345 | 1.069 ″ | +52 ppm |
| `23_31_59` | 10.03° | 0 | 184 | 1.256 ″ | −72 ppm |
| `22_56_41` | 9.01° | 0 | 37 | 1.056 ″ | −158 ppm |
| `22_59_14` | 8.54° | 125 | 53 | 1.099 ″ | −44 ppm |
| `23_37_17` | 5.70° | 125 | 77 | 1.392 ″ | **+1804 ppm** |
| `23_41_01` | 5.73° | 0 | 63 | 1.014 ″ | **+1222 ppm** |
| `23_42_43` | 5.45° | 125 | 132 | 1.095 ″ | **+1565 ppm** |

Everything from 15° down to 8.5° sits within ±160 ppm of the zenith scale. The three captures
at 5.5° are more than a thousand ppm out, **all in the same direction, at both gains and at
both detection settings** — so it is not a detection artefact and not a bad solve in the
ordinary sense. Something breaks between 8.5° and 5.7°, and there are two candidates, not one:
the **refraction correction**, whose differential across a ±2.9° field grows steeply toward the
horizon and whose weather here is assumed rather than measured (926.5 hPa, 25 °C, 35 %; the
scale carries ~63 ppm per 1 % of the refraction constant at the eclipse altitude); and
**model transfer**, since the frozen cubic-and-above come from a zenith field at 85° and a
field compressed this hard may simply not be described by them. This window cannot separate
the two.

**The eclipse blocks sit at 8.6°, inside the range that still holds but not far inside.** This
is a bound on the assumed weather rather than a measurement of it, and it is the first
empirical evidence in this cell that the assumption survives at the eclipse altitude at all.
Pointing B is excluded from every average below on its altitude, which was decided before
these scales were known.

### The term: nothing is resolved above per-star noise

`hu_atmosphere.py` now reports the decomposition the matrix reports beside its totals —
`structure = √(total² − floor²)`:

| construction | fields | total | floor | **structure** | at 63 stars |
|---|---|---|---|---|---|
| field-to-zenith, deep, 8.5–10° | 3 | 1.105 ″ | 0.771 ″ | 0.792 ″ | 1.176 ″ |
| **field-to-zenith, deep, the two with >126 stars** | 2 | **0.248 ″** | 0.383 ″ | **0.000 ″** | 0.551 ″ |
| field-to-zenith, zenith preset, 10° | 2 | 0.241 ″ | 0.622 ″ | 0.000 ″ | 0.381 ″ |
| consecutive pairs, deep, 8.5–10° | 3 | 1.049 ″ | 1.038 ″ | 0.147 ″ | 1.124 ″ |
| **consecutive pairs, deep, the one with >126 stars** | 1 | **0.058 ″** | 0.649 ″ | **0.000 ″** | 0.565 ″ |
| consecutive pairs, zenith preset, 10° | 2 | 1.707 ″ | 1.010 ″ | 1.376 ″ | 1.720 ″ |
| **the one field deep enough to resolve anything** (15°, 935 stars) | 1 | 0.142 ″ | 0.078 ″ | **0.119 ″** | 0.335 ″ |

Grouped by altitude instead, which is the axis an atmospheric term ought to follow:

| altitude | fields | stars | total | floor | structure |
|---|---|---|---|---|---|
| below 7.5° | 3 | 258 | 0.787 ″ | 0.819 ″ | 0.000 ″ |
| 7.5–12.0° | 4 | 569 | 1.228 ″ | 0.912 ″ | 0.823 ″ |
| above 12.0° | 1 | 935 | 0.142 ″ | 0.078 ″ | 0.119 ″ |

**There is no clean altitude trend, and the reason is that star count confounds it.** The
7.5–12° row is the highest of the three only because it contains both twilight fields; the
5.5° row is *lower* than it. Altitude and depth are anti-correlated in this dataset — the
fields nearest the eclipse geometry are the ones shot in twilight — so this window cannot
separate the two. The table is kept as the honest negative.

**Read the star-count column, not the altitude column.** Every construction with enough stars
to test the question returns 0.06–0.25 ″ and **no resolved structure**; every construction
that returns a large number does so on a field whose own photon noise is larger still. The
1.105 ″ row is one 47-star twilight field at +1.881 ″ against its own 1.331 ″ floor — 1.4 σ
from zero. The split is by star count and is stated in the tool rather than applied after
seeing the answers, but it is a split made after the fact and should be read as such.

**The term, provisionally: ±0.25 ″** — the total on the fields that can measure it, quoted
without subtracting the floor as the matrix requires. It sits between Station 1's ±0.11 ″ and
León's ±0.33 ″, which is where a 9–10° site belongs. The only resolved structure measurement
anywhere in either window is **0.119 ″ at 15° on 935 stars** (0.142 ″ total against a 0.078 ″
floor, 1.8 σ). The number did not move when the four extra pointing-B and pointing-C fields
landed, which is the best evidence available that ±0.25 ″ is not an artefact of which fields
happened to solve.

*(Superseded the same day by §3w: the two 10° fields in the tables above were stacks of
UNTRACKED captures and their residuals are drift smear. Rebuilt on per-frame medians the term
is **±0.10 ″**. The 15° row and the `cal 8 deg` rows are unaffected.)*

**And a warning against double-counting.** The 63-star column is ~0.5 ″ everywhere, but that
is sampling noise, not atmosphere: at the union's own star count the residual field is sparsely
sampled, and that is the same quantity as the union's ±0.430 ″ statistical error. Adding it as
a systematic would charge it twice. Unlike every other cell in the matrix, cell 4's null floor
**exceeds** its null total, because a 1 s exposure through six air masses is photon-starved.

Every horizon capture in both windows has now been reduced at both detection settings, except
`22_53_15`, which does not solve at either. Nothing in the window is left to run.

### The pathway of record under-reads a 1/r deflection by 14 %

Building the field-to-zenith null meant asking what the pathway does to a 1/r pattern, and
the answer is not "nothing". Method 2 of record fits the eclipse field with **the quadratic
free** — twelve polynomial terms: translation, rotation, scale, two shears, six quadratics —
and stage 3 then refits only translation, rotation and scale (mode 2's *r* column) jointly
with L. Whatever part of the deflection the two shears and six quadratics absorbed in stage 2
is gone before stage 3 sees it. On a Sun at the field centre with a symmetric star set that
part is zero (1/r is odd about the Sun, a quadratic is even); on the real star set it is not.

`hu_absorption.py` computes it from the star geometry alone — a unit-L deflection about the
eclipse Sun, the stage-2 freedoms projected out over every star the block matched, stage 3's
[N1, N2, Θ, S, L] refitted on the science-cut stars:

| star set | stage-2 fit on | stage 3 on | `constant` + free scale | `linear` | **`quadratic` (record)** |
|---|---|---|---|---|---|
| gain-125 block | 84 | 84 | 1.000 | 0.941 | **0.897** |
| gain-0 block | 74 | 74 | 1.000 | 0.929 | **0.841** |
| **two-witness union** | 84 and 74 | 63 | 1.000 | — | **0.863** |

`hu_inject.py` then measured it on the pipeline itself: every centroid of the gain-125 block
pushed radially away from the Sun by exactly **2.000 ″ of L** (0.07–0.42 px), stage 2 and
stage 3 run unchanged:

| rung | original L (Method 2) | injected L | **rise** | predicted |
|---|---|---|---|---|
| `quadratic`, free scale (record) | 2.215 ± 0.433 ″ | 4.002 ± 0.444 ″ | **+1.787 ″** | 2.000 × 0.897 = 1.794 |
| `constant`, free scale (control) | 2.039 ± 0.465 ″ | 4.031 ± 0.463 ″ | **+1.992 ″** | 2.000 × 1.000 = 2.000 |

The geometry and the pipeline agree to 0.3 %. (The test's Method 1 columns are not
meaningful: with the stage-2 scale free, the injection moves the base scale the import fixes.)

**What it means.** L = 2.129 ± 0.430 ″ is **0.863 × the sky's deflection**; undone, **2.467 ±
0.498 ″**, GR at 1.44 σ. Per block: 2.782/0.897 = 3.10 ″ and 1.596/0.841 = 1.90 ″. No other
cell has this: León's eclipse field is at `constant` against CAL_piLeo, Station 1's at
`constant` + free scale against the zenith, Bruns' at constant against L/R8 — all f = 1, as the
control column says. Cell 4 is the only cell that fits its eclipse field with the quadratic
free, and §3e chose that because `constant` against the zenith cost 0.18 ″ of *residual* (the
day–night low orders had moved). The choice was right for the residual and wrong for the
signal, and nobody asked what it did to the signal. The `constant` rung has its own problem —
its original-centroid L (2.039 ″) is *lower* than the quadratic rung's 2.215 ″, so the frozen
day–night low orders push the other way, and the injection cannot arbitrate between rungs,
only certify each one's response.

The same absorption acts on the field-to-zenith null, which is why the two are reported
together: the null charges the pathway what it does to a field with no deflection; f says what
it does to the deflection. If the record's L is divided by f, so is the null (×1.16).

**A decision for Douglas, not made here.** Three ways to carry it:

* (a) divide Method 2 of record by f = 0.863 — deterministic, geometry-only, verified on the
  pipeline: 2.129 → **2.467 ± 0.498 ″**;
* (b) move the eclipse field to **`constant` + free scale against the settled CalibS** — León's
  pathway rung for rung, f = 1, no scale import (the scale stays free), the low orders from a
  same-day 88-star calibration at the same altitude instead of from the night zenith. The
  per-block numbers on that pathway already exist from the `m1s_` runs' Method 2 columns
  (1.676 ± 0.540 and 3.508 ± 0.545 ″) but no union has been built on it;
* (c) leave 2.129 and state f beside it.

Recommendation: **(b) as the pathway, (a) as its check.** The ladder rule says the eclipse
field gets `constant`, and CalibS is exactly the same-day calibration the rule wants; (a) is
what (b) should reproduce within the CalibS quadratic's own noise. Method 1 (3.290 ± 0.681 ″,
f = 1 by construction) and the corrected Method 2 are 0.82 ″ apart instead of 1.16.

## 3t. The whole ladder against CalibS — and four pathways that agree once each is divided by its own f

Douglas, 2026-09-12: *"What is the value of L in a Method 1 calculation where we import only
the quadratic terms from CalibS but not the linear term?"* `tools/husillos2026/hu_rung.py`
runs all three rungs against the settled CalibS so the ladder is one table.

### That setting cannot be Method 1, and the reason is structural

`distortion_fixed_coefficients` names the highest order left **free**, so "quadratic and above
from CalibS, linear free" is **`linear`**. But **the plate scale is the isotropic part of the
linear term**. `distortion_polynomial.py:298–312` is explicit: only at `order_free == 0` does
the fitter run a linear fit, discard the stretch and skew, and then *replace* the scale with
the reference's `fix_platescale` — and even then only when `distortion_free_scale` is off. At
`linear` or above the linear coefficients are fitted on the eclipse field and the scale comes
with them. Every run below reads back `plate scale source: fitted on this field`, and the tool
asserts on that readback rather than on the arguments passed.

So the ask is **Method 2 with CalibS' quadratic**, not Method 1. Stage 3 still prints a
Method 1 line at that rung, but it holds a scale the same data just set — which is why its
uncertainty is 74.9 % against Method 2's 20.0 %. It is reported and not used.

### Stage 2, all six runs, rung read back from each run's own results

CalibS (settled, 88 stars, rms 0.6647 ″, ps 2.2030306 ″/px ± 19.0 ppm) is the reference for
all of them.

| rung | what is free | block | stars | rms | fitted scale | scale source |
|---|---|---|---|---|---|---|
| `constant` | constant only | gain 0 | 73 | 0.7915 ″ | 2.2030306 | **imported** |
| | | gain 125 | 84 | 1.0867 ″ | 2.2030306 | **imported** |
| `linear` | + linear (scale, rotation, 2 shears) | gain 0 | 73 | 0.7161 ″ | 2.2029526 | fitted |
| | | gain 125 | 84 | 0.8340 ″ | 2.2027922 | fitted |
| `quadratic` | + the six quadratics | gain 0 | 73 | 0.6951 ″ | 2.2029559 | fitted |
| | | gain 125 | 84 | 0.8027 ″ | 2.2027997 | fitted |

### The answer, and the three rungs beside it

Two-witness, 63 stars in the union, uncropped, G ≤ 13:

| rung | gain 0 | gain 125 | **union** | f | **union ÷ f** | GR at |
|---|---|---|---|---|---|---|
| `constant`, Method 1 (scale imported) | 2.110 ± 0.701 | 4.613 ± 0.710 | **3.290 ± 0.681 ″** | 1.000 | 3.290 ± 0.681 | 2.26 σ |
| `constant`, Method 2 (scale refit in stage 3) | 1.676 ± 0.540 | 3.508 ± 0.545 | **2.532 ± 0.453 ″** | 1.000 | 2.532 ± 0.453 | 1.72 σ |
| **`linear`** — *the question asked* | 1.744 ± 0.504 | 2.658 ± 0.574 | **2.244 ± 0.448 ″** | 0.911 | 2.463 ± 0.492 | 1.45 σ |
| `quadratic` vs CalibS | 1.603 ± 0.507 | 2.788 ± 0.541 | **2.135 ± 0.431 ″** | 0.860 | 2.483 ± 0.501 | 1.46 σ |
| `quadratic` vs the zenith — **the record** | 1.596 ± 0.507 | 2.782 ± 0.540 | **2.129 ± 0.430 ″** | 0.860 | 2.476 ± 0.500 | 1.45 σ |

**So: L = 2.244 ± 0.448 ″** for the pathway asked about, against the record's 2.129 ± 0.430 ″.

### The headline is the last column

Four pathways — scale imported, scale refit after a full CalibS freeze, CalibS' quadratic
with the linear free, and the record's own quadratic-free fit against the zenith — **span
2.463 to 2.532 ″, a spread of 0.069 ″**, against error bars of ±0.45–0.50 ″. Before the
division they span 2.129 to 2.532, a spread of 0.403 ″.

That is an independent confirmation of §3s's absorption fraction that uses no injection at
all: f was computed from the star geometry, and dividing by it collapses the rung dependence
of L to a twentieth of its error bar. The `constant` rung needs no correction (nothing it
frees can absorb a 1/r pattern) and lands in the same place. **The rung choice is not a real
degree of freedom in the answer once f is applied** — which is the strongest argument yet that
f is a property of the pathway and not an artefact of how it was measured.

Method 1 at 3.290 ″ is the one row that does not join, and §3k already says why: the imported
scale sits 21 ppm from what the blocks' own residuals want, and the 0.0324 ″/ppm lever turns
that into +0.68 ″.

### A correction to §3r's attribution

§3r said the 70 ppm between the blocks' stage-2 scales came from "the non-shared stars and the
free quadratic". The quadratic half is wrong, and this ladder shows it:

| what stage 2 leaves free | gain 0 − gain 125 |
|---|---|
| constant + linear + quadratic | +70.9 ppm |
| constant + linear (quadratic frozen from CalibS) | **+72.8 ppm** |
| constant only (linear and quadratic frozen), scale refit in stage 3 | **+20.9 ppm** |

Freezing the quadratic changes nothing; freezing the **linear** terms is what brings the
blocks together. That is the same object §3r's full affine already found — a **260 ppm
anisotropic compression and an 80 ppm shear** between the two block epochs at 8.6° altitude —
seen from the other side: it is a linear-map difference, and an isotropic scale fitted on two
different star sets splits it two different ways. Nothing about curvature, and nothing about
gain.

### What this does to §3s's open decision

Option (b) was "move the eclipse field to `constant` + free scale against the settled CalibS".
The ladder now prices every option on one star set, and the case for (b) is weaker than it
looked: its Method 2 row (2.532 ± 0.453 ″) is the *highest* of the four and its stage-2
residual is the *worst* (0.79 and 1.09 ″ against 0.70 and 0.80). The `linear` rung is the
better-behaved middle: it takes the field curvature from the 88-star same-day calibration
instead of fitting it on 73–84 eclipse stars, keeps the scale free so no ±19 ppm import
enters, costs only 0.02 ″ of stage-2 residual against the record's rung, and carries a
**smaller absorption correction** (f = 0.911 against 0.860).

Revised recommendation, still Douglas': **quote the record's rung with f applied — 2.476 ±
0.500 ″ — and carry the `linear` rung as the check at 2.463 ± 0.492 ″.** They differ by
0.013 ″. Whichever is chosen, the f division is the substantive change and the rung is not.

## 3u. The night maps, and what 23_37_17 settles about refraction

Two requests from Douglas on 2026-09-12, and they answer each other.

### `atmosphere_night_maps.png` — cell 4 joins the matrix's night-map series

*"Are we able to create a chart like this one with the Husillos atmosphere and zenith fields?
Let's confine it to only the 10 degree data … and only the data above 10 degrees. So just four
fields."* `tools/husillos2026/hu_atmos_maps.py`, published through `hu_record.publish()`.

The construction is cells 1 and 3's, not a new one: every night field re-fitted **the way a
calibration field is reduced** — cubic and above frozen from the reference, quadratic free —
so what is drawn is what a calibration fit cannot absorb. Cell 4's horizon fields are already
reduced exactly that way, so their stage-2 residuals *are* the map. Positions and arrows both
in sensor axes, `LSCALE = 0.0018` identical to the León and Bruns maps (legitimate because the
two plate scales agree to 0.1 %), crimson 1 ″ reference, green increasing-altitude arrow,
per-star median removed, 3 × MAD clip.

| panel | alt | gain | stars | rms | vertical | horizontal | **V/H** |
|---|---|---|---|---|---|---|---|
| ~~`23_31_59`, stack~~ | 10.03° | 0 | 182 | 1.223 ″ | 0.986 ″ | 0.724 ″ | 1.36 |
| ~~`23_34_38`, stack~~ | 10.01° | 125 | 339 | 1.006 ″ | 0.843 ″ | 0.548 ″ | 1.54 |
| **`23_31_59`, per-frame medians** (rev. 4) | 10.2° | 0 | 42 | 0.540 ″ | 0.509 ″ | 0.182 ″ | **2.80** |
| **`23_34_38`, per-frame medians** (rev. 4) | 10.2° | 125 | 75 | 0.371 ″ | 0.328 ″ | 0.172 ″ | **1.90** |
| `23_44_06` | 14.97° | 125 | 972 | 0.451 ″ | 0.339 ″ | 0.298 ″ | **1.14** |
| zenith | 81.09° | 0 | 2635 | 0.159 ″ | 0.117 ″ | 0.110 ″ | **1.06** |

*(The struck rows are revisions 1–3 of the chart: stacks of captures that §3w found were
untracked. Revision 4 draws those two panels from per-frame medians.)*

**The zenith panel is the control and it passes.** V/H = 1.06 overhead, the same value León's
zenith row gives, which is what proves the vertical excess below is not manufactured by the
decomposition — a mis-set vertical can only drive V/H *toward* 1. Above that floor the ratio
climbs toward the horizon: 1.14 at 15°, 1.36 and 1.54 at 10°. **This is the first time cell 4
has measured its vertical polarisation on fields with no Sun in them**, which is exactly what
§3h said it could not do and §3i said the horizon fields would allow. León gives 2.4 at the
same geometry, so Husillos is polarised in the same direction and less strongly.

Two honest caveats on the chart. The zenith panel is **not** the same construction — cell 4
has one zenith field, nothing can be frozen onto it from elsewhere, and it shows its own
free-quintic residuals, the machinery floor; the chart says so. And the panel rms values are
**totals**: §3s's bootstrap floors say photon noise is a large part of them at 10°, so arrow
length there is not all atmosphere. The 10.0° gain-125 panel also carries a **coherent swirl
that no linear term can absorb** (rotation is free in the fit), which is unexplained and worth
a look.

### 23_37_17 re-solved — the plate-scale blow-up at 5.5° is refraction, not model transfer

*"Can we try to solve this one again? We probably have a very good idea where it was
pointing."* We do, and it already solved: deep detection puts it at **RA 176.990, Dec +23.284,
alt 5.70°** — pointing B, the same field as `23_41_01` and `23_42_43`, not the 10° pointing its
folder position suggests. So the plate solve was never the problem.

`tools/husillos2026/hu_lowfield.py` fits every low field three ways on the same deep stack to
separate §3s's two candidates, which that section could not:

| capture | alt | fit | stars | rms | **scale vs the zenith** |
|---|---|---|---|---|---|
| `23_37_17` | 5.70° | frozen (of record) | 77 | 1.392 ″ | +1804 ppm |
| | | **free quintic, corrections ON** | 63 | **0.553 ″** | +1686 ppm |
| | | free quintic, corrections OFF | 38 | 0.390 ″ | **+9366 ppm** |
| `23_41_01` | 5.73° | frozen | 63 | 1.014 ″ | +1222 ppm |
| | | free, corr ON | 71 | 1.005 ″ | +1364 ppm |
| | | free, corr OFF | 43 | 0.473 ″ | **+9240 ppm** |
| `23_42_43` | 5.45° | frozen | 132 | 1.095 ″ | +1565 ppm |
| | | free, corr ON | 129 | 0.833 ″ | +1424 ppm |
| | | free, corr OFF | 99 | 0.808 ″ | **+9728 ppm** |
| `23_31_59` | 10.03° | frozen | 184 | 1.256 ″ | −72 ppm |
| | | free, corr ON | 187 | 0.928 ″ | −145 ppm |
| | | free, corr OFF | 154 | 0.896 ″ | +3378 ppm |
| `23_44_06` | 14.97° | frozen | 972 | 0.451 ″ | −43 ppm |
| | | free, corr ON | 971 | 0.431 ″ | −96 ppm |
| | | free, corr OFF | 969 | 0.431 ″ | +1698 ppm |

**Two separate effects, and the test separates them cleanly.**

*The scale is refraction.* Freeing the whole quintic barely moves it at any altitude (+1804 →
+1686, −72 → −145), so the frozen zenith model is not what puts it there. Turning the
correction off shows the size of the thing being corrected, and how well:

| alt | raw compression | left after correction | **correction accurate to** |
|---|---|---|---|
| 14.97° | +1698 ppm | −96 ppm | 5.7 % |
| 10.03° | +3378 ppm | −145 ppm | 4.3 % |
| 5.70° | +9366 ppm | +1686 ppm | **18.0 %** |
| 5.73° | +9240 ppm | +1364 ppm | **14.8 %** |
| 5.45° | +9728 ppm | +1424 ppm | **14.6 %** |

At 10° and 15° the correction removes slightly *more* than the whole compression and lands
within ~150 ppm. At 5.5° it under-removes by 15–18 %. That is the signature of a refraction
model leaving its valid range, which standard formulations do below roughly 5–10° altitude,
not of anything wrong with the captures.

**And the eclipse altitude is inside the working range.** This does not have to be
extrapolated: the `cal 8 deg` captures at **8.54° and 9.01°** — the eclipse geometry itself —
solve at **−44 and −158 ppm** from the zenith, the same small size as 10° and 15°, not the
thousand-ppm size of 5.5°. **The breakdown happens between 8.5° and 5.7°, below anything the
eclipse used.** It is the first empirical evidence in this cell that the assumed weather
(926.5 hPa, 25 °C, 35 %) survives where the science was taken, and it bounds §3r's "single
largest unquantified term" at roughly 150 ppm of scale rather than leaving it open.

*The shape is model transfer, and it is a different quantity.* Freeing the quintic improves
the residual everywhere and most where the transfer is longest: 1.392 → 0.553 ″ at 5.70°,
1.256 → 0.928 ″ at 10.03°, 0.451 → 0.431 ″ at 14.97°. So the zenith's frozen cubic-and-above
**is** wrong at low altitude, in shape rather than in scale. That is a caveat on the night maps
above, which freeze it by construction: a quarter of the 10° panels' residual is model
transfer, not atmosphere. It is also the correct behaviour for that chart — "what a calibration
fit cannot absorb" is defined to include model error — but the split should be stated, and now
it is.

## 3v. Husillos' night sky is rougher than León's — by 2× at the zenith and 3× at 10°, and it is structure, not noise

*(Superseded in part by §3w, same day: the 10° "structure" is the stacking of UNTRACKED
captures, not the sky. The magnitude test below is right that it is structure; the reading
that it is atmosphere is withdrawn. The zenith comparison stands. Kept as written because the
reasoning is what a reader should check against, and the correction is stated where it was
found.)*

Douglas, 2026-09-12, on the night maps: *"Although the vertical polarisation of the León site
may be greater, the absolute value of the atmospheric disturbance at Husillos looks
considerably larger, even at the zenith, but particularly near the horizon."*

It does, and before agreeing that the difference is the *atmosphere*, two things that are not
atmosphere had to come out, because both are larger at Husillos: per-star centroid noise (the
1 s gain-0 frames are read-noise limited by 11×; León's 4 s gain-101 frames are sky-limited)
and model transfer (one zenith field's quintic frozen onto 10°, against León's six-field cubic
average). `tools/husillos2026/hu_maps_bymag.py` applies the matrix's own discriminator
(`floor_vs_sampling.py`, §2 of `STEP3_CHARTS_AND_SETTINGS.md`): **bin the residuals in
magnitude.** Noise is magnitude-dependent, structure is not, so the bright-end asymptote is
the structure. The free-quintic fits from §3u sit beside the frozen ones, so the transfer's
share is visible too. León's horizon rows are rebuilt the way its map was drawn (per-star
medians over ~45 per-frame quadratic-free fits, corrections on) — the like-for-like row for
Husillos' frozen 10° fields.

| set | 4–8 | 8–9 | 9–10 | 10–11 | 11–12 | 12–13 | **bright (8–10)** | faint (11–13) | f/b |
|---|---|---|---|---|---|---|---|---|---|
| Husillos zenith (own free quintic) | 0.093 | 0.130 | 0.137 | 0.139 | 0.153 | 0.180 | **0.133 ″** | 0.167 | 1.25 |
| León zenith, 12 fields (six-field cubic frozen) | 0.086 | 0.070 | 0.066 | 0.065 | 0.063 | 0.069 | **0.068 ″** | 0.066 | 0.97 |
| Husillos 15°, frozen (the map) | 0.239 | 0.274 | 0.339 | 0.304 | 0.409 | 0.517 | 0.306 ″ | 0.463 | 1.51 |
| Husillos 15°, **free quintic** | 0.227 | 0.236 | 0.259 | 0.250 | 0.374 | 0.491 | **0.248 ″** | 0.432 | 1.75 |
| Husillos 10°, frozen, 2 fields (the map) | 1.125 | 1.202 | 1.244 | 1.212 | 0.947 | 0.858 | 1.223 ″ | 0.903 | 0.74 |
| Husillos 10°, **free quintic** (gain 0) | 1.083 | 0.763 | 0.891 | 0.911 | 0.779 | — | **0.827 ″** | 0.779 | 0.94 |
| León horizon, 9 windows at 8.5–12.4° | 0.236 | 0.281 | 0.276 | 0.258 | 0.258 | 0.266 | **0.279 ″** | 0.262 | 0.94 |

*(arcsec; one rms per field per bin, fields equal-weighted, a bin needs 8 stars.)*

**Read the bright columns and the f/b ratio.**

* **At the zenith Husillos is 2× León and it is structure**: 0.133 ″ against 0.068 ″ at G 8–10,
  with a faint/bright ratio of only 1.25, so noise adds little. This is the more telling of the
  two comparisons, because León's 0.068 is not a León number — Leakey gives 0.072 and Bruns
  0.052 on different optics in different years (`floor_vs_sampling.csv`). Three instruments
  sit at 0.05–0.07 ″ and Husillos sits at 0.13. And the constructions are not like-for-like in
  the direction that *favours* Husillos: a free quintic on its own field should leave less
  than a cubic frozen from five other fields, not more. What it is — the optic (this is the
  cell that needed a quintic, §HUSILLOS2026_ZENITH), or the night — is open.
* **At 10°, once model transfer is removed, Husillos is 3× León and it is structure**: the free
  quintic takes the bright end 1.223 → 0.827 ″, so a third of the map's residual was the frozen
  zenith model (§3u said a quarter of the rms; at the bright end it is a third); what remains
  is 0.827 ″ against León's 0.279 ″ with f/b = 0.94 — flat across magnitude, the matrix's own
  criterion for structure. Photon noise is *not* what makes the Husillos horizon panels look
  rough.
* **At 15° they nearly meet**: Husillos free 0.248 ″ against León's 0.279 ″ at 8.5–12°. But 15°
  is a gentler altitude than any León window, so at equal altitude Husillos is still the
  worse.

**What the excess is made of.** The free-quintic row has every smooth low-order error taken
out — model transfer and whatever the assumed weather does to the refraction correction at
cubic order and below — so its 0.83 ″ is high-order and it is not noise. That points at the
turbulent atmosphere through 5.6 air masses, and the polarisation says the same thing from a
different side: León's excess over its own zenith is almost all **vertical** (0.240 against
0.098 ″, V/H 2.4), while Husillos' is large in **both** components (vertical 0.84–0.99 ″,
horizontal 0.55–0.72 ″, V/H 1.4–1.5). A refraction-related term is vertical; an isotropic
turbulent term added on top of one lowers V/H exactly as seen. The simplest consistent
picture is **León's vertical structure plus a large isotropic component that León did not
have**, which is what a hot August plain at 743 m three hours after sunset, seen through six
air masses, would be expected to produce; León's site is at 1101 m. This is the most
consistent reading, not a measurement of the cause.

**One construction difference that could contribute and is not separated here.** León's map
is a per-star median over forty-five *separately solved* 4 s frames, so each frame's own
low-order atmospheric distortion is absorbed before the median; Husillos' is one 100 s stack
with one quadratic at the end. Over ~100–180 s of averaging the two should converge on the
same quasi-static field, but they are not the same operation, and a per-frame reduction of a
Husillos horizon capture would settle how much of the 3× is the construction. Cell 4 has the
frames to do it.

**What this means for the record.** §3s's atmosphere term (±0.25 ″, unresolved above noise)
stands — that number was built from *L* fitted on these residuals, and the structure seen here
is what the estimator's four freedoms could not project into a 1/r pattern. But the excess is
real and it is the physical reason cell 4's per-star noise (0.53 ″ between the blocks, §3f) is
the largest in the matrix: the eclipse field was shot through the same air.

## 3w. The first three `10 deg` captures were UNTRACKED — and that, not the sky, is the 3×

Douglas, 2026-09-12: *"Do the per-frame reduction of one horizon capture."* It was meant to
split §3v's 3× between site and construction. It did, and then it found why.

### The per-frame reduction, León's construction on 23_34_38

`tools/husillos2026/hu_perframe.py`: every frame 1–99 of `23_34_38` reduced alone — stage 1
on one frame at the zenith star-field preset (which is also León's per-frame regime,
`drive_horizon.STAGE1`), stage 2 against the same zenith quintic at the same rung as the
stack, corrections on at the frame's own mid-time from the SER trailer; then per star the
median over frames, a star needing 20. All 99 frames solved, 22–77 stars each (median 61),
rms 1.12–1.65 ″ each alone.

| same capture, bright end G 8–10 | stars | rms | vertical | horizontal | V/H |
|---|---|---|---|---|---|
| one 100-frame stack, zenith model frozen (the night map) | 339 | **1.171 ″** | 0.843 | 0.548 | 1.54 |
| one stack, whole quintic free | 351 | 1.006 ″ (all G) | | | |
| **per-frame median over 70 frames per star** | 75 | **0.393 ″** | 0.328 | 0.172 | **1.90** |
| León horizon, per-frame medians | — | 0.279 ″ | 0.240 | 0.098 | 2.4 |

A factor of **3.0 from the construction alone, on the same photons**. With the construction
matched, Husillos at 10° is 1.4× León, and its polarisation (V/H 1.90) is close to León's
2.4 rather than the stack's 1.54. Freeing the quintic on the stack changes nothing here
(1.006 ″ against 1.006), so the stack's excess is not the frozen model either.

### Why: the stack is smeared, and the smear is drift, not refraction

`hu_streak.py` measured every star's second moments in the stack and in one frame. The
stack's stars are streaked and the streak grows toward the edges — **along sensor x, not the
vertical**: σ_x 1.98 → 2.58 → 3.43 px from the centre band outward, σ_y 2.14 → 2.30 → 2.40;
the single frame is flat at ~2.2 / ~1.75 in every band. Streaks along RA growing with the
offset perpendicular to it are differential drift, not refraction (which would streak along
the vertical). The 99 single-frame solves then said it outright:

| over the 130 s capture | first | last | slope |
|---|---|---|---|
| **RA of the field centre** | 176.3776° | 176.9191° | **+15.09 ″/s = +0.545°** |
| Dec | 29.5996° | 29.5995° | −0.006 ″/s |
| roll | 326.1668° | 326.1648° | +0.02 ″/s |

**The sidereal rate to 0.3 %, with the roll constant to 0.0007°: an equatorial mount with
tracking off.** (A stationary alt-az pointing would rotate the field at the 5.8 ″/s
parallactic rate; it did not, which is what pins the mount as equatorial and the tracking as
the thing that was off.) The stacks' own alignment records, which the pipeline writes for
every capture, close the case:

| capture | frames | total drift | per frame | tracked? |
|---|---|---|---|---|
| `23_31_59` | 99 | **774 px** | **7.83 px** | **no** |
| `23_34_38` | 99 | **767 px** | **7.84 px** | **no** |
| `23_37_17` | 99 | **800 px** | **8.25 px** | **no** |
| `23_41_01` | 50 | 5.4 px | 0.60 px | yes |
| `23_42_43` | 47 | 5.4 px | 0.32 px | yes |
| `23_44_06` | 49 | 1.3 px | 0.17 px | yes |
| `22_56_41` (cal 8 deg) | 99 | 6.8 px | 2.53 px | yes |
| `22_59_14` (cal 8 deg) | 99 | 7.4 px | 1.06 px | yes |
| zenith `00_00_21` | 50 | 1.2 px | 0.14 px | yes |
| eclipse gain 125 | 126 | 2.4 px | 0.25 px | yes |
| eclipse gain 0 | 101 | 0.7 px | 0.26 px | yes |
| CalibS | 61 | 8.6 px | 0.29 px | yes |

7.8 px per 1.316 s frame is 15.04 ″/s × cos(29.6°) ÷ 2.2028 ″/px. Tracking came on between
`23_37_17` and `23_41_01` — the 14-frame `23_40_27` is very likely the moment it was
noticed. **Everything the science rests on was tracked**: both eclipse blocks, CalibS, the
zenith, and the `cal 8 deg` pair.

What an untracked stack does to its stars: each 1 s frame already carries a 6 px trail along
RA (the single frame's σ_x > σ_y says so), the aligner's one global shift can follow the
field's mean drift but not the cos(Dec) spread in drift rate across ±2.9° of declination, so
the edge stars are laid down as ~10 px streaks, and a windowed centroid on a uniform streak
lands wherever noise made it brightest. That is a magnitude-independent, non-smooth,
edge-weighted error of order a pixel — **exactly what §3v measured, correctly called
structure, and wrongly called the atmosphere.**

### What this corrects, and what it leaves standing

* **§3v's conclusion is withdrawn as stated.** Husillos' 10° fields are not "3× rougher than
  León's". Like for like, they are ~1.4× (0.393 against 0.279 ″), and the per-frame medians
  carry more sampling noise than León's (70 frames of 1.33 ″ against 45 of ~0.5 ″), so the
  true ratio is nearer 1.2. The "large isotropic component" was drift smear. The zenith
  comparison (0.133 against 0.068 ″) is untouched — that capture was tracked.
* **§3u's night map is wrong in its two 10° panels**: they are stacks of untracked captures,
  and the swirl in the `23_34_38` panel is a drift-stack artefact, not sky. The map will be
  redrawn from per-frame medians for those two panels (`23_31_59`'s per-frame run is under
  way); the 15° and zenith panels stand.
* **§3s's atmosphere term needs its 10° rows rebuilt.** The field-to-zenith nulls at 10°
  (+0.235 and −0.261 ″) and the deep 10° consecutive pair were fitted on the smeared stacks;
  their floors (0.46 and 0.29 ″) are partly smear. The 15° field and the `cal 8 deg` pair are
  unaffected. The term is to be re-derived on the per-frame medians before it is quoted
  again; the interim ±0.25 ″ should be read as an upper bound from contaminated inputs.
* **§3s's pointing map is wrong in one word**: the "re-pointing 0.66° in RA to hold 10°" was
  158 s of sidereal drift through a fixed telescope (0.0042°/s × 158 s = 0.66°); so was the
  0.88° between `23_37_17` and `23_41_01`. The declination changes were real slews.
* **§3u's refraction result stands.** All three 5.5° captures gave +1200–1800 ppm, two of them
  tracked; the scale blow-up is not a tracking artefact.

### Rebuilt on per-frame medians: both 10° captures, the map, the term

`23_31_59` reduced the same way (all 99 gain-0 frames solved, 22–40 stars each; 42 stars with
20 or more frames). The two untracked captures, like for like with León:

| per-frame medians, bright end G 8–10 | stars | rms (all) | vertical | horizontal | **V/H** |
|---|---|---|---|---|---|
| `23_31_59`, gain 0 | 42 | 0.540 ″ | 0.509 | 0.182 | **2.80** |
| `23_34_38`, gain 125 | 75 | 0.371 ″ | 0.328 | 0.172 | **1.90** |
| León horizon, 9 windows | — | 0.279 ″ | 0.240 | 0.098 | 2.4 |

Bright-end mean 0.442 ″ against León's 0.279 (`hu_maps_bymag.py`, row "PER-FRAME (2)"), and
the medians here carry ~0.2 ″ of their own sampling noise (70 frames of 1.34 ″ against León's
45 of ~0.5 ″), so the structure is nearer 0.39 ″: **1.4× León, polarised the same way (V/H
1.9–2.8 against 2.4)**. That is the end of §3v's question: Husillos' 10° sky is León's 10° sky
within the noise, once both are reduced the same way.

The night map (`atmosphere_night_maps.png`, revision 4, the third revision superseded) now
draws both 10° panels from the per-frame medians; the swirl is gone with the smear.

**The atmosphere term, rebuilt on the valid fields** (`hu_atmosphere.py`, variant `p`: the
per-frame medians in place of the untracked stacks, the tracked deep stacks elsewhere):

| field-to-zenith null, the cell's estimator | alt | N | total | floor | structure |
|---|---|---|---|---|---|
| `23_31_59`, per-frame medians | 10.2° | 40 | +0.034 ″ | 0.401 ″ | 0 |
| `23_34_38`, per-frame medians | 10.2° | 71 | −0.097 ″ | 0.251 ″ | 0 |
| `23_44_06`, tracked stack | 15.0° | 935 | +0.142 ″ | 0.078 ″ | 0.119 ″ |
| `22_59_14`, tracked stack (twilight) | 8.5° | 47 | +1.881 ″ | 1.220 ″ | noise |

**±0.10 ″** — the rms over the three fields that can carry the construction (0.034, 0.097,
0.142), quoted as the total per the matrix rule, with the only resolved structure the 0.119 ″
at 15°. Against the interim ±0.25 ″ (contaminated inputs), León's ±0.33 ″, Station 1's
±0.11 ″ and Bruns' ±0.15 ″. Two caveats travel with it: the per-frame floors (0.25–0.40 ″) are
overestimates, because the bootstrap perturbs by the median's whole residual rather than by a
median-over-70-frames' noise, so "unresolved" at 10° is conservative; and the eclipse-altitude
pair is still too thin to contribute. **The atmosphere is not what limits cell 4.** Its
largest term is the per-star noise of a 315 ms coronal field through six air masses (§3f),
and that is already in the ±0.430 ″.

### For 2027

Check tracking before every capture, not only after slews. Three of seven captures in the
best-populated horizon window were lost to it, and the loss was invisible in the folder
names, the frame counts, the headers and the plate solves — the RA drift and the alignment
record were the only witnesses, and nobody had asked them. And reduce horizon captures
frame by frame, as León did, whatever the tracking: a 100 s stack at 10° is smeared by
refraction evolution even when the mount is perfect.

## 4. A tool that does not work, and says so

`hu_eclipse_match.py` was written to match the detections against Gaia at the known pointing —
the Sun's apparent place at 18:29:43 UTC from Husillos is RA 142.107, Dec +14.909, so only roll
and a small translation are free. **It does not work, and it now refuses to report.**

Run on the **zenith** stack — a field stage 1 solved blind, at RA 281.7427, Dec +50.2092, roll
325.902 — it votes 3 pairs at the known roll, and its roll histogram peaks at 34.0° with a
peak-to-99th-percentile ratio of 1.05, which is flat. **A tool that cannot find a field it has
been given the answer to cannot be believed when it finds nothing in a field it has not**, so
`main` runs that control first and exits if it fails.

Two bugs were found and fixed along the way and are recorded because they will recur:

* **The catalogue arrays are radians, and sorted by declination.** Reading them as degrees returns
  an empty cone and no error at all — the first run reported "0 stars in the field" for a
  catalogue of 7 369 627.
* **A roll error of 0.25° swings a star at the 5756 px half-diagonal through 25 px**, so a
  translation vote binned at 4 px only adds up within ~900 px of the rotation centre. Scanning
  roll at 0.25° over the whole frame finds noise, which is what it found.

What is likely still wrong: the brightest detections are saturated or blended and so are not the
catalogue's brightest, and a 6 px tolerance on pair separations is smaller than the tens of pixels
of distortion across an 11 000 px baseline. Neither is fixed.

---

## 5. What to do next

1. ~~The blind solver needs a cleaner list, not a bigger one.~~ **Done** (§3b): the hot-pixel
   mask cut 304 centroids to 114 and the field solved in 1.1 s. **Stage 2 can now begin** — it
   needs a distortion reference (the zenith field at quintic) and the refraction correction on,
   since the Sun was at 8.6° altitude.
2. **Real darks are still worth asking for** (§3c): the mask reaches only pixels hot enough to
   clear 5 σ in one 1.0 s frame, and the mildly-hot ones that matter in a deep stack are beyond
   it. 1328 of `sn2_masked`'s 4061 centroids have an area of ≤ 2 px.
3. **The refraction correction must be on for anything at 8.6° altitude**, and the site is now
   known. Nothing in this document depends on it — these are counts and ratios — but a fit will.
4. ~~**`CalibS` is now the top priority**~~ — **done** (§3j–3k): reduced, importable at 40 ppm
   from the gain-0 block, and Method 1 run for the first time. What remains on it: the stage-1
   re-run under the eclipse detection settings, since the first used the zenith preset's
   thresholds and its ±25.2 ppm is the cell's binding term.
4b. **Stage 3 ignores `flag_is_outlier`** (§3e). It re-admitted the two stars that wreck the
   gain-0 block. Cell 2 reduced through `s1_pooled_fit.py` rather than the CLI's stage 3, which
   is probably why this has not bitten before. The two-witness rule removes those two stars
   independently (§3f), so it is not blocking, but it is still a defect.
4c. ~~**The 2.2 σ between the blocks needs a cause**~~ — **answered** (§3g): it is per-star
   noise, and the León union averages it. The blocks are no longer quoted separately.
4d. **The union is the reduction of record for cell 4** (§3g), and it will need re-running when
   CalibS arrives or a second zenith field changes the reference.
4f. **CalibS' ±19 ppm is a floor, not a budget** (§3n): its scale walked +18.6 ppm under two
   corrections that were each right. Method 1 walked 2.840 → 3.000 → 3.290 ″ with it and should
   not be quoted as a measurement in that state.
4g. ~~The plate scale drifts through totality~~ — **tested and rejected** (§3o). Open instead:
   why the two gain-0 halves differ by −79.7 ± 21.0 ppm when the gain-125 halves do not.
4h. ~~Gain 125's excess points at the atmosphere~~ — **superseded by §3q**: the Sun-centred
   structure follows the GAIN boundary and not the clock (3.3 σ across the gain in 21 s,
   1.5 σ at one gain in 20 s). The mechanism is saturation physics: sensor full-well at gain 0
   against ADC clipping at gain 125, occulter 592 vs 706 px. The V/H difference is real but is
   not the explanation of the block disagreement.
4i. ~~Run the occulter experiment~~ — **void** (§3q): a gate cannot move a star outside it.
   ~~The seeing~~ — **no step** (1.3 σ, wrong sign, and 30× too weak by estimate). **The
   half-block table was over-read** — and then §3r withdrew the whole thread: the
   "Sun-centred structure" was differential refraction between the block epochs, left in by a
   similarity fit on raw pixels and projected radially through eleven lopsided inner stars.
   On refraction-corrected displacements the blocks agree to 6 ppm. **There is no block
   systematic to explain; §3f stands.** The gain is a bystander.
4k. **Cell 4 has a first atmosphere term, provisional: ±0.25 ″** (§3s), the total on the
   fields with enough stars to measure it, floor not subtracted. **No structure is resolved
   on any of them** — the floor exceeds the total, which happens in no other cell, because
   1 s through six air masses is photon-starved. The only resolved structure in the window is
   0.117 ″ at 15° on 653 stars. Deep detection cracked the `cal 8 deg` window and it is the
   best geometry in the dataset: one tracked field, 2 min 33 s apart, straddling the eclipse
   altitude — but at 37 and 53 stars it measures nothing. **Do not add the 63-star column
   (~0.5 ″) as a systematic**: that is sampling noise and is already the union's ±0.430 ″.
4m. **The pathway of record under-reads the deflection by 14 %** (§3s) — fitting the eclipse
   field with the quadratic free lets f = 0.860 of a 1/r pattern through to stage 3, measured
   on the pipeline by injection (2.000 ″ in, 1.787 ″ out on the gain-125 block) and
   **confirmed independently by the rung ladder** (§3t): four pathways spanning 0.403 ″ of L
   collapse to a 0.069 ″ spread once each is divided by its own f. **Decision needed:** the
   f division is the substantive change; the rung is not. Recommended in §3t — quote the
   record's rung with f applied, **2.476 ± 0.500 ″**, with the `linear` rung's 2.463 ± 0.492 ″
   as the check.
4n. **The block scale disagreement lives in the LINEAR terms** (§3t): 72.8 ppm with the
   quadratic frozen, 70.9 ppm with it free, 20.9 ppm once the linear is frozen too. It is
   §3r's anisotropic affine seen from the other side. Open: whether the residual 20.9 ± 18.6
   ppm is anything at all, and whether real weather would remove the affine.
4o. **The assumed weather is no longer wholly unquantified** (§3u). Turning the correction off
   measures what it removes: at 10–15° it lands within ~150 ppm of a 1700–3400 ppm
   compression, and the eclipse-altitude captures at 8.54° and 9.01° sit at −44 and −158 ppm.
   The model breaks between 8.5° and 5.7°, below anything the eclipse used. **Still worth
   asking Joe for the real weather**, but the bound is now ~150 ppm of scale, not unknown.
4p. **The frozen zenith cubic-and-above is wrong in SHAPE at low altitude** (§3u): freeing the
   quintic takes the residual 1.392 → 0.553 ″ at 5.70° and 1.256 → 0.928 ″ at 10.03°, while
   moving the scale by under 80 ppm. The eclipse blocks are at 8.6° on a frozen zenith model;
   what this costs them has not been measured, and it is the natural next test.
4q. ~~The 10.0° gain-125 night map carries a coherent swirl~~ — **explained** (§3w): a stack
   of an untracked capture. Not sky.
4r. **The first three `10 deg` captures were untracked** (§3w: 770–800 px of sidereal drift
   over 99 frames). Their stacks are invalid for astrometry. ~~To do~~ — **done**: both
   reduced per frame, the night map redrawn (rev. 4), the term rebuilt: **±0.10 ″**, with
   the 10° sky 1.4× León's and polarised the same way. Left: a proper floor for per-frame
   medians (bootstrap over frames, not over the median's residual), which would only make
   the 10° rows better resolved. **2027: check tracking before every capture, and reduce
   horizon captures frame by frame.**
4k. *(superseded)* The atmosphere term is **±0.10 ″** (§3w), not ±0.25 ″.
4l. ~~Sun-centred or smooth field~~ — **neither** (§3r): per bin nothing reproduces on held-out
   stars, and the linear terms are the refraction ramp.
4j. **The gain-0 halves' uniform −79.7 ± 21.0 ppm scale step** (§3o, §3q) is unexplained and is
   not the corona.
4e. **The vertical nuisance is measured and deliberately not applied** (§3h). Re-measure it if
   the star count grows: V/H = 1.80 says the polarisation is real, only its smooth part is
   absent. Applying it would also need the estimator fitted in a frame rotated 14.9° from the
   sensor axes, which León did not need.
5. The Sun capture's **gain 125 against Sn2's gain 0** means their scales must not be carried
   across without measurement.
