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

**Conclusion: the vertical nuisance is not applied to cell 4, and the record's L stands at the
unfiltered fit.** Applying it would mean fitting five parameters, in a frame that first needs a
14.9° rotation, to absorb a structure the data says is not there — which is exactly the
"never choose an analysis parameter at the keyboard" trap. If more stars arrive (a second
zenith field, or CalibS deepening the reference) this is worth re-measuring, because the
V/H = 1.80 says the *physics* León's filter targets is present; only its smooth part is not.

### What this is and is not

* The two blocks' separate values (1.596 and 2.782 ″) **are superseded by the union** and should
  not be quoted alone; their 2.2 σ spread is per-star noise, which the union averages.
* **The weather is assumed** — 926.5 hPa is the standard atmosphere at 743 m, with 25 °C and 35 %
  as ordinary August values. At z = 81.4° the plate scale carries ~63 ppm per 1 % of the
  refraction constant, so this is the single largest unquantified term.
* **One zenith field, not seventeen** — no reference field-to-field term at all.
* **No atmospheric data** — Joe took none, so the ±0.11–0.33 ″ every other cell carries from
  zenith nulls has no counterpart.
* **No darks and no flats.**
* **The outer radial bound is inherited**, and is not applied.

So cell 4 stands at **L = 2.129 ± 0.430 (stat) ″**, the two-gain union under the two-witness
rule, GR at 0.88 σ — a real fit with an incomplete budget, not yet a matrix entry. What would
close it: **CalibS**, for a same-day same-altitude *imported* scale — the night zenith's is
1365 ppm from the eclipse field's and cannot be imported at all, and at 0.0324 ″/ppm this field
punishes an imported scale error harder than any other in the matrix; the **real weather**; and
more zenith fields for a reference term.

---

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
4. **`CalibS` (`20_30_18`, 145 frames) is now the top priority**, not merely next: the night
   zenith's scale is 1365 ppm from the eclipse field's, so the only route to an imported scale —
   and therefore to a real Method 1 — is a same-day, same-altitude calibration field. Its first
   ~80 frames are inside totality before C3 at 18:30:44.2.
4b. **Stage 3 ignores `flag_is_outlier`** (§3e). It re-admitted the two stars that wreck the
   gain-0 block. Cell 2 reduced through `s1_pooled_fit.py` rather than the CLI's stage 3, which
   is probably why this has not bitten before. The two-witness rule removes those two stars
   independently (§3f), so it is not blocking, but it is still a defect.
4c. ~~**The 2.2 σ between the blocks needs a cause**~~ — **answered** (§3g): it is per-star
   noise, and the León union averages it. The blocks are no longer quoted separately.
4d. **The union is the reduction of record for cell 4** (§3g), and it will need re-running when
   CalibS arrives or a second zenith field changes the reference.
4e. **The vertical nuisance is measured and deliberately not applied** (§3h). Re-measure it if
   the star count grows: V/H = 1.80 says the polarisation is real, only its smooth part is
   absent. Applying it would also need the estimator fitted in a frame rotated 14.9° from the
   sensor axes, which León did not need.
5. The Sun capture's **gain 125 against Sn2's gain 0** means their scales must not be carried
   across without measurement.
