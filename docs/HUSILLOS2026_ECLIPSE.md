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

## 3e. A first Method 2 — it runs, and it says nothing

Douglas, 2026-09-10: *"Let's do a quick Method 2 calculation with what we have so far."* Done,
by cell 2's pathway rung for rung (`docs/V1_4_0_TESTING.md` §5), through the window registered as
`analysis_window.WINDOWS['husillos2026']` before the fit was written:

| rung | setting | result |
|---|---|---|
| zenith reference | free **quintic**, gate 0.5 ″, refraction on | 2635 stars, rms 0.1594 ″, ps **2.2059136** |
| eclipse field | `constant` + `distortion_free_scale`, 20 ″ then 3 ″ | 73 stars, rms **0.8767 ″**, ps **2.2029020** |

The rung is read back from the run's own results, as CLAUDE.md requires: *fixed distortion order:
**constant***, *plate scale source: **fitted on this field***, *distortion_free_scale: **True***.

**The answer:**

| | L | plate scale |
|---|---|---|
| **Method 2** (scale fitted alongside) | **0.562 ± 6.248 ″** | 2.203136 ± 0.000408 (185 ppm) |
| Method 1 (scale taken as known) | 3.395 ± 3.845 ″ | 2.202902 ± 0.000045 (20 ppm) |

66 stars admitted, deflected-position rms 8.688 ″. **Both are consistent with GR's 1.75 ″ and
with zero. Neither is a measurement.** The Method 2 bar is 3.6× the quantity being measured.

### Why, and what it tells us to do

**The night zenith's plate scale cannot be imported into the eclipse field.** Fitted freely the
eclipse field wants **2.2029020** against the reference's **2.2059136** — **−1365 ppm**. Fixing
the scale to the reference instead of fitting it does not merely bias the answer, it destroys the
fit: *"second pass: 72 star(s) beyond 3.0 ″ after the refit removed (first gate 20.0 ″, pre-refit
rms 9.579 ″)"* — 72 of 73 stars gone.

That is not surprising once the altitude is taken seriously. At z = 81.4° refraction compresses
the field by k(1 + sec²z)/2 ≈ **6270 ppm**, and the fits confirm it: turning corrections on moved
the eclipse field's scale by −6900 ppm and the zenith field's by −235 ppm, against −6270 and −286
predicted. **So the eclipse scale carries ~63 ppm per 1 % error in the refraction constant**, and
the 1365 ppm gap is about 22 % — comfortably inside the uncertainty of *assumed* weather.

Three consequences, and they are the useful output of this exercise:

1. **Cell 4 cannot take its scale from a night zenith field.** It needs **CalibS** — same day,
   same altitude, minutes apart — which is exactly what the ladder's middle rung is for and is
   the next capture to reduce.
2. **The real weather matters here in a way it does not at the zenith.** 926.5 hPa / 25 °C / 35 %
   are assumed; at 63 ppm per 1 % they are worth more than a thousand ppm of plate scale.
3. **One 32 s block cannot carry Method 2.** The per-star residual is 0.8767 ″ against cell 2's
   0.13 ″, on 66 stars against cell 2's 639 observations — about 21× worse in the combination,
   which is what turns cell 2's ±0.084 ″ into ±3.8 ″ here. Cells 1–3 got their precision from many
   tiers pooled with a shared scale; Husillos has **one** science block and no second tier.

### What is missing from the budget — not uncertain, absent

* **one zenith field, not seventeen** — no reference field-to-field term at all
* **no atmospheric data** — Joe took none, so the ±0.11–0.33 ″ every other cell carries from
  zenith nulls has no counterpart here
* **the weather is assumed**
* **no darks and no flats**
* **the outer radial bound is inherited from cell 2**, not decided on cell 4's own data

So this is a number that exists, not a number that means anything. It is recorded so the next
session starts from a working pathway rather than a blank page, and **it should not go in the
matrix.**

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
4. **`CalibS` (`20_30_18`, 145 frames) is now the top priority**, not merely next: §3e shows the
   night zenith's scale is 1365 ppm from the eclipse field's and cannot be imported at all, so
   the only route to a usable scale is a same-day, same-altitude calibration field. Its first
   ~80 frames are inside totality before C3 at 18:30:44.2.
5. The Sun capture's **gain 125 against Sn2's gain 0** means their scales must not be carried
   across without measurement.
