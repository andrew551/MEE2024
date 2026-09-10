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
4. **`CalibS` (`20_30_18`, 145 frames) is the next capture to reduce**, since it carries the plate
   scale, and its first ~80 frames are inside totality before C3 at 18:30:44.2.
5. The Sun capture's **gain 125 against Sn2's gain 0** means their scales must not be carried
   across without measurement.
