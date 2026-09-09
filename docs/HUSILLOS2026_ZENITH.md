# Husillos 2026, the zenith fields: focal length, distortion order, and the AM5

**Date:** 2026-09-09. Matrix cell 4 (Joe Izen, Husillos 2026). Everything below is measured on
`G:\Joe Izen Husillos 2026` and, for the controls, on `G:\Leon Aug 2026`, with `v1.4.0-dev`.

Four questions were asked (Douglas, 2026-09-09): is the cubic enough on this full frame or is a
quintic needed; is the focal length the same as Leon 2026's; can the ZWO AM5's periodic error be
judged against Leon's Celestron AVX; and are the captures settled at the start and still tracking
at the end.

## 0. Three names, and what each one means

The names have to be kept apart, because two of them are the same country and one of them is a
date (Douglas, 2026-09-09):

| name | what it is |
|---|---|
| **the Spain 2026 eclipse** | the **event**: the total solar eclipse of 2026-08-12, whose path crossed northern Spain. It is not a dataset and not a matrix cell. |
| **Leon 2026** | **matrix cell 3** — the project's own station at León (42.60° N, 5.57° W). `G:\Leon Aug 2026`, an ASI2600 on a Celestron AVX. |
| **Husillos 2026** | **matrix cell 4** — Joe Izen's station at Husillos (below). `G:\Joe Izen Spain 2026` on the drive, a Zeus 455M PRO on a ZWO AM5. |

"Spain 2026" was used for Joe's data in `spain2026_prompt.md`, in this project's tools and in the
first version of this document. **That was wrong and is now corrected throughout**: León is in
Spain too, so the label cannot distinguish the two stations. The drive folder still reads
`G:\Joe Izen Spain 2026` and is left alone — it is read-only input.

**The two stations observed the same eclipse ~100 km apart**, which is why cell 3 is the control
for cell 4 everywhere below: Leon's zenith mosaic was shot 48 minutes after Joe's zenith capture,
on the same night, through the same telescope model with the same reducer and the same 3.76 µm
pixels. Reduced here at identical settings it is the reason these answers are as sharp as they
are.

### The Husillos site, exact

From `G:\Joe Izen Spain 2026\Husillos Spain.JPG` (Douglas, 2026-09-09) — Área de Servicio
Autocaravanas Husillos, Palencia:

| | |
|---|---|
| latitude | **42° 05′ 34.55″ N = +42.09293°** |
| longitude | **4° 31′ 37.26″ W = −4.52702°** |
| height | **743.0 m (2438 ft)** |
| C1 | 2026-08-12 17:33:46.4 UTC, alt +18.9°, az 273.1° |
| **C2** | **18:29:00.5 UTC**, alt +08.8°, az 282.0° |
| maximum | 18:29:52.5 UTC, alt +08.6°, az 282.2° |
| **C3** | **18:30:44.2 UTC**, alt +08.5°, az 282.3° |
| C4 | 19:22:27.6 UTC, alt −00.7°, az 290.8° |
| sunset | 19:23 UTC |
| **totality** | **1 m 43.5 s** (lunar-limb corrected; 1 m 43.7 s tabulated) |
| magnitude, obscuration | 1.01298, 100 % |
| umbral depth, path width | 78.12 % (115.1 km), 294.6 km |

Two things this settles. **The frame-derived C3 of the previous session was right**:
18:30:46 ± 2 s against the tabulated 18:30:44.2. **Its totality duration was not**: ~115 s was
read off the frames, against the true **103.7 s** — the C2 estimate was ~11 s early. And the Sun
is at **8.6° altitude** through totality, so **nothing on eclipse day is refraction-safe**; the
zenith fields below are, which is why the order and focal-length answers do not wait on it.

---

## 0b. Scope, and one thing done outside it

Douglas asked for the two zenith folders. §3.5 also reports a stage-1 failure on
`Capture/00_54_32`, which is **not** a zenith file and was not asked for. It was run while
looking for a second star-rich full-frame field to make the order test out of sample, before the
scope was clear. The finding is left in because it will matter when cell 4 reaches that capture,
but it is out of scope for this document and nothing here depends on it.

Tools: `tools/husillos2026/ser_track.py` (per-frame star tracking), `hu_zenith_order.py` (stage 1 and
2 at three orders, the radial diagnostic, the transfer test), `hu_mount_compare.py` (AM5 against
AVX with one estimator), `hu_am5_periodic.py` (the rate-scatter method — it did not deliver; §5).

---

## 1. What the two zenith folders actually contain

**Only one of them is a zenith field.**

| | `2026-08-12/zenith/23_24_56.ser` | `2026-08-13/zenith/00_00_21.ser` |
|---|---|---|
| frames | 16 | 50 |
| dimensions | **1280 × 1024**, Pan 4148 / Tilt 2682 | 9576 × 6388, full frame |
| exposure, gain | 1.0 s, 0 | 1.0 s, 0 |
| UTC start | 2026-08-12 21:24:56 | 2026-08-12 22:00:21 |
| duration, cadence | 16.8 s, 1.0508 s | 64.67 s, 1.3198 ± 0.0061 s |
| stage-1 centroids | 29 | 3211 |
| plate-solves? | **no** | yes, in 0.9 s |

Pan 4148 / Tilt 2682 is the exact centre of a 9576 × 6388 sensor, so the 12 August capture is a
**centred 2.1 % crop**: 0.78° × 0.63° against the full frame's 5.87° × 3.91°. It reaches 1808″ of
field radius where the full frame reaches 12699″, so it carries **no information at all about the
distortion order** — the terms in question live in the corners it does not have. Stage 1 finds 29
centroids in it and the solver fails after 77 s.

It is not useless: it says what the mount was doing. **The field drifts +3.72 px/frame in x and
+2.42 px/frame in y, dead straight through all sixteen frames** — 4.44 px per 1.0508 s frame, or
**9.32 ″/s**. The sidereal rate is 15.041 ″/s, so 9.32 ″/s is the rate at declination ±51.7°, and
the full-frame zenith capture 35 minutes later plate-solves at **Dec +50.21°**, where a stopped
mount drifts 9.63 ″/s. **Tracking was off for this capture.** The star images confirm it and
nothing else: the brightest star measures σ 2.05–2.15 px along the drift and 1.57–1.70 px across
it, and 4.22 px of trail in a 1.0 s exposure adds 4.22/√12 = 1.22 px in quadrature to a 1.65 px
PSF, giving 2.05 px. The whole elongation is the drift; there is no defocus and no vibration.

The camera settings agree: the mount's reported RA in that folder's four settings files advances
with the wall clock (05:36:33 → 05:38:55 over 142 s), which is a stopped mount reporting where it
happens to be pointing. Elsewhere it reports **Dec = +90°00′00″** — the home reading of a mount
that was never synced — so **the ASI Mount lines in this dataset cannot be used as pointing**.
Stage 1 says so itself on the good capture: *"solved position is 39.79° from the header (RA
18.033°, Dec 90.000°): nowhere near the telescope pointing"*.

So everything that follows rests on **one** full-frame zenith capture. That matters in §3.

---

## 2. The focal length: the same telescope, 0.15 mm apart

Both fields fitted free, `enable_corrections=False`, gate 0.5″, `max_star_mag_dist=13`, quintic:

| field | night | frames | stars | rms | plate scale | focal length |
|---|---|---|---|---|---|---|
| Husillos zenith, 12 Aug 22:00 UTC | 12 Aug | 50 × 1.0 s | 2680 | 0.1605″ | **2.2064323 ″/px** | **351.498 mm** |
| Leon Z1_base, 12 Aug 22:48 UTC | 12 Aug | 30 × 4 s | 2516 | 0.0929″ | 2.2073644 ″/px | 351.349 mm |
| Leon Z4_top_right, 12 Aug 23:01 UTC | 12 Aug | 30 × 4 s | 2977 | 0.0954″ | 2.2073773 ″/px | 351.347 mm |

**The two focal lengths differ by +425 ppm — 351.498 mm against 351.348 mm, a difference of
0.149 mm.** Both are the nominal FRA500 (90 mm f/5.6) with a 0.7× reducer, nominal 350 mm f/3.9.

That difference is real: it is fifteen to twenty times the project's own measured field-to-field
plate-scale scatter of 20–29 ppm (`docs/INSTRUMENT_COMPARISON.md` §11), and Leon's own two
pointings agree with each other to 6 ppm. But it is *small* — 0.04 % — and it is what two
independent assemblies of the same reducer will give from a slightly different reducer-to-sensor
spacing. **For the purposes of the reduction, treat the trains as the same design with their own
plate scales, and never carry a scale from one to the other.**

**Two traps in comparing these numbers, both of which would give a wrong answer.**

*Refraction cancels here, but only because both fields are near the zenith.* Neither fit applies a
refraction correction (no site is known for Joe's station), and an uncorrected field's fitted plate
scale is high by about `k` = 283 ppm at standard conditions. That term is nearly independent of
zenith distance — the tangential compression is exactly `k` at any `z`, and the mean isotropic
term is `k(1 + sec²z)/2` — so across the whole range 0° to 10° of zenith distance it varies by
**under 5 ppm**, against the 425 ppm being measured. Both fields are zenith pointings, so the
comparison is safe. It would not be safe against a low-altitude field.

*Do not compare either of these with Leon's canonical 2.2054043 ″/px.* That is CAL_piLeo, an
eclipse-day calibration fitted **with** corrections at a different focus, and Leon's own zenith
field on the same night reads **+892 ppm** away from it here. Night-to-day focus and the
correction setting are worth hundreds of ppm between them, which is the entire reason the
three-step ladder exists.

---

## 3. Cubic or quintic: **quintic**, and the septic is now excluded

An in-sample rms cannot answer this — more parameters always fit better — so four things were
measured instead. Free fits, same settings as above, on the full-frame zenith stack and on Leon's
two pointings.

### 3.1 How much the higher order actually moves the stars

Per star, on the stars both fits kept: the difference between the order-N and order-N+2 fitted
positions.

| field | half-diagonal | quintic − cubic, all | quintic − cubic, **outer 20 %** | septic − quintic, outer 20 % |
|---|---|---|---|---|
| **Husillos zenith** | **12699 ″ (3.53°)** | 0.066 ″ rms | **0.108 ″ rms** (max 0.41 ″) | **0.136 ″ rms** |
| Leon Z1 | 8294 ″ (2.30°) | 0.022 ″ rms | 0.033 ″ rms | 0.021 ″ rms |
| Leon Z4 | 8294 ″ (2.30°) | 0.023 ″ rms | 0.035 ″ rms | 0.024 ″ rms |

Husillos' field radius is 1.53× Leon's and the quintic's effect is **3.3× larger**, which is the
direction and roughly the size a radial term of that order predicts.

### 3.2 Out of sample, on a different star set

One field fitted free, then frozen into another with only the constant free. **Two halves of one
capture share their stars**, so anything that fits this field's own per-star quirks — a catalogue
position error, a blend — transfers between them and the test passes it. Leon's Z1 and Z4 are two
pointings 6.4° apart on the same night through the same optic, so only something belonging to the
**telescope** transfers.

| order | Husillos halves (same stars) | **Leon cross-pointing (different stars)** |
|---|---|---|
| cubic | 0.1645 ″ | 0.1323 ″ |
| quintic | 0.1671 ″ | **0.1309 ″** |
| septic | 0.1606 ″ | 0.1310 ″ |

On the honest test the **quintic gains 1.1 % out of sample and the septic gains nothing** — the
classic signature of the quintic being the right order for this optical design and the septic
over-fitting. The halves column is reported beside it precisely to show that it decides nothing.

### 3.3 The radial residual, with a tangential control

Mean signed residual per annulus. Centroid noise has no preferred direction, so the radial mean is
zero within its standard error unless a radial order is missing. The **tangential column is the
control**: on a rectangular sensor the outer annuli hold only the corners, so any anisotropic
field — uncorrected refraction is one, worth `k·tan²z` = 5.6 ppm ≈ 0.07 ″ at this field's corner —
fails to average away there and would read as a radial signal.

**The tangential control is null: the largest |σ| in any annulus of any of the eighteen fits is
2.42, against a radial column that reaches 10.4.** So the structure below is genuinely radial.

Largest |σ| of the radial mean, over the six annuli of each fit:

| field | cubic | quintic | septic |
|---|---|---|---|
| **Husillos zenith** (`with_f0`) | **5.83** | **9.90** | **9.61** |
| Husillos, `drop_last` | 6.23 | 9.95 | 9.46 |
| Leon Z1 | 3.59 | **1.06** | 1.15 |
| Leon Z4 | 3.44 | 3.57 | 2.75 |

Leon's standard errors are the *smaller* of the two (2.6–7.0 mas against Husillos' 3.8–14.9), so
this is not a sensitivity difference: **Husillos' radial structure is about three times more
significant than Leon's at every order.** Leon Z1 goes flat at quintic; Leon Z4 does not, which
is worth recording rather than rounding off — one of the two Leon pointings still shows 3.6 σ.

One caveat, and it bites: a least-squares residual is orthogonal to the fitted basis by
construction, so binned radial means *must* alternate in sign, and their amplitude is **not**
comparable across orders — the rise from 5.8 to 9.9 σ between Husillos' cubic and quintic is not
evidence that the quintic is worse. What is comparable is Leon against Husillos at the same order
with the same binning, and there the difference is stark.

### 3.4 What the closed cells say

Cell 2 (Mexico Station 1) is the **same IMX455 sensor**, 9576 × 6388, at 1.84847 ″/px — a
half-diagonal of 10637 ″ (2.955°). It needed the quintic, and `s1_septic_test.py` confirmed the
quintic sufficient out of sample. **Husillos' field is 19 % larger in angle than that**, at 3.528°.

### 3.5 The answer

**Use the quintic. The cubic is not sufficient.** The quintic moves fitted star positions by
0.108 ″ rms in the outer fifth of this frame; that is a coherent field rather than noise (Leon's
cross-pointing transfer confirms the term belongs to the telescope), and it is 3.3× what the same
comparison gives on Leon's smaller sensor. For scale: the deflection at 10 R☉ — the outer edge of
cell 2's admitted-star window — is 0.175 ″ for L = 1.75 ″, so the quintic term in the outer field
is about 60 % of the signal there. A term that size left in the reference is not a rounding error
against the quantity cell 4 exists to measure.

**And the septic is excluded — §3.6 has the measurement Douglas asked for.** The in-sample case
for it looked strong (6.8 % of rms, 0.1605 → 0.1496 ″), which is exactly why an in-sample number
cannot be allowed to decide.

### 3.6 Is the quintic sufficient?  Yes — the septic term does not reproduce

Douglas, 2026-09-09: *"septic will require a lot of stars to fit and likely will not be very
stable. Is quintic sufficient?"*  Both halves of that are now measured
(`tools/husillos2026/hu_order_stability.py`).

**How many stars, and where.** The basis is a 2-D binomial expansion, so the parameter count runs
away fast, and the added parameters have leverage only where the radius is large:

| order | free parameters, both axes | Husillos stars at r/R ≥ 0.75 | Leon stars at r/R ≥ 0.75 |
|---|---|---|---|
| cubic | 18 | | |
| quintic | 40 | **268 of 2702 (10 %)**, reaching r/R 0.94 | 465 of 2515 (18 %), reaching 0.99 |
| septic | 70 | | |

**The septic buys 30 more parameters and hands them 268 stars to be constrained by** — nine stars
per added parameter in the only region where those parameters do anything. That is the concern,
quantified.

**Does the term reproduce?** The sharp test is not whether the model changes but whether the
*added term* comes back the same when fitted twice. Take `T = model(order N) − model(order N−2)`
on two independent fits, and compare `T_A` with `T_B`. Median over the stars at r/R ≥ 0.75, in
mas:

| pair | term | ǀT_Aǀ | ǀT_Bǀ | ǀT_A − T_Bǀ | ratio |
|---|---|---|---|---|---|
| Husillos halfA/halfB *(same stars — see below)* | quintic − cubic | 131.4 | 112.9 | 68.6 | 1.91 |
| Husillos halfA/halfB *(same stars)* | septic − quintic | 106.2 | 112.9 | 54.1 | 1.96 |
| **Leon Z1/Z4 (different stars)** | **quintic − cubic** | 21.9 | 21.9 | 11.2 | **1.95** |
| **Leon Z1/Z4 (different stars)** | **septic − quintic** | 14.8 | 17.5 | 14.9 | **0.99** |

**Only the Leon rows are a test.** Husillos' two halves are the same star field with the same
catalogue errors and the same blends, so any term that fits this field's quirks reproduces
between them — which is why both its rows read ~1.9 and neither means anything. Leon's Z1 and Z4
are different pointings with different stars, so only something belonging to the telescope comes
back.

And there the answer is unambiguous: **the quintic term reproduces at 1.95× its own scatter; the
septic term is exactly the same size as the disagreement about it (0.99).** That is the
definition of fitting noise, and it agrees with the transfer test in §3.2, where the septic buys
nothing out of sample (0.1310 ″ against the quintic's 0.1309 ″).

**One trap that had to be fixed before any of this meant anything.** The distortion polynomial
keeps *linear* terms, and those are degenerate with the plate solution's own scale and roll: two
fits can split that degeneracy differently, and the difference between their polynomials then
carries a large linear part that the pointing and plate scale silently absorb, which no star ever
feels. Raw, the septic − quintic difference reads **530 mas** at the field edge; the same quantity
taken from the two fits' own residuals reads **106 mas**, and the two routes correlate at only
0.30. Project the similarity out of every difference and they agree to **0.5 %** — which is now
printed as a standing cross-check, so the two routes cannot drift apart again.

**The answer: use the quintic. It is sufficient, and it is necessary.** Necessary because on this
frame it moves outer stars by 131 mas, six times the 22 mas it moves them on Leon's smaller
sensor, and it reproduces. Sufficient because the only independent evidence in the data says the
next term does not reproduce and does not transfer.

**The one caveat, stated rather than buried.** The septic is rejected on *Leon's* field, where the
septic term is only 15 mas; on Husillos' larger frame it would be ~106 mas, and Husillos' own
halves cannot test it because they share their stars. Nothing in the data supports a septic
there — but nothing directly excludes it either, and the way to close that is a second Husillos
pointing. There is exactly one place to get one: the three 12 August zenith captures Joe's
settings files announce and the drive does not hold (`23_20_19`, `23_22_35`, `23_25_31`, all
1.0 s). It is still the cheapest single request in this dataset, and §1 has already shown that
the one delivered capture from that folder is a 16-frame centred crop that cannot help. **But the
reduction should not wait for it: fit at quintic.**

### 3.7 A note on the night sets

The other night sets do not substitute for a second zenith field. `cal 8 deg` and the `10 deg` mosaic are near the horizon
and far shallower — 1 and 3 sources per frame at the threshold that finds 391 on the zenith frame
— and `Capture/00_54_32` stops stage 1 outright (*"No star centroids were found on frame 1"*).
That capture is **not** empty: a 1 px matched filter finds ~2600 sources in it, the same as the
zenith frame, so its stars are there and its per-pixel signal-to-noise is not. It needs a more
sensitive stage-1 configuration than the zenith preset — worth knowing for the prompt's step 1,
which asks for it to be plate-solved, but a different estimator makes it a poor partner in a
transfer test.

---

## 4. The mount: settled at the start, tracking at the end

Measured per frame on the full-frame zenith capture, 150 stars in every one of the 50 frames.

* **Sky is 1759.0 ADU in every frame** and the median star flux varies by 2.1 % rms. No cloud, no
  sky change.
* **Drift is 2.19 ″/min** — −0.0178 ″/s in x, +0.0319 ″/s in y — for a total path of 2.69 ″ over
  the 64.67 s capture.
* **There is no settling transient at the start and no slew at the end.** A settling mount shows a
  departure that DECAYS over the first several frames and a new slew one that runs away; the
  departures from a straight-line fit here have a median of 0.171 px and no trend at either end.
  The capture began 16 minutes after the previous one ended, which is ample on any mount.
* **Frame 0 is the single worst frame, and by an amount that does not matter.** It is the largest
  departure from the line at 0.387 px (+0.273, +0.275) against a median of 0.171 px and a
  next-worst of 0.289 px (frame 11) — so Douglas' instinct is right in direction, and the size is
  0.85 ″, well inside the ±0.5 px the whole-pixel aligner rounds away in any case (§6). On every
  other measure frame 0 is ordinary: same sky (1759.0 ADU), 150/150 tracked stars, PSF second
  moment 3.69 px against a 3.66–3.81 px run. That is worth stating because the prompt records
  genuinely anomalous frame 0s elsewhere in this dataset — Sn2's has a sky of 2475 ADU against
  1800, and the Sun capture's reads 14634 against 48337.
* An apparent **field rotation of −2.5 ″ over the capture** (slope −0.0387 ″/s, 5 σ) is present.
  Treat it as an upper bound on true rotation: a similarity fit absorbs part of the change in the
  distortion pattern as the field translates, and the translation here is ~1 px. At face value it
  smears a corner star by 0.07 px across the stack.

---

## 5. AM5 against AVX, on the same optic

`docs/STEP3_2026.md` ("What an AM5 would have done") predicts that a strain-wave mount carries
**larger** periodic error than a good worm — *"order ±10–20 ″ over a period of minutes"* — and
notes that the project had never measured one. This is the first measurement, and it does not
support the prediction over the timescale that matters.

Same estimator on both, imported from `ser_track.py` rather than re-implemented.

| | **AM5** (Husillos zenith) | **AVX** (Leon Z1–Z6) |
|---|---|---|
| fields | 1 | 6 |
| capture length | 64.7 s | 127.6–128.0 s |
| exposure / cadence | 1.0 s / 1.320 s | 4 s / 4.41 s |
| stars per frame | 150 | 150 |
| per-frame position noise | 0.148 ″ | 0.028–0.127 ″ |
| drift rate | 2.19 ″/min | 0.46–15.86 ″/min |
| **rms about a straight line, common 60 s window** | **0.300 ″** | **0.703–1.271 ″** |
| rms about a line, whole capture | 0.420 ″ | 1.270–3.441 ″ |
| **curvature ǀd²x/dt²ǀ** | **0.0011 ± 0.0002 ″/s²** | **0.0013–0.0046 ″/s²** |

Two corrections without which this comparison would be meaningless, both applied:

* **Window length.** Curvature accumulates as T², so Leon's 128 s fields would read several times
  worse than Husillos' 65 s one on identical mount behaviour. Hence the common-60 s row, and hence
  the curvature row, which is a property of the motion rather than of the window.
* **Settling.** Leon's zenith set is a six-pointing **mosaic**: every field starts right after a
  slew, and the AVX needs ~30 s to settle (`docs/MATRIX_2026.md`). Split at the middle, the AVX's
  **first 60 s reads 1.06–2.13 ″ (median 1.34) and its last 60 s 0.63–1.24 ″ (median 0.90)** — so
  a third of its excess is settle. Joe's capture had 16 quiet minutes before it, so the fair
  comparison is against the AVX's *last* 60 s. It is still 2.1–3× the AM5's.

**Neither dataset can measure a period**, and neither pretends to: a worm period is 480–640 s and
a strain-wave period is minutes, against windows of 65 and 128 s. What curvature gives is a lower
bound on amplitude at each assumed period, since |accel| ≤ Aω²:

| assumed period | AM5, A ≥ | AVX, A ≥ |
|---|---|---|
| 180 s | 0.91 ″ | 1.08–3.79 ″ |
| 300 s | 2.54 ″ | 3.00–10.5 ″ |
| 450 s | 5.70 ″ | 6.76–23.7 ″ |
| 600 s | 10.1 ″ | 12.0–42.1 ″ |

**The judgement:** at any assumed period the AM5's implied minimum amplitude is **1.2–4× smaller
than the AVX's**, and over a settled 60 s window it wanders less than half as far. Two things
strengthen the direction rather than weakening it — Joe's 1 s exposures average atmospheric image
motion *less* than Leon's 4 s ones, and Joe's per-frame noise is 1.2–5× larger, so both biases run
against the AM5 and it still wins. **On this evidence the AM5 is the better mount for an eclipse
run of this length, and the ±10–20 ″ figure in `STEP3_2026.md` should be marked as an unverified
prediction that the first measurement does not support.** It is not refuted either: a lower bound
cannot exclude a large amplitude sitting at an unlucky phase.

**What would settle it, and why it is not here.** Portland's other reading is rate scatter across
captures at **one pointing** (`docs/PORTLAND_2026-07-29.md` §4). `hu_am5_periodic.py` implements
it and refuses to report: neither candidate group could be shown to share a pointing. The mount's
telemetry says Dec = +90° throughout and is useless (§1), and the frames put `Capture`'s second
and third captures 5276 and 6616 px from its first. A 7-minute sequence at one verified pointing —
which the missing 12 August zenith trio would also have been — is what this needs.

---

## 6. The stacking master changes the yield by 12 %, and here it decided whether the field solved

Douglas' theory (2026-09-09): *"it is always better to use a good frame in the middle of the
series than the first one at the beginning, which has a much higher likelihood to have a defect."*

**The mechanism is real and larger than expected. The direction, on this capture, is the
opposite.**

`mee2024/stacker_implementation.py` aligns **every frame against `files[0]`** — the code says so
in as many words — and `add_img_to_stack` rounds each shift to a whole pixel
(`shift = (round(shift[0]), round(shift[1]))`, then `np.roll`). This capture's entire drift is
1.2 px, so which frame is the master decides how the run distributes over four integer cells.

Six stacks of the same capture, differing only in which frames were used:

| stack | frames | master | in the master's rounding cell | initial centroids | plate-solve |
|---|---|---|---|---|---|
| `with_f0` | 0–49 | 0 | 26/50 = 52 % | **3220** | 0.9 s |
| `drop_last` | 0–48 | 0 | 26/49 = 53 % | **3221** | 0.9 s |
| `full49` | 1–49 | 1 | 35/49 = 71 % | 2950 | **FAILED** |
| `master_f1` | 1–48 | 1 | 35/48 = 73 % | 2933 | FAILED |
| `master_f2` | 2–49 | 2 | 37/48 = 77 % | 2846 | FAILED |

(Counts are stage 1's `n centroids initial`, before the sanity and window-refinement cuts; those
drop 6–11 from each and change nothing here.)

Depth is not the driver: 50 frames and 49 frames off the same master give 3220 and 3221; 49 and 48
off master 1 give 2950 and 2933. **The master is**, and the yield tracks how *spread* the rounded
shifts are — the runs whose frames straddle cells 52/48 find 12 % more sources than the run with
77 % in one cell. Sub-pixel dither across the integer grid is doing real work here, which is
consistent with this capture having no dither of its own (stage 1 logs *"no dark-free hot-pixel
search: the field moved only 1.3 px between frames, under the 3 px needed"*).

The consequence was not cosmetic. **With 2950 centroids the solver ran 14.3 s and failed; with
3212 it matched 30 stars in 0.9 s.** Dropping frame 0 — the standing F23 rule — is what broke it,
reproducibly, on both attempts.

**What to take from this.** The rule "prefer a middle frame" is right that the master matters, and
understates how much. But the master is not chosen for being un-defective alone: on a capture that
drifts less than a pixel it is also choosing the dither pattern. So F23 should be framed as *the
master is a run parameter that must be selectable and recorded*, not as *frame 0 is bad*. On this
capture frame 0 is measurably fine (§4) and was the best master of the three tried. A defective
first frame remains a real hazard elsewhere in this dataset — Sn2's frame 0 and the Sun capture's
frame 0 both are — which is the argument for making the master selectable rather than for a fixed
rule about which index to use.

---

## 7. What to do next, in order

1. **Ask Joe for the three missing 12 August zenith captures** (`23_20_19`, `23_22_35`,
   `23_25_31`). They are the only route to a second star-rich full-frame field, and they would
   confirm the septic verdict on Husillos' own optics, the plate-scale repeatability, and the AM5
   rate scatter at one pointing — three open items for the price of one request. None of them
   blocks the reduction.
2. **Fit cell 4's reference at quintic**, not cubic (§3.5), and **do not use the septic**
   (§3.6) — its term is the same size as the disagreement about it on the one independent test
   in the data. Item 1 would confirm that directly rather than by transfer from Leon.
3. **Do not carry Leon's plate scale to Joe's station**: the trains differ by 425 ppm (§2).
4. ~~Locate the site.~~ **Done** (§0): Husillos, +42.09293°, −4.52702°, 743 m. Everything above
   is refraction-safe only because both fields are near the zenith; nothing at 8° or 10° altitude
   is, and the eclipse field at 8.6° altitude certainly is not. The corrections can now be
   switched on for every non-zenith field.
5. **F23**: make the stacking master a recorded run parameter (§6), and note in its issue that the
   observed effect is dither, not defect.
6. **Ask Joe what gain and exposure the missing zenith captures used** (§7b), and record for 2027
   that a night calibration field must be shot at a gain and subframe length that put the read
   noise below the sky. Husillos' zenith was read-noise limited by 11× and lost 2.5× of SNR to
   the gain setting alone.

---

## 7b. Are the zenith frames underexposed?  Yes — read-noise limited by 11×

Douglas, 2026-09-09: the Husillos zenith captures are **1.0 s at gain 0**, against Leon's 4 s at
gain 101 and Portland's 4 s through a similar FRA500 on a similar Sony sensor.
`tools/husillos2026/hu_exposure.py` measures it, from the raw SER and FITS, with no dark frames
and no camera datasheet.

**"Underexposed" has one precise meaning here** and it is not that the picture looks dark. A
subframe is long enough when the **sky shot noise dominates the read noise**. Past that point the
sensor has stopped contributing and only photons matter, and a longer subframe buys nothing a
longer stack cannot. Below it, every extra frame pays the read noise again.

### How it was measured without a dark

* **Noise between consecutive frames, not across one frame.** A single frame's spatial scatter
  contains the fixed pattern — hot pixels, PRNU, amp glow — which is identical in every frame and
  therefore not noise at all for a stack. `std(frame_k − frame_k+1)/√2` removes all of it.
* **Sigma-clipped mean and standard deviation, never a MAD.** These are 16-bit integers with a
  few ADU of noise, so a MAD can only land on multiples of 1.4826/√2 = 1.048 ADU. The first pass
  of this tool returned exactly 4.19 ADU for four different captures and exactly 7.34 for three
  more — the estimator's grid, not the sensor.
* **The bias never has to be known.** Two exposures at the same gain *and the same offset*
  difference it away. Joe's `Capture` folder has 1.0 s and 0.315 s at gain 125, offset 50, three
  minutes apart. **The zenith captures cannot be paired with it** — they were shot at offset 220,
  and the offset *is* the bias — so the conversion gain is measured at gain 125 and carried to
  gain 0 by the 0.1 dB step these cameras use (+12.5 dB = ×4.217).

Measured: **EGAIN = 0.2466 e-/ADU at gain 125 → 1.04 e-/ADU at gain 0** (±20 %, dominated by the
5.45 ADU signal difference). Sanity: Leon's FITS header states 0.2608 e-/ADU at gain 101, which
scales to 0.834 at gain 0 for the IMX571, and a ~51 ke- full well over 65535 ADU implies ~0.78.
The three agree to the accuracy claimed.

### The numbers

| capture | exp | gain | read noise | sky | sky/read² | verdict |
|---|---|---|---|---|---|---|
| **Husillos zenith 08-13** | **1.0 s** | **0** | **4.73 e-** | **1.96 e-/px** | **0.09** | **read-noise limited by 11×** |
| Husillos `cal 8 deg` | 1.0 s | 0 | 4.73 e- | 1.96 e-/px | 0.09 | read-noise limited by 11× |
| Husillos `Capture` | 1.0 s | 125 | 1.38 e- | 1.96 e-/px | 1.03 | sky limited, just |
| **Leon Z1 / Z4** | **4 s** | **101** | **2.12 e-** | **7.85 e-/px** | **1.74** | **sky limited — correctly exposed** |
| Husillos eclipse Sn2 | 0.315 s | 0 | 4.73 e- | 238 e-/px | 10.6 | sky limited (the corona) |

The **sky rate is 1.96 e-/px/s** (±20 %), measured directly from the exposure pair rather than by
differencing two nearly-equal noises. It was measured at the `Capture` pointing, which is lower
in the sky than the zenith, so it is an **upper bound** for the zenith — the real ratio is worse
than 0.09, not better. Both stations share the FRA500 + 0.7× and 3.76 µm pixels, so one sky rate
in e-/px/s serves both.

**Two settings each cost about the same, and they multiply.** Gain 0 is the low-conversion-gain
mode: the Zeus 455M reads 4.73 e- there and 1.38 e- at gain 125, a factor 3.4 in noise thrown
away for nothing. And 1.0 s subframes pay that read noise fifty times in fifty seconds.

### What it cost

Stack variance per pixel is `(T/t)·read² + sky_rate·T`, so with the sky this faint the read term
is almost the whole budget:

| | stack noise | star signal | |
|---|---|---|---|
| Husillos, 50 × 1.0 s at gain 0 | 34.9 e- | 50 s | |
| Leon, 30 × 4.0 s at gain 101 | 19.3 e- | 120 s | |

**Leon collects 2.4× the star signal with 0.55× the noise: 4.3× the SNR.** And that is what the
depth shows, on fits run at identical settings:

| | matched stars | median G | stars at G 12–13 | sky area | faint stars per Mpx | astrometric rms |
|---|---|---|---|---|---|---|
| Husillos zenith | 2680 | 11.71 | 991 | 61.2 Mpx | **16** | 0.161 ″ |
| Leon Z1 | 2516 | 12.15 | 1389 | 26.1 Mpx | 53 | 0.093 ″ |
| Leon Z4 | 2977 | 12.24 | 1800 | 26.1 Mpx | **69** | 0.095 ″ |

Husillos has **2.3× more sky** and still matches fewer faint stars: per unit area Leon reaches
**4.3× more stars at G 12–13** — the same factor as the SNR, which is the consistency check that
the read noise really is the whole story. The astrometric residual follows: 0.161 ″ against
0.093 ″, and Husillos' residual climbs with magnitude (0.123 → 0.182 ″ from G < 10 to G 12–13)
where Leon's is flat (0.102 → 0.098 ″). **Leon is not photon-starved at G 13; Husillos is.**

### What would have fixed it, at the same 50 s of sky time

| | stack noise | SNR |
|---|---|---|
| gain 0, 1.0 s (what was shot) | 34.9 e- | ×1.00 |
| gain 0, 4.0 s | 19.4 e- | ×1.79 |
| **gain 125, 1.0 s** | **13.9 e-** | **×2.51** |
| **gain 125, 4.0 s** | **11.0 e-** | **×3.16** |

**The gain setting alone was worth 2.5×, for free, with no change to the schedule.** And there
was headroom for it: the brightest star in the zenith frame peaks at **40940 ADU of 65535 and
nothing in the frame saturates — zero saturated pixels**. Dynamic range was never the constraint.
At gain 125 the brightest few stars would clip, which for astrometry on G < 13 costs a handful
and no more.

### What this does and does not change

It does **not** invalidate anything above. The plate scale (§2) is a ratio of angles and is
unaffected by depth; the order verdict (§3) rests on Leon's cross-pointing test, where the
exposure was correct; the mount comparison (§5) already ran with Husillos' larger per-frame noise
counted against it, and the AM5 still won.

It **does** explain why Husillos' residual is 0.161 ″ against Leon's 0.093 ″ — that is exposure,
not optics — and it sets what to ask for. **If Joe still has the three missing 12 August zenith
captures, ask what gain and exposure they used**; and for 2027 the rule is one line: *on a
night calibration field, choose the gain that puts read noise below the sky, then choose the
subframe length that keeps it there.* On this rig at the zenith that is gain 125 and ≥ 1 s, or
gain 0 and ≥ 12 s.

---

## 8. Where the stacked frames are

Stage 1 writes the stack beside the centroid zip, not into it:

    <run>/CENTROID_OUTPUT<timestamp>/STACKED<timestamp>.fit          uint16,  122 MB
    <run>/CENTROID_OUTPUT<timestamp>/STACKED_FLOAT<timestamp>.fit    float32, 245 MB

with `CentroidsStackGood*.png`, `TWOD_RESIDUALS*.png`, `USEDSTARS*.png`,
`triangle_matches.png` and `LOG*.txt` alongside.

Everything for cell 4 now lives under **`D:\MEE2024 output\MEE_output\RECORD\husillos2026\`**
(`zenith_order/`, `track/`, `mount/`), and the one stack that matters is copied out under a name
that says what it is, with an index:

    RECORD/husillos2026/stacks/husillos_zenith_00_00_21_50frames_uint16.fit
    RECORD/husillos2026/stacks/husillos_zenith_00_00_21_50frames_float32.fit
    RECORD/husillos2026/stacks/husillos_zenith_00_00_21_centroids.png
    RECORD/husillos2026/stacks/husillos_zenith_00_00_21_residuals.png
    RECORD/husillos2026/stacks/00README.md

That is all 50 frames of `2026-08-13/zenith/00_00_21.ser`, 9576 × 6388, solving at RA 281.7427°,
Dec +50.2092°, roll 325.902°, 2.2064323 ″/px. Use the float32 file for anything that re-measures
the pixels; the uint16 one is the same stack rounded. `00README.md` lists the other nine runs and
which of them plate-solve.
