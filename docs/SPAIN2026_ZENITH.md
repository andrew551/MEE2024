# Spain 2026, the zenith fields: focal length, distortion order, and the AM5

**Date:** 2026-09-09. Matrix cell 4 (Joe Izen, Spain 2026). Everything below is measured on
`G:\Joe Izen Spain 2026` and, for the controls, on `G:\Leon Aug 2026`, with `v1.4.0-dev`.

Four questions were asked (Douglas, 2026-09-09): is the cubic enough on this full frame or is a
quintic needed; is the focal length the same as Leon 2026's; can the ZWO AM5's periodic error be
judged against Leon's Celestron AVX; and are the captures settled at the start and still tracking
at the end.

**León is in Spain and Joe's site is south of it: cells 3 and 4 are two stations at the same
eclipse.** Leon's zenith mosaic was shot 48 minutes after Joe's zenith capture, on the same night,
through the same telescope model with the same reducer and the same 3.76 µm pixels. Reduced here
at identical settings it is the control for every question below, and that is the single reason
these answers are as sharp as they are.

Tools: `tools/spain2026/ser_track.py` (per-frame star tracking), `sp_zenith_order.py` (stage 1 and
2 at three orders, the radial diagnostic, the transfer test), `sp_mount_compare.py` (AM5 against
AVX with one estimator), `sp_am5_periodic.py` (the rate-scatter method — it did not deliver; §5).

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
| Spain zenith, 12 Aug 22:00 UTC | 12 Aug | 50 × 1.0 s | 2680 | 0.1605″ | **2.2064323 ″/px** | **351.498 mm** |
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

## 3. Cubic or quintic: quintic at least, and the septic is not yet excluded

An in-sample rms cannot answer this — more parameters always fit better — so four things were
measured instead. Free fits, same settings as above, on the full-frame zenith stack and on Leon's
two pointings.

### 3.1 How much the higher order actually moves the stars

Per star, on the stars both fits kept: the difference between the order-N and order-N+2 fitted
positions.

| field | half-diagonal | quintic − cubic, all | quintic − cubic, **outer 20 %** | septic − quintic, outer 20 % |
|---|---|---|---|---|
| **Spain zenith** | **12699 ″ (3.53°)** | 0.066 ″ rms | **0.108 ″ rms** (max 0.41 ″) | **0.136 ″ rms** |
| Leon Z1 | 8294 ″ (2.30°) | 0.022 ″ rms | 0.033 ″ rms | 0.021 ″ rms |
| Leon Z4 | 8294 ″ (2.30°) | 0.023 ″ rms | 0.035 ″ rms | 0.024 ″ rms |

Spain's field radius is 1.53× Leon's and the quintic's effect is **3.3× larger**, which is the
direction and roughly the size a radial term of that order predicts.

### 3.2 Out of sample, on a different star set

One field fitted free, then frozen into another with only the constant free. **Two halves of one
capture share their stars**, so anything that fits this field's own per-star quirks — a catalogue
position error, a blend — transfers between them and the test passes it. Leon's Z1 and Z4 are two
pointings 6.4° apart on the same night through the same optic, so only something belonging to the
**telescope** transfers.

| order | Spain halves (same stars) | **Leon cross-pointing (different stars)** |
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
| **Spain zenith** (`with_f0`) | **5.83** | **9.90** | **9.61** |
| Spain, `drop_last` | 6.23 | 9.95 | 9.46 |
| Leon Z1 | 3.59 | **1.06** | 1.15 |
| Leon Z4 | 3.44 | 3.57 | 2.75 |

Leon's standard errors are the *smaller* of the two (2.6–7.0 mas against Spain's 3.8–14.9), so
this is not a sensitivity difference: **Spain's radial structure is about three times more
significant than Leon's at every order.** Leon Z1 goes flat at quintic; Leon Z4 does not, which
is worth recording rather than rounding off — one of the two Leon pointings still shows 3.6 σ.

One caveat, and it bites: a least-squares residual is orthogonal to the fitted basis by
construction, so binned radial means *must* alternate in sign, and their amplitude is **not**
comparable across orders — the rise from 5.8 to 9.9 σ between Spain's cubic and quintic is not
evidence that the quintic is worse. What is comparable is Leon against Spain at the same order
with the same binning, and there the difference is stark.

### 3.4 What the closed cells say

Cell 2 (Mexico Station 1) is the **same IMX455 sensor**, 9576 × 6388, at 1.84847 ″/px — a
half-diagonal of 10637 ″ (2.955°). It needed the quintic, and `s1_septic_test.py` confirmed the
quintic sufficient out of sample. **Spain's field is 19 % larger in angle than that**, at 3.528°.

### 3.5 The answer

**Use the quintic. The cubic is not sufficient.** The quintic moves fitted star positions by
0.108 ″ rms in the outer fifth of this frame; that is a coherent field rather than noise (Leon's
cross-pointing transfer confirms the term belongs to the telescope), and it is 3.3× what the same
comparison gives on Leon's smaller sensor. For scale: the deflection at 10 R☉ — the outer edge of
cell 2's admitted-star window — is 0.175 ″ for L = 1.75 ″, so the quintic term in the outer field
is about 60 % of the signal there. A term that size left in the reference is not a rounding error
against the quantity cell 4 exists to measure.

**The quintic-versus-septic question is open and this dataset cannot close it.** On Spain's frame
the septic still gains 6.8 % of in-sample rms (0.1605 → 0.1496 ″) and moves outer stars by
0.136 ″ — as much as the quintic does — where on Leon's frame it gains 1.5 % and transfers no
better. Settling it needs **a second star-rich full-frame Spain field at a different pointing**,
and there is exactly one place to get it: the three 12 August zenith captures Joe's settings files
announce and the drive does not hold (`23_20_19`, `23_22_35`, `23_25_31`, all 1.0 s). **Asking Joe
for those three files is the cheapest way to settle the order for cell 4**, and §1 has already
shown that the one delivered capture from that folder is a 16-frame centred crop that cannot help.

The other night sets do not substitute. `cal 8 deg` and the `10 deg` mosaic are near the horizon
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

| | **AM5** (Spain zenith) | **AVX** (Leon Z1–Z6) |
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
  worse than Spain's 65 s one on identical mount behaviour. Hence the common-60 s row, and hence
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
captures at **one pointing** (`docs/PORTLAND_2026-07-29.md` §4). `sp_am5_periodic.py` implements
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
   `23_25_31`). They are the only route to a second star-rich full-frame field, and they settle
   the quintic-versus-septic question, the plate-scale repeatability, and the AM5 rate scatter at
   one pointing — three open items for the price of one request.
2. **Fit cell 4's reference at quintic**, not cubic (§3.5). Record the septic as an open
   sensitivity until item 1 lands.
3. **Do not carry Leon's plate scale to Joe's station**: the trains differ by 425 ppm (§2).
4. **Locate the site.** Everything above is refraction-safe only because both fields are near the
   zenith; nothing at 8° or 10° altitude is, and the eclipse field is not.
5. **F23**: make the stacking master a recorded run parameter (§6), and note in its issue that the
   observed effect is dither, not defect.
