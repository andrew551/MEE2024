# Six optical trains on one footing — and the sign that made one of them look 16× worse

**Date:** 2026-08-26. Every number here was measured this week by re-reducing raw or
stage-1 data with `v1.4.0-dev` at identical settings. Nothing is quoted from a summary.

**This document corrects a conclusion currently in the repository.** `LEON_2026-08-11.md`
§18.11 states that *"Leon carries 16x the cubic of Bruns' FRA500 — the same telescope with
the same reducer … the strongest evidence yet that the train is misassembled."* That is an
artefact of a dropped sign. Measured on the very frames Bruns' Table 1 row was derived from,
Leon and that telescope agree to **0.4 %**. §5 below has the arithmetic.

A second correction, of attribution: the FRA500 in Bruns' Table 1 is **Kenneth Carrell's**
telescope and camera, imaged in Texas. Bruns' paper reports the measurement; the instrument
is Carrell's. §18.11 and `ROADMAP.md` §3a both call it "Bruns' FRA500".

---

## 1. Method, and why the sign is the whole story

Distortion is compared as the **signed isotropic cubic amplitude** `k`, in arcsec per deg³:
write the cubic displacement field in complex form around a circle and take the `e^{it}`
component (§18.6's decomposition; `B` is the isotropic radial term). Two properties make it
the right currency:

- **It is independent of pixel size and sensor size.** `rad/px³` is not — it scales with
  pixel size cubed — and "arcsec at the corner" is not, because sensors reach different
  angles. Both traps are already flagged in §18.11; this is the quantity that avoids them.
- **It carries a sign**, which the magnitude does not.

The extractor is validated against §18.8's published figures: it reproduces Leon's
`d(3000) = 3.1048″` as 3.1013 (0.1 %), both nights' scatter to 0.02 percentage points, and
the night-to-night gap as +4.98 % against the published +4.84 %.

**Why the sign cannot be dropped.** MEE reports in an angular gauge, the published tables in
the tangent plane, and `k_TAN = k_MEE + gauge` with the gauge **positive and of order 0.4**.
For an optic whose cubic is *smaller* than the gauge term — which is the FRA500's entire
selling point — using `|k_MEE|` does not scale the answer, it **reflects it across the gauge
constant**: −0.4986 + 0.41 = −0.089 becomes +0.4986 + 0.41 = +0.909. A −0.09 turns into a
+0.91 and a factor of 16 appears out of nowhere.

## 2. The gauge constant, measured rather than assumed

§18.11 records the constant as +0.41 by one method and +0.4587 ± 0.0106 by another, ~12 %
apart, and says resolving it is an open item. Two independent measurements now exist, both
from frames where Astrometrica's own solution sits alongside the MEE reduction of the *same
pixels*:

| source | fields | k_TAN − k_MEE |
|---|---|---|
| TV-85 + 0.8× (`I:\Don Bruns TV-85 calibration`, Sheliak + Rasalhague) | 2 | **+0.341 ± 0.042** |
| NP101is + ASI2600 (`I:\Don Bruns 2024`, HIP 29696/31096/32740/33018) | 4 | **+0.397 ± 0.036** |
| theoretical tan-minus-arc term, θ³/3 | — | **+0.3655** |

The two bracket the theoretical value and agree within their scatter. **Use +0.37 ± 0.04**,
and keep §18.11's rule of not quoting a conversion better than ~10 %.

Reading Astrometrica's logs requires one practical warning: it continues each polynomial
with a **bare carriage return** and terminates with CRLF. Read in text mode, Python treats
the bare CR as a line break, silently truncating the polynomial after its linear term — every
cubic coefficient reads as exactly zero, and the comparison returns a plausible-looking table
of nonsense. Read the file as bytes.

## 3. Six trains, one currency

Gauge +0.37 throughout. `sd` is field-to-field scatter of `k`.

| configuration | n | EFL | half-diag | k_MEE | k_TAN | corner | sd |
|---|---|---|---|---|---|---|---|
| TV-85 + 0.8× + ASI1600 (2022) | 2 | 473 | 1.75° | −7.21 | −6.84 | −36″ | — |
| 65PHQ + ASI533, London (2026-08) | 16 | 415 | 1.10° | −1.425 | −1.055 | −1.42″ | 1.9 % |
| 65PHQ + ASI294, Leakey (2024-04) | 22 | 415 | 1.61° | −1.311 | −0.941 | −3.93″ | 1.2 % |
| NP101is + ML8051, Bruns (2017) | 29 | 543 | 1.19° | +0.402 | +0.772 | +1.31″ | 21 % |
| NP101 + ASI2600, Bruns (2024) | 10 | 543 | 1.49° | +0.190 | +0.560 | +1.85″ | 16.5 % |
| FRA500 + 0.7× + ASI1600, Carrell (2024) | 3 | 364 | 1.75° | −0.496 | −0.126 | −0.67″ | 2.1 % |
| FRA500 + 0.7× + ASI2600, Portland (2026-07) | 6 | 351 | 2.31° | −0.464 | −0.094 | −1.15″ | 2.0 % |
| FRA500 + 0.7× + ASI2600, Leon 08-11 | 6 | 351 | 2.30° | −0.510 | −0.140 | −1.71″ | 1.3 % |
| FRA500 + 0.7× + ASI2600, Leon 08-12 | 6 | 351 | 2.30° | −0.486 | −0.116 | −1.42″ | 1.3 % |

## 4. Bruns' Table 1 checks out on magnitude; its signs do not survive

Converting Table 1's `rad/px³` independently and comparing:

| row | Table 1 \|k_TAN\| | ours | agreement |
|---|---|---|---|
| TV-85 + 0.8× | 6.73 | 6.87 (MEE) / 6.86 (Astrometrica) | **2 %** |
| NP101 native + ASI2600 | 0.595 | 0.587 (MEE) / 0.598–0.618 (Astrometrica) | **2 %** |
| 65PHQ native + ASI294 | 0.884 | 0.941 | 6 % |
| FRA500 + 0.7× + ASI1600 | 0.057 | 0.086–0.155 | **not testable** — see below |

Three of four reproduce to within 6 %, and the geometry reproduces even better: EFL 364 vs
our 363.5, 412 vs 415.0, 549 vs 543; FOV 167′ vs 167.3′. Table 1 is a sound compilation.

**But its sign column is not a reliable physical statement.** Astrometrica reports the TV-85's
X-cubic as −3.289E-15 and its Y-cubic as +3.150E-15 for the same image — not a real asymmetry,
because its *linear* terms are also opposite (+8.040E-6 and −8.040E-6): the frame is mirrored.
Normalising cubic by linear gives −4.09E-10 and −3.92E-10 per px², both negative, so the
distortion is unambiguously barrel. Table 1 lists **+3.2E-15**, which is the raw *Y*
coefficient. The same happens on the NP101 row. So the published sign is the sign of a raw
polynomial coefficient in a mirrored frame, and reading it as barrel-versus-pincushion
requires knowing each entry's axis convention.

**And the FRA500 row cannot be tested this way at all.** Its `k_TAN` is *smaller than the
gauge constant*, so the ±0.04 uncertainty in the constant is ±27 % of the answer and the
choice between 0.341 and 0.397 moves it from −0.155 to −0.086. Any statement of the form
"the FRA500 is N times better" inherits that.

## 5. The correction: Leon is not 16× Carrell's FRA500

The comparison that needs no gauge conversion at all, because both sides are in the same
gauge:

    Carrell FRA500 + 0.7x (the source of Table 1's row)   k_MEE = -0.4959
    Leon    FRA500 + 0.7x (mean of both nights)           k_MEE = -0.4981
    difference                                                    0.4 %

Same reduction, same settings, same extractor. Whatever conversion is applied afterwards
applies to both equally, so **no factor of 16 is possible**. §18.11's 0.909 for Leon is
exactly `|k_MEE| + 0.41`.

What follows for the conclusions built on it:

- *"the strongest evidence yet that the train is misassembled"* — **withdrawn.** The cubic
  does not distinguish these rigs.
- *"A correctly assembled FRA500 + 0.7x is 10x better than the NP101 native"* — the direction
  survives (§6 below) but the factor needs re-deriving with signs.
- *"Closing 16x would shrink every transfer error in this document by the same factor"* —
  there is no 16× to close.

**The misassembly evidence that does survive is the PSF, not the cubic.** Over a matched
angular range (0–0.3° to 1.2–1.45°), Portland's blur grows **1.32×** where Carrell's grows
**1.03×**, and Portland prefers the coma law `A + C r²` over the defocus law `A + B r⁴` —
reproducing §9.3's finding on the same rig, and matching F19's diagnosis of a back-focus
error that does not allow for the filter glass. That is unaffected by any of the above.

## 6. Which train is actually best, on Bruns' own metric

His V5 §"Analysis" argues only the component of cubic distortion **parallel to the deflection**
matters, and that placing the Sun in a corner minimises it. Applying that criterion — Sun in
one corner, stars inside 2 R⊙ excluded, usable where the parallel component stays below the
deflection:

| configuration | field | k_TAN | corner | **usable (C/D < 1)** |
|---|---|---|---|---|
| FRA500 + 0.7× + ASI2600 | 3.83 × 2.56° | −0.117 | −1.43″ | **6.46 sq°** |
| NP101 native + ASI2600 | 2.48 × 1.66° | +0.560 | +1.85″ | 2.69 sq° |
| 65PHQ native + ASI533 | 1.56 × 1.56° | −1.055 | −1.42″ | ~1.6 sq° |

**The FRA500 + 0.7× wins by 2.4× and 4×**, and the result is robust to the gauge uncertainty
(6.10–6.91 sq° across the full measured range, the others barely moving).

Note the corner distortions are nearly equal — 1.43, 1.85, 1.42″. **Ranking these trains on
worst-case distortion says they are interchangeable, and that is the wrong metric.** The
FRA500's advantage is that its low coefficient lets it keep usable stars much further out, so
its extra sky is usable rather than merely present.

## 7. Precision is absolute, not fractional

The per-field scatter column in §3 spans 1.2 % to 21 %, which invites the reading that some
campaigns measured the cubic far better than others. In arcsec they are equals:

| | cubic | per-field sd | **absolute** | on the mean |
|---|---|---|---|---|
| Bruns 2017, NP101is (29 fields) | 0.3555″ @ 1650 px | 20.8 % | 0.0740″ | **0.0140″** |
| Leon, FRA500 (12 fields) | 3.1048″ @ 3000 px | 1.31 % | 0.0407″ | **0.0117″** |

**A ratio of 1.19 on the quantity that reaches the deflection.** The percentage difference is
almost entirely that Bruns' optic has 8.7× less distortion to measure. Consequences:

- **Never rank instruments by fractional cubic stability.** It flatters the instrument with
  the most distortion.
- **Leon's 4.84 % night-to-night is 0.150″; the same fraction on Bruns' rig would be 0.017″.**
  Choosing a low-distortion optic shrinks every transfer error proportionally — which is the
  real argument for the FRA500, and a better one than the retracted 16×.

## 8. Stability: one measurement, and several non-detections

| | night-to-night | error | significance |
|---|---|---|---|
| Leon FRA500 (2026-08) | **+4.69 %** | ±0.74 % | **6.4 σ** |
| Leakey 65PHQ (2024-04) | −1.40 % | — | 18 sets vs 4 |
| Bruns NP101is (2017-08) | +1.66 % | **±7.99 %** | **0.2 σ** |

Bruns' 2017 night-to-night *looks* better than Leon's, and it is not evidence of anything:
its 68 % range is **−6.3 % to +9.6 %**, which contains Leon's 4.69 % comfortably. Its
per-field scatter is 21 % against Leon's 1.3 %, for the reason in §7. **Leon remains the only
campaign that has measured night-to-night cubic change to useful precision.**

**Bruns' §5.1 stability claim (±0.9 % and ±2.6 %) is a within-season figure**, measured over
nights in one spring at night focus both times. It does not cover the transfer the workflow
actually makes — night focus to daytime focus, 121–129 focuser steps at Leon — which no
campaign has measured.

## 9. Calibration has a shelf life: 8.2 % in 28 months

The same 65PHQ, at the same focal length to 0.02 % (415.0 vs 414.9 mm), measured twice:

| | k_MEE | ovality | ovality PA | trefoil A / E |
|---|---|---|---|---|
| Leakey, 2024-04 (ASI294) | −1.3165 | 2.05 % | **18.5°** | 0.73 / 5.55 % |
| London, 2026-08 (ASI533) | −1.4249 | 3.03 % | **64.5°** | 0.97 / 2.61 % |
| change | **+8.2 %** | | **+46°** | |

**The shape changed, not only the amplitude.** A power or spacing change moves `|B|` and
leaves the angular structure alone; here the ovality position angle rotated 46° and the
trefoil terms reorganised. That is the signature of an optical element shifting or
decentring — two years of use, transport to Texas and back, thermal cycling.

Three alternatives were tested and eliminated:

- **Higher-order absorption** from the different field radii (1.61° vs 1.10°): adding quintic
  terms moves the cubic by 0.1 %.
- **Tolerance mismatch** (the London series was reduced at 1.0, ours at 0.2): under 1 %.
- **Sensor tilt from the camera swap**: both of tilt's signatures are *smaller* in 2026, not
  larger — quadratic/cubic ratio 0.085 → 0.032, and the one-sided m=1 FWHM gradient
  23.8 % → 10.0 %. (Note also that the 65PHQ is a Petzval with an integral flattener, so the
  camera distance sets focus, not the aberration correction — the F19 back-focus mechanism
  does not apply to this design.)

**Consequence:** a telescope characterised well in advance does not stay characterised. This
does not affect Leon, calibrated the night before, but it bears on any plan that treats
Table 1 as a durable equipment guide, and on pre-season characterisation for 2027.

## 10. Focus sensitivity, measured for the first time

`I:\Leakey 2024 data\zenith 2` contains an unplanned focus sweep: 18 sets over 146 minutes
with the focuser moved once, 16110 → 16040, recorded in the FireCapture settings files.

| | |
|---|---|
| plate scale | **+8.9 ppm per step** (623 ppm over 70 steps; EFL scatter within each group 0.00–0.01 mm) |
| cubic | **+0.0138 % ± 0.0075 % per step** (0.97 % over 70 steps, 1.8 σ) |

A temperature control rules out thermal drift: within each focus group a 4–5 °C swing moves
the plate scale by a few ppm, against 623 ppm between groups.

Compare §9.2's Leon estimate of **0.64 % of cubic per focuser step** — 46× larger. Two
reasons not to read that as a contradiction. §18.6 already records that Leon's 8 recorded
steps sit *inside* the EAF's 15-step backlash and "cannot calibrate the slope"; this
measurement uses 70 steps with 8–10 sets per group. And **this is a 65PHQ with no reducer**,
where F19's mechanism — focusing changes the objective-to-reducer distance — does not apply.

So this is a **baseline, not a refutation**: without a reducer, focus barely moves the cubic.
It makes the reducer the prime suspect for Leon's focus sensitivity rather than the focuser or
the mount, and gives §9.3's sweep a control to be measured against.

## 11. Smaller results, recorded so they are not re-derived

- **Day–night plate scale.** Bruns 2017 **−524 ppm** (night 2.087827 → day 2.086732, both
  corrections-on, 15 night fields); Leon **−650 ppm** against 08-12 and **−850** against
  08-11. Same sign, same order — a property of the experiment class, not of one rig. Both are
  ~100× the ~6 ppm import tolerance F&L's eq. 23 implies, which is the quantitative case for
  re-measuring the scale during totality.
- **Within-night plate-scale drift**, Bruns 2017: **+2.78 and +1.52 ppm/min**, monotonic on
  both nights, alt/az fixed so refraction is constant. "The night's plate scale" is not a
  well-defined constant at the ppm level.
- **Reported standard error is optimistic.** On the same data the reported
  `platescale_relative_uncertainty` averages **3.4 ppm** while field-to-field scatter is
  **20–29 ppm**, and 12–14 ppm after removing the drift — a factor of 4–6.
- **Old-versus-new reduction agreement.** Re-reducing Bruns' L/R eclipse-day calibration
  reproduces the 2024 rms to 2 % and the day−night transfer to **14 ppm** (−510 vs −524).
  A **+30 ppm** plate-scale offset between reduction generations remains unexplained: it is
  not the reference set (−0.4 ppm), not F15's windowed centroid (+1.1 ppm), and not the
  tolerance. **Two candidates remain untested**: the refraction fix of §11.1, and the
  calibration frames themselves — the 2024 run used darks and this one did not, which an
  earlier draft of this list overlooked. See `CALIBRATION_FRAMES.md`.
- **Averaging both 2017 nights changes nothing** (−0.4, +0.2, +0.1 ppm on the three L/R
  variants), because they are statistically indistinguishable — the opposite of Leon, where
  §18.9 warns against averaging because the nights differ at 6 σ.
- **Vixen GP2 periodic error** (Leakey zenith 2): a component of **±6.1″ near 200 s**, which
  is 598/3 — the third harmonic of the 144-tooth worm. A pure worm-period sinusoid cannot
  produce the observed curvature (3× short). The **fundamental is unconstrained** (bootstrap
  0.0–9.3″): the 18 sets are a mosaic at ~290 s spacing, ~2 samples per cycle. The London
  HEQ5 Pro measurement of §1.4 (±7.7″ at 537 s) succeeded because that set repeated 9
  positions at ~127 s — 4.2 samples per cycle. Not comparable, and the GP2's reputation for
  low periodic error is neither confirmed nor refuted.

## 12. The Leon train against the 2017 experiment, head to head

The natural question once the 16x is withdrawn: how does the Leon rig actually compare with
the instrument that produced the best eclipse measurement ever made?

**Distortion — Leon is better, not merely comparable.**

| | field | k_TAN | at Bruns' corner (1.19 deg) | usable sky |
|---|---|---|---|---|
| Bruns 2017, NP101is + ML8051 | 1.91 x 1.43 deg | +0.772 | **+1.32"** | 1.96 sq deg |
| Leon 2026, FRA500 + 0.7x + ASI2600 | 3.83 x 2.56 deg | -0.128 | **-0.22"** | **6.30 sq deg** |

Six times less cubic per unit angle, and 3.2x more usable sky on Bruns' own criterion. The
larger field does not merely rescue the Leon rig; it wins on both terms at once.

**Image quality — the coma is real, and it is not worse than 2017 over the same field.**
Matched annulus by annulus over Bruns' own radial range:

| r (deg) | Bruns 2017 | Portland | ratio |
|---|---|---|---|
| 0.0-0.30 | 3.21" | 4.55" | 1.42 |
| 0.3-0.60 | 3.30" | 4.70" | 1.42 |
| 0.6-0.90 | 3.50" | 4.96" | 1.42 |
| 0.9-1.19 | 3.77" | 5.45" | 1.45 |
| **growth** | **1.175x** | **1.199x** | |

The field-dependence is the same to 2 %. The FRA500's coma only becomes conspicuous *beyond*
Bruns' field, reaching 1.32x by 1.45 deg -- territory the 2017 rig never sampled. Ellipticity
behaves better on the FRA500 too: it falls with radius (0.137 -> 0.091) where Bruns' rises
(0.130 -> 0.220).

So: **the Leon train's optics are at least as good as the 2017 rig's over comparable field,
and its distortion is substantially better** -- with the caveat that it is asked to work over
twice the radius, where the back-focus coma does bite.

Two limits on that. Portland's absolute FWHM is 1.42x worse at *every* radius, flat across
the field -- that is seeing and focus on a poor night (the batch summary records FWHM 5.9"),
not optics, but it means this rests on one Portland field and Leon's own zenith stacks would
be the better test. And **none of it touches the term that actually limits the campaign.**
Leon's problem was never a bad optic: it is that the cubic moved 4.84 % between calibration
nights and sits 121-129 focuser steps from the eclipse configuration. The retracted 16x at
least offered an explanation with a fix ("fit the spacer"); removing it leaves the transfer
error standing alone, unexplained and unbounded above. That is why section 9.3's focus sweep
matters more after this week's work than before it.

## 13. What to do with this

Proposals, not changes:

1. **Correct §18.11** — the 16×, and the attribution to Carrell. §10.5 and §11.4 lean on it.
2. **Adopt +0.37 ± 0.04** as the gauge constant, closing §18.11's stated open item, and note
   that Table 1's sign column needs its axis convention before it can be read physically.
3. **Add the absolute-versus-fractional rule to §3a** — it changes how Table 1 and §5.1 are
   read, and it is the honest argument for a low-distortion optic.
4. **Record the 28-month drift** wherever pre-season characterisation is discussed.
5. **The focus sweep of §9.3 is still the missing measurement**, and now has a no-reducer
   baseline to be compared against.

Data: `I:\Don Bruns TV-85 calibration`, `I:\Don Bruns 2024`, `I:\2017 eclipse images Don
Bruns`, `I:\Kenneth Carrell 2024\FRA500`, `I:\Leakey 2024 data`, `J:\Eclipse data\Toby
Portland data\2026-07-29`, `G:\SharpCap\2026-08-06\Zenith`. Reductions under
`D:\MEE2024 output\MEE_output\`.
