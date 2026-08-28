# CAL_piLeo step 2: the plate scale, and an error bar 15 % wider than reported

> **CANONICAL RESULT SUPERSEDED — 2026-08-27, on Douglas' direction.** The 17-frame stack
> below contained one frame that is physically a **~0.3 s exposure-transition artefact**
> (`0.3s8_29_17\CAL_piLeo_00001 (2).fits`, header claims 1.0 s; its sky of 1175 ADU
> matches the 0.3 s frames' 1173–1174 to 0.1 %, against 2773–2839 for true 1 s frames, and
> its timestamp overlaps the 0.3 s block's cadence). It is excluded. **The corrected
> canonical calibration is the 16-frame set, exactly the authoritative G: organisation:**
> all 6 of `1.0s8_29_19` (including the true-1 s transition frame at 18:29:25.799, sky
> 2839 ADU), the first 3 of `1.0s8_29_51` (pre-C3; the last ends 0.21 s before C3), and
> all 7 of `2.0s8_29_27`.
>
> | corrected canonical (16 frames) | |
> |---|---|
> | `observation_time` | **18:29:35** (true-exposure weighted mid) |
> | stars used | **74** |
> | rms | **0.5323 ″** |
> | **plate scale** | **2.2054197 ″/px** |
> | reported (HC0) | 21.7 ppm |
> | **uncertainty (HC3)** | **25.1 ppm** (jackknife 24.9) |
>
> The shift from the superseded value is **−11.7 ppm** (2.2054456 → 2.2054197), half the
> error bar, of the measured star-sample-sensitivity class. Every ladder, decomposition and
> conclusion below was derived on the 17-frame stack; the −12 ppm offset applies wherever
> its absolute plate scale is compared against anything (§9's day–night gaps move by the
> same amount), and none of the findings change. Reduction:
> `cal_pileo_step2/variant_A16_pure_tiers/`; canonical frame list:
> `cal_pileo_step2_frames.txt` (16 G: paths). **Step 3 must import this result file, not
> the superseded one.**
>
> **Further superseded, 2026-08-28**: the references changed to the **six 08-12 zenith
> files only** (the telescope was transported between night 1 and eclipse day and
> measurably changed — tilt dipole doubled, +197 ppm of scale; `docs/REFRACTION_2026.md`
> §16.2–16.3 on branch `refraction-leon-2026`). **The final canonical is 2.2054043 ″/px**
> (74 stars, rms 0.5318 ″, HC0 21.6 ppm, HC3-class ~25 ppm), reduction
> `cal_pileo_step2/canonical_16f_night2refs/`. That is the file step 3 imports.


> **Versioned into the repository 2026-08-28.** This document lived only on
> `D:\MEE2024 output\MEE_output` until then, and that copy has been deleted so there is no
> rival. Three artefacts it refers to moved with it:
> `cal_pileo_step2_frames.txt` → `calibration/cal_pileo_frames.txt`;
> `cal_pileo_step2/analysis/` → `tools/cal_pileo_step2/` (with a README recording that its
> shell drivers still point at `I:` and `H:`, neither of which is a valid source now);
> the zenith references → `calibration/zenith_cubic/`. The **reductions** stay on `D:` —
> they are regenerable from `G:` and this code, which is the line between the two places.

**Date:** 2026-08-26. Every number here was produced this week on `v1.4.0-dev` from the raw
frames on `I:\Leon 2026\2026-08-12\Eclipse\CAL_piLeo`. Reductions under
`D:\MEE2024 output\MEE_output\cal_pileo_step2\`.

This is step 2 of the three-step chain of `LEON_2026-08-11.md` §16 — the eclipse-day
calibration, which imports the zenith cubic and re-fits the low orders at the eclipse-day
configuration. It produces `platescale_relative_uncertainty`, which sets Method 1's transfer
term through F&L's equation 23.

---

## 1. The result

| | |
|---|---|
| frames | 17 pre-C3 (9 × 1 s + 8 × 2 s), uncalibrated |
| `observation_time` | **18:29:34** (exposure-weighted mid-point, 18:29:34.256) |
| stage 2 | cubic, `distortion_fixed_coefficients=quadratic`, 12 zenith references, `max_star_mag_dist=13`, corrections on |
| `distortion_fit_tol` | **1.0 ″** |
| stars used | **73** |
| rms | **0.5292 ″** |
| **plate scale** | **2.2054456 ″/px** |
| reported uncertainty (HC0) | 20.4 ppm |
| **corrected uncertainty (HC3)** | **23.5 ppm** (jackknife 23.3) |

A fresh stage 1 and stage 2, run into a new output tree, reproduced the previous session's run
**bit-identically**: star count, rms (0.529163683135836), plate scale (2.2054456202202464),
uncertainty, RA, Dec, roll and every cubic coefficient match in full float64. The pipeline is
deterministic on this field, so the earlier ladder can be read alongside these numbers.

The headline is in the last two rows, and it is narrow: **the plate scale is fine, the error bar
on it is not.** Nothing here changes a fitted value.

> **Provenance correction, 2026-08-27.** This reduction was run from the archival
> `I:\Leon 2026` copy; the authoritative raw tree is `G:\Leon Aug 2026`, in which Douglas
> corrected frames that carried inaccurate EXPTIME headers or sat in wrong exposure
> folders. Verification against G: (bidirectional, by DATE-OBS **and pixel-content hash**):
> all 17 used frames are **bit-identical** on G:, and the 10 unused 1.0 s/2.0 s frames on
> G: are all post-C3 — the pre-C3 selection missed nothing. So every fitted number in this
> document stands unchanged. Two labels do not: G:'s corrected organisation assigns
> `18_29_19/00001` to **0.3 s** (header claims 1.0) and `18_29_27/00001` to **1.0 s**
> (header claims 2.0) — precisely the two block-first frames this document's §8-era
> sky-level analysis flagged. The stack is therefore truly **1 × 0.3 s + 9 × 1.0 s +
> 7 × 2.0 s** (23.3 s of integration, not 25.0), and the true-exposure weighted mid-point
> is **18:29:34.995 UTC**, +0.74 s from the 18:29:34.256 used — worth **+1.3 ppm** on the
> plate scale at the measured 1.78 ppm/s, well inside the ±23.5 ppm error. The canonical
> frame list (`cal_pileo_step2_frames.txt`) now carries G: paths. The zenith reference
> chain is likewise verified: the 12 handoff files reproduce from local frames to all
> seven digits, and those frames are pixel-identical to G: (12/12 sampled). `I:` and
> `J:\Eclipse data` are archival backups and are not to be used for analysis.

> **Selection-sensitivity addendum, 2026-08-27** (Douglas' question: should the frame list
> have been chosen photometrically rather than by the C3 clock?). Two variants, both from
> the authoritative G: tree, each at its own true-exposure weighted mid-time, otherwise
> identical to the baseline:
>
> | variant | frames | stars | rms (″) | plate scale (″/px) | Δ vs baseline (ppm) |
> |---|---|---|---|---|---|
> | baseline (the settled 17) | 17 | 73 | 0.5292 | 2.2054456 | — |
> | pure 1 s + 2 s tiers (drop the accidental 0.3 s) | 16 | 74 | 0.5323 | 2.2054197 | −11.7 |
> | + the 3 photometrically-usable post-C3 frames (18_29_51 f4–f5, 18_29_57 f1) | 20 | 69 | 0.5199 | 2.2054338 | −5.4 |
>
> Both moves sit inside half the ±23.5 ppm error bar, and their size matches the already
> measured **star-sample/matching instability** (~6–12 ppm from single-assignment changes),
> not any photometric content of the frames — B20 actually *lost* four stars to its
> slightly raised background while its rms improved. The post-C3 frames are individually
> usable (sky +13–37 %, gradients 2–3×, unsaturated) and buy nothing, because integration
> is not the limiting resource on this field (§5). **The criterion is hereby restated
> photometrically**: exclude frames that are saturated, flooded, or taken while the
> background is non-stationary (rising 10–20 % per frame after C3) — which the end-before-C3
> clock rule approximates; the three marginal frames it also excludes are measured to be
> worth ≤5 ppm. The settled 17 stays canonical; selection freedom of one-to-three frames is
> a ~±10 ppm effect already inside the quoted error.

## 2. The reported standard error understates itself, and by more where leverage concentrates

`distortion_polynomial.py:209` reports **HC0** — White's heteroskedasticity-consistent standard
error, with no small-sample correction. HC0 is known to run low when a few points carry high
leverage, and at `tol 0.5` one star out of 31 carries **h = 0.46**. Two independent corrections:
**HC3**, which divides each residual's contribution by `(1 − h_i)²`, and a **delete-one
jackknife**, which refits N times leaving one star out and takes the spread.

| tol (″) | N (stars) | rms (″) | plate scale (″/px) | HC0 (ppm) | **HC3 (ppm)** | **jackknife (ppm)** | HC3/HC0 (ratio) | max h (dimensionless) |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 5 | — | — | — | — | **fails** | — | — |
| 0.3 | — | — | — | — | — | **fails** | — | — |
| 0.5 | 31 | 0.3440 | 2.2054476 | 19.21 | **26.21** | 25.78 | 1.36 | 0.459 |
| 0.7 | 55 | 0.4391 | 2.2054416 | 19.35 | **23.25** | 23.03 | 1.20 | 0.291 |
| **1.0** | **73** | **0.5292** | **2.2054456** | **20.36** | **23.50** | **23.34** | **1.15** | **0.272** |
| 1.5 | 90 | 0.7471 | 2.2054358 | 37.62 | 42.52 | 42.28 | 1.13 | 0.251 |
| 2.0 | 101 | 0.9067 | 2.2053969 | 40.66 | 44.82 | 44.59 | 1.10 | 0.213 |
| 999 | 105 | 1.0792 | 2.2054258 | 46.67 | 50.91 | 50.67 | 1.09 | 0.208 |

HC3 and the jackknife agree to ~1 % at every rung, which is the check that neither is an
artefact. `tol 0.2` fails with a clear diagnostic — "only 5 star(s) matched the catalogue, but a
cubic distortion fit needs at least 10" — and `0.3` fails the same way.

**The correction is 15 % at tol 1.0 and 36 % at tol 0.5**, and it shrinks steadily with N: 9 % by
105 stars. That is the expected behaviour — the small-sample penalty is real but it is not large
at the star counts this field delivers.

**Its practical consequence is comparative, not absolute.** Because the understatement is more
than twice as large at tol 0.5 as at tol 1.0, the *reported* figure ranks the two backwards:
19.21 against 20.36 says 0.5 is better, while 26.21 against 23.50 says it is worse. So the
reported number should not be used to choose between configurations. **Stay at tol 1.0**, as
§18.8 specifies; `tol 0.7` is nominally 1 % better on HC3 and that is noise. The *plate scale
itself* spans only **2.8 ppm** across tol 0.5 to 1.0 — the tolerance in that range does not move
the answer, only the error bar.

The understatement is worst where a fit is small and lopsided. Sub-stack A (36 stars) reports
HC0 = 56.6 ppm against HC3 = 149.2 and a jackknife of 147.0 — **a factor of 2.6**, because one
star there sits at leverage **0.72**.

*A note on method, since an earlier draft of this section got it wrong.* The correction was first
quoted from a **pairs bootstrap** (resample the N stars with replacement, refit, take the spread),
which gave 28.75 and 25.12 ppm. Those figures were ~15 % too high: the bootstrap draw
distribution is heavy-tailed, and summarising it with an RMS lets a few extreme resamples set the
answer — its own robust (interquartile) scale gives 24.62 and 22.82, in line with HC3 and the
jackknife. The resamples were checked for degeneracy and were not the problem (condition number
median 91 against 67 for the fit), so the bootstrap corroborates the direction; it is simply the
wrong summary statistic. HC3 is quoted throughout below because it agrees with the jackknife and
is what the code could actually report.

## 3. Anatomy: where the error comes from

The reported quantity has a closed form. Writing `u = x/w`, `v = y/w` with `w = 3124 px`
(`max(img_shape)/2`) and `w·s = 6890″ = 1.914°`:

    sigma_S/S  =  sqrt( sigma_x^2/sum(u^2)  +  sigma_y^2/sum(v^2) ) / (w · s)

Rebuilt from the residual files this reproduces the reported HC0 exactly — 19.21, 20.36 and
46.67 ppm at tol 0.5, 1.0 and 999 — so this is not a model of the code, it is the code.

At tol 0.5 the two terms are **x: 9.9 ppm, y: 17.9 ppm**. The y axis contributes 1.8× the x
axis, for two compounding reasons, neither of them a setting:

- the sensor is 1.5× narrower in y, so the lever arm `sum(v²)` is **half** `sum(u²)`;
- and `sigma_y/sigma_x = 1.34` on top of that.

That second factor is odd, and is §7.

## 4. What does not help — five levers closed

**Tolerance** — §2. `sigma²/N` is flat from 0.5 to 1.0: 31 stars at 0.208″/axis and 73 at
0.316″/axis carry the same constraint. Not a lever.

**Catalogue depth — dead.** `gaia_dr3_g15` is installed and in use ("ignoring gaia_dr3_g13:
already covered by gaia_dr3_g15"). Raising `max_star_mag_dist`, with `safety_limit_mag` to
match:

| `max_star_mag_dist` (Gaia G mag) | 13 | 13.5 | 14 | 15 |
|---|---|---|---|---|
| stars used (count) | 73 | 73 | 73 | 73 |

**The star count does not move at all.** All 132 detected centroids are brighter than G = 13,
so the *detection limit* binds, not the catalogue. (The plate scale wanders ~6 ppm across these,
from a few faint entries changing individual match assignments — worth knowing as a
matching-instability figure, but no gain.)

**Inverse-variance weighting — dead, and instructively so.** `do_cubic_fit(..., weights=1)` is
called with `weights=1` from its only call site; the parameter is plumbed, documented as "new
Oct'24: option for weighted centroids", and never used. Estimating the gain offline from the
binned empirical `sigma(magnitude)` — which credits *every* bit of the spread to noise and is
therefore an upper bound — gives **1.00 to 1.03×**. It buys nothing, because the residual is
not photon-limited:

| V (mag) | <8 | 8–9 | 9–10 | 10–10.5 | 10.5–11 | 11–11.5 | >11.5 |
|---|---|---|---|---|---|---|---|
| n (stars) | 9 | 22 | 27 | 22 | 20 | 4 | 1 |
| rms (tol 999) | 1.235″ | 0.956″ | 0.909″ | 0.835″ | 1.135″ | 1.342″ | 4.223″ |
| **median** | **0.608″** | **0.763″** | **0.569″** | **0.649″** | **0.858″** | 1.481″ | 4.223″ |

Read the median row, not the rms — F17 makes exactly this point, that a star sample's rms is
largely a readout of its outliers. **From V = 3.9 to V = 10.5 the median residual is flat within
0.57–0.76″, across a factor of ~500 in flux**, where photon-limited centroids would differ by
about twenty. Only beyond V = 11 does it climb, and there are five such stars, all cut at tol 1.0
anyway. **Whatever dominates this residual is not starlight**, which is why weighting by
brightness has nothing to work with.

(The bright bin's *rms* of 1.235″ is neither evidence against this nor a saturation effect: it is
set by two stars at 2.354″ and 2.459″, neither of them clipped, against a bin median of 0.608″.
An earlier draft attributed the bright-end rms to saturation; the per-frame measurement of §8
disproves that.)

**More frames — dead, and this is the important one.** Two further pre-C3 blocks exist and have
never been used: `18_29_17` and `18_29_46`, 6 × 0.3 s each, both ending well before C3 (−34.5 s
and −6.1 s). Adding them gives **29 frames, +71 % in count and +14 % in exposure**:

| | N (stars) | rms (″) | plate scale (″/px) | HC0 (ppm) | **HC3 (ppm)** |
|---|---|---|---|---|---|
| 17 frames, tol 1.0 | 73 | 0.5292″ | 2.2054456 | 20.36 | **23.50** |
| 29 frames, tol 1.0 | 76 | 0.5304″ | 2.2054298 | 21.22 | **24.53** |
| 17 frames, tol 0.7 | 55 | 0.4391″ | 2.2054416 | 19.35 | **23.25** |
| 29 frames, tol 0.7 | 55 | 0.4193″ | 2.2054034 | 18.14 | **21.82** |

**The rms does not move — 0.5292″ to 0.5304″, 0.2 %** — and the error does not improve. The
stacked frame yields *fewer* window-refined centroids (129 against 132), the star count gains 3,
and the plate scale shifts 7 ppm at tol 1.0 and 17 ppm at tol 0.7, inside the error bar but not
zero. **Keep the settled 17.** The extra frames add bookkeeping and a longer time baseline
(35.2 s, drift 9.85 px) for no measurable gain — which is exactly what §5 predicts, and is the
cleanest direct test of it in this document.

**And the 0.3 s frames on their own are an independent check worth having.** Reduced alone —
12 frames, 3.6 s of total exposure, sharing *no frames* with the 17 — they give:

| | N (stars) | rms (″) | plate scale (″/px) | HC0 (ppm) | HC3 (ppm) |
|---|---|---|---|---|---|
| 17 frames (9 × 1 s + 8 × 2 s) | 73 | 0.5292″ | **2.2054456** | 20.36 | 23.50 |
| 12 frames, 0.3 s only | 37 | 0.6480″ | **2.2054428** | 44.57 | 58.72 |

**1.3 ppm apart.** Their own errors are far larger than that, and they share the same stars, so
this bounds the frame-independent term again rather than validating the total (§5) — but it is a
second, disjoint frame set reaching the same answer. It also settles the saturation question
directly: the 0.3 s stack peaks at **14 336 ADU** (a single raw frame at 4138, 6 % of full scale),
so nothing in it is clipped, the V = 3.9 star is measured cleanly at 0.643″ against a field rms of
0.648″, and the plate scale does not move. **Saturation is not biasing this field's plate scale.**

**Calibration frames** — already settled at 2.3 ppm in `CALIBRATION_FRAMES.md`. Run
uncalibrated.

## 5. The error splits into 2 ppm that more data would fix and 23 ppm that it would not

The 17 frames divide into three blocks that **share no frames**, each reduced independently at
its own exposure-weighted mid-time:

| sub-stack | frames | mid-time (UTC) | N (stars) | rms (″) | plate scale (″/px) | HC0 (ppm) | **HC3 (ppm)** | max h (dimensionless) |
|---|---|---|---|---|---|---|---|---|
| A | 6 × 1 s | 18:29:22.7 | 36 | 0.6038″ | 2.2055013 | 56.6 | **149.2** | 0.721 |
| B | 8 × 2 s | 18:29:35.3 | 60 | 0.5580″ | 2.2054932 | 24.2 | **28.3** | 0.288 |
| C | 3 × 1 s | 18:29:51.9 | 34 | 0.6308″ | 2.2054936 | 41.7 | **58.9** | 0.524 |

**They agree to 3.6 ppm total, 2.1 ppm scatter** — against individual errors of 28 to 149 ppm.
That agreement is not a miracle and it is not a validation of the error bar. The three share
every star, the catalogue and the frozen zenith cubic; only the *centroid noise* is independent
between them. So their scatter bounds one term and one term only:

- **frame-independent (centroid noise): ≲ 2 ppm**
- **everything else: ~23 ppm** — star selection, catalogue, the transferred cubic, the
  distortion model.

**More frames, longer exposures and better signal-to-noise cannot touch the second term.** That
is the answer to whether ~20–24 ppm is the field's floor: for *this field and this star list*, it
is. The §12.5 and §13.2 recommendations — narrower magnitude spread, even sensor coverage, a
single 4 s exposure — remain right, but they work by changing **which stars are in the fit**,
not by collecting more photons on the same ones.

One flag for the record: the combined 17-frame reduction sits **21.8 ppm below** the
inverse-variance weighted mean of the three sub-stacks (2.2054456 against 2.2054937). That is
0.93σ of the combined stack's own HC3 error — not significant — but it is *below all three*, not
scattered about them. About half is traceable to star sample: the 17 stars in the
combined fit that are absent from B carry ~11 ppm of the linear terms between them. Worth
watching rather than acting on.

## 6. The refraction time-dependence, measured

The stack spans 33.7 s, during which the field drifts 9.2 px and the altitude falls 0.103°
(11.05″/s at az 269.9°, φ 42.74°). The refraction plate-scale term goes as `csc²h`, so
`d ln/dh = −2 cot h = −0.201 /deg`, and against §11.2's 3500 ppm at this altitude that predicts
**702 ppm/deg**, or 72 ppm across the stack.

Reducing a sub-stack at two assumed times measures it directly:

| sub-stack | Δt (s) | Δ plate scale (ppm) | rate (ppm/s) |
|---|---|---|---|
| A | 11 | 19.6 | **1.78** |
| C | 18 | 32.4 | **1.80** |

1.79 ppm/s ÷ 0.00307 °/s = **583 ppm/deg**, against 702 predicted — 17 % agreement by an
independent route, the same order §19.1 got for the temperature term.

Two consequences. First, **the exposure-weighted mid-point matters**: 18:29:19 rather than
18:29:34 would shift the plate scale by 27 ppm, larger than the whole error bar. Second, and
better news: after each sub-stack is corrected at its own time the three agree to 3.6 ppm, where
uncorrected they would differ by ~52 ppm across the 29 s baseline. **The refraction correction
removes its own time-dependence to better than 8 % of itself** — the first differential test of
the §11.1 refraction fix, and it passes.

## 7. Two things found on the way

**The transferred zenith cubic leaves no detectable residual.** Decomposing the stage-2 residual
into radial and tangential parts about the field centre and regressing the radial part on `r³`:

    coefficient at r = 1:   -0.048 +/- 0.097 arcsec   (0.5 sigma)

against a transferred cubic of 3.505″ at r = 1 (d(3000) = 3.1048″ scaled to 3124 px). That is
**1.4 % ± 2.8 %**, so ≤ 5.6 % at 2σ. It is consistent with §18.9's "≥ 2.4 % and unbounded above"
and is the first *upper* bound on that quantity. Read it with care: the free linear term has
already absorbed the part of a cubic residual collinear with `r` — the fitted `r` coefficient
comes out at exactly zero — so this constrains only the non-collinear component, and CAL_piLeo
cannot see a cubic error that looks like a scale change. It does not close §18.9.

**The residual is anisotropic, sensor-aligned, and specific to this field.** `sigma_y/sigma_x` is
1.34 at tol 0.5 and 1.0 (bootstrap 68 %: 1.16–1.54), rising to 2.01 on the untruncated set, with
the major axis at PA 92–98° in sensor coordinates — the short axis. It is **not** the sky
vertical (89.7° away), so not refraction or chromatic dispersion; **not** the drift direction, so
not trailing; and **not** the image, because adaptive moments on the stacked frame give a round
PSF (`sigma_y/sigma_x = 1.015`, FWHM 6.12″; the 1 s and 2 s sub-stacks give 6.01″ and 5.94″ —
identical, which independently rules out trailing).

It is also peculiar to CAL_piLeo. Across 130 stage-2 reductions on this machine:

| campaign | runs (count) | tol (″) | median N (stars) | median rms (″) | median σ_y/σ_x (ratio) |
|---|---|---|---|---|---|
| bruns2017_nights | 29 | 0.2 | 842 | 0.0561″ | 1.10 |
| bruns_np101 | 10 | 0.2 | 874 | 0.0552″ | 0.97 |
| carrell_fra500 | 3 | 0.2 | 771 | 0.0605″ | 0.94 |
| leakey_zenith | 25 | 0.2 | 288 | 0.0519″ | 1.02 |
| london65_tol02 | 3 | 0.2 | 490 | 0.0446″ | 1.16 |
| **portland_zenith** | 6 | 0.2 | 3024 | 0.0933″ | **0.85** |
| tv85 | 2 | 0.2 | 658 | 0.0940″ | 0.96 |
| **bruns2017_lr** | 10 | 0.2–0.5 | 66 | 0.1217″ | **1.00** |
| **CAL_piLeo, tol ≤ 1.0** | 1 | 1.0 | 73 | 0.5292″ | **1.34** |
| CAL_piLeo, loose tolerances | 22 | 2–999 | ~90 | ~1.07″ | 1.75–2.03 |

`portland_zenith` is the **same FRA500 + 0.7× + ASI2600**, so it is neither the camera nor the
optical train.

> **Corrected 2026-08-27, and the mystery is solved: the anisotropy IS the local vertical.**
> This section (and §3's "that second factor is odd") relied on `vertical_test.py`, whose
> hand-derived parallactic rotation turned out to be **90° off**. Re-deriving the vertical
> direction empirically — an affine fitted from the solved stars' astropy alt-az against
> their own pixels, self-checking at 2.5 px — shows the sensor −y axis lies **3.1° from the
> local vertical**, so the "sensor-aligned major axis at PA 92–98°" was the *vertical* all
> along, and the split is **0.424″ vertical / 0.317″ horizontal** (tol 1.0; 0.964/0.486 at
> tol 999), not the reverse. The night horizon campaign (`docs/REFRACTION_2026.md` §12–13,
> branch `refraction-leon-2026`) then closes the story: the night atmosphere on this
> sightline is vertical-major with the same character (quasi-static 0.15–0.32″ vertical
> against 0.07–0.13″ horizontal, per-6 s-frame jitter 0.43–0.61″ vertical), and the daytime
> vertical budget reproduces from night-measured numbers once the shorter integration is
> accounted for (17 × 1–2 s ≈ 25 s against the night blocks' 270 s). Nothing here is
> peculiar to the eclipse, the camera, or the optics — it is the ordinary anisotropic
> atmosphere at airmass ~6, mislabelled by one rotation error. `analysis/vertical_test.py`
> is superseded by the affine method (`refraction/analysis/m3_maps.py`).

**The comparison is not fully controlled, and the tol column says why**: every other campaign is
reduced at 0.2, which CAL_piLeo cannot reach (§2), and a tighter tolerance truncates the residual
distribution and would suppress any anisotropy in it. The best-matched row is `bruns2017_lr` —
the 2017 eclipse-day calibration, the same *role* in the chain, a comparable 66 stars — and it is
isotropic at 1.00, though at a quarter of CAL_piLeo's rms.

So: at ~2σ on 73 stars, and without a tolerance-matched control, this is suggestive rather than
established. The dominant reason the y term leads the error budget is still the sensor aspect
ratio (§3), not this. Recorded so it is not re-derived; not explained.

## 8. F16, measured rather than proxied

**F16 rejects individual saturated stars from the fit; it never drops a frame.** ROADMAP F16
specifies a peak-value flag set at stage 1 and honoured at stage 2 regardless of tolerance, and
Douglas' 2026-08-24 refinement requires it be measured **per raw frame**, not on the stack,
because a clip carried only by the long exposures is diluted away by the short ones.

That measurement now exists for this field. For each of the 73 fitted stars, its position was
mapped back into each of the 17 raw frames through the integer shift the stacker applied, and the
peak read there:

| | |
|---|---|
| stars clipped at 65535 ADU in ≥1 raw frame | **1 of 73** |
| stars above 60000 ADU in ≥1 raw frame | 1 of 73 |
| the clipped star | V = 3.9, clipped in **6 of the 8 × 2 s frames** |
| its peak in the 1 s frames | 49 206 ADU — unclipped |
| second-brightest star's peak, any frame | 20 253 ADU — a factor of 3 clear |
| **peak on the combined stack** | **43 378 ADU** |

**This confirms the ROADMAP's dilution argument quantitatively.** A stack-based test sees 43 378
and finds nothing to reject; the per-frame test finds a star clipped in six frames. (The ROADMAP
records the v1.3.6 stacks peaking at 59 612, below a 60 000 cut; the 17-frame stack sits lower
still.) **The per-frame requirement is not a refinement here, it is the difference between F16
working and F16 being inert.**

And its cost on this field is negligible:

| | N (stars) | rms (″) | max h (dimensionless) | HC0 (ppm) | HC3 (ppm) | jackknife (ppm) |
|---|---|---|---|---|---|---|
| as fitted | 73 | 0.5292″ | 0.272 | 20.36 | **23.50** | 23.34 |
| **F16 as specified — drop the 1 clipped star** | 72 | 0.5314″ | 0.272 | 20.67 | **23.89** | 23.73 |
| a V < 8 magnitude cut (7 stars) — *not* F16 | 66 | 0.5348″ | 0.498 | 22.71 | 26.44 | 30.44 |
| drop the single highest-leverage star | 72 | 0.5327″ | 0.328 | 22.82 | 27.33 | 27.83 |

**F16 costs 1.7 %.** The clipped star sits at radius 2147 px with leverage h = 0.064 — it is not
one of the fit's anchors — and its residual is 0.333″, *better* than the field's 0.529″ rms, which
is what §18.8 predicts for a fixed centroid window on a flat-topped star. **Enable F16 at step 2;
it is safe here.**

*An earlier draft of this document said the opposite*, on the strength of the third row — a
magnitude cut standing in for F16. That proxy was wrong in both directions: it removes seven
stars where F16 removes one, and the seven include high-leverage anchors that F16 never touches.
The lesson is the fourth row, which is the part that survives: **this fit is leverage-concentrated
enough that removing any single well-placed star costs ~19 %, while the rms barely moves.** Any
star-rejection rule should therefore be checked against the leverage distribution rather than
against the rms — F16 simply passes that check.

The free-order ladder makes the same geometric point from the other side:

| `distortion_fixed_coefficients` | free terms | N (stars) | rms (″) | plate scale (″/px) | HC0 (ppm) | HC3 (ppm) |
|---|---|---|---|---|---|---|
| `linear` | linear only | 67 | 0.5803″ | 2.2053856 | 27.72 | 29.67 |
| **`quadratic`** | **through quadratic** | **73** | **0.5292″** | **2.2054456** | **20.36** | **23.50** |
| `None` | through cubic | 72 | 0.5183″ | 2.2055017 | 61.32 | 68.89 |

Freezing the quadratic as well — taking it from the zenith night — makes the fit *worse* (rms
0.58 against 0.53) and moves the scale 27 ppm: a direct vindication of §16's design, that the low
orders have to come from the daytime field. And letting the cubic float triples the error, because
`x` and `x³` are 92 % collinear over a rectangular field (VIF ≈ 6, predicted factor 2.5, observed
3.0). That is the arithmetic behind §12.3's "cubic: not a measurement", and it explains why the
zenith fields' reported 1.2–1.3 ppm sits above what their star counts alone would suggest.

## 9. Against the zenith nights

The twelve zenith fields of §18.8, reduced the same week on the same rig:

| | fields (count) | mean plate scale (″/px) | field-to-field sd (ppm) | se on the mean (ppm) | reported HC0 per field (ppm) |
|---|---|---|---|---|---|
| zenith 08-11 | 6 | 2.2077996 | 7.0 | 2.8 | 1.27–1.40 |
| zenith 08-12 | 6 | 2.2073819 | 4.6 | 1.9 | 1.12–1.30 |
| **CAL_piLeo, eclipse day** | 1 | **2.2054456** | — | **23.5 (HC3)** | 20.4 |

    night-to-night, 08-11 minus 08-12          +189 ppm   (both corrections off)
    CAL_piLeo below 08-11, as reduced          -1066 ppm   (not like for like -- see below)
    CAL_piLeo below 08-12, as reduced           -877 ppm   (not like for like -- see below)

The +189 ppm night-to-night gap reproduces §12.4's "190 ppm at 32σ".

**One correction has to be applied before the last three numbers are read as physical, and
§18.1 already measured it.** The zenith fields were reduced with corrections **off**
(deliberately), while CAL_piLeo has them **on**. §18.1, field-matched across all twelve zenith
fields, records the shift from turning all three corrections on as **−221.9 ppm (sd 5.4 ppm)** on
the fitted plate scale, alongside **+0.065 % (sd 0.117)** on the transferred cubic.

Re-reducing `08-12 Z1_base` from its 30 raw 4 s frames reproduces that:

| 08-12 Z1_base | stars used | rms (″) | plate scale (″/px) | HC0 (ppm) |
|---|---|---|---|---|
| corrections off | 2399 | 0.0690 | **2.2073777** | 1.24 |
| corrections on | 2316 | 0.0678 | **2.2068874** | 1.22 |

−222 ppm, against §18.1's −221.9. (The corrections-off row also reproduces the §18.8 handoff value
to all seven digits.) Applying it:

    CAL_piLeo below 08-11, like for like        -844 ppm
    CAL_piLeo below 08-12, like for like        -655 ppm

**What is new here is only the mechanism, not the number.** An earlier draft of this section
estimated the correction at ~106 ppm by scaling §11.2's 3500 ppm at 9.87° by `csc²h`, and that
scaling is wrong: refraction compresses the field by `k·sec²z` along the vertical but by only `k`
along the horizontal, and `k ≈ 283 ppm` is a *constant*. At 9.87° the vertical term is 9623 ppm
and swamps it; at 79.9° the two are 292 and 283 ppm and the constant dominates the average.
**So the refraction plate-scale term does not fall to zero at the zenith — it floors at about
280 ppm.** That predicts 203 ppm against 222 measured, and it explains why §18.1's figure is as
large as it is at 80°.

**The two nights would not converge if corrections were turned on.** They sit at the same
altitude — 79.92° on 08-11 and 79.87° on 08-12, both at az 264° — so the shift is common to them:
§18.1's sd of 5.4 ppm across all twelve fields is the direct evidence, and a common fractional
shift of 222 ppm changes a 189 ppm gap by 0.04 ppm. **The night-to-night gap is not a refraction
artefact**, and §18.2 already classifies it as a diagnostic rather than an error term, since the
zenith plate scale is discarded at step 2 by construction.

The practical rule is §19.2's, extended: **never compare a plate scale reduced with corrections
on against one reduced with corrections off**, any more than across two assumed temperatures. On
this rig at these altitudes that mistake is worth 222 ppm.

Either way the conclusion of §12.2 stands and is large: **the eclipse-day optical configuration
sits roughly 0.1 % away from both calibration nights**, which is ~100× the ~6 ppm the transfer
needs, and is the quantitative reason the low-order terms must come from the daytime field.

**Two things worth flagging about the comparison.**

*The zenith per-field standard error is 1.1–1.4 ppm, while the fields scatter by 4.6–7.0 ppm* — a
factor of 4 to 5. At N = 1500–3500 the HC0/HC3 correction is only ~2 %, so this is **not** the
small-sample effect of §2. It is real field-to-field variation that no within-fit estimator can
see, and `INSTRUMENT_COMPARISON.md` §11 found the same factor of 4–6 on Bruns' data. The
implication for CAL_piLeo is uncomfortable: **there is only one field, so no such check is
available at all.** The 23.5 ppm is a within-field number, and the zenith and Bruns evidence both
say within-field numbers run several times low when a field-to-field test exists. 23.5 ppm should
therefore be read as a floor, not as the total.

*And this reduction does not reproduce §12.2's 2.2047–2.2048.* At 2.2054456 it sits **+302 ppm**
above it. §19.5 already records those runs as unreproducible — they are not on this machine, and
the search of 331 result files found nothing near 2.2047 — so the discrepancy cannot be chased to
its cause here. The frame selection differs (§12.2 predates the C3 finding of §19.4 and its "1.0 s
× 11" included two post-C3 frames), the reference-file import differs, and `observation_time`
is not recorded in any of them (F20). **Treat 2.2047–2.2048 as superseded**: the number in §1 is
the one that reproduces bit-identically from raw frames on the current pipeline.

## 10. What this does to Method 1

F&L equation 23 gives `δL = −h·δS` with `h = 1/mean(1/r²)`, a property of the star field.
`STAGE3_THEORY.md` §4 tabulates it for the candidate Leon geometries:

| field model | h (R⊙²) | δL/L per ppm (%/ppm) | at **23.5 ppm** (% of L) | at the reported 20.4 ppm (% of L) |
|---|---|---|---|---|
| Leon annulus, uniform 2–9 R⊙ | 18.0 | 0.99 % | **23.3 %** | 20.2 % |
| Leon, uniform on sensor, offsets absorbed | ~15 | 0.84 % | **19.7 %** | 17.1 % |
| Bruns-effective (stars closer in) | 6.8 | 0.37 % | **8.7 %** | 7.5 % |

F&L's own specification for σ_L ≤ 0.1″ on the Leon field is **δS ≤ 5.8 ppm**.

**So Andrew's open item 3 resolves to the pessimistic branch: the transfer term costs 9–23 % of
the deflection constant, not ~1 %.** Note that this conclusion does *not* rest on the HC0/HC3
correction — even the reported 20.4 ppm gives 7.5–20 %. The correction moves the answer by about
three percentage points; what puts it in the tens of percent is the size of the plate-scale error
itself, which every estimator agrees on to within 15 %.

Which end of the range applies is a step-3 question — it depends on where the eclipse stars fall
relative to the Sun, and `h` is a property of *that* field, not this one. Bruns reached 1.23 %
because his δS was 3.34 ppm *and* his stars sat close in; CAL_piLeo delivers neither.

The gap to 5.8 ppm is a factor of **4.1**, and §5 says it cannot be closed with more frames of
this field. The closed form of §3 prices the alternatives. Solving it for the star count that
reaches 5.8 ppm, holding the field's present 78 % coverage efficiency:

| per-star scatter, per axis (″) | stars needed (count) |
|---|---|
| 0.374 — what this field delivers | **1089** |
| 0.243 — the quality of its best 31 | **459** |
| 0.180 — even coverage, no outliers | **252** |
| 0.044 — what this rig reaches at night | **15** |

against **132 centroids detected and 105 matched**.

**The scatter is the more leveraged term**: σ enters linearly and N only as √N, so halving the
per-star error is worth quadrupling the star count. And the daytime figure is **8.5× the same
telescope and camera at night** — which is not a photon-noise gap (§4: the residual barely moves
across three magnitudes), so it is an unidentified systematic rather than a limit of the sky
brightness. Finding it is worth more than any amount of extra exposure, and §7's anisotropy —
present on this field and on no other reduction on this machine — is the one concrete lead.

## 11. What to do

1. **Carry forward `2.2054456 ″/px ± 23.5 ppm`** as the CAL_piLeo plate scale, at
   `observation_time = 18:29:34`, and quote the temperature it was reduced at (30.5 °C) beside
   it, per §19.2's rule.
2. **Treat `platescale_relative_uncertainty` as a lower bound, not the uncertainty.** It is HC0
   and runs 9–36 % low across this field's fits, and worse on small lopsided ones. HC3 is a
   one-attribute change in `distortion_polynomial.py:209` (`statsmodels` exposes `HC3_se`
   alongside `HC0_se`), and it agrees with a delete-one jackknife to 1 %; §5.3 of
   `STAGE3_THEORY.md` already proposes the jackknife at stage 3 for the same reason. **This
   would change a reported number that reaches σ_L through `plate_covariance2`, so by the
   ROADMAP §6 rule it is a results-changing change and needs its own validation** — it is not a
   tidy-up.
3. **Enable F16 at step 2** — measured at 1.7 % on this field (§8) — but only in its per-frame
   form. A stack-based test is inert here: the stack peaks at 43 378 where the raw 2 s frames
   clip at 65 535. Check any *other* star-rejection rule against the leverage distribution, not
   the rms.
4. **Record the 583 ppm/deg refraction rate** wherever `observation_time` is discussed — it is
   what makes the exposure-weighted mid-point non-optional, and F20 should capture the time
   alongside the temperature.
5. §13.2's 2027 field-design recommendation now has targets attached (§9): **~450 stars at the
   per-star quality of this field's best 31, or ~250 if the scatter can be brought to 0.18″.**
   Prioritise the scatter over the count — it enters linearly where the count enters as √N.
6. **Do not re-reduce the zenith fields with corrections on.** §18.2 shows step 2 imports only
   cubic-and-above, so the zenith plate scale is discarded by construction, and §18.1 measures
   the corrections' effect on the transferred cubic as +0.065 % ± 0.117 % — indistinguishable
   from zero and ~15× below the 1 % requirement. Nothing that propagates would change. The
   222 ppm matters only for the *comparison* in §9, which is a diagnostic.
7. **The open question this leaves is why the daytime per-star scatter is 8.5× the night-time
   figure on the same rig**, when it is flat with magnitude from V = 3.9 to V = 10.5 and so not
   photon-limited. That is the largest single unexplained term in the step-2 budget, and §7 is
   the lead.
8. **Treat 23.5 ppm as a floor.** §9 shows that where a field-to-field check exists — the zenith
   nights here, Bruns' fields in `INSTRUMENT_COMPARISON.md` §11 — within-field errors run 4–6×
   low. CAL_piLeo is a single field and admits no such check.

## Reproducing

Stage 1: the 17 frames listed in `cal_pileo_step2_frames.txt`, `sensitive_mode_stack=True`,
`centroid_gaussian_subtract=True`, `centroid_gaussian_thresh=4.0`, `min_area=2`,
`sigma_subtract=0.0`, `delete_saturated_blob=False`, `remove_edgy_centroids=True`,
`centroid_refine_window=True`, `centroid_window_sigma=2.0`, no darks or flats.

Stage 2: `--order cubic --date-from-header --fix-distortion <the 12 files> --set
distortion_fixed_coefficients=quadratic --set distortion_fit_tol=1.0 --set max_star_mag_dist=13`,
corrections on, site 42.740470 / −5.613780 / 1101 m, 30.5 °C / 896.6 hPa / 0.208 / 0.62 µm,
`observation_time=18:29:34`.

The twelve zenith reference files are the `inpipeline_windowed` set of §18.8, held at

    D:\MEE2024 output\MEE_output\Claude Code\HANDOFF_zenith_cubic\inpipeline_windowed\

The frame list is verified against the headers: all 17 end before C3 = 18:29:53.9. Note that the
`I:\` copy is organised flat by timestamp and is **clean** — the exposure-mismatched `(2)` files
§19.4 warns about belong to a differently organised copy and are not present here.

Re-running step 2 from that path reproduces §1 **bit-identically** (plate scale
2.2054456202202464, every coefficient equal), so the restored copy is the same input the numbers
here were produced from.

One note for anyone scripting this: the reference path contains a space, and an unquoted shell
variable turns it into `D:\MEE2024`, which fails inside `_open_distortion_files` rather than at
argument parsing. Pass the twelve paths as a quoted array.

All 25 runs are tabulated in `cal_pileo_step2/RUN_SUMMARY.csv`. The analysis scripts are in
`cal_pileo_step2/analysis/` and are pure post-processing of the `TWOD_RESIDUALS.csv` and
`CATALOGUE_MATCHED_ERRORS.csv` files each run writes — nothing in the pipeline was changed:

| script | what it produces |
|---|---|
| `se_analysis.py` | §3's closed form, HC0/HC3, the weighting bound |
| `bootstrap_se.py` | the superseded pairs bootstrap (see §2's method note) |
| `estimator_audit.py` | §2's audit of it: robust spreads, conditioning, jackknife |
| `saturation.py` | §8's per-frame saturation mask (F16 as specified) |
| `final_errors.py` | the HC0/HC3/jackknife table for every arm |
| `tol_ladder.py` | §2's table |
| `substacks.py` | §5's decomposition |
| `levers.py` | §8's F16 and leverage figures |
| `residual_structure.py` | §7's radial/cubic decomposition |
| `vertical_test.py` | §7's vertical-versus-drift-versus-sensor test |
| `psf_shape.py` | §7's adaptive-moment PSF shapes |
| `common.sh`, `run_stage1.sh` | the pipeline invocations themselves |

The design matrix in the analysis scripts is **image-centred** (`distortion_fitter.py:76`
subtracts `[img_shape[0]/2, img_shape[1]/2]` before fitting) while the CSVs record raw pixels.
Rebuilding it uncentred inflates every standard error by about 5×, which looks plausible and is
wrong; the check that it is right is that HC0 comes back at 19.21, 20.36 and 46.67 ppm exactly.
