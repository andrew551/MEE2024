# Stage-3 charts and settings to carry into the MEE program

**Written 2026-09-01**, at Douglas' request, as the specification for what the Bruns 2017
work should leave behind in the product rather than in `tools/`. Three things are wanted:
the charts, the atmospheric machinery, and the exact processing settings, all reusable
for the Leon 2026 and Mexico 2024 reductions so the three datasets can be compared
directly.

## 1. The charts to add to stage 3

All four are produced today by `tools/matrix_bruns/b17_charts_record.py` (chart revision
9; every revision archived under `chart_versions/revNN_*`). They are dataset-agnostic
apart from the constants listed in §3.

| chart | what it shows | why it earns its place |
|---|---|---|
| **deflection vs radius** | per-star radial deflection against R/R⊙, the Method-1 curve, the total-error band, Einstein and Newton | the headline result; outliers beyond 2.5 σ are labelled with magnitude and σ |
| **displacement vectors** (NEW) | each star's measured shift after the fitted pointing offset and rotation are removed, drawn as a true vector on the sky in RA/Dec | the only chart that shows the *tangential* scatter, which is what separates measurement noise from deflection. Bruns' data: tangential rms 0.114″ vs radial-about-fit 0.085″ |
| **L and plate scale** | 1σ covariance ellipses for Method 1 and Method 2 against the imported scale, in ppm | makes the scale/L degeneracy visible and shows why Method 1 is primary |
| **atmosphere night maps** (NEW) | quadratic-free residual quiver maps for every night calibration field, in sensor axes | the atmospheric error floor, measured rather than assumed — see §2 |

Chart rules learned the hard way this session, all of which the implementation must keep:

* **arrows are vectors, not projections.** Plotting the radial *projection* makes every
  arrow point outward by construction and hides the tangential scatter entirely;
* **arrow lengths are asserted against the data at runtime**, and the axes are padded
  beyond the sensor — arrows leaving the axes were being silently clipped, which looked
  like missing stars;
* **positions and vectors must share one frame.** An early atmosphere map plotted the
  alt/az decomposition as arrow components over sensor-pixel positions;
* **state the variant in the title** (magnitude cut, link size) — a chart that does not
  say which fit it describes gets quoted as the wrong number;
* **every chart writes a versioned copy**; superseded versions are never deleted;
* **draw the field chart in the frame the physics lives in.** Leon's moved to alt/az
  (2026-09-02): the instrument sits 3.6° from that frame, so the footprint comes out
  nearly axis-aligned and the chart's vertical axis is the direction the atmosphere is
  polarised along, which is what the nuisance-in view exists to show. RA/Dec hid it at a
  45° roll. Bruns' chart stays in RA/Dec because his ROLL is ~0.3° and the two frames
  coincide there;
* **annotate an outlier with the evidence, not just the sigma** — Leon's charts name how
  many exposure tiers saw the star, which is what says whether the consistency vet could
  have acted on it at all.
* **arrow lengths are asserted against their own scale bar, not only against the axes.**
  Found 2026-09-01 while copying the construction for Leon: the Bruns field chart's
  sensor-to-sky conversion returned arcseconds multiplied by the plate scale, so every
  arrow was drawn 2.087× longer than the "1 arcsec" bar beside it, through nine
  revisions of review. The in-axes assertion could not see it. Revision 10 fixes it and
  asserts that a unit sensor displacement round-trips to one arcsec of sky; the Leon
  chart carries the same assertion from its first revision.

## 2. The atmospheric error floor

`tools/matrix_bruns/b17_atmosphere2.py` is the reference implementation, and the method
generalises to any campaign that has night calibration fields:

1. re-fit each night field **constant-only against the previous field of the same night**
   — the same construction the science field uses against its calibration;
2. impose the eclipse Sun's frame position, apply the science radial and magnitude cuts;
3. fit L. True deflection is zero, so whatever comes back is manufactured by that
   atmosphere and that estimator;
4. the rms over fields is the systematic.

**Two traps, both of which produced wrong answers first** (§ the record):

* the field's **own observation time** must be passed — the refraction correction is
  applied at the altitude it implies, and a placeholder time leaves a per-pointing
  systematic that masquerades as atmosphere (it produced ±1.40″ instead of ±0.15″);
* pair fields **within one night**. Fields 06–10 of each Bruns group are the *following*
  night, and pairing across the gap measures the +85 ppm night-to-night plate-scale step.

Measured results, for reuse as reference values:

| set | geometry | fields | quasi-static residual | vertical | horizontal | V/H | null-test L systematic |
|---|---|---|---|---|---|---|---|
| Leon 2026 zenith | alt 79–83° | 12 | **0.067″** (0.058–0.075) | 0.048″ | 0.046″ | 1.1 | **±0.12″** one-sided |
| Leakey 2024 zenith | alt 79–88°, a second instrument | 21 | **0.078″** (0.035–0.212) | 0.050″ | 0.059″ | 0.9 | **±0.12″** one-sided |
| Bruns 2017 night | alt 53–55°, the eclipse-day pointings | 29 | **0.100″** (0.041–0.175) | 0.072″ | 0.068″ | 1.1 | **±0.150″** one-sided, **±0.059″ R-E-L bracketed** |
| Leon 2026 horizon | alt 8.5–12.4°, the eclipse geometry | 9 | **0.260″** (0.177–0.350) | 0.240″ | 0.098″ | 2.4 | ±0.33″ one-sided |

Built by `tools/step3_atmosphere_table.py` and `tools/step3_zenith_floor.py`. **Every
campaign needs a zenith row**, for two reasons that only became clear once Leon's was
filled in:

* it is the **control on the V/H column**. A mis-set vertical direction can only drive a
  measured V/H toward 1, so isotropy at zenith (1.06, confirmed by a sensor-axis split of
  y/x = 0.77 that needs no ephemeris) proves the 2.4 at alt 9–12° is not an artefact of
  the decomposition. The science field's own displacements reproduce it (V/H 2.5 raw, 2.1
  after the nuisance);
* it is the **floor under the null test**. The zenith null — ten constant-only pairs of
  consecutive same-night fields, 2 min 34 s apart, which is the science chain's own
  cadence — returns **±0.12″** (±0.14 at a 42-star sample) from a residual field with
  essentially no atmosphere in it. Refitted with the scale free it falls to ±0.06: half of
  that floor is plate-scale *drift* over the 2.6 minutes (7 ppm), the rest is shape. Above
  it, Bruns' ±0.15 leaves only **±0.05″** of atmosphere and Leon's ±0.33 leaves ±0.30.
  Quote the total, report the decomposition beside it, and do not subtract the floor — it
  is measured on one instrument at one focus with corrections off.

**The floor is angular, not a sampling artefact, and the magnitude cut is what mixes
photon noise into it** (`tools/floor_vs_sampling.py`). Binning the same residuals by
magnitude splits the structure from the noise, because structure displaces bright and faint
stars alike while centroid noise does not. At G 8-10 the three floors agree to 1.4x in
ARCSEC (Leakey 0.072, Leon 0.068, Bruns 0.052) and spread 2.5x in PIXELS (0.063, 0.031,
0.025) — so it is an angle, and Leakey's 1.9x finer sampling buys nothing. At the G ≤ 11
science cut, though, Bruns' number is about half his own faint-star noise (0.100 against a
0.052 structure floor; faint/bright 2.88, he being the one undersampled instrument at
1.64 px FWHM) while Leon's is untouched (0.067 against 0.068, faint/bright 0.97). **So the
G ≤ 11 column is not a structure ordering** — at the bright end it reverses to Bruns <
Leon ≈ Leakey. Keep the G ≤ 11 cut, because the budgets are computed at it, and read the
bright-end column beside it. The nulls already price each instrument's own photon noise in
their bootstrap floors.

**Two instruments now anchor the zenith row** (`tools/leakey_zenith_floor.py`): Leon's
FRA500 gives 0.067″ and ±0.12″, Leakey's Askar 65PHQ at another site in another year gives
0.078″ and ±0.12″, both isotropic. A zenith floor measured on one instrument may therefore
be used as the pedestal under another's night fields — which is what this table does. On
both, freeing the plate scale takes the null to ±0.06–0.07: **the floor is about two-thirds
plate-scale drift.** Group a campaign's fields by focus before pairing them; Leakey's scale
steps 610 ppm at a refocus, and the one pair that straddles it reads −0.76″.

**Match the null's construction to the science design, or it charges the wrong thing.**
The one-sided null (field against the previous field) is the analogue of Leon's one-sided
CAL_piLeo. Bruns' eclipse field was fitted against the *mean* of L (before) and R (after),
which cancels anything linear in time across the gap; redone that way, his night null is
**±0.087″** instead of ±0.150 and does not move when the scale is freed — a bracket
removes the drift and leaves pure shape (`tools/matrix_bruns/b17_bracket_null.py`). The
one-sided ±0.150 stays in the record as what a one-sided design would have inherited.
**Better still, rehearse and null the science SEQUENCE.** Bruns' night fields run R, E, L on
a two-minute cadence with the eclipse pointing midway between the calibration pointings, so
the null can be built exactly as his eclipse fit was: **±0.059″**, against ±0.150 one-sided.
The one-sided means against R and L alone are −0.039 and +0.036″ — equal and opposite, a
gradient across the sky — and the bracket's mean is −0.001. That cancellation is what two
calibration fields either side buy, and it is measurable only if the rehearsal repeats the
sequence (`tools/matrix_bruns/b17_lr_bracket_null.py`).
**Never take the L−R half-split as the scale term AND an atmosphere term.** The 45 ppm
L−R split is the measured size of the sky gradient the bracket cancels, and the bracketed
null measures its residue directly; charging both counts the same gradient twice. Use the
bracket's own HC3 (10.3 ppm) with the null, or the 22.5 ppm half-split alone, never both.

**The null does not double-count the scale term**: its overlap is leverage × the
reference field's own scale uncertainty, which is 0.02–0.03″ for the Bruns and zenith sets
(references at 1–5 ppm) and 0.05–0.10″ inside Leon's ±0.33 (H3 references at 5–9 ppm) —
under 0.02″ in quadrature on any total. Its overlap with the stat term is its own bootstrap
floor, 0.03–0.10″, likewise negligible in quadrature.

The horizontal components are nearly equal; the **vertical** component is what grows
toward the horizon, and it is the component that couples to a radial deflection signal.
That is the whole explanation for why Leon was limited and Bruns was not.

## 3. The Bruns 2017 processing settings, as run

Reduction of record: `tools/matrix_bruns/b17_bruns_method.py`, tree
`D:\MEE2024 output\MEE_output\matrix_bruns2017_brunsmethod\`.

**Stage 1** (both masters, and the L/R calibration fields):

```
sensitive_mode_stack=True        centroid_gaussian_subtract=True
centroid_gaussian_thresh=4.0     min_area=2          sigma_subtract=0.0
centroid_refine_window=False     background_subtraction_mode=Gaussian
delete_saturated_blob=False      remove_edgy_centroids=True
```

`Gaussian` + `centroid_refine_window=False` is the **Bruns-compatible convention**: it is
what reproduces his published value. The background mode is worth ~19 ppm of plate scale
and the estimator under 2 ppm — the background is the lever, not the estimator.

**Preprocessing** (tool-level, `b17_s0.py`): tier-mean coronal model, 10 px Gaussian blur,
subtracted per frame, +2000 ADU pedestal, forbidden disk painted at the pedestal with
radius max(1.25 R⊙, 99th-percentile saturation radius + 10 px). Bruns' own method.

**Stage 2**: `--order cubic`, `distortion_fixed_coefficients=constant`,
`distortion_fit_tol=2.0`, `max_star_mag_dist=13`, `rough_match_threshhold=36`,
corrections ON; site 42°44′11″N 106°19′05″W, 2400 m, 13.0 °C, 770.0 mb, humidity 0.4,
λ 0.625 µm; frozen reference = the L+R8 bracket (imported scale **2.0867533 ″/px**, HC3 12.64 and
13.46 ppm on the two fields → **9.23 ppm** on their mean).

**Stage 3 / estimator**: Method 1 (imported scale), no nuisance term (Bruns had none);
catalogue G ≤ 11 (the standard; the error rises sharply beyond — 0.418″ per-star scatter
at G 11–13 against 0.16″ below), doubles dropped at 10″, blends dropped, R > 1.45 R⊙;
close-in pair linked from the 0.09 s master by the **14-star** offset (the new standard;
Bruns used 7, and the choice is L-neutral at 0.013″).

**Variants on record** (all tables in the same tree):

| cut | link | N | L (″) | total σ |
|---|---|---|---|---|
| G ≤ 10.5 | 14 | 22 | 1.794 ± 0.062 | 0.111 |
| **G ≤ 11** | **14** | **27** | **1.764 ± 0.060** | **0.113** |
| G ≤ 11 | 7 (Bruns') | 27 | 1.777 ± 0.064 | 0.115 |
| G ≤ 13 | 14 | 39 | 1.718 ± 0.086 | 0.132 |

Bruns 2018 published **1.7512 ± 3.4 % = ±0.060 ″**. Every variant agrees within 0.05″. The
totals are chart revision 12 (2026-09-02), carrying the **R-E-L bracketed** atmosphere term
±0.059 rather than the one-sided ±0.150 (see §2), and the bracket HC3 of the convention of
record, **9.23 ppm**, rather than the windowed pair's 10.3.

## 4. What has to be true before the three datasets are compared

* **one convention across cells.** Leon's headline was reduced windowed+annular; measured
  in this convention it moves −0.08″, so the two are already comparable, but Mexico must
  be reduced here from the start;
* **one chart set.** These four charts, same construction, same error decomposition;
* **the atmosphere floor measured per campaign** by §2, not inherited;
* **the magnitude cut justified per instrument** — G ≤ 11 is measured for Bruns, and the
  same scan should be run on each new dataset rather than assumed.

## 5. Known defects the reductions of record still carry

1. **The coronal-model trench**: the tool-level preprocessing blurs the tier mean with the
   saturated core included, over-subtracting a ~30 px ring just outside it. Fixed in the
   pipeline (`subtract_coronal_background`, masked blur), not yet in the tools.
2. **Rim artefacts in the alignment** (ROADMAP F29): with the pipeline mask off, per-frame
   centroids are unfiltered, and 0.8 rim detections per frame reach the alignment against
   12.8 real stars. Per-frame, not per-star, so it moves the star sample rather than
   biasing astrometry.
3. **F28** blocks the pipeline path from replacing the tool chain: the per-frame coronal
   model leaves too few stars to plate solve. Until it is closed these results cannot be
   reproduced from the exe.

## 6. The Leon 2026 processing settings, as run (added 2026-09-01)

Reduction of record: the tool chain of `docs/STEP3_2026.md`, tables and charts in
`D:\MEE2024 output\MEE_output\step3_record\`, copied into `RECORD/leon2026/`.

**Stage 1** (CAL_piLeo and the four science tiers alike; `step3_s0_v4.py`,
`cal_pileo_step2/canonical_16f_night2refs`):

```
sensitive_mode_stack=True        centroid_gaussian_subtract=True
centroid_gaussian_thresh=4.0     min_area=2          sigma_subtract=0.0
centroid_refine_window=True      centroid_window_sigma=2.0
background_subtraction_mode=annular
delete_saturated_blob=False      remove_edgy_centroids=True
```

`annular` + `centroid_refine_window=True` is Leon's convention. The estimator choice is
per-instrument on purpose: Leon's optics show a brightness-dependent centroid bias of
172–299 mas beyond r = 2500 px in twelve zenith fields out of twelve
(`docs/LEON_2026-08-11.md` §18.3), which the windowed estimator removes. The background
mode was an inherited default until the A/B of 2026-09-01 (`tools/step3_background_ab.py`,
`step3_bg_ab/`) measured it on Leon alone; see `docs/STEP3_2026.md` for the numbers.

**Preprocessing** (tool-level, `step3_s0_blursub.py` then `step3_s0_v4.py`): per-tier
unshifted mean, 10 px Gaussian blur, subtracted per frame, +2000 ADU pedestal; saturated
pixels dilated 10 px and painted; forbidden disk at max(1.25 R⊙, tier 99th-percentile
saturation radius + 20 px) → 0.1 s 632, 0.3 s 699, 0.6 s 756, 1.2 s 821 px (the record
was measured at margin 20; the margin is 10 in the current tool), centred on the
ephemeris Sun (3171, 3232).

**Stage 2**: `--order cubic`, six 08-12 zenith references frozen at
`distortion_fixed_coefficients=quadratic` for CAL_piLeo (`distortion_fit_tol=1.0`,
imported scale **2.2054043 ″/px**, 74 stars, rms 0.5318 ″, observation_time 18:29:35);
the science tiers `distortion_fixed_coefficients=constant` against that CAL result,
`distortion_fit_tol=2.0`, `max_star_mag_dist=13`, `rough_match_threshhold=36`,
corrections ON; site 42.740470 N, −5.613780 E, 1101 m; CAL 30.5 °C / 896.6 hPa, science
29.2 °C / 896.7 hPa, humidity 0.208, λ 0.62 µm; tier mid-times 0.1 s 18:28:32, 0.3 s
18:28:34, 0.6 s 18:28:33, 1.2 s 18:28:32.

**Stage 3 / estimator** (`tools/step3_s2_union.py`, the union of the 0.6 s and 1.2 s
tiers): **the two-witness admission rule — a star is admitted only if BOTH tiers detected
it**, so the cross-tier consistency vet can act on every member (adopted 2026-09-02;
36 stars of the 42 matches). Gaia G ≤ 11, epoch 2026.61, refraction/aberration corrected
per tier time;
doubles dropped at 10 ″; blends dropped; gates 8 ″ (offset pass) then 4.5 ″ (collect — it
must exceed the anchor's 2.4–2.9 ″ physical displacement); per-star median across
tiers; cross-tier vet at 3×MAD with a 1.5 ″ floor (it removes G 9.10); R > 2 R⊙ about
(3171, 3232); Method 1 with the imported scale; **vertical-deg-2 nuisance on** (the S1
gate's verdict); the below-Sun anchor G 7.71 at 2.17 R⊙ in; 200-star bootstrap, seed 3.

**Error decomposition, matching cell 1's** (L = 1.914 ± 0.637 ± 0.675 ± 0.33, total
0.985): stat from the bootstrap; **scale** = the
CAL_piLeo HC3-class 25 ppm × the leverage measured by injecting a 1 ppm uniform scale
error into this field's geometry with this estimator (0.0278 ″/ppm with the nuisance;
naive eq-23 h·R⊙ gives 0.0257); **atmosphere** = the S1 gate's max over the three M5
night windows (±0.33), with the cell-1 statistic (rms over windows, ±0.22) stated beside
it. The scale term had been left out of the quoted Leon headline; it is the largest term.

