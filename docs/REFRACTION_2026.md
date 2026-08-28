# The Leon refraction data: what it is, and a plan for using it

*(Branch record for `refraction-leon-2026`. The original strategy copy lives at
`D:\MEE2024 output\MEE_output
efraction\REFRACTION_2026_STRATEGY.md`; this file is the
living version and will accumulate results as M2–M8 run.)*

**Date:** 2026-08-26. Status: **strategy — nothing here has been measured yet** except the
inventory (§1), the weather extraction (§2) and the pilot solvability run (§5.M1). Sources:
`leon_horizon_v1.15.scs`, `leon_refraction_mosaic_v2.scs`, the frame headers on
`G:\Leon Aug 2026`, the logger file on `I:\Leon location and weather data`, and
`docs/LEON_2026-08-11.md` (§4.4, §5, §11, §16, §18, §19) with `CAL_PILEO_STEP2.md`.

The purpose, in one sentence: the eclipse was measured at altitude 9.5–9.9°, where the
pipeline's standard refraction model (astropy/ERFA class) is asked to remove a ~10 000 ppm
vertical field compression with almost no free parameters to absorb its error — and these
two datasets exist to **measure that model against the real atmosphere at the real
sightlines**, rather than assume it.

This analysis is deliberately non-standard, probably eclipse-2026-specific, and should live
on its own branch as auxiliary tooling (proposed: branch `refraction-leon-2026`, code under
`tools/refraction/`, running record in `docs/REFRACTION_2026.md`). Nothing under `mee2024/`
changes, so by the ROADMAP §6 classification it cannot move a released number.

---

## 1. What exists on disk (measured from headers, 2026-08-26)

All frames: 6 s exposure, gain 101, offset 50, sensor at +10 °C, red filter, camera clamped
at the eclipse rotation. Every planned block is complete — no aborts, no missing fields.

### The horizon sets — three windows, two nights

| window | UTC | fields | frames | FOCUSPOS (steps) | FOCTEMP (°C) |
|---|---|---|---|---|---|
| **N1** (pre-eclipse night) | 08-11 23:16 – 23:31 | H1, H2, H3 | 3 × 45 | **17049** | 29.1–29.4 |
| **N2** (post-eclipse, before mosaic) | 08-12 22:31 – 22:46 | H1, H2, H3 | 3 × 45 | **17041** | 30.6–31.1 |
| **N3** (post-eclipse, after mosaic) | 08-13 00:22 – 00:37 | H1, H2, H3 | 3 × 45 | **17037** | 30.1–30.3 |

The fields (fixed alt-az GOTO, then sidereal tracking):

| field | alt (°) | az (°) | role |
|---|---|---|---|
| H1 | 9.6 | 281 | the Sun's mid-totality position; covers the lower ~70 % of the science frame |
| H2 | 11.6 | 281 | +2° — brackets the science band from above |
| H3 | 10.5 | 271 | the CAL_piLeo sightline (CAL_piLeo reduced at alt 9.87°, az 269.9°) |

**N2 and N3 are the same observing night**, bracketing the mosaic by ~2 h: the post-eclipse
sequence was horizon (22:31) → zenith set (22:48) → mosaic (23:10–00:20) → horizon again
(00:22). So the "two post-eclipse sets" give a same-night repeat under a measured 2.2 K /
0.9 hPa atmospheric change — a built-in weather-sensitivity lever — while N1 is the
different-night control.

**The focus check passes exactly where it can pass** (the check requested before planning):

| block | FOCUSPOS (steps) | same night's zenith set | match |
|---|---|---|---|
| N1 horizon | 17049 | zenith 08-11: **17049** | exact |
| N2 horizon | 17041 | zenith 08-12: **17041** | exact |
| N3 horizon | 17037 | (no zenith set at 17037) | 4 steps from 08-12's — inside the EAF's ~15-step backlash (§12.4) |
| CAL_piLeo, eclipse day | 17170 | — | the 121–129-step daytime offset of §10.2 (17170−17049 = 121, 17170−17041 = 129) |

So N1 and N2 are at **identical recorded focus** to their own night's distortion solution —
the zenith cubic imports with no focus caveat at all — and N3 carries a 4-step,
inside-backlash caveat.

### The meridian mosaic — one run, complete

80 fields × 5 frames, alt 5° (south) → 88.92° (zenith straddle, fields 40/41) → 5° (north),
UTC 08-12 23:10:26 → 08-13 00:20:33 (70 min; §4.4's "23:10–23:50" window should be extended —
a proposed correction, §8). Every field has all 5 frames.

**FOCUSPOS steps 17041 → 17037 exactly at field 63 — the pier flip** (00:05:21 UTC; the
script predicted the flip between fields 62 and 63, pole at alt 42.60°). A 4-step change
coincident with the mount reversing the gravity load on the focuser, inside backlash: most
plausibly the *reading* settled, not the optics. Whether the optics moved is itself
measurable from the mosaic (§5.M4): fields 62/63 are adjacent in altitude and time across
the flip, and every matched pair from 63 on straddles it.

### Weather (logger, 30 s cadence, local = UTC+2; extraction reproduces §4.4)

| window | n (records) | T (°C) | RH (%) | P (hPa) |
|---|---|---|---|---|
| N1 horizon | 32 | 23.5 ± 0.1 | 37.1 | 896.3 |
| N2 horizon | 32 | 23.7 ± 0.3 | 37.9 | 897.5 |
| mosaic | 142 | 21.7 ± 0.5 | 48.6 | 898.4 |
| N3 horizon | 32 | 21.5 ± 0.1 | 50.0 | 898.4 |
| eclipse totality (ref) | 6 | 30.5 ± 0.1 | 20.8 | 896.7 |

Placement per §4.4: N1 was logged from the box (humidity reads ~10 points low; night box
temperature is trustworthy per §4.2); everything on the post-eclipse night is from the
spreader in free air — the best-instrumented windows of the campaign. Two logger caveats,
both already established: the file's own "Original time zone UTC+1" header is wrong (the
timezone was established in §4), and the file must never be averaged whole — its last hours
(pressure rising to 926.8 hPa) are the descent from site.

## 2. Why refraction is the right suspect, made precise

The pipeline applies the standard model to catalogue positions ("corrections on"). What the
fit can absorb of the model's error depends on which polynomial orders are free — and the
three-step chain of §16 progressively freezes them:

| step | free orders | exposed to model error at |
|---|---|---|
| 1 (zenith) | all, through cubic | nothing (model ~flat at 80°) |
| 2 (CAL_piLeo) | constant + linear + quadratic | **cubic and above** |
| 3 (eclipse field) | **constant only** | **linear and above** |

The standard-model signal sizes at alt 9.6°, on this 2.5°-tall frame (from R ≈ k·tan ζ with
k = 58.3″; the pipeline's model is better than this, the orders of magnitude are right):

| term | size | absorbed by |
|---|---|---|
| absolute refraction | ~330″ | the free constant, everywhere |
| vertical compression (linear) | **~10 100 ppm** | step 2's free linear; **frozen at step 3** |
| horizontal compression (linear) | ~283 ppm (constant with altitude — the "zenith floor" measured 2026-08-26) | same |
| quadratic sag across the frame | ~6″ | step 2's free quadratic; frozen at step 3 |
| **cubic structure across the frame** | **~0.7–1″** | **nothing — frozen from the zenith night at both steps** |
| time drift of the compression | ~600–700 ppm/deg × 10.9″/s descent ≈ 1.8 ppm/s | `observation_time` (exposure-weighted mid-point) |

So: a fractional model error ε costs step 2 roughly ε × 1″ of *structured, unfittable*
residual (against its observed 0.53″ rms — LEON §5's "0.5″ of curvature survives the fit" is
this line), and costs step 3 ε_differential × 45″ across the frame at the linear term alone,
where ε_differential is the model error *difference* between the CAL_piLeo reduction
(alt 9.87°, az 270°, 18:29:34 UTC) and the eclipse frames (alt 9.5–9.9°, az 281°, minutes
later). The differential is what matters because the imported low orders already carry the
model error as realized on the cal sightline — §19.2's common-mode principle. **H3 − H1 is,
by design, a direct measurement of that differential geometry** (cal sightline vs eclipse
sightline, 2.1° apart in the same sky), which no amount of zenith or mosaic data provides.

What the deflection is protected by, and what it is not: the cal/science common mode
protects the *amplitude*; the night data's job is to measure the parts that are **not**
common — the sightline differential, the within-frame shape beyond quadratic, and the
stability of both against weather (N1 vs N2 vs N3 spans 2.2 K and 2.1 hPa naturally).

### The built-in mini-sweep

Each 45-frame block tracks one star field **descending 0.87° during the block**
(10.9 ″/s of altitude at az 281°, φ 42.74°). Per-frame astrometry therefore gives ~45
samples of scale-vs-altitude *on the same stars*, so catalogue and distortion errors cancel
frame-to-frame. With corrections OFF that measures the raw compression slope
(~600 ppm/deg — until now known only from the model); with corrections ON the same slope
should be ~0 and its measured value **is** the model's local error slope. Expected
sensitivity: per-frame plate-scale error of order 5–20 ppm over a 0.87° lever with 45 points
resolves the slope to ~5–10 ppm/deg, i.e. tests the ~600 ppm/deg model term at the ~1–2 %
level, per field, per window — nine independent times.

## 3. What each dataset is for (and is not)

**The horizon sets** are the local instrument: model shape and slope *at the two sightlines
the science depends on*, at the exact focus of their night's distortion solution, three
times, under measured weather. They cannot measure the daytime/eclipse atmosphere (§7).

**The mosaic** is the global instrument: the model's shape over 5°–89° of altitude (97× in
differential refraction), the azimuthal-homogeneity assumption (40 matched pairs at equal
zenith distance on opposite azimuths — any pair difference is azimuthal asymmetry, not
refraction), and the pier-flip diagnostic at field 62/63. It validates the model *class*;
it does not sample the science azimuth (281°) below the pole — which is exactly why the
horizon sets exist, and why the matched pairs matter: they bound how far meridian
conclusions transfer to other azimuths.

**The three windows together** are the stability instrument: same measurement under
2.2 K / 2.1 hPa / 12-point-RH natural variation, plus a same-night repeat (N2 vs N3) and a
different-night control (N1).

## 4. The decision the horizon data settles first

The largest unexplained term in the step-2 budget is that CAL_piLeo's daytime per-star
scatter (0.53″ rms, flat with magnitude, horizontally elongated σ_y/σ_x ≈ 1.34) is ~8.5×
the same rig's night-time zenith figure. The first horizon reduction settles which of three
worlds we are in, and the rest of the program branches on it:

| night result at H1/H3 | reading | consequence |
|---|---|---|
| rms ≈ 0.06–0.12″, isotropic (zenith-like) | the sightline is benign; the daytime excess is **eclipse-specific** (coronal sky gradient, thermal transient, wind) | refraction model likely fine; program closes early with a bound; the eclipse-day mystery moves to daytime physics |
| rms ≈ 0.3–0.6″ with the **same horizontal anisotropy** | it is the **sightline** — low-altitude differential image motion / turbulence anisotropy, day and night alike | scatter is irreducible per-frame; stacking statistics and exposure choice for step 3 become the lever; model may still be fine |
| rms elevated with a **structured vertical residual** after corrections ON | **refraction model error** isolated | measure it, template it (M6), correct step 2/3 or bound them |

These are distinguishable in the first afternoon of M2, on stacked blocks alone.

## 5. The measurement program

Everything below runs the existing pipeline (`python -m mee2024.cli stack` / `distortion`)
through thin drivers in `tools/refraction/` — the same pattern as the step-2 session, no
package changes. Distortion references: the **same night's six zenith files** (exact-focus
match, §1); a control pass with all 12 prices that choice. Weather per window from §1's
table; `observation_time` per frame or per block, exposure-weighted, mid-exposure per the
script's own instruction.

**M0 — inventory and environment. Done** (this document, §1–§2;
`refraction/INVENTORY.csv`).

**M1 — pilot solvability. Done** — see result below. Gate: if a 10-frame stack at alt 9.6°
solves and fits, everything else is mechanics.

**M2 — the horizon ladders** (the core). For each window × field: stage 1 per frame (45
single-frame reductions) and per block (one 45-frame stack); stage 2 on each at
`distortion_fixed_coefficients=quadratic`, corrections ON and OFF (same centroids, two
cheap fits). Outputs per window×field: S(t) and S(alt) with corrections OFF (raw slope vs
model) and ON (residual slope = model error); rotation(t); rms(t); the stacked residual map
in sensor coordinates; per-frame scatter statistics (the differential-image-motion
measurement, and the night answer to the anisotropy question). Cost: ~405 stage-1 +
~810 stage-2 runs ≈ **8–10 h machine time**, overnight, unattended. Trim to N2 + N3 first
(the eclipse-night atmosphere) if a first look is wanted sooner; N1 follows as control.

**M3 — the sightline differential.** H3 − H1 per window, both correction states: Δscale
(ppm), Δresidual-map (mas), Δslope (ppm/deg). This is the number that feeds Method 1's
transfer term — the non-common-mode part of the refraction correction between the cal and
eclipse sightlines — and its night-to-night spread is its own error bar. H2 − H1 gives the
same differential in pure altitude at fixed azimuth (the science band's top vs bottom;
the upper 30 % of the science frame is reached by H1–H2 interpolation, not extrapolation).

**M4 — the mosaic sweep.** Stage 1 per field (5-frame stacks), stage 2 at quadratic-free,
corrections ON (primary) and OFF (one in five fields, for the raw curve): plate scale,
rotation, rms, FWHM and star count vs altitude over 5–89°; the corrections-ON scale should
be **flat** — the altitude where it departs, and the shape of the departure, is the model's
validity boundary at this site. Then the 40 matched-pair differences (azimuthal asymmetry
bound), and the field-62/63 discontinuity (did the pier flip move the optics, or only the
focuser reading?). Cost ≈ **4–5 h machine**, parallelisable with M2 on a second evening.

**M5 — the step-3 rehearsal** (the highest-value single product). Reproduce the real chain
at night, where the deflection is absent and truth is the catalogue: same-night zenith
cubic → step-2-like fit on **H3** (quadratic-free, corrections ON) → step-3-like fit on
**H1** with H3's result as `--fix-distortion` reference and
`distortion_fixed_coefficients=constant` (the key exists in the mapping; verified). Every
arcsecond of structure in H1's residual map is then exactly the class of error the eclipse
reduction will inherit. Project that map onto the eclipse fit's own basis — the 1/r field
around the eclipse-day Sun position within the frame, weighted by the actual expected star
geometry (catalogue to V = 11) — and the output is **a forecast of δL, in arcsec and as %
of 1.75″, from refraction-model error, before step 3 is ever run**. Three windows give
three forecasts and a spread.

**M6 — conditional: the empirical template.** Only if M2/M5 find a stable structured
residual: build the night-mean residual map (N2+N3 primary), smooth it (low-order in
frame coordinates), and apply it as a pre-correction to (a) the CAL_piLeo step-2 rerun and
(b) later, step 3 — then re-measure plate scale, rms, anisotropy, and the M5 forecast with
the template in. Acceptance rule, fixed now: the template must *reduce* the night H1
constant-only residual in all three windows, not merely in the window that built it. If M5's
forecast is small, M6 is skipped and its place is an error-budget line.

**M7 — byproducts** (cheap, from M2/M4 outputs): FWHM and limiting magnitude vs altitude
and night (the SCI_ladder exposure-tier check the horizon script was also designed for);
sky level vs altitude; the differential-image-motion statistics at airmass 5.8 (what a 45×
stack actually averages); periodic-error signature in the per-frame pointing constants.

**M8 — the write-up**: `docs/REFRACTION_2026.md` on the branch, LEON register; conclusions
that touch the main-line reductions (step 2/3 error budget lines, any §4.4/§18 corrections)
are carried back to `v1.4.0-dev` as proposals, not edits from this branch.

Order: M2 (N2+N3) → M3 + M5 quick-look → decide M6 → M4 overnight → N1 control → M7 → M8.

## 6. Pilot result (M1)

Ten frames of N2 H1 (60 s integration at alt 9.6°, gain 101), zenith stage-1 regime,
stage 2 quadratic-free against the 12 zenith references:

| stage 2 | stars used | rms (″) | plate scale (″/px) | HC0 (ppm) | solved alt (°) |
|---|---|---|---|---|---|
| corrections ON | 154 | 0.4264 | 2.2072262 | 29.6 | 8.94 |
| corrections OFF | 316 | 0.4291 | 2.2163670 | 10.6 | — |

Four readings, in decreasing order of confidence:

1. **It works.** A 60 s night stack at alt ~9° platesolves and fits cleanly through the
   standard chain. The program is mechanics from here.
2. **The ON−OFF scale difference is 4142 ppm** — the model's whole local correction, against
   ~4260 ppm expected from scaling §11.2's 3500 ppm at 9.87° by csc²h to 8.94°. The model's
   amplitude is right at the few-percent level out of the box.
3. **The night sightline is CAL_piLeo-class, not zenith-class**: rms 0.43″ against 0.53″
   (day) and 0.06–0.07″ (zenith). Provisionally this is row 2 of §4's decision matrix — the
   daytime scatter excess is a property of the sightline, not of the eclipse — **but** a
   10-frame stack carries ~0.5″ of within-stack refraction-evolution smear at the frame
   edges (64 s × ~9 mas/s of differential stretch), which per-frame reduction removes. The
   per-frame ladder decides between "atmosphere" and "stacking smear"; do not quote row 2
   yet.
4. **A 155 ppm tension, flagged for M2 to settle**: corrections-ON at H1 gives 2.2072262,
   while the same night's zenith field at the *identical* focus (17041), corrections-ON,
   gave 2.2068874 — a +155 ppm gap where near-zero is expected if the model were exact,
   i.e. ~3.7 % of the local refraction term, at ~4σ of the pilot's own (HC0) error. Possible
   readings: genuine model error at airmass ~6, within-stack smear bias, or the tol-1.0
   selection (the ON fit kept only 154 of 316 stars — the cut rejected half the field, so
   the tolerance is shaping the sample at this altitude and M2 should sweep it).

Practical notes for M2 from the pilot: single frames will carry ~150–250 usable centroids
(fine); `distortion_fit_tol = 1.0″` is too tight for mapping work at this altitude — run
the ladder at 1.0 / 2.0 / 5.0 and keep the maps from the loose end; the AVX pointed ~0.6°
below the commanded altitude (solved 8.94° vs GOTO 9.6°), harmless since all analysis uses
solved positions.

## 7. What this data cannot do — stated up front

- **It cannot measure the daytime or eclipse-time atmosphere.** Totality sat 7–9 K warmer
  than the night windows, with its own cooling transient (§4.4: a damped ~3 K drop) and
  whatever boundary-layer response the eclipse itself drove. The night data validates the
  model's *shape*; the amplitude transfers by P/T scaling (±2 K → ±1.2 ppm where it
  matters, §19.2); an eclipse-specific *shape* anomaly is irreducible from any of this and
  stays a caveat on step 3 whatever we find.
- **The mosaic does not sample the science azimuth** below the pole; its matched pairs
  bound, but do not eliminate, the azimuth question. The horizon sets carry that load.
- **Night-to-night comparisons are not same-star**: a fixed alt-az at a different clock
  time is a different star field (N1 vs N2 H1 differ by ~10° of RA). Comparisons live in
  the sensor/alt-az frame — scale, slope, residual maps — not in per-star differences,
  except within a block, where the mini-sweep is same-star by construction.
- **6 s at gain 101 will clip the brightest stars** at these star densities. The step-2
  session measured the cost of losing single anchors (leverage, not rms); with hundreds of
  stars per frame the effect is diluted ~10×, and the F16 per-frame mask from
  `analysis/saturation.py` is reusable if it matters.

## 8. Corrections and additions this recon already owes the record

Proposals for `v1.4.0-dev` (not edits from this branch): §4.4's mosaic window extends to
00:20:33 UTC (140 logger records, not 80 — the mean moves ≲0.1 K / 0.1 hPa); §4.4 could
usefully note N2/N3 bracket the mosaic on one night; the pier-flip focus step
(17041 → 17037 at field 63) belongs wherever §12.4 discusses backlash, whatever M4 finds.

## 9. Branch, code, and effort

- **Branch `refraction-leon-2026`** off `v1.4.0-dev`, created when approved. Contents:
  `tools/refraction/` (inventory.py, weather.py, drive_horizon.py, drive_mosaic.py,
  residual_maps.py, step3_rehearsal.py, report.py — thin drivers over the CLI),
  `docs/REFRACTION_2026.md`. **No changes under `mee2024/`**, so nothing on the branch can
  alter a pipeline number; if M6 ever graduates into the package, that is a
  results-changing change and takes the full §6 validation path.
- Reductions land under `D:\MEE2024 output\MEE_output\refraction\`.
- Machine time: M2 ≈ 8–10 h + M4 ≈ 4–5 h, both unattended overnight; analysis on top.
- Everything above is reproducible from `INVENTORY.csv`, the logger file, and the two
  `.scs` scripts; no hand steps.


---

## 10. M2 results — the horizon ladders (run overnight 2026-08-26/27)

All 405 frames of all nine field-windows reduced per frame, corrections ON and OFF,
zero failures; per-frame results in `refraction/perframe_results.csv`, per-field summary in
`refraction/m2_fieldwindow_summary.csv`. Corrections-ON zenith references: 08-12 measured
directly (2.2068874 ″/px); 08-11 derived via §18.1's −221.9 ppm shift (sd 5.4 ppm).

| win/field | UTC | alt mean (°) | ON slope (ppm/deg) | ON offset vs zenith (ppm) | rms, median per frame (″) |
|---|---|---|---|---|---|
| N1/H1 | 23:16 | 10.32 | −145 ± 33 | +115 ± 10 | 0.71 |
| N1/H2 | 23:21 | 12.30 | +2 ± 37 | +65 ± 9 | 0.69 |
| N1/H3 | 23:26 | 11.21 | +130 ± 23 | +27 ± 8 | 0.69 |
| N2/H1 | 22:31 | 8.60 | −107 ± 28 | +88 ± 8 | 0.73 |
| N2/H2 | 22:36 | 11.09 | −52 ± 20 | −78 ± 6 | 0.50 |
| N2/H3 | 22:41 | 9.45 | +33 ± 21 | +15 ± 5 | 0.62 |
| N3/H1 | 00:22 | 8.55 | +125 ± 55 | +336 ± 15 | 0.73 |
| N3/H2 | 00:27 | 10.62 | −71 ± 29 | +107 ± 8 | 0.63 |
| N3/H3 | 00:32 | 9.45 | +37 ± 31 | +143 ± 8 | 0.62 |

(The mount's alt-az pointing wandered ±0.7–1.0° night to night, so "H1/H2/H3" sample
different altitudes per window — the table's alt column, from the solves, is the truth.
Corrections-OFF slopes ran −500 to −920 ppm/deg and the model removes ~85–90 % of them;
the corrections-ON columns are the residuals. All N3 offsets carry the 4-step focus caveat,
worth ≲50 ppm common-mode; N3 slopes and within-window differentials are free of it.)

**Findings.**

1. **The standard model leaves scale residuals of −80 to +340 ppm below ~12.5° altitude,
   structured and non-stationary.** Within one window the mini-sweep residual slope is
   measured to ±20–55 ppm/deg; across windows those slopes range −145 to +130 ppm/deg with
   no universal altitude dependence. The residual is weather, not a fixed model defect: it
   changed by ~100–250 ppm between 22:31 and 00:22 UTC on one night, and the pre-eclipse
   night's profile does not match either epoch of the post-eclipse night.
2. **The spatial differential — the Method-1-relevant number — is the stable part.** The
   focus-free within-window H3−H1 differential, scaled to the ~0.3° cal-to-eclipse
   altitude separation: **−29.6 ± 12 ppm (N1), −25.5 ± 10 ppm (N2), −64.6 ± 17 ppm (N3)**.
   Two windows on *different nights* agree at ~−27 ppm; the late-night epoch doubles it.
   The night-based estimate of the refraction-differential term in the CAL→eclipse
   transfer is therefore **~25–65 ppm, epoch-dependent** — comparable to or larger than
   the 23.5 ppm statistical error on the CAL_piLeo plate scale. What protects the eclipse
   reduction is *simultaneity*: CAL_piLeo and the science frames are within a minute, so
   the hours-scale wander cancels and only this spatial term survives.
3. **The daytime-scatter mystery of `CAL_PILEO_STEP2.md` §9/§11 is resolved: it was the
   sightline all along.** Night per-frame rms at 9.4–9.5° is 0.62″, and the 10-frame pilot
   stack gave 0.43″ — against CAL_piLeo's daytime 0.53″ (stacked) and the zenith 0.06″.
   The "8.5× the night-time figure" compared day-at-9.9° against night-at-80°; at matched
   altitude the day/night ratio is ~1.2. Partial frame-averaging (0.62″ per frame →
   0.43″ over 10 frames, not the 0.20″ of independent noise) says roughly half the
   per-star error is quasi-static over minutes — atmospheric microstructure that stacking
   cannot remove, which is also why CAL_piLeo's error floor (§5 there) is what it is.

**Consequences for the program.** M6's template idea is dead at the lowest altitudes (the
thing it would template is non-stationary) and weakened everywhere; the effort moves to
M5 — measuring what step 3 actually inherits, with the ~25–65 ppm differential as the
input range — and to §16.1's check of the 1/R residual gradient in the eclipse data
itself. M4 (the mosaic) gains a specific extra job: fields 1–62 vs 63–80 straddle the
4-step focus change, so the matched pairs calibrate the N3 focus confound directly.


## 11. M5 results — the step-3 rehearsal (2026-08-27)

The full chain, rehearsed at night where the deflection is absent: same-night zenith cubic
→ step-2-like fit on a mid-block 10-frame H3 stack (the CAL_piLeo analogue: 470–778 stars,
rms 0.41–0.46″ — sightline-class, as §10 predicts) → constant-only fits of all 45 H1
frames per window against that reference, `distortion_fit_tol = 999` as §16 prescribes.
Per-star median residuals (≥20 frames each), MAD-clipped, smoothed to a cubic surface,
sampled at the real eclipse-star geometry, and fitted with the Method-1 estimator
(offsets + roll + L/R). The machinery validates end to end: the empirical sky→sensor
affine reproduces the 0.74° Sun offset to 3 px, and an injected L = 1.000″ is recovered
to 1.0000 in both estimator variants.

**The real step-3 field, for the record**: 132 stars G ≤ 11 outside 2 R⊙ (2.0–10.3 R⊙),
Sun at px (3208, 3293); **F&L h = 19.8 R⊙²**, i.e. **1.07 % of L per ppm** of imported
plate-scale error by the naive eq.-23 route — the highest-sensitivity row of
`STAGE3_THEORY.md` §4's table now has its measured value.

| window | stars used | field rms (″) | linear part (ppm) | **δL, Method 1 (″)** | δL (% of L) | δL, scale-free (″) | boot se (″) | clipped |
|---|---|---|---|---|---|---|---|---|
| N1 | 212 | 0.604 | −60 | **−0.968** | **−55 %** | −0.066 | 0.051 | 2 |
| N2 | 106 | 0.662 | −81 | **+0.229** | **+13 %** | −0.723 | 0.100 | 1 |
| N3 | 115 | 0.849 | −130 | **+0.410** | **+23 %** | +3.419 | 0.261 | 3 |

(The linear parts track §10's independently measured H3−H1 differentials — −88/−73/−193
ppm — as they must; that is the cross-check that the rehearsal measures the same physics.
N1's reference sat *above* its field, the inverted geometry, hence its opposite sign and
stronger shape content.)

**Findings.**

1. **A step-3-style fit inheriting a calibration from a field ~0.9° away picks up δL of
   0.2–1.0″ (13–55 % of L), sign and size set by that hour's atmosphere.** Scaled to the
   real CAL-to-eclipse separation (~0.5–0.6° between frame centres), the night-based
   forecast for the refraction-inheritance term is of order **0.1–0.6″** — the largest
   single threat to L now quantified anywhere in this project.
2. **The geometry helps, by a measured factor ~6.** For scale-like residual content the
   coupling is ~3 mas of δL per ppm (N2: 0.229″/81 ppm; N3: 0.410″/130 ppm) against the
   naive h·δS bound of 18.7 mas/ppm — the free offsets+roll absorb most of a
   frame-centred gradient, and the Sun sits well off frame centre. But *shape* content
   couples up to 5× harder (N1: 16 mas/ppm), so the suppression cannot be assumed, only
   measured — which is what this rehearsal is for.
3. **The scale-free estimator is catastrophically fragile here, demonstrated on real
   residuals**: −0.72″ to +3.42″ (−41 % to +195 % of L) across windows. F&L's eq.-7/12
   verdict and `STAGE3_THEORY.md` §6's cluster mechanism, reproduced at night with no
   deflection in the sky. Method 2 at this altitude is a diagnostic, never a result.
4. **`tol 999` admitted 1–3 persistent catalogue mismatches per window (up to 64.5″),
   which survive frame averaging** because the same wrong star matches every frame. An
   early version of this analysis was wrecked by them (δL of −40″) until MAD-clipped.
   The real step-3 field has 132 stars: expect the same 1–3 junk entries, each reaching
   the deflection fit unchallenged. **F16-at-step-3 is not optional**, and the eclipse
   analysis needs the same robust-aggregation protection — this is §16.8's warning with
   a measured casualty count.

**What this reframes.** With a single one-sided calibration field at this altitude, the
inheritance term cannot be calibrated away from night data (§10: it is weather). The
defences that remain are (a) **simultaneity** — CAL_piLeo and the science frames are
within a minute, killing the temporal part; (b) **the eclipse data's own diagnostics** —
the Method-1-vs-Method-2 gap (linear in exactly this error, per `STAGE3_THEORY.md` §5.4)
and §16.1's 1/R-residual-gradient check, both of which are now essential rather than
advisory; and (c) the spatial part being the *repeated* −27 ppm/0.3° of §10 rather than
the late-night extreme, which would put the term nearer 0.1″ than 0.6″. Which defence
holds is measurable only from the eclipse frames themselves — step 3 should be run with
all three diagnostics from the start.


## 12. M3 results — the residual maps (2026-08-27)

Per-star decomposition of all nine field-windows from the M2 corrections-ON quadratic-free
fits: the **quasi-static** residual (median over the ~45 frames — what stacking keeps) and
the **per-frame jitter** (scatter about it — what stacking averages), rotated into the
local alt-az frame. Figures: `refraction/m3_maps/m3_quiver_maps.png`,
`m3_vertical_profiles.png`; per-field statistics in `m3_stats.csv`.

Condensed (ranges over the nine field-windows, alt 8.5–12.4°):

| component | vertical (″) | horizontal (″) | anisotropy V/H (ratio) |
|---|---|---|---|
| quasi-static rms | 0.15–0.32 | 0.07–0.13 | ~2.3 |
| jitter, median per 6 s frame | 0.43–0.61 | 0.19–0.31 | ~2.5 |
| jitter after a 45-frame stack | 0.06–0.09 | 0.03–0.05 | — |

**Findings.**

1. **The unabsorbed structure is a wavefield, not a polynomial.** The vertical-residual
   profiles oscillate with **0.2–0.3″ amplitude on ~0.5–1° altitude scales**, and where
   two fields overlap in altitude (minutes apart, ~10° apart in azimuth) the curves do
   not agree — the structure is not a function of altitude alone. The maps show the same
   thing spatially: coherent patches of common vertical displacement. This is
   gravity-wave-class modulation of the refracting layers, and it explains at a stroke
   why §10's within-window slopes wander between windows: a 0.87° mini-sweep samples one
   or two phases of a wave, and the fitted "slope" is whatever phase it caught. No
   polynomial order fixes this, and no night template transfers it.
2. **Stacking hits a floor set by the quasi-static term.** Jitter integrates down
   (0.5–0.7″ per 6 s frame → 0.07–0.10″ over 45 frames) but the 0.17–0.35″ quasi-static
   field does not. For the eclipse science frames the same arithmetic applies: beyond
   ~20–40 frames the error budget is the wavefield, not photon or seeing noise — which
   also sets the per-star error floor Method 1's fit will see (~0.25″, vertically
   polarised).
3. **§10's finding 3 was half right, and the correction sharpens it.** The *vertical*
   components match perfectly: CAL_piLeo's daytime stacked residual is 0.32″ vertical
   against the night quasi-static 0.15–0.32″ — the vertical part of the daytime error is
   fully explained by the sightline atmosphere. **The horizontal does not match: 0.42″
   by day against 0.07–0.13″ at night, a 3–6× excess.** So the daytime anomaly is
   specifically an extra ~0.4″ *horizontal* quasi-static term with no night counterpart.
   Candidates, untested: the totality sky-brightness gradient (the Sun sat ~11° west of
   the cal field in azimuth — a horizontal gradient pulls centroids horizontally via the
   background estimate), or wind shake. Testable cheaply from the CAL_piLeo frames
   themselves by measuring the background gradient direction — flagged for the step-3
   preparation, since the eclipse field sits even closer to the corona.

Method note: the corr-ON tol-2.0 fits carried zero persistent mismatches into these maps
(the clip count was 0 in all nine field-windows) — the M5 mismatch problem is specific to
`tol 999`, which is one more datum for the F16-at-step-3 case.


## 13. The sky-gradient test — mechanism dead, and a 90° error found and fixed (2026-08-27)

§12.3 proposed the totality sky-brightness gradient as the cause of a daytime "horizontal
excess" in CAL_piLeo's residuals. The test killed the mechanism three independent ways and
then killed the premise.

**The gradient, measured.** On the 17-frame stack: sky 3901 ADU, gradient 81 ADU/kpx —
**almost purely vertical, brighter toward the horizon** (−80.8 ADU/kpx along altitude
against +5.7 along azimuth; cos of the angle to the Sun's direction = +0.11). It is the
twilight-ring glow below the field, not the corona 11° west. Per frame the fractional
gradient is stable at ~2 %/kpx of sky while the sky itself runs 726 → 5700 ADU through
totality (and the post-C3 frames saturate whole — sky 65535 ADU by 18:30:05).

**The centroid mechanism, priced.** After annular background subtraction, the surviving
linear term shifts a σ_w = 2 px windowed centroid by g·2πσ_w⁴/F_w: median **0.016″**,
faintest-quartile 0.034″, maximum 0.050″ — 20–60× below the observed scatter, and
uncorrelated with the observed along-gradient residuals (r = +0.09 on 105 stars, below
the 0.20 two-sigma line). Dead on direction, dead on size, dead on correlation.

**The premise was wrong: the daytime anisotropy is vertical, not horizontal.** The
step-2-era `vertical_test.py` rotated residuals into the vertical frame with a hand-derived
parallactic angle that is **90° off**. The convention-proof affine (astropy alt-az of the
solved stars against their own pixels; sensor −y lands 3.1° from the local vertical) gives
CAL_piLeo **0.424″ vertical / 0.317″ horizontal** (tol 1.0) — the exact swap of the old
claim. §10 finding 3 and §12 finding 3 are amended accordingly: there is **no daytime
horizontal anomaly**. Day and night are vertical-major alike, and the daytime vertical
budget closes from night-measured numbers once integration time is accounted (the CAL
stack is ~25 s of integration against the night blocks' 270 s: quasi-static ~0.3″ plus
jitter/√17 with per-1–2 s-frame jitter scaled from the 6 s measurement ≈ 0.42″ predicted
against 0.42″ observed; the horizontal closes within a factor ~2 of the same crude
scaling). The CAL_piLeo residual is the ordinary anisotropic atmosphere at airmass ~6 —
nothing eclipse-specific, nothing instrumental.

**A capture artefact found in passing, for F14/F7 and the §19.4 file:** the first frame
of a CAL_piLeo block can carry the **previous block's exposure** while its header claims
the new one — proven by sky level: `18_29_19/00001` reads 1180 ADU (the 0.3 s level, not
1 s), `18_29_27/00001` reads 2848 ADU (the 1 s level, not 2 s), `18_29_46/00001` reads
5716 ADU (the 2 s level, not 0.3 s); `18_29_51/00001` is clean, so the rule is not
universal. Two affected frames sit in the step-2 stack; the effect on the reduction is
negligible (centroids are exposure-independent and the exposure-weighted mid-time moves
at the ~1 ppm level) but the headers lie, which matters to anything that trusts EXPTIME.


## 14. M4 results — the meridian mosaic (2026-08-27)

78 of 80 fields reduced in both correction states (stage 1 five-frame stacks, stage 2
quadratic-free over the frozen 08-12 zenith cubic, per-field logger weather). Casualties:
the alt-5° pair only — M01 failed to platesolve at all (the ~3.7 % vertical compression at
airmass ~10 exceeds the pattern matcher's tolerance; the solver's own hard floor is
between alt 5.0° and 5.6° on this rig), and M80's corrections-ON match failed the same
way. Per-field results: `refraction/m4_mosaic/m4_fields.csv`; curves: `m4_curves.png`.
A weather-lookup bug in the first pass (the logger axis shifted one hour by a naive
timestamp on a UTC+1 machine, feeding the late fields pack-up descent pressure) was found,
fixed, and 77 fields re-reduced; the M2/M5 results never used interpolated weather.

The corrections-ON deviations from the same-focus zenith-ON reference, decomposed jointly
(74 fields above alt 10°, robust fit, per-field wave scatter 14.4 ppm):

| term | value | independent cross-check |
|---|---|---|
| within-night scale drift | **+1.4 ± 0.6 ppm/min** | M2's N2→N3 offsets: +1.2–2.3 ppm/min; Bruns 2017 within-night: +1.5–2.8 ppm/min (§11 of `INSTRUMENT_COMPARISON.md`) |
| north−south asymmetry, unflipped, equal zenith distance | **−56 ± 14 ppm** | same class as the H3−H1 sightline differentials (60–130 ppm over 10° az + 0.9° alt) |
| pier flip + 4 EAF focus steps | **−41 ± 30 ppm** | ≈ −10 ± 8 ppm/step against §12.2's ~11 ppm/step |
| altitude-band residuals, 10–90° | **−49 to +39 ppm (se 21–28)** | the model-validity curve proper |

**Findings.**

1. **The standard model holds to ≲50 ppm everywhere above alt 10°** once drift, asymmetry
   and the flip are removed — a gentle ≤±40 ppm S-shape at the edge of significance.
   Below 10° the per-field deviations blow up to the ±900 ppm class (the wavefield at
   full strength plus genuine model breakdown), and by 5° the plate solver itself dies.
   The validity boundary of the whole approach at this site is **alt ≈ 10°** — and the
   eclipse was at 9.6°.
2. **The within-night drift is now measured three independent ways** and lands at
   +1.4 ± 0.6 ppm/min here. It also closes the N3 puzzle: N3's +250 ppm offset shift over
   111 minutes is mostly drift (+155 ppm expected) plus wave noise — the focus confound
   contributes at most the −41 ± 30 ppm the flip term measures.
3. **Azimuthal homogeneity fails at the −56 ± 14 ppm level (4σ)** between north and south
   sky at equal zenith distance. Refraction depends on zenith distance alone only in a
   horizontally homogeneous atmosphere; this site's atmosphere is not one, at exactly the
   amplitude the H-field differentials implied. Any transfer of a calibration across ~10°
   of azimuth carries a ~50–100 ppm atmospheric term — which is the measured, quantitative
   form of F&L's warning about one-sided calibrations, and applies directly to
   CAL_piLeo (az 270°) feeding the eclipse field (az 281°).
4. **The matched-pair scatter (22.4 ppm clean pairs) and the joint-fit scatter
   (14.4 ppm per 30 s field)** set the wavefield noise floor at mosaic depth — consistent
   with the M2/M3 quasi-static amplitudes projected onto a plate scale.

With M4 done, the mosaic's three assigned jobs are complete: the model-validity curve
(≲50 ppm above 10°, breakdown below), the azimuth bound (−56 ± 14 ppm), and the focus-step
calibration (−10 ± 8 ppm/step). The refraction program's remaining item is the step-3
reduction itself, with the M5 diagnostics wired in.


## 15. The mosaic as an instrument-stability monitor (Douglas' request, 2026-08-27)

The stability band: **27 unflipped fields at solved alt 50–80°** (M23–M36 south, M46–M58
north) — all at **FOCUSPOS 17041, identical to the same-night zenith set**, 810 s of
integration against the zenith set's 720 s, well clear of the pier flip and the sub-10°
zone. Free-cubic reruns in `mosaic/*/freecubic/`; d(3000) extractor validated against the
handoff (reproduces both nights' means to 0.13 % and the per-field sd to 0.02 pp).

**Plate scale** (corrections ON): mean 2.2070818 ″/px; raw field-to-field sd 23.1 ppm;
**drift-detrended sd 13.0 ppm** (se 2.5 ppm) against the zenith set's 4.6 ppm — the excess
consistent with 30 s visits averaging the wavefield 2× less and the band's ±40 ppm
altitude structure (§14). Band mean sits **+88 ppm above** zenith-2 ON, inside the
+60–130 ppm the measured +1.4 ppm/min nightly drift predicts for the 45–90 min separation.
**No new instrument term**: the plate scale behaves as the already-measured atmosphere
plus drift.

**The cubic does not.** Free-cubic d(3000) across the band: **2.8096 ″ ± 0.55 % (se),
per-field sd 2.83 % — 7.3 % below the same-night, same-focus zenith value of 3.0297 ″**,
on both branches alike (south −6.7 %, north −7.9 %). The deficit survives every control:

| control | result |
|---|---|
| tolerance 0.2 / 0.5 / 1.0 on three fields | moves d(3000) ≤ 1.1 % |
| stack depth: 08-12 Z1 re-reduced from 5 frames, mosaic-style | 3.124–3.139 ″ vs its 30-frame 3.099 ″ — **+1 %, not −7 %** |
| time within the band (joint fit with altitude) | −0.6 ± 2.5 %/h — null |
| altitude | **M36 at solved alt 78.7° — inside the zenith set's own 78.5–83.4° — shows the full deficit** |
| focus | identical, 17041 steps |

With altitude, time, tolerance, focus and stack depth excluded, what separates the two
sequences is **the mosaic itself began with a large slew** (zenith neighbourhood →
az 180°, alt 5°), and the deficit is a **step, not a trend**: the optical train's cubic
changed by ~7 % at fixed focus within one night, most plausibly a mechanical settling of
an element at the slew — F19's reducer-spacing mechanism supplies the lever (§9.2's
~0.64 %-per-focuser-step class sensitivity means a ~0.1 mm shift suffices). A post-flip
check (M63–M66, a second mechanical event) is **inconclusive** — those fields are too low
for clean free cubics (one fit collapsed to 46 stars); the residual candidate not excluded
is the 6 s-vs-4 s exposure difference, whose only identified mechanism (clipped-star
population) was measured at ≤ 0.65 % on the zenith data.

**What this does to the stability question.** It was "night-to-night 4.84 % at 6σ,
systematic ≥ 2.4 % and unbounded above" (§18.6/§18.9). It is now: **the cubic can step
~7 % within a single night at fixed focus on a mechanical event.** The zenith→eclipse
transfer crosses several slews, 129 focuser steps and nine hours; its cubic systematic
should be budgeted at the ≥7 % class, not ≥2.4 %. Through the measured propagation of
`cubic_into_deflection` (2.4 % of frozen cubic → +1.7 % on Method-1 L), a 7 % error is
worth **~+5 % on L** — in tension with CAL_piLeo's own residual bound on the non-collinear
cubic error (≤5.6 % at 2σ, step-2 record §7), a tension that step 3's 1/R-gradient
diagnostic is now required to resolve. The one benign reading — that the mosaic's cubic
is the *eclipse-relevant* one and the zenith sets are the outliers — would be worth a
dedicated test on the N2/N3 horizon stacks if a cubic can be coaxed from them.


## 16. The cubic step is temperature, not the slew (Douglas' hypothesis, confirmed 2026-08-27)

§15 attributed the −7.3 % cubic step to mechanical settling at the mosaic's opening slew.
Douglas proposed temperature instead — night 2 has the logger on the spreader in free air,
one focuser position (17041 steps) across the zenith set and the whole pre-flip mosaic, and
the air cooled 23.7 → 21.5 °C with a **non-monotonic** curve (a ~1.3 K recovery around
23:50–00:10), which is exactly the feature that separates temperature from generic time
drift. Figure: `refraction/night2_temperature.png`.

**Plate scale (corrections ON, refraction removed with each point's own logger weather —
what remains is the optics), 55 points:**

| test | result |
|---|---|
| vs temperature alone | **−39.7 ppm/K, r = −0.82** |
| vs time alone | +57.7 ppm/h, r = +0.53 |
| joint fit | dps/dT = **−35.1 ± 3.8 ppm/K**; residual time term +26.5 ± 8.5 ppm/h |
| within-mosaic only (1.4 K range, bumps included) | −28.8 ppm/K, r = −0.52 |
| between-sequence (zenith → mosaic) | −43.9 ppm/K |

Temperature beats time outright, the within- and between-sequence slopes agree, and the
sign is the physical one (cooling contracts the train, shortens the EFL, raises ″/px).
**The night-2 plate-scale "drift" of §14 is thermal expansion at ~−35 ppm/K at a fixed
focuser.** This also retires the drift as a mystery: 1.4 K/h of cooling × 35 ppm/K
reproduces its magnitude.

**Cubic d(3000), 33 points:** the between-sequence step is **+3.7 %/K equivalent**
(3.026″ at 23.5 °C → 2.810″ at 21.5 °C) — squarely inside the prediction from two
independently measured couplings, 5–9 focuser steps/K (§12.4) × 0.64 %/step (§9.2) =
**3.2–5.8 %/K** for a train whose focuser is not moved as it cools. Within the band the
cubic does *not* track the air's short-term bumps (−1.7 %/K, r = −0.30, wrong sign, weak)
— which is what a **tube that lags the air by tens of minutes** produces: the optic
integrates the slow 2 K decline and ignores the 20-minute wiggles. The plate scale's
partial bump-tracking against the cubic's absence of it suggests the two live in
different parts of the train (fast-responding spacing vs slower lens-cell temperature).
The slew hypothesis of §15 is withdrawn as unnecessary; a mechanical component cannot be
fully excluded but nothing requires it.

**What this changes.** The cubic instability is not a random mechanical event — it is a
**deterministic temperature dependence at fixed focuser position, ~+3.7 %/K** (warmer =
larger cubic). Consequences:

1. §18.6's 4.84 % night-to-night gap now has a candidate cause of the right size: the two
   zenith nights differed by ~1.3 K (24.2 vs 22.9 °C boxed/spreader readings, §4.4) —
   ~+4.8 % predicted at +3.7 %/K. What §5.1-style nightly refocusing does or does not
   compensate becomes the sharp question.
2. **The eclipse transfer question becomes a sign question.** Totality sat at ~30.5 °C
   against the zenith calibrations' ~23.5–24.2 °C: ΔT ≈ +6–7 K → the eclipse-day cubic
   was plausibly **~+25 % above the frozen night value** — *before* accounting for the
   +129-step daytime refocus, which compensates thermal focus shift but compensates the
   cubic only if the focuser moves the element whose spacing drives F19's mechanism.
   Which element the EAF moves now decides the sign and size of the largest systematic in
   the chain; `STAGE3_THEORY.md` §6's over-correction mechanism assumed the opposite sign
   from focus-position reasoning. The 1/R-gradient diagnostic at step 3 arbitrates
   empirically.
3. Preliminary-status caveat, stated plainly: one night, 33 cubic points, sequences and
   temperature partially confounded, tube-lag inferred not measured. The 08-11 night
   (17049 steps, warmer) offers one independent check; a purpose-built
   temperature-vs-cubic monitor belongs on the 2027 list next to §9.3's focus sweep.


### 16.1 Figure correction, and the two follow-up tests (2026-08-27)

**Figure erratum**: `night2_temperature.png`'s cubic panel placed the band points at
~29 °C — a time-of-day wrap bug in that script's temperature lookup (the §16 *numbers*
came from the corrected decomposition and stand). Superseded by
`platescale_vs_temperature.png` (no flipped fields, night 1 added, tight scale) and
`cubic_vs_temperature.png` (correct axis: band at 20.8–22.2 °C).

**Can temperature explain night 1's plate scale? No — and the failure is informative.**
Six new corrections-ON reductions of the 08-11 zenith fields (FOCUSPOS 17049) give a
night-1 mean of 2.2073186 ″/px, **+197.5 ppm** above night 2's 2.2068828. At
−39.7 ppm/K that would need night 1 ~5 K *cooler*; the box logger read it **+0.75 K
warmer** (unexplained +227 ppm) and even §19.3's box-runs-warm correction (air ~22.15 °C)
leaves **+144 ppm unexplained**. Meanwhile the *cubic* night-to-night gap (+4.96 %)
matches the +1.3 K box ΔT at +3.7 %/K almost exactly. So between the two nights the train
underwent a **scale-only change (~+150–230 ppm of plate scale, cubic-neutral)** that
temperature does not account for — the 8 focuser steps would need −18 to −28 ppm/step
against the −3.0 to −3.6 measured below, so §12.4's "something optical did change"
survives, now sharpened to: something that moved the EFL without moving the cubic.

**Can a reasonable daytime temperature explain CAL_piLeo's focal length? Partly — with
the refocus carrying the rest at a plausible rate.** CAL_piLeo (corrections ON,
2.2054197 ″/px, FOCUSPOS 17170) sits **−663 ppm** below the night-2 zenith mean, across
ΔT ≈ +5–7 K and +129 focuser steps:

| daytime T | temperature share | residual over 129 steps |
|---|---|---|
| 30.5 °C (assumed) | −278 ppm (42 %) | **−2.98 ppm/step** |
| ~28.5 °C (FOCTEMP-corrected, §19.3) | −199 ppm (30 %) | **−3.60 ppm/step** |

The implied focus–scale coupling of −3.0 to −3.6 ppm/step is a new number for this train
(sign: racking toward daytime focus lowers ″/px), and it cross-checks: the mosaic flip's
−4 steps contribute ~+13 ppm of its −41 ± 30 ppm term, leaving the flip mechanics at
−54 ± 30 — plausible. With two unknowns fitted to one equation this is **consistency,
not proof** — but temperature-plus-refocus now accounts for the entire day–night
plate-scale difference with no third mechanism required. In focal-length language:
−39.7 ppm/K of plate scale = **EFL +14.4 µm/K** on the 363.5 mm train, against
~+8.4 µm/K for an aluminium tube alone — a ratio of 1.7 that the reducer-spacing
leverage (F19) comfortably supplies.


### 16.2 Clarifications and the back-focus hypothesis for night 1 (2026-08-27)

**The regression set.** The quoted −39.7 ppm/K (r = −0.82) never contained the flipped
fields — they were plotted grey but excluded from the fit, which is why the redrawn
figure's slope is unchanged (−39.2 ppm/K, r = −0.83, same 55 points; the 0.5 ppm/K is
rounding in the zenith temperatures). Including the 7 flipped fields *would* change it —
to −32.2 ppm/K with r collapsing to −0.24, courtesy of the −607 ppm M63/64 outlier — which
is exactly why they were excluded.

**Night 1: the back-focus re-seat hypothesis.** A change in the objective–reducer spacing
(the F19 focus mechanism) cannot explain night 1's +197 ppm: that route carries
0.64 %/step of cubic, so +197 ppm ≈ 60 steps-equivalent would drag the cubic by ~38 %,
against the measured +4.96 % (fully accounted by the 1.3 K temperature difference). The
element that moved must be **scale-active but cubic-neutral — which is the
reducer-to-sensor distance (back-focus)**. For a 0.7× reducer, magnification
m = 1 − d/f_red; with m ≈ 0.727 and d ≈ 60 mm, f_red ≈ 220 mm and dm/m ≈ 0.62 %/mm =
**6 200 ppm per mm of back-focus**. The observed +197 ppm therefore corresponds to
**~32 µm of back-focus re-seat** — and the telescope was dismounted, transported by car,
and remounted between the nights. The nightly Bahtinov refocus then restores sharpness
(moving the *focuser*, compensating focus) while the magnification change from the
altered reducer–sensor spacing remains. This reconciles the two facts: transport does not
change the objective's focal length, but ~30 µm of camera-assembly re-seating changes the
*system* EFL by exactly the observed amount, at near-zero cubic cost. It is also F19's
own mechanism — back-focus error — whose other signature is coma; a night-1-vs-night-2
comparison of the radial FWHM growth in the zenith stacks is the available confirming
test.

**The CAL_piLeo table of §16.1, restated.** Both rows are the **plate scale**
decomposition (the cubic appears nowhere in it); the two rows differ only in **which
daytime air temperature is assumed**, since the totality reading came from the shaded box
(30.5 °C as logged; ~28.5 °C if the box ran ~2 K warm in daylight, §19.3). Under either
assumption the same two-term model — temperature at −39.7 ppm/K plus the 129-step refocus
— absorbs the full −663 ppm, with the focus coupling landing at −3.0 or −3.6 ppm/step
respectively.


### 16.3 The coma test, the night-2-only workflow, FOCTEMP validated, and the day–night
cubic number (2026-08-28)

**The coma test detected the transport change — as a tilt.** Radial FWHM growth
(1.2–1.45° over 0–0.3°) across all twelve zenith stacks: **night 1: 1.216 ± 0.008;
night 2: 1.335 ± 0.042** — a ~6σ difference, with night 2 at Portland's conspicuous-coma
value. Both nights prefer the coma law (A + C·r²) over defocus (A + B·r⁴), night 2
decisively; and the m=1 tilt dipole (the 65PHQ analysis's own tilt fingerprint)
**doubled**: 0.510″ (8.7 % of mean FWHM) at PA −67° → 0.996″ (15.2 %) at PA −101°. The
compound picture closes every fingerprint at once: the transport re-seated the
camera/drawtube assembly **mostly as a tilt** (coma growth and dipole, ~zero scale cost)
**with a small axial component** (~32 µm → the +197 ppm scale step, ~zero coma cost) —
and neither moves the isotropic cubic, which stayed on its thermal track. A pure 32 µm
axial shift would have predicted an invisible coma change; the detection is the tilt.

**Workflow change (Douglas' direction): the chain freezes the cubic from night 2 only.**
Night 2 shares the eclipse day's mechanical state (no dismount between them); night 1 is
the pre-transport optic, now measurably different in tilt, coma, scale — and its cubic
(3.176″) differs from night 2's (3.026″) by the thermal +4.96 %. The canonical CAL_piLeo
reduction re-run with the six 08-12 references only:

| canonical step 2 (16 frames, night-2 references) | |
|---|---|
| stars used | 74 |
| rms | 0.5318 ″ |
| **plate scale** | **2.2054043 ″/px** |
| uncertainty | HC0 21.6 ppm → HC3-class ~25 ppm |

(−7.0 ppm from the 12-reference value — inside noise; the change buys mechanical
consistency, not precision. Step 3 imports this result and the 08-12 references.)

**The canonical temperature fit, once and for all**: **−39.7 ± 3.8 ppm/K, r = −0.82,
n = 55** (per-field logger temperatures; the earlier −39.2/−0.83 used bulk zenith
temperatures, and "−40/−35" were prose roundings of the same fit and its joint-fit
variant). The chart now carries slope, error and r.

**FOCTEMP validated as a differential thermometer.** Across 46 samples spanning the whole
of night 2 (22:48–00:13 UTC), FOCTEMP − spreader-logger = **+7.74 ± 0.67 K**, stable.
Applied to totality's FOCTEMP of 36.9 °C: **daytime air ≈ 29.2 ± 0.7 °C**, against the
30.5 °C shaded-box reading — Douglas' preference for the focuser thermometer is
supported, and 29.2 °C becomes the best daytime air estimate. (To be clear on the earlier
confusion: FOCTEMP itself read 36.9 °C, motor-warmed; 28.5–29.2 °C are *air estimates*
obtained by subtracting its measured offset; 30.5 °C was the box logger.)

**The day–night cubic change from temperature**: ΔT(air) = 29.2 − 23.5 = **+5.7 K** at
+3.7 %/K gives **+21 %** (range +18 to +33 % across the coupling's 3.2–5.8 %/K span).
That is the amount by which the eclipse-day cubic exceeded the frozen night-2 value *if
the 129-step daytime refocus does not compensate the spacing* — the §16 sign question,
now with its number attached. The corresponding CAL_piLeo scale decomposition at 29.2 °C:
temperature −226 ppm (34 %), refocus −437 ppm at −3.4 ppm/step.


### 16.4 The cubic–temperature claim, downgraded on its own evidence (2026-08-28)

Douglas challenged §16's cubic–temperature story on two grounds: night 1 is inadmissible
as an anchor (the transport re-seat means its cubic may carry a mechanical offset of its
own), and the night-2-only trend is unconvincing. The decisive check is the tube's own
sensor: if the cubic responds to the *optic's* temperature, it should track FOCTEMP.
It does not:

| | FOCTEMP (°C) | d(3000) (″) |
|---|---|---|
| zenith sequence (6 fields) | 30.6–30.8 (mean 30.66) | 3.0257 |
| band (27 fields) | 29.5–30.2 (mean 29.88) | 2.8096 |

The tube cooled only **−0.78 K** between the sequences (the air dropped ~1.9 K — the lag
is real), so a tube-thermal reading of the −7.1 % cubic step requires **+9.2 %/K** —
nearly triple the +3.2–5.8 %/K the (5–9 steps/K × 0.64 %/step) chain supplies, and that
0.64 %/step is itself §9.2's weakly-founded estimate. Within the band, cubic vs FOCTEMP
gives r = −0.04 — no tracking at all; the pooled r = +0.60 is purely the two-sequence
contrast, a single data point wearing 33 costumes.

**Status after the downgrade.** What survives, because it survived every methodology
control: **the cubic stepped −7.3 % between two same-night, same-focus sequences.** What
does not survive: a confident temperature attribution, and with it §16.3's "+21 %
day–night cubic" prediction, which rested on the air-temperature coupling. Candidate
causes for the step, none established: tube-thermal with a coupling ~2–3× the estimated
chain; a mechanical settling at the inter-sequence slew (resurrected, though disfavoured
by Douglas and unproven); an unidentified systematic of the mosaic's observing mode
(the 6 s-vs-4 s exposure difference remains bounded at ≤0.65 % by the clipping test but
not excluded as a family). The budget statement for step 3 accordingly reverts to the
robust form: **the cubic is unstable at the ~7 % level within a night on this train, with
unresolved cause and therefore unknown sign at eclipse time** — the 1/R-gradient and
M1-vs-M2 diagnostics in the eclipse data are the only arbiter, now without a prior on the
direction.

**The NP101 contrast (Douglas' first point), endorsed with the recorded numbers.** The
FRA500 + 0.7× wins on cubic *size* (6× less per unit angle than the NP101 native,
`INSTRUMENT_COMPARISON.md` §12) — but every stability pathology found this week runs
through the external reducer: the F19 spacing lever, the −3.4 ppm/step focus–scale
coupling, the tilt-coma of §16.3, the −7.3 % step. The two no-reducer controls on record
point the other way: the Leakey 65PHQ moved its cubic by only **+0.0138 ± 0.0075 %/step**
over a 70-step sweep (46× less than the Leon estimate) with plate scale steady to a few
ppm over 4–5 K; and Bruns' own NP101 night-to-night cubic change was **+1.66 % ± 7.99 %**
— weakly constrained, but with nothing demanding instability, and his transfer error was
protected twice over: a probably-stable train *and* the absolute-error rule of §7 (his
cubic was small in arcseconds, so even fractional wobble cost little). **Bruns' stability
assumption was likely sound for his train.** The 2027 design lesson sharpens accordingly:
the criterion is not cubic size but cubic size × stability, and a reducer-free or
integral-corrector optic that carries a larger-but-frozen cubic may beat a
small-but-mobile one — a qualification that now attaches to §6's "usable sky" ranking.


### 16.5 The two thermometers reconciled: a lagged tube, and the levers apportioned
(2026-08-28, from Douglas' tube-length datum and the FOCTEMP refit)

**Plate scale vs FOCTEMP: −109.2 ± 4.7 ppm/K, r = −0.95, n = 55** — a tighter correlation
than the air fit (−39.7 ± 3.8 ppm/K, r = −0.82) over a 1.3 K FOCTEMP span against the
air's 2.3 K. The same ~90 ppm of scale swing, correlated better against the slower sensor.

**The reconciliation.** The 535 mm aluminium tube follows the air with a lag of tens of
minutes; FOCTEMP's motor housing has comparable thermal mass, so FOCTEMP matches the
tube's *phase* (hence the superior r) while under-representing its *amplitude*; the air
matches the amplitude but not the phase (hence r = −0.82 with the bumps partially
smeared). The amplitude anchor is Douglas' tube length: aluminium at 23 ppm/K × **535 mm
(front lens to focal plane — the tube is physically longer than the focal length)** =
12.3 µm/K, against the measured 39.7 ppm/K × 363.5 mm = 14.4 µm/K — **a ratio of 1.17**
(§16.1's "1.7×" wrongly used the EFL as the length). The true tube coupling is therefore
bracketed −40 to −109 ppm/K with the mechanical arithmetic favouring the air end
(−40 to −60 ppm/K).

**The levers, apportioned across the train** (the useful engineering summary):

| element | length | thermal expansion | owns |
|---|---|---|---|
| tube, lens → reducer | ~480 mm | ~11 µm/K | **plate scale** (the −40…−109 ppm/K) and the **cubic lever** (objective–reducer spacing) |
| reducer → sensor | **55 mm** | **1.3 µm/K** | ~8 ppm/K of scale (via 6 200 ppm/mm), **~zero cubic** — thermally the *stable* section, exactly as Douglas argued; its risk is mechanical re-seat (§16.2/§16.3), not temperature |

**The cubic question, reopened to leading-hypothesis status — with a softer coupling.**
The thermometer-independent constraint is the between-sequence *ratio*: scale stepped
+88 ppm while the cubic stepped −7.1 %, i.e. **0.081 %-of-cubic per ppm-of-scale**. If
both ride the objective–reducer lever, the focuser calibration predicts 0.64/3.4 =
0.188 %/ppm — a factor 2.3 high. Closure requires **cubic-per-step ≈ 0.27 %/step** (§9.2's
0.64 was explicitly soft — eight recorded steps inside a 15-step backlash), and/or part of
the scale's thermal path bypassing the reducer entirely (the objective's own dn/dT moves
scale with zero cubic, diluting the ratio). Under the lagged-tube reading, §16.4's
"implausible +9.2 %/K" dissolves: per kelvin *of FOCTEMP* the scale moves −109 ppm, so
9.2/109 = the same 0.084 %/ppm ratio — no separate implausibility. Status: **thermal via
the shared lever, with cubic-per-step ~0.3 %/step, is again the leading explanation of
the −7.3 % step**, unproven (within-sequence cubic tracking remains undetected, though
the expected signal there is marginal against the 2.3–2.8 % per-field noise); a
mechanical component is not excluded. The day–night cubic transfer stays a **range, not a
number** — the daytime tube was sun-heated, so no night-calibrated thermometer constrains
it — and the eclipse data's 1/R-gradient diagnostic remains the arbiter, unchanged.


### 16.6 The reducer's lever: −109 ppm/K is the physically correct coupling
(2026-08-28, prompted by Douglas' challenge and the 540 mm tube length)

§16.5's "amplitude anchor" — measured 14.4 µm/K against aluminium's 12.3, ratio 1.17 —
was a **category error**: it compared EFL-equivalent microns to mechanical microns, which
are different quantities once a reducer leverages the geometry. The correct arithmetic:

The chief-ray plate scale of the reduced system is EFL_eff = f_obj · d/s, where d = 55 mm
(reducer→sensor) and s = 75.7 mm (reducer→native focal plane; from m = 0.727 and d,
giving f_red = 201.5 mm). Tube expansion acts on the **s arm, not on the EFL**: the
objective→reducer section (540 − 55 = 485 mm of aluminium) grows 11.15 µm/K, and
11.15 µm/K over a 75.7 mm arm is **147 ppm/K** from that term alone. In full, with the
objective's own thermal focal shift β (ppm/K):

    d(scale)/dT = −(170 − 5.6 β) ppm/K        [540 mm tube, aluminium]

| objective df/f (ppm/K) | plate scale (ppm/K) |
|---|---|
| 0 (athermal) | −170 |
| +11 | **−109** |
| +20 | −58 |

**β = +11 ppm/K — a mildly expanding apo objective, entirely plausible — reproduces the
measured −109.2 ± 4.7 ppm/K exactly.** The naive no-lever formula (12.4 µm/K ÷ 363.5 mm
= 34 ppm/K) is wrong by the lever ratio ~s/EFL. Two independent cross-checks close at
the factor-≤2 level through the same lever: thermal focus travel 11.2 µm/K ≈ **22
steps/K at ~0.5 µm/step, against §12.4's own on-train measurement of 17–20 steps/K**;
and the focus–scale coupling predicts 6.6 ppm/step against the measured 3.0–3.6.

**Consequences.** (1) FOCTEMP is vindicated as the tube's thermometer in *amplitude and
phase*; the air fit's −39.7 ppm/K is the attenuated apparent coupling of a lagging tube
against an over-swinging air signal (attenuation ≈ 1.3/2.3 ✓). The §16.5 bracket
collapses: **the train's thermal plate-scale coupling is ~−109 ppm/K of tube
temperature.** (2) The cubic closes to the same picture at the current precision:
22 steps/K × ~0.27 %/step ≈ 6 %/K against the observed 9.2 %/K (FOCTEMP-referenced),
factor 1.5. (3) One honest new tension: at −109 ppm/K, the raw day–night FOCTEMP
difference (+6.24 K) predicts −681 ppm of scale — the *entire* observed −663 ppm —
leaving nothing for the 129-step refocus's −430 ppm. The books balance only because
daytime FOCTEMP is sun-loaded (the +7.74 K offset is a night calibration), so the
optic's true daytime ΔT is smaller than FOCTEMP implies, and/or the refocus partly
*compensates* thermal expansion — which is what refocusing is for. The daytime
thermometry cannot be untangled from night data; the step-3 diagnostics remain the
arbiter of the transferred calibration, unchanged.


### 16.7 The day–night plate-scale anomaly: refocus should restore the geometry, and
never does (Douglas, 2026-08-28)

**The theorem, stated properly.** The daytime refocus exists precisely to compensate tube
expansion. On this train the focuser moves the reducer+sensor assembly together, so a
correct refocus restores s (reducer to native focal plane) to the focus condition with d
mechanically fixed — i.e. it restores the *reducer geometry*, and the refocused day–night
scale difference should be only the objective's own thermal term:

    Δscale(day−night, refocused) ≈ −β · ΔT_objective ≈ −11 ppm/K × (+5–7 K) ≈ −60 to −80 ppm

**The observation: −663 ppm at Leon — an order more.** And Bruns 2017 shows **−524 ppm**
day–night on the NP101is. The ideal-restoration picture fails in practice, on both rigs,
in the same direction and order — the record already called the day–night gap "a property
of the experiment class, not of one rig"; this section explains why that is the right
description and what the candidate physics is.

**The NP101 is not actually lever-free.** It is a Petzval: a front objective plus a rear
group *fixed in the tube*. Tube expansion changes the group separation — an internal
version of exactly our reducer lever — and the focuser, which moves only the
camera/drawtube, **cannot restore an element spacing that is set by the tube itself**. So
"no external reducer" does not mean "no lever": both trains carry a tube-set element
separation that refocusing cannot touch, and both show ~−500–700 ppm. (The clean control
remains the Leakey 65PHQ — also a Petzval, but §10's sweep showed its *focus* barely
moves anything; its day–night behaviour was never measured.)

**The Leon contributor stack for the −663 ppm**, honestly bracketed: the restorable-theory
term −60 to −80 ppm (β·ΔT); the unrestored lever share if the daytime refocus criterion
(the Venus-terminator procedure, §18.10) lands tens of steps from true focus (~−100 ppm
per 30 steps at the measured coupling); sun-load on the objective cell (β acts on the
*lens* temperature, which under daytime sun-load can exceed air ΔT several-fold —
unmeasured); and the daytime refraction-model residual at alt 9.87°, which M2 measured at
the −80 to +340 ppm scale-level on this very sightline class — a contribution of either
sign that no night data constrains. The gap is over-determined by uncertain terms, which
is the deep reason it has resisted explanation across campaigns.

**Why this does not threaten L — the design already knew.** The chain discards the night
plate scale by construction (§18.2) and imports the scale from a calibration field taken
*in the daytime configuration, simultaneously* (CAL_piLeo; Bruns' L/R) — F&L's italicised
"same position" rule is precisely the defence against this anomaly, written in 1944. The
day–night gap is a diagnostic of the train, not an error term in the deflection; its
lesson lands on 2027 design (measure everything in the flight configuration) rather than
on the 2026 reduction.


### 16.8 The FRA500 is itself a Petzval (Douglas, with the optical layouts, 2026-08-28)

Douglas supplied the maker's layouts: the FRA500 is a **front triplet apo plus a
two-element flattener fixed mid-tube** — the same Petzval architecture as the 65PHQ
(3+2 "quintuplet") and the NP101is. This corrects §16.6/§16.7's framing in one place and
sharpens it in three:

1. **§16.6's arithmetic survives, but β is reinterpreted.** The lever formula treated
   "f_obj(T)" as a simple objective with thermal coefficient β. In fact the native 500 mm
   system's focal length depends on the **tube-set separation between the triplet and the
   internal flattener**, which no focuser can restore. The fitted β = +11 ppm/K is
   therefore the native Petzval's *effective* thermal coefficient — glass dn/dT, cell,
   and the internal-separation lever combined — not a glass property. The external
   reducer's s-lever term (147 ppm/K on the 75.7 mm arm) is unchanged; the decomposition
   ds/dT = df_native/dT − dL/dT already routes the internal effects through f_native.
2. **Every optic in this project is a Petzval; the discriminator is where the leverage
   sits.** NP101is: internal rear group, no external reducer — its −524 ppm day–night
   implies an effective native coefficient of ~+35–50 ppm/K over a plausible ΔT, i.e. a
   stronger internal lever (its rear group sits closer to the focal plane; lever strength
   scales as 1/s of the group). FRA500 native: internal lever worth ≲ +11 ppm/K
   effective. FRA500 + 0.7×: the external reducer adds the dominant 13.2 ppm/µm lever on
   a 76 mm arm. 65PHQ: internal group only, and the Leakey sweep measured its focus/scale
   response as nearly nil — the most thermally benign architecture of the three as flown.
3. **The Leakey control's meaning sharpens**: it did not show "Petzvals are stable" — it
   showed that *moving the camera* on a Petzval whose internal separation is tube-set
   does little, which is exactly the geometry statement. Its day–night (tube-thermal)
   behaviour was never measured and remains the missing control for the class.

The 2027 selection rule of §16.4 gains its mechanism: cubic size × stability, where
stability is set by the tube-set separations' lever arms — shortest levers (or cemented/
athermalised spacings) win, and an external reducer on a short back-focus is the worst
configuration of the ones flown.


## 17. 2027 candidate: the SQA85, and its acceptance test (Douglas, 2026-08-28)

Douglas proposes the Askar SQA85 (85 mm f/4.8 quintuplet, 408 mm, total axial length
440 mm, layout supplied) — short focal length, large aperture, **designed to work without
an external reducer**. Against this campaign's measured criteria:

| | FRA500 + 0.7× (as flown) | SQA85 (candidate) |
|---|---|---|
| plate scale (″/px, IMX571) | 2.134 | 1.901 |
| field (°) | 3.70 × 2.47 (9.2 sq°) | 3.30 × 2.21 (7.3 sq°, 0.79×) |
| corner radius | 8.5 R⊙ | 7.5 R⊙ |
| deflection at limb (px) | 0.82 | 0.92 |
| tube / focal length | 1.07 | 1.08 |
| external reducer | **yes — the dominant measured instability** | **none** |

What "no external reducer" removes, in this campaign's measured terms: the 13.2 ppm/µm
back-focus lever on a 76 mm arm (the −109 ppm/K thermal path's multiplier), the
−3.3 ppm/step focus–scale coupling, and the threaded-assembly mechanical modes — the
transport tilt (coma growth 1.216 → 1.335) and the ~32 µm re-seat (+197 ppm). What it
does **not** remove: the internal levers. The layout shows a rear element close to the
focal plane — a short-arm, tube-set spacing of exactly the class that gives the NP101is
its ~+35–50 ppm/K effective coefficient. The NP101 lesson stands: reducer-free is not
lever-free, and the SQA85's native thermal coefficient must be **measured, not assumed**.
Costs to note: 21 % less calibration sky per field, and a slightly smaller corner radius
(7.5 vs 8.5 R⊙ — mildly *lowering* F&L's h, which is favourable for Method 1's scale
sensitivity).

**The acceptance test, from this branch's toolkit — one night plus a transport cycle:**

1. **Two 6-field zenith sets a few hours apart**, logger on the spreader: free-cubic
   d(3000) stability, and the thermal plate-scale coupling via the FOCTEMP-analogue +
   logger (the §16.6 lever analysis then yields the native effective β directly).
2. **A deliberate ~100-step focus sweep** (Leakey protocol, §10): the focus–scale and
   focus–cubic couplings, outside backlash.
3. **A dismount–transport–remount cycle, then repeat one zenith set**: the re-seat modes
   — scale step, tilt dipole, coma-law growth (§16.2/§16.3 machinery applies unchanged).

Pass criteria worth setting in advance: native |β| ≲ 15 ppm/K, cubic per-night stability
≲ 1 %, transport-cycle scale step ≲ 50 ppm and dipole change ≲ 3 % of FWHM — numbers the
FRA500 + 0.7× would have failed on two of four. Douglas is awaiting good zenith data for
the SQA85; when it exists, items 1–2 run with the existing scripts as-is.


### 16.9 CORRECTION to §16.6: the lever arithmetic done properly, and what it does not
explain (from Douglas' NP101is schematic, 2026-08-28)

Douglas supplied Bruns' own schematic: the NP101is is a **4-element, 2-group
Nagler-Petzval** — front doublet at ~f/11 (≈1100 mm), rear doublet just ahead of the
focuser reducing the system to 540 mm at f/5.4, acting natively as the field flattener.
Confirmed: **every optic in this project is a long-focus objective plus a reducing group**;
the FRA500 + 0.7× differs only in where that group sits.

**§16.6's arithmetic was wrong, and its "exact closure" was a one-parameter fit.** It used
lever = 1/s (13.2 ppm/µm) and then solved for the objective coefficient β that reproduced
the measurement — fitting, not predicting. The correct chief-ray treatment of a two-group
train (objective f_obj, rear group at L, sensor d₀ beyond it, so EFL = L + d₀(1 − L/f_red))
gives two levers, the first pleasingly general:

    dEFL/dL = m   (the reduction factor itself)        dEFL/dd₀ = 1 − L/f_red

| train | m | L (mm) | d₀ (mm) | aluminium only, fixed focus |
|---|---|---|---|---|
| FRA500 + 0.7× | 0.727 | 424 | 55 | **−15.7 ppm/K** |
| NP101is | 0.491 | 550 | 270 | **−11.1 ppm/K** |

1. **§16.8's claim that the NP101is has the *stronger* internal lever is withdrawn.** It has
   the weaker one — but only by 1.4×, because its stronger reduction (m = 0.49 against
   0.73) largely offsets its longer expanding section. Architecture is a weak
   discriminator; §16.4's "cubic size × stability" rule survives, but the stability half
   **cannot be read off a layout** — it has to be measured.
2. **Aluminium expansion explains at most 40 % of the measured coupling, possibly 14 %.**
   Predicted −15.7 ppm/K against −39.7 (air-referenced) to −109.2 (FOCTEMP-referenced).
   The remainder must lie in terms this arithmetic omits: the objective's own thermal
   focal shift (glass dn/dT plus cell — the real β, unmeasured), the internal doublet
   air-spacings, and FOCTEMP under-swinging the true optical temperature so its per-K
   slope over-estimates the physical coefficient. **The physical coupling is bracketed
   −16 to −40 ppm/K; −109 ppm/K is an empirical FOCTEMP-referenced slope and must be
   quoted as such, not as a material property.**

**Nothing downstream changes**: step 2 and step 3 use *measured* plate scales, never a
thermal model. §16.7's day–night anomaly is if anything strengthened — a smaller thermal
share leaves more of the −663/−524 ppm to the focus criterion, sun-load and the daytime
refraction residual. And §17's acceptance test already specifies *measuring* a candidate's
native coefficient, which this section shows is the only reliable route.
