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
