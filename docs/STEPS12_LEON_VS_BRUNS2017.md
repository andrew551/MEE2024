# Steps 1 and 2, side by side: Leon 2026 against Bruns 2017 at the same stage

**Date:** 2026-08-27. Every number measured on `v1.4.0-dev` with identical settings and
estimators: Leon zenith from `G:\Leon Aug 2026` (this batch, all 12 fields, reproducing the
handoff to all seven digits); Bruns from the re-reductions of his raw frames made during the
instrument-comparison week (29 night fields, 4 L/R variants), residuals re-analysed here.
Uncertainty estimators: **HC0** = White's robust standard error as the pipeline reports it;
**HC3** = the leverage-corrected version; **jackknife** = delete-one-star refits. All are
*within-field, star-resampling* errors; the field-to-field (f2f) scatter column is the
between-field truth they must be judged against.

## Step 1 — the zenith calibrations (four datasets, three estimators each)

| dataset | fields | stars/field | HC0 med (ppm) | HC3 med (ppm) | jackknife med (ppm) | night mean (″/px) | f2f sd (ppm) | se of mean (ppm) | f2f ÷ HC3 (ratio) |
|---|---|---|---|---|---|---|---|---|---|
| Leon 08-11 | 6 | 1448–1942 | 1.32 | 1.34 | 1.33 | 2.2077996 | 6.97 | 2.85 | 5.2 |
| Leon 08-12 | 6 | 2179–3520 | 1.21 | 1.21 | 1.21 | 2.2073819 | 4.56 | 1.86 | 3.8 |
| Bruns 08-19 | 14 | 535–1297 | 3.43 | 3.51 | 3.50 | 2.0876816 | 28.88 | 7.72 | 8.2 |
| Bruns 08-20 | 15 | 559–1313 | 3.06 | 3.10 | 3.10 | 2.0878601 | 19.73 | 5.09 | 6.4 |

Night-to-night plate-scale gaps: **Leon +189 ppm** (≈ 32 σ of the se-of-mean, §12.4's number
reproduced), **Bruns +85 ppm**. Bruns' larger f2f scatter is partly his measured within-night
drift (+1.5–2.8 ppm/min over fields spanning hours; ~12–14 ppm after detrending, per the
instrument-comparison record), against Leon's 23-minute field sequences.

Two lessons this table settles:

- **At zenith star counts the estimator choice is irrelevant**: HC3/HC0 ≤ 1.02 everywhere,
  jackknife identical. The program's HC0 is fine at step 1.
- **The real uncertainty at every stage is atmospheric coherence, not star statistics**:
  field-to-field scatter runs **3.8–8.2×** the within-field error on all four datasets, both
  rigs, both years. Nothing about Leon is anomalous — Bruns had the same coherent-atmosphere
  floor, larger.

(Reminder from §18.2: the zenith plate scale is discarded at step 2 by construction — this
table is a stability diagnostic and an estimator test-bench, not an input to L.)

## Step 2 — the eclipse-day calibration fields

| | Leon CAL_piLeo (canonical, 16 frames) | Bruns L | Bruns R (8 frames) | Bruns L+R combined |
|---|---|---|---|---|
| geometry | one field, az 270°, alt 9.9° — **one-sided** | −7.4° from Sun | +7.4° from Sun | **bracketing the eclipse field** |
| stars | 74 | 105 | 110 | 215 |
| rms (″) | 0.532 | 0.220 | 0.235 | — |
| plate scale (″/px) | 2.2054197 | 2.0867534 | 2.0868474 | 2.0868004 |
| HC0 (ppm) | 21.7 | 13.4 | 13.6 | ~9.5 |
| **HC3 (ppm)** | **25.1** | **14.5** | **14.7** | **~10.3** |
| jackknife (ppm) | 24.9 | 14.4 | 14.6 | — |
| L−R split | — | | | **45.0 ppm** (half-width 22.5) |
| the published-era figure | — | | | Bruns quoted δS = **3.34 ppm** (moment formula) |

## What the whole picture says

1. **The estimator hierarchy matters only below ~150 stars.** Zenith: no difference. Step-2
   fields: HC3 runs 8–15 % above HC0 (Leon 21.7 → 25.1; Bruns 13.4 → 14.5). The pipeline's
   HC0 was built and validated in the regime where the choice is invisible; the eclipse-day
   fields are where it opens.

2. **Bruns' honest per-field calibration error was ~14.5 ppm, not the 3.34 ppm his moment
   formula supplied** — a factor 4.3, the same reported-versus-honest factor this project
   has now measured on four independent datasets. The 4× optimism at the foundation of the
   best eclipse measurement ever made is the strongest argument yet for quoting HC3-class
   errors and field-to-field checks.

3. **The bracket is the decisive structural difference — worth more than star counts,
   altitude, or estimators.** Bruns' two fields disagree by 45.0 ppm; because the eclipse
   field sat midway, averaging them cancels the linear part of the atmospheric differential
   and *measures* its size (22.5 ppm half-width). Leon's night campaign independently
   measured the same phenomenon on the CAL-to-eclipse geometry at **25–65 ppm** — the two
   campaigns agree on the atmosphere; only Bruns' design could cancel and bound his own.
   CAL_piLeo, one-sided, inherits the differential uncancelled.

4. **What reaches L, using each field's measured coupling** (Leon: 0.17 % of L per ppm, the
   M5-measured coupling with the Sun off frame centre; Bruns: 0.37 %/ppm, his effective
   geometry): at the same stage, honestly assessed, Bruns stood at roughly **10–15 ppm
   effective δS → 4–6 % of L**, with his residual differential second-order after
   averaging — against his published 1.23 %. Leon stands at **25.1 ppm statistical → ~4 %
   of L**, essentially matching Bruns — *plus* the uncancelled one-sided differential
   (25–65 ppm night-measured → 4–11 % of L at scale-like coupling, more if the residual
   arrives as shape), which is the M5 forecast of 6–35 % and the reason step 3 carries its
   diagnostics.

5. **Leon's raw precision is better everywhere it can be** — zenith per-field 1.2–1.3 ppm
   against Bruns' 3.1–3.5 (bigger sensor, deeper stacks), calibration rms comparable at
   twice the airmass — and none of that closes the gap that matters, because the budget is
   set by the atmosphere's spatial structure and by what the calibration *geometry* can
   cancel. The 2027 lesson writes itself: two calibration fields bracketing the Sun, per
   F&L, per Bruns, per every number in this table.

Sources: `refraction/zenith12/` (this batch), `bruns2017_nights/`, `bruns2017_lr/`,
`cal_pileo_step2/variant_A16_pure_tiers/` (the canonical CAL_piLeo), analysis in
`refraction/analysis/three_estimators.py`. Bruns' fields re-reduced from his raw frames with
the same pipeline and settings as Leon's, so every comparison is like for like.
