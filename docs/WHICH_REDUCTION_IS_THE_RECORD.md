# Which reduction is the record, and where its output lives

**Written 2026-09-01**, because several variants were run over 2026-08-29..09-01 and only
two of them are quoted. Anything not listed here is an experiment, not a result.

## The two numbers, and the chain that produced each

### Cell 1 — Bruns 2017: **L = 1.777 ± 0.065 (stat) ± 0.084 (scale) ± 0.15 (atmosphere) ″**

Total σ ≈ 0.18. Bruns 2018 published 1.752 ± 0.060 — agreement to 0.025″ (0.4 σ).
GR at 0.14 σ; Newton excluded at 4.9 σ.

**Reduced by Bruns' own procedure** (his § quoted by Douglas, 2026-09-01): ONE 0.62 s
master from all 34 EA+EB frames; the two close-in stars carried from the 0.09 s master by
the seven-brightest-common-stars offset link (measured link se 0.08″); Method 1 with the
imported bracket scale, no nuisance term (his method had none). The v-deg2 variant gives
1.680 ± 0.081 and Method 2 gives 1.842 ± 0.116 (scale −9.5 ppm from imported), both
reported alongside.

| step | what | where |
|---|---|---|
| calibration | 15 night fields → cubic frozen; L and R8 refit, **Gaussian bg + footprint moments**; bracket mean 2.0867533 ″/px | `matrix_bruns2017_like2024/{L,R8}/stage2/` |
| preprocessing | tier-mean blur-10px coronal subtraction, forbidden disk painted at the pedestal | `matrix_bruns2017/{EA,E2,EB}/preprocessed/` (shared) |
| the 0.62 s master | all 34 EA+EB frames, one stack; constant-only against the bracket (39 matched, rms 0.4993″) | `matrix_bruns2017_brunsmethod/master062/` |
| the 0.09 s master | the E2 stack in the same convention | `matrix_bruns2017_like2024/E2/` |
| the link + the fit | Bruns' 7-star offset + Method 1 (**tools, not the program** — F27) | `tools/matrix_bruns/b17_bruns_method.py` |
| atmosphere term | 22 constant-only night nulls, consecutive same-night pairs | `tools/matrix_bruns/b17_atmosphere2.py`, `matrix_bruns2017_atmosphere3/` |

**Graphical output — start at `RECORDruns2017\`** (a copy of the summary set lives
there precisely so it can be found):
* summary charts: `RECORD/bruns2017/record_deflection.png`, `record_field.png`,
  `record_covariance.png` (originals beside the reduction in
  `matrix_bruns2017_brunsmethod/`), plus the star table `bruns_method_star_table.csv`
* the program's own plots: `matrix_bruns2017_brunsmethod/master062/CENTROID_OUTPUT*/` and
  `master062/stage2/DISTORTION_OUTPUT*/`; the calibration fields under
  `matrix_bruns2017_like2024/{L,R8}/`
* the M3-style atmosphere maps: `RECORD/bruns2017/atmosphere_night_maps.png`

### Cell 3 — Leon 2026: **L = 1.98 ± 0.60 (stat) ± 0.33 (atmosphere) ″**

Quoted in the windowed+annular convention. Re-reduced in cell 1's convention it gives
1.897 — a **−0.08 ″** shift, so the headline is convention-robust and the two cells are
comparable.

| step | what | where |
|---|---|---|
| calibration | six 08-12 zenith cubics → CAL_piLeo 16 frames, 2.2054043 ″/px | `cal_pileo_step2/canonical_16f_night2refs/` |
| preprocessing | coronal subtraction + forbidden disk | `step3_s0_v4/` (frames frozen) |
| stage 1 + 2 | constant-only against CAL_piLeo | `step3_prelim_L/{0p6s,1p2s}/` |
| stage 3 | union estimator + two-pass rematch (**tools**) | `tools/step3_s2_union.py`, `step3_rematch.py` |
| atmosphere term | M5 night nulls, S1 gate (failed honestly → ±0.33 quoted) | `tools/step3_s1_estimator.py` |

**Graphical output**
* summary charts: `step3_s2_plots/` — `field_radec.png`, `field_altaz.png`, `covariance.png`, `deflection_method1.png`, `deflection_method2.png`
* the program's own plots (65 files): `step3_prelim_L/*/stage2_constant/DISTORTION_OUTPUT*/` and `*/stage3/`
* convention cross-check: `step3_bruns_convention/`

## What is NOT the record

| tree | what it was | why not quoted |
|---|---|---|
| `matrix_bruns2017/` (windowed) | the first cell-1 reduction, L = 1.556 | superseded by the convention ruling; its two charts are renamed `SUPERSEDED_windowed_1.556_*` |
| `matrix_bruns2017_like2024/{EA,EB}` per-tier | the like-2024 convention with EA/EB stacked separately, L = 1.720 | superseded 2026-09-01: Bruns stacked all 0.62 s frames as ONE master and linked the inner pair in, so the record now follows his procedure (`_brunsmethod/`) |
| `matrix_bruns2017_moment/` | rollback attempt 1 | mis-designed — turned the sensitive flag off, giving a different detector |
| `matrix_bruns2017_gate/`, `step3_gate/` | mask-as-gate rerun | preserving the saturated core created a high-variance patch; detections collapsed |
| `matrix_bruns2017_modelfix/`, `step3_modelfix/` | masked-blur rerun | stopped part-way, superseded by the pipeline-path attempt |
| `matrix_bruns2017_pipeline/` | raw frames through the pipeline | blocked by F28 — the per-frame coronal model leaves too few stars to plate solve |
| `matrix_bruns2017_atmosphere/`, `_atmosphere2/` | atmosphere attempts 1 and 2 | both invalid; see the record. `_atmosphere3/` is the valid one |

## Known defects in the record, measured and bounded

1. **The coronal model carves a trench** just outside the saturated core (naive blur
   includes the core's plateau). Fixed in the pipeline; the reductions of record still
   carry it. Re-running is F28-blocked for the pipeline path and pending for the tool path.
2. **Rim artefacts reach the alignment** (F29): 0.8 per frame against 12.8 real stars on
   Bruns EA, because the tool chain runs with the pipeline mask off. Per-frame, not
   per-star, so it moves the star sample rather than biasing astrometry.

Neither changes the quoted numbers; both are why a clean re-run is the next piece of work.
