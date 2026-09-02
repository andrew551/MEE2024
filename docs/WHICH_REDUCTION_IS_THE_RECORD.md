# Which reduction is the record, and where its output lives

**Written 2026-09-01, cell 3 revised the same night**, because several variants were run over 2026-08-29..09-01 and only
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
| atmosphere term | 22 constant-only night nulls, consecutive same-night pairs (±0.150, one-sided). **2026-09-02: the same nulls redone with a bracketing reference, as his eclipse field was fitted, give ±0.087 and contain no scale-like part; proposed as the term matching his design, not yet applied** | `tools/matrix_bruns/b17_atmosphere2.py`, `b17_bracket_null.py`, `matrix_bruns2017_atmosphere3/` |

**Graphical output — start at `RECORD/bruns2017\`** (a copy of the summary set lives
there precisely so it can be found):
* summary charts: `RECORD/bruns2017/record_deflection.png`, `record_field.png`,
  `record_covariance.png` (originals beside the reduction in
  `matrix_bruns2017_brunsmethod/`), plus the star table `bruns_method_star_table.csv`
* the program's own plots: `matrix_bruns2017_brunsmethod/master062/CENTROID_OUTPUT*/` and
  `master062/stage2/DISTORTION_OUTPUT*/`; the calibration fields under
  `matrix_bruns2017_like2024/{L,R8}/`
* the M3-style atmosphere maps: `RECORD/bruns2017/atmosphere_night_maps.png`

### Cell 3 — Leon 2026: **L = 1.914 ± 0.637 (stat) ± 0.675 (scale) ± 0.33 (atmosphere) ″**

Total σ ≈ 0.985. GR at 0.17 σ; Newton at 1.05 σ. The 0.6+1.2 s union under the
**two-witness rule** (a star is admitted only if both tiers detected it): 36 stars,
h = 25.9 R☉², vertical-deg-2 nuisance, the below-Sun star in.

Two revisions this session, both recorded in `docs/STEP3_2026.md`:

* **2026-09-01, the scale term** — the quoted headline had carried only stat and
  atmosphere. The imported plate scale's HC3-class 25 ppm, measured on this field's
  geometry by injection (0.027 ″ of L per ppm with the nuisance on), is the largest term
  in the budget;
* **2026-09-02, the two-witness rule** (Douglas' ruling) — six of the 42 matches were
  detected in one tier only, so the cross-tier consistency vet could never act on them,
  and one of those six sat +3.5 σ off the curve. Admitting only two-witness stars moves L
  by −0.06 ″ and leaves nothing beyond 2.5 σ. The superseded 42-star value,
  **L = 1.976 ± 0.596**, is kept as `record_deflection_all_matches.png`.

Quoted in the windowed+annular convention. Re-reduced end to end in cell 1's convention
(Gaussian + moments) it gives 1.897 — a **−0.08 ″** shift. The 2×2 on Leon alone
(`tools/step3_background_ab.py`, `step3_bg_ab/`, 2026-09-02): the background axis is
worth +0.14 ″ (windowed+Gaussian 2.115, three union stars fewer), the estimator axis
−0.38 ″ (moments+annular 1.595) — on Leon the estimator is the larger lever, the
reverse of Bruns; details in `docs/STEP3_2026.md` ("Leon brought to the cell-1
standard"). The headline is convention-robust and the two cells are comparable; they do
not share a convention, and the choice is per-instrument on purpose (Leon's optics carry a
brightness-dependent centroid bias the windowed estimator exists to remove).

| step | what | where |
|---|---|---|
| calibration | six 08-12 zenith cubics → CAL_piLeo 16 frames, 2.2054043 ″/px ± 25 ppm | `cal_pileo_step2/canonical_16f_night2refs/` |
| preprocessing | coronal subtraction + forbidden disk | `step3_s0_v4/` (frames frozen) |
| stage 1 + 2 | constant-only against CAL_piLeo | `step3_prelim_L/{0p6s,1p2s}/` |
| stage 3 | union estimator + two-pass rematch (**tools**, F27) | `tools/step3_s2_union.py`, `step3_rematch.py` |
| the star table | the 42 matches with an `ntier` column; the record is the 36 with two witnesses | `step3_record/leon_union_star_table.csv` (+ `_sans_anchor`, `_full4`, `leon_union_meta.json`) |
| atmosphere term | M5 night nulls, S1 gate (max over three windows, ±0.33); re-derived by cell 1's construction ±0.22 rms / 0.31 max | `tools/step3_s1_estimator.py`, `tools/step3_atmosphere.py`, `step3_record/atmosphere_nulls.csv` |
| scale term | 25 ppm × the leverage measured by injection on the record's geometry | `tools/step3_charts_record.py`, `step3_record/record_summary.json` |
| structure | one 0.6+1.2 s master built and rejected (re-admits the G 9.10 corrupted centroid; 2.52 ± 0.61) | `tools/step3_master_vs_union.py`, `step3_record/master0612/` |

**Graphical output — start at `RECORD/leon2026\`** (a copy of `step3_record/`):
* the four charts of the spec, **chart revision 2**: `record_deflection.png` (variants
  `_sans_anchor`, `_no_nuisance`, `_full4`, `_two_witness`), `record_field.png`
  (displacement vectors in **alt/az**, nuisance removed) and `record_field_raw.png`
  (nuisance left in — the vertical atmosphere visible, V/H 2.5),
  `record_covariance.png`, `atmosphere_night_maps.png` (9 horizon windows + 12 zenith
  fields, Bruns style) with `atmosphere_floor_table.csv`, `zenith_floor.csv` and
  `zenith_nulls.csv` beside it, and `master_0p6s_annotated.png` /
  `master_1p2s_annotated.png` (yellow = both exposures, red = one only); every revision under
  `step3_record/chart_versions/`, superseded copies under `RECORD/leon2026/superseded_*`
* the 2026-08-29 chart set (`field_radec`, `field_altaz`, `covariance`,
  `deflection_method1/2`) is kept under `RECORD/leon2026/superseded_2026-09-01_2312/`
* the program's own plots (65 files): `step3_prelim_L/*/stage2_constant/DISTORTION_OUTPUT*/` and `*/stage3/`
* convention cross-checks: `step3_bruns_convention/` (both axes switched), `step3_bg_ab/` (one axis at a time)

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
