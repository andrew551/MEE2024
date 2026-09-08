# Which reduction is the record, and where its output lives

**Written 2026-09-01, cell 3 revised the same night**, because several variants were run over 2026-08-29..09-01 and only
two of them are quoted. Anything not listed here is an experiment, not a result.

**Updated 2026-09-06** with cell 2, which closed on 2026-09-05. Two things a reader should
know before quoting from this page: the current budget for every cell is the one in
[`MATRIX_2026.md`](MATRIX_2026.md), which is maintained and this page is not; and cell 1's
atmosphere term is quoted in the headline below as the design-matched ±0.059 while its own
table row still says that term was proposed and *not yet applied*. That contradiction is
unresolved here — do not quote cell 1's total without checking the matrix first.

This file is the index of `D:\MEE2024 output\MEE_output\RECORD\`, and
`tools/sync_record.py` copies it there; the repository copy is the one to edit. Besides the
three cell folders below, RECORD holds **`refraction/`**: the Leon campaign's six figures
(two publication-grade, with their caption facts in its `README.md`), the tables
`docs/REFRACTION_2026.md` cites, and the withdrawn first temperature figure under
`superseded_2026-08-27/`. It is not a cell — it measures no L — but it is where the
atmosphere term of cells 3 and 4 comes from.

## The three numbers, and the chain that produced each

### Cell 1 — Bruns 2017: **L = 1.764 ± 0.060 (stat) ± 0.075 (scale) ± 0.059 (atmosphere) ″**

Total σ ≈ 0.113 (chart revision 12, 2026-09-02; the scale term is the bracket HC3 of the pair the reduction of record uses (9.23 ppm, Gaussian + moments), not the windowed pair’s 10.3). Bruns 2018 published 1.7512 ± 3.4 %
= ±0.060 ″. **GR at 0.11 σ; Newton excluded at 7.9 σ.** The atmosphere term is the **R-E-L bracketed** null of 2026-09-02 — the construction his eclipse fit actually used — not the one-sided ±0.150 the charts carried through revision 10. The 14-star link is the
standard; the 7-star link Bruns used gives 1.777 ± 0.064, L-neutral at 0.013 ″.

**Reduced by Bruns' own procedure** (his § quoted by Douglas, 2026-09-01): ONE 0.62 s
master from all 34 EA+EB frames; the two close-in stars carried from the 0.09 s master by
the seven-brightest-common-stars offset link (measured link se 0.08″); Method 1 with the
imported bracket scale, no nuisance term (his method had none). The v-deg2 variant gives
1.680 ± 0.081 and Method 2 gives 1.842 ± 0.116 (scale −9.5 ppm from imported), both
reported alongside.

| step | what | where |
|---|---|---|
| calibration | 15 night fields → cubic frozen; L and R8 refit, **Gaussian bg + footprint moments**; bracket mean 2.0867533 ″/px, HC3 12.64 and 13.46 ppm → **9.23 ppm** for the mean | `matrix_bruns2017_like2024/{L,R8}/stage2/` |
| preprocessing | tier-mean blur-10px coronal subtraction, forbidden disk painted at the pedestal | `matrix_bruns2017/{EA,E2,EB}/preprocessed/` (shared) |
| the 0.62 s master | all 34 EA+EB frames, one stack; constant-only against the bracket (39 matched, rms 0.4993″) | `matrix_bruns2017_brunsmethod/master062/` |
| the 0.09 s master | the E2 stack in the same convention | `matrix_bruns2017_like2024/E2/` |
| the link + the fit | Bruns' 7-star offset + Method 1 (**tools, not the program** — F27) | `tools/matrix_bruns/b17_bruns_method.py` |
| atmosphere term | 22 one-sided night nulls (±0.150). **2026-09-02: his night fields run R, E, L on a two-minute cadence with the eclipse pointing midway between the calibration pointings, so the null can be built exactly as his eclipse fit was — against the MEAN of R and L. That gives ±0.059 (8 triplets), and the one-sided means against R and L alone are −0.039 and +0.036, equal and opposite. Proposed as the design-matched term, giving total 0.118; not yet applied** | `tools/matrix_bruns/b17_atmosphere2.py`, `b17_bracket_null.py`, `b17_lr_bracket_null.py`, `matrix_bruns2017_atmosphere3/` |

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

### Cell 2 — Mexico 2024, Station 1: **L = 1.804 ± 0.084 (stat) ± 0.11 (atmosphere) ″**

Total σ ≈ 0.138. **GR at 0.4 σ; Newton excluded at 6.7 σ; L/L_Newton = 2.06 ± 0.16.**
Dittrich et al. 2025 published 1.839 ± 0.239 ″ from this data, which this sits 0.15 σ from on
an error bar 2.8x smaller. Closed 2026-09-05; the derivation is the cell-2 block of
[`MATRIX_2026.md`](MATRIX_2026.md) and the working log in [`STEP3_2026.md`](STEP3_2026.md).

**Method 2 throughout** — the plate scale is fitted alongside L, because the eclipse and
zenith fields differ by ~600 ppm of focal length and no scale can be imported between them.
The joint scale is 1.8473626 ″/px, 0.2 ppm from the published value; the L–scale correlation
is −0.786 against the paper's −0.783, a property of the field geometry both analyses had to
find independently. A scale fitted *without* L runs ~46 ppm low, because the deflection is
absorbed into it — never compare one of those with a published scale.

| step | what | where |
|---|---|---|
| calibration | 17 zenith fields, quintic, reference fit at a **0.5 ″ gate** (chosen on corner coverage, not on rms) | `station1_record/zenith_recentroid_tol/tol0p5/` (the per-field fits under `zenith_recentroid/<timestamp>/`; the 0.1, 0.2 and 1.0 gates beside it) |
| preprocessing | per-frame coronal subtraction (blur σ 10 px, 2000 ADU pedestal) with the disk occulter, dark + flat | `station1_record/eclipse_corona/<tier>/`, four tiers `0p25s_1810`, `0p3s_1811`, `0p3s_1813`, `0p4s_1812`, re-stacked from raw |
| stage 1 + 2 | windowed + annular centroids; **two-pass match**, gates 20 ″ then 3 ″, `distortion_free_scale` | `station1_record/eclipse_corona/<tier>/stage2_twopass_reftol0p5/` (`mee2024/distortion_fitter.py`; the other gates' trees beside it) |
| stage 3 | pooled Method 2 over **every observation** of the four exposure tiers: per-block offset, rotation and scale, one L, 17 parameters on 1278 coordinates, one 4-MAD vet | **`station1_record/pooled_fit/twopass/`** (`pooled_rows.csv`, `pooled_summary.json`; `tools/matrix_station1/s1_pooled_fit.py --ref twopass`). `pooled_fit/twopass_reftol0p5/` is the same fit from the gate scan, L identical, bootstrap 0.081 against 0.084 from its seed |
| the star sample | G ≤ 13, 2–10 R☉ — 639 observations of 192 stars. The outer cut is set by the zenith-vs-eclipse annulus comparison, not by L | `station1_record/reference_tolerance*.csv`, `blocks_alone.csv` (`s1_reference_tolerance.py`, `s1_blocks_alone.py`) |
| errors | star bootstrap and a cluster-robust sandwich, which agree; a weighted mean over blocks is too small because the blocks share their stars | in `pooled_summary.json` above |
| atmosphere term | 16 zenith Method-2 nulls (±0.109 ″) scaled by airmass^0.73 to the eclipse altitude — mostly the estimator floor, not the sky | `station1_record/zenith_nulls/`, `zenith_nulls.csv`, `zenith_floor.csv` |

Reported **beside** the budget rather than folded into it: the admission rule 0.05 ″, the
model order 0.03 ″, the reference gate 0.01–0.02 ″, the coronal blur 0.01 ″. The flat's
±0.2–0.4 ″ lever is **withdrawn**: it was a vet-selected-subset artefact, and comparing two
reductions on stars each of them chose for itself is the mistake that produced it.

**Graphical output — start at `RECORD/mexico2024\`**: `record_deflection.png` (with
`_all4` and `_per_star`), `record_field.png`, `record_covariance.png`, the four
`master_<tier>_annotated.png`, `station1_star_table.csv` and `record_summary.json`; chart
revision 5, every revision under `station1_record/charts/chart_versions/`, superseded copies
in dated `superseded_*` folders. Built by `tools/matrix_station1/s1_charts_record.py`; the
originals are in `station1_record/charts/`.

Everything else under `station1_record/` (24.7 GB) is the work that chose those settings —
`eclipse_tiers/` (moments against windowed, per tier), `eclipse_corona_s15/` (the 15 px
blur), `moments_on_corona/`, `septic_test/` and `order_test/` (model order), `darks_flats/`
and `eclipse_caldecomp/` (the calibration arms), `zenith_flat_test*/` and
`flat_mechanism.log` (the withdrawn flat lever), `reference_convention/` (A/B/C), the
`zenith_*` nulls and floors, and `pooled_fit/twopass_grid_*` (the admission × magnitude ×
vet grid). Each is named in the cell-2 block of `MATRIX_2026.md` where its number is used.

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
| `matrix_bruns2017_bgtest/` | Bruns' R6 night field re-stacked with the Gaussian background, one archive (2026-08-31) | a probe for the background axis of cell 1's 2×2, superseded by the convention ruling of 2026-09-01 |
| `matrix_bruns2017_windowed_annular/`, `_w0p7/`, `matrix_bruns2017_brunsmethod_windowed/`, `_w0p7/` | cell 1 re-run end to end in the Station 1 convention with the 2 px and the 0.7 px window (2026-09-05, `b17_windowed_annular.py`) | the finding is in the record ("the window must be narrower than the PSF"); the trees are its evidence, not a reduction of record |
| `matrix_bruns2017_night_estimator/` | Bruns' 29 night fields under each estimator (`b17_night_estimator.py`) | feeds the zenith-row and estimator comparisons in `MATRIX_2026.md`; not a reduction of L |
| `F16_ladder/` | CAL_piLeo's 18 frames stacked in sensitive mode without calibration, 2026-08-24 — the first F16 (saturated stars) exposure-ladder test | superseded by `f16_cal_pileo_test/`, `_test2/` (2026-08-29) and `cal_pileo_step2/`; **deleted 2026-09-06** (1.1 GB), the only folder removed in the housekeeping pass |
| `bruns2017_freecubic/` | free-cubic fits of Bruns' 2017-08-19 night calibration, 2026-08-25 | the night-to-night cubic variation (4.8 %) that `INSTRUMENT_COMPARISON.md` builds on; evidence, 3 MB |
| `station2_transfer/`, and its record set `RECORD/mexico2024st2/` (rev02, 2026-09-08) | Mexico 2024 **Station 2**: fifteen zenith fields, the trimmed L/R bracket, and the eclipse field under the Station 1 technique — 17 stars per tier, pooled Method 2 L = 1.770 ± 1.067 ± 0.12 ″, the 13 both-tier stars averaged 1.100 ± 0.958 ″ | not a cell: the bar spans Einstein and Newton, and one inner star moves L by 0.7 ″. It is kept as a record set because it carries the second NP101is's atmosphere null (±0.12 ″) and the night-to-day scale drop (−723 ppm) that the transfer attempt rests on. `docs/STEP3_2026.md`, "Station 2's eclipse field" and "Station 2: an external plate scale" |
| `psf_carrell/`, `psf_london/` | `stars.json` PSF profiles for Carrell's FRA500 + ASI1600 and London's ASI533, 2026-08-26 | two of the six trains in `INSTRUMENT_COMPARISON.md`'s PSF section, beside the cited `psf_leakey/`, `psf_portland/`, `psf_bruns2017/`; evidence, 3 MB |

## Known defects in the record, measured and bounded

1. **The coronal model carves a trench** just outside the saturated core (naive blur
   includes the core's plateau). Fixed in the pipeline; the reductions of record still
   carry it. Re-running is F28-blocked for the pipeline path and pending for the tool path.
2. **Rim artefacts reach the alignment** (F29): 0.8 per frame against 12.8 real stars on
   Bruns EA, because the tool chain runs with the pipeline mask off. Per-frame, not
   per-star, so it moves the star sample rather than biasing astrometry.

Neither changes the quoted numbers; both are why a clean re-run is the next piece of work.
