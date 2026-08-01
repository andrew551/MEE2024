# Solver bench results

Stage-over-stage A/B record for the plate solver rebuild
(`docs/PLATESOLVER_V2_DESIGN.md`). Every section is appended by
`python tools/solver_bench.py compare A.json B.json --md docs/bench/BENCH.md`
against the committed JSON of the previous stage. Corpus: `tools/solver_bench.py`,
`CORPUS_VERSION = 1`, 88 cases.

## S0 baseline — v1 @ 54522e1 (2026-07-31, `s0_baseline.json`)

Overall: correct 61/80 solvable (76.3%), **wrong solves 0**, junk rejected **8/8**,
real fields **2/2**. DB load 14.8 s; median solve 7–13 s, failures 13–17 s (the
mirrored retry doubles the work).

| family | correct | notes |
|---|---|---|
| fov (midlat) | 6/10 | 0.6°, 1° fail (star-list depth); 2–8° pass; 10°, 12°, 18° fail — matches the measured envelope in PLATESOLVER_DESIGN.md §1 |
| fov (galplane) | 5/10 | 2.4–10° pass (10° passes here — density helps); 0.6°, 1°, 2°, 12°, 18° fail |
| reliability | 31/32 | one 8° draw fails. Milder than the documented 6/8 & 4/8 — this corpus's default ordering scatter (0.3) is gentler than the original protocol; the scatter family probes it directly |
| noise | 2/4 | 0.3, 2 px pass; 4, 8 px fail (the documented edge is ~4–5 px; this draw lands just past it) |
| scatter | 11/12 | one failure at scatter 0.6 (midlat) — the ordering-scatter axis S2b/S5a target |
| sparse_detect | 2/3 | 7 detections fail, 10 and 12 pass — exactly the documented floor |
| pole | **0/4** | both poles, both draws. Predicted: roll is ill-conditioned near the pole, so candidate rolls scatter beyond TOL_ROLL and the consensus splits. The S4 (quaternion consensus) target. Pole fields carry 100–150 stars, so this is the solver, not the field |
| rollwrap | 3/3 | roll 0 / 57 / 359.9 all pass at 2.4° — the wrap split is probabilistic, not deterministic; kept in the corpus as an S4 canary |
| junk | 8/8 rejected | zero false positives — the gate every stage must hold |
| real | 2/2 | zwo1 + zwo3 solve in 7.2 s each |

Failed solvable cases run 13–17 s because failure triggers the full mirrored retry;
candidate counts on failures reach ~2.6 M summed across both passes (~1.3 M per pass,
matching the memory incident in `progress.md`).

## S1 — Gaia pattern DB, same algorithm (`s1_gaia.json`)

v2 = the ported algorithm reading `patdb_g12_t17` (112,660 anchors / 17.24 M
triangles built from `gaia_dr3_g12`), with verification against the same Gaia
catalogue instead of the bundled Tycho npz. Same structural parameters, same
invariant, same tolerance — only the catalogue and format changed.

**Gate: PASS.** Wrong solves 0, junk 8/8, real fields 2/2 (parity check: v2 agrees
with v1's solutions on both real fields to 0.33″ in ra/dec, 0.005° in roll, 1.3×10⁻⁴
in scale — the roll conventions ported bit-for-bit). Overall correct 61 → 62/80.

What moved, and why:

- **Verification margins are transformed.** On the ordering-scatter family v2 matches
  81–88 stars where v1 matched 27–50 on identical fields — Gaia's sub-mas positions
  make nearly every catalogue star matchable at 36″ where Tycho's 2.5″ tail lost many.
  This is the headroom S3 will spend when it tightens the tolerance.
- `fov_galplane_fov2` and the 8 px-noise case flipped to solving. Median solve times
  dropped ~25–45 % per family (verification against the mmap'd catalogue replaces the
  in-solve Tycho work); DB load 14.8 s → 12.0 s.
- **One knife-edge regression, root-caused**: `sparse_detect_n10` (10 detections). Both
  solvers find the identical true pointing; v1 verified 9/10 against threshold 9 and
  squeaked in, v2 verifies 8/10 against the same 9 — the ~3× denser verification
  catalogue lets a fainter neighbour disqualify one marginal match via the 2×
  confusion test. The honest fix is S3's: adapt verification depth to the detection
  count (a G<12 comparison set adds only confusers, never matches, when just 10 bright
  stars are detected). Until then the measured sparse floor for v2 is ~12 detections
  vs v1's ~10, on this seed.
- The two `scatter 0.6` flips (one fixed, one broke, family total unchanged at 11/12)
  are churn at the hardest ordering-scatter setting: the DB's brightness ranking is now
  G-band rather than V-band, so *which* marginal draw survives changes. This is the
  axis S2b's dimmer-legs decision and S5a's anchor sampling are aimed at.
- Poles still 0/4, as expected — the consensus chart is untouched until S4.

## v1@54522e1 vs v2@1ec236e (2026-07-31T23:44:47)

corpus v1, 88 shared cases. DB load 14.81 s -> 12.01 s.

| family | correct | wrong | median time (s) |
|---|---|---|---|
| fov | 10/20 -> 11/20 | 0 -> 0 | 12.83 -> 6.92 |
| junk | 8/8 -> 8/8 | 0 -> 0 | 14.46 -> 12.93 |
| noise | 2/4 -> 3/4 | 0 -> 0 | 16.18 -> 14.18 |
| pole | 0/4 -> 0/4 | 0 -> 0 | 15.37 -> 13.79 |
| real | 2/2 -> 2/2 | 0 -> 0 | 7.2 -> 6.45 |
| reliability | 31/32 -> 31/32 | 0 -> 0 | 7.19 -> 6.42 |
| rollwrap | 3/3 -> 3/3 | 0 -> 0 | 11.38 -> 10.16 |
| scatter | 11/12 -> 11/12 | 0 -> 0 | 11.91 -> 7.11 |
| sparse_detect | 2/3 -> 1/3 | 0 -> 0 | 2.37 -> 2.5 |

overall correct rate 0.7625 -> 0.775; wrong solves 0 -> 0; junk rejected 8/8 -> 8/8

case flips:
- fixed: fov_galplane_fov2_s0
- fixed: noise_midlat_fov2.4_noise_px8_s0
- BROKE: scatter_midlat_fov2.4_mag_order_scatter0.6_s0
- fixed: scatter_midlat_fov2.4_mag_order_scatter0.6_s1
- BROKE: sparse_detect_midlat_fov2.4_n_detect10_s0

## S2 — Kendall shape invariant + single-pass mirror (`s2_kendall.json`)

v2 reading `patdb_g12_t17k` (same anchors/legs as t17; invariant = Kendall shape
sphere with 3-bit permutation codes; calibrated tolerance 0.005). Mirror coverage is
now part of one query pass: extraction is shared and the mirror pool is only queried
and clustered when the normal pool fails. `patdb_g12_t17k` is the auto-selected
default from this stage; `patdb_g12_t17` remains the named rollback.

**Gate: PASS.** Wrong solves 0, junk 8/8, real fields 2/2. Overall 62 → **64/80**;
**reliability 32/32** (the 8° ordering draw fixed) and **scatter 12/12** (the S1
casualty recovered) — the shape metric keeps marginal draws that the distorted
(ratio, dphi) metric lost. `fov_galplane_12°` now solves: the first movement past
the documented 10° ceiling, from shape-exact matching alone.

Measured mechanics:

- Solved-case candidates 1.27 M → 758 k (0.6×); solve medians down 20–25 % across
  families; failure median 13.0 → 10.2 s (extraction+query shared with the mirror
  pool). Not the naive halving: at tolerance 0.005 consensus still dominates
  failures — the deep cuts belong to S3.
- **Calibration honesty** (scratch run on 1609 true pairs over 23 fields): true-pair
  Kendall distances run median ~0.001, q99 0.002–0.007 per clean field, growing with
  FOV and noise exactly as the §1.1 error budget predicts. The dev spike's 0.001 was
  right for clean stacked fields; this corpus's 3 px distortion floor sets 0.005.
  Consequence: the candidate reduction at *equal stringency* is modest; the 10–50×
  cut arrives with S3's per-image adaptive tolerance.
- **A structural find**: the Kendall rep is vertex-symmetric, so the same physical
  star triple stored under 2–3 anchors matches one image triangle repeatedly with
  identical solutions — 36,544 raw ≥4-clusters vs S1's 160 on the same field. The
  consensus now gates on *distinct image triples*, vectorised; warm kendall solves
  went from 11.3 s (slower than S1) to 5.2 s (faster). An S5 option is recorded: the
  dedupe could move into the DB (store each triple once), which is S2b's dimmer-legs
  question from the other direction.
- One boundary churn: the 8 px-noise case (fixed by S1) fails again — 8 px of
  centroid noise pushes true-pair shape distances beyond the calibrated 0.005, while
  S1's coarser metric happened to keep it. It sits beyond v1's documented envelope
  (~4–5 px) and beyond the calibration set; S3's noise-adaptive radius is the
  designed fix. Sparse-10 (S3) and poles 0/4 (S4) unchanged, as assigned.

## S3 (+S2b) — the tolerance model; dimmer-legs judged and declined (`s3_adaptive.json`, `s3_dimmer.json`)

The query radius is now the calibrated model
`r = 0.0006 + 4.8·(2√2·ε/S_img) + 0.93·(θ_db/2)²` per triangle (fitted on 4,677
identity-verified true pairs; the curvature coefficient ≈ 1 *confirms* the design
doc's projective prediction), with ε from `platesolve_noise_px` (default 0.3),
candidates re-cut with their actual catalogue-triangle size, an escalation ladder
(×3 on total failure, radii capped at 0.02, candidates budgeted at 1.5 M/pool), and
verification depth capped at 8× the detection count so sparse fields are not judged
against a comparison set they could never match. 0.81 % of true pairs canonicalise
onto the mirror point (noisy chirality of near-degenerate triangles) — the
irreducible loss band the ≥4 consensus absorbs.

**Gate: PASS.** 64 → **65/80**, wrong solves 0, junk 8/8, real fields 2/2. The
8 px-noise case is recovered by the escalation ladder. Success-path medians
collapse: real fields 5.4 → **1.6 s**, reliability 2.7 s, scatter 2.7 s; solved-case
candidates 758 k → 518 k median, and on the real (low-noise) fields ~156 k — the S1
verification margin being spent, as designed. The cost: failed solves now pay the
ladder — junk 10 → 28 s, poles 11 → 35 s (bounded by the candidate budget; S4 turns
poles into successes and junk fields are rare in practice).

**sparse-10, honestly**: the depth fix works (the true candidate's matches went
6 → 8 of 10 once the cap respected the bbox-vs-frame geometry), but the case still
fails because the acceptance threshold has a *floor* of ~9 = 3 defining stars + 3
addon (the deliberate safety margin) + x1 ≈ 3 — density cannot push it lower. Fixing
it means revisiting the addon for the tiny-λ regime, which is acceptance-statistics
work (the corrected-p-value experiment in the design doc), not tolerance work. Left
failing, with the mechanism pinned.

**S2b decision — pre-registered rule applied: NOT adopted.** At identical size
(230 MB, same anchor/triangle counts), `patdb_g12_t17kd` ties the scatter sweep
12/12 (the axis the rule names) and *breaks* `fov_galplane_8°`, `fov_galplane_12°`
and one 8° reliability draw — 65 → 62/80. Physical cause: storing only
dimmer-than-anchor legs thins patterns of exactly the bright stars that wide fields'
top-18 window depends on. The dedupe idea's remaining home is S5's index layout, not
the pattern content. `patdb_g12_t17k` stays the default.

## v2@1ec236e vs v2@292d79e (2026-08-01T00:53:58)

corpus v1, 88 shared cases. DB load 12.01 s -> 13.21 s.

| family | correct | wrong | median time (s) |
|---|---|---|---|
| fov | 11/20 -> 12/20 | 0 -> 0 | 6.92 -> 5.48 |
| junk | 8/8 -> 8/8 | 0 -> 0 | 12.93 -> 10.17 |
| noise | 3/4 -> 2/4 | 0 -> 0 | 14.18 -> 10.14 |
| pole | 0/4 -> 0/4 | 0 -> 0 | 13.79 -> 10.65 |
| real | 2/2 -> 2/2 | 0 -> 0 | 6.45 -> 5.37 |
| reliability | 31/32 -> 32/32 | 0 -> 0 | 6.42 -> 4.97 |
| rollwrap | 3/3 -> 3/3 | 0 -> 0 | 10.16 -> 5.51 |
| scatter | 11/12 -> 12/12 | 0 -> 0 | 7.11 -> 5.28 |
| sparse_detect | 1/3 -> 1/3 | 0 -> 0 | 2.5 -> 2.04 |

overall correct rate 0.775 -> 0.8; wrong solves 0 -> 0; junk rejected 8/8 -> 8/8

case flips:
- fixed: fov_galplane_fov12_s0
- fixed: reliability_midlat_fov8_s3
- BROKE: noise_midlat_fov2.4_noise_px8_s0
- fixed: scatter_midlat_fov2.4_mag_order_scatter0.6_s0

## v2@292d79e vs v2@26f6660 (2026-08-01T02:33:59)

corpus v1, 88 shared cases. DB load 13.21 s -> 12.74 s.

| family | correct | wrong | median time (s) |
|---|---|---|---|
| fov | 12/20 -> 12/20 | 0 -> 0 | 5.48 -> 3.43 |
| junk | 8/8 -> 8/8 | 0 -> 0 | 10.17 -> 27.69 |
| noise | 2/4 -> 3/4 | 0 -> 0 | 10.14 -> 16.75 |
| pole | 0/4 -> 0/4 | 0 -> 0 | 10.65 -> 34.94 |
| real | 2/2 -> 2/2 | 0 -> 0 | 5.37 -> 1.59 |
| reliability | 32/32 -> 32/32 | 0 -> 0 | 4.97 -> 2.67 |
| rollwrap | 3/3 -> 3/3 | 0 -> 0 | 5.51 -> 2.87 |
| scatter | 12/12 -> 12/12 | 0 -> 0 | 5.28 -> 2.73 |
| sparse_detect | 1/3 -> 1/3 | 0 -> 0 | 2.04 -> 2.96 |

overall correct rate 0.8 -> 0.8125; wrong solves 0 -> 0; junk rejected 8/8 -> 8/8

case flips:
- fixed: noise_midlat_fov2.4_noise_px8_s0

## v2@26f6660 vs v2@26f6660 (2026-08-01T02:34:00)

corpus v1, 88 shared cases. DB load 12.74 s -> 13.01 s.

| family | correct | wrong | median time (s) |
|---|---|---|---|
| fov | 12/20 -> 10/20 | 0 -> 0 | 3.43 -> 3.45 |
| junk | 8/8 -> 8/8 | 0 -> 0 | 27.69 -> 23.17 |
| noise | 3/4 -> 3/4 | 0 -> 0 | 16.75 -> 14.22 |
| pole | 0/4 -> 0/4 | 0 -> 0 | 34.94 -> 29.51 |
| real | 2/2 -> 2/2 | 0 -> 0 | 1.59 -> 1.28 |
| reliability | 32/32 -> 31/32 | 0 -> 0 | 2.67 -> 2.29 |
| rollwrap | 3/3 -> 3/3 | 0 -> 0 | 2.87 -> 2.62 |
| scatter | 12/12 -> 12/12 | 0 -> 0 | 2.73 -> 2.32 |
| sparse_detect | 1/3 -> 1/3 | 0 -> 0 | 2.96 -> 2.28 |

overall correct rate 0.8125 -> 0.775; wrong solves 0 -> 0; junk rejected 8/8 -> 8/8

case flips:
- BROKE: fov_galplane_fov8_s0
- BROKE: fov_galplane_fov12_s0
- BROKE: reliability_midlat_fov8_s3

