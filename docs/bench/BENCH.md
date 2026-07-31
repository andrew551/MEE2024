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

