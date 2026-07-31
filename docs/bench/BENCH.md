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
