# Step 3 — the eclipse-field reduction: plan, inputs, and gates

**Written 2026-08-28** at the close of the refraction/stability session, for the session
that will execute it. The reasoning behind every requirement here is in
`docs/REFRACTION_2026.md` (§10–§17) and `D:\MEE2024 output\MEE_output\CAL_PILEO_STEP2.md`;
this file is the operational contract.

## Why this step 3 differs from 2017 and 2024 — three measured reasons

1. **The atmosphere is inside the fit.** The eclipse sat at alt 9.6°, at the measured
   validity boundary of the standard refraction model (good to ≲50 ppm above 10°,
   ±150–900 ppm-class residuals below). The residual is a spatially coherent wavefield
   (0.2–0.3″ quasi-static, 0.5–1° patches) — so classical per-star statistics overstate
   the information by the patch count, several-fold.
2. **The calibration is one-sided.** No L/R bracket: the CAL-to-eclipse sightline
   differential was night-measured at 25–65 ppm (epoch-dependent), uncancellable by
   design, and M5's rehearsal projects residual fields of the measured classes into
   **δL = −0.97 / +0.23 / +0.41 ″ across three real atmospheres** — order 0.1–0.6″
   (6–35 % of L) at the true separation.
3. **`tol 999` is booby-trapped.** 1–3 persistent catalogue mismatches per ~100-star
   field (up to 64.5″) survive frame averaging — measured, with casualties.

## Canonical inputs (do not re-derive; provenance is settled)

| input | value / location |
|---|---|
| raw frames | `G:\Leon Aug 2026\2026-08-12\Eclipse\SCI_ladder\` — tiers `0.1s/ 0.3s/ 0.6s/ 1.2s/` + `discard/` (one ~50 ms default-setting frame). **G: only**; `I:` and `J:\Eclipse data` are archival backups, never analysis sources. Folder organisation is truth; EXPTIME headers lie on block-first frames after a SET EXPOSURE change (verify by sky level). |
| CAL_piLeo result to import | `D:\MEE2024 output\MEE_output\cal_pileo_step2\canonical_16f_night2refs\` — **2.2054043 ″/px**, 74 stars, rms 0.5318″, `observation_time 18:29:35`, HC0 21.6 ppm (quote HC3-class ~25 ppm) |
| zenith references | the **six 08-12 files only** (night-2; the two-night 12-file set is superseded — the telescope was transported between night 1 and eclipse day and measurably changed: tilt dipole ×2, +197 ppm scale). `D:\MEE2024 output\MEE_output\Claude Code\HANDOFF_zenith_cubic\inpipeline_windowed\08-12_Z*.txt`; d(3000) carried forward = **3.0297″**, not 3.1048 |
| site / weather (totality) | 42.740470 / −5.613780 / 1101 m; T 30.5 °C (box logger) with **29.2 ± 0.7 °C the better air estimate** (FOCTEMP − 7.74 K, night-validated); P 896.6–896.8 hPa; RH 0.208; λ 0.62 µm |
| contacts | C2 ≈ 18:28:07, C3 = 18:29:53.9 UTC (LEON §3.1) |
| eclipse geometry | Sun alt 9.70° az 281.2°, R⊙ = 947.1″; frame centre 0.74° above the Sun (RA 142.68° Dec +15.40°); camera PA 45.2° E of N (ROLL ≈ 315.5°); Sun at px ≈ (3208, 3293) |
| star field | 132 stars G ≤ 11 outside 2 R⊙ (2.0–10.3 R⊙); **F&L h = 19.8 R⊙²** → 1.07 %/ppm naive, ~0.17 %/ppm through the real basis for scale-like content (M5) |

## Design requirements (each carries a measured justification)

- **Per-frame reduction, per tier**, then robust per-star aggregation (medians, MAD clip)
  — never a blind tol-999 stack. The M2/M3 machinery (`tools/refraction/`) is proven on
  810 reductions.
- **F16 per-frame saturation masks per tier** (`analysis/saturation.py` pattern): short
  tiers own the unsaturated inner stars, 1.2 s owns depth; combine at catalogue level.
- **Chain fidelity**: zenith cubic (6 files) → CAL_piLeo low orders →
  `distortion_fixed_coefficients=constant`, corrections ON, per-frame mid-exposure
  `observation_time` (the refraction scale term moves 1.78 ppm/s at this altitude).
- **The three diagnostics, wired in from the start**: (1) Method-1 vs Method-2 gap
  (Method 2 is diagnostic only — it swung −41 % to +195 % on real night fields);
  (2) the 1/R residual-gradient check of LEON §16.1 — now the arbiter of the cubic
  transfer, which is uncertain at the ≥7 %-class with **unknown sign** (the −7.3 %
  within-night step, cause unresolved; §16.4–16.6); (3) error bars by **spatial block
  bootstrap** at the 0.5–1° coherence scale, not per-star OLS.
- **The nuisance estimator (the S1 gate)**: joint fit [N1, N2, Θ, L·(1/R)û + smooth
  vertical nuisance], outer stars (R ≳ 7 R⊙) constraining the nuisance. Validate on the
  night nulls first: the M5 rehearsal data (`refraction/m5_rehearsal/`, three windows,
  zero true deflection) must show the estimator shrinking the raw −0.97/+0.23/+0.41″
  biases toward the noise floor AND recovering an injected L = 1.751″ unbiased at ≤2 %.
  If the gate fails, L is quoted with the ±0.1–0.6″ systematic stated, per the
  decision-before-looking rule.

## Phases

- **S0** — SCI_ladder inventory: headers, per-tier counts, timing vs C2/C3, the discard
  triage, per-frame sky levels (exposure-transition artefacts!), saturation radii per
  tier, drift; one pilot solve near the Sun. *An evening; read-only.*
- **S1** — nuisance estimator built and gated on the night nulls (above). *The decisive
  phase; analysis only.*
- **S2** — per-frame reduction of all usable SCI frames through the chain. *Machine
  hours, overnight, drivers adapted from `drive_horizon.py`.*
- **S3** — L: exact-2017-replica Method 1 (comparability), the nuisance estimator
  (headline), Method 2 (diagnostic); block-bootstrap errors; the systematic budget
  assembled from the campaign's measured numbers.
- **S4** — write-up, with the 2017/2024/Leon methods table.

Branch suggestion: `step3-leon-2026` off `refraction-leon-2026` (which is `v1.4.0-dev`
plus docs/tools only, so everything rides along and nothing under `mee2024/` changes).

## Figures and the campaign comparison

`docs/figures/` holds the six current figures; `tools/refraction/FIGURES.md` maps each to
the script that regenerates it and records the caption facts. `docs/STEPS12_LEON_VS_BRUNS2017.md`
is the like-for-like steps 1–2 comparison against Bruns 2017 (all four zenith datasets with
three uncertainty estimators, and the calibration-field comparison).

## Standing traps (each cost this project real time at least once)

- Confirm the data source against `G:\Leon Aug 2026` **before** any reduction.
- Paths contain spaces: quote everything; Python drivers over shell one-liners; launch
  detached jobs from the repo (no spaces) and **verify the log is alive before waiting**.
- The weather logger CSV is UTF-16, local = UTC+2, and must be read with **timezone-aware
  datetimes** (a naive `.timestamp()` on this UTC+1 machine shifts the axis 1 h); never
  average the file whole (it contains the pack-up descent).
- `platescale_relative_uncertainty` is HC0 and under-reports by 15–36 % at eclipse-field
  star counts; quote HC3/jackknife-class.
- `analysis/vertical_test.py` is superseded (90° rotation error); use the alt-az affine
  (`m3_maps.py`).
- The venv, never system Python. Write only to `D:\MEE2024 output\MEE_output` and the
  repo.
