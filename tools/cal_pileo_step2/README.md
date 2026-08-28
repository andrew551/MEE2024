# CAL_piLeo step-2 analysis

The scripts behind `docs/CAL_PILEO_STEP2.md` — the eclipse-day calibration that produces
the plate scale step 3 imports. They are pure post-processing of the `TWOD_RESIDUALS.csv`
and `CATALOGUE_MATCHED_ERRORS.csv` files each stage-2 run writes; **nothing in the pipeline
was changed to produce any number in that document.**

They lived only on `D:` until 2026-08-28. Versioned because the reductions they read are
regenerable but the reasoning is not, and because `saturation.py` is named in
`docs/STEP3_PLAN.md` as the pattern step 3's F16 per-frame masks must follow.

| script | what it answers |
|---|---|
| `se_analysis.py` | rebuilds the reported standard error from a residual file: HC0 as the pipeline reports it, then HC3 and inverse-variance weights, separating "optimistic estimator" from "field at its floor" |
| `bootstrap_se.py` | pairs bootstrap of the plate-scale error, to adjudicate HC0 against HC3 |
| `estimator_audit.py` | audits that bootstrap — 31 stars against 6 parameters per axis makes some draws near-degenerate, and an RMS summary lets a few blown-up coefficients set the answer |
| `final_errors.py` | the settled position: HC0, HC3 and a delete-one jackknife on every arm. The bootstrap is dropped as headline (heavy-tailed draws put its RMS ~15 % above its own robust scale) |
| `substacks.py` | the three disjoint sub-stacks A/B/C — the split that separates ~2 ppm of centroid noise from ~23 ppm of everything else |
| `tol_ladder.py` | the tolerance ladder rescored on the bootstrap, because HC0 runs lowest exactly where leverage concentrates |
| `levers.py` | what would actually move the error: refits on modified star lists |
| `saturation.py` | **which stars F16 would reject** — the per-raw-frame mask, shifted by the stacker's integer offset and accumulated ("clipped in N of 17 frames") |
| `residual_structure.py` | is the residual noise or an unmodelled field term? Tests for spatial structure and a radial component |
| `psf_shape.py` | adaptive second moments of the stacked stars (a fixed box at this PSF size measures the background, not the star) |
| `vertical_test.py` | **SUPERSEDED** — its parallactic rotation is 90° off. Kept for the record; the method is the alt-az affine in `tools/refraction/m3_maps.py`. Its own header says so. |
| `common.sh`, `run_stage1.sh` | the stage-1 shell drivers as run at the time |

## Three path traps before re-running anything

These scripts are a record of what was run in August 2026. All three of these paths are
now wrong:

1. **`run_stage1.sh` reads from `I:/Leon 2026`.** For **Leon 2026 data specifically**,
   `G:\Leon Aug 2026` is the sole authoritative tree: it is where Douglas corrected frames
   carrying wrong EXPTIME headers or sitting in the wrong exposure folder, so the `I:` and
   `J:\Eclipse data` copies of *that* campaign are superseded backups. The original
   reduction was verified bit-identical against `G:` after the fact
   (`docs/CAL_PILEO_STEP2.md`, provenance correction of 2026-08-27), which is why its
   numbers stand; a fresh run must point at `G:`.

   This rule does **not** generalise to `I:` as a whole. `I:` carries primary data for
   other campaigns that this project legitimately analyses — `I:\Don Bruns 2024`,
   `I:\Kenneth Carrell 2024`, `I:\Leakey 2024`, and `I:\Papers` — and
   `J:\Eclipse data` holds archival material that has also been used. Those are historical
   datasets with no `G:` counterpart; only the 2026 Leon campaign has one.

2. **`common.sh` sets `REFDIR` to `H:/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed`.**
   `H:` is the read-only transfer folder to Andrew. The references now live in
   `calibration/zenith_cubic/`, and the canonical chain uses the **six `08-12` files only**
   — not the twelve that path resolves to.

3. **`run_stage1.sh` sources `common.sh` from a session scratchpad directory** that no
   longer exists. Source it from this folder instead.

They are left as they were rather than quietly repointed, because a script that records
what produced a published number is worth more than a script that runs today. Repoint a
copy.

## Reproducing the canonical reduction

Settings are in `docs/CAL_PILEO_STEP2.md` under "Reproducing". The frame list is
`calibration/cal_pileo_frames.txt` (sixteen `G:` paths), the references are the six
`08-12` files in `calibration/zenith_cubic/`, and `observation_time` is **18:29:35**.
That gives 2.2054043 ″/px, 74 stars, rms 0.5318 ″.

One scripting note that has bitten twice: the reference path contains a space, and an
unquoted shell variable turns it into `D:\MEE2024`, which fails inside
`_open_distortion_files` rather than at argument parsing. Pass the paths as a quoted array.
