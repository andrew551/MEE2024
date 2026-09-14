# The `docs/` folder — what is here, and what to trust

Index written 2026-09-06. Twenty-nine documents had accumulated with no map, and several
carried status lines the code had long since contradicted. This page says what each file is
**for**, and — where it matters — what in it has been overtaken.

Two conventions the project follows, worth knowing before you read anything here:

* **Records are not rewritten.** A document dated to a campaign or a session stays as it was
  written, including the wrong turns, because how a mistake was made is usually the lesson.
  When a later measurement withdraws something, a banner is added at the point of the claim
  and the original text is left standing. So a date on a document is load-bearing.
* **The live documents are few.** Most of this folder is evidence. Four files are maintained.

---

## Start here

| | |
|---|---|
| [`ONBOARDING.md`](ONBOARDING.md) | What the project measures and why the rules are the rules. Read this first. |
| [`../CLAUDE.md`](../CLAUDE.md) | The same rules in short form, loaded automatically by Claude Code. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How data flows through the three stages, and the on-disk contract between them. |

## Maintained — these are kept current

| | |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | The live plan. §1 measurements, §2 closed fixes, §3 features F1–F31, §4 open questions, §6 the release order. Long because it is the argument, not a summary. |
| [`V1_4_0_TESTING.md`](V1_4_0_TESTING.md) | The work plan for `v1.4.0-dev`, the branch holding everything that can change a measured number. Read it before working on that branch. |
| [`MATRIX_2026.md`](MATRIX_2026.md) | The four-dataset matrix: one tool chain against three eclipses and a full moon. **The current error budget for every cell is here**, and this is the file to trust when two documents disagree. |
| [`STEP3_2026.md`](STEP3_2026.md) | The running record of the eclipse-field reductions, newest last. Numbers with units, supersessions marked in place. |

## Which number is the record

[`WHICH_REDUCTION_IS_THE_RECORD.md`](WHICH_REDUCTION_IS_THE_RECORD.md) — for each closed
cell, which run is quoted and where its output lives. Anything not listed there is an
experiment, not a result. It lags the matrix; check both.

As of 2026-09-06 three cells are closed and one is open:

| cell | dataset | result |
|---|---|---|
| 1 | Bruns 2017, Casper WY | L = 1.764 ″, Method 1, reduced by Bruns' own procedure |
| 2 | Mexico 2024, Station 1 | L = 1.804 ± 0.084 (stat) ± 0.11 (atmosphere) ″, Method 2 |
| 3 | Leon 2026 | L = 1.914 ″ with a much wider bar, Method 1, one-sided calibration |
| 4 | Portland moon 2026-07-29 | the null check, L = 0 |

General relativity predicts 1.751 ″ at the solar limb.

## Campaign and dataset records

Each is the record of one dataset, dated, and stays as written.

| | |
|---|---|
| [`LEON_2026-08-11.md`](LEON_2026-08-11.md) | The 2026 Leon campaign: calibration, distortion, temperature, and §18.11's gauge trap — **read §18.11 before comparing any distortion coefficient with any other program.** The 16× conclusion built on it is withdrawn; the banner there points to the correction. |
| [`PORTLAND_2026-07-29.md`](PORTLAND_2026-07-29.md) | The Portland zenith set, the dress rehearsal for Leon. |
| [`PORTLAND_MOON_2026-07-29.md`](PORTLAND_MOON_2026-07-29.md) | Can it work beside a full Moon? Yes. This dataset is the matrix's null cell. |
| [`CAL_PILEO_STEP2.md`](CAL_PILEO_STEP2.md) | The CAL_piLeo plate scale. Its headline result is superseded **in place** — the banner at the top gives the corrected 16-frame set. |
| [`STEPS12_LEON_VS_BRUNS2017.md`](STEPS12_LEON_VS_BRUNS2017.md) | Leon against Bruns 2017 at the same pipeline stage, three error estimators each, 2026-08-27. |
| [`INSTRUMENT_COMPARISON.md`](INSTRUMENT_COMPARISON.md) | Six optical trains on one footing. §5 is the dropped-sign correction that withdraws LEON §18.11's conclusion. |
| [`REFRACTION_2026.md`](REFRACTION_2026.md) | The Leon refraction data and the plan for it — the branch record for `refraction-leon-2026`. Written as strategy; the ladder has since run. |
| [`LEON_SCRIPT_REVIEW.md`](LEON_SCRIPT_REVIEW.md) | Review of the capture scripts for the gain 101 → 0 change, against the Portland findings. |

## The step-3 reduction

| | |
|---|---|
| [`STEP3_PLAN.md`](STEP3_PLAN.md) | The operational contract step 3 was executed against, and the three measured reasons it differs from 2017 and 2024. |
| [`STEP3_CHARTS_AND_SETTINGS.md`](STEP3_CHARTS_AND_SETTINGS.md) | The specification for what the reduction work should leave behind in the product: the chart set, the atmospheric machinery, the exact settings. |

## Method and reference notes

| | |
|---|---|
| [`STAGE3_THEORY.md`](STAGE3_THEORY.md) | Freundlich & Ledermann 1944, Bruns 2017, and what the code actually computes. Names the correspondence between every estimator and its ancestor. |
| [`PSF_REVIEW.md`](PSF_REVIEW.md) | What the literature says about PSF modelling and centroiding, judged in this project's regime. |
| [`CALIBRATION_FRAMES.md`](CALIBRATION_FRAMES.md) | Do darks and flats earn their place? Separates what is measured from what is merely observed and what is untested. |
| [`FIELD_PRESETS.md`](FIELD_PRESETS.md) | The two standard stage-1 configurations, taken from the reductions of record rather than from intent. |
| [`CATALOGUE_INVENTORY.md`](CATALOGUE_INVENTORY.md) | What star catalogues the project holds, with row counts and epochs. |

## Design documents

Written before the code, kept for the reasoning. Each now states what shipped, because
several of them said "not yet implemented" long after they were.

| | |
|---|---|
| [`STARCAT_DESIGN.md`](STARCAT_DESIGN.md) | The star-catalogue abstraction. **Implemented** as `mee2024/starcat/`. |
| [`PLATESOLVER_DESIGN.md`](PLATESOLVER_DESIGN.md) | Measured behaviour of the classic solver, and the statistical groundwork under both solvers. |
| [`PLATESOLVER_V2_DESIGN.md`](PLATESOLVER_V2_DESIGN.md) | The v2 rebuild. **Implemented, and now the default.** |
| [`UI_DESIGN.md`](UI_DESIGN.md) | What the app window is and why it exists beside the classic interface. |
| [`UI_ROADMAP.md`](UI_ROADMAP.md) | Two interfaces — keep both, freeze one, or port selectively. Partly overtaken; the app window exists. |

## Subfolders

| | |
|---|---|
| [`bench/`](bench/) | The error budget, the solver and centroid benchmarks, the hot-pixel and PSF studies, and the test-frame register. `ERROR_BUDGET.md` is the one to read. |
| [`releases/`](releases/) | Release notes, one file per version. `MEE2024.spec` copies the matching one beside the binary at build time. v1.3.6 has none — that is the release the project lost, and the reason version numbers are never reused. |

---

## Not in this folder

**The outputs.** The repository carries the documents; the figures, star tables and summaries
they cite live in the output tree on one machine, `F:\MEE_output\`, which is
far too large to version. **It has moved twice in three weeks**: `D:\MEE_output` until
2026-08-26, then `D:\MEE2024 output\MEE_output`, and since 2026-09-14 `F:\MEE_output`
— the D: drive had 28 GB left and this tree was 245 GB of it. Older handoffs, transcripts and the
paths recorded inside the output files themselves still name the D: locations; they mean
this tree. The documents and tools here were rewritten at the move, except where they
recount the history above. All three stages were then re-run from F: on a real capture
and reproduced the recorded run field for field, down to the 49-frame alignment record
(`tools/husillos2026/hu_pipeline_check.py`). Its curated index is **`RECORD\`**: for each finished piece of work,
the outputs someone would need to check it, re-quote it or publish from it — one folder per
matrix cell in a common shape (`bruns2017`, `leon2026`, `mexico2024`), plus `refraction` for
the Leon campaign physics that feeds the atmosphere term. Every file there is a copy of one in
the tree; superseded versions go in dated `superseded_*` folders and are never deleted. The
cell folders are written by their chart tools; `tools/sync_record.py` keeps the rest in step,
including a copy of `WHICH_REDUCTION_IS_THE_RECORD.md`, the folder's own index. The repository
holds no figures of its own: a `docs/figures/` with six refraction charts existed until
2026-09-06, but no document displayed them and the record cited the D: originals, so they
moved to `RECORD/refraction/` with everything else.

**The transcripts.** Every Claude Code session is a `.jsonl` under the user's
`~/.claude/projects/`, a folder the tool prunes on a timer and the desktop app has lost to a
crash before. `tools/export_transcripts.py` (Andrew's, 2026-08-20; in the repository since
2026-09-06) writes each session as readable Markdown with an `INDEX.md`, conversation text in
full and tool output truncated. Each machine keeps its own archive and runs the export on a
daily Windows scheduled task named **MEE transcript export**: Douglas' at
`D:\MEE2024 output\MEE_transcripts\`, Andrew's under his OneDrive Documents. The archive on
`I:\MEE_transcripts\` is Andrew's export of 2026-08-25. Re-running is safe; a session whose
title changed replaces its earlier file.

The **session handoffs** — the running notes that carry one working session into the next —
live outside the repository, under `F:\MEE_output\`, as
`next_session_prompt_<date>.md` plus a current one for the project in hand. They are working
notes addressed to whoever picks the work up next, not project documentation, and they are
not a substitute for the records above. A copy of one of them lived here as `NEXT_SESSION.md`
until 2026-09-06; it was two handoffs out of date by then and was removed rather than
maintained in two places.
