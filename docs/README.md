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
(`tools/husillos2026/hu_pipeline_check.py`). 

**The tree is organised by dataset, one folder per campaign** (Douglas, 2026-09-17: the
husillos2026 folder "is a nice way to organise things ... this mirrors the organisation in
`F:\MEE_output\RECORD`"). It was not always: until 2026-09-17 the top level held 59 entries
with one campaign's working folders scattered across it -- thirteen `matrix_bruns2017_*`
beside ten `step3_*` beside `cal_pileo_step2`, in alphabetical order and therefore in no
order at all.

The tree is **12 folders** now, one per dataset, with the instrument work beside them.
Eclipse campaigns: `bruns2017\`, `mexico2024\` (both stations), `leon2026\`,
`husillos2026\`, and `refraction\`. Instrument and calibration work:
`bruns_calibrations\`, `askar65phq\`, `carrell2024\`, `portland2026\`, `perfect_optic\`
and `tan_gauge_examples\`. Folders were moved, not renamed, so every old basename still
identifies its folder and almost every rewritten path was a prefix insertion.

Three things are worth stating because each was a trap:

* **A folder name is not evidence of what is in it.** `bruns_np101`, `bruns_rerun` and
  `bruns2600_rerun` are the 2024 instrument and shim tests from `E:\ZenithCals`, not the
  2017 eclipse; `tv85` is Bruns data from **2022**, not 2024; and `step3_record` holds 39
  Leakey 2024 files beside its 52 Leon ones as controls, so the first results file found in
  it names the wrong dataset. Every folder was classified by reading `source_files` and
  `observation_date` inside its own results files.
* **One folder is deliberately not named for a year.** `bruns_calibrations\` spans 2022 and
  2024, so no year fits it. It was `bruns2024\` for about an hour before `tv85` joined it.
* **`refraction\` is Leon data but stays a sibling of `leon2026\`**, because `RECORD\`
  keeps the same split. It is also the one folder whose name is an ordinary English word
  used several hundred times in these documents -- 575 mentions, of which 32 are paths -- so
  a rewrite there has to match the path and not the word.

`askar65phq\` groups by **telescope** rather than campaign: Leakey (2024) and London (2026)
are two owners of one instrument, and the comparison between them is the point. The FITS
headers confirm it rather than the plate scales alone -- `TELESCOP = Askar 65PHQ` and
`TELESCOP = 65PHQ`, `FOCALLEN = 416`.

The rewrites were done in two passes, 175 references then 132, and checked by resolving
every `MEE_output` path mentioned anywhere: 388 resolve and none still names a moved folder.
**Only `F:` paths were repointed.** A drive letter is part of the history here: the first
pass matched any drive and rewrote `docs/LEON_2026-08-11.md` §18.8 as quoted in
the calibration README (now at `F:\MEE_output\leon2026\calibration\`), which
records that the folder once sat at
`D:\MEE_output\HANDOFF_zenith_cubic\` -- a statement about August, not a live path. It was
undone by hand, and the second pass carries an explicit non-`F:` guard. That guard silently
matched nothing on its first writing, because the captured prefix ends in the separator and
the drive letter is therefore not at the end of it; a unit test on sample paths caught it
before the pass ran.

**The frozen calibration inputs left the repository on 2026-09-17.** The Leon zenith
solutions and the CAL_piLeo frame list were versioned here as `calibration/` from
2026-08-28, on the argument that they are inputs to the chain rather than outputs and every
published number is pinned to them. Douglas: *"I am uncomfortable with output data like ...
calibration ending up in the repo."* They are now
`F:\MEE_output\leon2026\calibration\`, with the reasoning and a **SHA-256 manifest of all
eighteen files** kept here as `docs/CALIBRATION_INPUTS.md`. The manifest is the point: the
output tree has no version history, so a checksum under version control is what still lets
anyone confirm the pinned values have not moved. A copy of it sits beside the data as
`SHA256SUMS.tsv`, and `sha256sum -c` against it reports 18 of 18 OK.

Its curated index is **`RECORD\`**: for each finished piece of work,
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
daily Windows scheduled task named **MEE transcript export**, writing to
`OneDrive\Documents\MEE_transcripts\` on each machine. The archive is deliberately in
OneDrive rather than beside the output tree: it exists because transcripts get pruned
and the desktop app crashes, and a copy that shares a disk with the original answers
neither. It is 9 MB. Douglas' archive was on `D:` and then `F:` until 2026-09-14, when
a SECOND daily task was found exporting the same sessions to OneDrive from a forked,
hardcoded copy of the script; the fork and the duplicate task were removed and the
repository's copy now writes the one archive. The archive on
`I:\MEE_transcripts\` is Andrew's export of 2026-08-25. Re-running is safe; a session whose
title changed replaces its earlier file.

The **session handoffs** — the running notes that carry one working session into the next —
live outside the repository, under `F:\MEE_output\`, as
`next_session_prompt_<date>.md` plus a current one for the project in hand. They are working
notes addressed to whoever picks the work up next, not project documentation, and they are
not a substitute for the records above. A copy of one of them lived here as `NEXT_SESSION.md`
until 2026-09-06; it was two handoffs out of date by then and was removed rather than
maintained in two places.
