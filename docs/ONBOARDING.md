# Getting started on MEE

For a new contributor, human or assisted. [`../CLAUDE.md`](../CLAUDE.md) holds the rules in
short form and is loaded automatically by Claude Code; this file explains why they are the
rules, and what the project is actually trying to measure.

## What the program is for

In 1919 Eddington measured starlight bending around the eclipsed Sun. This pipeline does the
same measurement with modern equipment: photograph the star field around a totally eclipsed
Sun, measure where each star appears, compare with where the catalogue says it should be, and
fit the deflection constant.

The whole difficulty is that the signal is small and everything else is larger. Deflection at
the solar limb is 1.75″, falling as 1/r. The errors competing with it — distortion,
refraction, alignment, centroiding — are measured in this repository in
[`bench/ERROR_BUDGET.md`](bench/ERROR_BUDGET.md), and several are the same size or bigger.
That is why the codebase is unusually full of measurements: nearly every design decision here
was settled by measuring, and reversed at least once.

## Set up

Python 3.12+ -- the floor is numpy's and scipy's, not the code's. From a clone:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expect **876 passed, 26 skipped**. The skips need `--runslow`: the 127 MB triangle database
or a network call.

You do not need any observational data to work on this. The suite generates synthetic star
fields in `tests/conftest.py`. Nothing to download, nothing to ask for.

`tests/data/fits/` held 3.5 GB of real frames until 2026-08-21 and no longer exists: it was
gitignored, no test read it, and every one of its 82 files was verified byte-identical
against the source drive before deletion. What remains under `tests/data/` — `fields/` and
`gaia/` — *is* tracked and *is* read, so leave it alone.

The folder was kept whole rather than scattered back into the source captures, because two
of its six subfolders are curated *subsets* and the selection cannot be reconstructed from the
raw data. It now lives at `I:\MEE test frames\fits\`, which is what the benchmark and
release-check commands in `docs/bench/` and `RELEASING.md` name.

[`docs/bench/TEST_FRAMES.md`](bench/TEST_FRAMES.md) is the record: every frame with its size
and SHA-256, which capture each came from, and which are subsets. Without that drive those
commands cannot run, which costs nothing for development — they reproduce published
measurements and are not part of the suite.

The bundled Gaia G<10 catalogue means a fresh install plate-solves offline immediately.
Deeper catalogues (`gaia_dr3_g13`, `gaia_dr3_g15`) download on demand from the repository's
own releases, and unpack into a per-user data directory.

## The three interfaces, and why there are three

| | what it is | who uses it |
|---|---|---|
| classic UI | `UI_handler.py`, FreeSimpleGUI, forty-odd options on one form | the original; still the only interface that can run stage 3 and enable astrometric corrections |
| app window | `mee2024/ui/`, a local HTTP server and an HTML frontend | the current one — batch runs, live progress, eight options |
| CLI | `mee2024/cli.py` | scripting and unattended runs |

They are not layers; they are three front ends onto the same pipeline. The recurring bug
class is a fix that lands in one of them: correctness then depends on which interface you
used. When you fix something, ask whether it belongs where the data is rather than where the
button is.

The convergence question — whether the classic UI can be retired — is open, and gated on the
app window learning to do the three things only the classic one can. See
[`UI_ROADMAP.md`](UI_ROADMAP.md).

## The pipeline

**Stage 1** (`stacker_implementation.py`) reads frames, subtracts a master dark, divides by a
master flat, masks hot pixels, aligns and stacks, finds centroids, and plate-solves. Output is
a `CENTROID_OUTPUT*` folder with the stack, the centroids and a solve.

**Stage 2** (`distortion_fitter.py`) matches the centroids against a star catalogue and fits a
distortion polynomial. This is where the astrometric quality is decided, and where the
regression baselines live.

**Stage 3** (`eclipse_analysis.py`) fits the deflection constant from an eclipse field.

Frames are addressed as paths, including inside SER video containers (`capture.ser#42`), so
the one-frame-one-path model survives.

## Working style this project expects

**Measure before claiming.** The documents here are full of "this looked right and was
wrong". If you assert that a change improves accuracy, show the number. If you cannot measure
it, say that instead.

**A finding survives its explanation.** More than once the observation was real and the
reason given for it was not. When a result is challenged, re-derive it from the raw data
rather than defending the previous account.

**Say what you did not do.** Partial work reported as complete is the expensive failure here,
because the next decision gets made on it.

**Prefer reporting to correcting.** Where the data is wrong — headers that lie about exposure,
site coordinates 18 km off — the pipeline reports the discrepancy rather than silently
fixing it. Silent correction destroys the evidence that something is wrong upstream.

## Contributing

Branch from `v1.4.0-dev` — not `main`, which carries released code. Name the branch for the
work (`f7-header-harvest`), keep it short-lived, and open a pull request back into
`v1.4.0-dev`. Personal long-lived branches diverge and become unmergeable, especially when
large multi-file edits are involved.

Before opening a PR: the full fast suite passes, and you can say which kind of change it is —
additive-only, or capable of moving a measured number. That distinction decides which release
it can go in, and it is the first question that will be asked.

Commit messages here are prose that says what changed and why, with the measurement if there
was one. Look at `git log` for the register.

## The documents worth knowing

| | |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | the live plan — measurements, closed fixes, features F1–F13, release order |
| [`bench/ERROR_BUDGET.md`](bench/ERROR_BUDGET.md) | what limits the measurement, with numbers |
| [`LEON_2026-08-11.md`](LEON_2026-08-11.md) | the 2026 eclipse campaign, and why refraction now dominates |
| [`V1_4_0_TESTING.md`](V1_4_0_TESTING.md) | what the next release has to prove |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | how the pieces fit |
| [`../RELEASING.md`](../RELEASING.md) | building and publishing, including the catalogue releases |
| [`releases/`](releases/) | what each version actually contained |

The roadmap is long because it is the argument, not a summary. §6 is the part that says what
happens next.
