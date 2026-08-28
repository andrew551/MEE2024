# MEE — working notes for Claude Code

Astrometry pipeline for the Modern Eddington Experiment: stack frames, find centroids,
plate-solve, fit distortion, measure light deflection during a total eclipse.

Read [`docs/ONBOARDING.md`](docs/ONBOARDING.md) before the first substantial change. It has
the reasoning; this file has the rules.

## Commands

```
.venv/Scripts/python.exe -m pytest tests/ -q     # 869 pass, 26 skip with a catalogue
                                                 # installed; 868 pass, 27 skip without.
.venv/Scripts/python.exe -m pytest tests/ -q --runslow   # + triangle DB and network tests
.venv/Scripts/python.exe -m PyInstaller MEE2024.spec     # -> dist/MEE_v<version>.exe
```

```
.venv/Scripts/python.exe tools/smoke_exe.py dist/MEE_v<version>.exe --expect-version v<version>
```

Always the venv, never system Python — PyInstaller bundles whatever the interpreter can see,
and a system Python sweeps up unrelated packages.

**The executable is the product.** Almost every user runs the exe rather than the package, and
a green test suite is not evidence a release works: `pytest` exercises the source tree, while
PyInstaller decides at build time which modules exist. A missing hidden import fails in one
subcommand of the bundle and nowhere else. Run `tools/smoke_exe.py` on any exe before it goes
anywhere, and `tools/inspect_exe.py` to see what was bundled.

## Rules that are not obvious from the code

**Never rename `APP_NAME`.** It is `"MEE2024"` in `mee2024/MEE2024util.py` and it feeds
`get_data_root()` — the star catalogues, pattern databases and triangle database live under
it. Renaming strands every existing install and forces a multi-GB re-download. The *product*
is called MEE; `APP_NAME` is an opaque storage key users never see. Same for the `MEE2024`
FITS keyword, which other people's scripts may read.

**The version lives in two files** — `MEE2024util._version()` and `version` in `setup.cfg`.
Bump both. `tests/test_mee2024util.py` enforces it; v1.3.9 shipped with them disagreeing.

**Never reuse a version number.** If a build has been given to anyone, the next build gets a
new number even if only a filename changed. v1.3.6 was a real, field-tested binary that no
tag recorded, and the confusion cost days.

**Changes are classified by whether they can alter a measured number.** That is the release
split (`docs/ROADMAP.md` §6): additive-only changes ride on the previous version's field
testing, anything that moves a fit needs its own validation on real data. Say which a change
is when proposing it.

**The stage-2 regression baselines in `tests/test_stage2_regression.py` are pinned
deliberately.** They exist to catch a change that quietly moves astrometry. Do not re-derive
them to make a test pass; if a change moves them, that is the finding — name the change that
did it. Same for the junk-field false-positive rate, which must stay at zero.

**Close the app before touching catalogues.** A running instance memory-maps catalogue and
pattern-DB files, and Windows then refuses to install, remove or rebuild them.
`database_cache.release_catalogues()` and `pattern_db.release_databases()` exist for the
in-app path.

**`dist/` is gitignored.** Release notes are written to `docs/releases/v<version>.md`, which
`MEE2024.spec` copies beside the binary at build time. Three releases' notes once existed
only in `dist/`, one `git clean` from gone.

**Tests must never touch the real settings directory.** An autouse fixture in
`tests/conftest.py` redirects both config paths. Several tests used to write to
`AppData\Local\MEE2024\MEE2024\` for real.

**An interface should only apply settings it can show.** The classic UI displays forty-odd
options and the app window eight, so they keep separate files: `MEE_config.txt` and
`MEE_app_config.txt`. Do not make the app window read or write the shared one.

## Shape of the code

| | |
|---|---|
| `mee2024/stacker_implementation.py` | stage 1 — stacking, centroids, alignment |
| `mee2024/distortion_fitter.py` | stage 2 — catalogue match and distortion fit |
| `mee2024/eclipse_analysis.py` | stage 3 — the deflection constant |
| `mee2024/platesolve2/`, `platesolve_triangle.py` | v2 solver (default) and the classic one |
| `mee2024/calibration.py` | master darks and flats, the calibration library |
| `mee2024/starcat/` | catalogue download, storage, star labels |
| `mee2024/ui/` | the app window (server + frontend); `UI_handler.py` is the classic UI |
| `mee2024/cli.py` | every command-line entry point |

Three interfaces reach the same pipeline: the classic UI, the app window, and the CLI. A fix
that lives in one front end's options assembly is in the wrong place — that was bug I9.

## Conventions

Comments explain *why*, especially where the obvious approach was tried and failed; several
carry the measurement that settled it. Match that. Docstrings on non-trivial functions say
what the previous behaviour was when it was wrong, because that is what stops it coming back.

Prose in docs and commit messages is plain and specific — no marketing register, and numbers
rather than adjectives.

## Where the project is

`docs/ROADMAP.md` is the live plan: §1 measurements, §2 closed fixes, §3 features F1–F13, §6
the release order. v1.3.9 is released; `v1.4.0-dev` holds everything that changes results,
starting with F7 (header harvest) and refraction. See `docs/V1_4_0_TESTING.md`.
