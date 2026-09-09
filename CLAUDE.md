# MEE — working notes for Claude Code

Astrometry pipeline for the Modern Eddington Experiment: stack frames, find centroids,
plate-solve, fit distortion, measure light deflection during a total eclipse.

Read [`docs/ONBOARDING.md`](docs/ONBOARDING.md) before the first substantial change. It has
the reasoning; this file has the rules.

## Commands

```
.venv/Scripts/python.exe -m pytest tests/ -q     # 931 pass, 26 skip with a catalogue
                                                 # installed; 930 pass, 27 skip without.
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

**The three-step reduction ladder is a rule, not a style.** `docs/V1_4_0_TESTING.md` §5 has the
table: the zenith reference is fitted free (`distortion_fixed_coefficients=None`), the **daytime
L/R calibration** field gets `quadratic`, and only the **eclipse field** gets `constant`. Bruns'
L/R8 and Leon's CAL_piLeo were both fitted `quadratic`; a station with no daytime calibration
(Station 1) fits its eclipse field `constant` + `distortion_free_scale` straight against the
zenith. On 2026-09-08 Station 2's bracket was given the eclipse field's settings by analogy with
Station 1 — one rung off — and it cost 0.27 ″ of deflection constant, because a frozen quadratic
cannot absorb a day-night change in the low-order distortion. Check `fixed distortion order` in
the run's own `distortion_results.txt` before believing any claim about which pathway was used.

**Never choose an analysis parameter at the keyboard.** The admitted-star window lives in
`tools/analysis_window.py` with the document that fixes it (cell 2: G ≤ 13, 2–10 R⊙,
`docs/MATRIX_2026.md`), and `tests/test_analysis_window.py` fails any new window constant that
cites no source. On 2026-09-08 Station 2 was fitted at 2–5.5 R⊙ — a bound in no document, nobody
asked for — which dropped three of seventeen stars per tier and moved L by 0.20 ″ and the error
bar by a third. The same week the daytime calibration was given the eclipse field's fixed-order
setting. Both were documented; both were missed by reasoning from analogy instead of reading. If a
number is not in a document or the registry, it is invented, and inventing one is worse than
stopping to ask.

**A chart joins a set; copy the set, do not recall it.** Every record chart in the project
draws RA **ascending to the right** (`s1_charts_record.py`, `b17_charts_record.py`), which is
not the sky convention. On 2026-09-08 two new charts were drawn reversed from habit, one of
them commented "as cell 2 draws it" — a citation written from memory, which reads exactly
like a checked one. `tests/test_chart_conventions.py` now fails a reversed RA axis. Open the
neighbouring tool before matching a convention, and never write a citation you have not read.

**A catalogue id is 19 digits: never let one near a float, and never guess when one will not parse.** `int(float(id))` changes 73 % of Gaia source ids, `np.uint64` against the int64 `source_id.npy` miscompares, and `df.iterrows()` on an all-numeric frame delivers the id as float64. None of these raise — they make a star match ITSELF, whose catalogue entry sits 0.6–1.0 ″ away once proper motion is applied, so a broken id becomes a plausible double star. On 2026-09-09 that produced three different confident wrong answers in one afternoon, and a "if the id is unreadable, exclude anything within 0.5 ″" fallback turned each parse failure into a finding. Compare like with like, refuse a float-shaped id, and check any such count against a second implementation. `tests/test_star_id_handling.py`.

**Changes are classified by whether they can alter a measured number.** That is the release
split (`docs/ROADMAP.md` §6): additive-only changes ride on the previous version's field
testing, anything that moves a fit needs its own validation on real data. Say which a change
is when proposing it.

**The stage-2 regression baselines in `tests/test_stage2_regression.py` are pinned
deliberately.** They exist to catch a change that quietly moves astrometry. Do not re-derive
them to make a test pass; if a change moves them, that is the finding — name the change that
did it. Same for the junk-field false-positive rate, which must stay at zero.

**Distortion coefficients are in MEE's own angular gauge, not the tangent plane.** Astrometrica,
ASTAP and every published table use TAN; the two differ by a universal radial term,
`k_TAN ≈ k_MEE + ~0.4 ″/deg³`. Compare without it and the programs look like they disagree by a
factor of three — this has now been "discovered" twice and written up as a bug once. Ratios are
gauge-dependent too, because the term is additive: two optics 1.5× apart in TAN gauge can look
10× apart in MEE gauge. And `rad/px³` scales with pixel size cubed, so convert to `″/deg³`
before comparing across cameras. `ROADMAP.md` §"The reference-projection gauge" and
[`docs/LEON_2026-08-11.md`](docs/LEON_2026-08-11.md) §18.11 have the numbers and the worked
validation.

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

`docs/ROADMAP.md` is the live plan: §1 measurements, §2 closed fixes, §3 features F1–F31, §6
the release order. v1.3.9 is released; `v1.4.0-dev` holds everything that changes results,
starting with F7 (header harvest) and refraction. **Read [`docs/V1_4_0_TESTING.md`](docs/V1_4_0_TESTING.md)
before working on that branch** — it is the work plan, and it records that the refraction
correction was broken outright until `v1.4.0-dev` fixed it.
