# MEE2024 — development progress log

A running record of what has changed, what has been measured, and what was decided.
Newest first. Design detail lives in `docs/ARCHITECTURE.md` (how the pipeline works) and
`docs/STARCAT_DESIGN.md` (the star-catalogue redesign).

---

## Current state

| | |
|---|---|
| Branch | `refactor/test-cli-foundation` |
| Tests | 500 passing (`pytest` = 479 fast + 21 behind `--runslow`) |
| Lint | pyflakes clean apart from three intentional import probes/shims |
| Pipeline | stages 1–3 headless from the CLI, and from the new app window |
| Catalogues | Gaia G<12 and 12<G<13 published as GitHub release assets, fetched on first use; Hipparcos + labels bundled |
| Interfaces | app window by default, `mee2024 gui` (classic, unchanged), CLI |
| Version | v1.0.1; Windows exe built from `MEE2024.spec` |

Design docs: `docs/CATALOGUE_INVENTORY.md` (catalogue unification),
`docs/PLATESOLVER_DESIGN.md` (solver measurements, statistics, improvement plan),
`docs/PLATESOLVER_V2_DESIGN.md` (the solver rebuild: theory and stage plan),
`docs/UI_DESIGN.md` (UI strategy, what is built, and the P2 question).

### Measured baseline — do not regress these

Both example fields are ZWO zenith star fields, 5 frames, 3 s exposures, taken
**2023-10-29** (from the FITS `DATE-OBS`).

Stage 1:

| field | frames | pixels | centroids | platescale fitted | predicted from optics |
|---|---|---|---|---|---|
| zwo1 | 5 | 9576×6388 | 3966 | 1.8511″/px | 1.8466″/px (0.24%) |
| zwo3 | 5 | 4656×3520 | 1328 | 1.8716″/px | 1.8662″/px (0.29%) |

Stage 2 (`guess_date` seeded with 2020-01-01, blind):

| field | order | RMS | stars | guessed date | error | nn_corr |
|---|---|---|---|---|---|---|
| zwo3 | cubic | 113.0 mas | 433 | 2023-11-02 | +4 d | 0.386 |
| zwo3 | quintic | 109.6 mas | 434 | 2023-10-28 | −1 d | 0.166 |
| zwo3 | septic | 107.2 mas | 434 | 2023-10-27 | −2 d | 0.132 |
| zwo1 | cubic | 525.2 mas | 1396 | 2023-03-19 | −224 d | 0.921 |
| zwo1 | quintic | 111.9 mas | 1564 | 2023-09-06 | −54 d | 0.353 |
| zwo1 | septic | 100.9 mas | 1564 | 2023-09-20 | −39 d | 0.191 |

All of the above is asserted by `tests/test_stage2_regression.py`, offline.

---

## 2026-07-31 — S1: the Gaia pattern database, and the port that reads it

**`mee2024/platesolve2/` is real.** Same algorithm as v1 — same invariant, tolerance,
consensus, estimator — reading a new database built from the offline Gaia catalogue,
with verification switched to that same catalogue (leaving it on Tycho would have kept
2.5″ errors inside the acceptance statistics). `patdb_g12_t17`: 112,660 anchors /
17.24 M triangles / 213 MB, built by `mee2024 build-pattern-db` in ~3 minutes — the
1.4 M scalar KD queries v1's builder spends minutes on become two batched
multi-threaded calls. Format: directory-of-npy + manifest (per-file SHA-256, pinned
dtypes, prefix-sum triangle offsets so sparse-sky anchors simply own fewer rows —
v1's "edge case handling unimplemented" is gone). The solver reads the pattern width
from the manifest, ending the silent g=18 builder/solver coupling. A missing database
raises with the exact build command; the orientation fit gains the Kabsch determinant
correction v1 omits.

**A/B against the S0 baseline (docs/bench/BENCH.md): gate passed.** Wrong solves 0,
junk 8/8, real fields 2/2 with v1-parity to 0.33″ / 0.005° roll / 1.3×10⁻⁴ scale.
Overall 61 → 62/80; `fov_galplane_2°` and the 8 px-noise case flipped to solving;
median times down 25–45 % per family. The headline number is verification margin:
**81–88 stars matched where v1 managed 27–50** on identical fields — that margin is
what S3 spends when it tightens the shape tolerance.

**One knife-edge regression, root-caused and deliberately accepted.** With 10
detections both solvers find the same true pointing, but v2 verifies 8/10 against
threshold 9 where v1 scraped 9/10 — the ~3× denser Gaia comparison set lets a faint
neighbour disqualify one marginal match via the 2× confusion ratio. Fix assigned to
S3: verification depth should track detection count (faint catalogue stars can only
disqualify, never help, when only bright stars were detected). Two ordering-scatter
draws also churned (one fixed, one broke — G-band vs V-band ranking); that axis
belongs to S2b/S5a.

12 new fast tests give v2 what v1 never had: a build → solve → verify → contract
end-to-end test in CI, on a miniature in-test database. The slow synthetic/junk tests
now parametrise over both solvers.

---

## 2026-07-31 — solver v2 designed; stage S0 landed (bench harness + v1 baseline)

**The rebuild is designed and gated.** `docs/PLATESOLVER_V2_DESIGN.md` carries the
theory and the stage plan: Gaia pattern DB (S1) → Kendall shape invariant with a
conjugate-query mirror pass (S2) → adaptive tolerance (S3) → quaternion consensus (S4)
→ progressive anchors + mmap/bucket index (S5) → multi-scale layers + platescale hints
(S6) → a quad layer only if the S6 bench still shows a blind >8° gap (S7). Each stage
isolates one variable and must hold: zero wrong solves, junk all rejected, both real
fields solving.

Decisions from the analysis, in brief: the `dev-platesolve` spike's Kendall shape
coordinates are verified correct and adopted (rewritten — its tip does not import);
**quaternions win nothing for the final orientation fit** (Wahba/SVD is equivalent;
microseconds either way) but are the right *consensus clustering* metric — v1 clusters
roll unwrapped and its (centre, roll) chart is singular at the poles; the spike's
double-cover handling is measurably broken (missing `abs()`, un-negated twins) and gets
the correct canonicalise-plus-boundary-twins treatment; its p-value acceptance test is
post-hoc (feeds matched radii into its own null) and is set aside in favour of the
production estimator with the local-density fix; its "dimmer legs" DB rule becomes a
benchable variant (S2b), not an assumption, judged on the ordering-scatter sweep.

**S0 is landed:**

- `tools/solver_bench.py` — 88 frozen cases (FOV sweep both density regimes,
  reliability draws, noise/scatter sweeps, sparse fields, poles, roll-wrap, junk, the
  two real fields), deterministic per-case seeds, `run`/`compare`/`list`, results
  committed under `docs/bench/`.
- **v1 baseline** (`docs/bench/s0_baseline.json`, summarised in `docs/bench/BENCH.md`):
  correct 61/80 solvable, wrong solves **0**, junk **8/8** rejected, real fields 2/2 at
  7.2 s. Reproduces the measured envelope — and adds one new fact: **pole fields fail
  0/4** with 100–150 stars available, confirming the predicted roll ill-conditioning in
  the consensus chart rather than any lack of stars. That is the S4 target, now pinned
  by a number rather than an argument.
- `options['platesolver']` ('triangle' default | 'v2') with a four-line dispatch facade
  in `platesolve_triangle.platesolve`; `mee2024/platesolve2/` exists as a stub that
  names the design doc. No pipeline call site changed.
- Slow solver tests now **skip with instructions** when the triangle DB is absent
  (`skip_unless_triangle_db` in `tests/fixture_catalogue.py`) instead of triggering
  `database_cache`'s silent multi-minute inline rebuild mid-run.

Found while testing, not caused by this work: `mee2024/resources/star_labels/names.txt`
is CRLF-corrupted on this checkout (git line-ending conversion shifted the byte offsets
`hip_name_offset.npy` indexes into), so 6 label tests fail with byte-shifted names.
Needs a `.gitattributes -text` rule plus a renormalised file; tracked separately.

---

## 2026-07-30 — v1.0.1: catalogues fetch themselves, and two diagnostic views

### Catalogues arrive without being asked for

Selecting an offline catalogue used to fail at stage 2 if its archive was absent — minutes
after pressing Run, with the stacking already done. `download.prepare_catalogue()` now runs
*before stage 1* and fetches what is missing, from both the app and the CLI. Set
`auto_download_catalogue: false` to refuse instead (a 138 MB download on a metered
connection is a reasonable thing to decline); the refusal names `--fetch`.

Download progress was reported in raw bytes, so a 138 MB transfer read as
`45088768 / 137952319`. Progress events now carry `unit='bytes'` and the frontend renders
`45 MB of 138 MB · 33%`.

Two things found while doing this:

- **`_default_catalogue()` capped the depth it had.** It returned the first *installed
  archive name*, so with both archives present a run used `gaia_dr3_g12` — G<12 — while
  the 12<G<13 extension sat on disk unused. It now returns `gaia_offline`, which reads
  every archive present.
- **`size_bytes` for the deep archive was wrong** (188,985,889 against an actual
  188,640,212), so `--check-remote` failed against a correctly uploaded file. The sha256
  was right, and downloads verify against that and the server's own Content-Length, so
  fetching was never broken — only the preflight.

### Asking too deep is now said out loud

A magnitude limit past the catalogue's depth was silently truncated, which reads as a poor
field rather than a catalogue that does not go that far. It now warns at the lookup itself
(so no caller can bypass it), before the run as a preflight, and live in the UI as you type
— and it names the remedy. The two archives are disjoint magnitude slices, so the UI says
where G<13 actually comes from: both installed, badged **recommended**, with the extension
labelled as carrying no bright stars on its own.

### Advanced analysis: surfaces and a residual-correlation map

Hidden behind a toggle, since the flat field map answers the usual question:

- **Rotatable displacement surfaces** with every measured star drawn on them, replacing the
  three matplotlib 3-D windows the old code opened. A star sits off the surface by exactly
  its residual, so too low an order shows as coherent undulation rather than even
  peppering. On a good fit residuals are ~100× smaller than the distortion, so ×5–×100
  exaggeration is offered and labelled.
- **Residual-correlation map**: the detector in cells, each averaging how far residuals
  inside it agree with their nearest neighbour's — the per-star quantity the single
  `nn_corr` score averages. A warm patch localises an optical imperfection instead of
  smearing it into one number.

Both are drawn on a canvas from one `analysis` event (columnar, ~30 kB) rather than sent as
images, so rotating and re-binning need no re-run.

**Bin count is derived, not guessed.** A star's nn-correlation is a cosine spanning −1..1,
so a cell holding two or three of them is noise that reads as structure. `bins ≈
√(N_stars/8)`, bounded to 4–24: a 430-star field gets 7 bins at 8.8 stars per cell, and the
legend turns amber below 4 per cell. Verified on a synthetic field with a coherent patch
injected into one corner — the map reports **0.737** inside it against **0.024** elsewhere.

Also fixed: below 820 px the controls column left the results pane a few pixels wide,
collapsing every plot to nothing. The layout now stacks.

---

## 2026-07-30 — new UI: event bus (P0) and app shell (P1)

**P0 — `mee2024/events.py`.** Typed pipeline events on an ambient `ContextVar` bus, so
emitting needs no argument threaded through five layers. Sinks: `ListSink` (the UI and
tests), `JsonlSink`, `TextSink`, `CallbackSink`. A broken sink cannot take a run down.
`ProgressReporter.loop` now emits stage/progress events, so **every** reporter — and any
frontend watching — stays in step with no subclass doing extra work. Emissions added for
frame alignment, centroid counts, solve candidates and results, stage metrics, and a
stacked-image preview. New CLI flags `--events-jsonl PATH` and `--events-text` make any run
machine-readable, which milestones B and E want regardless of the UI.

**P1 — `mee2024/ui/`.** `runner.py` runs the pipeline in a worker thread with cooperative
cancellation and three presets (auto / quick look / deep); `server.py` serves a
token-guarded API on an ephemeral localhost port; `frontend.html` is one self-contained
file (no CDN, works offline, dark themed); `app.py` opens a native window via pywebview
with a browser fallback. One transport serves both, so there is a single frontend.
`mee2024 ui` launches it; `mee2024 gui` still launches the classic interface untouched.

**Verified against real data.** A stage-2 run driven through the UI reproduced the CLI
numbers exactly — 109.6 mas, 434 stars, `nn_corr` 0.166, date recovered 2023-10-28 — and
rendered six graded score cards with plain-language captions. 57 new tests, none of which
open a window.

**Three bugs the live test found that unit tests had not:**

- **`do_stack` used `os.mkdir`**, which cannot create a missing parent — choosing a
  not-yet-existing output folder failed several layers deep with a bare `WinError 3`. This
  hit the CLI's `-o` equally. Now `makedirs`, and the runner validates the folder up front.
- **The frontend treated any payload `error` key as a transport failure**, while the state
  payload used `error` for the run's own error — so polling broke at exactly the moment a
  run failed. Renamed to `run_error`; `api()` now keys off the HTTP status alone.
- **My own `png_event` converted a whole frame to float64 before downsampling** (125 MiB
  for a 3520×4656 stack, more again for RGBA). It now strides *then* casts, emits 8-bit
  greyscale, and carries a stdlib PNG encoder so IMAGE events work without Pillow.

**Architectural finding for P2.** Running the pipeline in a thread inside a long-lived
server means the cached triangle database (~1 GB resident with its KD-tree) coexists with
the pipeline's transient full-frame arrays. On a memory-pressured machine that is enough to
fail — a stage-1 run died needing another 125 MiB in `get_centroids_blur`, and later 89 MiB
for 1.3 M triangle candidates. The CLI escapes it because each run is a fresh process.
**Recommendation: move the run into a subprocess**, forwarding events over a pipe; the bus
already serialises to JSON, so the frontend needs no change, and cancellation becomes
immediate instead of cooperative. Cheaper complementary win: `get_centroids_blur` allocates
full-frame float64 arrays from uint16 input — float32 would halve peak memory there, worth
measuring against the milestone-E benchmark rather than assuming.

---

## 2026-07-30 — plate solver measured end to end; estimator hole found and fixed

**New instrument: `tools/synthetic_field.py`** — ground-truth centroid lists synthesized
from the offline Gaia catalogue (gnomonic projection → cubic distortion → detection
incompleteness → noise). This is what makes the solver measurable; full results and the
statistical theory are in `docs/PLATESOLVER_DESIGN.md`.

**Measured envelope** (113,121 anchors × 153 = 17.3 M triangle DB): solves 2–8° reliably
at mid-latitudes and in the galactic plane; fails at 1° (DB star-list depth) and at 10°
(the brightest-18-per-pattern-disc window plus projective breakdown of similarity
invariants). Solves down to 10 detected stars, up to ~4 px centroid noise. 5–9 s per
solve after a 10.4 s DB load. Junk fields: 3/3 rejected.

**The reliability problem in numbers**: over 8 random detection-ordering draws per
pointing — 8/8 at mid-lat, but **6/8 at a sparse high-latitude pointing and 4/8 at 8°
FOV**. The failure mode is which bright stars survive detection ordering, i.e. the strict
brightest-`f` anchor prefix. This is the cheapest big win (sample anchors from ~2f).

**False-positive estimator audited against the exact max-of-N-Poisson quantile.** The
Lambert-W approximation is excellent (within 1–2 everywhere, covered by the +3 addon) —
but it assumed all-sky mean density, and the galactic plane runs 3–10× that, making the
threshold **unsafe by 12–23 matches in dense fields**. Fixed: the threshold now uses the
local density from the bbox star count that `match_centroids` already had in hand.
Verified against exact quantiles by new tests; the real-field corpus still passes.

**Theory, checked against measurement**: the ≥4-triangle consensus rule is exactly the
information budget (45 bits of solution space / 14.3 bits per triangle at tolerance 0.01);
predicted 865 chance matches per query triangle vs ~900–1000 observed. Current tolerance
is ~50× coarser than centroid noise supports — because the *Tycho DB* is the noisy party —
so rebuilding the pattern DB from the offline Gaia catalogue is the gain that unlocks the
rest (finer tolerance → fewer candidates → faster and more reliable). The Kendall
shape-space idea (from the other branch) is endorsed with the metric argument; quads are
the right tool only beyond ~8–10° where projective invariants are needed.

**UI strategy written** (`docs/UI_DESIGN.md`): the pipeline is already headless, so the
one missing piece is a typed event bus extending `ProgressReporter` (P0, small, useful to
milestones B/E regardless). Recommended shell: pywebview + a self-contained local HTML
frontend (unlimited visual ceiling, tiny packaging, Win/macOS first-class, `--browser`
fallback). Three modes: Simple (drop files + Auto + score cards), Advanced (today's
options), Live (watch folder; per-frame solve at the measured 5–9 s is fast enough for
instant pointing/quality feedback during acquisition). Legacy FreeSimpleGUI kept.

---

## 2026-07-30 — Hipparcos closes the bright-star gap; deep extension built

**Deep extension built:** `gaia_dr3_g12_13`, 4,281,806 stars (12<G<13), 10 min, 213 MB.
Stacked with the base archive that is **7,369,627 stars to G<13**, matching the archive
count exactly. The eclipse-field case is now decisive — a 10°×10° field at G<13 returns an
identical 45,573 stars in **0.03 s offline against 6.9 s online (206×)**.

**Hipparcos-2 added as the bright fill and the label source.** One 6 MB download serves
both, and both are small enough to bundle in the wheel:

- `mee2024/resources/hipparcos2/` — 117,955 stars, 5.9 MB, epoch 1991.25 with proper motion
- `mee2024/resources/star_labels/` — 117,955 HIP entries, 99,525 Gaia crossmatches, 50
  proper names, 2.1 MB

All four stars Gaia lacks now resolve correctly and are flagged precision-grade:

| star | V | via | label |
|---|---|---|---|
| Sirius | −1.46 | hipparcos | `Sirius` |
| Vega | +0.03 | hipparcos | `Vega` |
| Arcturus | −0.05 | hipparcos | `Arcturus` |
| Canopus | −0.74 | hipparcos | `Canopus` |

**The Hp→G transformation was fitted, not cited.** A quadratic in B−V trained on the 96,767
stars with both Hp and a measured Gaia G: **robust σ = 0.038 mag** (plain rms 0.269, the
tail being variables and binaries Gaia resolved but Hipparcos did not). A cubic overfits
the blue end. `band='G_est_from_Hp'` records the provenance so nothing treats it as
photometry.

**Measured: Tycho is nearly redundant as a bright fill.** Over 14 random 5°×5° boxes
(350 sq deg, 19,129 Gaia stars) — Hipparcos adds **5** stars, Tycho adds **1 more** after
it, Tycho alone would add 5. They overlap almost entirely. Tycho stays for plate solving
(the triangle database needs 700k stars, which only Tycho provides) but its fill role is
marginal.

**The merge is verified safe:** on both example fields it adds *zero* stars and reproduces
the pure-Gaia result exactly (109.6 mas / 434 / nn_corr 0.172). It only acts where Gaia is
genuinely incomplete — in a field containing Sirius it adds exactly one star.

**Guard added:** `StarTable.is_precision_grade()` plus enforcement in `distortion_fitter`,
so a Tycho star can reach the plate solver but never the distortion fit. This is what the
per-star `origin` column was for.

Labels are a **sorted-key index with `np.searchsorted`**, not a hash table — smaller,
memory-mappable, and one call resolves a whole field.

---

## 2026-07-30 — all-sky G<12 built; two of my earlier claims corrected

**`gaia_dr3_g12` is built and verified: 3,087,821 stars, 33 min, 160 MB on disk.** Stage 2
through it reproduces the online run exactly on both fields (109.6 mas / 434 / 2023-10-28
and 111.9 mas / 1564 / 2023-09-06).

### Correction 1 — the speed benefit is field-dependent, not general

I implied offline would simply be faster. On the small zwo3 field it is **not**: 21 s
online vs 23 s offline, because the query is already quick there and the plate solve
dominates. The benefit appears with density and depth — a 10°×10° dense field near the
galactic plane, identical 18,886 stars both ways:

| | stars | time |
|---|---|---|
| online | 18,886 | 10.13 s |
| offline | 18,886 | **0.02 s** |

**632×.** That is exactly the mag-13 eclipse-field case. For a small calibration field,
offline buys reproducibility and no network dependency, not speed.

### Correction 2 — the bright-star gap is worse than I said, and Tycho does not close it

I claimed part of "Gaia misses the brightest stars" was our own `BETWEEN 3` query floor.
That floor was real but **minor**. The dominant cause is that **Gaia DR3 has no entry at
all** for the brightest stars — they saturate the instrument. Querying with no magnitude
filter whatsoever:

| star | V | what Gaia DR3 holds at that position |
|---|---|---|
| Sirius | −1.46 | 20 sources, brightest G = **8.52** (unrelated field stars) |
| Vega | +0.03 | 13 sources, brightest G = **14.58** |
| Arcturus | −0.05 | 2 sources, brightest G = **15.12** |
| Canopus | −0.74 | 1 source, G = **19.13** |

Zero NULL photometry — the stars are simply absent.

And a second gap I had not anticipated: **the bundled Tycho catalogue is also missing the
very brightest stars.** It has Arcturus (V=+0.16) and Canopus (V=−0.63) but not Sirius or
Vega, and only 8 entries brighter than V=1 where the sky has ~15:

| | V<1 | V<2 | V<3 | V<4 | V<5 | V<6 |
|---|---|---|---|---|---|---|
| bundled Tycho | 8 | 40 | 151 | 479 | 1535 | 4813 |
| real sky (approx) | 15 | 50 | 170 | 520 | 1600 | 4800 |

Tycho-2 shunts ~120 very bright stars into *Supplement 1*, a separate file from the
`tyc_main.dat` this npz was built from. So the merge fixes Arcturus and Canopus but not
Sirius or Vega.

**Implication for the design:** use **Hipparcos** as the bright fill, not Tycho-2.
Hipparcos is complete to V≈7.3, contains every naked-eye star, and has better bright-star
astrometry than Tycho-2. We already needed a HIP crossmatch for the label layer, so one
small artefact solves both problems. Tycho then fills only V≈7–9 where HIP thins out.

Practical impact of the gap is probably modest — such stars are saturated in real frames
and largely removed by the blob masking — but it would bite on a plate solve of a field
containing one.

---

## 2026-07-30 — offline catalogue builder, and it reproduces the online results exactly

**The offline path is now testable end to end.** `tools/build_gaia_offline.py` builds a
catalogue; `mee2024 --catalogue <name> distortion ...` uses it with no network.

Stage 2 through a real offline catalogue directory, against the online numbers:

| field | order | online | offline |
|---|---|---|---|
| zwo3 | quintic | 109.6 mas, 434 stars, 2023-10-28, nn_corr 0.166 | **identical** |
| zwo1 | quintic | 111.9 mas, 1564 stars, 2023-09-06, nn_corr 0.353 | **identical** |

Not "within tolerance" — the same numbers.

**Gaia query latency dominates the build: ~18–28 s per query regardless of size.** A
mag-5 all-sky build (2168 stars, 18 queries) took 318 s, essentially all latency. So the
declination step defaults to **10°, not 1°** — a 1° step would mean 180 stripes and turn a
15-minute build into hours for nothing. Row counting is one `GROUP BY` query rather than
one per stripe, halving the query count. Builds are resumable: each chunk is cached as an
`.npy` keyed on its stripe band.

Real sizing for all-sky G<12, measured rather than guessed: **3,087,821 stars, 29 queries,
10–22 min, 148 MB**. The design doc's original "hours" estimate was wrong.

Build sequence that worked, each step validating more than the last:

1. `--max-mag 5` all sky — 2168 stars. Proves query/chunk/assemble/flag/write/verify.
2. `--max-mag 12 --region ...` around the known fields — 4585 stars in one 28 s query.
   This is what made the offline stage-2 test above possible.
3. all sky `--max-mag 12` — the real artefact.

Wiring: `database_cache.open_catalogue` now resolves starcat provider names and locally
built catalogue directories, so a catalogue built by the tool is usable by name
immediately. `distortion_fitter` routes double-star lookup through the provider rather
than calling `gaia_search.lookup_nearby` directly.

**One real behavioural difference offline:** `nn_sep`/`nn_mag` are computed among the
catalogue's own members, so a companion fainter than the catalogue limit is not flagged,
whereas the online path queried to G<17. It changes nothing here because
`remove_double_tab2` defaults to false and the flag is only recorded, but it is a genuine
reduction in sensitivity and the manifest records the depth covered. Photometrically, a
companion at Δm shifts a centroid by ~`10^(-0.4Δm)` of the separation, so G=14 beside G=10
at 10″ matters (~250 mas) and G=17 does not (~16 mas) — neighbours to about `limit + 4`
capture what counts, and the flat G<17 was more than needed.

---

## 2026-07-30 — star catalogue: step 2, the `starcat` package

Added `mee2024/starcat/`. Nothing is wired into the pipeline yet, so the measured baseline
is untouched — this is the new machinery sitting alongside the old.

- **`table.py` — `StarTable`.** One columnar type replacing both the raw `(N, 6)` array
  and `StarData`. float64 positions, NaN-for-unknown **per star**, per-star `origin`, and
  operations that return new tables. Two bugs found while writing its tests: `select()`
  with a slice returned numpy *views*, so `__copy__` aliased its parent (fixed with a
  `may_share_memory` check); and the epoch round trip is accurate to **0.023 mas**, four
  orders of magnitude below what we achieve.
- **`providers.py`** — `CatalogueProvider` plus `TychoProvider`, `GaiaOnlineProvider`,
  `GaiaOfflineProvider`, `MergedProvider`, and a name registry (`gaia`, `gaia_offline`,
  `tycho`, `merged`, `merged_offline`). The Gaia query no longer imposes a `G>3` floor.
  The merge keeps a Tycho star only where Gaia has nothing within 2″ *and* it is brighter
  than V=9, tagging it `ORIGIN_TYCHO` so the precision fit can exclude it.
- **`store.py`** — the offline format: a directory of uncompressed `.npy` files,
  memory-mapped, declination-sorted, with a one-degree band index, a manifest carrying
  SHA-256 per column, and pack/unpack/verify. A two-degree query touches under 10% of the
  rows, so lookup cost does not grow with catalogue size.
- **`download.py`** — release registry, download with progress, checksum verification,
  extraction. **Zenodo entries are placeholders**: URL and hash are `None`, so
  `--fetch` explains how to build locally instead of failing obscurely. Filling in a real
  deposition is a one-line edit per catalogue. Depth is split into a `gaia_dr3_g12` base
  (~139 MB) and an optional `gaia_dr3_g12_13` extension (~215 MB), which a provider can
  stack — this is the answer to wanting mag-13 stars for eclipse fields without forcing a
  354 MB download on everyone.
- **`mee2024 catalogue [--list|--verify NAME|--fetch NAME]`** as the user-facing surface.
- **The propagation gate is now a permanent offline test.** The fixtures carry Gaia's own
  `ESDC_EPOCH_PROP_POS` positions as ground truth, so
  `test_offline_propagation_reproduces_gaias_own_answer` asserts agreement below 0.1 mas
  with no network. It also caught that `StarTable` is *more* accurate than `StarData` was:
  passing radial velocity to `apply_space_motion` is worth 2.45 mas on one zwo3 star.

82 new tests. Next: step 3, migrate `platesolve_triangle` off raw arrays.

---

## 2026-07-30 — star catalogue: step 1 (`ff51002`)

**Verified stage 2 works** on both fields, including the guess-date feature. See the table
above.

**Epoch-propagation gate passed.** The main risk in going offline was that Gaia propagates
positions server-side with `ESDC_EPOCH_PROP_POS`, and offline we must use astropy
`apply_space_motion` locally. Measured over the 2016.0 → 2023.84 baseline: **0.000 mas
median and maximum** across all 2394 stars in both fields. Parallax clamping is
irrelevant; dropping radial velocity costs ≤2.5 mas. Offline loses no accuracy.

**Added an offline, deterministic stage-2 regression test.** The Gaia response is replayed
from a 116 KB fixture, so the whole table above is verified with no network. The replaying
provider is deliberately shaped like the future `GaiaOffline` provider, so one seam serves
both.

**Findings**

- **RMS does not predict date accuracy.** zwo1/quintic and zwo3/quintic have near-identical
  RMS (112 vs 110 mas) but date errors of 54 days vs 1 day. What tracks it is the
  **nearest-neighbour error correlation**, already computed: 0.13–0.17 → days, 0.35 →
  weeks, 0.92 → months. Use `nn_corr`, not RMS, as the headline quality signal.
- **How good is the blind date fit, really?** Least squares on `Δθᵢ = μᵢ(t − t_ref)` gives
  `σ_t = σ_resid / √Σ|μᵢ|²`. Predicted vs actual: zwo3 **16.3 d predicted, 1 d actual
  (0.06σ — a lucky draw)**; zwo1 **24.5 d predicted, 54 d actual (2.2σ — worse than
  statistical, because the distortion polynomial absorbs part of the proper-motion
  pattern)**. So the honest capability is **2–4 weeks**, not one day. Precision is also
  dominated by a few fast movers rather than the whole field: for zwo3 the RMS proper
  motion is 83 mas/yr but the *median* is 9.2 mas/yr.
- **Bundled Tycho positions degrade steeply with magnitude** — 0.135″ at V<8 rising to
  2.46″ at V=11–11.5, with no systematic offset. Consistent with Tycho-2 proper-motion
  error amplified over the 32.75-year propagation from 1991.25. Fine for plate solving
  (36″ tolerance), unusable for a 0.35″ deflection measurement.
- **`select_in_box` hard-codes `phot_g_mean_mag BETWEEN 3 AND max`**, silently dropping the
  150 Gaia sources with G<3 *and* all 5.46 M with NULL photometry. Part of "Gaia misses the
  brightest stars" is our own query.
- **Gaia source ids must never touch a float.** 19 digits do not fit a 53-bit mantissa.
  An early version of the capture tool compared the wrong stars because of this and
  reported a spurious 2266 mas propagation outlier.

**Decisions taken**

| | |
|---|---|
| Offline catalogue depth | **G < 12** (~2.9 M stars, ~139 MB) |
| Hosting | **Zenodo** (citable DOI); placeholder URL until an artefact exists |
| Tycho bright fill | **V < 9**, plate solving only, never the precision fit |
| Unified type name | **`StarTable`**, retiring `StarData` |
| Label layer | Full HIP crossmatch + IAU names, ~1.9 MB, bundled in the wheel |

---

## 2026-07-29 — test and CLI foundation (`4bacfc4`)

Groundwork so that pipeline changes can be measured rather than eyeballed.

- **Decoupled the pipeline from the GUI.** `mee2024/config.py` holds the option defaults
  (previously trapped in `main.py`, so importing them pulled in FreeSimpleGUI and spawned
  a subprocess). `mee2024/progress.py` provides Null/Text/Gui progress reporters. No
  pipeline module imports FreeSimpleGUI any more.
- **Added `mee2024/cli.py`**: `stack`, `distortion`, `eclipse`, `run`, `config`,
  `build-triangle-db`, with `--no-display` and `--set key=value`. No arguments still opens
  the GUI.
- **Fixed 9 bugs**, each with a regression test. The one that changes results:
  `rough_match_threshhold` was divided by **33600 instead of 3600**, making the match
  tolerance 9.33× tighter than the GUI claimed — configs tuned against the old behaviour
  (e.g. `rough_match_threshhold=200`) should be revisited. Also: `np.cos` of a declination
  in degrees in the double-star query; `StarData.__copy__` assigning to `newone.epch`;
  an off-by-one frame index in the residual plot; the eclipse report labelling Method 2 as
  Method 1; two unguarded `plt.show()` calls plus a `NameError` that made headless stage 3
  impossible; tab 3 validating tab 2's output directory; `icoord_to_vector` mutating its
  caller's array.
- **Removed 969 lines of dead code** against 366 added — a 205-line commented block in
  `transforms.py`, four `__main__` blocks with hardcoded `D:/` paths, an unreachable
  Legendre branch, an unused multiprocessing loop, ~40 unused imports, and five options
  nothing read. `tests/test_config_coverage.py` now cross-checks options against
  `DEFAULT_OPTIONS` in both directions so that class of removal can't break the GUI again.
- **Wrote `docs/ARCHITECTURE.md`.**

**Open finding, needs a decision:** the two centroid finders use pixel conventions that
differ by **exactly half a pixel in each axis** — 1.31″ diagonally at 1.85″/px, larger than
the signal. `simple_get_centroids` treats an integer index as a pixel corner; the sensitive
path treats it as the centre. A constant offset is absorbed by the plate solve so it does
not bias *L*, but the two modes are not interchangeable and non-sensitive-mode stage-1
RA/Dec is off by ~0.9″ per axis. `do_stack` already compensates at *display* time only.
Pinned by `test_the_two_centroid_finders_disagree_by_half_a_pixel`; unifying it would shift
every historical result, so it is left as a deliberate choice.

---

## Next

Milestones A and C are done; the UI is through P1 plus the analysis views above.

1. **Milestone D — plate-solve robustness.** Now a staged rebuild with an A/B gate:
   S0 (bench + baseline) and S1 (Gaia pattern DB + verification switch) are landed;
   next is S2, the Kendall shape invariant with the conjugate-query mirror pass.
   Designed in `docs/PLATESOLVER_V2_DESIGN.md` (which supersedes the improvement list
   in `docs/PLATESOLVER_DESIGN.md` §5). S3 note from the S1 bench: verification depth
   should adapt to the detection count.
2. **Milestone B — auto-calibration and a quality score.** The score cards and the nn_corr
   grading exist; `mee2024/quality.py` and `mee2024 autocal` do not.
3. **Milestone E — centroid backend rig.** Not started. The half-pixel convention note
   above is the first thing it should pin down.
4. **Zenodo** for a citable DOI, replacing the GitHub release URLs — two lines per
   catalogue in `RELEASES`, and the checksums do not change.
5. **Mac/Linux builds** — deferred deliberately.

### Unresolved questions

- `safety_limit_mag`: mag-13 stars are detectable in eclipse fields and scientifically
  useful, but Gaia is slow to serve them and the offline archive doubles in size to cover
  them. Proposal: keep the default at 13, build the archive in two magnitude-banded parts
  (G<12 base, 12<G<13 optional), and have the offline provider declare its own depth and
  fall back to online beyond it.
- Whether to unify the half-pixel centroid convention (above).
- What `mee2024/TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_dataB.npz` in the
  working tree is — possibly an earlier wide-field experiment, relevant to milestone D.
