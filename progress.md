# MEE2024 — development progress log

A running record of what has changed, what has been measured, and what was decided.
Newest first. Design detail lives in `docs/ARCHITECTURE.md` (how the pipeline works) and
`docs/STARCAT_DESIGN.md` (the star-catalogue redesign).

---

## Current state

| | |
|---|---|
| Branch | `refactor/test-cli-foundation` |
| Tests | 671 fast, 26 more behind `--runslow`, all passing in a clean `.venv` (`python -m venv .venv` + `requirements.txt`) |
| Pipeline | stages 1–3 headless from the CLI, and from the new app window |
| Plate solver | **v2 by default** (Gaia + Kendall + quaternion consensus + FOV layers; `docs/bench/BENCH.md`); falls back to the classic Tycho solver when no pattern DB is installed; `platesolver='triangle'` selects it deliberately |
| Pattern DBs | `patdb_g12_t17k` primary (230 MB) + optional `patdb_g13_t06k` (334 MB) / `patdb_g12_t40k` (60 MB) layers, built locally with `mee2024 build-pattern-db`; `LAYER_SET` picks the newest installed per scale; not yet published as release assets |
| Catalogues | **`gaia_dr3_g13`** (G<13, 7.37 M stars) is the standard archive, offline by default and fetched/merged on first use; `g10` (24 MB) bundled in the exe, `g15` reserved for the deep tier; Hipparcos + labels bundled. Two user choices: `gaia` (offline + bright fill) and `gaia_online` |
| Interfaces | app window by default, `mee2024 gui` (classic, unchanged), CLI |
| Version | v1.3.1; Windows exe built from `MEE2024.spec`, carrying the compact catalogue |

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

Stage 2 (`guess_date` seeded with 2020-01-01, blind; re-pinned at v1.1.0 for the
v2-seeded fit — rms/stars/nn_corr are equivalent to the v1-seeded values, and the
date shifts sit well inside the honest σ_t of ~16 d (zwo3) / ~25 d (zwo1); the old
zwo3-quintic "−1 d" was a lucky 0.06 σ draw):

| field | order | RMS | stars | guessed date | error | nn_corr |
|---|---|---|---|---|---|---|
| zwo3 | cubic | 112.5 mas | 432 | 2023-10-25 | −4 d | 0.385 |
| zwo3 | quintic | 108.9 mas | 433 | 2023-10-16 | −13 d | 0.167 |
| zwo3 | septic | 106.6 mas | 433 | 2023-10-19 | −10 d | 0.134 |
| zwo1 | quintic | 115.1 mas | 1565 | 2023-09-07 | −53 d | 0.351 |
| zwo1 | septic | 104.2 mas | 1565 | 2023-09-23 | −37 d | 0.191 |

All of the above is asserted by `tests/test_stage2_regression.py`, offline.

---

## 2026-08-01 — four UI bugs, a pointing check, and an executable that was 2.7 GB

**Closing the browser tab left the process running** -- and the first fix for it was
worse than the bug. A closed tab tells a server nothing, so the beacon-on-`pagehide`
plus idle-watchdog approach was right in outline and wrong in both details: the page
only polls *during a run*, so an open, idle page made no requests at all and the
watchdog shut the server down beneath it, leaving a frozen tab; and `pagehide` is not
proof of a close, since it also fires for navigation and for the back/forward cache,
from which a page can return. Now: a 5-second heartbeat runs whenever the page is
open, `pagehide` starts a 3-second countdown that any later request cancels,
`pageshow` cancels it explicitly, and the idle backstop is 150 s -- long on purpose,
because browsers throttle background-tab timers to about one a minute and a
backgrounded tab is still open. A run or an active watch always wins.

**pywebview was never a declared dependency**, so an install from source silently lost
the native app window that the README promises -- and with it the platform file
dialogs, since a browser tab cannot open one. That is why the app appeared in a
browser and why the native picker did not show up. It is now in `requirements.txt` and
`install_requires` (marker-guarded to Windows and macOS; Linux additionally needs
system webkit2gtk), and the launcher says which interface it chose and why instead of
quietly degrading. A `.venv` recipe is documented in the README: an environment
carrying unrelated heavyweight packages also inflates the packaged executable, as the
2.7 GB build showed.

`mee2024 ui --keep-alive` still keeps the server up regardless, for serving a browser
on another machine or reloading the tab freely while developing.

**The 3-D surface and correlation map grew on every scroll and drag.** `fitCanvas`
read the drawing height back out of `canvas.getAttribute('height')` — but assigning
`canvas.height` *is* that attribute, so `height = h × dpr` fed itself once per
redraw, inflating by 25 % a tick on a 1.25× display. The intended size is remembered
once and the CSS box pinned, so the backing store cannot move the layout either.

**The config file was never written by the app at all.** Only the CLI and the classic
GUI ever wrote it, so the app window forgot the last-used folder, catalogue and preset
on every launch. A run now saves them and `hello()` offers them back. On location:
`%LOCALAPPDATA%\MEE2024\MEE2024\MEE_config.txt` (`mee2024 config --show-path`) is
correct and stays — Program Files is read-only without administrator rights and cannot
hold per-user settings. The absence was the bug, not the address.

**Cubic is the distortion default everywhere** (the config default always was; the UI
select and the `auto` preset said quintic).

**Stage 1 now scores the solve against the mount.** Capture software records its own
`RA`/`DEC` (or `OBJCTRA`/`OBJCTDEC`), which is an independent check on the whole
chain: under 0.5° reads as good alignment, a few degrees as workable, tens of degrees
as something wrong upstream. Measured 0.05° on the new ZWO dataset.

**Native file dialogs are wired up at last.** The `native_dialog` hook has existed
unused since P1; `/api/pick` now uses it in the native window and reports
unavailability in browser mode, where the built-in picker remains the fallback. A
cancelled dialog stays distinguishable from having no dialog.

**Pattern databases are built on first use, not downloaded.** They are derived from
the star catalogue, so publishing them would mean shipping hundreds of megabytes a
machine can compute in seconds. Measured: 69 MB and **19 s** from the compact
catalogue (both real fields and 6 of 7 spot cases, and 0.1–0.3 s per solve — the
sparser query balls are much faster), against 230 MB and ~3 min for the full one.
`ensure_pattern_db()` picks the right build and reports progress like any stage.

**The executable was 2.7 GB.** PyInstaller followed optional `torch` references in
scikit-image and scikit-learn and bundled the whole CUDA stack — `cublasLt64_12.dll`
alone is 473 MB — so the release artefact's size depended on what the build machine
happened to have installed. Worse, **`excludes` does not prevent it**: excluding a
package prunes its Python modules, but binaries a hook already contributed survive,
and `excludes=['torch']` still shipped 1.25 GB of CUDA libraries. The spec now filters
the TOC tables after `Analysis`, which is the only place a size guarantee can be made.
Verified which heavy packages are genuinely used before filtering: cv2 (stage 1) and
statsmodels (the distortion and eclipse fits) stay; torch, cupy, jax and numba are
imported nowhere.

---

## 2026-08-01 — the duplicate-catalogue bug: a good field that matched nothing

**Reported symptom**: stage 1 solved a real 3008² field (RA 280.60, Dec 50.83, 1.56°,
1217 centroids, 100/100 verification stars), then stage 2 died in the polynomial fit
with `ValueError: Found array with 0 sample(s) (shape=(0, 9))`.

**Cause, introduced by the v1.2.0 catalogue work.**
`GaiaOfflineProvider.from_installed()` read *every* installed archive and concatenated
them — correct while the archives were guaranteed disjoint magnitude slices, and wrong
the moment they were not. With `gaia_dr3_g13` (a superset of the base + extension pair
it was merged from) and `gaia_dr3_g10` (a subset of it) also installed, every star was
listed two or three times. Duplicate entries do not merely waste memory: they defeat
the "nearest match must be twice as close as the runner-up" ambiguity test, because the
runner-up *is* the duplicate sitting on top of the winner. Measured on the reported
field: **190 stars matched within 36″, and the confusion test rejected all 190**. The
plate solver was unaffected only because it pins the single archive named in its
pattern-database manifest.

**Fix**: `choose_non_overlapping()` selects a coherent archive set using both ends of
each archive's magnitude range — the new `magnitude_min` manifest field, measured from
the data for archives written before it existed. The superset wins over the parts it
contains; the genuinely complementary base + extension pair is still read together; a
lone compact archive is still used. Four fast tests pin those cases.

**Also added**, because the symptom pointed nowhere near the cause: when many stars
match within the threshold but almost none survive the ambiguity test, stage 2 now says
so and names duplicate catalogue entries as the likely reason.

**Verified on the reported data**: the field now fits **185 stars at 0.092″ rms with
nn_corr 0.039** — a better fit than either ZWO reference field.

---

## 2026-08-03 — v1.3.1: batch rows say how well, and a doomed run stops in two frames

**Each finished field now carries its numbers.** `rms mas · n stars` beside the tick, so a
glance down twenty fields answers "did these measure well?" without opening any of them.
Taken from that field's own slice of the event stream — scoped by sequence number, because in
a batch the previous field's metrics are still in the sink — rather than by re-opening two
zips to recover numbers that had just gone past. Falls back to the centroid count when a
field stopped after stage 1, and says `not solved` when the plate solve failed.

**The source folder stays on screen whichever way a field ended.** It used to be replaced by
the error text, which removed the one thing wanted first: which folder to go and look at. Path
and error are now separate lines, and the row's tooltip carries the full source path and the
output path it was written to.

**Fail fast on unmatchable frames.** Centroid finding is the expensive part — seconds per
frame on a full sensor — and it ran over *every* frame before the first alignment was
attempted. Since every frame is aligned against frame 0, a set whose first two frames cannot
be matched was always going to fail, just after paying for all of them. The first pair is now
done up front and checked; on good data this costs nothing at all (each frame is still
centroided exactly once, which is pinned by a test), and on bad data a hundred-frame run
becomes a two-frame one. Measured on ten noise frames: **2 centroided instead of 10**.

The failure message names the offending frame and says what was skipped and why, so a run
dying two frames in reads as deliberate rather than as frames being silently dropped for
some other reason.

One thing found while writing it: `attempt_align` *raises* rather than returning `None` for a
failed match, so the first version's `if probe is None` was dead code — and the same `shift2
is None` branch in `_align_frames` is defensive-only for the same reason.

---

## 2026-08-03 — v1.3.0: recursive batch folder processing

Capture software writes a folder per field — `session / field / time / frames` — so the
useful unit of work is often a tree, not a list of frames. **Batch folders** (a checkbox
top-right) switches the input to a single folder, treats every folder beneath it that
*directly* holds frames as its own field, and mirrors that layout into the output.

On a replica of the reported layout: `5 field(s), 18 frame(s), from 11 folder(s)`, with the
`.CameraSettings.txt` sidecars ignored and a folder of frames ending the descent, so a stray
`thumbnails/` inside a capture folder cannot become a sixth field. Output mirrors rather than
flattens, because capture software reuses timestamps across nights and two `22_02_22` folders
would otherwise collide.

**Two limits, because one is not enough.** The requested cap on fields is there — 20 by
default, editable in the UI — but a field count alone does not protect anyone: a drive root
is an enormous *nearly empty* tree, where the walk itself is the cost rather than the runs.
So there is also a cap on directories examined (2000). Both **refuse outright rather than
truncating**: quietly processing the first twenty of two hundred would leave the user
believing the job had finished, which is worse than stopping.

**A failing field does not abandon the batch.** A night of observing is too expensive to lose
to one bad folder, so each field is caught, recorded and reported and the run continues;
cancellation is different and stops at the next field boundary, which is what Stop should
mean here. The **Stop button** now sits beside Run in the header, visible only while running.

The catalogue and the pattern database are prepared **once** rather than per field, which is
why `_run_one` was factored out of `_work` instead of looping the existing entry point.

**Verified both ways round on real data.** Five synthetic fields with no stars: every field
attempted, each row `✗ failed` with its reason, summary `0 of 5 succeeded, 5 failed`, and the
batch itself finishing `done` rather than aborting — the isolation exercised for real, not
just in a test. Then two genuine fields (the Rasalhague frame, and three of the eclipse
lights) through **both stages** in 160 s:

```
A_rasalhague_28_47     done   1 frame   centroid + distortion zip
B_eclipse_field_04_15  done   3 frames  centroid + distortion zip
```

each written to its own mirrored folder. 41 new tests over the bounds, the refusals, the
mirroring, name collisions, per-field failure isolation and cancellation.

---

## 2026-08-03 — v1.2.8: a bundled catalogue the app offered but could not open

Reported from the executable: selecting `gaia_dr3_g10` plate-solved fine and then died in
stage 2 with `FileNotFoundError: 'gaia_dr3_g10'` out of the *legacy Tycho reader*, which had
been handed a catalogue name and tried to `open()` it as a CSV file.

**The cause was two functions disagreeing about where a catalogue may live.**

| | user's data directory | bundled inside the exe |
|---|---|---|
| `CatalogueRelease.is_installed()` — feeds the app's catalogue list | yes | **yes** |
| `database_cache._installed_catalogue_dir()` — opens it | yes | **no** |

So the executable advertised the archive it ships inside itself, and then could not find it:
the lookup fell through every branch to the legacy reader, whose error message named a
catalogue rather than a path and pointed nowhere useful. Only the exe build could hit it,
since only there is a catalogue bundled rather than downloaded. Availability and location
now consult the same candidates, installed copy first, since a downloaded archive is the
deeper one. Verified by copying g10 to a bundled-only location: it resolves, opens as
`gaia_offline (offline, G<10.0) [gaia_dr3_g10]`, and serves a lookup.

**A second, quieter wrong thing in the same report.** The log said

> gaia_dr3_g10 only contains stars to G<10 … **Install gaia_dr3_g13** to reach G<13.

two lines above *building the pattern database from gaia_dr3_g13*. The advice was telling
the user to install what they already had, which wastes their time and makes the rest of the
message look untrustworthy too. When the deeper archive is present the problem is the
*choice* of catalogue, so it now says so instead.

---

## 2026-08-03 — v1.2.7: the name fix, actually wired up

**v1.2.6's positional name lookup did nothing in the app, and I reported it as working.**
The resolver was correct and I verified it against the real bundled index — 50 of 50 named
stars — but I verified it *in isolation*, not on the path the UI reads. Positions were
wired into stage 1's `emit_from_solution` and **not** into stage 2's `star_labels.emit`.
The frontend deliberately lets the stage-2 event supersede stage 1's, because that one knows
which stars the fit discarded as doubles — so the event that reaches the screen carried no
`ra`/`dec`, the positional lookup was never called, and every label fell back to a
magnitude. Reported by the user still seeing `G 2.1`.

Both call sites now pass sky positions, and stage 1 passes the real observation epoch rather
than defaulting to 2024.0 — that was breaking nothing here, a year or two of drift being
well inside the 10″ match radius, but naming stars *by position* against the wrong epoch is
a trap set for the first fast-moving star to come along.

**Verified through the real pipeline this time**, `do_stack` → `match_and_fit_distortion`,
checking both events:

```
stage      stack:  93 stars, 1 named -> 'Rasalhague', tier 'named', mag 2.11
stage distortion: 248 stars, 1 named -> 'Rasalhague', tier 'named', mag 2.11
```

A unit test on the resolver could not have caught this, so the wiring itself is now pinned:
`test_both_star_label_emitters_pass_sky_positions` parses the argument list of every
`star_labels.emit*` call in both modules and fails if `ra`/`dec`/`epoch` are missing.

---

## 2026-08-03 — v1.2.6: named stars get their names, and dropped stars say why

**"Why is Rasalhague labelled G 2.1?" turned out to be a general failure, not one star.**
The name is in the index — `names.txt` maps it to HIP 86032, and asking by HIP returns
`Rasalhague`. But a Gaia source_id reaches a name only through **Gaia's own crossmatch to
Hipparcos** (`gaiadr3.hipparcos2_best_neighbour`), and that table covers 99,525 of 117,955
HIP stars. Which ones does it miss? Measured: **46 of the 49 named stars.** Vega, Sirius,
Betelgeuse, Polaris, Arcturus, Rigel, Canopus, Antares — only three worked. Gaia struggles
with the brightest stars, and the named stars are the brightest there are, so the label
feature was broken for precisely its entire audience.

**Fixed by resolving names from the sky instead.** Named stars are few and far apart, and
their Hipparcos positions come from the catalogue already bundled for the bright fill, so
`LabelIndex.names_by_position` is a brute-force match against about fifty candidates
propagated to the observation epoch, cached per epoch. No index rebuild, no archive access,
no new bundled data. **Verified against the real bundled index: 50 of 50 named stars
resolve, where 3 did before.** It also upgrades a correct-but-worse `HIP 86032` to
`Rasalhague`, and is a pure fallback — an id that already resolves is untouched, and a
failure leaves the magnitude label.

**And a star dropped from the fit now says why it was dropped.** A missing proper motion
means a stale position, which is a different thing from a bad measurement — but it was
reported as an `outlier`, which sends anyone investigating in exactly the wrong direction.
The run now says so in words, with the brightest affected magnitude and the worst miss, and
separates the three cases in `METRICS`: `n_dropped_no_proper_motion`, `n_dropped_double`,
`n_dropped_unexplained`. The flags were always computed and written to the CSV; nothing had
ever surfaced them.

---

## 2026-08-03 — v1.2.5: the brightest stars stop being thrown away

**Rasalhague, the brightest star in its frame, was being discarded — and not for any of the
reasons that looked likely.** Traced through the whole pipeline on
`tests/data/fits/rasalhague`:

| stage | verdict |
|---|---|
| in the catalogue? | **yes** — `gaia:4493746564376875520`, G=2.11, precision-grade, no duplicate within 0.05° |
| saturated-blob removal | doesn't touch it |
| centroid detection | found, **flux rank 1 of 1801** |
| `min_area`, sanity check | pass (65469 → 63560 → 55009 → 42629) |
| stage-1 plate solve | **matched**, 0.00 px, to the mag 2.11 entry |
| stage-2 distortion fit | **dropped**: `error(") = 1.845` against a 1.0 tolerance |

Two guesses of mine were wrong on the way, which is worth recording. I expected the
**saturated-blob remover** — but it needs ≥20000 connected saturated pixels *after* 8×8
mean-downsampling, and this core is 4×4 (16 px at ≥99% of peak). It looks like a blob when
stretched; numerically it is a compact star. Then I expected the **2× confusion filter**,
α Ophiuchi being a close binary — but Gaia does not resolve the 0.5″ companion and the
bright fill added no duplicate, so there was nothing for the ratio test to trip on.

**The real cause is a Gaia data gap: `flag_missing_pm`.** `pmra`, `pmdec` and `parallax` are
all NaN, because Gaia's brightest stars often get two-parameter solutions — they saturate
its detectors. Without a proper motion the position cannot be propagated, so it stays at the
2016.0 catalogue epoch while the frame is 2022.63: 6.6 years times ~250 mas/yr is ≈1.6″,
essentially the whole 1.845″ residual. **It fails the outlier cut precisely because it is
bright**, and nothing said so — `remove_missing_pm` is off by default, so it was swept up by
the outlier test and reported as though the *measurement* were bad.

That it is staleness and not saturation is unambiguous: every other star of the brightest
fifteen sat at **0.01–0.25″**.

**It is systematic.** Fraction of the catalogue with no proper motion: **21.2% at G<4**,
8.7% at G 4–5, 7.1% at 5–6, 4.1% at 6–7, 1.2% at 8–10, 0.8% at 10–13. The brighter the
star, the likelier it is discarded.

**Fixed by borrowing the proper motion, on the fly.** Hipparcos has good motions for exactly
these stars and is already merged in to fill the bright end, so `fill_proper_motion` lends
one where Gaia has none. Merged at lookup time rather than baked into a unified archive:
that keeps `gaia_online` working identically, keeps the data sources separable, and avoids
re-packing and re-publishing a 320 MB asset with a pinned hash for something that costs
microseconds. It is also where the bright *star* fill already happens.

**Position from Gaia, motion from Hipparcos** — better than either alone. Gaia's position is
good to about a milliarcsecond at its own epoch and Hipparcos' motion adds a few mas over a
decade, where a pure Hipparcos position would carry thirty years of its own propagation
error. **Measured: Rasalhague's residual falls from 1.845″ to 0.408″, it is no longer an
outlier, and every other star is unchanged** (0.144 vs 0.147, 0.253 vs 0.254, …). The filled
values — pmra 108.07, pmdec −221.57 mas/yr, parallax 67.13 mas — match the published ones
for α Oph.

Ordering matters and is the subtle part: the fill must happen while positions are still at
the **catalogue** epoch, because propagation is what turns a missing motion into a stale
position. So `lookup(epoch=None)` gained a meaning — "leave the positions where the
catalogue has them" — and the merge fills, then propagates. Providers that do not implement
it, including a table returned with no epoch at all, fall back to the plain lookup, so this
can only improve on the previous behaviour; that fallback is pinned by a test, having been
found by one.

Each fill source is used only within **its own magnitude ceiling**: Tycho's positions reach
~2.5″ by V=11, so past that a 2″ match radius would be adopting random neighbours rather
than identifying the same star. A magnitude agreement of 2 mag is also required, since Gaia
reports G and Hipparcos Hp/V.

---

## 2026-08-03 — v1.2.4: hot pixels found without a dark frame

Darks are not always taken, and when they are they are not always usable — the ones in the
bundled example were shot 45 minutes late and run three times hotter than the lights. So
`mee2024/hotpixels.py` now finds hot pixels from the dither instead: **a star is fixed to
the sky, a hot pixel to the detector**, so asking whether a bright site persists at a fixed
detector pixel or at a fixed sky position separates them with no dark anywhere in the
statistic. The exploration, the rules that did worse, and the figures are in
`docs/bench/HOTPIX.md`.

**Measured end to end, with the darks withheld from the run:** 161 hot pixels found from
the dither alone out of 5237 bright candidates, of which **160 are confirmed by the
dark-based mask the run never saw — 99.4% precision**. One centroid was dropped and the
alignment redone. The plate solve landed at RA 155.4052978**49**°, against
155.4052978**17**° for the run that did use darks: the same answer to 3×10⁻⁸ degrees.
Stage 1 went from 138 s to **142 s**, matching the 4.8 s the search was measured to cost.

**Darks are still better when they are good.** The dark mask holds 296 pixels; only 161 of
those were bright enough in the lights to be candidates at all, so the dark-free path found
essentially every one it could see but not the 136 milder ones. Those contaminate less, but
they are not nothing — this is a fallback, not a replacement.

**The combination matters more than the measurement.** The obvious discriminant, detector
persistence *minus* sky persistence, manages 0.962 average precision and only 21.7% recall
at zero false positives, because an absolute gap conflates a faint hot pixel with a bright
star. The **log ratio** reaches 0.996 and 96.3%. The threshold sits mid-plateau — anything
from 1.0 to 3.5 gives 98–100% precision — so it is not tuned to an edge.

**Can a hot pixel reach the aligner?** In principle yes, on this data no, and the reason is
circumstantial rather than structural: centroids rank by *integrated* flux and `min_area` is
4, so the one hot centroid among 388 ranked **106th**, far outside the brightest 30 the
aligner uses — a saturated pixel at 16380 ADU integrates to 280 against 11127 for the
brightest star. That protection fails for hot clusters or hot columns, and in sparse fields
where the brightest 30 is most of the list. Hot pixels vote for shift (0,0) because they do
not move, so with a large dither enough of them could pull the alignment onto zero. Hence
cleaning the lists and realigning rather than trusting the first pass.

**Ordering, and why it is not the obvious one.** The dark path masks *before* centroid
finding, so a hot pixel never becomes a centroid. The dark-free path cannot: it needs the
shifts, which need the centroids. So it aligns, searches, **filters the existing centroid
lists**, and realigns — re-detecting would cost more than everything else here put together,
and a bad centroid only has to be dropped. `_align_frames` was extracted to run twice, and
`FRAME_ALIGNED` events moved out of it so they are emitted **once, from whichever alignment
turned out to be final**; otherwise a discarded first pass would report shifts the run never
used.

**Guards, each because it can bite.** Fewer than three frames, or dither under 3 px, is
declined *with the reason logged* — with no dither the two measures are identical by
construction and every star would be flagged, so silent nonsense is the real risk. The
saturated-blob mask is excluded from candidates and the list is capped at 200k: a lunar or
solar disc would otherwise contribute ~10⁶ sites which classify correctly but waste the work,
and an uncapped allocation on unseen data is not something to ship. `hot_pixel_dark_free`
turns the path off.

Still open: using it as a cross-check when darks *are* present. It found at least one real
hot pixel the dark's own 20σ cut had missed (row 3219, col 4195: 27σ detector persistence,
−1.3σ sky, 63 px from any star).

---

## 2026-08-02 — v1.2.3: the stack keeps its ADU, and hot pixels stop pretending to be stars

Three defects found by working through `tests/data/fits/example_with_darks`, a 12-bit
5644×8288 set with 7 lights and 13 darks. Each of them let a run complete without
complaining, which is why they had survived.

**The stacked FITS was a display stretch saved as science data.** The line was

```python
stacked16 = ((stacked - min) / (max - min) * 65535).astype(np.uint16)
```

so every output filled 16 bits regardless of what went in: 12-bit data came back 16-bit,
the black point moved to wherever the darkest pixel happened to be, and the numbers stopped
meaning ADU. The gain is part of the measurement, so it is kept now — the stack is written
in the input frames' own units with `BITDEPTH`, `NCOMBINE` and the version in the header.
**Measured on the example: the stack now spans −268…15846 ADU against a 12-bit-times-four
full scale of 16380, where before it was stretched across 0…65535.**

**Hot pixels survive dark subtraction, and the arithmetic says why.** They clip, and
clipping is not linear: light and dark both sit at full scale and their difference says
nothing about the sky. Measured on this data, the residual *after* subtracting the master
dark was still **35 sigma above the background at the median hot site and 258 sigma at the
worst**. Being fixed to the detector while the field is dithered, they then smear across
the stack as a small constellation of fake stars — exactly what was reported. They are now
found from the master dark's own distribution (bulk median 306 ADU, robust sigma 5,
99.999th percentile 396, so a 20-sigma cut at ~406 sits far outside the honest pixels) and
**excluded from the stack rather than subtracted**: 296 pixels of 46.8 million. Excluding
them from the *count* as well as the sum is what removes them instead of diluting them —
each sky position simply loses whichever frames had a bad pixel under it.

Measured by running the example both ways and differencing the two stacks, which isolates
the change exactly (counting "isolated spikes" was useless — at 1.15 arcsec/pixel a faint
star is nearly a spike too, so that measure was dominated by real stars). The exclusion
changed **1,731 sky positions** — 296 bad pixels seen at about six dither positions each —
and of those, **761 had been contaminated by more than 10 sigma, 111 by more than 40, and
9 by more than 100**, the worst by 1045 ADU or 176 sigma. That population of fake stars is
the reported smudge. The cost is one frame of seven at those 1,731 positions.

**These darks do not match these lights, and now the pipeline says so.** The master dark's
median is 306 ADU against the lights' 116: they were taken 45–55 minutes later and are
about three times hotter. Subtracting them drives the background to −190 ADU. The old
min–max stretch hid this completely. The stack now carries a recorded `PEDESTAL` (subtract
it to recover ADU, rather than clipping most of the frame to zero) and warns, naming the
likely cause. That is a data problem, not a code one, but it should be visible.

**Bit depths must now agree.** `BITDEPTH` where the camera writes it — the container's
`BITPIX` describes neither the sensor nor the scaling — falling back to `BITPIX`, with
frames that declare nothing skipped rather than assumed. Mixing depths subtracts numbers a
fixed factor too large and looks like bad data rather than an error.

**A fourth, found on the way:** the flat was used raw, so dividing by it scaled the frame
by thousands. The output stretch had been renormalising that away. Flats are normalised to
unit median now, which is what a flat is for.

**The stacked image displays inverted by default** — dark stars on white, the way a plate
is traditionally examined, and easier on faint stars. It is a composite over the drawn
image rather than a change to the PNG, so the toggle is instant; markers and labels are
drawn after the inversion and carry their own palette, since pale blue on white is
unreadable.

---

## 2026-08-02 — v1.2.2: Clear clears everything, and master calibration frames are kept

**"Clear" dropped only the light frames.** Darks and flats stayed selected, and stayed
selected *silently* — the only trace of them is a line of small print under the buttons —
so the next run could be calibrated with frames chosen for a different session. It now
clears the card, and says so: the button reads **Clear all**.

**The combined dark and flat are written to the output folder** as
`DARK_STACK<timestamp>.fit` / `FLAT_STACK<timestamp>.fit`, with `NCOMBINE`, `COMBTYPE` and
the program version in the header so a master frame reused months later can still say what
it is. `save_dark_flat` already existed but defaulted off and fired on a single frame;
it now defaults **on** and writes **only when two or more frames were actually combined** —
one frame averaged is a copy of its input, and a copy under a new name reads like a product
that it is not. Lifted out of `do_stack` into `save_calibration_stacks()` so the rule is
testable, which it now is, at the boundary and either side of it.

---

## 2026-08-02 — v1.2.1: every plot now agrees with the picture it describes

**The 3-D surfaces were upside down, and nobody could see it.** `project()` mapped +z to
*downward* on screen. On the signed dx/dy views that is invisible — a distortion field is
as plausible inverted as not — but the moment the |displacement| surface arrived, which
cannot go negative, its floor hung above its peak. Fixed at the source, so all three views
change together. Two things fell out of the same reading: the depth key had the **opposite
sign to its own screen mapping**, so the far side of a surface could paint over the near
side; and the *earlier* commit's legend edit ("amber above the surface, blue below") was
premature — it described the physics while the renderer still drew z downward, so it was
wrong when written and is right now. Both halves are needed; neither is right alone.

**Everything now uses the image's own axes.** `y` in this pipeline is a row offset, so it
increases *downward*, and every plot drew it upward: the 3-D surfaces, the residual
correlation map (which explicitly flipped `j`), the distortion field and magnitude panels,
and the residual scatter. A map of where the optics misbehave is useless if it mirrors the
frame. All of them are now top-left origin, like the image.

**Aspect ratios are honest in pixel space.** The correlation map drew *square cells*, but a
cell covers W/nbins × H/nbins pixels — on a 3520×4656 sensor that stretched the detector
into a square. The grid now carries the frame's shape. (The 3-D surfaces already shared one
length scale across x and y; the matplotlib panels already had `set_aspect('equal')`.)

Verified numerically rather than by eye: the projection's five sign conventions checked
one at a time, the correlation map's filled quadrant tracked to the frame quadrant that
holds the stars, and the drawn grid measured at **1.502** against an expected 1.500.

**The stacked image gained a zoom control** (1–4×) and its preview is rendered at 1600 px
instead of 900. The backing store deliberately stays at the image's own size whatever the
zoom — scaling it would be ~800 MB at 4× on a full frame — so zoom only widens the CSS box
and the container scrolls.

**A deep build can now be left alone.** `tools/build_gaia_offline.py` prints a running
`rows/s · elapsed · ETA` revised from measured rate (the static up-front estimate assumed
a fast archive, which the last two days have not been), and **chunks are written to a
`.part` file and renamed**. The rename is atomic, so a build killed mid-write leaves a
chunk either absent or whole — previously a truncated cache entry would have been counted
as complete on the next run and never refetched, a silent hole in the sky. `RELEASING.md`
carries the g15 command, its ~6–8 GB assembly peak, and what to expect.

---

## 2026-08-01 — one settings panel, and the stacked image gets named stars

**Simple and advanced modes are one mode.** Two modes meant two places for a setting to
be, and the simple one hid controls people wanted; now there is a single always-advanced
panel that collapses. The display-distortion checkbox is gone — the field plot is always
drawn, because it costs one basis evaluation and answers the first question anyone asks of
a fit. `max_star_mag_dist` defaults to 13, matching the standard archive's depth. The date
is a three-way choice (**guess it / read the FITS header / type one**) rather than a
checkbox plus a text field that contradicted each other; reading the header is new, and
falls back to guessing with a warning when the frames carry no date. **Removing double
stars and removing stars with no proper motion are now separate checkboxes** — and
separate in the fit, which they were not: asking for either used to get both
(`distortion_fitter.py`, `remove_missing_pm` added to the defaults). The scoreboard gained
a **"Vs. telescope"** card: how far the solved position sits from the FITS header's, in
degrees, with the same 0.5°/5° grading the log message uses.

**The stacked preview is darker, and its stars have names.** The stretch anchored its black
point at the 25th percentile, which rendered the sky mid-grey — but a star field is almost
entirely sky, so most of the frame *is* that percentile. The black point now sits at the
median with the white point at 99.9%, and a γ=0.65 curve lifts the faint stars back without
lifting the sky: the background goes dark and the stars are visible.

Over that image, the identified stars are drawn as a live overlay from a new `STARS` event
(`mee2024/star_labels.py`), and a **slider adds labels in tiers — none → named → HIP →
bright Gaia → all** (the last at a smaller font, because a wide field identifies hundreds).
Positions travel as columns rather than baked into the PNG, so changing the tier costs no
run: how many names fit is a question about screen space, not about the data. Both stages
emit; the distortion fit's event supersedes the solve's, because it worked from a deeper
catalogue and knows something the solve did not — **which stars it discarded as double
stars, which the frontend crosses out in red** with a legend saying why (a close companion
pulls the measured centre off the catalogue position). Only the stars the fit used, plus
those crossed-out doubles, are drawn: an outlier dropped for another reason would otherwise
look like a star that had been measured. Verified end to end in the browser — 4 synthetic
stars, tier stepping, and a pixel check that the eliminated one is red and the kept one is
not.

Threading that needed a small solver change: `platesolve2.verify.match_centroids` now
returns the **catalogue ids** of the stars it matched, carried out as `matched_ids`, which
is what lets a label be a name rather than a magnitude.

### Measured: what a deeper archive would cost, and why the time figure is a range

**Size is settled.** Two independently built archives agree on bytes per star to within
1.5% — `gaia_dr3_g12` packs 3.09 M stars into 138.0 MB and the 12–13 extension packs
4.28 M into 188.6 MB, i.e. **~44.4 B/star zipped, ~49.7 B/star on disk**. So:

| tier | stars | download | on disk |
|---|---|---|---|
| G < 14 | 16.8 M | ~750 MB | ~0.84 GB |
| G < 15 | 36.9 M | ~1.64 GB | ~1.83 GB |

**Build time is not settled, and the honest answer is a range spanning two orders of
magnitude.** The recorded all-sky G<12 build fetched 3.09 M rows in 29 queries in 10–22
minutes — **2,300–5,200 rows/s**. An uncapped async probe run today fetched one 1° stripe
(222,483 rows at G<15) in **3,951 s — 56 rows/s**, a factor of 45 slower. Nothing in our
code differs between the two; the archive was simply loaded. Extrapolating today's rate
gives 182 h for G<15, which I do not believe as a *typical* cost, and quoting it as one
would be dressing up a bad-day sample as a measurement.

What can be said: at the historically measured rate a G<15 build is **~2–4.5 hours**, with
a hard floor of ~1–1.4 h set by query latency alone (~185 chunks at 200 k rows each, 20–28 s
of latency per query regardless of size). On a day like today it is a multi-day job. The
build is resumable — each stripe chunk is cached as an `.npy` keyed on its band — so a slow
archive costs patience rather than restarts, which is what makes the wide range tolerable.
**Recommendation: an overnight run, started when the archive is quiet, not a foreground
task.**

### Measured: the query shape is not the problem

The obvious suspicion was that the probe asked badly — 1° stripes where the builder uses
10°, a `dec BETWEEN` filter on a non-clustering column, astroquery's default VOTable XML,
and the full-width `gaia_source` row. Four variants at matched row count (~35 k at G<15),
run back to back so they saw similar archive load:

| strategy | rows/s |
|---|---|
| A `dec` range, `gaia_source`, VOTable — what the builder does | 45 |
| B `dec` range, `gaia_source`, CSV | 43 |
| C `dec` range, **`gaia_source_lite`** | 50 |
| D **`source_id` (HEALPix level-4) range**, `gaia_source` | 49 |

**A 15% spread — nothing.** `source_id` is Gaia's clustering key (it encodes the level-12
HEALPix index in its top bits, `source_id = hpx12·2³⁵ + n`), so a pixel range should be a
contiguous scan rather than a filtered one; it is not measurably faster. Neither is halving
the row width, nor skipping XML parsing. The four ran over ~45 minutes, so drifting archive
load is confounded with the variant — but that confound cannot hide a 45× effect, and there
is no effect to hide.

Combining with the earlier 222 k-row probe gives **~200 s fixed per job plus ~59 rows/s
marginal**: linear in rows, so chunking differently only moves the fixed term, which is
already small. The variable that actually swings the answer by 50× is the archive's own
throughput on the day. **No builder change is justified** — `tools/build_gaia_offline.py`
stays as it is.

Nor is the server the lever. **GAVO's `gaia.dr3lite`** — a deliberately slimmed Gaia DR3,
the most promising candidate precisely because it is built for sweeps like this — returned
the same band at **15 rows/s, three times slower than ESA**. VizieR answered 503 and gave
no measurement (and an earlier attempt through astroquery's `TapPlus` measured nothing at
either mirror: a 406 from AIP, and a certificate failure at VizieR because `TapPlus` POSTs
through raw `http.client` and so uses the system trust store rather than `certifi` — client
problems, not slow servers). Across five ESA queries over two hours the rate held at
44–50 rows/s, so this was a sustained bad day rather than a bad moment.

**Remaining levers, none of them code:** when the build is run, and an authenticated ESA
session (logged-in users are given higher priority than anonymous ones). Everything about
*how* we ask has now been measured and does not matter.

---

## 2026-08-01 — v1.2.0: one catalogue, two choices, offline by default

**The catalogue set is now one archive and two options.** `gaia_dr3_g13` (7,369,627
stars, 366 MB, verified, brightest G=1.73) replaces the G<12 + 12<G<13 pair as the
standard depth: the split saved 138 MB and cost more confusion than that was worth, and
an extension-only install is a genuine footgun. `mee2024 catalogue --merge` builds it
from an installed pair without downloading anything — and **recomputes the double-star
neighbour flags over the union**, which the separate archives structurally could not do
(203,016 stars now carry a close-neighbour flag). The two other tiers are the ends of the
range rather than more slices: **`gaia_dr3_g10`** (482,106 stars, **24 MB**, bundled into
the Windows executable so it solves offline out of the box, and kept out of source
control — the spec takes it from wherever it is installed) and `gaia_dr3_g15` (GB scale,
special needs).

**Two user-facing catalogue choices, not six.** `gaia` is the installed offline archive
plus the bright fill — which is what "merged" always was, to two decimal places, so it
takes the honest name with the footnote shown live in the picker — and `gaia_online`
queries the archive per field. Tycho, Hipparcos, `merged` and `merged_offline` stay
registered for tests and advanced use but are off the menu: neither Tycho nor Hipparcos
alone is a catalogue to reduce a plate against.

**Offline is the default.** `gaia` reads an installed archive when there is one and only
falls back online until then, so the minutes-per-field online path is now opt-in
(`gaia_online`). Declining the first-use download warns and keeps working rather than
failing. Double-star depth is clamped to the catalogue with a stated note; the arithmetic
is in `_lookup_neighbours` (a companion at Δm displaces a centroid by ~10^(−0.4Δm) of its
separation, so the offline limit costs little and buys milliseconds).

**Two robustness fixes the work forced out, both from a real failure.** The installed
`gaia_dr3_g12` turned out to be **half-installed — every data column present, no
manifest** — because a running instance held the files memory-mapped and Windows refused
the completing write. Every "is it installed?" check answered no, so the archive had
silently vanished from the app's view (which is why a fresh run went online), and the
first merge cheerfully produced a "G<13" archive containing nothing brighter than G=12.
Now: `broken_catalogues()` detects data-without-manifest, `--merge` refuses while one
exists (naming the repair), a merged base whose brightest star is fainter than G=8 is
rejected as a mislabelled extension, and **`mee2024 catalogue --repair`** rebuilds a lost
manifest after validating the data (equal column lengths, declination genuinely sorted,
band index recomputed and compared, finite in-range positions, depth consistent). It
recovered the archive with no download.

**A null result, recorded rather than shipped:** rebuilding the 0.6° pattern layer from
the deeper G<13 star list changed nothing — identical pass/fail on every sub-degree case.
Deeper *legs* were never the constraint; anchor density is (a 0.24 deg² field holds ~1.6
anchors at the shipped density). Sub-degree solving wants the GB-scale anchor-dense layer,
which is the `g15`-tier conversation, not a free win.

---

## 2026-08-01 — v1.1.0: the rebuilt solver becomes the default

**`platesolver` defaults to `'v2'`**, with automatic fallback: a fresh install has
neither the pattern database nor the offline catalogue, so `preflight()` checks
both and quietly uses the classic Tycho solver until
`mee2024 catalogue --fetch gaia_dr3_g12` + `mee2024 build-pattern-db` (≈3 minutes)
unlock v2 — at which point installing the optional t06/t40 layers extends blind
coverage to 1°–18° with no further configuration. A config migration moves
pre-v1.1.0 `platesolver='triangle'` settings to `'v2'` once, with a note;
re-choosing `'triangle'` afterwards sticks.

**Verified end to end, with one honest re-pin.** The full suite passes with v2
seeding the pipeline; rms, star counts and nn_corr reproduce the baseline to
within measurement (108.9 vs 109.6 mas, 433 vs 434 stars, nn_corr 0.167 vs 0.166).
The **blind date guesses moved** — zwo3-quintic from −1 d to −13 d — because the
date+distortion fit is partially degenerate and a solver seed 0.33″ away settles
in a neighbouring optimum. That is not a regression: the honest capability is
σ_t ≈ 16 d for this field, and the old −1 d was on record as a lucky 0.06 σ draw.
The regression pins are re-measured for the v2-seeded fit and the blind-date test
now asserts the *capability* (21 d, the UI's green threshold) rather than the
luck. The date-guess degeneracy is worth revisiting on its own — fitting date and
distortion jointly with a proper-motion prior would shrink it — noted alongside
the acceptance-statistics work.

---

## 2026-08-01 — S6: blind at every platescale; the quad question answered by data

**Three FOV layers, one solver.** The same builder produced a 0.6° layer (deep
star list, 334 MB) and a 4° layer (bright sparse anchors, 60 MB) beside the 1.7°
primary; candidates from every layer merge into a single consensus, since scale and
orientation are physical quantities that don't know which layer found them. The
failure ladder gained one rung — all layers — after the primary's rungs, so a
standard field pays nothing. Installing a layer *is* the configuration.

**Bench (corpus v5, now 104 cases): 77 → 89/96 solvable (0.927), twelve flips all
fixed, none broken, zero wrong solves.** The blind envelope is **1°–18° in both
density regimes** — the owner's goal of blind solving at every platescale down to
the floor, delivered. The remaining sub-degree cases are the G<12 catalogue floor
itself (midlat) and a dense-field case a fatter locally-built t06 handles (its
1.2 GB draft solved it; the shipped 334 MB trim gives it back — parameters
recorded).

**S7 (quads/pentas): declined, by measurement.** The 4° layer alone solves 10–18°
at ~1 s including the new `widedist` family — wide fields with the 8 px optical
distortion wide lenses actually have. Two lessons the numbers teach: wide fields
are information-cheap (5× fewer patterns per layer beats the curvature-inflated
tolerance, exactly as the information argument predicted), and real wide-lens
distortion dwarfs projection curvature while respecting *no* invariant, projective
or otherwise — tolerance modelling is the only defence, and the S3 model already
provides it per candidate. Filed alongside a v2.1 idea: feed the pipeline's own
fitted distortion polynomial back into the solver's tolerance floor for a fixed
instrument.

---

## 2026-08-01 — S5: the index disappears; anchors go progressive

**The pattern database no longer has a load time.** Kendall triangle columns are
bucket-sorted at build time over an (x, y) grid — the shape sphere is
two-dimensional on disk, z being derived — and a query is a gather of grid cells
from the memory-mapped files plus an exact metric ball, asserted identical to the
KD-tree by test. **DB load 12–13 s → 0.0 s; a cold process solves a real field in
3.3 s total; warm solves 1.8 s.** The ~1 GB resident tree is gone, which also
retires the solver's share of the UI-server memory concern from `docs/UI_DESIGN.md`.

**Anchors are now a ladder, not a prefix.** Brightest-9 first, then ranks 9–17
merged in, then noise escalation — so saturated artifacts outranking every real
star (the classic real-frame failure v1's design doc named) cost one extra query
instead of the field. Measured on the new `artifact` corpus family: with 12 fakes
poisoning the whole first round — provably unsolvable single-round — **both midlat
cases flip to solving, 4/8 → 6/8**, in 11 s median instead of 33 s of failing. The
two sparse 12-artifact cases fail either way (6 real stars in the top 18 is below
the matching floor, whoever anchors). Honest cost, recorded: junk rejection exhausts
the ladder, 36 → 58 s — only fields with no sky pay it.

Chain: v1 61/80 → S1 62 → S2 64 → S3 65 → S4 69/80 → S5 75/88 solvable (corpus
grew twice: un-lopsided poles, then artifacts), **zero wrong solves at every
stage**.

---

## 2026-08-01 — S4: the poles fall — to three fixes, honestly attributed

**Poles 0/4 → 4/4** (corpus v2; 69/80 overall, wrong solves 0, junk 8/8). The stage
was built for the quaternion consensus key, and the bench then forced an honest
attribution: three defects had stacked up at the poles, and the chart singularity
was only one of them.

1. **The instrument lied at the poles.** The synthetic generator's RA-window dropped
   in-frame stars on the far side of the pole, so polar test fields were lopsided
   with elongated triangles. Fixed (full RA circle when the field reaches the pole);
   corpus bumped to v2 and the S3 reference re-run per protocol.
2. **A real solver bug v1 shares**: the verification region is built from the field
   *corners*, but a pole-containing field's declination extreme — the pole — lies
   between corners, so verification excluded the stars nearest the pole and every
   correct candidate starved. Invisible until now: no earlier consensus ever brought
   a polar field as far as verification. Fixed with a pole-aware bbox.
3. **The chart defect is real but survivable**: candidate orientations at the pole
   scatter 3.5× anisotropically in the roll-like components — near-threshold for the
   legacy (roll, centre) key, which nonetheless chains through on full-frame fields.

With 1+2 fixed, both consensus keys solve the poles; the quaternion key's measured
delta is zero. **Adopted anyway**: it removes the roll-wrap and pole defects by
construction rather than by margin, at zero measured cost (`v2_consensus=legacy` is
the rollback). The satisfying by-product: the per-candidate orientation map turns
out to be a *reflection* (det = −1, the pixel/sky handedness flip) — which finally
explains v1's mysterious "+90/+180 roll" convention shifts: they compensate for
decoding Euler angles from an improper matrix. The quaternion path composes with a
fixed reflection first, so its conversion is exact.

Recorded for S5: a size-aware consensus radius (candidate orientation noise scales
as ε/S, exactly like the S3 match radius — the same physics, one level up).

---

## 2026-08-01 — S3: the tolerance becomes a physical model; dimmer-legs judged

**The query radius is no longer a constant.** Per triangle:
`r = 0.0006 + 4.8·(2√2·ε/S) + 0.93·(θ_db/2)²`, fitted on 4,677 identity-verified
true pairs (the generator now returns its stars' source ids, so true pairs come from
identity, not from a query that would truncate the tail). The curvature coefficient
landing at ~1 is the design doc's projective prediction, measured. ε comes from the
new `platesolve_noise_px` option (default 0.3, stacked-image grade); total failure
escalates ×3 with radii capped and candidates budgeted, so a noisier-than-assumed
image costs a retry rather than widening every search. Verification depth now tracks
the detection count (8× cap), so sparse fields are compared against stars they could
actually have detected.

**Bench (docs/bench/BENCH.md): 64 → 65/80, gates all hold.** The 8 px-noise case is
recovered by escalation. Success-path medians collapse — real fields 5.4 → **1.6 s**
warm, reliability/scatter ~2.7 s — with solved-case candidates 758 k → 518 k median
and ~156 k on the real low-noise fields: the S1 verification margin, spent as
designed. Failure paths pay for the ladder (junk 28 s, poles 35 s, budget-bounded);
S4 turns the poles into successes, which reclaims most of that.

**sparse-10 stays failed, now with the mechanism pinned all the way down**: the
depth fix restored its matches 6 → 8 of 10, but the acceptance threshold floors at
~9 = 3 defining stars + 3 addon + x1≈3 — the addon is the deliberate safety margin,
and lowering it is acceptance-statistics work (the corrected p-value experiment),
not tolerance work.

**S2b, decided by the pre-registered rule: dimmer-legs is NOT adopted.** Identical
size, scatter sweep tied 12/12 — and three wide-FOV/reliability cases break
(65 → 62/80). Storing only dimmer-than-anchor legs thins patterns of exactly the
bright stars the wide-field top-18 window depends on. The idea's remnant lives in
S5's index layout, not in pattern content.

---

## 2026-08-01 — S2: triangles move onto the Kendall shape sphere

**The invariant is now geometry, not bookkeeping.** `patdb_g12_t17k` stores each
triangle as its point on the Kendall shape sphere — where isotropic centroid noise is
isotropic shape noise — plus a 3-bit permutation code that pairs image and catalogue
vertices positionally, with the pixel↔sky handedness flip absorbed by opposite
chirality conventions (verified against pipeline-projected fields to the curvature
limit). A mirrored field is one reflected query point, so the transpose-and-re-solve
retry is gone from the v2 path: extraction and query are shared, and the mirror pool
is clustered only if the normal one fails.

**Bench vs S1 (docs/bench/BENCH.md): gate passed, 62 → 64/80.** Reliability **32/32**
and scatter **12/12** — the marginal ordering draws the distorted (ratio, dphi)
metric lost are exactly what the uniform metric keeps. `fov_galplane_12°` solves: the
first movement past the documented ~10° ceiling. Solve medians −20–25 %; failures
13.0 → 10.2 s. Junk 8/8, wrong solves 0, real fields 2/2 at 5.2–5.4 s.

**Two findings the staged protocol was built to surface:**

- *Calibration, measured rather than assumed*: over 1609 true triangle pairs across
  23 synthetic fields, true-pair shape distances have median ~0.001 and per-field q99
  of 0.002–0.007, scaling with FOV and noise exactly as the design doc's error budget
  predicts. The dev spike's tolerance of 0.001 was correct *for clean stacked
  fields*; this corpus's 3 px edge distortion sets an equal-stringency radius of
  0.005 — so the big candidate cut (10–50×) is S3's adaptive-tolerance win, not
  S2's. What S2 delivers instead is statistical uniformity, single-pass mirror, and
  the two reliability families going clean.
- *Vertex symmetry meets storage redundancy*: the same physical star triple lives
  under 2–3 anchors in the DB, and the shape rep — unlike (ratio, dphi) — is the same
  from every one of them, so every match arrives in duplicate with an identical
  implied solution: 36,544 raw consensus clusters where S1 had 160. The consensus
  loop now gates on distinct image triples (pure numpy); warm kendall solves went
  11.3 s → 5.2 s, overtaking S1's 6.2 s. Recorded for S5: the dedupe could move into
  the database itself, which is the dimmer-legs question (S2b) seen from the other
  side.

One boundary churn, accepted and assigned: the 8 px-noise case returns to failing
(true-pair distances exceed the calibrated radius; S1's coarse metric kept it by
luck). It is outside v1's documented envelope and awaits S3's noise-adaptive radius.

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

1. **Milestone D — plate-solve robustness.** The staged rebuild S0–S6 is landed:
   Gaia DB, Kendall invariant, calibrated tolerance model, quaternion consensus +
   pole-aware verification, bucket index + progressive anchors, multi-scale FOV
   layers. **v1's 61/80 → 89/96 solvable (0.93); blind envelope 1°–18° in both
   density regimes; poles 4/4; artifact-poisoned fields recovered; DB load 0 s,
   cold solve 3.3 s, warm 1.8 s; zero wrong solves at every stage.** S7
   (quads/pentas): **declined by measurement** — the 4° layer covers 10–18°
   including heavy-distortion wide optics. **v2 is the pipeline default since
   v1.1.0** (automatic fallback to the classic solver on fresh installs).
   Remaining polish candidates: ship the layer DBs as release assets, size-aware
   consensus radius, junk early-abort (~81 s ladder exhaustion on skyless
   fields), the acceptance-threshold floor (sparse-10; corrected p-value
   experiment), the date+distortion degeneracy (a proper-motion prior would
   shrink the blind-date spread), and the v2.1 idea of feeding the fitted
   distortion polynomial back into the solver's tolerance floor. Designed
   in `docs/PLATESOLVER_V2_DESIGN.md`; stage record in `docs/bench/BENCH.md`.
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
