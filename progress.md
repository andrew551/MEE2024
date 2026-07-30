# MEE2024 — development progress log

A running record of what has changed, what has been measured, and what was decided.
Newest first. Design detail lives in `docs/ARCHITECTURE.md` (how the pipeline works) and
`docs/STARCAT_DESIGN.md` (the star-catalogue redesign).

---

## Current state

| | |
|---|---|
| Branch | `refactor/test-cli-foundation` |
| Tests | 290 passing (`pytest` = 276 fast + 14 behind `--runslow`) |
| Lint | pyflakes clean apart from two intentional PyInstaller import shims |
| Pipeline | stages 1–3 all runnable headlessly from the CLI |

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

1. **starcat step 2** — `mee2024/starcat/`: `StarTable`, provider protocol, adapters. Not
   wired up.
2. **step 3** — `TychoOffline`; migrate `platesolve_triangle` off raw arrays.
3. **step 4** — `GaiaOnline` returns `StarTable`; migrate stage 2. Remove the G>3 floor.
4. **step 5** — `GaiaOffline` + builder + Zenodo download.
5. **step 6** — Gaia+Tycho merge and the catalogue-check report.
6. **step 7** — label layer.
7. Then milestones D (plate-solve robustness), B (auto-calibration score), E (centroid rig).

### Unresolved questions

- `safety_limit_mag`: mag-13 stars are detectable in eclipse fields and scientifically
  useful, but Gaia is slow to serve them and the offline archive doubles in size to cover
  them. Proposal: keep the default at 13, build the archive in two magnitude-banded parts
  (G<12 base, 12<G<13 optional), and have the offline provider declare its own depth and
  fall back to online beyond it.
- Whether to unify the half-pixel centroid convention (above).
- What `mee2024/TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_dataB.npz` in the
  working tree is — possibly an earlier wide-field experiment, relevant to milestone D.
