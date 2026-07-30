# Star catalogue redesign — offline Gaia, unified star data, Gaia+Tycho merge

Status: **design, not yet implemented**. Written 2026-07-29 against commit `4bacfc4`.

Goal: a single star-data abstraction, an offline Gaia catalogue that auto-downloads on
first use, and a merged Gaia+Tycho catalogue that fixes the bright-end gap — without
changing any number the pipeline currently produces except deliberately.

---

## 1. Where we are (measured, not assumed)

### 1.1 Stage 2 works, and the guess-date feature works remarkably well

Run on the two supplied ZWO zenith fields. `guess_date` starts from 2020-01-01 and is
told nothing about the observation; the true date comes from the FITS `DATE-OBS`
(**2023-10-29** for both).

| field | FOV | order | RMS | stars | guessed date | error | nn corr |
|---|---|---|---|---|---|---|---|
| zwo3 | 2.4×1.8° | cubic | 113.0 mas | 433 | 2023-11-02 | **+4 d** | 0.386 |
| zwo3 | | quintic | 109.6 mas | 434 | 2023-10-28 | **−1 d** | 0.166 |
| zwo3 | | septic | 107.2 mas | 434 | 2023-10-27 | **−2 d** | 0.132 |
| zwo1 | 4.9×3.3° | cubic | 525.2 mas | 1396 | 2023-03-19 | −224 d | 0.921 |
| zwo1 | | quintic | 111.9 mas | 1564 | 2023-09-06 | −54 d | 0.353 |
| zwo1 | | septic | 100.9 mas | 1564 | 2023-09-20 | −39 d | 0.191 |

Conclusions:

- **The pipeline is intact.** ~100–113 mas RMS on 3-second exposures, and blind date
  recovery to one day on the narrow field.
- **RMS alone does not predict date accuracy.** zwo1/quintic and zwo3/quintic have
  almost identical RMS (112 vs 110 mas) but date errors of 54 days vs 1 day. What tracks
  date accuracy is the **nearest-neighbour error correlation**: 0.13–0.17 gives days,
  0.35 gives weeks, 0.92 gives months. Spatially correlated residuals are unmodelled
  distortion, and distortion can mimic the coherent pattern that proper motion imprints.
  `nn_corr` is therefore the right headline diagnostic for the milestone-B quality score,
  and it is already computed.
- **The wide field needs a higher order.** Cubic leaves 525 mas and `nn_corr`=0.92 on
  zwo1. Any auto-calibration mode should pick the order from `nn_corr`, not from RMS.

### 1.2 The bundled Tycho positions degrade steeply with magnitude

Crossmatch of the bundled `compressed_tycho2024epoch.npz` against Gaia DR3, both at
epoch 2024, over the zwo3 field (255 Tycho vs 1451 Gaia sources):

| Tycho V | n | median separation |
|---|---|---|
| 4–8 | 10 | 0.135″ |
| 8–9 | 25 | 0.312″ |
| 9–10 | 41 | 0.637″ |
| 10–10.5 | 44 | 1.465″ |
| 10.5–11 | 85 | 2.094″ |
| 11–11.5 | 44 | 2.464″ |

There is **no systematic offset** — for the well-matched subset the mean offset is
+0.03″ in RA·cos(dec) and +0.11″ in Dec, i.e. no coordinate-frame or epoch bug. It is
random scatter growing with magnitude, the signature of Tycho-2 proper-motion error
amplified by the 32.75-year propagation from 1991.25 to 2024. (I could not isolate the
root cause further without the original `tyc_main.dat`, and the published Tycho-2 PM
errors look too small to explain 2″ — worth revisiting, but it does not change the design
decision below.)

**This is decisive for the merge design:** Tycho is fine for plate solving (36″ match
tolerance, statistical acceptance test) and unusable for the precision fit (we are
measuring 0.35″ deflections at 0.1″ RMS). So:

> Tycho stars may enter the catalogue **for plate-solving completeness only**. They must
> never reach the distortion or deflection fit. This requires **per-star provenance**,
> which is precisely what today's single `has_pm` table-level flag cannot express.

### 1.3 The bright-end gap is partly our own query

`gaia_search.select_in_box` hard-codes `phot_g_mean_mag BETWEEN 3 AND {max_mag}`.

- **150** Gaia DR3 sources have G < 3 and are silently dropped by that lower bound.
- **5,455,339** sources have no `phot_g_mean_mag` at all; a SQL `BETWEEN` excludes NULLs,
  so those are dropped too. Some are genuinely bright stars with unusable photometry.

So "Gaia misses the brightest stars" is two separate effects: a real catalogue gap, and a
self-inflicted query floor. The floor should just be removed.

### 1.4 Offline catalogue size

| depth | Gaia DR3 sources | raw at 48 bytes/star |
|---|---|---|
| G < 11 | ~1.2 M | ~58 MB |
| G < 12 | ~2.9 M | ~139 MB |
| **G < 13** | **7,369,627** | **~354 MB** |

Of the G<13 set, **58,097 (0.8%)** have no proper motion and no parallax — again per-star,
not per-table.

---

## 2. The unified star table

Replace `StarData` (a Gaia-result-table wrapper) and the raw `(N,6)` float array that
`database_lookup2` returns with one columnar type.

```python
# mee2024/starcat/table.py
class StarTable:
    ra        : (N,)  float64   radians, ICRS, valid at self.epoch
    dec       : (N,)  float64   radians
    mag       : (N,)  float32   see band
    band      : str             'G' | 'V' | 'G_est_from_V'
    ids       : (N,)  int64     Gaia source_id, or encoded TYC1-TYC2-TYC3
    origin    : (N,)  uint8     ORIGIN_GAIA | ORIGIN_TYCHO
    epoch     : float           Julian year
    pmra      : (N,)  float32   mas/yr, already *cos(dec); NaN if unknown
    pmdec     : (N,)  float32   mas/yr; NaN if unknown
    parallax  : (N,)  float32   mas; NaN if unknown
    nn_sep    : (N,)  float32   arcsec to nearest catalogue neighbour; NaN if not computed
    nn_mag    : (N,)  float32   that neighbour's magnitude
    vectors   : (N,3) float64   unit vectors, cached, invalidated on epoch change
```

Design choices and why:

- **Columnar numpy, not an astropy `Table` or `SkyCoord`, as the source of truth.**
  `SkyCoord` construction is expensive and it cannot be memory-mapped or sliced cheaply,
  yet `select_indices` is called repeatedly and the offline DB needs mmap. `SkyCoord` is
  built lazily and cached only when epoch propagation or AltAz conversion needs it.
- **`float64` for ra/dec.** `float32` gives ~0.1″ resolution at these coordinates —
  larger than the signal. Non-negotiable.
- **NaN for "unknown", per star.** Fixes the `has_pm` table-level flag, which is wrong
  the moment Gaia and Tycho are in the same table, and wrong today for the 0.8% of Gaia
  stars with no PM.
- **`origin` per star.** Lets the distortion fitter reject Tycho-only stars (§1.2).
- **Immutable-style operations.** `at_epoch()` and `select()` return new tables;
  `update_epoch()` and `select_indices()` stay as thin in-place wrappers so
  `distortion_fitter` and `gravity_sweep` keep working during migration. The current
  in-place `update_epoch` combined with a `__copy__` that shares array references is a
  latent aliasing trap (and is how the `epch` typo went unnoticed).

Accessors keep their present names — `get_ra()`, `get_dec()`, `get_ra_dec()`,
`get_vectors()`, `get_mags()`, `get_ids()`, `get_pmotion()`, `get_parallax()`,
`nstars()` — so callers do not churn. New: `has_proper_motion()`, `is_gaia()`,
`select()`, `at_epoch()`, `concat()`, `to_dir()`, `from_dir()`.

## 3. Provider protocol

One interface behind which online, offline, Tycho and merged catalogues are
interchangeable:

```python
# mee2024/starcat/providers.py
class CatalogueProvider(Protocol):
    name: str
    is_offline: bool
    def lookup(self, ra_range, dec_range, max_magnitude, epoch) -> StarTable: ...
    def lookup_neighbours(self, table, radius_arcsec, max_magnitude) -> StarTable: ...
```

| implementation | notes |
|---|---|
| `GaiaOnline` | today's `dbs_gaia`, minus the `BETWEEN 3` floor |
| `GaiaOffline` | the downloaded catalogue; propagates epochs locally |
| `TychoOffline` | wraps the bundled npz; `origin=ORIGIN_TYCHO`, no PM |
| `Merged(primary, secondary, ...)` | Gaia first, Tycho only to fill the bright end |
| `CachedProvider(path)` | replays a saved response — the test fixture (§6) |

`database_cache.open_catalogue` becomes a provider registry keyed on a catalogue name
(`'gaia'`, `'gaia_offline'`, `'tycho'`, `'merged'`), preserving its existing cache role.

## 4. Offline catalogue on disk

A **directory of uncompressed `.npy` files opened with `mmap_mode='r'`**, not a single
`.npz`.

```
<user_data_dir>/MEE2024/catalogues/gaia_dr3_g13_v1/
    manifest.json      format version, provenance, epoch, mag limit, star count, SHA-256 per file
    ra.npy dec.npy     float64, sorted by declination
    pmra.npy pmdec.npy parallax.npy mag.npy   float32
    source_id.npy      int64
    nn_sep.npy nn_mag.npy                     float32
    dec_index.npy      int64, 181 entries: first row of each 1-degree declination band
```

- **Why not `.npz`:** a compressed archive must be fully decompressed to read one star.
  354 MB per stage-2 run is unacceptable. Memory-mapped `.npy` reads only the pages
  touched.
- **Why dec-sorted with a 1-degree band index:** every query the pipeline makes is a
  bounding box (`get_bbox` → ra_range, dec_range). A dec band index turns that into a
  contiguous slice plus an RA filter — a few hundred KB read instead of 354 MB. No
  HEALPix dependency, and RA wrap-around needs no special case because it only affects
  the in-memory filter, exactly as `database_lookup2.lookup_objects` already handles it.
- **Distribution:** the directory zipped. Downloaded once, unpacked to `user_data_dir`,
  every file hash-verified against the manifest. Same location pattern as the existing
  `get_triangle_db_path()`; add `get_catalogue_dir(name)`.
- **Builder** (`tools/build_gaia_offline.py`, not shipped in the wheel): ADQL in
  declination stripes, resumable — each chunk is cached as an `.npy` keyed on its stripe
  band, so an interrupted build resumes exactly where it stopped.

### Measured build cost, and why the chunking looks the way it does

**Gaia async query latency dominates: ~18–28 s per query regardless of how little it
returns.** A mag-5 all-sky build (2168 stars, 18 queries) took 318 s — essentially all
latency. Two consequences shaped the design:

- **Prefer few large stripes.** The default declination step is **10°**, not 1°. A 1°
  step would mean 180 stripes and turn a 15-minute build into a multi-hour one for no
  benefit. Stripes denser than `ROWS_PER_QUERY` (200 000) are split in RA automatically.
- **Count with a single `GROUP BY` query**, not one per stripe. Counting stripe by stripe
  would have doubled the total query count.

Real sizing for the all-sky G<12 build, from the counting query:

| | |
|---|---|
| stars with G < 12 | **3,087,821** |
| populated 10° declination stripes | 18 |
| densest stripe | 252,730 rows |
| queries required | **29** |
| wall time | **10–22 min** |
| on-disk size | **148 MB** |

This is much cheaper than the "hours" originally assumed.

### Recommended build sequence

Three scales, each validating more than the last:

1. **Plumbing** — `--max-mag 5`, all sky. ~2000 stars, minutes. Proves query, chunk,
   assemble, neighbour flags, write, verify.
2. **Real pipeline test** — `--max-mag 12 --region ...` around known fields. Full depth
   over a small sky area; one query, ~30 s. This is the artefact that lets stage 2 run
   genuinely offline against real data.
3. **The artefact** — all sky, `--max-mag 12`.

### Double-star flagging: a real difference offline

`nn_sep`/`nn_mag` are computed among the catalogue's own members with a KD-tree, which is
free. A companion **fainter than the catalogue limit is therefore not flagged**, whereas
the online path queried to G<17 (`double_star_mag`). In the example fields this changes
nothing, because `remove_double_tab2` defaults to false and the flag is only recorded —
but it is a genuine reduction in sensitivity, and the manifest records the depth the flags
actually cover. A `--neighbour-depth` option exists for a deeper (much slower) pass.

Photometric reasoning for how deep is worth going: a companion at Δm shifts the centroid
by roughly `10^(-0.4Δm)` of the separation. At 10″ separation a G=14 companion beside a
G=10 star shifts it ~250 mas — significant. A G=17 companion shifts it ~16 mas —
negligible against our 100 mas. So neighbours to about `limit + 4` mag capture what
matters; a flat G<17 was more than needed.

## 4a. Bright-star label layer

Gaia source ids are 19 digits and unreadable on a plot. HIP numbers and proper names are
what people want to see — but only a tiny fraction of stars have either, so putting the
columns in the main catalogue would mean a 2.9 M-row field that is >99% empty.

Instead: a **separate, tiny, sorted-key side table**, structured like the platesolve
database — plain binary arrays, memory-mapped, no Python-level dictionary.

```
mee2024/resources/star_labels_v1/          (bundled in the wheel, ~2 MB)
    manifest.json     format version, provenance, source catalogues, row count
    gaia_id.npy       int64  sorted ascending   <- the lookup key
    hip.npy           int32  HIP number, 0 = none
    name_offset.npy   int32  byte offset into names.txt, -1 = unnamed
    tyc_id.npy        int64  sorted ascending   <- second key, for Tycho-origin stars
    tyc_hip.npy       int32
    names.txt         UTF-8, newline-separated proper names
```

**Sorted array + `np.searchsorted`, not a hash table.** It is smaller (a hash table needs
30–50% empty slack), it memory-maps directly, it needs no load/rehash step, and one
`searchsorted` call resolves an entire field's worth of ids at once. That is the
"hashtable-like layer" without the hashtable's overhead.

Scope and size:

| scope | rows | size |
|---|---|---|
| IAU proper names only (IAU-CSN) | ~450 | ~12 KB |
| naked-eye stars, V < 6.5 | ~9,100 | ~150 KB |
| **full Hipparcos crossmatch** | **118,218** | **~1.9 MB** |

Recommend the **full HIP crossmatch**: at 1.9 MB it is a quarter of the Tycho catalogue
already bundled, and HIP's completeness limit (V≈9) coincides exactly with the V<9 Tycho
fill limit chosen in §8 — so every Tycho-origin star we add for plate solving can carry a
readable HIP label. It is small enough to **ship in the wheel** rather than in the
downloaded archive, so labels work even before any offline catalogue is fetched, and it
can be revised independently of the 139 MB main download.

Built from authoritative precomputed crossmatches, not our own positional matching:
`gaiadr3.hipparcos2_best_neighbour` (source_id ↔ HIP) and
`gaiadr3.tycho2tdsc_merge_best_neighbour` (source_id ↔ TYC), with proper names from the
IAU Catalog of Star Names keyed on HIP.

```python
# mee2024/starcat/labels.py
class LabelIndex:
    def hip_for(self, ids, origin) -> np.ndarray          # int32, 0 where unknown
    def name_for(self, ids, origin) -> list[str | None]
    def label_for(self, ids, origin) -> list[str]         # 'Vega' > 'HIP 91262' > 'gaia:2...'
```

`label_for` falls back down that chain, so it is always safe to call. Three existing
annotation sites become readable through it: `distortion_fitter`'s `ID` column and its
rough-fit plot, and `eclipse_analysis`, which currently strips the `gaia:` prefix by
slicing `id_i[5:]`.

## 5. Merge policy

```
for each Tycho star:
    if a Gaia source lies within 2":   drop the Tycho entry (Gaia wins)
    else if V < BRIGHT_FILL_LIMIT:     keep it, origin=ORIGIN_TYCHO, band='G_est_from_V'
    else:                              drop it (Tycho is too imprecise to help, §1.2)
```

- Magnitudes are transformed to an approximate G with the published Gaia EDR3 relation
  `G − V ≈ −0.01760 − 0.006860(B−V) − 0.1732(B−V)²`; without a colour we use the measured
  median offset of **−0.15 mag** and record `band='G_est_from_V'` so nothing mistakes it
  for a real G. Magnitude is only used for limiting cuts, plot labels and brightness
  ordering, so approximate is fine — but it must be labelled.
- `distortion_fitter` filters to `is_gaia()` before fitting. Tycho stars reach the plate
  solver and nothing else.

### Gaia-vs-Tycho sanity report

`mee2024 catalogue-check --ra-range .. --dec-range ..` reproduces §1.2 on demand:
counts per magnitude bin, crossmatch rate, positional agreement (separation and its RA/Dec
decomposition, vs magnitude), and magnitude agreement after the colour transform. Run as a
`slow`+`network` test over a few sky regions so a catalogue regression is caught.

## 6. Regression fixtures to lock in *before* refactoring

This is the first work item, because it is what makes the rest safe.

Cache the Gaia response for each example field **at ref_epoch 2016.0, unpropagated**
(~600 rows, a few tens of KB) into `tests/data/gaia/<field>.npy`. That single artefact
serves three purposes:

1. A **fast, offline, deterministic stage-2 regression test** — assert the RMS, star
   count and guessed date from the table in §1.1, with no network.
2. A **miniature offline catalogue** to exercise `GaiaOffline` and the dec-band index.
3. The **epoch-propagation gate**: comparing a locally propagated fixture against the
   online `ESDC_EPOCH_PROP_POS` result measures the one number that decides whether the
   offline path is acceptable.

### Epoch-propagation gate: **PASSED**

This was the main technical risk. Today Gaia propagates positions server-side with
`ESDC_EPOCH_PROP_POS`; offline we must do it locally with astropy `apply_space_motion`.
Measured over both fields for the 2016.0 → 2023.84 baseline
(`tools/capture_gaia_fixture.py`):

| field | stars | median | p90 | max |
|---|---|---|---|---|
| zwo1 | 1808 | 0.000 mas | 0.000 | **0.000** |
| zwo3 | 586 | 0.000 mas | 0.000 | **0.000** |

Local propagation reproduces the server-side result **exactly**, below the 0.001 mas
printing precision, for all 2394 stars. Clamping parallax to ≥1 mas (as `StarData` does
today) makes no difference. Dropping `radial_velocity` costs at most 0.019 mas (zwo1) and
2.454 mas (zwo3) — so RV is worth the 4 bytes/star (+12 MB at G<12), but even without it
the error is 40× below our 100 mas RMS.

**Conclusion: the offline path loses no accuracy.** Carry `radial_velocity`.

> An earlier run of this measurement reported a 2266 mas outlier. That was a bug in the
> capture tool — `source_id` was routed through `float64`, which cannot represent a
> 19-digit Gaia identifier, so the comparison paired the wrong stars. Worth remembering:
> **Gaia source_ids must never touch a float.**

## 7. Migration order, with a verification gate at each step

Each step must leave the suite green and reproduce §1.1 before the next begins.

1. **Fixtures + propagation gate** (§6). No production code changes.
2. **`mee2024/starcat/`**: `StarTable`, providers, adapters, unit tests. Nothing wired up.
3. **`TychoOffline` + migrate `platesolve_triangle`** off raw arrays. This changes the
   `solution['matched_stars']` contract that `do_stack` indexes positionally — the field
   fixtures in `tests/data/fields/` must still solve to identical RA/Dec/roll/platescale.
4. **`GaiaOnline` returns `StarTable`**; migrate `distortion_fitter`, `gravity_sweep`,
   `refraction_correction`. Gate: reproduce §1.1 exactly.
5. **`GaiaOffline` + builder**. Gate: offline vs online agree per §6 on both fields.
6. **`Merged` + the catalogue-check report**. Gate: plate-solve success on the bright-star
   field the merge is meant to fix.
7. **Label layer** (§4a) and its builder, plus wiring into the three annotation sites.
   Independent of steps 3–6; can be done at any point after step 2.
8. **UI/CLI catalogue selector + auto-download** with hash verification and a progress bar.

Removing the `BETWEEN 3` floor (§1.3) is a one-line change that belongs in step 4, and it
changes results on fields containing a G<3 star — worth calling out in its own commit.

---

## 8. Decisions taken (2026-07-29)

1. **Depth: G < 12** — ~2.9 M stars, ~139 MB. This still exceeds the default
   `max_star_mag_dist` of 12, so it covers every normal run. `safety_limit_mag` should
   drop from 13 to 12 for the offline provider, or deeper queries fall back to online.
2. **Hosting: Zenodo** — citable DOI and permanent versioned archiving, appropriate for an
   artefact underpinning published results. The manifest hash remains the integrity check.
3. **Tycho fill limit: V < 9** — median position error ~0.3″, far inside the 36″
   plate-solve tolerance, and roughly doubles the bright-star fill compared with V < 8.
4. **Name: `StarTable`**, retiring `StarData` once callers have moved. The semantics
   change enough (per-star provenance, NaN-for-unknown, immutable operations) that a new
   name keeps the migration auditable.
