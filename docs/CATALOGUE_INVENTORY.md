# Catalogue inventory and unification plan

Written 2026-07-30. Every number here is measured, not estimated.

## 1. What we now hold

| # | artefact | rows | size | location | native epoch | band | role |
|---|---|---|---|---|---|---|---|
| 1 | `compressed_tycho2024epoch.npz` (Tycho-2) | 1,034,887 | 8.1 MB | bundled in wheel | **2024.0, pre-propagated, no PM stored** | V | triangle-DB source; plate-solve verification |
| 2 | `TripleTriangle_pattern_data.npz` | ~171k anchors × 153 triangles | 169 MB | user data dir, generated on first run | derived | — | lost-in-space plate solving |
| 3 | `hipparcos2` | 117,955 | 5.9 MB | bundled in wheel | 1991.25 + PM | `G_est_from_Hp` | bright fill; precision-grade |
| 4 | `star_labels` | 117,955 HIP, 99,525 Gaia crossmatches, 50 names | 2.1 MB | bundled in wheel | — | — | readable plot labels |
| 5 | `gaia_dr3_g12` | 3,087,821 | 154 MB | user data dir, downloaded/built | 2016.0 + PM | G | **primary precision catalogue** |
| 6 | `gaia_dr3_g12_13` | 4,281,806 | 213 MB | user data dir, optional | 2016.0 + PM | G | deep extension for eclipse fields |
| 7 | `gaia_test_g5`, `gaia_test_zwo` | 2,168 / 4,585 | 0.4 MB | user data dir | 2016.0 | G | build-pipeline test fixtures |

Bundled in the wheel: **16.1 MB**. Optional downloads: **154 MB** base, **213 MB** deep.
Stacked, 5+6 give 7,369,627 stars to G<13, matching the archive count exactly.

## 2. Measured coverage and accuracy

### Bright end (the reason Hipparcos was added)

Gaia DR3 has **no entry at all** for the brightest stars — they saturate the instrument.
With no magnitude filter, the brightest Gaia source within 0.02° of Sirius is G=8.52; of
Vega, G=14.58; of Canopus, G=19.13 — all unrelated field stars, with zero NULL photometry.

- **18,430** Hipparcos-2 stars have no Gaia DR3 counterpart; **2,629** of those are
  brighter than Hp=7.
- The bundled Tycho does not close it either: it has Arcturus and Canopus but **not Sirius
  or Vega**, and only 8 entries brighter than V=1 where the sky has ~15. Tycho-2 moves
  ~120 very bright stars into a separate *Supplement 1* file, which is not what this npz
  was built from.
- Hipparcos-2 has all of them, with proper motions and parallaxes.

### What each fill actually contributes

Measured over 14 random 5°×5° boxes (350 sq deg, 19,129 Gaia stars):

| fill source | stars added |
|---|---|
| Hipparcos (tried first) | **5** |
| Tycho, *after* Hipparcos | **1** |
| Tycho alone, without Hipparcos | 5 |

So the two overlap almost entirely, and **Tycho's marginal value as a bright fill is about
1 star per 350 sq deg** — roughly 120 all-sky. Hipparcos does essentially the whole job.

### Positional accuracy by source

| source | accuracy | fit for the precision fit? |
|---|---|---|
| Gaia DR3 | sub-mas | yes |
| Hipparcos-2 | ~1 mas at epoch, propagated | yes |
| Tycho (bundled npz) | 0.135″ at V<8 → **2.46″ at V=11–11.5** | **no** |

Tycho's degradation is random, not a systematic offset, and grows with magnitude — the
signature of Tycho-2 proper-motion error amplified over the 32.75-year propagation from
1991.25 to the frozen 2024 epoch.

`StarTable.is_precision_grade()` encodes this, and `distortion_fitter` now enforces it, so
a Tycho star can reach the plate solver but never the distortion fit.

### Magnitude systems, unified

Three native bands (V, Hp, G) are all mapped to approximate Gaia G, with the `band`
attribute recording which transformation was used so nothing is mistaken for a measured G:

- Tycho V → G: measured median offset **−0.15 mag**.
- Hipparcos Hp + B−V → G: quadratic fitted on the 96,767 stars with both, **robust σ =
  0.038 mag** (plain rms 0.269, the tail being variables and binaries Gaia resolved but
  Hipparcos did not). A cubic was tried and overfits the blue end.
- A merged table reports `band='G_mixed'`.

Magnitude is used only for limiting cuts, brightness ordering and plot labels, so 0.04 mag
is far better than needed — but the labelling matters so nobody treats it as photometry.

## 3. Where the fragmentation still is

Three storage formats coexist:

1. **`database_lookup2` npz** — a `(N,3)` array of `[ra_rad, dec_rad, mag]`, expanded on
   load into an `(N,6)` table with unit vectors. Loaded **entirely into memory** on every
   run. Only Tycho uses it.
2. **starcat directory** — 11 columns of `.npy`, memory-mapped, declination-sorted, with a
   manifest and per-column SHA-256. Gaia and Hipparcos use it.
3. **triangle pattern npz** — anchors, patterns and triangle shape invariants, plus a
   KD-tree built at load.

Format 3 is a *derived index*, not a star list, and should stay separate. Format 1 is pure
legacy: one catalogue, one reader, one full-memory load.

Identifier namespaces share a single `int64` column, disambiguated only by `origin`: Gaia
`source_id` (~10¹⁸), HIP (<120,000), packed TYC (<10⁹). This works, and `label_for` always
takes `origin`, but a caller that ignored `origin` could mistake HIP 32349 for a Gaia id.

## 4. Recommendations, ranked by value

**R1 — Migrate the bundled Tycho to the starcat format.** Retires format 1 and its
whole-catalogue memory load, leaving exactly one reader for star lists. Prerequisite for
step 3 of the starcat plan (moving `platesolve_triangle` off raw arrays).

**R2 — Make `merged_offline` the default catalogue once step 4 lands.** This is the
user-visible win: it closes the bright-star gap, works with no network, and is already
verified to change nothing on ordinary fields (identical RMS, star count and `nn_corr` on
both example fields, because the merge correctly adds nothing where Gaia is complete).

**R3 — Publish to Zenodo.** The two Gaia archives currently exist only on this machine.
Everything except the URL and hash is wired up.

**R4 — Rebuilding Tycho with proper motions at its native epoch: LOW priority.** My first
instinct was that this mattered, because freezing positions at 2024 bakes in up to ~150 mas
of avoidable error for a 2017 observation. On inspection it does not matter, because Tycho
is only used where that error is irrelevant:

- the **triangle database** keys on *shape* invariants (side ratio, subtended angle) with a
  1% tolerance over 1.7° patterns — 0.6″ of proper motion is ~10⁻⁴ of the pattern scale;
- **plate-solve verification** matches at `rough_match_threshhold` = 36″;
- the **precision fit** now excludes Tycho outright.

So the frozen epoch is harmless in every path Tycho is actually used in. Worth doing only
if Tycho is ever promoted to a role that needs accuracy — and Hipparcos has taken that role.

**R5 — Consider dropping Tycho from the merge fill chain.** It contributes ~1 star per
350 sq deg once Hipparcos is tried first. Keeping it costs almost nothing and it is
already fenced off from the fit, so this is a tidiness question, not a correctness one.

## 5. The unified picture

```
star lists  ->  one format (starcat directory: mmap, dec-sorted, manifest+checksums)
                one type   (StarTable: float64 positions, per-star NaN, per-star origin)
                one interface (CatalogueProvider.lookup / lookup_neighbours)

derived index -> triangle pattern database, separate by nature, generated from Tycho

priority order for a lookup:
    1. Gaia DR3        precision astrometry, the workhorse        G < 12 (+ optional 13)
    2. Hipparcos-2     bright fill, precision-grade, labels       G < 9
    3. Tycho-2         plate solving only, never the fit          G < 9, marginal

magnitudes -> all approximate Gaia G; `band` records the provenance
identifiers -> int64, meaning determined by `origin`; resolve through LabelIndex
epochs     -> every source stores its native epoch plus proper motion, and propagates at
              query time. Verified to reproduce Gaia's server-side propagation to 0.000 mas
```
