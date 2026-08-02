# Publishing releases

Three kinds of artefact, and only two are ever uploaded:

| artefact | where it comes from | released? |
|---|---|---|
| **Star catalogue** `gaia_dr3_g13` | built once from the Gaia archive | yes, tag `catalogues-v1` |
| **The program** `MEE_2024_v<version>.exe` | `python -m PyInstaller MEE2024.spec` | yes, tag `v<version>` |
| **Plate-solving pattern databases** | derived on the user's own machine | **no — see Part 3** |

Catalogues and software are separate releases because the catalogues change far less often
than the code: pinning them together would mean re-uploading hundreds of megabytes for
every patch, and would break the download URLs compiled into older builds.

---

# Part 1 — the star catalogue

The application already knows this URL, and `mee2024/starcat/download.py` must agree with
it exactly or the download silently 404s:

```
https://github.com/andrew551/MEE2024/releases/download/catalogues-v1/gaia_dr3_g13.zip
```

Three things must match: the repository is **`andrew551/MEE2024`** and **public**, the tag
is **`catalogues-v1`**, and the asset is **`gaia_dr3_g13.zip`**.

## Step 1 — pack the archive and record its identity

```bash
mee2024 catalogue --pack gaia_dr3_g13
```

This writes `gaia_dr3_g13.zip` and prints its size and SHA-256, which go into the
`gaia_dr3_g13` entry of `RELEASES` in `mee2024/starcat/download.py`.

**Already done for the current archive** — the registry now carries:

```
size_bytes  319_719_061                                                       (320 MB)
sha256      897e6bc2ef32a4faf04c9294a48dde3318fd43edc6bd2041581a2cffc66453f0
url         _github_asset('gaia_dr3_g13.zip')
```

Repack only if the archive itself changes; then update both numbers, because the
application refuses any download whose hash does not match. **Do not rename, re-zip or
re-compress the file** — zip output is not byte-reproducible, so a second `--pack` of the
same data yields a different hash. Upload exactly the file `--pack` produced.

## Step 2 — upload

Needs the GitHub CLI (<https://cli.github.com/>, one-off `gh auth login`). The
`catalogues-v1` release already holds the two superseded archives, so this adds to it
rather than replacing it, and older builds keep working:

```bash
gh release upload catalogues-v1 gaia_dr3_g13.zip --repo andrew551/MEE2024
```

If the tag does not exist yet, create it in one step instead:

```bash
gh release create catalogues-v1 gaia_dr3_g13.zip --repo andrew551/MEE2024 --title "Star catalogues" --notes "Offline Gaia DR3 catalogue for MEE2024: G < 13, 7,369,627 stars, with double-star neighbour flags computed across the whole archive. Install with: mee2024 catalogue --fetch gaia_dr3_g13. Derived from ESA/Gaia/DPAC data."
```

**Never pass `--draft`.** A draft release is invisible to anonymous downloads, so the app
gets a 404 while the release looks fine in your browser. To stage privately, use `--draft`
and then `gh release edit catalogues-v1 --draft=false` when ready.

## Step 3 — verify, without downloading a third of a gigabyte

```bash
mee2024 catalogue --check-remote
```

`[OK  ] gaia_dr3_g13 ... reachable, <size> MB` is the goal. `[skip]` means no URL is
configured yet (step 1 was missed). `FAIL ... 404` means the tag, repository or asset name
does not match, or the release is still a draft. A **size mismatch** means a different or
re-compressed file was uploaded and the recorded hash no longer applies.

Then confirm a real download on a machine that does not already have it — it verifies the
SHA-256, unpacks, and checks every column against the manifest before accepting:

```bash
mee2024 catalogue --fetch gaia_dr3_g13
```

## The other tiers

`gaia_dr3_g10` (24 MB) is **bundled inside the executable** rather than released — Part 2.

`gaia_dr3_g15` is a placeholder for a deep archive that does not exist yet. Build it in a
terminal of its own:

```bash
python tools/build_gaia_offline.py --name gaia_dr3_g15 --max-mag 15
```

Then follow Part 1 to publish it. What to expect:

* **~1.6 GB to download, ~1.8 GB installed**, 36.9 M stars.
* **Between two hours and several days.** Gaia's throughput has been measured to swing
  50× between days and no query shape changes it (`progress.md`); the builder prints a
  running `rows/s · elapsed · ETA` per chunk, revised from what it actually measures, so
  believe that rather than the up-front floor it prints first.
* **Interrupting it is safe.** Every chunk is cached under
  `<catalogue root>/.build_gaia_dr3_g15/stripe_<band>_<part>.npy` and skipped on the next
  run, and each is written to a `.part` file and renamed, so a kill mid-write leaves no
  half-file to be mistaken for a complete one. Ctrl-C and restart the same command as
  often as you like; only the assembly at the end has to run start to finish.
* **Peak memory ~6–8 GB** in that final assembly: every chunk is concatenated and a
  36.9 M-point KD-tree is built for the double-star neighbour flags. On a smaller machine,
  build it in declination halves with `--region` and merge.

The superseded `gaia_dr3_g12` and `gaia_dr3_g12_13` assets stay published so existing
installations keep working and `mee2024 catalogue --merge` still has a pair to merge — but
they are **no longer offered in the app** (`CatalogueRelease.offered`), and the first-use
download no longer falls back to them: handing someone a G<12 archive when they asked for
the standard one is worse than reporting the failure.

---

# Part 2 — the program

```bash
python -m PyInstaller MEE2024.spec --noconfirm
```

Run it **from the repository root** — the code uses absolute `from mee2024 import ...`
imports, so the root must be on the path. Produces `dist/MEE_2024_v<version>.exe`, one
file, no Python needed on the target machine. The filename follows `_version()` in
`mee2024/MEE2024util.py`, so bump that (and `setup.cfg`) first.

Built and tested with Python 3.9; the full suite also passes on newer interpreters.

**The build bundles the compact star catalogue** so a fresh install plate-solves offline
immediately. The spec looks for `gaia_dr3_g10` in the build machine's catalogue directory
(or `mee2024/resources/catalogues/`) and prints which it used:

```
spec: bundling gaia_dr3_g10 from C:\Users\...\MEE2024\catalogues\gaia_dr3_g10
```

If it prints `not found` the exe still works, but every first run needs a download. Create
the tier in about a second from any deeper installed catalogue:

```bash
python -c "from mee2024.starcat import download; download.build_compact_tier()"
```

It is deliberately **not** in source control: 24 MB of generated data that any machine with
a deeper archive can reproduce.

## Check before shipping

```bash
dist/MEE_2024_v1.2.0.exe --version
```
```bash
dist/MEE_2024_v1.2.0.exe catalogue
```

The second must list `gaia_dr3_g10` as installed — that proves the bundle arrived *and*
that the runtime finds it inside the archive. Then double-click it: the app window opens
(the default since v1.0.0), and `MEE_2024_v1.2.0.exe gui` still opens the classic
interface. Run a small dataset through it and confirm the plate solve succeeds on a machine
with no catalogue in its data directory — that is the whole point of the bundle.

## Publish

```bash
gh release create v1.2.0 "dist/MEE_2024_v1.2.0.exe" --repo andrew551/MEE2024 --title "MEE2024 v1.2.0" --notes "Windows executable, no Python installation required. Double-click for the app window, or run it from a terminal for the command line; the classic interface is still available with: MEE_2024_v1.2.0.exe gui. This build bundles the compact Gaia catalogue (G < 10) and so plate-solves offline immediately -- fetch gaia_dr3_g13 for fainter stars. New: a rebuilt plate solver that solves blind from 1 to 18 degrees at about a second a field, one standard star catalogue instead of two overlapping ones, native file dialogs, and the plate-solving database is built automatically on first use."
```

---

# Part 3 — why the pattern databases are not released

The plate solver reads pattern databases, and it would be natural to publish them beside
the catalogues. They should not be, for a reason worth stating plainly: **they are derived
data.** Every byte is computable from the star catalogue the user already has, so
publishing them means shipping hundreds of megabytes a machine can produce in seconds, and
then keeping those artefacts in step with every change to the builder.

Measured on this machine:

| built from | legs per anchor | triangles | size | build time | solves |
|---|---|---|---|---|---|
| `gaia_dr3_g10` | 8 | 3.2 M | 69 MB | **19 s** | both real fields, 6 of 7 spot cases, 0.1–0.3 s per solve |
| `gaia_dr3_g13` | 18 | 17.2 M | 230 MB | ~3 min | the full measured envelope (`docs/bench/BENCH.md`) |

So the program builds one on first use instead. `platesolve2.ensure_pattern_db()` runs
before the first solve, picks the small fast build from the compact catalogue or the full
one from a deeper archive, and reports progress like any other pipeline stage. The user is
told what is happening and waits twenty seconds, once — no download, no instructions,
nothing to choose. Installing a deeper catalogue later earns a better database next time.

If that wait ever proves unpopular, publishing prebuilt databases is a small change: add
entries to `RELEASES` with `kind='patterndb'` and follow Part 1. Do it only with a
measurement in hand — a 230 MB download to save three minutes of local computation is a
poor trade, and a 69 MB download to save twenty seconds is a worse one.

---

# Moving to Zenodo later

Zenodo gives a citable DOI, which matters once results are published. The switch is a
two-line edit per catalogue in `RELEASES` — replace `url` and fill in `doi`. The `sha256`
values stay the same as long as the same files are uploaded, so existing installations are
unaffected. Users can also override the source with no new software release:

```bash
mee2024 catalogue --set-source gaia_dr3_g13 --url URL --sha256 HASH
```

## Attribution

Gaia data is freely redistributable with credit to **ESA/Gaia/DPAC**. The bundled
Hipparcos-derived catalogue and star-label index come from the Hipparcos-2 reduction
(van Leeuwen 2007) and Gaia's own `hipparcos2_best_neighbour` crossmatch. Each catalogue
records its sources in its `manifest.json` provenance field.

---

# After any release

Update the **Current state** table at the top of `progress.md` (version, what is
published) and add an entry describing what changed.
