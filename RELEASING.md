# Publishing releases

Three kinds of artefact, and only two are ever uploaded:

| artefact | where it comes from | released? |
|---|---|---|
| **Star catalogue** `gaia_dr3_g13` | built once from the Gaia archive | yes, tag `catalogues-v1` |
| **The program** `MEE_v<version>.exe` | `python -m PyInstaller MEE2024.spec` | yes, tag `v<version>` |
| **Release notes** `docs/releases/v<version>.md` | written by hand, **before** the build | yes, as the release body |
| **Plate-solving pattern databases** | derived on the user's own machine | **no — see Part 3** |
| **Calibration libraries** | built by the observer from their own darks and flats | **no** — they describe one camera at one setpoint, so they are neither shareable nor ours to ship. `mee2024 calibrate` builds one; Part 3's argument applies unchanged. |

Write the notes **before** building. The spec copies `docs/releases/v<version>.md` to
`dist/release-notes-v<version>.md` as its last step, so the notes end up beside the binary
they describe, and warns if they are missing. The copy in `dist/` is a build artefact; the
tracked file is the source. `dist/` is gitignored, which is how the v1.3.7 and v1.3.8 notes
came to exist on one machine and nowhere else.

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

Build **from the project's own `.venv`**, not a system Python:

```bash
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-build.txt
```
```bash
.venv/Scripts/python -m PyInstaller MEE2024.spec --noconfirm
```

The venv holds exactly the declared dependencies, so PyInstaller cannot sweep up something
heavy that merely happened to be installed on the machine — which is how an earlier build
reached 2.7 GB on CUDA libraries no part of this project uses. `MEE2024.spec` still strips
those by name, but from a clean venv that filter is a safety net rather than the mechanism.

Run it **from the repository root** — the code uses absolute `from mee2024 import ...`
imports, so the root must be on the path. Produces `dist/MEE_v<version>.exe`, one
file, no Python needed on the target machine. The filename follows `_version()` in
`mee2024/MEE2024util.py`, so bump that (and `setup.cfg`) first.

Built and tested with Python 3.12 -- what CI pins, and what the shipped v1.3.9 exe bundles.
This line claimed 3.9 until 2026-08-20. That was stale prose rather than a record of any build
machine, and it led a collaborator to conclude the released binaries carried a 3.9-only bug.
**Check the exe, not this file:** the bundled `python3XX.dll` names the interpreter outright.

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

Start here, because it is the part that can be automated:

```bash
python tools/smoke_exe.py dist/MEE_v1.3.9.exe --expect-version v1.3.9     --lights "I:/MEE test frames/fits/00_23_49/Zenith_0000[1-3].fits"
```

24 checks: the exe runs, reports the version the spec claims, lists all eleven subcommands,
survives `--help` on every one of them, performs its read-only commands, exits **non-zero**
on bad input, and -- with `--lights` -- stacks real frames end to end and exits **0** on
success. Takes about a minute.

That last one is not ceremony. `mee2024 stack` returned its output archive to `sys.exit()`,
which treats a non-integer as an error message, so every successful run reported failure to
the shell. It shipped in v1.3.5 and survived to v1.3.8, and the test suite could not see it
because the source had the same bug. The per-subcommand `--help` sweep is the cheap catch for
the other bundle-only failure: a hidden import missing from one code path and no other.

The suite passing says nothing about any of this -- `pytest` runs the source tree, and almost
every user runs the executable. They are not the same program.


```bash
dist/MEE_v1.3.1.exe --version
```

**Actually run it.** A broken bundle builds perfectly and dies on the first import —
PyInstaller 6.12 against numpy 2.5 produced an exe that failed with `No module named
'numpy._core._exceptions'`, and nothing in the build log hinted at it.

```bash
python tools/inspect_exe.py dist/MEE_v1.3.1.exe --expect-python 3.12
```

This reads the archive and gates the release on it: the bundled catalogue, the UI frontend,
the star-label index, Hipparcos, Tycho, the absence of any GPU/ML stack, and the interpreter
the bundle carries. Pass `--expect-python` and a build from the wrong venv fails here rather
than shipping -- the exe embeds whatever interpreter built it, so that one is the only Python
version most users will ever run, and no document in this repository is evidence about it. Reading the
archive is the only honest check — running the exe and seeing `gaia_dr3_g10 ... installed`
proves nothing on a build machine, because that catalogue is installed in its own data
directory, which is exactly where the runtime looks first.

Then double-click it: the app window opens
(the default since v1.0.0), and `MEE_v1.3.1.exe gui` still opens the classic
interface. Run a small dataset through it and confirm the plate solve succeeds on a machine
with no catalogue in its data directory — that is the whole point of the bundle.

## Publish

```bash
gh release create v1.3.1 "dist/MEE_v1.3.1.exe" --repo andrew551/MEE2024 --title "MEE v1.3.1" --notes-file docs/releases/v1.3.1.md

The notes live in `docs/releases/`, under version control: they are the record of what a
build contained, and `dist/` is ignored, so notes kept there vanish with the first
`git clean`. Older text: "Windows executable, no Python installation required. Double-click for the app window, or run it from a terminal for the command line; the classic interface is still there with: MEE_v1.3.1.exe gui.

The headline changes since v1.0.1:

- **A rebuilt plate solver**, on by default. Solves blind from 1 to 18 degrees at about a second a field, poles included, from the Gaia catalogue rather than Tycho.
- **Works offline out of the box.** The compact Gaia catalogue is bundled in the executable, the standard G < 13 archive downloads on first use, and the plate-solving database builds itself.
- **Batch folder processing.** Point it at a night of captures and every folder of frames is processed as its own field, with results mirrored into the output. One bad field does not stop the rest.
- **The stacked FITS keeps its numbers.** Input bit depth and ADU are preserved instead of being stretched to fill 16 bits; hot pixels are excluded rather than subtracted; and bright stars are no longer discarded just because Gaia has no proper motion for them.
- **A rebuilt interface**: one settings panel, native file dialogs, named stars labelled over the stacked image, and distortion views that finally share the image's orientation."

New: recursive batch folder processing. Tick \"Batch folders\", pick the folder above a night's captures, and every folder of frames beneath it is processed as its own field, with the results written to a matching folder under the output. A failing field does not abandon the rest, each row reports its rms and star count, and there is a Stop button.

Also since v1.2.0:
- The stacked image displays inverted by default (dark stars on white), with a zoom and a toggle. Identified stars are labelled over it, with a slider for how many names to show and a red cross on double stars the fit discarded.
- Bright stars are no longer thrown away. Gaia has no proper motion for 21% of stars brighter than G=4, so their positions could not be brought to the observation epoch and the distortion fit discarded them as outliers; the motion is now borrowed from Hipparcos. Named stars are also labelled properly -- Gaia's crossmatch reaches only 3 of the 49, so names are resolved by position instead.
- The stacked FITS keeps the input's bit depth and ADU instead of being stretched to fill 16 bits, with a recorded PEDESTAL if dark subtraction drives the background negative.
- Hot pixels are excluded from the stack rather than subtracted (saturation clips, so a dark cannot remove them), found from the darks or, when there are none, from the dither.
- Flats are normalised, master darks and flats are saved for reuse, and lights, darks and flats must share a bit depth.
- One settings panel instead of simple/advanced modes, with Auto or Custom, and a three-way date choice.
- Every plot now shares the image's orientation and aspect ratio, and the 3-D surfaces are no longer drawn upside down. A magnitude-of-displacement surface joins the x and y ones.
- Stacking gives up after two frames when the first two cannot be matched, instead of centroiding every frame first.
- A catalogue bundled in the executable can now actually be opened (it was offered and then failed), and Clear clears the darks and flats too."
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
