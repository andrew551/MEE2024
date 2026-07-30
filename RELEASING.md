# Publishing releases

Two separate releases, deliberately kept apart:

| release | tag | contains | changes |
|---|---|---|---|
| **Star catalogues** | `catalogues-v1` | two `.zip` archives, 327 MB total | rarely |
| **Software** | `v1.0.0` | `MEE_2024_v1.0.0.exe`, 187 MB | every version |

They are separate because the catalogues change far less often than the code. Pinning them
together would mean re-uploading 327 MB for every patch release, and would break the
download URLs already compiled into older builds.

---

# Part 1 — the star catalogues

The application **already knows these URLs**. `mee2024/starcat/download.py` expects exactly:

```
https://github.com/andrew551/MEE2024/releases/download/catalogues-v1/gaia_dr3_g12.zip
https://github.com/andrew551/MEE2024/releases/download/catalogues-v1/gaia_dr3_g12_13.zip
```

so three things must match exactly, or the download silently 404s:

1. the repository is **`andrew551/MEE2024`** and is **public**
2. the tag is **`catalogues-v1`** — not `catalogues_v1`, not `v1-catalogues`
3. the asset filenames are **`gaia_dr3_g12.zip`** and **`gaia_dr3_g12_13.zip`**

## What to upload

Both files are already built and waiting in `dist/upload/`:

| file | size (bytes) | sha256 |
|---|---|---|
| `gaia_dr3_g12.zip` | 137,952,319 | `f4a579e369c41b6d7099bac6b20d58c69f6b750092cd62f4085c77c670fbc5cb` |
| `gaia_dr3_g12_13.zip` | 188,640,212 | `28607c07a9f60c89f09ba653eac06f890ff89631d28252e9af7af63be1adb71b` |

Those hashes are compiled into the application, which refuses any download that does not
match. **Do not rename, re-zip or re-compress the files** — that changes the hash and every
download will then fail verification. Upload them exactly as they are.

If you ever do need to rebuild them, re-pack and then update the two `sha256` values in
`RELEASES` in `mee2024/starcat/download.py`:

```bash
mee2024 catalogue --pack gaia_dr3_g12       # prints the new sha256
mee2024 catalogue --pack gaia_dr3_g12_13
```

## Option A — the command line (recommended, least room for error)

Needs the GitHub CLI: <https://cli.github.com/>. One-off sign-in: `gh auth login`.

```bash
cd C:/Users/Andrew/Documents/mee26/MEE2024
```

```bash
gh release create catalogues-v1 "dist/upload/gaia_dr3_g12.zip" "dist/upload/gaia_dr3_g12_13.zip" --repo andrew551/MEE2024 --title "Star catalogues v1" --notes "Offline Gaia DR3 star catalogues for MEE2024. gaia_dr3_g12.zip is G<12, 3,087,821 stars. gaia_dr3_g12_13.zip is 12<G<13, 4,281,806 stars, an optional extension for deep eclipse fields. Install with: mee2024 catalogue --fetch gaia_dr3_g12. Derived from ESA/Gaia/DPAC data."
```

The upload takes a few minutes for 327 MB. It prints the release URL when finished.

> **Do not pass `--draft`.** A draft release is invisible to anonymous downloads, so the
> app would get a 404 even though the release looks fine in your browser. If you want to
> stage it privately first, use `--draft` and then publish with
> `gh release edit catalogues-v1 --draft=false` when ready.

## Option B — the GitHub website

1. Go to **<https://github.com/andrew551/MEE2024/releases>** and click **Draft a new
   release**.
2. **Choose a tag** → type `catalogues-v1` → click **"+ Create new tag: catalogues-v1 on
   publish"**. Leave the target as `main`.
3. Release title: `Star catalogues v1`.
4. Description: anything you like; the text from Option A is a reasonable starting point.
5. Drag **both** files from `dist/upload/` into the *"Attach binaries..."* box. Wait for
   both to reach 100% — a 327 MB upload is not quick, and clicking Publish early attaches
   nothing.
6. Leave **"Set as a pre-release" unticked**.
7. Click **Publish release** — *not* "Save draft".

## Verify it worked

This is the important step, and it needs no download:

```bash
mee2024 catalogue --check-remote
```

Expected:

```
[OK  ] gaia_dr3_g12
       https://github.com/andrew551/MEE2024/releases/download/catalogues-v1/gaia_dr3_g12.zip
       reachable, 138 MB
[OK  ] gaia_dr3_g12_13
       https://github.com/andrew551/MEE2024/releases/download/catalogues-v1/gaia_dr3_g12_13.zip
       reachable, 189 MB

All catalogue assets are reachable; `mee2024 catalogue --fetch NAME` will work.
```

If it says `FAIL ... 404 not found`, one of the three exact-match conditions above is
wrong, or the release is still a draft. If it reports a **size mismatch**, a different or
re-compressed file was uploaded and the hashes in `download.py` no longer apply.

Then confirm a real download works, ideally on a machine that does not already have the
catalogue:

```bash
mee2024 catalogue --fetch gaia_dr3_g12
```

It downloads, checks the SHA-256, unpacks, and verifies every column against the manifest
before accepting it.

---

# Part 2 — the Windows executable

## Build

```bash
cd C:/Users/Andrew/Documents/mee26/MEE2024
python -m PyInstaller MEE2024.spec --noconfirm
```

Run it **from the repository root** — the code uses absolute `from mee2024 import ...`
imports, so the root has to be on the path. Produces `dist/MEE_2024_v1.0.0.exe`, about
187 MB, self-contained, no Python needed on the target machine.

Built and tested with Python 3.9. The full test suite passes on 3.9 and 3.14.

## Check before shipping

```bash
dist/MEE_2024_v1.0.0.exe --version
dist/MEE_2024_v1.0.0.exe catalogue
dist/MEE_2024_v1.0.0.exe ui --browser
```

Then double-click it: the **new app window** should open. That is the default from v1.0.0.
`MEE_2024_v1.0.0.exe gui` still opens the classic interface, and
`mee2024 config --set default_interface=classic` makes the classic one the default again.

## Publish

```bash
gh release create v1.0.0 "dist/MEE_2024_v1.0.0.exe" --repo andrew551/MEE2024 --title "MEE2024 v1.0.0" --notes "Windows executable, no Python installation required. Double-click to open the new interface, or run it from a terminal for the command line. The classic interface is still available with: MEE_2024_v1.0.0.exe gui. Star catalogues are downloaded separately on first use."
```

The executable deliberately does **not** contain the 327 MB of star catalogues. It fetches
them on first use, or they can be installed from a local file with
`mee2024 catalogue --install`.

---

# Moving to Zenodo later

Zenodo gives a citable DOI, which matters once results are published. The switch is a
two-line edit per catalogue in `RELEASES` in `mee2024/starcat/download.py` — replace `url`
and fill in `doi`. The `sha256` values stay the same as long as the same files are
uploaded, so existing installations are unaffected.

Users can also override the source without any new software release:

```bash
mee2024 catalogue --set-source gaia_dr3_g12 --url URL --sha256 HASH
```

## Attribution

Gaia data is freely redistributable with credit to **ESA/Gaia/DPAC**. The bundled
Hipparcos-derived catalogue and star-label index come from the Hipparcos-2 reduction
(van Leeuwen 2007) and Gaia's own `hipparcos2_best_neighbour` crossmatch. Each catalogue
records its sources in its `manifest.json` provenance field.
