# Publishing the catalogues and the Windows build

## 1. Catalogue archives (GitHub release)

The app already knows the URLs — `mee2024/starcat/download.py` points at a release tagged
`catalogues-v1` with the SHA-256 of each archive baked in. Until the assets exist those
URLs return 404, reported as an actionable message rather than a traceback.

Pack them (the hashes below are what the code expects, so re-pack only if the catalogue
itself is rebuilt):

```bash
mee2024 catalogue --pack gaia_dr3_g12       # 138 MB
mee2024 catalogue --pack gaia_dr3_g12_13    # 189 MB
```

| asset | size | sha256 |
|---|---|---|
| `gaia_dr3_g12.zip` | 137,952,319 | `f4a579e369c41b6d7099bac6b20d58c69f6b750092cd62f4085c77c670fbc5cb` |
| `gaia_dr3_g12_13.zip` | 188,985,889 | `28607c07a9f60c89f09ba653eac06f890ff89631d28252e9af7af63be1adb71b` |

Create the release and upload:

```bash
gh release create catalogues-v1 \
  gaia_dr3_g12.zip gaia_dr3_g12_13.zip \
  --title "Star catalogues v1" \
  --notes "Gaia DR3 offline catalogues for MEE2024. G<12 base (3,087,821 stars) and the
optional 12<G<13 extension (4,281,806 stars). Install with
\`mee2024 catalogue --fetch gaia_dr3_g12\`, or download and
\`mee2024 catalogue --install <file>\`. Derived from ESA/Gaia/DPAC data."
```

Then `mee2024 catalogue --fetch gaia_dr3_g12` works on any machine, verifying the download
against the manifest checksums before installing.

**Why a separate `catalogues-v1` tag** rather than attaching to the software release: the
catalogues change far less often than the code, and pinning them to a software version
would mean re-uploading 327 MB with every patch release.

**Attribution.** Gaia data is freely redistributable with credit to *ESA/Gaia/DPAC*. The
Hipparcos-derived bundle credits van Leeuwen (2007). Both are recorded in each catalogue's
`manifest.json` provenance field.

**Moving to Zenodo later** is a two-line edit to `RELEASES` in `download.py` (URL + DOI),
and existing installs are unaffected. Users can also override without any code change:

```bash
mee2024 catalogue --set-source gaia_dr3_g12 --url URL --sha256 HASH
```

## 2. Windows executable

```bash
python -m PyInstaller MEE2024.spec --noconfirm
```

Run from the repository root. Produces `dist/MEE_2024_v1.0.0.exe` (187 MB, self-contained).
Built and tested with Python 3.9.

Verify before shipping:

```bash
dist/MEE_2024_v1.0.0.exe --version
dist/MEE_2024_v1.0.0.exe catalogue
dist/MEE_2024_v1.0.0.exe ui --browser        # prints a URL with a session token
dist/MEE_2024_v1.0.0.exe distortion <a centroid_data zip> -o out --no-display
```

Attach it to a normal software release:

```bash
gh release create v1.0.0 dist/MEE_2024_v1.0.0.exe --title "MEE2024 v1.0.0"
```

The exe does **not** contain the star catalogues (327 MB); it downloads them on first use
from the `catalogues-v1` release, or they can be installed from a local file.
