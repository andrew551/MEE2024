# MEE2024
Modern Eddington Experiment codebase

## Installation

### Windows

The Windows executable (see releases) will run on Windows 10 and 11 computers without having to install Python. It carries the compact star catalogue (Gaia G < 10), so it plate-solves offline the moment it starts; fetch the standard `gaia_dr3_g13` archive when you need fainter stars.

### Mac/Linux

- The recommended way to install MEE2024 is via pip in the terminal. As a pre-requisite, this requires an install of Python 3.9+, with Python added to path (see https://www.python.org/downloads/).

- (Linux only: you may need to install tkinter for python: sudo apt-get install python3-tk)

- Note that a terminal is required both to install and launch the program.

- Then to either install or run, paste the following into a terminal (you may want to save the command to a local re-usable bash file):

```
set -e

APP_NAME="mee2024"
ENV_DIR="$HOME/.mee2024env"
REPO_URL="git+https://github.com/andrew551/MEE2024.git"

echo "Using environment: $ENV_DIR"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

if ! command -v mee2024 >/dev/null 2>&1; then
    echo "Installing / reinstalling MEE2024..."
    pip install --upgrade pip
    pip install --upgrade "$REPO_URL"
fi

echo "Launching MEE2024..."
exec mee2024
```
After installing, you may use this to run MEE2024:
```
source "$HOME/.mee2024env/bin/activate"
mee2024
```


### Installation from Source

To run (and potentially edit) the Python source code, install Python from python.org
(on Windows, check the box to add Python to PATH). Then, from the repository root,
work in a virtual environment so the project has exactly its own dependencies and
nothing else:

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
```
```bash
.venv/bin/python -m pip install -r requirements.txt              # macOS / Linux
```

Then run it with that interpreter:

```bash
.venv/Scripts/python.exe mee2024/main.py
```

`.venv/` is git-ignored. Keeping the environment clean is not just tidiness: an
environment carrying unrelated heavyweight packages (a GPU build of PyTorch, say)
inflates the packaged executable, because PyInstaller follows optional references to
them from scikit-image and scikit-learn.

**`pywebview` is what gives you the native app window** — and with it the platform's
own file dialogs. Without it `mee2024` falls back to your default browser, which
cannot open a native dialog; the app says which one it is using and why on start-up.
It is in `requirements.txt` for Windows and macOS; on Linux it additionally needs
system webkit2gtk (e.g. `sudo apt install gir1.2-webkit2-4.0`).

Tests:

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```
```bash
.venv/Scripts/python.exe -m pytest
```

## The app window

Double-clicking the executable, or running `mee2024` with no arguments, opens the app
window — a native window using the platform's own web view. Choose your light frames, pick
a processing preset, press Run. Progress, a stacked-image preview, the fitted distortion
field and graded score cards all appear in the window; no plot pop-ups.

There is also a **Watch** mode: point it at a folder and frames are stacked and solved as
they arrive, so you get pointing and quality feedback while the telescope is still on the
field. A frame is only opened once it has stopped changing, so a file still being written
is never read half-finished.

Under **Advanced analysis** — hidden until you open it, since the field map above answers
the usual question — are two diagnostic views:

- the fitted displacement as a **rotatable surface** with every measured star drawn on it.
  A star sits off the surface by exactly its residual, so a distortion order that is too
  low shows as the scatter undulating coherently above and below rather than peppering it
  evenly. Residuals can be exaggerated ×5 to ×100 to see their structure on a good fit.
- a **residual-correlation map**: the detector divided into cells, each showing how far
  residuals inside it point the same way as their nearest neighbour's. Near zero is
  uncorrelated noise; a warm patch is a local optical imperfection the global polynomial
  has not absorbed. The single "residual structure" score card is this map averaged. The
  bin count is chosen from the star count so each cell holds enough stars to mean
  something, and can be changed live.

```
mee2024 ui              # explicitly, if you prefer
mee2024 ui --browser    # same interface, in your default browser
```

The classic interface is still there and unchanged:

```bash
mee2024 gui                                     # open it once
mee2024 config --set default_interface=classic  # ...or make it the default again
```

## Command line

With arguments, `mee2024` runs headlessly — which is what the test suite and any batch
processing use:

```
mee2024 stack       LIGHTS... [--dark ...] [--flat ...] [-o DIR]   # stage 1
mee2024 distortion  DATA.zip  [--order quintic]                    # stage 2
mee2024 eclipse     DISTORTION.zip                                 # stage 3
mee2024 run         LIGHTS... [--eclipse]                          # stages back to back
mee2024 config      --show | --set key=value
mee2024 build-triangle-db                                          # regenerate the platesolve database
```

## Star catalogues

There are two catalogue choices, and the first is the right one almost always:

| choice | what it is |
|---|---|
| **Gaia** (default) | the installed offline archive, plus the ~100 stars so bright that Gaia records nothing for them at all (Sirius, Vega, Arcturus, Canopus…), filled from Hipparcos. Milliseconds per field |
| Gaia archive (online) | queries the ESA archive for every field: minutes per run, and needs a connection. Under Advanced, for when you deliberately want it |

The standard archive is **`gaia_dr3_g13`** — Gaia DR3 to G < 13, 7.37 M stars — which
covers ordinary runs (the default magnitude limit is 12) and the fainter stars an
eclipse field wants. Selecting Gaia with nothing installed offers to download it;
declining is fine, the run falls back to the online archive and says so.

```bash
mee2024 catalogue                            # what is installed, and where
mee2024 catalogue --fetch gaia_dr3_g13       # ...or just press Run and accept
mee2024 catalogue --check-remote             # are the published archives reachable?
```

Two other tiers exist for the ends of the range: **`gaia_dr3_g10`** (G < 10, 24 MB,
bundled inside the Windows executable so it solves offline out of the box) and
**`gaia_dr3_g15`** (G < 15, several GB, for work that genuinely needs it).

Asking for stars fainter than the installed catalogue reaches is reported rather than
silently truncated — a mag-14 request against a G<13 archive warns and names the fix.
The same honesty applies to double stars: companions fainter than the archive cannot be
flagged, so the search depth is clamped to the catalogue and said out loud. The cost is
small (a companion at Δm displaces a centroid by ~10^(−0.4Δm) of its separation) and the
speed is worth it.

### Older installs, and repairs

If you already have the original `gaia_dr3_g12` + `gaia_dr3_g12_13` pair, merge them
into the standard archive rather than downloading anything — this also recomputes the
double-star neighbour flags across the union, which the two separate archives could not
do:

```bash
mee2024 catalogue --merge                    # -> gaia_dr3_g13, then --remove the parts
```

An interrupted download or unpack can leave an archive with all its data and no
manifest, after which everything reports it as absent. That is repairable without
re-downloading — the data is validated (column lengths, declination ordering, the band
index, position ranges, depth) before a new manifest is written:

```bash
mee2024 catalogue --repair gaia_dr3_g12
```

**Close the program before catalogue operations.** A running instance holds the archive
memory-mapped, and Windows will refuse the write — which is how a half-installed archive
happens in the first place.

Catalogues can also be built from scratch with `tools/build_gaia_offline.py` (all-sky
Gaia G<12 is ~3.1 M stars, 29 queries, 10–22 min).

To move one to another machine without re-downloading from Gaia:

```bash
mee2024 catalogue --pack gaia_dr3_g13          # -> gaia_dr3_g13.zip, prints its sha256
# copy the zip across, then on the other machine:
mee2024 catalogue --install gaia_dr3_g13.zip
```

Every column carries a SHA-256 in the catalogue's manifest, so `--install` verifies the
transfer and refuses to leave a corrupt catalogue in place. `mee2024 catalogue --verify NAME`
re-checks an installed one at any time.

### The plate-solving databases

The rebuilt solver reads pattern databases built from the star catalogue, one per field
scale, in the same user-data directory. Build the standard one once:

```bash
mee2024 build-pattern-db                       # patdb_g12_t17k, ~230 MB, ~3 minutes
```

Optional layers extend blind solving to the ends of the range — a 0.6° layer for fields
around a degree and a 4° layer out past 18°. Installing a layer is the entire
configuration; the solver uses whatever is present, and consults the extra layers only
when the primary one does not solve a field.

Any option can be overridden with `--set key=value` (repeatable), and `--no-display`
suppresses every plot window so a run can complete unattended. For example:

```bash
mee2024 stack data/*.fit -o out/ --no-display --set min_area=3 --set sigma_subtract=2.5
```

`docs/ARCHITECTURE.md` describes the three pipeline stages, the file contracts between
them, the coordinate conventions, and where the astrometric error budget is spent.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`pytest --runslow` additionally runs the plate-solve regression corpus, which needs the
triangle database (built automatically on first use).

## Tips

A small platesolve database is built into the executable (derived from the Tycho catalogue).

An internet connection is required to connect to the Gaia database.

Note that when the program is run for the first time, it may take a few minutes to perform some one-off precomputation.



## **Usage**

Run the Python file / executable obtained following installation.
Select the images to be stacked (the "Light frames").
Dark frame(s) and Flat frame(s) can also be selected, if desired.

It is recommended to choose an _Output folder_ or else the output files will be written to the same folder which contains the image data.

Select "Show graphics" if you want to view the intermediate graphical analysis (optional).
Choose the number of bright stars to be identified in the stacked image (the default of 100 is a reasonable choice).

The output FITS stacked image is by default resized to 16-bit (the same as the input). A second 32-bit floating point (FP) FITS file can also be saved (optional).
The program does its calculations in 32-bit FP to preserve accuracy. The 32-bit FP file will be twice the size of the 16-bit file.

The Dark and Flat stacked images can also be saved (optional); the format is 32-bit FP FITS. These stacks can be used in subsequent processing of Light frames.

"Remove big bright object" is useful when images contain the Sun or Moon. It can be kept enabled for star fields with no Sun or Moon.
The _blob_radius_extra_ parameter determines the extra exclusion zone outside the saturated region. The "extra" distance is measured in pixels.
The _centroid_gap_blob_ parameter determines which centroids outside the extra exclusion zone should be ignored. The "gap" distance is measured in pixels.
The default parameters are (100, 30) but neither is particularly sensitive.
The purpose of this function is to limit the centroid search to areas away from the Moon and the solar corona.

"Sensitive stacking mode" should only be use if images contain the Sun or Moon or are taken on a bright sky (e.g. twilight).
For dark-sky star fields, this mode will take too long and is not recommended.

"Use sensitive mode on stacked result" can be left on for most images which require accurate centroid finding, but the sensitivity parameters should adjusted accordingly.
A lower _sigma_thresh_ will increase the sensitivity (between 4 and 7 are typical values).
A smaller _min_area_ will mean centroids of smaller pixel size will be found (between 1 and 4 are typical values).
A higher _sigma_subtract_ will increase the background cutoff, thereby eliminating more noise and reducing the number of centroids found (between 0 and _sigma_thresh_ -2 are typical values). For good dark-sky images, (5.0, 4, 3.0) are reasonable values.

"Remove centroids near edges" will remove extraneous centroids associated with edge effects near the Moon or the solar corona.
This function can be left on, just like "Remove big bright object", even when processing normal star fields.

The file called _MEE_config.txt_ (saved in your appdata or userdata) stores the program parameters, including the input and output directories.
It is automatically updated each time the program is run, and can also be manually edited for advanced use (all standard parameters can be edited via the GUI).
