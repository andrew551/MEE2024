# MEE2024
Modern Eddington Experiment codebase

## Installation

### Windows

The Windows executable (see releases) will run on Windows 10 and 11 computers without having to install Python.

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

- To run (and potentially edit) the Python source code, install the most recent version of Python from python.org (make sure to check the box to add Python to PATH on windows).

- To install requirements: pip install -r requirements.txt

- Run with python: python mee2024/main.py

## The app window

```
mee2024 ui
```

Opens a native window (using the platform's own web view) with a simplified workflow:
choose your light frames, pick a processing preset, press Run. Progress, a stacked-image
preview, and graded score cards appear in the window — no plot pop-ups. `--browser` uses
your default browser instead, serving the same interface from the same local server.

The classic interface is still there and unchanged:

```
mee2024 gui
```

## Running it

Double-clicking the executable (or running `mee2024` with no arguments) opens the new app
window. The classic interface is still there:

```bash
mee2024 gui                                     # the classic interface, once
mee2024 config --set default_interface=classic  # ...or make it the default again
```

## Command line

Running `mee2024` with no arguments opens the classic GUI, as before. With arguments it
runs headlessly, which is what the test suite and any batch processing use:

```
mee2024 stack       LIGHTS... [--dark ...] [--flat ...] [-o DIR]   # stage 1
mee2024 distortion  DATA.zip  [--order quintic]                    # stage 2
mee2024 eclipse     DISTORTION.zip                                 # stage 3
mee2024 run         LIGHTS... [--eclipse]                          # stages back to back
mee2024 config      --show | --set key=value
mee2024 build-triangle-db                                          # regenerate the platesolve database
```

## Offline star catalogues

`mee2024 catalogue` lists what is installed and where. Catalogues are built locally with
`tools/build_gaia_offline.py` (all-sky Gaia G<12 is ~3.1 M stars, 29 queries, 10-22 min).

To move one to another machine without re-downloading from Gaia:

```bash
mee2024 catalogue --pack gaia_dr3_g12          # -> gaia_dr3_g12.zip, prints its sha256
# copy the zip across, then on the other machine:
mee2024 catalogue --install gaia_dr3_g12.zip
```

Every column carries a SHA-256 in the catalogue's manifest, so `--install` verifies the
transfer and refuses to leave a corrupt catalogue in place. `mee2024 catalogue --verify NAME`
re-checks an installed one at any time.

The plate-solving triangle database (`TripleTriangle_pattern_data.npz`, ~128 MB, in the
same user-data directory) is a plain `.npz` and can simply be copied to save the few
minutes it takes to regenerate.

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
