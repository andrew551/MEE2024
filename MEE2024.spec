# -*- mode: python -*-
"""
PyInstaller spec for a self-contained MEE2024 build.

Run from the repository root (not from mee2024/), so that `mee2024` is importable as a
package -- the code uses absolute `from mee2024 import ...` imports throughout:

    python -m PyInstaller MEE2024.spec --noconfirm

Produces dist/MEE_v<version>.exe on Windows, where the version is read from the
package rather than written here, so the two cannot drift apart. Double-clicking it opens
the new app window; `MEE_v<version>.exe gui` opens the classic one, and every CLI
subcommand works too.

Bundling notes, each of which was needed to make the build actually run:

* The package data is not just the Tycho .npz any more. `resources/*` does not recurse, so
  the Hipparcos catalogue and the star-label index need explicit entries, as does the
  single-file UI frontend.
* scikit-image, scikit-learn and astropy load submodules dynamically, so PyInstaller's
  static analysis misses them; the existing `import skimage.data._fetchers` and
  `sklearn.metrics._pairwise_distances_reduction.*` shims in the source cover the worst of
  it, and the hidden imports below cover the rest.
* pywebview is optional at runtime: if it is missing the app falls back to the browser.
  It is included when present so the packaged app gets a real window.
"""

import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

package = os.path.join(os.getcwd(), 'mee2024')
if not os.path.isdir(package):
    raise SystemExit('run this spec from the repository root, not from mee2024/')

sys.path.insert(0, os.getcwd())
from mee2024.MEE2024util import _version, get_catalogue_root  # noqa: E402  (needs the path above)

exe_name = f'MEE_{_version()}'

datas = []
datas += collect_data_files('astroquery', includes=['CITATION'])

# Bundled catalogues and the UI. Note the two different destinations, which are not
# interchangeable:
#
#   * MEE2024util.resource_path() joins sys._MEIPASS directly, so anything it looks up
#     must land at the archive ROOT ("resources/..."), not under "mee2024/".
#   * ui/server.py finds the frontend relative to its own __file__, which under PyInstaller
#     is _MEIPASS/mee2024/ui/, so frontend.html must land there.
#
# And `resources/*` does not reach into subdirectories, so the Hipparcos catalogue and the
# label index are listed file by file -- the same trap that had to be fixed in setup.cfg.
datas += [(os.path.join(package, 'resources', 'compressed_tycho2024epoch.npz'),
           'resources')]
for subdir in ('hipparcos2', 'star_labels'):
    source = os.path.join(package, 'resources', subdir)
    if os.path.isdir(source):
        for name in os.listdir(source):
            datas.append((os.path.join(source, name), f'resources/{subdir}'))
datas += [(os.path.join(package, 'ui', 'frontend.html'), 'mee2024/ui')]

# The compact star catalogue, so the executable solves plates offline out of the box
# instead of falling back to minutes-per-field archive queries. It is deliberately NOT
# in source control (24 MB of generated data): the build takes it from wherever it is
# installed on this machine, and simply ships without it if absent. Build it with
#     mee2024 catalogue --merge            (or --fetch gaia_dr3_g12)
#     python -c "from mee2024.starcat import download; download.build_compact_tier()"
BUNDLED_CATALOGUE = 'gaia_dr3_g10'
_catalogue_source = None
for candidate in (os.path.join(package, 'resources', 'catalogues', BUNDLED_CATALOGUE),
                  os.path.join(str(get_catalogue_root()), BUNDLED_CATALOGUE)):
    if os.path.isfile(os.path.join(candidate, 'manifest.json')):
        _catalogue_source = candidate
        break
if _catalogue_source:
    for name in os.listdir(_catalogue_source):
        datas.append((os.path.join(_catalogue_source, name),
                      f'resources/catalogues/{BUNDLED_CATALOGUE}'))
    print(f'spec: bundling {BUNDLED_CATALOGUE} from {_catalogue_source}')
else:
    print(f'spec: {BUNDLED_CATALOGUE} not found; the executable will need to download '
          f'a catalogue on first use')

hiddenimports = [
    'skimage.data._fetchers',
    'sklearn.metrics._pairwise_distances_reduction._datasets_pair',
    'sklearn.metrics._pairwise_distances_reduction._middle_term_computer',
    'sklearn.neighbors._partition_nodes',
    'scipy.special.cython_special',
    'astropy.constants',
    'FreeSimpleGUI',
]
hiddenimports += collect_submodules('astropy.coordinates.builtin_frames')
try:
    import webview  # noqa: F401  -- probe: bundle the native window when available
    hiddenimports += collect_submodules('webview')
except Exception:
    pass

a = Analysis(
    [os.path.join(package, 'main.py')],
    pathex=[os.getcwd()],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # matplotlib's interactive backends and the test frameworks are dead weight in a
    # bundle that renders to Agg and to the web frontend.
    #
    # The machine-learning stacks matter more than they look: scikit-image and
    # scikit-learn reference torch, tensorflow and friends in optional code paths, so
    # PyInstaller's analysis follows them if they happen to be installed on the build
    # machine -- which turned one build into a 2.7 GB executable instead of ~215 MB.
    # Nothing here uses them, so excluding them is free, and doing it explicitly means
    # the executable's size no longer depends on what else the builder has in their
    # environment.
    excludes=['tkinter.test', 'pytest', '_pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
              'torch', 'torchvision', 'torchaudio', 'tensorflow', 'tensorboard', 'keras',
              'jax', 'jaxlib', 'transformers', 'sympy', 'IPython', 'notebook',
              'jupyter', 'jupyter_core', 'nbconvert', 'nbformat', 'zmq', 'tornado',
              'numba', 'llvmlite', 'dask', 'bokeh', 'plotly', 'seaborn', 'pyarrow'],
)

# This used to *strip* GPU/ML files from the tables, because `excludes` alone could not:
# excluding a package stops its modules being collected, but binaries a PyInstaller hook
# has already contributed survive it, and torch's CUDA libraries (1.25 GB of cuBLAS, cuFFT
# and cuSPARSE) sailed straight through `excludes=['torch']` into a 2.7 GB executable.
#
# The build environment is the real fix. Releases are built from the project's own .venv
# (see RELEASING.md), which holds only the declared dependencies, so there is nothing
# stray to collect -- measured: the strip removed exactly zero files. Editing the tables
# is therefore no longer earning its keep, and silently mutating a build is a poor way to
# discover you built it in the wrong place. What remains is the check, which is the part
# that was actually load-bearing: fail loudly, and say what to do about it.
UNWANTED = {'torch', 'torchvision', 'torchaudio', 'cupy', 'cupy_backends',
            'cupyx', 'tensorflow', 'tensorboard', 'keras', 'jax', 'jaxlib',
            'nvidia', 'triton', 'transformers', 'numba', 'llvmlite'}


def _check_no_unwanted(*tables):
    found = set()
    for table in tables:
        for entry in table:
            top = re.split(r'[\\/]', entry[0])[0].lower().split('.')[0]
            if top in UNWANTED:
                found.add(top)
    if found:
        raise SystemExit(
            f'spec: refusing to build -- {", ".join(sorted(found))} would be bundled, and '
            f'nothing here imports them. This means the build environment carries packages '
            f'the project does not declare (a torch install adds ~1.25 GB of CUDA '
            f'libraries). Build from the project venv instead:\n'
            f'    .venv/Scripts/python -m pip install -r requirements.txt '
            f'-r requirements-build.txt\n'
            f'    .venv/Scripts/python -m PyInstaller MEE2024.spec --noconfirm')


_check_no_unwanted(a.binaries, a.datas, a.pure)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=exe_name,
    onefile=True,
    # console=True so that `MEE_v<version>.exe --help`, the CLI subcommands and any error
    # traceback remain visible. The UI opens its own window on top of it.
    console=True,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name=f'{exe_name}.app',
        bundle_identifier='org.mee2024.mee2024',
    )
