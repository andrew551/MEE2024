# -*- mode: python -*-
"""
PyInstaller spec for a self-contained MEE2024 build.

Run from the repository root (not from mee2024/), so that `mee2024` is importable as a
package -- the code uses absolute `from mee2024 import ...` imports throughout:

    python -m PyInstaller MEE2024.spec --noconfirm

Produces dist/MEE_2024_v<version>.exe on Windows, where the version is read from the
package rather than written here, so the two cannot drift apart. Double-clicking it opens
the new app window; `MEE_2024_v<version>.exe gui` opens the classic one, and every CLI
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
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

package = os.path.join(os.getcwd(), 'mee2024')
if not os.path.isdir(package):
    raise SystemExit('run this spec from the repository root, not from mee2024/')

sys.path.insert(0, os.getcwd())
from mee2024.MEE2024util import _version           # noqa: E402  (needs the path above)

exe_name = f'MEE_2024_{_version()}'

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
    # bundle that renders to Agg and to the web frontend
    excludes=['tkinter.test', 'pytest', '_pytest', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=exe_name,
    onefile=True,
    # console=True so that `MEE_2024.exe --help`, the CLI subcommands and any error
    # traceback remain visible. The UI opens its own window on top of it.
    console=True,
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name=f'{exe_name}.app',
        bundle_identifier='org.mee2024.mee2024',
    )
