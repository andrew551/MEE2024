# -*- mode: python -*-
import sys
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files("astroquery", includes=["CITATION"])
datas += [("resources/compressed_tycho2024epoch.npz", "resources")]

a = Analysis(
    ["MEE2024Stacker.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MEE_2024_v0.5.3",
    onefile=True,
    console=True,
)

# macOS only: wrap in .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="MEE_2024_v0.5.3.app",
        bundle_identifier="org.mee2024.mee2024",
    )