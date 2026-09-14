"""Keep the RECORD folder's non-cell entries in step with their sources.

RECORD (`F:\\MEE_output\\RECORD`) is the curated index to a ~140 GB output
tree: for each finished piece of work, the outputs someone would need to check it,
re-quote it, or publish it. The three cell folders (bruns2017, leon2026, mexico2024) are
written by their own chart tools with a RECORD-copy switch. This script covers the rest:

  * `RECORD/WHICH_REDUCTION_IS_THE_RECORD.md` -- a copy of the repository document that
    says which run is quoted for each cell. The repository copy is the source; a hand-kept
    copy of it on D: had drifted (2026-09-06), which is why this exists.
  * `RECORD/refraction/` -- the Leon refraction work is not a cell (it measures no L) but
    it feeds the atmosphere term of two, and its six figures were the only ones the
    repository carried. They now live here with the tables the record cites, and an index
    with the caption facts from `tools/refraction/FIGURES.md`.

Run from the repository root with the venv:

    .venv/Scripts/python.exe tools/sync_record.py            # copy what changed
    .venv/Scripts/python.exe tools/sync_record.py --check    # report only, no writes

Copies are byte-compared first, so an unchanged file is neither rewritten nor reported as
new. Nothing is ever deleted from RECORD; superseded material goes in dated
`superseded_*` folders, as in the cell folders.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = r"F:\MEE_output"
RECORD = os.path.join(OUT, "RECORD")
REFRACTION_SRC = os.path.join(OUT, "refraction")

# (source relative to REFRACTION_SRC, destination relative to RECORD/refraction)
REFRACTION_FILES = [
    # the six figures, in the order tools/refraction/FIGURES.md ranks them
    ("stability_vs_tube_temperature.png", "stability_vs_tube_temperature.png"),
    ("m4_mosaic/m4_curves.png", "m4_curves.png"),
    ("m3_maps/m3_quiver_maps.png", "m3_quiver_maps.png"),
    ("m3_maps/m3_vertical_profiles.png", "m3_vertical_profiles.png"),
    ("platescale_vs_temperature.png", "platescale_vs_temperature.png"),
    ("cubic_vs_temperature.png", "cubic_vs_temperature.png"),
    # the tables docs/REFRACTION_2026.md cites, and the ones behind the figures
    ("INVENTORY.csv", "INVENTORY.csv"),
    ("m2_fieldwindow_summary.csv", "m2_fieldwindow_summary.csv"),
    ("m3_maps/m3_stats.csv", "m3_stats.csv"),
    ("m4_mosaic/m4_fields.csv", "m4_fields.csv"),
    ("perframe_results.csv", "perframe_results.csv"),
    ("band_cubic_results.csv", "band_cubic_results.csv"),
    ("band_fields.csv", "band_fields.csv"),
    # the original strategy document the repository record grew from
    ("REFRACTION_2026_STRATEGY.md", "REFRACTION_2026_STRATEGY.md"),
    # the withdrawn first figure, kept where the erratum can point at it
    ("night2_temperature.png", "superseded_2026-08-27/night2_temperature.png"),
]

REFRACTION_INDEX = """# Leon 2026 refraction -- the record set

Copied from `F:\\MEE_output\\refraction\\` by `tools/sync_record.py`; the
argument these belong to is `docs/REFRACTION_2026.md` in the repository, and
`tools/refraction/FIGURES.md` names the script that regenerates each figure. This is not a
matrix cell -- it measures no deflection constant -- but it is where the atmosphere term
of the Leon and Portland cells comes from, and two of its figures are publication-grade.

## The two figures flagged for publication

**`stability_vs_tube_temperature.png`** -- the campaign's summary figure. Plate scale and
free-cubic d(3000) against normalised tube temperature (FOCTEMP - 7.74 K), night 1 shown
but unfitted. The FOCTEMP sensor is coupled to the telescope so it supplies the dynamics;
the calibrated free-air logger supplies the scale, through the measured +7.74 +/- 0.67 K
offset. Caption facts: plate-scale fit -109.2 +/- 4.7 ppm/K at r = -0.95 (n = 55, unflipped
night-2 fields). This is an empirical FOCTEMP-referenced slope -- the physical coupling is
bracketed -16 to -40 ppm/K (REFRACTION_2026.md 16.9-16.11) -- so do not quote -109 as a
material property. Regenerate with `tools/refraction/tube_temperature_chart.py`
(offset `OFF`, reference scale `REF` at the top of the script).

**`m4_curves.png`** -- the meridian mosaic, 78 of 80 fields: corrections-ON plate scale
against altitude, with rms and star count. Caption facts: the standard refraction model
holds to <~50 ppm above alt 10 deg once drift (+1.4 ppm/min), north-south asymmetry
(-56 +/- 14 ppm) and the pier flip are removed; below 10 deg the residuals reach the
+/-900 ppm class and the plate solver itself fails between alt 5.0 and 5.6 deg. The eclipse
was at 9.6 deg. Regenerate with `tools/refraction/m4_analysis.py`.

## The rest

| file | what it is |
|---|---|
| `m3_quiver_maps.png` | nine quasi-static residual maps -- the wavefield, arrows in arcsec (`m3_maps.py`) |
| `m3_vertical_profiles.png` | vertical residual against altitude within the frame -- oscillatory, not polynomial (`m3_maps.py`) |
| `m3_stats.csv` | per-field statistics behind the two M3 figures |
| `platescale_vs_temperature.png`, `cubic_vs_temperature.png` | plate scale and cubic against both thermometers side by side (`final_charts.py`); superseded as the summary figure by `stability_vs_tube_temperature.png`, kept because REFRACTION_2026.md 16.1 cites them as the corrected pair |
| `m4_fields.csv` | the mosaic, per field |
| `m2_fieldwindow_summary.csv` | the nine horizon field-windows: slopes, offsets, differentials |
| `perframe_results.csv` | the per-frame reductions behind the temperature story |
| `band_cubic_results.csv`, `band_fields.csv` | the mosaic stability band: plate scale and free-cubic d(3000) |
| `INVENTORY.csv` | every frame in the refraction data, with its role |
| `REFRACTION_2026_STRATEGY.md` | the strategy as first written, 2026-08-26; the repository's REFRACTION_2026.md is the living version |
| `superseded_2026-08-27/night2_temperature.png` | the first temperature figure. **Withdrawn**: its cubic panel places the band points at ~29 C through a time-of-day wrap bug in the script's temperature lookup; the numbers it illustrated came from the corrected decomposition and stand (REFRACTION_2026.md 16.1). Kept so the erratum has something to point at |
"""


def copy_if_changed(src: str, dst: str, check: bool) -> str:
    if not os.path.exists(src):
        return "MISSING SOURCE"
    existed = os.path.exists(dst)
    if existed and filecmp.cmp(src, dst, shallow=False):
        return "unchanged"
    if not check:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return "updated" if existed else "new"


def write_if_changed(text: str, dst: str, check: bool) -> str:
    current = open(dst, encoding="utf-8").read() if os.path.exists(dst) else None
    if current == text:
        return "unchanged"
    if not check:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return "updated" if current is not None else "new"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args(argv)
    check = args.check

    if not os.path.isdir(OUT):
        print(f"output root not found: {OUT}", file=sys.stderr)
        return 2

    rows: list[tuple[str, str]] = []

    # 1. the record document
    src = os.path.join(REPO, "docs", "WHICH_REDUCTION_IS_THE_RECORD.md")
    dst = os.path.join(RECORD, "WHICH_REDUCTION_IS_THE_RECORD.md")
    rows.append(("RECORD/WHICH_REDUCTION_IS_THE_RECORD.md", copy_if_changed(src, dst, check)))
    stale = os.path.join(OUT, "WHICH_REDUCTION_IS_THE_RECORD.md")
    if os.path.exists(stale):
        rows.append(("MEE_output/WHICH_REDUCTION_IS_THE_RECORD.md",
                     "STALE hand-kept copy beside the prompts; the RECORD one is maintained"))

    # 2. the refraction set
    for rel_src, rel_dst in REFRACTION_FILES:
        rows.append((f"RECORD/refraction/{rel_dst}",
                     copy_if_changed(os.path.join(REFRACTION_SRC, rel_src),
                                     os.path.join(RECORD, "refraction", rel_dst), check)))
    rows.append(("RECORD/refraction/README.md",
                 write_if_changed(REFRACTION_INDEX, os.path.join(RECORD, "refraction", "README.md"), check)))

    width = max(len(r[0]) for r in rows)
    for name, status in rows:
        print(f"  {name:<{width}}  {status}")
    n_missing = sum(1 for _, s in rows if s.startswith("MISSING"))
    n_changed = sum(1 for _, s in rows if s in ("new", "updated"))
    verb = "would change" if check else "changed"
    print(f"{n_changed} {verb}, {n_missing} missing source(s)")
    return 1 if n_missing else 0


if __name__ == "__main__":
    sys.exit(main())
