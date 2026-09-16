"""Write the tangent-plane coefficients for a stage-2 fit that has already been run.

Douglas, 2026-09-15: "Where can I find an example output?"

A fresh reduction writes `distortion_results_TAN.txt` beside its own `distortion_results.txt`
automatically. Every fit made before 2026-09-15 has no such file, and there are hundreds of
them. This converts any of them without re-reducing, because the conversion needs nothing the
stage-2 file does not already carry -- the coefficients, the plate scale, the order -- plus the
sensor size.

FINDING THE SENSOR SIZE, which is the only awkward part. `distortion_results.txt` does not
record it. It does record `source_data`, the stage-1 archive, which carries `img_shape`; but on
this machine those paths were written before the output tree moved off `D:` on 2026-09-14, so
they are stale and are remapped here. Failing that, a `centroid_data*.zip` beside the fit is
used, and failing that `--shape`.

    .venv/Scripts/python.exe tools/tan_export.py <path> [<path> ...] [--out DIR] [--shape NY,NX]

Each <path> may be a distortion_results.txt, a distortion_data*.zip, or a folder to search.
"""
import argparse
import glob
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mee2024 import distortion_polynomial as dp  # noqa: E402
import numpy as np  # noqa: E402

#: the output tree moved on 2026-09-14; paths recorded before then name the old roots
REMAP = [(r'D:\MEE2024 output\MEE_output', r'F:\MEE_output'),
         (r'D:/MEE2024 output/MEE_output', r'F:/MEE_output'),
         (r'D:\MEE_output', r'F:\MEE_output')]


def results_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, '**', 'distortion_results.txt'),
                                recursive=True))
    if path.endswith('.zip'):
        return [path]
    return [path]


def load(path):
    if path.endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            return json.load(z.open('distortion_results.txt'))
    return json.load(io.open(path, encoding='utf-8'))


def shape_from_stage1(j, near):
    """(ny, nx) from the stage-1 archive this fit came from, or None."""
    cand = []
    src = j.get('source_data')
    if src:
        cand.append(src)
        for old, new in REMAP:
            if src.startswith(old):
                cand.append(new + src[len(old):])
    for up in (near, os.path.dirname(near), os.path.dirname(os.path.dirname(near)),
               os.path.dirname(os.path.dirname(os.path.dirname(near)))):
        cand += glob.glob(os.path.join(up, 'centroid_data*.zip'))
    for c in cand:
        if c and os.path.isfile(c):
            try:
                with zipfile.ZipFile(c) as z:
                    # older archives put everything under a `data/` prefix, which is why
                    # distortion_fitter tries both; without the second name every 2024-era
                    # archive reported "sensor size not found" even when it was sitting there
                    member = 'results.txt' if 'results.txt' in z.namelist() else 'data/results.txt'
                    s = json.load(io.TextIOWrapper(z.open(member), encoding='utf-8',
                                                   errors='replace')).get('img_shape')
                if s:
                    return int(s[0]), int(s[1])
            except Exception:                                   # noqa: BLE001
                continue
    return None


def _name_for(f):
    """A distinguishing name for --out, since every results file lives in a folder called
    `distortion` under a folder called `stage2` -- taking the parent alone collided all three
    example fields onto one filename."""
    parts = os.path.abspath(f).split(os.sep)[:-1]
    skip = {'distortion', 'stage2', 'mee_output'}
    keep = [p for p in parts if p.lower() not in skip
            and not p.startswith('DISTORTION_OUTPUT') and ':' not in p]
    return '_'.join(keep[-3:]).replace(' ', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default=None, help='write here instead of beside the fit')
    ap.add_argument('--shape', default=None, help='NY,NX if it cannot be found automatically')
    a = ap.parse_args()
    forced = tuple(int(v) for v in a.shape.split(',')) if a.shape else None
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    for path in a.paths:
        for f in results_files(path):
            try:
                j = load(f)
            except Exception as e:                              # noqa: BLE001
                print('%s: cannot read (%s)' % (f, e)); continue
            if j.get('gauge') == dp.TAN_GAUGE_MARK:
                print('%s: already a TAN export, skipped' % f); continue
            shape = forced or shape_from_stage1(j, os.path.dirname(os.path.abspath(f)))
            if not shape:
                print('%s: sensor size not found -- pass --shape NY,NX' % f); continue
            order = j['distortion order']
            opts = {'distortionOrder': order, 'distortion_fixed_coefficients': 'None'}
            names = dp.get_coeff_names(opts)
            cx = [j['distortion coeffs x'].get(n, 0.0) for n in names]
            cy = [j['distortion coeffs y'].get(n, 0.0) for n in names]
            q = (np.radians(j['platescale (arcseconds/pixel)'] / 3600.0),
                 np.radians(j['RA']), np.radians(j['DEC']), np.radians(j['ROLL']))
            tan = dp.tangent_plane_coefficients(q, cx, cy, shape, opts)
            tan['converted_from'] = os.path.abspath(f)
            tan['converted_by'] = 'tools/tan_export.py (the fit itself was not re-run)'
            dest = (os.path.join(a.out, _name_for(f) + '_TAN.txt') if a.out else
                    os.path.join(os.path.dirname(os.path.abspath(f)),
                                 'distortion_results_TAN.txt'))
            with io.open(dest, 'w', encoding='utf-8') as fp:
                json.dump(tan, fp, sort_keys=False, indent=4)
            print('%s  (%dx%d, %s)  gauge %.4f "/deg^3, %.2f px at the corner\n   -> %s'
                  % (os.path.basename(os.path.dirname(f)), shape[1], shape[0], order,
                     tan['gauge term radial cubic (arcsec/deg^3)'],
                     tan['gauge term at the corner (pixels)'], dest))


if __name__ == '__main__':
    main()
