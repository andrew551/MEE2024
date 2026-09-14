"""Does the pathway of record under-read a 1/r deflection?  Inject one and measure the response.

hu_absorption.py says, from the geometry alone, that fitting the eclipse field with the
quadratic free (the cell's Method 2 pathway: `fixed distortion order: quadratic`, scale free,
cubic and above from the zenith) lets only f = 0.86-0.90 of a 1/r deflection through to
stage 3, because the two shears and the six quadratics that stage 2 frees absorb the rest and
stage 3 does not refit them.  That is linear algebra on the star positions; this is the
pipeline itself.

THE TEST.  Take the gain-125 block's stage-1 centroids as they are, move every centroid
radially AWAY from the eclipse Sun (5043, 3386) px by an extra DL * R_sun / r -- an added
deflection of exactly DL = 2.000 arcsec on top of whatever the sky already carries -- and run
stage 2 and stage 3 unchanged.  The measured L must rise by 2.000 x f:

    rung                       predicted rise (hu_absorption.py, 84 stars)
    quadratic (record)         2.000 x 0.897 = 1.79 "
    constant + free scale      2.000 x 1.000 = 2.00 "   (control: everything stage 2 frees,
                                                        stage 3 refits)

Everything else -- reference, gates, cuts, site, time, the Method 2 estimator -- is the
record's own invocation (hu_step3._stage2_one, hu_union._stage3).  A rise of 2.00 on the
quadratic rung would mean hu_absorption.py is wrong; a rise near 1.79 means the record's
2.129 " is f x the sky's deflection.

Outputs under F:/MEE_output/husillos2026/absorption/.

    .venv/Scripts/python.exe tools/husillos2026/hu_inject.py
"""
import glob
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from analysis_window import WINDOWS  # noqa: E402
from hu_step3 import BLOCKS, PY, SITE, czip, refpath, results, run  # noqa: E402

WIN = WINDOWS['husillos2026']
HUS = r'F:\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'absorption')
PS, R_SUN_AS = 2.2028, 947.1
SUNPX, SUNPY = 5043.0, 3386.0
DL = 2.000                     # arcsec of deflection constant added
BLOCK = 'gain125'
RUNGS = (('quadratic', ['--set', 'distortion_fixed_coefficients=quadratic',
                        '--set', 'distortion_free_scale=True']),
         ('constant', ['--set', 'distortion_fixed_coefficients=constant',
                       '--set', 'distortion_free_scale=True']))


def inject(src_zip, dst_dir):
    """A copy of the stage-1 zip with every centroid pushed radially outward by DL R_sun / r."""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src_zip))
    if os.path.exists(dst):
        return dst
    zin = zipfile.ZipFile(src_zip)
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('STACKED_CENTROIDS_DATA.csv'):
                t = pd.read_csv(io.BytesIO(data), index_col=0)
                rx, ry = t['px'].values - SUNPX, t['py'].values - SUNPY
                r_px = np.hypot(rx, ry)
                shift_px = DL * R_SUN_AS / (r_px * PS) / PS      # arcsec -> px, 1/r in arcsec
                t['px'] = t['px'] + shift_px * rx / r_px
                t['py'] = t['py'] + shift_px * ry / r_px
                buf = io.StringIO()
                t.to_csv(buf)
                data = buf.getvalue().encode('utf-8')
                print('   injected %+.3f " of L into %d centroids: radial push %.3f-%.3f px'
                      % (DL, len(t), shift_px.min(), shift_px.max()))
            zout.writestr(item, data)
    return dst


def stage2(tag, cz, tmid, rung_args):
    d = os.path.join(OUT, 's2_' + tag)
    if not results(d):
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic',
             '--fix-distortion', refpath(), *rung_args,
             '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *SITE,
             '--set', 'observation_time=' + tmid,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results(d)
    z = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    return j, (z[0] if z else None)


def stage3(tag, z):
    d = os.path.join(OUT, 's3_' + tag)
    fs = glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True)
    if not fs:
        run([PY, '-m', 'mee2024.cli', 'eclipse', z,
             '--set', 'eclipse_method=Method 1 & 2',
             '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
             '--set', 'limit_radial_sun_radii=False',
             '--set', 'remove_double_stars_eclipse=False',
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage3.log'))
        fs = glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True)
    if not fs:
        return None
    txt = io.open(fs[0], encoding='utf-8', errors='replace').read()
    out = {}
    for m in re.finditer(r'Method (\d) results: L=([-\d.]+)\D+([\d.]+), platescale=([\d.]+)', txt):
        out['M' + m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    n = re.search(r'number of stars[^\d]*(\d+)', txt)
    out['n'] = int(n.group(1)) if n else -1
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tag, src, tmid = [b for b in BLOCKS if b[0] == BLOCK][0]
    cz0 = czip(src)
    cz1 = inject(cz0, os.path.join(OUT, 's1_inject_' + tag))
    print()
    print('%-28s %5s %8s %10s %14s %14s %9s' % ('run', 'N', 'rms(")', 'ps ("/px)',
                                                 'Method 1 L', 'Method 2 L', 'M2 rise'))
    for rung, args in RUNGS:
        base = None
        for kind, cz in (('original', cz0), ('injected', cz1)):
            name = '%s_%s_%s' % (tag, rung, kind)
            j, z = stage2(name, cz, tmid, args)
            if not j or not z:
                print('%-28s stage 2 FAILED' % name)
                continue
            assert j['fixed distortion order'] == rung, j['fixed distortion order']
            r = stage3(name, z)
            if not r:
                print('%-28s stage 3 FAILED' % name)
                continue
            m1, m2 = r.get('M1', (np.nan,) * 3), r.get('M2', (np.nan,) * 3)
            rise = '' if base is None else '%+.3f' % (m2[0] - base)
            if base is None:
                base = m2[0]
            print('%-28s %5d %8.4f %10.7f %6.3f +- %.3f %6.3f +- %.3f %9s'
                  % (name, j['#stars used'], j['final rms error (arcseconds)'],
                     j['platescale (arcseconds/pixel)'], m1[0], m1[1], m2[0], m2[1], rise))
    print()
    print('injected %.3f ".  Predicted rise on the quadratic rung: %.3f x f; hu_absorption.py'
          % (DL, DL))
    print('gives f = 0.897 for this block (1.794 "), and 1.000 (2.000 ") for the constant rung.')


if __name__ == '__main__':
    main()
