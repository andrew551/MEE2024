"""Re-run Don Bruns' zenith calibration fields through today's MEE, and compare with Astrometrica.

Douglas, 2026-09-15: "can we actually run this on the data in I:\\Don Bruns 2600MM tests" --
then "I:\\Don Bruns 2024 ... has both MEE and Astrometrica results. These were done with cubic
correction. Let's see how close the results are this time."

Both folders hold complete stage-1 archives, so stage 2 can be run again rather than trusting a
2024 results file, and the four 2024 fields multiply one comparison into five.

  --set 2600mm   one field (HIP 29696), MEE quintic in 2024
  --set 2024     four fields (HIP 29696, 31096, 32740, 33018), MEE cubic in 2024

WHY RE-RUNNING MATTERS RATHER THAN READING THE STORED FITS.  The 2600MM field compared at
0.034 " against Astrometrica from its stored 2024 quintic and 0.018 " once re-fitted at cubic
with today's pipeline. The stored 2024 cubic fits in the other folder compare at 0.064 ". Either
the pipeline has improved since, or those fields differ; re-running them is what tells the two
apart, and it is the same archives either way.

Three things the archives need. Their members sit under a `data/` prefix where today's reader
expects them at the root, so they are repackaged. `I:` is READ ONLY, so everything is written to
`F:\\MEE_output\\bruns_rerun`. And the settings are READ from each field's own 2024 results file
-- magnitude cut, tolerance, observation date, and the order inferred from the coefficient count
-- rather than chosen here.

    .venv/Scripts/python.exe tools/bruns_rerun.py [--set 2024|2600mm] [--order cubic|as2024]
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))
from tools.astrometrica_compare import compare, pairs_2024, SETS  # noqa: E402

PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
OUT = r'F:\MEE_output\bruns_rerun'
ORDER_OF = {6: 'quadratic', 10: 'cubic', 15: 'quartic', 21: 'quintic'}


def repackage(src_zip, tag):
    """The stage-1 archive with its `data/` prefix stripped, on F:."""
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, 'centroid_data_%s.zip' % tag)
    if os.path.isfile(dst):
        return dst
    with zipfile.ZipFile(src_zip) as z, zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as out:
        for n in z.namelist():
            if not n.endswith('/'):
                out.writestr(n.split('/', 1)[1] if n.startswith('data/') else n, z.read(n))
    return dst


def jobs(which):
    """(tag, stage-1 zip, Astrometrica log, 2024 MEE results) per field."""
    # the tag prefixes the SET, because both folders contain a HIP 29696 and an unprefixed
    # tag made the 2600mm run silently reuse the 2024 field's output folder
    if which == '2600mm':
        f = SETS['2600mm']
        return [('2600mm_HIP29696', os.path.join(f, 'data.zip'),
                 os.path.join(f, 'Walter HIP29696 MEE2024float.txt'),
                 os.path.join(f, 'output', 'data20240320154050distortion_results.txt'))]
    f = SETS['2024']
    out = []
    for hip, log, mee in pairs_2024(f):
        z = glob.glob(os.path.join(f, 'HIP*%s*data.zip' % hip))
        if not z:
            print('HIP %s: no stage-1 archive, skipped' % hip)
            continue
        out.append(('2024_HIP' + hip, z[0], log, mee))
    return out


def rerun(tag, zsrc, old_results, order_arg):
    j24 = json.load(io.open(old_results, encoding='utf-8'))
    order24 = ORDER_OF[len(j24['distortion coeffs x'])]
    order = order24 if order_arg == 'as2024' else order_arg
    z = repackage(zsrc, tag)
    d = os.path.join(OUT, tag, 'stage2_' + order)
    if not glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True):
        os.makedirs(d, exist_ok=True)
        cmd = [PY, '-m', 'mee2024.cli', 'distortion', z, '--order', order,
               '--set', 'distortion_fixed_coefficients=None',
               '--set', 'max_star_mag_dist=%g' % j24['star max magnitude'],
               '--set', 'distortion_fit_tol=%g' % j24['error tolerance (as)'],
               '--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
               '--set', 'observation_date=' + j24['observation_date'],
               '--no-display', '--quiet', '-o', d]
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    f = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return (f[0] if f else None), j24, order24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='2024', choices=sorted(SETS))
    ap.add_argument('--order', default='cubic',
                    help="MEE's fit order (default cubic, matching Astrometrica and MEE's own "
                         'default); `as2024` uses whatever each 2024 run used')
    a = ap.parse_args()

    todo = jobs(a.set)
    print('re-running %d field(s) from %s through today\'s MEE, order %s'
          % (len(todo), SETS[a.set], a.order))
    print()
    print('%-10s %-24s %-24s %s' % ('', 'stored 2024 fit', 'today', ''))
    print('%-10s %8s %7s %6s %8s %7s %6s %10s'
          % ('field', 'stars', 'rms "', 'TAN "', 'stars', 'rms "', 'TAN "', 'ppm scale'))
    rows = []
    for tag, zsrc, log, old in todo:
        new, j24, order24 = rerun(tag, zsrc, old, a.order)
        if not new:
            print('%-10s FAILED' % tag)
            continue
        r24 = compare(log, old, verbose=False)
        rnew = compare(log, new, verbose=False)
        jn = json.load(io.open(new, encoding='utf-8'))
        dps = 1e6 * (jn['platescale (arcseconds/pixel)'] - j24['platescale (arcseconds/pixel)']) \
            / j24['platescale (arcseconds/pixel)']
        rows.append((r24, rnew))
        print('%-10s %8d %7.4f %6.4f %8d %7.4f %6.4f %10.1f'
              % (tag, r24['mee_stars'], r24['mee_rms'], r24['tan'],
                 rnew['mee_stars'], rnew['mee_rms'], rnew['tan'], dps))
    if len(rows) > 1:
        print('%-10s %8s %7.4f %6.4f %8s %7.4f %6.4f'
              % ('MEAN', '', np.mean([a_['mee_rms'] for a_, _ in rows]),
                 np.mean([a_['tan'] for a_, _ in rows]), '',
                 np.mean([b['mee_rms'] for _, b in rows]),
                 np.mean([b['tan'] for _, b in rows])))
    print()
    print('TAN " is the residual against Astrometrica after the gauge conversion and the')
    print('linear (Jacobian) step -- the smaller the better.')


if __name__ == '__main__':
    main()
