"""A telescope with NO distortion at all, put through the real pipeline.

Douglas, 2026-09-16: "Can you create a simulation of a perfect telescope optics and generate the
corresponding distortion_results and distortion_field in both the MEE and TANGENT gauges. Use
cubic polynomial and assume the same 2600MM camera."

WHAT A PERFECT OPTIC IS.  A flawless lens forms a GNOMONIC image: a star in direction (X, Y, Z)
about the boresight lands at tangent-plane coordinates (Y/X, Z/X), and nowhere else.  So the
simulation takes real catalogue positions, projects them gnomonically with a real plate
solution, and writes those as the measured pixel positions.  Nothing else about the frame is
invented: the same stars, the same field, the same plate scale, only the optics made perfect.

WHAT IT SHOULD SHOW, and why it is worth running.  In the TANGENT gauge the fit must come back
flat -- there is no distortion to find.  In MEE's own gauge it must come back with the
projection term, ~1.5 arcsec into the corners of this sensor, because MEE's frame is
(declination, RA x cos(dec)) and not a tangent plane.  The pair of charts is the cleanest
possible statement of what `Distortion_field.png` measures from, with no real optics in the way.

The stars are taken from a real reduction's CATALOGUE_MATCHED_ERRORS, which carries the
catalogue positions already brought to the observation epoch, so the synthetic frame matches
against the same catalogue the real one did.  Stage 2 then plate-solves and matches it from
scratch, exactly as it would a real capture: nothing here hands it the answer.

    .venv/Scripts/python.exe tools/simulate_perfect_optic.py [--order cubic]
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
#: a real reduction of Bruns' 2600MM field, for its stars and its plate solution
REAL = r'F:\MEE_output\bruns_rerun\2600mm_HIP29696\stage2_cubic'
OUT = r'F:\MEE_output\perfect_optic'


def one(pattern):
    m = glob.glob(pattern, recursive=True)
    if not m:
        raise SystemExit('not found: %s\nrun tools/bruns_rerun.py --set 2600mm first' % pattern)
    return m[0]


def ideal_pixels(ra_deg, dec_deg, q, shape):
    """Where a PERFECT (gnomonic) optic puts each catalogue position, in pixels.

    The rotation is the pipeline's own (`transforms.detransform_vectors`); only the projection
    differs -- tangent plane instead of (dec, RA cos dec).
    """
    scale, ra, dec, roll = q
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    v = np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    rot = Rotation.from_euler('zyx', [-ra, dec, -roll]).apply(v)
    col = rot[:, 1] / rot[:, 0] / scale
    row = rot[:, 2] / rot[:, 0] / scale
    return col + shape[1] / 2.0, row + shape[0] / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', default='cubic')
    a = ap.parse_args()

    res = json.load(io.open(one(os.path.join(REAL, '**', 'distortion_results.txt')),
                            encoding='utf-8'))
    zf = zipfile.ZipFile(one(os.path.join(REAL, '**', 'distortion_data*.zip')))
    name = [n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(name))
    t.columns = [c.strip() for c in t.columns]

    src = one(os.path.join(os.path.dirname(REAL), 'centroid_data*.zip'))
    with zipfile.ZipFile(src) as z:
        s1 = json.load(io.TextIOWrapper(z.open('results.txt'), encoding='utf-8',
                                        errors='replace'))
    shape = s1['img_shape']
    # distortion_fitter writes ROLL as degrees(q[3]) - 180 ("this dodgy +/- 180 thing", its own
    # comment), so the fitter's roll has to be put back before the rotation is rebuilt. Without
    # this the stars land on the opposite side of the field and the sanity check below trips at
    # ~4300 px, which is how the slip was caught.
    q = (np.radians(res['platescale (arcseconds/pixel)'] / 3600.0),
         np.radians(res['RA']), np.radians(res['DEC']), np.radians(res['ROLL'] + 180.0))

    px, py = ideal_pixels(t['RA(catalog)'].values, t['DEC(catalog)'].values, q, shape)
    # sanity: the perfect positions must sit within a few pixels of the real ones, since the
    # only difference is the distortion. A convention slip would show up here as a huge number.
    off = np.hypot(px - t['px'].values, py - t['py'].values)
    print('%d stars, sensor %dx%d, %.5f "/px'
          % (len(t), shape[1], shape[0], res['platescale (arcseconds/pixel)']))
    print('perfect minus real position: median %.2f px, max %.2f px  (the real distortion)'
          % (np.median(off), off.max()))
    if off.max() > 50:
        raise SystemExit('that is too large to be distortion -- check the projection convention')

    os.makedirs(OUT, exist_ok=True)
    arc = os.path.join(OUT, 'centroid_data_perfect.zip')
    with zipfile.ZipFile(arc, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('results.txt', json.dumps(dict(
            s1, n_centroids=len(px), platesolved=False,
            source_files=['SIMULATED: a perfect gnomonic optic, tools/simulate_perfect_optic.py'],
            starttime='perfect')))
        z.writestr('STACKED_CENTROIDS_DATA.csv',
                   pd.DataFrame({'px': px, 'py': py,
                                 'area (pixels)': 20.0,
                                 'flux (noise-normed)': 10 ** (-0.4 * t['magV'].values) * 1e7
                                 }).to_csv())
    d = os.path.join(OUT, 'stage2_' + a.order)
    if not glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True):
        os.makedirs(d, exist_ok=True)
        cmd = [PY, '-m', 'mee2024.cli', 'distortion', arc, '--order', a.order,
               '--set', 'distortion_fixed_coefficients=None',
               '--set', 'max_star_mag_dist=13',
               '--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
               '--set', 'observation_date=' + res['observation_date'],
               '--no-display', '--quiet', '-o', d]
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    f = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    if not f:
        print('FAILED -- tail of the log:')
        print(open(os.path.join(d, 'stage2.log')).read()[-1200:])
        sys.exit(1)
    j = json.load(io.open(f[0], encoding='utf-8'))
    print()
    print('the fit on a perfect optic: %d stars, rms %.4f ", plate scale %.5f "/px'
          % (j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)']))

    sys.path.insert(0, REPO)
    from mee2024 import distortion_polynomial as dp
    o = {'distortionOrder': a.order, 'distortion_fixed_coefficients': 'None'}
    names = dp.get_coeff_names(o)
    for lbl, path in (('MEE gauge      ', f[0]),
                      ('TANGENT gauge  ', os.path.join(os.path.dirname(f[0]),
                                                       'distortion_results_TAN.txt'))):
        if not os.path.isfile(path):
            print('%s missing' % lbl); continue
        dd = json.load(io.open(path, encoding='utf-8'))
        X, Y, DX, DY = dp.distortion_field([dd['distortion coeffs x'][n] for n in names],
                                           [dd['distortion coeffs y'][n] for n in names],
                                           shape, o)
        m = np.hypot(DX, DY) * dd['platescale (arcseconds/pixel)']
        print('   %s peak %6.3f "   rms %6.3f "' % (lbl, m.max(), np.sqrt((m ** 2).mean())))
    print()
    print('charts and results in %s' % d)


if __name__ == '__main__':
    main()
