"""Method 1 driven by the RIGHT bracket alone and the LEFT bracket alone, not their mean.

Douglas, 2026-09-08: the two tiers agree once they are handed the same scale, so pooling them is
fine -- but the imported scale itself must be wrong, because the answer excludes GR. The bracket
is a mean of two fields taken on either side of totality; if one of them was disrupted (cloud,
scattered coronal light, mount drift) the mean carries half of that disruption. So fit L three
ways, on the pooled sample each time, and look at the two bracket fields themselves.

Bruns' design intends the mean: the linear atmospheric differential between the two sides
cancels in the average, and the L-R split bounds what is left (docs/MATRIX_2026.md, cell 1). That
argument only holds if both sides are sound.

All three runs use the quadratic-free bracket, which is the convention Bruns and Leon used
(s2_bracket_convention.py).

  .venv/Scripts/python.exe tools/matrix_station2/s2_bracket_lr_split.py
"""
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.analysis_window import WINDOWS

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/mexico2024/station2"
ECL = os.path.join(OUT, "eclipse")
NX, NY, PS = 4656, 3520, 1.8672511
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2
_W = WINDOWS['mexico2024_station2']          # 2-10 R_sun, G <= 13; see the registry
MAGCUT, RMIN, RMAX = _W.mag, _W.rmin, _W.rmax
GR, NEWTON = 1.7512, 0.8756
TIERS = (("100ms", "18:12:07", "100"), ("075ms", "18:13:20", "36"))
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
        '--set', 'observation_wavelength=0.633', '--set', 'observation_temp=15.2',
        '--set', 'observation_humidity=0.24']


def res_path(d):
    g = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return g[0] if g else None


def load(sub, tag):
    f = glob.glob(os.path.join(ECL, tag, sub, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if not f:
        return None
    d = pd.read_csv(f[0])
    d = d[d['magV'] <= MAGCUT].copy()
    d['rx'] = (d['px'] - SUNPX) * PS
    d['ry'] = (d['py'] - SUNPY) * PS
    d['R'] = np.hypot(d['rx'], d['ry'])
    d['Rsun'] = d['R'] / R_SUN_AS
    d['tier'] = tag
    return d[(d['Rsun'] >= RMIN) & (d['Rsun'] <= RMAX)].reset_index(drop=True)


def pooled_L(d):
    """Per-tier offset and rotation, one L, no scale -- Method 1 pooled over both tiers."""
    n = len(d); Z = np.zeros(n)
    px, py = d['px'].values, d['py'].values
    ux, uy = d['rx'].values / d['R'].values, d['ry'].values / d['R'].values
    cx, cy = [], []
    for tag, _, _ in TIERS:
        m = (d['tier'].values == tag).astype(float)
        cx += [m, Z, -m * (py - NY / 2) * PS]
        cy += [Z, m, m * (px - NX / 2) * PS]
    cx.append(ux * R_SUN_AS / d['R'].values); cy.append(uy * R_SUN_AS / d['R'].values)
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    y = np.concatenate([d['dx_arcsec'].values, d['dy_arcsec'].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c[-1], float(np.sqrt(np.mean(r ** 2)))


def boot(d, draws=600, seed=7):
    rng = np.random.default_rng(seed); ids = d['ID'].unique(); out = []
    for _ in range(draws):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([d[d['ID'] == i] for i in pick], ignore_index=True)
        if s['tier'].nunique() < 2:
            continue
        try:
            out.append(pooled_L(s)[0])
        except Exception:
            pass
    return float(np.std(out, ddof=1))


VARIANTS = (('right', ('right',), 'stage2_m1_right'),
            ('left', ('left',), 'stage2_m1_left'),
            ('mean of both', ('right', 'left'), 'stage2_method1_quadfree'))

print('bracket fields, quadratic-free:')
scales = {}
for side in ('right', 'left'):
    j = json.load(open(res_path(os.path.join(OUT, 'bracket_quadfree', side)), encoding='utf-8'))
    scales[side] = j['platescale (arcseconds/pixel)']
    print('  %-5s %3d stars  rms %.4f"  ps %.7f' % (side, j['#stars used'],
                                                    j['final rms error (arcseconds)'], scales[side]))
mean = 0.5 * (scales['right'] + scales['left'])
print('  L-R split %.1f ppm (half-width %.1f ppm = %.2f " of L at 0.0165 "/ppm)'
      % (1e6 * abs(scales['left'] - scales['right']) / mean,
         0.5e6 * abs(scales['left'] - scales['right']) / mean,
         0.0165 * 0.5e6 * abs(scales['left'] - scales['right']) / mean))

print('\n%-14s %-11s %-24s %-9s %s' % ('imported from', 'scale ("/px)', 'L pooled (arcsec)', 'fit rms', 'vs GR'))
for name, sides, sub in VARIANTS:
    for tag, tm, rmt in TIERS:
        d = os.path.join(ECL, tag, sub)
        os.makedirs(d, exist_ok=True)
        if not res_path(d):
            refs = [res_path(os.path.join(OUT, 'bracket_quadfree', s)) for s in sides]
            cmd = [PY, '-m', 'mee2024.cli', 'distortion',
                   glob.glob(os.path.join(ECL, tag, 'centroid_data*.zip'))[0],
                   '--order', 'cubic', '--fix-distortion', *refs,
                   '--set', 'distortion_fixed_coefficients=constant',
                   '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                   '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=' + rmt,
                   *SITE, '--set', 'observation_time=' + tm,
                   '--no-display', '--quiet', '-o', d]
            with open(os.path.join(d, 'stage2.log'), 'w') as fh:
                subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    D = pd.concat([load(sub, t) for t, _, _ in TIERS], ignore_index=True)
    L, rms = pooled_L(D)
    s = boot(D)
    ps_used = scales[sides[0]] if len(sides) == 1 else mean
    print('%-14s %.7f   %+7.3f +- %-13.3f %.3f"     %+.1f sigma'
          % (name, ps_used, L, s, rms, (L - GR) / s))
