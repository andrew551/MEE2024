"""Station 2's eclipse field, each exposure tier fitted on its own under both methods.

Douglas, 2026-09-08: the pooled Method 1 (2.523 arcsec) sits far from Method 2 (1.770), and
Method 1 is meant to be the tighter of the two. Per-tier fits separate three possibilities:
statistics, a tier-dependent problem, or the imported scale being wrong for the eclipse field.

Method 1 = the stage-2 runs with the plate scale fixed to the bracket value; L is fitted with
offset and rotation free and NO scale term.  Method 2 = scale free per tier.
For each tier the script also fits a free scale ON THE METHOD 1 residuals: that term is what
the field wants to add to the imported scale, in ppm, and it is the diagnostic.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.analysis_window import WINDOWS

OUT = r"D:/MEE2024 output/MEE_output/station2_transfer"
NX, NY, PS = 4656, 3520, 1.8672511
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2
_W = WINDOWS['mexico2024_station2']          # 2-10 R_sun, G <= 13; see the registry
MAGCUT, RMIN, RMAX = _W.mag, _W.rmin, _W.rmax
GR, NEWTON = 1.7512, 0.8756
TIERS = (('100ms', '0.100 s'), ('075ms', '0.075 s'))
SUB = {'m2': {'100ms': 'eclipse/100ms/stage2', '075ms': 'eclipse/075ms/stage2_rmt36'},
       'm1': {'100ms': 'eclipse/100ms/stage2_method1', '075ms': 'eclipse/075ms/stage2_method1'}}


def load(method, tag):
    f = glob.glob(os.path.join(OUT, SUB[method][tag], '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    d = pd.read_csv(f[0])
    d = d[d['magV'] <= MAGCUT].copy()
    d['rx'] = (d['px'] - SUNPX) * PS
    d['ry'] = (d['py'] - SUNPY) * PS
    d['R'] = np.hypot(d['rx'], d['ry'])
    d['Rsun'] = d['R'] / R_SUN_AS
    return d[(d['Rsun'] >= RMIN) & (d['Rsun'] <= RMAX)].reset_index(drop=True)


def solve(d, with_scale):
    """One tier: offset (2), rotation, optionally scale, and L."""
    n = len(d); Z = np.zeros(n); one = np.ones(n)
    px, py = d['px'].values, d['py'].values
    ux, uy = d['rx'].values / d['R'].values, d['ry'].values / d['R'].values
    cx = [one, Z, -(py - NY / 2) * PS]
    cy = [Z, one, (px - NX / 2) * PS]
    lab = ['N1', 'N2', 'Th']
    if with_scale:
        cx.append((px - NX / 2) * PS); cy.append((py - NY / 2) * PS); lab.append('S')
    cx.append(ux * R_SUN_AS / d['R'].values); cy.append(uy * R_SUN_AS / d['R'].values)
    lab.append('L')
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    y = np.concatenate([d['dx_arcsec'].values, d['dy_arcsec'].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    return dict(zip(lab, c)), float(np.sqrt(np.mean(resid ** 2)))


def boot(d, with_scale, draws=600, seed=7):
    rng = np.random.default_rng(seed)
    ids = d['ID'].unique(); out = []
    for _ in range(draws):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([d[d['ID'] == i] for i in pick], ignore_index=True)
        try:
            out.append(solve(s, with_scale)[0]['L'])
        except Exception:
            pass
    return float(np.std(out, ddof=1))


IMP = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, 'bracket_cubic', n, '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for n in ('right', 'left')]))
BR = {n: json.load(open(glob.glob(os.path.join(OUT, 'bracket_cubic', n, '**', 'distortion_results.txt'),
                                  recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)']
      for n in ('right', 'left')}
print('imported bracket scale %.7f "/px  (right %.7f, left %.7f, spread %.1f ppm)'
      % (IMP, BR['right'], BR['left'], 1e6 * abs(BR['left'] - BR['right']) / IMP))

print('\n%-9s %-8s %5s %5s | %-24s | %-24s | scale the M1 residuals still want'
      % ('tier', 'method', 'stars', 'obs', 'L (arcsec)', 'fit rms (arcsec)'))
rows = {}
for tag, lab in TIERS:
    for meth, with_scale, name in (('m1', False, 'Method 1'), ('m2', True, 'Method 2')):
        d = load(meth, tag)
        c, rms = solve(d, with_scale)
        s = boot(d, with_scale)
        extra = ''
        if meth == 'm1':
            cs, rms_s = solve(d, True)          # free the scale on the Method 1 residuals
            ppm = 1e6 * cs['S']
            extra = ('  %+.0f ppm -> L = %+.3f, rms %.3f' % (ppm, cs['L'], rms_s))
            rows[(tag, 'm1_free')] = (cs['L'], ppm)
        print('%-9s %-8s %5d %5d | %+7.3f +- %-13.3f | %-24.3f |%s'
              % (lab, name, d.ID.nunique(), len(d), c['L'], s, rms, extra))
        rows[(tag, meth)] = (c['L'], s, rms, c.get('S'))

# how much L moves per ppm of scale on this field, measured directly
d = load('m1', '100ms')
base = solve(d, False)[0]['L']
lev = []
for eps in (-200e-6, 200e-6):
    dd = d.copy()
    dd['dx_arcsec'] = dd['dx_arcsec'].values + eps * (dd['px'].values - NX / 2) * PS
    dd['dy_arcsec'] = dd['dy_arcsec'].values + eps * (dd['py'].values - NY / 2) * PS
    lev.append((solve(dd, False)[0]['L'] - base) / (eps * 1e6))
print('\nleverage on this field: %.4f arcsec of L per ppm of imported scale (%.3f arcsec per 100 ppm)'
      % (np.mean(lev), 100 * np.mean(lev)))
print('Method 2 joint scale minus imported: see the gap below')
for tag, lab in TIERS:
    L2, _, _, S2 = rows[(tag, 'm2')]
    print('  %-8s Method 2 free scale %+7.1f ppm from the reference; '
          'Method 1 residuals want %+7.1f ppm more than the bracket'
          % (lab, 1e6 * S2, rows[(tag, 'm1_free')][1]))
