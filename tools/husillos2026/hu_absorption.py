"""How much of a 1/r deflection survives cell 4's stage-2 pathway to reach stage 3?

The pathway of record fits the eclipse field against the zenith with the QUADRATIC FREE
(`fixed distortion order: quadratic`, `distortion_free_scale: true`, read back from both
blocks' distortion_results.txt): twelve free polynomial terms -- translation, rotation,
scale, two shears, six quadratics.  Stage 3 (mee2024/eclipse_analysis.py, mode 2) then
refits translation and rotation (a 3-D rotation of the unit vectors) and the scale (the r
column of [1/r, r]) jointly with L.  Whatever part of the 1/r pattern the two shears and the
six quadratics absorbed in stage 2 is NOT refit in stage 3, so L comes out low by that
fraction.  On a Sun at the field centre with a symmetric star set the fraction is zero (1/r is
odd about the Sun, a quadratic is even); on a real star set it is whatever the geometry says.

The same absorption acts on the field-to-zenith null (hu_atmosphere.py), which is why the
two are read together: the null charges the pathway what the pathway does to a field with no
deflection, and this number says what the pathway does to the deflection itself.

For each block's matched list and for the two-witness union: a unit-L deflection field about
the eclipse Sun (5043, 3386) px at the science cuts; project out the stage-2 freedoms; refit
[N1, N2, Th, S, L]; report L per unit L.  `constant` + free scale (Station 1's pathway) is
the control -- everything it frees is refit in stage 3, so it must return 1.000.

    .venv/Scripts/python.exe tools/husillos2026/hu_absorption.py
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from analysis_window import WINDOWS  # noqa: E402

HUS = r'F:\MEE_output\husillos2026'
WIN = WINDOWS['husillos2026']
NX, NY = 9576, 6388
PS, R_SUN_AS = 2.2028, 947.1
SUNPX, SUNPY = 5043.0, 3386.0
W_NORM = NX / 2.0
L_RECORD, L_RECORD_ERR = 2.129, 0.430


def matched(pattern):
    z = glob.glob(pattern, recursive=True)
    if not z:
        return None
    zf = zipfile.ZipFile(z[0])
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    return t


def stage2_freedoms(px, py, kind):
    """Columns of what stage 2 leaves free, block-diagonal in (dx, dy)."""
    xs, ys = (px - NX / 2) / W_NORM, (py - NY / 2) / W_NORM
    n = len(px)
    one, Z = np.ones(n), np.zeros(n)
    if kind == 'constant + free scale':
        # translation per axis, one isotropic scale
        cols = [np.r_[one, Z], np.r_[Z, one], np.r_[xs, ys]]
    else:
        polys = [one, xs, ys]
        if kind == 'quadratic':
            polys += [xs * xs, xs * ys, ys * ys]
        cols = [np.r_[p, Z] for p in polys] + [np.r_[Z, p] for p in polys]
    return np.column_stack(cols)


def stage3_design(px, py, rx, ry, R):
    n = len(px)
    Z = np.zeros(n)
    ux, uy = rx / R, ry / R
    cols_x = [np.ones(n), Z, -(py - NY / 2) * PS, (px - NX / 2) * PS, ux * R_SUN_AS / R]
    cols_y = [Z, np.ones(n), (px - NX / 2) * PS, (py - NY / 2) * PS, uy * R_SUN_AS / R]
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])


def survival(px, py, kind):
    rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS
    R = np.hypot(rx, ry)
    ell = np.r_[rx / R * R_SUN_AS / R, ry / R * R_SUN_AS / R]      # arcsec per unit L
    A = stage2_freedoms(px, py, kind)
    c, *_ = np.linalg.lstsq(A, ell, rcond=None)
    left = ell - A @ c                                              # what stage 3 sees
    B = stage3_design(px, py, rx, ry, R)
    k, *_ = np.linalg.lstsq(B, left, rcond=None)
    return k[-1]


def cuts(t):
    t = t[t['magV'] <= WIN.mag]
    px, py = t['px'].values.astype(float), t['py'].values.astype(float)
    R = np.hypot((px - SUNPX) * PS, (py - SUNPY) * PS)
    keep = R > WIN.rmin * R_SUN_AS
    return px[keep], py[keep]


def pipeline_f():
    """f as the pipeline actually produces it for the union of record.

    Stage 2's quadratic is fitted on EVERY star the block matched (84 and 74), not on the 63
    the union keeps, so the absorbed part is the projection over the block's own stars,
    evaluated at the union's; the union then takes the per-star mean of the two blocks'
    displacements (median of two) and stage 3 fits the 63.  hu_inject.py measured this on
    the gain-125 block with the pipeline itself: 2.000 " in, 1.787 " out.
    """
    uz = os.path.join(HUS, 'step3', 'union', 'union_2witness.zip')
    if not os.path.exists(uz):
        return None
    U = matched(uz)
    U['ID'] = U['ID'].astype(str).str.strip()
    ids = set(U['ID'])
    left = {}
    per_block = {}
    for name in ('gain125', 'gain0'):
        t = matched(os.path.join(HUS, 'step3', 'eclipse_%s' % name, '**', 'distortion_data*.zip'))
        t['ID'] = t['ID'].astype(str).str.strip()
        px, py = t['px'].values.astype(float), t['py'].values.astype(float)
        rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS
        R = np.hypot(rx, ry)
        ell = np.r_[rx / R * R_SUN_AS / R, ry / R * R_SUN_AS / R]
        A = stage2_freedoms(px, py, 'quadratic')
        c, *_ = np.linalg.lstsq(A, ell, rcond=None)
        res = ell - A @ c
        n = len(px)
        # the block's own stage 3, on its science-cut stars, after a fit on all of them
        keep = (t['magV'].values <= WIN.mag) & (R > WIN.rmin * R_SUN_AS)
        B = stage3_design(px[keep], py[keep], rx[keep], ry[keep], R[keep])
        k, *_ = np.linalg.lstsq(B, np.r_[res[:n][keep], res[n:][keep]], rcond=None)
        per_block[name] = (int(keep.sum()), k[-1])
        for i, sid in enumerate(t['ID'].values):
            if sid in ids:
                left.setdefault(sid, []).append((px[i], py[i], res[i], res[n + i]))
    rows = [(sid, np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]),
             np.mean([v[2] for v in vals]), np.mean([v[3] for v in vals]))
            for sid, vals in left.items() if len(vals) == 2]
    px, py, dx, dy = (np.array([r[i] for r in rows]) for i in (1, 2, 3, 4))
    rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS
    R = np.hypot(rx, ry)
    B = stage3_design(px, py, rx, ry, R)
    k, *_ = np.linalg.lstsq(B, np.r_[dx, dy], rcond=None)
    return per_block, len(rows), k[-1]


def main():
    sets = [('gain 125 block', matched(os.path.join(HUS, 'step3', 'eclipse_gain125', '**',
                                                    'distortion_data*.zip'))),
            ('gain 0 block', matched(os.path.join(HUS, 'step3', 'eclipse_gain0', '**',
                                                  'distortion_data*.zip')))]
    # the Method 2 union of record lives under step3/union/host_<block>/ (hu_union.py);
    # the Method 1 unions are under m1*_host_<block>/ and are not the pathway in question
    uni = sorted(glob.glob(os.path.join(HUS, 'step3', 'union', '**', 'union_2witness*.zip'),
                           recursive=True))
    for u in uni:
        rel = os.path.relpath(u, os.path.join(HUS, 'step3', 'union'))
        if 'm1' in rel or 'method2_' in rel:
            continue
        sets.append(('two-witness union (%s)' % rel[:34], matched(u)))
    print('fraction of a 1/r deflection reaching stage 3, per unit L, at G <= %g, R > %g R_sun'
          % (WIN.mag, WIN.rmin))
    print('%-46s %5s %22s %10s %10s' % ('star set', 'N', 'constant + free scale', 'linear',
                                        'quadratic'))
    worst = 1.0
    for name, t in sets:
        if t is None:
            print('%-46s  (not found)' % name)
            continue
        px, py = cuts(t)
        f = [survival(px, py, k) for k in ('constant + free scale', 'linear', 'quadratic')]
        worst = min(worst, f[2])
        print('%-46s %5d %22.4f %10.4f %10.4f' % (name, len(px), *f))
    print()
    pf = pipeline_f()
    if pf:
        per_block, n_u, f_u = pf
        print('AS THE PIPELINE DOES IT (stage-2 fit on every matched star, stage 3 on the cut):')
        for name, (n, f) in per_block.items():
            print('   %-12s stage 3 on %3d stars   f = %.4f%s'
                  % (name, n, f, '   (hu_inject.py measured 1.787 / 2.000 = 0.894)'
                     if name == 'gain125' else ''))
        print('   %-12s stage 3 on %3d stars   f = %.4f   <- the union of record'
              % ('union', n_u, f_u))
        print()
        print('L_record %.3f +- %.3f " (Method 2, two-witness union) is f x the sky\'s deflection:'
              % (L_RECORD, L_RECORD_ERR))
        print('undone, %.3f +- %.3f ".  Whether to undo it is a record decision, not this'
              % (L_RECORD / f_u, L_RECORD_ERR / f_u))
        print('tool\'s; the field-to-zenith null (hu_atmosphere.py) is read at the same fraction.')


if __name__ == '__main__':
    main()
