"""Does the error really rise beyond mag 11? Douglas' 2017-era finding, tested.

Douglas, 2026-08-31: "I found in analysing the 2017 data that it was better to stay below
mag 11 as the error started to go up after that, presumably because of S/N issues with
the dimmer centroids."

The project's cuts are already at that limit -- `eclipse_limiting_mag` 11 in stage 3, and
every union in this matrix cuts at G 11.0 -- but the limit has been inherited rather than
re-measured on this data. This scans it: rebuild the cell-1 union from a catalogue deep
enough to admit faint stars, then refit L at a series of magnitude cuts and watch both the
statistical error and the per-star residual scatter.

Two quantities, because they answer different halves of the question:
  * the bootstrap error on L -- what the extra stars buy;
  * the rms residual of the stars in the faint bin -- whether they are worth having.
More stars always reduce a bootstrap error if they carry signal; the test of Douglas'
finding is whether the faint bins' own scatter grows fast enough to cancel that.
"""
import os, sys

os.environ['B17_LIMIT_MAG'] = '13.0'
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
src = src.replace('matrix_bruns2017', 'matrix_bruns2017_like2024')
src = src.replace("'stage2_constant'", "'stage2'")
exec(compile(src.split("print()\nfor t in (")[0], 'b17_union_magscan', 'exec'))

import numpy as np

rng = np.random.default_rng(7)
print()
print('cell-1 union (like-2024 convention, the reduction of record), catalogue to G 13')
print('%6s %5s %9s %9s %10s %12s' % ('magcut', 'N', 'L base', 'L v-deg2', '+- stat', 'faint-bin rms'))
prev = None
for cut in (9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0):
    U, rx, ry, R = build_union(('EA', 'EB'), 2.0)
    keep = U.mag.values <= cut
    if keep.sum() < 12:
        print('%6.1f %5d  -- too few stars' % (cut, keep.sum()))
        continue
    Us, rxs, rys, Rs = U[keep], rx[keep], ry[keep], R[keep]
    n = len(Us)
    Lb = fit_L(Us, rxs, rys, Rs)
    Lv = fit_L(Us, rxs, rys, Rs, nuis_deg=2)
    boots = []
    for _ in range(300):
        k = rng.integers(0, n, n)
        try:
            boots.append(fit_L(Us.iloc[k], rxs[k], rys[k], Rs[k], nuis_deg=2))
        except Exception:
            pass
    se = float(np.std(boots, ddof=1))
    # scatter of the stars this cut just admitted, about the fitted model
    A, labels = design(Us.px.values, Us.py.values, rxs, rys, Rs, 2)
    c, *_ = np.linalg.lstsq(A, np.concatenate([Us.dx.values, Us.dy.values]), rcond=None)
    resid = np.concatenate([Us.dx.values, Us.dy.values]) - A @ c
    resid = np.hypot(resid[:n], resid[n:])
    newly = (Us.mag.values > (prev if prev else 0)) & (Us.mag.values <= cut)
    fb = ('%12.3f' % np.sqrt(np.mean(resid[newly]**2))) if newly.sum() >= 2 else ('%12s' % '-')
    print('%6.1f %5d %+9.3f %+9.3f %10.3f %s   (+%d stars)'
          % (cut, n, Lb, Lv, se, fb, int(newly.sum())))
    prev = cut

print()
print('per-magnitude-bin residual scatter about the fitted model (all stars in, G <= 13):')
U, rx, ry, R = build_union(('EA', 'EB'), 2.0)
n = len(U)
A, labels = design(U.px.values, U.py.values, rx, ry, R, 2)
c, *_ = np.linalg.lstsq(A, np.concatenate([U.dx.values, U.dy.values]), rcond=None)
resid = np.concatenate([U.dx.values, U.dy.values]) - A @ c
resid = np.hypot(resid[:n], resid[n:])
for lo, hi in ((6, 9), (9, 10), (10, 11), (11, 12), (12, 13)):
    m = (U.mag.values > lo) & (U.mag.values <= hi)
    if m.sum():
        print('  G %2d-%2d: %2d stars, rms residual %.3f arcsec' % (lo, hi, m.sum(),
                                                                   np.sqrt(np.mean(resid[m]**2))))
