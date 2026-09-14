"""Is the moving structure Sun-centred, or a generic smooth field that only looks radial?

hu_radial_vs_time.py found the inner-minus-outer radial residual moving on a 10-30 s
timescale with +-0.3-0.5 px amplitude -- the atmosphere's signature.  That reading rests on
one projection, radial-from-the-Sun averaged over 11 inner stars, and 11 stars lopsided in
azimuth will make ANY smooth field look Sun-centred when so projected.  This is the
discriminator the radial average could not make.

For each 15-frame sub-stack the per-star 2-D residuals (after translation, rotation and
scale about the Sun are removed) are fitted two ways over the same stars:

    RADIAL     d = a * (R_sun / r) * r_hat                      1 parameter, Sun-centred
    QUADRATIC  d = c1 x + c2 y + c3 x^2 + c4 xy + c5 y^2 (each axis)  10 parameters, generic

and both are compared with what white noise alone would remove, and by held-out stars
(star-split cross-validation, the test that settled the zenith order and rejected the vertical
nuisance surface in section 3h).  A Sun-centred atmosphere is not a thing; a Sun-centred
pipeline effect is what section 3q ruled out.  If the radial term explains the field and the
quadratic does not, something about the Sun is still unexplained; if the quadratic explains it
and the radial term is what it looks like when projected, the atmosphere reading stands.

The per-bin quadratic coefficients are also printed as a time series: a real wavefield drifts
smoothly in them; a fixed gain artefact would step at frame 171 -> 2.

    .venv/Scripts/python.exe tools/husillos2026/hu_field_shape.py
"""
import os

import numpy as np
import pandas as pd

HUS = r'F:\MEE_output\husillos2026'
CSV = os.path.join(HUS, 'seeing', 'radial_vs_time_stars.csv')
SUNX, SUNY, PS, RS = 5043.0, 3386.0, 2.2028, 947.1
W = 4788.0            # NX / 2, field-normalised coordinates for the quadratic


def bases(px, py):
    rx, ry = px - SUNX, py - SUNY
    R = np.hypot(rx, ry)
    Rsun = R * PS / RS
    # radial: a displacement of a/Rsun arcsec along r_hat, in px
    rad = np.column_stack([rx / R / (Rsun * PS), ry / R / (Rsun * PS)])
    x, y = (px - SUNX) / W, (py - SUNY) / W
    Q = np.column_stack([x, y, x * x, x * y, y * y])
    return rad, Q


def fit_radial(d2, rad):
    b = np.concatenate([rad[:, 0], rad[:, 1]])
    a = float(b @ d2 / (b @ b))
    return a, d2 - a * b


def fit_quad(d2, Q):
    n = Q.shape[0]
    Z = np.zeros((n, 5))
    M = np.vstack([np.column_stack([Q, Z]), np.column_stack([Z, Q])])
    c, *_ = np.linalg.lstsq(M, d2, rcond=None)
    return c, d2 - M @ c, M


def crossval(d2, build, n, rng, k=200):
    """Held-out gain of a model: fit on half the stars, score the other half."""
    gains = []
    for _ in range(k):
        m = rng.permutation(n) < n // 2
        mm = np.concatenate([m, m])
        try:
            fit_fn = build(mm)
        except np.linalg.LinAlgError:
            continue
        held = d2[~mm]
        pred = fit_fn(~mm)
        gains.append(1 - np.std(held - pred) / np.std(held))
    return 100 * np.mean(gains) if gains else np.nan


def main():
    t = pd.read_csv(CSV)
    rng = np.random.default_rng(11)
    print('%-8s %-8s %4s %9s %9s %9s %9s %9s %9s'
          % ('block', 'frames', 'n', 'rms px', 'radial', 'quad', 'noise1', 'noise10', 'radial a'))
    print('%-8s %-8s %4s %9s %9s %9s %9s %9s %9s'
          % ('', '', '', '', 'held-out', 'held-out', 'expect', 'expect', '(arcsec)'))
    series = []
    for (block, f0), g in t.groupby(['block', 'f0'], sort=False):
        g = g.sort_values('star')
        n = len(g)
        px, py = g.px.values, g.py.values
        d2 = np.concatenate([g.res_x.values, g.res_y.values])
        rad, Q = bases(px, py)
        b = np.concatenate([rad[:, 0], rad[:, 1]])
        a, _ = fit_radial(d2, rad)
        c, _, M = fit_quad(d2, Q)
        rms = float(np.std(d2))

        def build_radial(mm):
            aa = float(b[mm] @ d2[mm] / (b[mm] @ b[mm]))
            return lambda held: aa * b[held]

        def build_quad(mm):
            cc, *_ = np.linalg.lstsq(M[mm], d2[mm], rcond=None)
            return lambda held: M[held] @ cc

        gr = crossval(d2, build_radial, n, rng)
        gq = crossval(d2, build_quad, n, rng)
        # what k parameters remove from white noise on 2n components
        e1 = 100 * (1 - np.sqrt(1 - 1 / (2 * n)))
        e10 = 100 * (1 - np.sqrt(max(1 - 10 / (2 * n), 0)))
        print('%-8s %3d-%-3d %4d %9.3f %+8.1f%% %+8.1f%% %8.1f%% %8.1f%% %+9.3f'
              % (block, f0, g.f0.iloc[0] + 14, n, rms, gr, gq, e1, e10, a))
        series.append(dict(block=block, f0=f0, t_s=g.t_s.iloc[0], a=a, **{
            'c%d' % i: c[i] for i in range(10)}))
    s = pd.DataFrame(series)
    print()
    print('held-out gain ABOVE the white-noise expectation is real structure; a model that')
    print('scores below it on held-out stars is fitting noise.')
    print()
    print('THE QUADRATIC COEFFICIENTS AS A TIME SERIES (x-axis terms, field units; a step at the')
    print('gain boundary would be an artefact, a smooth drift the atmosphere):')
    print('%-8s %-6s %8s %8s %8s %8s %8s   %8s' % ('block', 'f0', 'x', 'y', 'x2', 'xy', 'y2',
                                                   'radial a'))
    for _, q in s.iterrows():
        print('%-8s %-6d %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f   %+8.3f'
              % (q.block, q.f0, q.c0, q.c1, q.c2, q.c3, q.c4, q.a))


if __name__ == '__main__':
    main()
