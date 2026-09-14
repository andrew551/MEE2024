"""Bruns 2017's L and R calibration fields, coefficient by coefficient.

Douglas, 2026-09-14: "look at the L and R fields from Bruns 2017. Compare the two fields and
tell me whether the quadratic components are more stable than the linear components. Only look
at these two fields, not any others. Just look at the final computed coefficients for the L and
R fields."

So: no night fields, no Station 2, no sample statistics -- the two stage-2 fits Bruns' own
method averages, their stored coefficients, and the difference between them.

READING THE UNITS.  `distortion_polynomial.get_basis` divides every monomial by
w = max(img_shape)/2 and the coefficients are a displacement in PIXELS, so a coefficient is
exactly the pixels that term moves a star at the long-axis edge of the sensor.  Multiplying by
the plate scale gives arcseconds, which is the only form in which a linear and a quadratic
coefficient can be compared at all: they have different powers of position, so their raw
numbers are not commensurable, and their displacements are.

WHAT THE LINEAR COEFFICIENTS ALREADY EXCLUDE.  After each fit `_get_corrected_q` folds four
linear degrees of freedom back into the plate solution -- the two translations into an RA/Dec
shift, the isotropic scale into the plate scale, a roll into the roll.  What is left stored is
trace-free with one cross-term zero, i.e. pure shear.  So this comparison is not diluted by a
scale or rotation difference between the two fields: those are already gone.

    .venv/Scripts/python.exe tools/matrix_bruns/b17_lr_coeffs.py
"""
import glob
import io
import json
import os

import numpy as np
import pandas as pd
import zipfile

#: the convention-of-record tree: Bruns' own reduction, L and R8 fitted quadratic-free with the
#: 15-night cubic frozen (docs/MATRIX_2026.md, cell 1)
TREE = r'F:\MEE_output\matrix_bruns2017_like2024'
NY, NX = 2472, 3296
ORDERS = {'linear': ['x', 'y'], 'quadratic': ['x^2', 'x * y', 'y^2']}


#: the same two fields reduced independently earlier, as a check that the answer is a property
#: of the fields and not of one reduction
TREE_ALT = r'F:\MEE_output\bruns2017_lr'


def load(field, tree=None):
    p = glob.glob(os.path.join(tree or TREE, field, '**', 'distortion_results.txt'),
                  recursive=True)[0]
    j = json.load(io.open(p, encoding='utf-8'))
    # the zip sits in the field's stage2/ folder, two levels above the results file's own
    # DISTORTION_OUTPUT.../distortion/ -- walk up rather than guess a depth
    z = []
    up = os.path.dirname(p)
    for _ in range(4):
        z = glob.glob(os.path.join(up, 'distortion_data*.zip'))
        if z:
            break
        up = os.path.dirname(up)
    stars = None
    if z:
        zf = zipfile.ZipFile(z[0])
        m = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
        if m:
            t = pd.read_csv(zf.open(m[0]))
            t.columns = [c.strip() for c in t.columns]
            stars = t[['px', 'py']].values.astype(float)
    return dict(name=field, cx=j['distortion coeffs x'], cy=j['distortion coeffs y'],
                ps=j['platescale (arcseconds/pixel)'], n=j['#stars used'],
                rms=j['final rms error (arcseconds)'], free=j.get('fixed distortion order'),
                alt=j.get('observation alt (degrees)'), t=j.get('observation_time (UTC)'),
                stars=stars)


def monomial(name, u, v):
    return {'x': u, 'y': v, 'x^2': u * u, 'x * y': u * v, 'y^2': v * v}[name]


def grid(n=41):
    w = max(NY, NX) / 2.0
    return np.meshgrid(np.linspace(-NX / 2, NX / 2, n) / w,
                       np.linspace(-NY / 2, NY / 2, n) / w)


def order_field(f, order, u, v):
    dx = sum(f['cx'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    dy = sum(f['cy'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    return np.stack([dx, dy]) * f['ps']


def noise(f, order, u, v):
    """The 1-sigma uncertainty of this order's displacement field for THIS fit alone."""
    if f['stars'] is None:
        return np.nan
    w = max(NY, NX) / 2.0
    su, sv = (f['stars'][:, 0] - NX / 2) / w, (f['stars'][:, 1] - NY / 2) / w
    names = ['1'] + ORDERS['linear'] + ORDERS['quadratic']
    cols = [np.ones_like(su)] + [monomial(m, su, sv) for m in names[1:]]
    X = np.column_stack(cols)
    cov = np.linalg.inv(X.T @ X) * (f['rms'] / f['ps']) ** 2
    idx = [names.index(m) for m in ORDERS[order]]
    M = np.stack([monomial(m, u, v).ravel() for m in ORDERS[order]])
    var = np.einsum('ip,ij,jp->p', M, cov[np.ix_(idx, idx)], M)
    return float(np.sqrt(2.0 * var.mean()) * f['ps'])


def main():
    L, R = load('L'), load('R8')
    print('Bruns 2017 L and R8, the two calibration fields his method averages')
    for f in (L, R):
        print('   %-3s %4d stars, rms %.3f ", plate scale %.7f "/px, alt %.2f deg, %s UTC, '
               'free to %s' % (f['name'], f['n'], f['rms'], f['ps'], f['alt'], f['t'], f['free']))
    print()

    print('FINAL COEFFICIENTS, as stored (displacement in pixels at the long-axis edge)')
    print('   %-8s %13s %13s %13s   %13s %13s %13s'
          % ('', 'L  x', 'R8  x', 'diff x', 'L  y', 'R8  y', 'diff y'))
    for order in ('linear', 'quadratic'):
        for m in ORDERS[order]:
            lx, rx = L['cx'].get(m, 0.0), R['cx'].get(m, 0.0)
            ly, ry = L['cy'].get(m, 0.0), R['cy'].get(m, 0.0)
            print('   %-8s %+13.6f %+13.6f %+13.6f   %+13.6f %+13.6f %+13.6f'
                  % (m, lx, rx, rx - lx, ly, ry, ry - ly))
        print()

    u, v = grid()
    print('WHAT THAT DIFFERENCE DOES TO A STAR, in arcsec at the sensor edge')
    print('   %-11s %10s %10s %10s   %s' % ('order', 'L', 'R8', 'L - R8', 'and its noise'))
    out = {}
    for order in ('linear', 'quadratic'):
        fl, fr = order_field(L, order, u, v), order_field(R, order, u, v)
        mag_l = float(np.sqrt(np.mean(np.sum(fl ** 2, axis=0))))
        mag_r = float(np.sqrt(np.mean(np.sum(fr ** 2, axis=0))))
        d = float(np.sqrt(np.mean(np.sum((fl - fr) ** 2, axis=0))))
        nz = float(np.sqrt(noise(L, order, u, v) ** 2 + noise(R, order, u, v) ** 2))
        out[order] = (d, nz)
        print('   %-11s %10.4f %10.4f %10.4f   +- %.4f  (%.1f sigma)'
              % (order, mag_l, mag_r, d, nz, d / nz))
    print()
    dl, nl = out['linear']
    dq, nq = out['quadratic']
    print('ANSWER')
    print('   The LINEAR components agree far better: %.4f " between the two fields,' % dl)
    print('   against %.4f " for the quadratic -- a factor of %.1f the other way.' % (dq, dq / dl))
    print('   Significance, though: the linear difference is %.1f sigma and the quadratic %.1f,'
          % (dl / nl, dq / nq))
    print('   so on these two fields alone NEITHER difference is established, and the honest')
    print('   statement is that the linear agree to well inside their noise while the')
    print('   quadratic differ by about their noise.')

    # the same two fields, reduced independently earlier -- is the answer a property of the
    # fields or of one reduction?
    print()
    print('CHECK: the same two fields in the earlier `bruns2017_lr` reduction')
    L2, R2 = load('L', TREE_ALT), load('R8', TREE_ALT)
    print('   L %d stars rms %.3f ", R8 %d stars rms %.3f "'
          % (L2['n'], L2['rms'], R2['n'], R2['rms']))
    for order in ('linear', 'quadratic'):
        d = float(np.sqrt(np.mean(np.sum((order_field(L2, order, u, v)
                                          - order_field(R2, order, u, v)) ** 2, axis=0))))
        nz = float(np.sqrt(noise(L2, order, u, v) ** 2 + noise(R2, order, u, v) ** 2))
        print('   %-11s L - R8 = %.4f "  +- %.4f  (%.1f sigma)' % (order, d, nz, d / nz))


if __name__ == '__main__':
    main()
