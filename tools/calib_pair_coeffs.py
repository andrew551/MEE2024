"""A calibration bracket's two fields, coefficient by coefficient.

Douglas, 2026-09-14: "look at the L and R fields from Bruns 2017. Compare the two fields and
tell me whether the quadratic components are more stable than the linear components. Only look
at these two fields, not any others. Just look at the final computed coefficients for the L and
R fields." -- then: "Now do the same comparison for the Mexico 2024 Station 2 bracket".

So: no night fields, no sample statistics -- the two stage-2 fits a bracket averages, their
stored coefficients, and the difference between them.  One tool for both cells rather than a
copy per cell, because four private copies of one calculation is how this project lost a week
in September (CLAUDE.md, on record_charts).

READING THE UNITS.  `distortion_polynomial.get_basis` divides every monomial by
w = max(img_shape)/2 and the coefficients are a displacement in PIXELS, so a coefficient is
exactly the pixels that term moves a star at the long-axis edge of the sensor.  Multiplying by
the plate scale gives arcseconds, which is the only form in which a linear and a quadratic
coefficient can be compared at all: they carry different powers of position, so their raw
numbers are not commensurable and their displacements are.

WHAT THE LINEAR COEFFICIENTS ALREADY EXCLUDE.  After each fit `_get_corrected_q` folds four
linear degrees of freedom back into the plate solution -- the two translations into an RA/Dec
shift, the isotropic scale into the plate scale, a roll into the roll.  What is left stored is
trace-free with one cross-term zero, i.e. pure shear.  So this comparison is not diluted by a
scale or rotation difference between the two fields: those are already gone.

EACH PAIR IS RUN TWICE, through two independent reductions of the SAME two fields, so that an
answer which is a property of one reduction cannot be mistaken for a property of the fields.

    .venv/Scripts/python.exe tools/calib_pair_coeffs.py [--pair bruns|s2]
"""
import argparse
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

ORDERS = {'linear': ['x', 'y'], 'quadratic': ['x^2', 'x * y', 'y^2']}

PAIRS = {
    'bruns': dict(
        label='Bruns 2017 L / R8 -- the bracket his method averages',
        tree=r'F:\MEE_output\bruns2017\matrix_bruns2017_like2024',
        alt=r'F:\MEE_output\bruns2017\bruns2017_lr',
        fields=('L', 'R8'), shape=(2472, 3296),
        note='both fields quadratic-free with the 15-night cubic frozen'),
    's2': dict(
        label='Mexico 2024 Station 2 left / right bracket',
        tree=r'F:\MEE_output\mexico2024\station2\bracket_quadfree',
        alt=r'F:\MEE_output\mexico2024\station2\bracket_freecubic',
        fields=('left', 'right'), shape=(3520, 4656),
        note='quadratic-free, the convention Bruns and Leon used (s2_bracket_convention.py);'
             ' the check tree leaves the cubic free too'),
}


def load(tree, field):
    p = glob.glob(os.path.join(tree, field, '**', 'distortion_results.txt'), recursive=True)[0]
    j = json.load(io.open(p, encoding='utf-8'))
    # the zip sits a few levels above the results file and the trees disagree on how many --
    # walk up rather than count separators
    stars, up = None, os.path.dirname(p)
    for _ in range(4):
        z = glob.glob(os.path.join(up, 'distortion_data*.zip'))
        if z:
            zf = zipfile.ZipFile(z[0])
            m = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
            if m:
                t = pd.read_csv(zf.open(m[0]))
                t.columns = [c.strip() for c in t.columns]
                stars = t[['px', 'py']].values.astype(float)
            break
        up = os.path.dirname(up)
    return dict(name=field, cx=j['distortion coeffs x'], cy=j['distortion coeffs y'],
                ps=j['platescale (arcseconds/pixel)'], n=j['#stars used'],
                rms=j['final rms error (arcseconds)'], free=j.get('fixed distortion order'),
                alt=j.get('observation alt (degrees)'), t=j.get('observation_time (UTC)'),
                stars=stars)


def monomial(name, u, v):
    return {'x': u, 'y': v, 'x^2': u * u, 'x * y': u * v, 'y^2': v * v}[name]


def grid(shape, n=41):
    ny, nx = shape
    w = max(shape) / 2.0
    return np.meshgrid(np.linspace(-nx / 2, nx / 2, n) / w,
                       np.linspace(-ny / 2, ny / 2, n) / w)


def order_field(f, order, u, v):
    dx = sum(f['cx'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    dy = sum(f['cy'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    return np.stack([dx, dy]) * f['ps']


def noise(f, order, u, v, shape):
    """1-sigma on this order's displacement field for THIS fit, from sigma^2 (X'X)^-1."""
    if f['stars'] is None:
        return np.nan
    ny, nx = shape
    w = max(shape) / 2.0
    su, sv = (f['stars'][:, 0] - nx / 2) / w, (f['stars'][:, 1] - ny / 2) / w
    names = ['1'] + ORDERS['linear'] + ORDERS['quadratic']
    X = np.column_stack([np.ones_like(su)] + [monomial(m, su, sv) for m in names[1:]])
    cov = np.linalg.inv(X.T @ X) * (f['rms'] / f['ps']) ** 2
    idx = [names.index(m) for m in ORDERS[order]]
    M = np.stack([monomial(m, u, v).ravel() for m in ORDERS[order]])
    var = np.einsum('ip,ij,jp->p', M, cov[np.ix_(idx, idx)], M)
    return float(np.sqrt(2.0 * var.mean()) * f['ps'])


def compare(A, B, shape, show_table):
    u, v = grid(shape)
    if show_table:
        print('   FINAL COEFFICIENTS, as stored (pixels of displacement at the long-axis edge)')
        print('   %-8s %12s %12s %12s   %12s %12s %12s'
              % ('', A['name'] + ' x', B['name'] + ' x', 'diff x',
                 A['name'] + ' y', B['name'] + ' y', 'diff y'))
        for order in ('linear', 'quadratic'):
            for m in ORDERS[order]:
                ax, bx = A['cx'].get(m, 0.0), B['cx'].get(m, 0.0)
                ay, by = A['cy'].get(m, 0.0), B['cy'].get(m, 0.0)
                print('   %-8s %+12.6f %+12.6f %+12.6f   %+12.6f %+12.6f %+12.6f'
                      % (m, ax, bx, bx - ax, ay, by, by - ay))
            print()
    out = {}
    for order in ('linear', 'quadratic'):
        fa, fb = order_field(A, order, u, v), order_field(B, order, u, v)
        mag = [float(np.sqrt(np.mean(np.sum(f ** 2, axis=0)))) for f in (fa, fb)]
        d = float(np.sqrt(np.mean(np.sum((fa - fb) ** 2, axis=0))))
        nz = float(np.sqrt(noise(A, order, u, v, shape) ** 2 + noise(B, order, u, v, shape) ** 2))
        out[order] = (mag[0], mag[1], d, nz)
    return out


def run(key):
    P = PAIRS[key]
    shape = P['shape']
    A, B = (load(P['tree'], f) for f in P['fields'])
    print('=' * 79)
    print(P['label'])
    print('   %s' % P['note'])
    for f in (A, B):
        print('   %-6s %4d stars, rms %.3f ", ps %.7f "/px, alt %s, %s UTC, free to %s'
              % (f['name'], f['n'], f['rms'], f['ps'],
                 '%.2f deg' % f['alt'] if f['alt'] is not None else '?', f['t'], f['free']))
    print()
    res = compare(A, B, shape, show_table=True)
    print('   WHAT THAT DIFFERENCE DOES TO A STAR, in arcsec at the sensor edge')
    print('   %-11s %9s %9s %10s %9s   %s'
          % ('order', A['name'], B['name'], 'difference', 'noise', 'sigma'))
    for order in ('linear', 'quadratic'):
        a, b, d, nz = res[order]
        print('   %-11s %9.4f %9.4f %10.4f %9.4f   %5.1f' % (order, a, b, d, nz, d / nz))

    A2, B2 = (load(P['alt'], f) for f in P['fields'])
    r2 = compare(A2, B2, shape, show_table=False)
    print()
    print('   CHECK -- the same two fields, independently reduced (%s)'
          % os.path.basename(P['alt']))
    print('   %-6s %4d stars rms %.3f ", %-6s %4d stars rms %.3f ", free to %s'
          % (A2['name'], A2['n'], A2['rms'], B2['name'], B2['n'], B2['rms'], A2['free']))
    for order in ('linear', 'quadratic'):
        _, _, d, nz = r2[order]
        print('   %-11s difference %.4f "  +- %.4f  (%.1f sigma)' % (order, d, nz, d / nz))

    dl, dq = res['linear'][2], res['quadratic'][2]
    print()
    print('   ANSWER: the %s components differ less, %.4f " against %.4f " -- a factor of %.1f.'
          % (('linear', dl, dq, dq / dl) if dl < dq else ('quadratic', dq, dl, dl / dq)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', choices=sorted(PAIRS), default=None)
    a = ap.parse_args()
    for key in ([a.pair] if a.pair else ['bruns', 's2']):
        run(key)
        print()


if __name__ == '__main__':
    main()
