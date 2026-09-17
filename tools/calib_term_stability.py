"""How stable are a calibration field's LINEAR terms against its QUADRATIC terms?

Douglas, 2026-09-14: "The Bruns 2017 method is to use both the linear and quadratic terms
from the two external L/R fields. We have recently started deploying different versions of
Method 2: one which uses the night-time zenith values (this I think can never be correct --
there is too much thermal change in the telescope between night and day); using the
coefficients derived from the eclipse field; using the quadratic terms from the external
field and the linear term from the eclipse field itself. The last method I would like to
start calling Method 3 from now on.  As a first step, we verify from the Bruns 2017 data and
the Mexico 2024 Station 2 data the relative stability of the linear and quadratic terms of
the L/R fields."

METHOD 3, defined.  Stage 2 fits the eclipse field with `distortion_fixed_coefficients=linear`
-- constant and linear free on the eclipse field itself, quadratic and above frozen from an
external calibration field -- and stage 3 fits the plate scale alongside L, as Method 2 does.
Note that "Method 1" and "Method 2" in this pipeline name only how stage 3 treats the PLATE
SCALE (pinned vs fitted, docs/ARCHITECTURE.md); the rung is a separate axis.  Method 3 is
therefore Method 2's stage 3 on a `linear` rung, and it is the pathway cell 4 already ran as
the `linear` row of HUSILLOS2026_ECLIPSE.md section 3t.

Method 3 is worth having only if the quadratic terms transfer from another field better than
the linear terms do.  So: take every repeated calibration-field fit in the matrix, ask what
each ORDER does to a star's position, and measure how much that varies field to field.

THE GAUGE.  `distortion_polynomial.get_basis` builds monomials in (x, y) divided by
w = max(img_shape)/2, and the coefficients are a displacement in PIXELS.  So a coefficient IS
the pixels that term moves a star at the long-axis edge, and multiplying by the fit's own
plate scale puts both instruments in arcseconds with no gauge conversion -- the trap CLAUDE.md
warns about does not arise, because nothing is compared in rad/px^n.

WHAT THE STORED LINEAR TERMS ALREADY ARE.  `_get_corrected_q` folds four linear degrees of
freedom back into the plate solution after each fit: the two translations become a shift in
RA/Dec, the isotropic scale becomes a new plate scale, and a roll shift zeroes the y-coefficient
of the x-correction.  Iterated to convergence that leaves the stored linear map as
[[a, 0], [b, -a]] -- trace zero, one axis cross-term zero -- whose symmetric part is the shear
(a along the axes, b/2 at 45 degrees) and whose antisymmetric part is b/2 of rotation.  Since
a rotation has already been absorbed at every iteration, the surviving b is 2x the 45-degree
shear.  So the stored linear terms are PURE SHEAR: none of what is left gets absorbed
downstream, and all of it contaminates a deflection.  The tool asserts the two structural
zeros rather than assuming them, so a change in the parametrisation is caught instead of
silently misread.

THE NOISE FLOOR, without which none of this can be read.  The night fields carry 700-1300
stars at 0.05 " rms; the eclipse-day brackets carry ~120 at 0.21 ".  Scatter between two noisy
fits is not instability.  Each fit's own matched-star table gives its design matrix, so the
coefficient covariance is sigma^2 (X'X)^-1 exactly, propagated to a displacement field per
order and compared against the observed scatter.  What is quoted as real is the deconvolved
part, sqrt(observed^2 - noise^2).

    .venv/Scripts/python.exe tools/calib_term_stability.py
"""
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

#: monomials of each order, spelled as distortion_polynomial.get_coeff_names spells them
ORDERS = {
    'constant': ['1'],
    'linear': ['x', 'y'],
    'quadratic': ['x^2', 'x * y', 'y^2'],
    'cubic': ['x^3', 'x^2 * y', 'x * y^2', 'y^3'],
}
SEQUENCE = ['constant', 'linear', 'quadratic', 'cubic']
DEPTH = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5}

#: (label, root holding one folder per field, glob under it, sensor (ny, nx), note).  The trees
#: do not agree on depth -- the Bruns ones interpose a `stage2\` level, Station 2's does not --
#: so a field's name is the first path component under its root, never a count of separators.
DATASETS = [
    ('bruns_nights', r'F:\MEE_output\bruns2017\bruns2017_nights', r'*\**\distortion_results.txt',
     (2472, 3296), 'Bruns 2017 night calibrations, three pointings x 10, 19-20 Aug'),
    ('bruns_bracket', r'F:\MEE_output\bruns2017\matrix_bruns2017_like2024', r'{L,R8}\**\distortion_results.txt',
     (2472, 3296), "Bruns 2017 eclipse-day L/R bracket -- the pair his method averages"),
    ('s2_bracket', r'F:\MEE_output\mexico2024\station2_transfer\bracket_quadfree', r'*\**\distortion_results.txt',
     (3520, 4656), 'Mexico 2024 Station 2 eclipse-day bracket'),
]


def load(root, pattern):
    """Every stage-2 fit under the root, newest per field, with its own stars and scale."""
    if '{' in pattern:                       # small brace expansion, so a pair can be listed
        head, rest = pattern.split('{', 1)
        names, tail = rest.split('}', 1)
        pats = [head + n + tail for n in names.split(',')]
    else:
        pats = [pattern]
    seen = {}
    for pat in pats:
        for p in glob.glob(os.path.join(root, pat), recursive=True):
            key = os.path.relpath(p, root).split(os.sep)[0]
            if key not in seen or os.path.getmtime(p) > os.path.getmtime(seen[key]):
                seen[key] = p
    out = []
    for key in sorted(seen):
        p = seen[key]
        j = json.load(io.open(p, encoding='utf-8'))
        free = j.get('fixed distortion order')
        free = j['distortion order'] if free in (None, 'None') else free
        out.append(dict(name=key, cx=j['distortion coeffs x'], cy=j['distortion coeffs y'],
                        ps=j['platescale (arcseconds/pixel)'], n=j['#stars used'],
                        rms=j['final rms error (arcseconds)'], free=free,
                        stars=_stars(os.path.dirname(p))))
    return out


def _stars(d):
    """The fit's own matched stars (px, py), for the design matrix. None if the zip is gone."""
    for up in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        z = glob.glob(os.path.join(up, 'distortion_data*.zip'))
        if z:
            zf = zipfile.ZipFile(z[0])
            m = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
            if m:
                t = pd.read_csv(zf.open(m[0]))
                t.columns = [c.strip() for c in t.columns]
                return t[['px', 'py']].values.astype(float)
    return None


def norm(px, py, shape):
    ny, nx = shape
    w = max(shape) / 2.0
    return (px - nx / 2.0) / w, (py - ny / 2.0) / w


def monomial(name, u, v):
    return {'1': np.ones_like(u), 'x': u, 'y': v, 'x^2': u * u, 'x * y': u * v, 'y^2': v * v,
            'x^3': u ** 3, 'x^2 * y': u * u * v, 'x * y^2': u * v * v, 'y^3': v ** 3}[name]


def free_monomials(free):
    out = []
    for o in SEQUENCE:
        if DEPTH[o] <= DEPTH[free]:
            out += ORDERS[o]
    return out


def grid(shape, n=41):
    """Normalised sensor coordinates: u over the long axis in [-1, 1], v the short axis."""
    ny, nx = shape
    w = max(shape) / 2.0
    return np.meshgrid(np.linspace(-nx / 2, nx / 2, n) / w,
                       np.linspace(-ny / 2, ny / 2, n) / w)


def field_of(fit, order, u, v):
    """Displacement in ARCSEC contributed by one order's terms, over the grid."""
    dx = sum(fit['cx'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    dy = sum(fit['cy'].get(m, 0.0) * monomial(m, u, v) for m in ORDERS[order])
    return np.stack([dx, dy]) * fit['ps']


def noise_of(fit, order, u, v, shape):
    """rms arcsec this order's displacement field is uncertain by, from sigma^2 (X'X)^-1.

    One axis at a time is what the pipeline fits, and x and y use the same design matrix and
    are independent, so the VECTOR magnitude has twice the scalar variance.
    """
    if fit['stars'] is None:
        return np.nan
    su, sv = norm(fit['stars'][:, 0], fit['stars'][:, 1], shape)
    names = free_monomials(fit['free'])
    X = np.column_stack([monomial(m, su, sv) for m in names])
    try:
        cov = np.linalg.inv(X.T @ X) * (fit['rms'] / fit['ps']) ** 2      # pixels^2
    except np.linalg.LinAlgError:
        return np.nan
    idx = [names.index(m) for m in ORDERS[order] if m in names]
    if not idx:
        return np.nan
    M = np.stack([monomial(m, u, v).ravel() for m in ORDERS[order] if m in names])   # k x npts
    var = np.einsum('ip,ij,jp->p', M, cov[np.ix_(idx, idx)], M)           # pixels^2, one axis
    return float(np.sqrt(2.0 * var.mean()) * fit['ps'])                   # both axes, arcsec


def rms_over(fields):
    return float(np.sqrt(np.mean(np.sum(np.asarray(fields) ** 2, axis=1))))


def shear(fit):
    """The stored linear map as shear, in arcsec at the edge, with its structure checked."""
    a, c = fit['cx'].get('x', 0.0), fit['cx'].get('y', 0.0)
    b, d = fit['cy'].get('x', 0.0), fit['cy'].get('y', 0.0)
    scale = 0.5 * (a + d)
    assert abs(scale) < 1e-4 * max(1e-9, abs(a)) + 1e-9, (fit['name'], 'trace not zero', a, d)
    assert abs(c) < 1e-4 * max(1e-9, abs(b)) + 1e-9, (fit['name'], 'x-of-y not zero', c)
    return np.array([a, 0.5 * b]) * fit['ps']          # (axis-aligned shear, 45-degree shear)


def report(label, note, fits, shape):
    u, v = grid(shape)
    print('=' * 79)
    print('%s -- %s' % (label, note))
    print('%d fits, sensor %dx%d, free to %s' % (len(fits), shape[1], shape[0], fits[0]['free']))
    for f in fits[:4]:
        print('   %-8s %4d stars, rms %.3f ", ps %.7f "/px' % (f['name'], f['n'], f['rms'], f['ps']))
    if len(fits) > 4:
        print('   ... and %d more, %d-%d stars, rms %.3f-%.3f "'
              % (len(fits) - 4, min(f['n'] for f in fits), max(f['n'] for f in fits),
                 min(f['rms'] for f in fits), max(f['rms'] for f in fits)))
    print()
    print('   %-10s %9s %9s %9s %9s   %s'
          % ('order', 'mean', 'scatter', 'noise', 'REAL', 'real/mean'))
    print('   %-10s %9s %9s %9s %9s' % ('', 'arcsec', 'arcsec', 'arcsec', 'arcsec'))
    res = {}
    n = len(fits)
    for order in ('linear', 'quadratic', 'cubic'):
        if DEPTH[order] > DEPTH[fits[0]['free']]:
            continue
        fs = [field_of(f, order, u, v) for f in fits]
        mean = np.mean(fs, axis=0)
        obs = rms_over([f - mean for f in fs])
        # each fit's own noise, and the shrinkage from measuring scatter about a sample mean
        noise = float(np.sqrt(np.mean([noise_of(f, order, u, v, shape) ** 2 for f in fits])
                              * (1.0 - 1.0 / n)))
        real = float(np.sqrt(max(0.0, obs ** 2 - noise ** 2)))
        mag = rms_over([mean])
        res[order] = dict(mean=mag, obs=obs, noise=noise, real=real)
        print('   %-10s %9.4f %9.4f %9.4f %9.4f   %8.2f'
              % (order, mag, obs, noise, real, real / mag if mag else np.nan))
    sh = np.array([shear(f) for f in fits])
    print()
    print('   the stored linear terms are pure shear (scale, rotation and the two shifts are')
    print('   already in the plate solution): axis %+.4f +- %.4f ", 45 deg %+.4f +- %.4f "'
          % (sh[:, 0].mean(), sh[:, 0].std(ddof=1), sh[:, 1].mean(), sh[:, 1].std(ddof=1)))
    return res


def night_vs_day(nights, bracket, shape):
    """How far the night mean sits from the eclipse-day mean, per order.

    This is the question behind "the night-time zenith values can never be correct -- there is
    too much thermal change in the telescope between night and day".  The stability figures
    above are repeatability WITHIN one regime; this is the step BETWEEN them, and it is the
    number that says whether a night calibration may be frozen into a daytime fit.

    One confound, stated rather than hidden: the bracket fits carry the night average as their
    frozen cubic, so any real day-night change in the cubic is pushed down into the bracket's
    linear and quadratic.  That inflates this comparison rather than deflating it, so a SMALL
    step here would be conclusive and a large one is an upper bound.
    """
    u, v = grid(shape)
    print('=' * 79)
    print('NIGHT vs ECLIPSE DAY -- the same instrument, %d night fields against %d day fields'
          % (len(nights), len(bracket)))
    print()
    print('   %-10s %9s %9s %9s   %s' % ('order', 'step', 'noise', 'step/noise', 'verdict'))
    print('   %-10s %9s %9s' % ('', 'arcsec', 'arcsec'))
    for order in ('linear', 'quadratic'):
        a = np.mean([field_of(f, order, u, v) for f in nights], axis=0)
        b = np.mean([field_of(f, order, u, v) for f in bracket], axis=0)
        step = rms_over([a - b])
        na = np.mean([noise_of(f, order, u, v, shape) ** 2 for f in nights]) / len(nights)
        nb = np.mean([noise_of(f, order, u, v, shape) ** 2 for f in bracket]) / len(bracket)
        noise = float(np.sqrt(na + nb))
        print('   %-10s %9.4f %9.4f %9.1f   %s'
              % (order, step, noise, step / noise if noise else np.nan,
                 'REAL, and large' if step > 3 * noise else
                 'real' if step > noise else 'not resolved'))
    print()


def main():
    print('Relative stability of the LINEAR and QUADRATIC terms of a calibration field')
    print('(all figures are arcseconds of star displacement at the sensor edge)')
    print()
    summary = {}
    loaded = {}
    for label, root, pattern, shape, note in DATASETS:
        fits = load(root, pattern)
        loaded[label] = (fits, shape)
        if len(fits) < 2:
            print('%s: %d fits, need 2 -- skipped\n' % (label, len(fits)))
            continue
        summary[label] = report(label, note, fits, shape)
        print()
    if 'bruns_nights' in summary and 'bruns_bracket' in summary:
        night_vs_day(loaded['bruns_nights'][0], loaded['bruns_bracket'][0],
                     loaded['bruns_nights'][1])
    print('=' * 79)
    print('DOES THE QUADRATIC TRANSFER BETTER THAN THE LINEAR?')
    print('%-15s %11s %11s %11s' % ('dataset', 'linear REAL', 'quad REAL', 'ratio lin/quad'))
    for label, res in summary.items():
        lin = res.get('linear', {}).get('real', np.nan)
        qd = res.get('quadratic', {}).get('real', np.nan)
        print('%-15s %11.4f %11.4f %11s'
              % (label, lin, qd, ('%.2f' % (lin / qd)) if qd else 'n/a'))


if __name__ == '__main__':
    main()
