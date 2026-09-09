"""Is the quintic sufficient?  The order is worth using only if its term beats its own wobble.

Douglas, 2026-09-09: *"septic will require a lot of stars to fit and likely will not be very
stable. Is quintic sufficient?"*  That is the right test and `hu_zenith_order.py` did not make it.
An in-sample rms always improves with more parameters, and even an out-of-sample transfer only
says whether a term helps somewhere -- neither says whether the term is REPRODUCIBLE.  This does.

The measurement is the one `tools/matrix_station1/s1_reference_tolerance.py` calls (b) -- the
difference between two independent fits' own distortion polynomials, in mas -- with one change
that turned out to matter.  s1's version samples a uniform grid over the frame, and **a uniform
grid runs into the extreme corners, where a rectangular sensor has no stars and a high-order
polynomial extrapolates freely.**  Measured on this data 2026-09-09: on the grid the septic's
worst point wobbles 881 mas and its apparent contribution reads 1273 mas, against 136 mas
measured where the stars actually are.  Both readings are inflated by the same empty region, so
the ratio survives -- but the absolute numbers are meaningless and would be quoted.  So this
tool evaluates the polynomials **at the matched star positions of the field**, through
`distortion_polynomial.get_basis`, never a second implementation of the basis.

**And one thing more, without which every number here is four to eight times too big.**  The
distortion polynomial keeps LINEAR terms, and those are degenerate with the plate solution's own
scale and roll: two fits of the same optic can split that degeneracy differently, and the
difference between their polynomials then contains a large linear part that the pointing and
plate scale silently absorb and no star ever feels.  Measured 2026-09-09: the raw polynomial
difference between halfA's quintic and septic reads 530 mas at the field edge, while the same
quantity taken from the two fits' own residuals -- which have each fit's linear solution already
in them -- reads 106 mas.  The correlation between the two routes is 0.30.  So the similarity
component (translation, rotation, uniform scale) is projected out of every difference before it
is measured, and `main` prints the residual-route cross-check beside the answer so that the two
never drift apart again.

Two independent pairs exist:

  * **halfA against halfB** -- the same zenith capture split into 24 and 25 frames, each stacked,
    solved and fitted on its own.  Same sky, same stars, half the photons: this isolates how much
    the model moves under pure statistical resampling.  It is the direct answer to "will it be
    stable".
  * **Leon Z1 against Leon Z4** -- two pointings 6.4 deg apart on the same night through the same
    optic.  A real field-to-field figure, on the smaller sensor.

Then the decision rule, which needs both numbers and is why this tool exists:

    a term is worth fitting when what it ADDS is larger than how much it WOBBLES.

"Adds" is the model difference between order N and order N-2 (`hu_zenith_order.py report`);
"wobbles" is the half-to-half scatter of the order-N model measured here.  A term that adds
0.14 " in the corners and wobbles 0.20 " there is not a measurement of the optics, it is a
measurement of this particular star field.

  .venv/Scripts/python.exe tools/husillos2026/hu_order_stability.py
"""
import contextlib
import glob
import io
import json
import os

import numpy as np
import pandas as pd

from mee2024 import distortion_polynomial as dp
from mee2024.config import get_default_options

OUT = r"D:/MEE2024 output/MEE_output/RECORD/husillos2026/zenith_order"
ORDERS = ('cubic', 'quintic', 'septic')

#: (label, sensor, plate scale, the two independent fits to compare)
PAIRS = [
    ('Husillos zenith, halfA vs halfB (same field, half the frames each)',
     (9576, 6388), 2.2064323, ('halfA', 'halfB')),
    ('Leon, Z1 vs Z4 (two pointings 6.4 deg apart)',
     (6248, 4176), 2.2073708, ('leonZ1', 'leonZ4')),
]


def results_file(stack, order):
    r = glob.glob(os.path.join(OUT, 's2_%s_%s' % (stack, order), '**',
                               'distortion_results.txt'), recursive=True)
    return r[0] if r else None


def stars(stack, order):
    """The matched star positions of a fit, as (px, py) in image pixels."""
    r = glob.glob(os.path.join(OUT, 's2_%s_%s' % (stack, order), '**', 'TWOD_RESIDUALS.csv'),
                  recursive=True)
    return pd.read_csv(r[0]) if r else None


def model_at(path, order, shape, px, py):
    """This fit's distortion displacement at the given image positions, in pixels.

    `distortion_field` would do this on a grid; the basis call underneath it is reused directly
    so that arbitrary positions can be asked for -- same code, same normalisation `w`, same
    exclusion of coeff[0] (the constant, which the pointing already absorbed).
    """
    opts = get_default_options()
    opts['distortionOrder'] = order
    opts['distortion_fixed_coefficients'] = 'None'
    opts['distortion_reference_files'] = path
    with contextlib.redirect_stdout(io.StringIO()):     # _open_distortion_files prints the dicts
        cx, cy, _, _ = dp._open_distortion_files(opts)
    y = np.asarray(py) - shape[1] / 2.0
    x = np.asarray(px) - shape[0] / 2.0
    w = max(shape) / 2.0
    basis = dp.get_basis(y, x, w, 1, opts)
    return (basis @ np.asarray(list(cx.values())[1:]),
            basis @ np.asarray(list(cy.values())[1:]))


def strip_similarity(px, py, dx, dy, shape):
    """Remove the translation + rotation + uniform scale a plate solution would absorb.

    What is left is the part of a displacement field that actually reaches the astrometry.  In
    complex form the similarity is d = c + z*p with p = x + iy, so it is one linear least squares
    over the two real parameters of c and the two of z.
    """
    p = (np.asarray(px) - shape[0] / 2.0) + 1j * (np.asarray(py) - shape[1] / 2.0)
    d = np.asarray(dx) + 1j * np.asarray(dy)
    A = np.column_stack([np.ones_like(p), p])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ coef
    return r.real, r.imag


#: radius bins as a fraction of the half-diagonal; the last is "where the stars run out"
BINS = ((0.0, 0.5), (0.5, 0.75), (0.75, 1.0))


def main():
    rows = []
    for label, shape, ps, (a, b) in PAIRS:
        half_diag = np.hypot(shape[0] / 2, shape[1] / 2)
        # one star list for the whole comparison, so every number below is measured at the same
        # points: the stars the quintic fit of the first field kept
        sd = stars(a, 'quintic')
        if sd is None:
            print('%s: no residual file' % label)
            continue
        px, py = sd['px'].to_numpy(), sd['py'].to_numpy()
        rg = np.hypot(px - shape[0] / 2, py - shape[1] / 2) / half_diag
        print('=' * 104)
        print(label)
        print('%d stars, reaching r/R = %.2f of the half-diagonal' % (len(px), rg.max()))
        print('=' * 104)
        hdr = ''.join('%14s' % ('r/R %.2f-%.2f' % bb) for bb in BINS)
        print('%-8s %11s %s %10s' % ('order', 'n stars', hdr, 'all stars'))
        for order in ORDERS:
            fa, fb = results_file(a, order), results_file(b, order)
            if not fa or not fb:
                print('%-8s   one of the two fits is missing' % order)
                continue
            dxa, dya = model_at(fa, order, shape, px, py)
            dxb, dyb = model_at(fb, order, shape, px, py)
            ddx, ddy = strip_similarity(px, py, dxa - dxb, dya - dyb, shape)
            d = np.hypot(ddx, ddy) * ps * 1000.0                    # mas
            na = json.load(open(fa, encoding='utf-8'))['#stars used']
            nb = json.load(open(fb, encoding='utf-8'))['#stars used']
            cells = [float(np.median(d[(rg >= lo) & (rg < hi)])) for lo, hi in BINS]
            print('%-8s %5d/%-5d %s %7.1f mas'
                  % (order, na, nb, ''.join('%10.1f mas' % c for c in cells),
                     np.sqrt((d ** 2).mean())))
            rec = dict(pair=label, order=order, n_a=na, n_b=nb,
                       all_mas=float(np.sqrt((d ** 2).mean())), worst_mas=float(d.max()))
            for (lo, hi), c in zip(BINS, cells):
                rec['wobble_%.2f_%.2f_mas' % (lo, hi)] = c
            rows.append(rec)
        print()

    t = pd.DataFrame(rows)
    if t.empty:
        return
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, 'order_stability.csv'), index=False)

    lo_hi, out_lo, out_hi = BINS[-1], BINS[-1][0], BINS[-1][1]
    print('=' * 104)
    print('THE DECISION: what the term adds against how much it wobbles, at r/R %.2f-%.2f'
          % (out_lo, out_hi))
    print('=' * 104)
    print('%-26s %-8s %13s %13s %8s   %s'
          % ('field', 'order', 'adds (mas)', 'wobble (mas)', 'ratio', 'verdict'))
    for label, shape, ps, (a, b) in PAIRS:
        half_diag = np.hypot(shape[0] / 2, shape[1] / 2)
        sd = stars(a, 'quintic')
        if sd is None:
            continue
        px, py = sd['px'].to_numpy(), sd['py'].to_numpy()
        rg = np.hypot(px - shape[0] / 2, py - shape[1] / 2) / half_diag
        outer = (rg >= out_lo) & (rg < out_hi)
        for low, high in (('cubic', 'quintic'), ('quintic', 'septic')):
            f_lo, f_hi = results_file(a, low), results_file(a, high)
            row = t[(t.pair == label) & (t.order == high)]
            if not f_lo or not f_hi or row.empty:
                continue
            dxl, dyl = model_at(f_lo, low, shape, px, py)
            dxh, dyh = model_at(f_hi, high, shape, px, py)
            adx, ady = strip_similarity(px, py, dxh - dxl, dyh - dyl, shape)
            add = np.hypot(adx, ady) * ps * 1000.0
            adds = float(np.median(add[outer]))
            wob = float(row['wobble_%.2f_%.2f_mas' % lo_hi].iloc[0])
            r = adds / wob if wob else np.inf
            print('%-26s %-8s %10.1f    %10.1f    %6.2f   %s'
                  % (label.split(',')[0], high, adds, wob, r,
                     'worth fitting' if r > 1 else 'fitting the star field, not the optics'))
    print()
    print('ratio > 1: the term is bigger than the scatter in the term -- worth fitting.')
    print('ratio < 1: the fit is chasing this star field rather than the optics.')
    print()
    term_reproduces()
    print()
    cross_check()
    print('->', os.path.join(OUT, 'order_stability.csv'))


def term_reproduces():
    """Does the added TERM come back the same when it is fitted twice?  The sharpest statistic.

    The table above compares whole models, which the low-order coefficients every order shares
    dominate.  This is narrower: take T = model(order N) - model(order N-2) on each of two
    independent fits, strip the similarity from each, and compare T_A against T_B.  A term that
    is a property of the optics comes back; a term that is fitting the star field does not.

    Read the two pairs differently, and the difference IS the result.  Husillos' halves share
    their stars -- same sky, same catalogue errors, same blends -- so ANY term that fits this
    field's quirks reproduces between them, and the pair cannot separate a real term from a
    fitted one.  Leon's Z1 and Z4 are different pointings with different stars, so only something
    belonging to the telescope reproduces there.
    """
    print('does the added term reproduce?  T = model(N) - model(N-2), r/R >= %.2f, mas'
          % BINS[-1][0])
    print('%-26s %-18s %10s %10s %11s %8s' % (
        'field', 'term', '|T_A|', '|T_B|', '|T_A-T_B|', 'ratio'))
    for label, shape, ps, (a, b) in PAIRS:
        half_diag = np.hypot(shape[0] / 2, shape[1] / 2)
        sd = stars(a, 'quintic')
        if sd is None:
            continue
        px, py = sd['px'].to_numpy(), sd['py'].to_numpy()
        rg = np.hypot(px - shape[0] / 2, py - shape[1] / 2) / half_diag
        outer = rg >= BINS[-1][0]

        def term(stack, low, high):
            fl, fh = results_file(stack, low), results_file(stack, high)
            if not fl or not fh:
                return None
            dxl, dyl = model_at(fl, low, shape, px, py)
            dxh, dyh = model_at(fh, high, shape, px, py)
            return strip_similarity(px, py, dxh - dxl, dyh - dyl, shape)

        for low, high in (('cubic', 'quintic'), ('quintic', 'septic')):
            ta, tb = term(a, low, high), term(b, low, high)
            if ta is None or tb is None:
                continue
            ma = np.hypot(*ta) * ps * 1000
            mb = np.hypot(*tb) * ps * 1000
            dd = np.hypot(ta[0] - tb[0], ta[1] - tb[1]) * ps * 1000
            r = np.median(ma[outer]) / np.median(dd[outer])
            print('%-26s %-18s %6.1f mas %6.1f mas %7.1f mas %7.2f'
                  % (label.split(',')[0], '%s - %s' % (high, low),
                     np.median(ma[outer]), np.median(mb[outer]), np.median(dd[outer]), r))
    print('ratio: how many times bigger the term is than the disagreement about it. Only the')
    print('Leon row is a real test -- Husillos halves share their stars (see the docstring).')


def cross_check():
    """The same 'adds' by a route that touches no polynomial code, so the two cannot drift apart.

    residual = catalogue - transform(measured), and each fit's transform carries its own
    pointing, roll, plate scale AND distortion.  So for a star both fits kept, the difference of
    their residuals is the whole mapping difference, with the linear solution already in it.  It
    should reproduce the similarity-stripped polynomial difference; if it stops doing so,
    something in `model_at` or `strip_similarity` has broken.
    """
    print('cross-check: the same quantity from the fits\' own residuals (no polynomial code)')
    print('%-26s %-8s %13s %13s' % ('field', 'order', 'polynomial', 'residuals'))
    for label, shape, ps, (a, b) in PAIRS:
        half_diag = np.hypot(shape[0] / 2, shape[1] / 2)
        for low, high in (('cubic', 'quintic'), ('quintic', 'septic')):
            sl, sh = stars(a, low), stars(a, high)
            if sl is None or sh is None:
                continue
            m = sl.merge(sh, on='ID', suffixes=('_l', '_h'))
            if m.empty:
                continue
            px, py = m['px_l'].to_numpy(), m['py_l'].to_numpy()
            rg = np.hypot(px - shape[0] / 2, py - shape[1] / 2) / half_diag
            outer = rg >= BINS[-1][0]
            res = np.hypot(m['dx_arcsec_l'] - m['dx_arcsec_h'],
                           m['dy_arcsec_l'] - m['dy_arcsec_h']).to_numpy() * 1000.0
            dxl, dyl = model_at(results_file(a, low), low, shape, px, py)
            dxh, dyh = model_at(results_file(a, high), high, shape, px, py)
            adx, ady = strip_similarity(px, py, dxh - dxl, dyh - dyl, shape)
            poly = np.hypot(adx, ady) * ps * 1000.0
            print('%-26s %-8s %10.1f mas %10.1f mas'
                  % (label.split(',')[0], high, np.median(poly[outer]), np.median(res[outer])))


if __name__ == '__main__':
    main()
