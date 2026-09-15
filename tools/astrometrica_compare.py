"""Can MEE reproduce Astrometrica's printed coefficients on the same image?

Douglas, 2026-09-15: "Let's look at the folder I:\\Don Bruns 2600MM tests. Are we able to
reproduce the astrometrica coefficients in the MEE output file?"

That folder holds the one thing needed to answer it: ONE stacked image reduced by both
programs. Astrometrica 4.13 read MEE's own `STACKED_FLOAT20240320154050.fit`, so both saw
identical pixels, and both printed a polynomial.

  Astrometrica  X, Y in RADIANS as a cubic in (x', y') = pixels from (3124, 2088)
  MEE           a correction in PIXELS as a quintic in (x, y)/w from the image centre

Astrometrica's origin is the image centre too -- 3124 x 2 = 6248, 2088 x 2 = 4176 -- and the
two tangent points agree to under an arcsecond, so nothing has to be guessed about where
either program pointed.

WHAT IS COMPARED, AND WHY IT IS NOT JUST THE NUMBERS.  The two write different quantities in
different bases, so a coefficient cannot be read against a coefficient directly. What CAN be
compared is the mapping each one defines from pixel to sky. The recorded recipe (ROADMAP,
"The reference-projection gauge") has two steps -- a basis rescale through the linear Jacobian,
then the gauge term -- and this does exactly that, in that order:

  1. evaluate both mappings on a grid;
  2. absorb the basis and pointing difference into a best-fit LINEAR map (this is the
     Jacobian step, and it is where `det J < 0` -- Astrometrica's image is mirrored relative
     to MEE's -- stops being a trap, because a fitted 2x2 simply carries the sign);
  3. what is left is the nonlinear disagreement, which is the gauge question.

Step 3 is run TWICE: once against MEE's native coefficients and once against the tangent-plane
export. If the export is doing its job the second residual is much the smaller, and the
remaining difference is the two programs' genuine disagreement about the optics.

Finally MEE's mapping is refitted in Astrometrica's own basis and the 20 coefficients are
printed side by side, which is the question as asked.

    .venv/Scripts/python.exe tools/astrometrica_compare.py
"""
import io
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mee2024 import distortion_polynomial as dp      # noqa: E402
from mee2024 import transforms                        # noqa: E402

SRC = r'I:\Don Bruns 2600MM tests'                     # READ ONLY
LOG = os.path.join(SRC, 'Walter HIP29696 MEE2024float.txt')
MEE = os.path.join(SRC, 'output', 'data20240320154050distortion_results.txt')
NX, NY = 6248, 4176                                    # origin 3124, 2088 is the centre
#: Astrometrica's cubic terms, in its own printed order
ATERMS = ['1', "x'", "y'", "x'^2", "x'*y'", "y'^2", "x'^3", "x'^2*y'", "x'*y'^2", "y'^3"]


def astrometrica_coeffs():
    """The X = ... and Y = ... polynomials, as two lists of ten floats.

    Each WRAPS over three lines in the log -- linear, quadratic, cubic -- so the block runs to
    the start of the next labelled section, not to the end of the line.
    """
    txt = io.open(LOG, encoding='utf-8', errors='replace').read()
    out = {}
    for axis, stop in (('X', r'\n\s*Y = '), ('Y', r'\n\s*Origin:')):
        m = re.search(r'\n\s*%s = (.*?)(?=%s)' % (axis, stop), txt, re.S)
        if not m:
            raise SystemExit('no %s = block in the Astrometrica log' % axis)
        out[axis] = [float(v) for v in
                     re.findall(r'[+-]?\d\.\d+E[+-]\d+', m.group(1))]
        if len(out[axis]) != 10:
            raise SystemExit('%s: expected 10 terms, read %d' % (axis, len(out[axis])))
    return out['X'], out['Y']


def astrometrica_basis(xp, yp):
    return np.column_stack([np.ones_like(xp), xp, yp, xp * xp, xp * yp, yp * yp,
                            xp ** 3, xp * xp * yp, xp * yp * yp, yp ** 3])


def mee_order(j):
    """The fitted order. This 2024-era results file predates the `distortion order` key, so it
    is read back off the coefficients themselves rather than assumed."""
    if j.get('distortion order'):
        return j['distortion order']
    n = len(j['distortion coeffs x'])
    for name, k in dp.mapping.items():
        if (k + 2) * (k + 1) // 2 == n:
            return name
    raise SystemExit('cannot infer the distortion order from %d coefficients' % n)


def mee_sky(j, col, row, tangent_plane):
    """Where MEE puts each pixel, as two angles in radians.

    tangent_plane=False returns MEE's OWN frame (declination, RA*cos(dec)); True converts to
    the tangent plane, which is what tangent_plane_coefficients does.
    """
    scale = np.radians(j['platescale (arcseconds/pixel)'] / 3600.0)
    order = mee_order(j)
    opts = {'distortionOrder': order, 'distortion_fixed_coefficients': 'None'}
    names = dp.get_coeff_names(opts)
    cx = [j['distortion coeffs x'].get(n, 0.0) for n in names]
    cy = [j['distortion coeffs y'].get(n, 0.0) for n in names]
    w = max(NY, NX) / 2
    basis = dp.get_basis(row, col, w, 1, opts)
    dcol = basis @ np.asarray(cx[1:])
    drow = basis @ np.asarray(cy[1:])
    icoords = np.column_stack([(row + drow), (col + dcol)]) * scale   # (dec, RA*cos dec)
    if not tangent_plane:
        return icoords[:, 1], icoords[:, 0]
    v = transforms.icoord_to_vector(icoords)
    return v[:, 1] / v[:, 0], v[:, 2] / v[:, 0]


def linear_residual(a1, a2, b1, b2):
    """Absorb pointing and basis into a best-fit linear map; return the residual in arcsec."""
    A = np.column_stack([np.ones_like(a1), a1, a2])
    r = []
    for target in (b1, b2):
        c, *_ = np.linalg.lstsq(A, target, rcond=None)
        r.append(target - A @ c)
    return np.degrees(np.hypot(r[0], r[1])) * 3600.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mee', default=MEE,
                    help='a MEE distortion_results.txt for the same image (default: the 2024 '
                         'one stored in the test folder)')
    a = ap.parse_args()
    ax, ay = astrometrica_coeffs()
    j = __import__('json').load(io.open(a.mee, encoding='utf-8'))
    print('ONE image, two programs')
    print('   Astrometrica 4.13  cubic,  914 stars, dRA = dDe = 0.04 ", 1.43 "/px')
    print('   MEE                %-8s %d stars, rms %.4f ", %.7f "/px'
          % (mee_order(j), j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)']))
    print()

    n = 60
    col = np.linspace(-NX / 2, NX / 2, n)
    row = np.linspace(-NY / 2, NY / 2, n)
    C, R = np.meshgrid(col, row)
    C, R = C.ravel(), R.ravel()
    B = astrometrica_basis(C, R)                       # Astrometrica's own x', y'
    AX, AY = B @ np.asarray(ax), B @ np.asarray(ay)

    print('AFTER the linear (Jacobian) step, what is left between the two programs:')
    for label, tp in (('MEE native gauge      ', False), ('MEE tangent-plane export', True)):
        m1, m2 = mee_sky(j, C, R, tangent_plane=tp)
        res = linear_residual(m1, m2, AX, AY)
        print('   %s  rms %.4f "   max %.4f "' % (label, np.sqrt(np.mean(res ** 2)), res.max()))
    print()

    # MEE's mapping expressed in Astrometrica's basis, which is the question as asked
    m1, m2 = mee_sky(j, C, R, tangent_plane=True)
    A = np.column_stack([np.ones_like(m1), m1, m2])
    lin = [np.linalg.lstsq(A, t, rcond=None)[0] for t in (AX, AY)]
    fit_x, fit_y = [np.linalg.lstsq(B, A @ c, rcond=None)[0] for c in lin]

    # A ratio on a near-zero coefficient says nothing, so the disagreement is also reported as
    # what it DOES to a star: the difference per order, evaluated over the field.
    order_of = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    print('where the remaining disagreement lives, as star displacement over the field:')
    for k, label in ((0, 'constant'), (1, 'linear'), (2, 'quadratic'), (3, 'cubic')):
        sel = [i for i, o in enumerate(order_of) if o == k]
        dx = B[:, sel] @ (np.asarray(ax)[sel] - fit_x[sel])
        dy = B[:, sel] @ (np.asarray(ay)[sel] - fit_y[sel])
        d = np.degrees(np.hypot(dx, dy)) * 3600.0
        print('   %-10s rms %.4f "   max %.4f "' % (label, np.sqrt(np.mean(d ** 2)), d.max()))
    # MEE is quintic here and Astrometrica is cubic, so part of the residual is content
    # Astrometrica structurally cannot carry. How much?
    proj = [np.linalg.lstsq(B, t, rcond=None)[0] for t in (A @ lin[0], A @ lin[1])]
    lost = np.degrees(np.hypot(A @ lin[0] - B @ proj[0], A @ lin[1] - B @ proj[1])) * 3600.0
    print("   %-10s rms %.4f \"   max %.4f \"   (MEE's quartic+quintic, which a cubic cannot hold)"
          % ('>cubic', np.sqrt(np.mean(lost ** 2)), lost.max()))
    print()

    print("MEE's mapping refitted in Astrometrica's basis, against what Astrometrica printed:")
    print('   %-10s %18s %18s %10s' % ('term', 'Astrometrica', 'MEE', 'ratio'))
    for axis, a_c, m_c in (('X', ax, fit_x), ('Y', ay, fit_y)):
        for t, a_v, m_v in zip(ATERMS, a_c, m_c):
            rat = (m_v / a_v) if a_v else float('nan')
            print('   %s %-8s %18.9E %18.9E %10.3f' % (axis, t, a_v, m_v, rat))
        print()


if __name__ == '__main__':
    main()
