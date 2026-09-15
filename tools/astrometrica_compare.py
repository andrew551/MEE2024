"""Does MEE reproduce Astrometrica's coefficients? Measured on fields both programs reduced.

Douglas, 2026-09-15: "Are we able to reproduce the astrometrica coefficients in the MEE output
file?" -- then "I:\\Don Bruns 2024 ... has both MEE and Astrometrica results. These were done
with cubic correction. Let's see how close the results are this time."

Two folders on `I:` (READ ONLY) hold fields reduced by both programs:

  Don Bruns 2600MM tests   one field, MEE quintic, and a stage-1 archive that can be re-run
  Don Bruns 2024           FOUR fields, MEE CUBIC throughout -- like for like with Astrometrica,
                           which only fits a cubic

WHAT IS COMPARED, AND WHY IT IS NOT COEFFICIENT AGAINST COEFFICIENT.  The two write different
quantities in different bases, so the numbers cannot be read off against each other. What can
be compared is the mapping each defines from pixel to sky. The recorded recipe (ROADMAP, "The
reference-projection gauge") has two steps, and this does them in order:

  1. evaluate both mappings on a grid;
  2. absorb pointing and basis into a best-fit LINEAR map -- the Jacobian step, and where
     `det J < 0` stops being a trap, because a fitted 2x2 simply carries the mirror;
  3. what is left is the nonlinear disagreement, which is the gauge question.

Step 3 runs twice, against MEE's native coefficients and against the tangent-plane conversion.
If the conversion is doing its job the second is much the smaller.

    .venv/Scripts/python.exe tools/astrometrica_compare.py                 # all four 2024 fields
    .venv/Scripts/python.exe tools/astrometrica_compare.py --set 2600mm    # the quintic field
    .venv/Scripts/python.exe tools/astrometrica_compare.py --log L --mee M # any pair
"""
import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mee2024 import distortion_polynomial as dp      # noqa: E402
from mee2024 import transforms                        # noqa: E402

SETS = {
    '2024': r'I:\Don Bruns 2024',
    '2600mm': r'I:\Don Bruns 2600MM tests',
}
ATERMS = ['1', "x'", "y'", "x'^2", "x'*y'", "y'^2", "x'^3", "x'^2*y'", "x'*y'^2", "y'^3"]


def astrometrica_coeffs(log):
    """The X and Y polynomials and the origin. Each polynomial WRAPS over three lines."""
    txt = io.open(log, encoding='utf-8', errors='replace').read()
    out = {}
    for axis, stop in (('X', r'\n\s*Y = '), ('Y', r'\n\s*Origin:')):
        m = re.search(r'\n\s*%s = (.*?)(?=%s)' % (axis, stop), txt, re.S)
        if not m:
            raise SystemExit('%s: no %s = block' % (log, axis))
        out[axis] = [float(v) for v in re.findall(r'[+-]?\d\.\d+E[+-]\d+', m.group(1))]
        if len(out[axis]) != 10:
            raise SystemExit('%s: %s has %d terms, expected 10' % (log, axis, len(out[axis])))
    o = re.search(r'Origin:\s*x0\s*=\s*([\d.]+),\s*y0\s*=\s*([\d.]+)', txt)
    if not o:
        raise SystemExit('%s: no Origin line' % log)
    x0, y0 = float(o.group(1)), float(o.group(2))
    n = re.search(r'(\d+) of (\d+) Reference Stars used: dRA = ([\d.]+)", dDe = ([\d.]+)"', txt)
    stars, dra = (int(n.group(1)), float(n.group(3))) if n else (0, float('nan'))
    return out['X'], out['Y'], (int(2 * y0), int(2 * x0)), stars, dra


def abasis(xp, yp):
    return np.column_stack([np.ones_like(xp), xp, yp, xp * xp, xp * yp, yp * yp,
                            xp ** 3, xp * xp * yp, xp * yp * yp, yp ** 3])


def mee_order(j):
    """Read back off the coefficients: the 2024-era files predate the `distortion order` key."""
    if j.get('distortion order'):
        return j['distortion order']
    n = len(j['distortion coeffs x'])
    for name, k in dp.mapping.items():
        if (k + 2) * (k + 1) // 2 == n:
            return name
    raise SystemExit('cannot infer the order from %d coefficients' % n)


def mee_sky(j, col, row, shape, tangent_plane):
    """Where MEE puts each pixel, as two angles in radians."""
    scale = np.radians(j['platescale (arcseconds/pixel)'] / 3600.0)
    opts = {'distortionOrder': mee_order(j), 'distortion_fixed_coefficients': 'None'}
    names = dp.get_coeff_names(opts)
    cx = [j['distortion coeffs x'].get(n, 0.0) for n in names]
    cy = [j['distortion coeffs y'].get(n, 0.0) for n in names]
    basis = dp.get_basis(row, col, max(shape) / 2, 1, opts)
    icoords = np.column_stack([row + basis @ np.asarray(cy[1:]),
                               col + basis @ np.asarray(cx[1:])]) * scale
    if not tangent_plane:
        return icoords[:, 1], icoords[:, 0]
    v = transforms.icoord_to_vector(icoords)
    return v[:, 1] / v[:, 0], v[:, 2] / v[:, 0]


def linear_residual(a1, a2, b1, b2):
    A = np.column_stack([np.ones_like(a1), a1, a2])
    r = [t - A @ np.linalg.lstsq(A, t, rcond=None)[0] for t in (b1, b2)]
    return np.degrees(np.hypot(r[0], r[1])) * 3600.0


def compare(log, mee, verbose=True):
    ax, ay, shape, astars, adra = astrometrica_coeffs(log)
    j = json.load(io.open(mee, encoding='utf-8'))
    ny, nx = shape
    n = 60
    C, R = np.meshgrid(np.linspace(-nx / 2, nx / 2, n), np.linspace(-ny / 2, ny / 2, n))
    C, R = C.ravel(), R.ravel()
    B = abasis(C, R)
    AX, AY = B @ np.asarray(ax), B @ np.asarray(ay)

    out = {'field': os.path.basename(log), 'order': mee_order(j),
           'mee_stars': j['#stars used'], 'mee_rms': j['final rms error (arcseconds)'],
           'ast_stars': astars, 'ast_rms': adra}
    for key, tp in (('native', False), ('tan', True)):
        m1, m2 = mee_sky(j, C, R, shape, tangent_plane=tp)
        res = linear_residual(m1, m2, AX, AY)
        out[key] = float(np.sqrt(np.mean(res ** 2)))
        out[key + '_max'] = float(res.max())

    m1, m2 = mee_sky(j, C, R, shape, tangent_plane=True)
    A = np.column_stack([np.ones_like(m1), m1, m2])
    lin = [np.linalg.lstsq(A, t, rcond=None)[0] for t in (AX, AY)]
    fit_x, fit_y = [np.linalg.lstsq(B, A @ c, rcond=None)[0] for c in lin]
    order_of = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    if verbose:
        print('   per order, as star displacement over the field:')
        for k, label in ((0, 'constant'), (1, 'linear'), (2, 'quadratic'), (3, 'cubic')):
            sel = [i for i, o in enumerate(order_of) if o == k]
            d = np.degrees(np.hypot(B[:, sel] @ (np.asarray(ax)[sel] - fit_x[sel]),
                                    B[:, sel] @ (np.asarray(ay)[sel] - fit_y[sel]))) * 3600.0
            print('      %-10s rms %.4f "' % (label, np.sqrt(np.mean(d ** 2))))
        # the mirror, stated rather than implied
        M_mee = np.array([[fit_x[1], fit_x[2]], [fit_y[1], fit_y[2]]])
        print('   axes: determinant of the MEE->Astrometrica map is %+.3f'
              % np.sign(np.linalg.det(np.array([[ax[1], ax[2]], [ay[1], ay[2]]])))
              if M_mee is not None else '')
    return out


def pairs_2024(folder):
    """Astrometrica log and MEE results paired by HIP number ('HIP 29696' vs 'HIP29696')."""
    def hip(p):
        m = re.search(r'HIP\s*(\d+)', os.path.basename(p))
        return m.group(1) if m else None
    logs = {hip(p): p for p in glob.glob(os.path.join(folder, 'Astrometrica*.log'))}
    mees = {hip(p): p for p in glob.glob(os.path.join(folder, '*distortion_results.txt'))}
    return [(k, logs[k], mees[k]) for k in sorted(set(logs) & set(mees))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='2024', choices=sorted(SETS))
    ap.add_argument('--log', default=None)
    ap.add_argument('--mee', default=None)
    a = ap.parse_args()

    if a.log and a.mee:
        rows = [('pair', a.log, a.mee)]
    elif a.set == '2024':
        rows = pairs_2024(SETS['2024'])
    else:
        rows = [('HIP29696', os.path.join(SETS['2600mm'], 'Walter HIP29696 MEE2024float.txt'),
                 os.path.join(SETS['2600mm'], 'output',
                              'data20240320154050distortion_results.txt'))]

    print('MEE against Astrometrica 4.13, on fields both programs reduced')
    print('%-10s %-8s %14s %14s %11s %11s %8s' %
          ('field', 'order', 'MEE stars/rms', 'Ast stars/rms', 'native "', 'TAN "', 'factor'))
    tot = []
    for name, log, mee in rows:
        r = compare(log, mee, verbose=False)
        tot.append(r)
        print('%-10s %-8s %7d %6.4f %7d %6.3f %11.4f %11.4f %8.1f'
              % (name, r['order'], r['mee_stars'], r['mee_rms'], r['ast_stars'], r['ast_rms'],
                 r['native'], r['tan'], r['native'] / r['tan']))
    if len(tot) > 1:
        print('%-10s %-8s %14s %14s %11.4f %11.4f %8.1f'
              % ('MEAN', '', '', '', np.mean([r['native'] for r in tot]),
                 np.mean([r['tan'] for r in tot]),
                 np.mean([r['native'] / r['tan'] for r in tot])))


if __name__ == '__main__':
    main()
