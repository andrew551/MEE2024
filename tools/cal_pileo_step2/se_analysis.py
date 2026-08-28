"""Rebuild the plate-scale standard error from a stage-2 residual file.

Reproduces distortion_polynomial.py's HC0 figure, then recomputes it with HC3 and
with inverse-variance weights, to separate "the estimator is optimistic" from
"the field is at its floor".
"""
import sys, json
import numpy as np, pandas as pd

W = 3124.0          # max(img_shape)/2 for the 6248 x 4176 ASI2600
CX, CY = 6248 / 2, 4176 / 2   # distortion_fitter.py:76 centres the plate before fitting

def design(px, py):
    """[1, x/w, y/w, x^2/w^2, xy/w^2, y^2/w^2] -- get_basis(y=py, x=px) at quadratic,
    with the constant prepended exactly as sm.add_constant does. Coordinates are
    image-centred: the CSV records raw pixels, the fit does not."""
    x, y = (px - CX) / W, (py - CY) / W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def sandwich(X, e2):
    """(X'X)^-1 X' diag(e2) X (X'X)^-1 -- the HC family, e2 chooses the member."""
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * e2[:, None]).T @ X
    return XtXi @ meat @ XtXi

def se_platescale(X, ex, ey, weights=None):
    """The two linear coefficients, combined as the code combines them, in ppm."""
    if weights is None:
        weights = np.ones(len(ex))
    sw = np.sqrt(weights)
    Xw = X * sw[:, None]
    h = np.einsum('ij,jk,ik->i', Xw, np.linalg.inv(Xw.T @ Xw), Xw)
    out = {}
    for name, e in (('HC0', 1.0), ('HC3', None)):
        f = np.ones(len(ex)) if name == 'HC0' else 1.0 / (1 - h)**2
        cx = sandwich(Xw, weights * ex**2 * f)
        cy = sandwich(Xw, weights * ey**2 * f)
        out[name] = np.hypot(cx[1, 1]**0.5, cy[2, 2]**0.5) / W * 1e6
    return out, h

def report(path, label):
    d = pd.read_csv(path)
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    print(f'\n=== {label}  N={n} ===')
    unw, h = se_platescale(X, ex, ey)
    print(f'  unweighted   HC0 = {unw["HC0"]:6.2f} ppm   HC3 = {unw["HC3"]:6.2f} ppm'
          f'   (HC3/HC0 = {unw["HC3"]/unw["HC0"]:.2f})')
    print(f'  leverage: max h = {h.max():.3f}, mean = {h.mean():.3f} (p/n = {X.shape[1]/n:.3f})')

    # sigma against magnitude, in the only currency the weights can use
    d = d.assign(e2=ex**2 + ey**2)
    bins = pd.cut(d.magV, [0, 8, 9, 10, 10.5, 11, 11.5, 12, 13])
    g = d.groupby(bins, observed=True).agg(n=('e2', 'size'), rms_px=('e2', lambda v: np.mean(v)**0.5))
    g['rms_as'] = g.rms_px * 2.2054
    print('  sigma vs magnitude:')
    for b, r in g.iterrows():
        print(f'    {str(b):14s} n={int(r.n):3d}  rms = {r.rms_as:6.3f}"')

    # inverse-variance weights from that binned model -- an UPPER bound on what
    # weighting can buy, because it credits every bit of the spread to noise
    lut = dict(zip(g.index, g.rms_px))
    sig = np.array([lut[b] for b in bins])
    wgt = 1.0 / sig**2
    wls, _ = se_platescale(X, ex, ey, weights=wgt)
    print(f'  inv-var WLS  HC0 = {wls["HC0"]:6.2f} ppm   HC3 = {wls["HC3"]:6.2f} ppm'
          f'   -> gain {unw["HC0"]/wls["HC0"]:.2f}x on HC0')
    return unw, wls

base = r"D:/MEE2024 output/MEE_output/step2_ladder"
for tag, run in [("tol 0.5", "tolsweep_0.5/DISTORTION_OUTPUT20260825143416__20260825033557"),
                 ("tol 1.0", "tolsweep_1.0/DISTORTION_OUTPUT20260825143422__20260825033557"),
                 ("tol 5.0", "tolsweep_5.0/DISTORTION_OUTPUT20260825143434__20260825033557")]:
    report(f"{base}/{run}/distortion/TWOD_RESIDUALS.csv", tag)
