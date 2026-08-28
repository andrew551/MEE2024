"""Bootstrap the plate-scale standard error, to adjudicate HC0 against HC3.

distortion_polynomial.py reports HC0, which is known to run low when a few stars carry
high leverage -- and at tol 0.5 one star carries h = 0.46 out of 31. A pairs bootstrap
resamples (design row, residual) together and refits, so it makes no small-sample
assumption at all and says which of the two sandwich estimators to believe.
"""
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0

def design(px, py):
    x, y = (px - CX) / W, (py - CY) / W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def hc(X, e, j, kind):
    XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    f = 1.0 if kind == 'HC0' else 1.0/(1-h)**2
    return (XtXi @ ((X * (e**2*f)[:, None]).T @ X) @ XtXi)[j, j]**0.5

def run(path, label, nboot=20000):
    d = pd.read_csv(path)
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    rep = {k: np.hypot(hc(X, ex, 1, k), hc(X, ey, 2, k))/W*1e6 for k in ('HC0', 'HC3')}

    rng = np.random.default_rng(2026)
    XtXi = np.linalg.inv(X.T @ X)
    bx, by = XtXi @ X.T @ ex, XtXi @ X.T @ ey
    fx, fy = X @ bx, X @ by          # fitted part, so residuals are resampled about it
    out = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        Xi = X[i]
        try:
            Q = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError:
            continue
        cx = (Q @ Xi.T @ (fx[i] + ex[i]))[1]
        cy = (Q @ Xi.T @ (fy[i] + ey[i]))[2]
        out.append(np.hypot(cx - bx[1], cy - by[2]))
    out = np.array(out)
    # the bootstrap spread of the two linear coefficients, combined as the code combines them
    boot = np.sqrt(np.mean(out**2)) / W * 1e6
    print(f'{label:10s} N={n:3d}   HC0 {rep["HC0"]:6.2f}   HC3 {rep["HC3"]:6.2f}   '
          f'bootstrap {boot:6.2f} ppm   (boot/HC0 = {boot/rep["HC0"]:.2f})')
    return rep, boot

base = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"
print('plate-scale standard error, three estimators, definitive 17-frame stack:')
for tag, sub in [("tol 0.5", "definitive_tol0.5"), ("tol 1.0", "definitive_tol1.0"),
                 ("tol 999", "definitive_tol999")]:
    import glob
    f = glob.glob(f"{base}/{sub}/**/TWOD_RESIDUALS.csv", recursive=True)[0]
    run(f, tag)
