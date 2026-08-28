"""The tolerance ladder, scored on the bootstrap rather than on HC0.

HC0 runs low exactly where leverage concentrates, which is the low-tolerance end -- so
choosing the tolerance by the reported standard error picks the configuration whose error
is most understated. This rescores the ladder on a pairs bootstrap.
"""
import glob, json
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0
B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"

def design(px, py):
    x, y = (px - CX)/W, (py - CY)/W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def stats(d, nboot=8000):
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    hc = {}
    for k in ('HC0', 'HC3'):
        f = np.ones(n) if k == 'HC0' else 1/(1-h)**2
        c = lambda e: (XtXi @ ((X*(e**2*f)[:, None]).T @ X) @ XtXi)
        hc[k] = np.hypot(c(ex)[1, 1]**.5, c(ey)[2, 2]**.5)/W*1e6
    bx, by = XtXi @ X.T @ ex, XtXi @ X.T @ ey
    fx, fy = X @ bx, X @ by
    rng = np.random.default_rng(2026)
    acc = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        Xi = X[i]
        try: Q = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError: continue
        acc.append(np.hypot((Q @ Xi.T @ (fx[i]+ex[i]))[1]-bx[1],
                            (Q @ Xi.T @ (fy[i]+ey[i]))[2]-by[2]))
    return hc['HC0'], hc['HC3'], np.sqrt(np.mean(np.array(acc)**2))/W*1e6, h.max()

rows = []
for tol, sub in [(0.5, 'definitive_tol0.5'), (0.7, 'tol_0.7'), (1.0, 'definitive_tol1.0'),
                 (1.5, 'tol_1.5'), (2.0, 'tol_2.0'), (999, 'definitive_tol999')]:
    g = glob.glob(f"{B}/{sub}/**/distortion_results.txt", recursive=True)
    if not g: continue
    j = json.load(open(g[0]))
    d = pd.read_csv(glob.glob(f"{B}/{sub}/**/TWOD_RESIDUALS.csv", recursive=True)[0])
    hc0, hc3, bs, hmax = stats(d)
    rows.append((tol, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)'], hc0, hc3, bs, hmax))

print(f"{'tol':>5s} {'N':>4s} {'rms':>7s} {'platescale':>11s} {'HC0':>7s} {'HC3':>7s} "
      f"{'boot':>7s} {'max h':>6s}")
best = min(rows, key=lambda r: r[6])
for r in rows:
    mark = '  <-- lowest honest error' if r is best else ''
    print(f"{r[0]:5g} {r[1]:4d} {r[2]:7.4f} {r[3]:11.7f} {r[4]:7.2f} {r[5]:7.2f} "
          f"{r[6]:7.2f} {r[7]:6.3f}{mark}")
ps = np.array([r[3] for r in rows if r[0] <= 1.0])
print(f"\nplate scale across tol 0.5-1.0: spread {(ps.max()-ps.min())/ps.mean()*1e6:.1f} ppm")
