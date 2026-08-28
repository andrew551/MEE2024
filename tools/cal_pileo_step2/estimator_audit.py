"""Audit the bootstrap that claimed HC0 understates the plate-scale error.

The worry: 31 stars against 6 parameters per axis. A pairs bootstrap resamples with
replacement, so some draws are near-degenerate -- and summarising the draws with an RMS
lets a few blown-up coefficients set the answer. If that is what happened, the bootstrap
is an artefact and HC0 is not being convicted by it.

Checks: a robust spread instead of RMS, the condition number of each resampled design,
and a delete-one jackknife, which cannot degenerate the same way.
"""
import glob
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0
B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"

def design(px, py):
    x, y = (px-CX)/W, (py-CY)/W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def audit(sub, label, nboot=20000):
    d = pd.read_csv(glob.glob(f"{B}/{sub}/**/TWOD_RESIDUALS.csv", recursive=True)[0])
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    XtXi = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    hc = {}
    for k, f in (('HC0', np.ones(n)), ('HC3', 1/(1-h)**2)):
        c = lambda e: (XtXi @ ((X*(e**2*f)[:, None]).T @ X) @ XtXi)
        hc[k] = np.hypot(c(ex)[1,1]**.5, c(ey)[2,2]**.5)/W*1e6

    cond0 = np.linalg.cond(X.T @ X)
    bx, by = XtXi @ X.T @ ex, XtXi @ X.T @ ey
    rng = np.random.default_rng(2026)
    dx, dy, conds = [], [], []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        Xi = X[i]
        M = Xi.T @ Xi
        conds.append(np.linalg.cond(M))
        try: Q = np.linalg.inv(M)
        except np.linalg.LinAlgError: continue
        dx.append((Q @ Xi.T @ ex[i])[1] - bx[1])
        dy.append((Q @ Xi.T @ ey[i])[2] - by[2])
    dx, dy, conds = np.array(dx), np.array(dy), np.array(conds)

    def comb(sx, sy): return np.hypot(sx, sy)/W*1e6
    rms   = comb(np.sqrt(np.mean(dx**2)), np.sqrt(np.mean(dy**2)))
    sd    = comb(dx.std(ddof=1), dy.std(ddof=1))
    # robust: normal-consistent scale from the interquartile range
    iqr   = lambda v: (np.percentile(v,75)-np.percentile(v,25))/1.349
    robust= comb(iqr(dx), iqr(dy))
    # trimmed: drop the worst 1% of resamples by conditioning
    keep  = conds <= np.percentile(conds, 99)
    trim  = comb(dx[keep].std(ddof=1), dy[keep].std(ddof=1))

    # delete-one jackknife -- no resampling, cannot degenerate
    jx, jy = [], []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        Q = np.linalg.inv(X[m].T @ X[m])
        jx.append((Q @ X[m].T @ ex[m])[1]); jy.append((Q @ X[m].T @ ey[m])[2])
    jx, jy = np.array(jx), np.array(jy)
    jack = comb(np.sqrt((n-1)/n*np.sum((jx-jx.mean())**2)),
                np.sqrt((n-1)/n*np.sum((jy-jy.mean())**2)))

    print(f'\n=== {label}  N={n}, p=6 per axis, p/n={6/n:.2f} ===')
    print(f'  HC0 (reported) {hc["HC0"]:6.2f} ppm     HC3 {hc["HC3"]:6.2f} ppm')
    print(f'  bootstrap: RMS {rms:6.2f}   sd {sd:6.2f}   IQR-robust {robust:6.2f}   '
          f'cond-trimmed {trim:6.2f}')
    print(f'  delete-one jackknife    {jack:6.2f} ppm')
    print(f'  cond(X\'X): fit {cond0:8.1f};  resamples median {np.median(conds):8.1f}, '
          f'p99 {np.percentile(conds,99):9.1f}, max {conds.max():.3g}')
    print(f'  max leverage {h.max():.3f}   (p/n = {6/n:.3f})')

for sub, lab in [('definitive_tol0.5', 'tol 0.5'), ('definitive_tol1.0', 'tol 1.0'),
                 ('definitive_tol999', 'tol 999'), ('subA_owntime', 'sub-stack A')]:
    audit(sub, lab)
