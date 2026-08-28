"""What would actually move the CAL_piLeo plate-scale error?

Each row refits the linear+quadratic model on a modified star list and reports the
bootstrap standard error, which is the honest one (HC0 runs 25-52% low here). Offline,
because these are questions about the star list, not about the pipeline.
"""
import glob
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0
BASE = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"

def design(px, py):
    x, y = (px - CX)/W, (py - CY)/W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def boot_se(d, nboot=8000, seed=2026):
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    XtXi = np.linalg.inv(X.T @ X)
    bx, by = XtXi @ X.T @ ex, XtXi @ X.T @ ey
    fx, fy = X @ bx, X @ by
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        Xi = X[i]
        try: Q = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError: continue
        acc.append(np.hypot((Q @ Xi.T @ (fx[i]+ex[i]))[1] - bx[1],
                            (Q @ Xi.T @ (fy[i]+ey[i]))[2] - by[2]))
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    return np.sqrt(np.mean(np.array(acc)**2))/W*1e6, n, h.max(), \
           np.sqrt(np.mean(d.error_arcsec**2))

def show(tag, d):
    se, n, hmax, rms = boot_se(d)
    print(f'  {tag:38s} N={n:3d}  rms={rms:.4f}"  max h={hmax:.3f}  se={se:6.2f} ppm')
    return se

f999 = glob.glob(f"{BASE}/definitive_tol999/**/TWOD_RESIDUALS.csv", recursive=True)[0]
f10  = glob.glob(f"{BASE}/definitive_tol1.0/**/TWOD_RESIDUALS.csv", recursive=True)[0]
a, b = pd.read_csv(f999), pd.read_csv(f10)

print('from the untruncated match list (tol 999, 105 stars):')
base999 = show('as fitted', a)
show('drop V < 8 (F16: the saturated end)', a[a.magV >= 8])
show('drop V < 9', a[a.magV >= 9])
show('drop V > 11.5 (the faint end)', a[a.magV <= 11.5])
show('keep 8 <= V <= 11.5 (both ends)', a[(a.magV >= 8) & (a.magV <= 11.5)])

print('\nfrom the recommended fit (tol 1.0, 73 stars):')
base10 = show('as fitted  [THE BASELINE]', b)
show('drop V < 8 (F16)', b[b.magV >= 8])
show('drop the single highest-leverage star', b.drop(
    index=int(np.argmax(np.einsum('ij,jk,ik->i', design(b.px.values, b.py.values),
              np.linalg.inv(design(b.px.values, b.py.values).T @ design(b.px.values, b.py.values)),
              design(b.px.values, b.py.values))))))
print(f'\n  for reference, F&L eq.23 spec for sigma_L <= 0.1" on the Leon field: 5.8 ppm')
