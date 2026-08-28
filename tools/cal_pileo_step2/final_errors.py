"""Consistent standard errors for every arm: HC0 as reported, HC3, and a delete-one jackknife.

The pairs bootstrap used earlier is dropped as the headline: its draw distribution is
heavy-tailed, so summarising it with an RMS sat ~15% above its own robust scale. HC3 and
the jackknife agree with each other and with the bootstrap's robust spread, and HC3 is
what the code could actually report.
"""
import glob, json
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0
B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"

def design(px, py):
    x, y = (px-CX)/W, (py-CY)/W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def errors(sub):
    j = json.load(open(glob.glob(f"{B}/{sub}/**/distortion_results.txt", recursive=True)[0]))
    d = pd.read_csv(glob.glob(f"{B}/{sub}/**/TWOD_RESIDUALS.csv", recursive=True)[0])
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    Q = np.linalg.inv(X.T @ X)
    h = np.einsum('ij,jk,ik->i', X, Q, X)
    out = {}
    for k, f in (('HC0', np.ones(n)), ('HC3', 1/(1-h)**2)):
        c = lambda e: (Q @ ((X*(e**2*f)[:, None]).T @ X) @ Q)
        out[k] = np.hypot(c(ex)[1,1]**.5, c(ey)[2,2]**.5)/W*1e6
    jx, jy = [], []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        Qk = np.linalg.inv(X[m].T @ X[m])
        jx.append((Qk @ X[m].T @ ex[m])[1]); jy.append((Qk @ X[m].T @ ey[m])[2])
    jx, jy = np.array(jx), np.array(jy)
    out['jack'] = np.hypot(np.sqrt((n-1)/n*np.sum((jx-jx.mean())**2)),
                           np.sqrt((n-1)/n*np.sum((jy-jy.mean())**2)))/W*1e6
    return (j['#stars used'], j['final rms error (arcseconds)'],
            j['platescale (arcseconds/pixel)'], out['HC0'], out['HC3'], out['jack'], h.max())

arms = [('tol 0.5',            'definitive_tol0.5'),
        ('tol 0.7',            'tol_0.7'),
        ('tol 1.0  BASELINE',  'definitive_tol1.0'),
        ('tol 1.5',            'tol_1.5'),
        ('tol 2.0',            'tol_2.0'),
        ('tol 999',            'definitive_tol999'),
        ('29 frames tol 1.0',  'all29_tol1.0'),
        ('29 frames tol 0.7',  'all29_tol0.7'),
        ('sub-stack A',        'subA_owntime'),
        ('sub-stack B',        'subB_owntime'),
        ('sub-stack C',        'subC_owntime'),
        ('fixed=linear',       'order_linear'),
        ('fixed=None (cubic)', 'order_cubic')]
print(f"{'arm':20s} {'N':>4s} {'rms':>7s} {'platescale':>11s} {'HC0':>7s} {'HC3':>7s} "
      f"{'jack':>7s} {'HC3/HC0':>8s} {'max h':>6s}")
for lab, sub in arms:
    try: n, rms, ps, a, b, c, hm = errors(sub)
    except Exception as e: print(f"{lab:20s} -- {e}"); continue
    print(f"{lab:20s} {n:4d} {rms:7.4f} {ps:11.7f} {a:7.2f} {b:7.2f} {c:7.2f} "
          f"{b/a:8.2f} {hm:6.3f}")
