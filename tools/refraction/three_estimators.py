"""HC0 / HC3 / delete-one jackknife for any set of stage-2 runs, order-aware.

The free-basis width depends on distortion_fixed_coefficients: a free-cubic fit has 10
columns (constant..cubic), a quadratic-free fit 6. HC0 must reproduce the reported
platescale_relative_uncertainty exactly -- that is the per-run validity check.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

MAP = {"constant": 0, "linear": 1, "quadratic": 2, "cubic": 3, "None": 3, None: 3}

def design(px, py, order, nx, ny):
    w = max(nx, ny) / 2
    x, y = (px - nx/2) / w, (py - ny/2) / w
    cols = [np.ones_like(x)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            cols.append(y**j * x**(i - j))
    return np.column_stack(cols), w

def three(run_dir, nx=6248, ny=4176):
    r = glob.glob(os.path.join(run_dir, "**", "distortion_results.txt"), recursive=True)
    if not r:
        return None
    j = json.load(open(r[0]))
    d = pd.read_csv(glob.glob(os.path.join(run_dir, "**", "TWOD_RESIDUALS.csv"),
                              recursive=True)[0])
    fixed = j.get("fixed distortion order")
    order = MAP.get(fixed, 3) if fixed not in (None, "None") else MAP[j["distortion order"]]
    X, w = design(d.px.values, d.py.values, order, nx, ny)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    Q = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, Q, X)
    out = {}
    for k, fw in (("HC0", np.ones(n)), ("HC3", 1/(1-h)**2)):
        c = lambda e: (Q @ ((X*(e**2*fw)[:, None]).T @ X) @ Q)
        out[k] = np.hypot(c(ex)[1, 1]**.5, c(ey)[2, 2]**.5)/w*1e6
    jx, jy = [], []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        Qk = np.linalg.inv(X[m].T @ X[m])
        jx.append((Qk @ X[m].T @ ex[m])[1]); jy.append((Qk @ X[m].T @ ey[m])[2])
    jx, jy = np.array(jx), np.array(jy)
    jack = np.hypot(np.sqrt((n-1)/n*np.sum((jx-jx.mean())**2)),
                    np.sqrt((n-1)/n*np.sum((jy-jy.mean())**2)))/w*1e6
    rep = j["platescale_relative_uncertainty"]*1e6
    return dict(n=n, date=j.get("observation_date"), ps=j["platescale (arcseconds/pixel)"],
                rms=j["final rms error (arcseconds)"], rep=rep, hc0=out["HC0"],
                hc3=out["HC3"], jack=jack, ok=abs(out["HC0"]-rep)/max(rep,1e-9) < 0.02,
                free=order)

if __name__ == "__main__":
    base = sys.argv[1]
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 6248
    ny = int(sys.argv[3]) if len(sys.argv) > 3 else 4176
    rows = []
    for sub in sorted(os.listdir(base)):
        p = os.path.join(base, sub)
        if not os.path.isdir(p): continue
        t = three(p, nx, ny)
        if t:
            rows.append((sub, t))
            print(f"{sub:14s} date={t['date']} N={t['n']:4d} free_order={t['free']} "
                  f"ps={t['ps']:.7f} rms={t['rms']:.4f} rep={t['rep']:6.2f} "
                  f"HC0={t['hc0']:6.2f} HC3={t['hc3']:6.2f} jack={t['jack']:6.2f} "
                  f"{'OK' if t['ok'] else 'HC0-MISMATCH'}")
