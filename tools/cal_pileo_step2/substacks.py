"""The three disjoint sub-stacks, against the combined reduction.

A, B and C share no frames, so their scatter measures the part of the error that is
independent between frames -- centroid noise. They share the same stars, the same
catalogue and the same frozen cubic, so everything else is common-mode and invisible in
that scatter. The comparison therefore splits the total error into the part more data
would fix and the part it would not.
"""
import glob, json
import numpy as np, pandas as pd

W, CX, CY = 3124.0, 3124.0, 2088.0
B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"

def design(px, py):
    x, y = (px-CX)/W, (py-CY)/W
    return np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y])

def boot(sub, nboot=8000):
    j = json.load(open(glob.glob(f"{B}/{sub}/**/distortion_results.txt", recursive=True)[0]))
    d = pd.read_csv(glob.glob(f"{B}/{sub}/**/TWOD_RESIDUALS.csv", recursive=True)[0])
    X = design(d.px.values, d.py.values)
    ex, ey = d.dx_px.values, d.dy_px.values
    n = len(d)
    Q0 = np.linalg.inv(X.T @ X)
    bx, by = Q0 @ X.T @ ex, Q0 @ X.T @ ey
    fx, fy = X @ bx, X @ by
    rng = np.random.default_rng(2026)
    acc = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        Xi = X[i]
        try: Q = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError: continue
        acc.append(np.hypot((Q @ Xi.T @ (fx[i]+ex[i]))[1]-bx[1], (Q @ Xi.T @ (fy[i]+ey[i]))[2]-by[2]))
    return (j['platescale (arcseconds/pixel)'], j['#stars used'],
            j['final rms error (arcseconds)'],
            j['platescale_relative_uncertainty']*1e6,
            np.sqrt(np.mean(np.array(acc)**2))/W*1e6)

print('each sub-stack reduced at its OWN exposure-weighted mid-time:')
rows = {}
for tag, sub, t in [('A  6 x 1 s', 'subA_owntime', '18:29:22.7'),
                    ('B  8 x 2 s', 'subB_owntime', '18:29:35.3'),
                    ('C  3 x 1 s', 'subC_owntime', '18:29:51.9')]:
    ps, n, rms, hc0, bs = boot(sub)
    rows[tag] = (ps, bs)
    print(f'  {tag}  t={t}  N={n:3d}  rms={rms:.4f}"  ps={ps:.7f}  HC0={hc0:5.1f}  boot={bs:5.1f} ppm')

ps = np.array([v[0] for v in rows.values()])
se = np.array([v[1] for v in rows.values()])
mean = ps.mean()
print(f'\n  sub-stack spread: {(ps.max()-ps.min())/mean*1e6:5.1f} ppm total, '
      f'sd {ps.std(ddof=1)/mean*1e6:.1f} ppm on 3 values')
w = 1/se**2
wmean = (ps*w).sum()/w.sum()
print(f'  inverse-variance weighted mean of the three: {wmean:.7f}')

cps, cn, crms, chc0, cbs = boot('definitive_tol1.0')
print(f'\n  combined 17-frame reduction:                 {cps:.7f}  '
      f'(HC0 {chc0:.1f}, bootstrap {cbs:.1f} ppm)')
print(f'  combined minus sub-stack weighted mean: {(cps-wmean)/mean*1e6:+.1f} ppm  '
      f'({abs(cps-wmean)/mean*1e6/cbs:.2f} sigma of the combined bootstrap error)')

print(f"""
  Reading:
    the three sub-stacks share no frames, so their {ps.std(ddof=1)/mean*1e6:.1f} ppm scatter bounds the
    frame-independent (centroid-noise) part of the error;
    they share every star, the catalogue and the frozen cubic, so the remaining
    ~{np.sqrt(max(cbs**2 - (ps.std(ddof=1)/mean*1e6)**2,0)):.0f} ppm of the combined stack's {cbs:.0f} ppm is common-mode and does not
    fall with more frames or longer exposure.""")
