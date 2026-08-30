"""Attribute the FULL-union move 2.6 -> 2.2: G 9.10 removal vs zero-point re-estimation.
Also: sans-anchor fits on the limit-12 unions (anchor-leverage numbers)."""
import os, sys
import numpy as np
lim = sys.argv[1]
os.environ['S2_LIMIT_MAG'] = lim
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools"
src = open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])
rng = np.random.default_rng(7)

def se(U, rx, ry, R):
    n = len(U); boots = []
    for _ in range(200):
        k = rng.integers(0, n, n)
        try: boots.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=2))
        except Exception: pass
    return float(np.std(boots, ddof=1))

for tiers, nm in ((('0p6s','1p2s'), '0.6+1.2'), (('0p1s','0p3s','0p6s','1p2s'), 'FULL')):
    U, rx, ry, R = build_union(tiers)
    anchor = np.hypot(U.px.values-3161, U.py.values-4163) < 6
    g910 = np.hypot(U.px.values-3005, U.py.values-2263) < 8
    for tag, sel in (('all', np.ones(len(U), bool)),
                     ('sans G9.10', ~g910),
                     ('sans anchor', ~anchor)):
        Us, rxs, rys, Rs = U[sel], rx[sel], ry[sel], R[sel]
        Lv = fit_L(Us, rxs, rys, Rs, nuis_deg=2)
        h = 1/np.mean((R_SUN_AS/Rs)**2)
        print(f'limit {lim} {nm:8} {tag:12} N={len(Us):3d} h={h:5.1f} Rsun^2 '
              f'L v-deg2 {Lv:+.3f} +- {se(Us, rxs, rys, Rs):.3f}" (stat)', flush=True)
