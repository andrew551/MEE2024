"""Cell-1 finishers: the G7.52-excluded inner fit + the standard charts."""
import os, sys
sys.argv = ['x']
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools/matrix_bruns"
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rng = np.random.default_rng(3)
def boot(U, rx, ry, R, nd):
    n = len(U); bs = []
    for _ in range(200):
        k = rng.integers(0, n, n)
        try: bs.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=nd))
        except Exception: pass
    return float(np.std(bs, ddof=1))

U2, rx2, ry2, R2 = build_union(('EA','E2','EB'), 1.45)
g752 = np.hypot(U2.px.values-2102, U2.py.values-1241) < 6
Ux, rxx, ryx, Rx = U2[~g752], rx2[~g752], ry2[~g752], R2[~g752]
Lb = fit_L(Ux, rxx, ryx, Rx); Lv = fit_L(Ux, rxx, ryx, Rx, nuis_deg=2)
h = 1/np.mean((R_SUN_AS/Rx)**2)
print(f'INNER sans G7.52: N={len(Ux)} h={h:.1f} Rsun^2  L base {Lb:+.3f} +- {boot(Ux,rxx,ryx,Rx,None):.3f}  '
      f'L v-deg2 {Lv:+.3f} +- {boot(Ux,rxx,ryx,Rx,2):.3f} (stat)')

# ---- charts: field + deflection vs radius (v-deg2, R>2 default union)
OUTP = r'D:/MEE2024 output/MEE_output/matrix_bruns2017'
U, rx, ry, R = build_union(('EA','EB'), 2.0)
A, labels = design(U.px.values, U.py.values, rx, ry, R, 2)
b = np.concatenate([U.dx.values, U.dy.values])
c, *_ = np.linalg.lstsq(A, b, rcond=None)
iL = labels.index('L')
clean = b - (A@c - A[:, iL]*c[iL])
n = len(U)
defl = clean[:n]*(rx/R) + clean[n:]*(ry/R)

fig, axd = plt.subplots(figsize=(9.5, 6.5))
axd.axhline(0, color='black', lw=1)
axd.scatter(R/R_SUN_AS, defl, s=30, label='0.62 s union (R > 2)')
# the two E2 inner stars at their per-frame-verified values (b17_perframe2.py:
# same-frame reference, quadratic local background, 11 raw frames each)
axd.errorbar([1.62], [1.632], yerr=[0.204], fmt='s', ms=8, color='tab:green',
             capsize=3, label='G 7.09 (E2, per-frame verified)')
axd.errorbar([1.49], [0.463], yerr=[0.229], fmt='s', ms=8, color='tab:olive',
             capsize=3, label='G 7.52 (E2, per-frame verified)')
xx = np.linspace(1.3, 5.3, 100)
axd.plot(xx, c[iL]/xx, color='black', label=f'fit L = {c[iL]:.3f}"')
axd.plot(xx, L_REF/xx, color='green', lw=1, alpha=0.7, label='GR 1.751"')
axd.plot(xx, 0.8756/xx, color='orange', lw=1, alpha=0.7, label='Newton 0.876"')
axd.set_xlabel('radial position (solar radii)', fontsize=13)
axd.set_ylabel('radial deflection (arcsec)', fontsize=13)
axd.set_title('Bruns 2017 through the Leon chain: deflections, v-deg2 nuisance removed')
axd.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUTP, 'deflection_b17.png'), dpi=140); plt.close(fig)

fig, axf = plt.subplots(figsize=(9, 7))
axf.scatter(U.px.values, U.py.values, s=30, color='tab:blue')
q = axf.quiver(U.px.values, U.py.values, clean[:n]*(rx/R), clean[n:]*(ry/R),
               angles='xy', scale_units='xy', scale=0.005, color='tab:red', width=0.004)
axf.quiverkey(q, 0.88, 0.06, 1.0, '1 arcsec', coordinates='axes')
axf.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS/PS, color='black'))
axf.add_patch(Circle((SUNPX, SUNPY), 2*R_SUN_AS/PS, fill=False, ls=':', color='gray'))
for k in range(n):
    if U.mag.values[k] <= 8.7:
        axf.annotate(f' {U.mag.values[k]:.1f}', (U.px.values[k], U.py.values[k]), fontsize=7)
axf.set_xlim(0, NX); axf.set_ylim(0, NY); axf.set_aspect(1)
axf.set_xlabel('px'); axf.set_ylabel('py')
axf.set_title(f'Bruns 2017 eclipse field: {n} stars, radial components of the '
              f'L-view displacements (x200)')
fig.tight_layout(); fig.savefig(os.path.join(OUTP, 'field_b17.png'), dpi=140); plt.close(fig)
print('charts ->', OUTP)
