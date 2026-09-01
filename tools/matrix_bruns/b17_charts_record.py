"""The summary chart set for cell 1's REDUCTION OF RECORD (L = 1.720 arcsec).

`b17_charts.py` draws the same three charts for the *windowed+annular* reduction, which
the convention ruling superseded (that one gives 1.556). Its output therefore describes a
number the project no longer quotes, and the two files sit in `matrix_bruns2017/`. This
draws the record: the like-2024 convention -- Gaussian background, footprint moments --
whose tree is `matrix_bruns2017_like2024/`, and writes there, so the charts live beside
the reduction they describe.

Every title carries the reduction's identity for exactly that reason: a chart that does
not say which reduction made it is how the wrong number gets quoted a month later.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

sys.argv = ['x']
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
src = src.replace('matrix_bruns2017', 'matrix_bruns2017_like2024')
src = src.replace("'stage2_constant'", "'stage2'")
exec(compile(src.split("print()\nfor t in (")[0], 'b17_union_record', 'exec'))

OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024"
GR, NEWTON = 1.7512, 0.8756
TAG = 'Bruns 2017, reduction of record: Gaussian background + footprint moments'

U, rx, ry, R = build_union(('EA', 'EB'), 2.0)
n = len(U)
A, labels = design(U.px.values, U.py.values, rx, ry, R, 2)
b = np.concatenate([U.dx.values, U.dy.values])
c, *_ = np.linalg.lstsq(A, b, rcond=None)
iL = labels.index('L')
Lfit = c[iL]
resid = b - A@c
dof = len(b) - len(c)
cov = (float(resid @ resid)/dof) * np.linalg.inv(A.T @ A)
sL = float(np.sqrt(cov[iL, iL]))
clean = b - (A@c - A[:, iL]*c[iL])
defl = clean[:n]*(rx/R) + clean[n:]*(ry/R)
rng = np.random.default_rng(3)
boots = []
for _ in range(300):
    k = rng.integers(0, n, n)
    try:
        boots.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=2))
    except Exception:
        pass
se = float(np.std(boots, ddof=1))
h = 1/np.mean((R_SUN_AS/R)**2)
SCALE_ERR, ATM_ERR = 0.105, 0.150
tot = float(np.hypot(np.hypot(se, SCALE_ERR), ATM_ERR))
print('record union: N=%d  L = %.3f +- %.3f (stat) +- %.3f (scale) +- %.3f (atm), total %.3f'
      % (n, Lfit, se, SCALE_ERR, ATM_ERR, tot))

# ---- 1. deflection vs radius
fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.axhline(0, color='black', lw=1)
ax.scatter(R/R_SUN_AS, defl, s=32, color='tab:blue', label='%d stars (0.62 s tiers, R > 2 R$_\\odot$)' % n)
xx = np.linspace(1.9, (R/R_SUN_AS).max()+0.3, 200)
ax.plot(xx, Lfit/xx, color='black', lw=2,
        label='fit  L = %.3f $\\pm$ %.3f (stat) $\\pm$ %.3f (scale) $\\pm$ %.3f (atm)"'
              % (Lfit, se, SCALE_ERR, ATM_ERR))
ax.fill_between(xx, (Lfit-tot)/xx, (Lfit+tot)/xx, color='black', alpha=0.10,
                label='total $\\pm$%.2f"' % tot)
ax.plot(xx, GR/xx, color='green', lw=1.4, label='Einstein  1.751"')
ax.plot(xx, NEWTON/xx, color='orange', lw=1.4, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec)', fontsize=13)
ax.set_title('Deflection vs radius — %s' % TAG, fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_deflection.png'), dpi=140)
plt.close(fig)

# ---- 2. the field, with the measured displacements
fig, ax = plt.subplots(figsize=(9.5, 7.5))
ax.scatter(U.px.values, U.py.values, s=30, color='tab:blue')
q = ax.quiver(U.px.values, U.py.values, clean[:n]*(rx/R), clean[n:]*(ry/R),
              angles='xy', scale_units='xy', scale=0.004, color='tab:red', width=0.004)
ax.quiverkey(q, 0.86, 0.05, 1.0, '1 arcsec', coordinates='axes')
ax.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS/PS, color='black'))
ax.add_patch(Circle((SUNPX, SUNPY), 2*R_SUN_AS/PS, fill=False, ls=':', color='gray'))
for k in range(n):
    if U.mag.values[k] <= 8.7:
        ax.annotate(' %.1f' % U.mag.values[k], (U.px.values[k], U.py.values[k]), fontsize=7)
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect(1)
ax.set_xlabel('px'); ax.set_ylabel('py')
ax.set_title('Field and radial displacements (x250) — %s' % TAG, fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_field.png'), dpi=140)
plt.close(fig)

# ---- 3. L against the imported plate scale, with the eq-23 coupling
fig, ax = plt.subplots(figsize=(9, 7))
dS = 10.3e-6
pc = h*R_SUN_AS*dS
C = np.array([[se**2 + pc**2, -pc*dS*PS], [-pc*dS*PS, (dS*PS)**2]])
mu = np.array([Lfit, PS])
vals, vecs = np.linalg.eigh(C)
ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
for kk, ls in ((1, '-'), (2, ':')):
    ax.add_patch(Ellipse(mu, 2*kk*np.sqrt(vals[1]), 2*kk*np.sqrt(vals[0]), angle=ang,
                         fill=False, color='darkred', linestyle=ls,
                         label='%d$\\sigma$ (stat + scale)' % kk))
ax.scatter(*mu, marker='+', s=110, color='darkred')
ax.axvline(GR, color='green', lw=1.4, label='Einstein')
ax.axvline(NEWTON, color='orange', lw=1.4, ls='--', label='Newton')
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('imported plate scale (arcsec / pixel)', fontsize=13)
ax.set_title('L against the imported plate scale (h = %.1f R$_\\odot^2$) — %s' % (h, TAG),
             fontsize=10)
ax.legend(loc='lower left', fontsize=9)
ax.autoscale_view()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=140)
plt.close(fig)
print('charts ->', OUT)
