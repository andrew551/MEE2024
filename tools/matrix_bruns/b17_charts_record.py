"""The summary chart set for cell 1's reduction of record: BRUNS' OWN METHOD.

The reduction: one 0.62 s master (all 34 EA+EB frames), the two close-in stars carried
from the 0.09 s master by Bruns' seven-brightest-common-stars offset link, Method 1 with
the imported bracket scale (tools/matrix_bruns/b17_bruns_method.py). That gives
L = 1.777 +- 0.065 (stat) against Bruns 2018's published 1.752 +- 0.060.

Chart-history note, because two earlier versions of this file were wrong in ways Douglas
caught from the pictures alone: the first drew the SUPERSEDED windowed reduction; the
second fixed the convention but (a) plotted (dx*ux, dy*uy) as arrows -- an elementwise
product that is not a vector, so nothing pointed outward, (b) omitted the two inner stars
without a word, (c) had no Method 2, no numbers on the covariance chart, an unexplained
dotted circle, and magnitudes on an arbitrary subset of stars. And when those were fixed
the quiver arrows drew ~1.5x their stated key length -- matplotlib's quiverkey is
unreliable under scale_units='xy'. The arrows are now drawn MANUALLY in data coordinates
with a manual scale bar, so length is exact by construction.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
VX, VY = 0.447, -0.895
GR, NEWTON = 1.7512, 0.8756
SCALE_PPM, ATM_ERR = 10.3e-6, 0.150
TAG = 'Bruns 2017 — reduction of record (Bruns\u2019 method: one 0.62 s master + linked 0.09 s inner stars)'

tab = pd.read_csv(os.path.join(OUT, 'bruns_method_star_table.csv'))
rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
linked = (tab.src == 'E2-linked').values
n = len(tab)


def solve(with_scale=False, nuis_deg=None):
    xs, ys = (tab.px.values-NX/2)/W_NORM, (tab.py.values-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(tab.py.values-NY/2)*PS]
    cols_y = [Z, np.ones(n), (tab.px.values-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((tab.px.values-NX/2)*PS)
        cols_y.append((tab.py.values-NY/2)*PS)
        labels.append('S')
    cols_x.append(ur*R_SUN_AS/R)
    cols_y.append(vr*R_SUN_AS/R)
    labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for jj in range(nuis_deg+1-i):
                if i == 0 and jj == 0:
                    continue
                cols_x.append(VX*xs**i*ys**jj)
                cols_y.append(VY*xs**i*ys**jj)
                labels.append('v%d%d' % (i, jj))
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([tab.dx.values, tab.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    iL = labels.index('L')
    clean = b - (A@c - A[:, iL]*c[iL])
    defl = clean[:n]*(rx/R) + clean[n:]*(ry/R)
    return c, labels, cov, defl


c1, l1, cov1, defl = solve()                       # Method 1 base -- Bruns' method
cv, lv, _, _ = solve(nuis_deg=2)
c2, l2, cov2, _ = solve(with_scale=True)
L1, Lv, L2 = c1[l1.index('L')], cv[lv.index('L')], c2[l2.index('L')]
sL1 = float(np.sqrt(cov1[l1.index('L'), l1.index('L')]))
h = 1/np.mean((R_SUN_AS/R)**2)
SCALE_ERR = h*R_SUN_AS*SCALE_PPM
SE_STAT = 0.065                                    # bootstrap, b17_bruns_method.py
tot = float(np.hypot(np.hypot(SE_STAT, SCALE_ERR), ATM_ERR))
print('record: N=%d  L(M1) = %.3f +- %.3f stat +- %.3f scale +- %.3f atm (total %.3f); '
      'v-deg2 %.3f; M2 %.3f' % (n, L1, SE_STAT, SCALE_ERR, ATM_ERR, tot, Lv, L2))

# ---- 1. deflection vs radius
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
m = ~linked
ax.scatter(R[m]/R_SUN_AS, defl[m], s=38, color='tab:blue', zorder=4,
           label='%d stars, 0.62 s master' % int(m.sum()))
ax.scatter(R[linked]/R_SUN_AS, defl[linked], s=70, marker='D', color='tab:red', zorder=5,
           label='2 close-in stars, 0.09 s master (Bruns 7-star link, se 0.08")')
for k in np.where(linked)[0]:
    ax.annotate('  G %.2f' % tab.mag.values[k], (R[k]/R_SUN_AS, defl[k]), fontsize=9,
                color='tab:red')
xx = np.linspace(1.35, (R/R_SUN_AS).max()+0.3, 300)
ax.fill_between(xx, (L1-tot)/xx, (L1+tot)/xx, color='black', alpha=0.10,
                label='total $\\pm$%.2f" (stat %.3f, scale %.3f, atm %.3f)'
                      % (tot, SE_STAT, SCALE_ERR, ATM_ERR))
ax.plot(xx, L1/xx, color='black', lw=2.2,
        label='RECORD (Method 1, Bruns\u2019 method)  L = %.3f"' % L1)
ax.plot(xx, Lv/xx, color='tab:purple', lw=1.4, ls='-.',
        label='with the vertical nuisance:  L = %.3f"' % Lv)
ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
ax.set_title(TAG, fontsize=10)
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_deflection.png'), dpi=140)
plt.close(fig)

# ---- 2. the field, arrows drawn by hand so lengths are exact
ARROW = 300.0                                      # px of arrow per arcsec of deflection
fig, ax = plt.subplots(figsize=(10.5, 8))
ux, uy = rx/R, ry/R
for k in range(n):
    col = 'tab:red' if linked[k] else 'tab:blue'
    x0, y0 = tab.px.values[k], tab.py.values[k]
    ax.annotate('', xy=(x0 + defl[k]*ux[k]*ARROW, y0 + defl[k]*uy[k]*ARROW),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6,
                                shrinkA=0, shrinkB=0))
    ax.annotate(' %.1f' % tab.mag.values[k], (x0, y0), fontsize=6.5,
                color=('tab:red' if linked[k] else 'black'))
ax.scatter(tab.px.values[~linked], tab.py.values[~linked], s=22, color='tab:blue',
           zorder=5, label='0.62 s master (%d stars)' % int((~linked).sum()))
ax.scatter(tab.px.values[linked], tab.py.values[linked], s=70, marker='D',
           color='tab:red', zorder=5, label='0.09 s master, linked (2 stars)')
ax.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS/PS, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
ax.add_patch(Circle((SUNPX, SUNPY), 1.45*R_SUN_AS/PS, fill=False, ls='--',
                    color='tab:red', alpha=0.6, zorder=3,
                    label='1.45 R$_\\odot$ — inner admission limit'))
# manual scale bar: exact by construction
bx, by = 150, 220
ax.plot([bx, bx + 1.0*ARROW], [by, by], color='black', lw=2.5)
ax.annotate('1 arcsec of deflection', (bx + 0.5*ARROW, by + 45), ha='center', fontsize=9)
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect(1)
ax.set_xlabel('px'); ax.set_ylabel('py')
ax.set_title('Radial deflection vectors, all G magnitudes labelled — %s' % TAG,
             fontsize=9.5)
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_field.png'), dpi=140)
plt.close(fig)

# ---- 3. L against plate scale, both methods, all numbers on the chart
fig, ax = plt.subplots(figsize=(9.5, 7))
pc = h*R_SUN_AS*SCALE_PPM
C1 = np.array([[sL1**2 + pc**2, -pc*SCALE_PPM*PS], [-pc*SCALE_PPM*PS, (SCALE_PPM*PS)**2]])
mu1 = np.array([L1, PS])
iL2, iS2 = l2.index('L'), l2.index('S')
C2 = np.array([[cov2[iL2, iL2], cov2[iL2, iS2]*PS],
               [cov2[iS2, iL2]*PS, cov2[iS2, iS2]*PS**2]])
mu2 = np.array([L2, PS*(1 + c2[iS2])])


def draw(cov, mu, color, name):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for kk, ls in ((1, '-'), (2, ':')):
        ax.add_patch(Ellipse(mu, 2*kk*np.sqrt(vals[1]), 2*kk*np.sqrt(vals[0]), angle=ang,
                             fill=False, color=color, linestyle=ls,
                             label='%d$\\sigma$ — %s' % (kk, name)))
    ax.scatter(*mu, marker='+', s=110, color=color, zorder=5)


draw(C1, mu1, 'darkred', 'Method 1 (scale imported)')
draw(C2, mu2, 'tab:blue', 'Method 2 (scale free)')
ax.annotate('Method 1:  L = %.3f $\\pm$ %.3f"\n scale %.7f "/px (imported, $\\pm$%.1f ppm)'
            % (L1, np.sqrt(C1[0, 0]), PS, 1e6*SCALE_PPM), mu1,
            textcoords='offset points', xytext=(12, 16), fontsize=9, color='darkred')
ax.annotate('Method 2:  L = %.3f $\\pm$ %.3f"\n scale %.7f "/px (%+.1f ppm from imported)'
            % (L2, np.sqrt(C2[0, 0]), mu2[1], 1e6*c2[iS2]), mu2,
            textcoords='offset points', xytext=(12, -38), fontsize=9, color='tab:blue')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.axvline(NEWTON, color='orange', lw=1.5, ls='--', label='Newton 0.876"')
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('plate scale (arcsec / pixel)', fontsize=13)
ax.set_title('L and plate scale, both methods (h = %.1f R$_\\odot^2$, %d stars) — %s'
             % (h, n, TAG), fontsize=8.5)
ax.legend(fontsize=8.5, loc='lower left')
ax.autoscale_view()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=140)
plt.close(fig)
print('charts ->', OUT)
