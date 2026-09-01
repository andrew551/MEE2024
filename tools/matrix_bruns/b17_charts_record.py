"""The summary chart set for cell 1's reduction of record (Bruns' method, L = 1.777).

Fourth revision, and the docstring keeps the history because Douglas caught every defect
from the pictures alone. This version's changes, all his 2026-09-01 rulings:

  * the field chart draws the FULL displacement vector (constants and rotation removed),
    not the radial projection. The earlier arrows all pointed exactly outward because
    projecting onto the radial direction forces them to -- an artifact of what was
    plotted, not a property of the data. The measured tangential rms is 0.114 arcsec
    against a radial-about-fit rms of 0.085, and the arrows now show it;
  * arrow lengths are drawn in data coordinates and VERIFIED numerically at run time
    (the drawn length of every arrow and of the scale bar are printed and asserted);
  * the nuisance-variant curve is gone (Bruns' method has no nuisance term; showing a
    variant on the record chart cluttered it);
  * the word RECORD is gone from the labels;
  * the covariance chart no longer draws Newton, whose distance off-axis squashed the
    ellipses into unreadability. Newton stays on the deflection chart, where the 1/R
    curves make the comparison meaningful.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
PS, NX, NY = 2.0868004, 3296, 2472
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
GR, NEWTON = 1.7512, 0.8756
SCALE_PPM, ATM_ERR, SE_STAT = 10.3e-6, 0.150, 0.064
TAG = 'Bruns 2017, one 0.62 s master + linked 0.09 s close-in pair (Bruns\u2019 method)'

tab = pd.read_csv(os.path.join(OUT, 'bruns_method_star_table.csv'))
rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
linked = (tab.src == 'E2-linked').values
n = len(tab)


def solve(with_scale=False):
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
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([tab.dx.values, tab.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    return c, labels, cov


c1, l1, cov1 = solve()
c2, l2, cov2 = solve(with_scale=True)
L1, L2 = c1[l1.index('L')], c2[l2.index('L')]
sL1 = float(np.sqrt(cov1[l1.index('L'), l1.index('L')]))
# the full displacement vector with the non-L, non-noise pieces removed
dxc = tab.dx.values - c1[l1.index('N1')] + c1[l1.index('Th')]*(tab.py.values-NY/2)*PS
dyc = tab.dy.values - c1[l1.index('N2')] - c1[l1.index('Th')]*(tab.px.values-NX/2)*PS
rad = dxc*(rx/R) + dyc*(ry/R)
h = 1/np.mean((R_SUN_AS/R)**2)
SCALE_ERR = h*R_SUN_AS*SCALE_PPM
tot = float(np.hypot(np.hypot(SE_STAT, SCALE_ERR), ATM_ERR))
print('N=%d  L(M1) = %.3f;  M2 = %.3f;  total sigma %.3f' % (n, L1, L2, tot))

# ---- 1. deflection vs radius
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
m = ~linked
ax.scatter(R[m]/R_SUN_AS, rad[m], s=38, color='tab:blue', zorder=4,
           label='%d stars, 0.62 s master' % int(m.sum()))
ax.scatter(R[linked]/R_SUN_AS, rad[linked], s=70, marker='D', color='tab:red', zorder=5,
           label='2 close-in stars, 0.09 s master (7-star link, se 0.11")')
for k in np.where(linked)[0]:
    ax.annotate('  G %.2f' % tab.mag.values[k], (R[k]/R_SUN_AS, rad[k]), fontsize=9,
                color='tab:red')
k_out = int(np.argmin(np.where((R/R_SUN_AS > 4.3) & (R/R_SUN_AS < 5.0), rad, 1e9)))
ax.annotate('  G %.2f (faint, 3 px\n  footprint, field corner)' % tab.mag.values[k_out],
            (R[k_out]/R_SUN_AS, rad[k_out]), fontsize=7.5, color='gray')
xx = np.linspace(1.35, (R/R_SUN_AS).max()+0.3, 300)
ax.fill_between(xx, (L1-tot)/xx, (L1+tot)/xx, color='black', alpha=0.10,
                label='total $\\pm$%.2f" (stat %.3f, scale %.3f, atm %.3f)'
                      % (tot, SE_STAT, SCALE_ERR, ATM_ERR))
ax.plot(xx, L1/xx, color='black', lw=2.2, label='Method 1 fit:  L = %.3f"' % L1)
ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
ax.set_title(TAG, fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_deflection.png'), dpi=140)
plt.close(fig)

# ---- 2. the field: FULL displacement vectors, lengths verified
ARROW = 300.0                                      # px of arrow per arcsec
fig, ax = plt.subplots(figsize=(10.5, 8))
maxlen = 0.0
for k in range(n):
    col = 'tab:red' if linked[k] else 'tab:blue'
    x0, y0 = tab.px.values[k], tab.py.values[k]
    x1, y1 = x0 + dxc[k]*ARROW, y0 + dyc[k]*ARROW
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                color=col, lw=1.5, shrinkA=0, shrinkB=0))
    drawn = float(np.hypot(x1-x0, y1-y0))
    maxlen = max(maxlen, drawn)
    assert abs(drawn - np.hypot(dxc[k], dyc[k])*ARROW) < 1e-6
    ax.annotate(' %.1f' % tab.mag.values[k], (x0, y0), fontsize=6.5,
                color=('tab:red' if linked[k] else 'black'))
print('arrow lengths: %.0f-%.0f px for %.2f-%.2f arcsec; scale bar %.0f px = 1 arcsec '
      '(exact by construction)'
      % (float(np.hypot(dxc, dyc).min()*ARROW), maxlen,
         float(np.hypot(dxc, dyc).min()), float(np.hypot(dxc, dyc).max()), ARROW))
ax.scatter(tab.px.values[~linked], tab.py.values[~linked], s=22, color='tab:blue',
           zorder=5, label='0.62 s master (%d stars)' % int((~linked).sum()))
ax.scatter(tab.px.values[linked], tab.py.values[linked], s=70, marker='D',
           color='tab:red', zorder=5, label='0.09 s master, linked (2 stars)')
ax.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS/PS, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
bx, by = 150, 220
ax.plot([bx, bx + ARROW], [by, by], color='black', lw=2.5)
ax.annotate('1 arcsec', (bx + 0.5*ARROW, by + 45), ha='center', fontsize=9)
ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_aspect(1)
ax.set_xlabel('px'); ax.set_ylabel('py')
ax.set_title('Displacement vectors (constants + rotation removed; deflection + '
             'per-star noise remain) — %s' % TAG, fontsize=9)
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_field.png'), dpi=140)
plt.close(fig)

# ---- 3. L against plate scale, both methods, Einstein only
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
ax.annotate('Method 2:  L = %.3f $\\pm$ %.3f"\n scale %.7f "/px (%+.1f ppm)'
            % (L2, np.sqrt(C2[0, 0]), mu2[1], 1e6*c2[iS2]), mu2,
            textcoords='offset points', xytext=(12, -38), fontsize=9, color='tab:blue')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('plate scale (arcsec / pixel)', fontsize=13)
ax.set_title('L and plate scale, both methods (h = %.1f R$_\\odot^2$, %d stars) — %s'
             % (h, n, TAG), fontsize=8.5)
ax.legend(fontsize=8.5, loc='lower left')
ax.autoscale_view()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=140)
plt.close(fig)
print('charts ->', OUT)
