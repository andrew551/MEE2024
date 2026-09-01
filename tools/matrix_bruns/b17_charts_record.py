"""The summary chart set for cell 1's reduction of record (Bruns' method, L = 1.777).

Fifth revision, all from Douglas' chart reviews. Changes this round, each with its cause:

  * THE MISSING VECTORS ARE FOUND: arrows whose endpoints left the sensor area were
    silently CLIPPED at the axes limits (G 7.09's arrow ends at py ~ 2714 on a 2472-px
    axis; the 'missing' outer stars all sat near edges with outward arrows). The axes now
    extend 450 px beyond the sensor, the sensor edge is drawn as a grey rectangle, and a
    runtime assertion fails the script if any endpoint leaves the axes.
  * both scale bars (1 arcsec, and the typical per-star scatter) sit in the top-left
    with the legend; titles shortened to fit one line;
  * the two inner stars lose their magnitude labels on the deflection chart
    (inconsistent with the rest and illegibly close together);
  * the covariance chart: one-sigma ellipses only; the y-axis is plate scale MINUS the
    imported value in ppm, which kills matplotlib's orphaned '+2.086...' offset text; and
    it states explicitly that its ellipses carry stat+scale only, with the atmosphere
    term quoted beside them -- the deflection chart's band is the total, and the two now
    cross-reference instead of looking inconsistent;
  * a fourth chart: the G <= 13 variant (39 stars), with stars more than 3 sigma from
    the fit labelled -- Douglas' request, and the visual form of the mag-11 finding
    (the error grows 37 % and L drifts down when the faint stars come in).
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
PS, NX, NY = 2.0868004, 3296, 2472
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
GR, NEWTON = 1.7512, 0.8756
SCALE_PPM, ATM_ERR, SE_STAT = 10.3e-6, 0.150, 0.064
TAG = 'Bruns 2017, Bruns\u2019 method'


def load(name):
    t = pd.read_csv(os.path.join(OUT, name))
    rx_, ry_ = (t.px.values-SUNPX)*PS, (t.py.values-SUNPY)*PS
    return t, rx_, ry_, np.hypot(rx_, ry_)


def solve(t, rx_, ry_, R_, with_scale=False):
    m = len(t)
    ur, vr = rx_/R_, ry_/R_
    Z = np.zeros(m)
    cols_x = [np.ones(m), Z, -(t.py.values-NY/2)*PS]
    cols_y = [Z, np.ones(m), (t.px.values-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((t.px.values-NX/2)*PS)
        cols_y.append((t.py.values-NY/2)*PS)
        labels.append('S')
    cols_x.append(ur*R_SUN_AS/R_)
    cols_y.append(vr*R_SUN_AS/R_)
    labels.append('L')
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([t.dx.values, t.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    dxc = t.dx.values - c[labels.index('N1')] + c[labels.index('Th')]*(t.py.values-NY/2)*PS
    dyc = t.dy.values - c[labels.index('N2')] - c[labels.index('Th')]*(t.px.values-NX/2)*PS
    if with_scale:
        dxc -= c[labels.index('S')]*(t.px.values-NX/2)*PS
        dyc -= c[labels.index('S')]*(t.py.values-NY/2)*PS
    return c, labels, cov, dxc, dyc


tab, rx, ry, R = load('bruns_method_star_table.csv')
linked = (tab.src == 'E2-linked').values
n = len(tab)
c1, l1, cov1, dxc, dyc = solve(tab, rx, ry, R)
c2, l2, cov2, _, _ = solve(tab, rx, ry, R, with_scale=True)
L1, L2 = c1[l1.index('L')], c2[l2.index('L')]
sL1 = float(np.sqrt(cov1[l1.index('L'), l1.index('L')]))
rad = dxc*(rx/R) + dyc*(ry/R)
tanc = -dxc*(ry/R) + dyc*(rx/R)
h = 1/np.mean((R_SUN_AS/R)**2)
SCALE_ERR = h*R_SUN_AS*SCALE_PPM
tot = float(np.hypot(np.hypot(SE_STAT, SCALE_ERR), ATM_ERR))
star_rms = float(np.sqrt(np.mean((rad - L1*R_SUN_AS/R)**2 + tanc**2)))
print('N=%d  L(M1) = %.3f; M2 = %.3f; total sigma %.3f; per-star scatter %.3f arcsec'
      % (n, L1, L2, tot, star_rms))

# ---- 1. deflection vs radius (record)
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
m = ~linked
ax.scatter(R[m]/R_SUN_AS, rad[m], s=38, color='tab:blue', zorder=4,
           label='%d stars, 0.62 s master' % int(m.sum()))
ax.scatter(R[linked]/R_SUN_AS, rad[linked], s=70, marker='D', color='tab:red', zorder=5,
           label='2 close-in stars, 0.09 s master (7-star link)')
k_out = int(np.argmin(np.where((R/R_SUN_AS > 4.3) & (R/R_SUN_AS < 5.0), rad, 1e9)))
ax.annotate('  G %.2f: faint, 3 px footprint,\n  field corner (removing it: +0.014")'
            % tab.mag.values[k_out], (R[k_out]/R_SUN_AS, rad[k_out]), fontsize=7.5,
            color='gray')
xx = np.linspace(1.35, (R/R_SUN_AS).max()+0.3, 300)
ax.fill_between(xx, (L1-tot)/xx, (L1+tot)/xx, color='black', alpha=0.10,
                label='total $\\pm$%.2f" (stat %.3f + scale %.3f + atm %.3f)'
                      % (tot, SE_STAT, SCALE_ERR, ATM_ERR))
ax.plot(xx, L1/xx, color='black', lw=2.2, label='Method 1 fit:  L = %.3f"' % L1)
ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
ax.set_title('Deflection vs radius — %s' % TAG, fontsize=12)
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_deflection.png'), dpi=140)
plt.close(fig)

# ---- 2. the field: full displacement vectors, nothing clipped
ARROW = 300.0
PAD = 450
fig, ax = plt.subplots(figsize=(10.5, 8.2))
ax.add_patch(Rectangle((0, 0), NX, NY, fill=False, color='gray', lw=1.2,
                       label='sensor edge'))
for k in range(n):
    col = 'tab:red' if linked[k] else 'tab:blue'
    x0, y0 = tab.px.values[k], tab.py.values[k]
    x1, y1 = x0 + dxc[k]*ARROW, y0 + dyc[k]*ARROW
    assert -PAD < x1 < NX+PAD and -PAD < y1 < NY+PAD, 'arrow %d leaves the axes' % k
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                color=col, lw=1.5, shrinkA=0, shrinkB=0))
    ax.annotate(' %.1f' % tab.mag.values[k], (x0, y0), fontsize=6.5,
                color=('tab:red' if linked[k] else 'black'))
ax.scatter(tab.px.values[~linked], tab.py.values[~linked], s=22, color='tab:blue',
           zorder=5, label='0.62 s master (%d stars)' % int((~linked).sum()))
ax.scatter(tab.px.values[linked], tab.py.values[linked], s=70, marker='D',
           color='tab:red', zorder=5, label='0.09 s master, linked (2)')
ax.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS/PS, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
leg = ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
# both scale bars directly under the legend box, top-left
bx, by = -PAD + 120, NY - 640
ax.plot([bx, bx + ARROW], [by, by], color='black', lw=2.5)
ax.annotate('1 arcsec of displacement', (bx + ARROW + 40, by - 18), fontsize=8)
ax.plot([bx, bx + star_rms*ARROW], [by - 150, by - 150], color='gray', lw=4)
ax.annotate('per-star scatter (%.2f")' % star_rms, (bx + ARROW + 40, by - 168),
            fontsize=8, color='gray')
ax.set_xlim(-PAD, NX+PAD); ax.set_ylim(-PAD, NY+PAD); ax.set_aspect(1)
ax.set_xlabel('px'); ax.set_ylabel('py')
ax.set_title('Displacement vectors (pointing + rotation removed) — %s' % TAG, fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_field.png'), dpi=140)
plt.close(fig)

# ---- 3. L against plate scale, 1-sigma only, ppm axis
fig, ax = plt.subplots(figsize=(9.5, 7))
pc = h*R_SUN_AS*SCALE_PPM
C1 = np.array([[sL1**2 + pc**2, -pc*SCALE_PPM*1e6], [-pc*SCALE_PPM*1e6, (SCALE_PPM*1e6)**2]])
mu1 = np.array([L1, 0.0])                          # ppm offset from the imported scale
iL2, iS2 = l2.index('L'), l2.index('S')
C2 = np.array([[cov2[iL2, iL2], cov2[iL2, iS2]*1e6],
               [cov2[iS2, iL2]*1e6, cov2[iS2, iS2]*1e12]])
mu2 = np.array([L2, 1e6*c2[iS2]])


def draw(cov, mu, color, name):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), angle=ang,
                         fill=False, color=color, lw=1.6, label='1$\\sigma$ — %s' % name))
    ax.scatter(*mu, marker='+', s=110, color=color, zorder=5)


draw(C1, mu1, 'darkred', 'Method 1 (scale imported)')
draw(C2, mu2, 'tab:blue', 'Method 2 (scale free)')
ax.annotate('Method 1:  L = %.3f $\\pm$ %.3f" (stat+scale)' % (L1, np.sqrt(C1[0, 0])),
            mu1, textcoords='offset points', xytext=(12, 12), fontsize=9, color='darkred')
ax.annotate('Method 2:  L = %.3f $\\pm$ %.3f"\n scale %+.1f ppm from imported'
            % (L2, np.sqrt(C2[0, 0]), mu2[1]), mu2,
            textcoords='offset points', xytext=(12, -30), fontsize=9, color='tab:blue')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.annotate('atmosphere $\\pm$%.2f" not drawn; total error $\\pm$%.2f"' % (ATM_ERR, tot),
            (0.02, 0.02), xycoords='axes fraction', fontsize=8.5, color='gray')
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('plate scale $-$ imported 2.0868004 (ppm)', fontsize=12)
ax.set_title('L and plate scale, both methods — %s' % TAG, fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.autoscale_view()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=140)
plt.close(fig)

# ---- 4. the G <= 13 variant, outliers labelled
t13, rx13, ry13, R13 = load('bruns_method_star_table_mag13.csv')
c13, l13, _, dx13, dy13 = solve(t13, rx13, ry13, R13)
L13 = c13[l13.index('L')]
rad13 = dx13*(rx13/R13) + dy13*(ry13/R13)
resid13 = rad13 - L13*R_SUN_AS/R13
rms13 = float(np.sqrt(np.mean(resid13**2)))
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
faint = t13.mag.values > 11.0
ax.scatter(R13[~faint]/R_SUN_AS, rad13[~faint], s=38, color='tab:blue', zorder=4,
           label='G $\\leq$ 11 (%d stars)' % int((~faint).sum()))
ax.scatter(R13[faint]/R_SUN_AS, rad13[faint], s=38, marker='s', color='tab:orange',
           zorder=4, label='G 11–13 (%d stars)' % int(faint.sum()))
for k in np.where(np.abs(resid13) > 3*rms13)[0]:
    ax.annotate('  G %.2f (%+.1f$\\sigma$)' % (t13.mag.values[k], resid13[k]/rms13),
                (R13[k]/R_SUN_AS, rad13[k]), fontsize=7.5, color='crimson')
xx = np.linspace(1.35, (R13/R_SUN_AS).max()+0.3, 300)
ax.plot(xx, L13/xx, color='black', lw=2, label='fit:  L = %.3f $\\pm$ 0.088"' % L13)
ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec)', fontsize=13)
ax.set_title('The G $\\leq$ 13 variant: %d stars, stat error +37%% vs G $\\leq$ 11 — %s'
             % (len(t13), TAG), fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_deflection_g13.png'), dpi=140)
plt.close(fig)
print('G13: N=%d L=%.3f rms13=%.3f; >3sigma: %s'
      % (len(t13), L13, rms13,
         ['G %.2f (%+.1f sig)' % (t13.mag.values[k], resid13[k]/rms13)
          for k in np.where(np.abs(resid13) > 3*rms13)[0]]))
print('charts ->', OUT)
