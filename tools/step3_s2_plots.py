"""The standard stage-3 charts, produced from the S2 union fit (0.6+1.2 s, anchor in).

Reuses step3_s2_union.py's build by executing it, then draws the classic trio:
  1. the eclipse field in RA/Dec, Sun disk to scale, stars labelled;
  2. the L-vs-plate-scale covariance ellipses, Method 1 (scale imported, eq-23 term added
     by hand at the corrected units) and Method 2 (scale free, covariance from the fit);
  3. radial deflection vs radial position with the fitted L/r curve, for both methods --
     the plotted deflections are the per-star displacements with the fitted offsets, roll
     and NUISANCE removed, projected radially (i.e. what the L column actually sees).
PNGs to D:\\MEE2024 output\\MEE_output\\step3_s2_plots\\.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])          # build machinery + tier tables, skip reports

OUT = r"D:/MEE2024 output/MEE_output/step3_s2_plots"
os.makedirs(OUT, exist_ok=True)
SUN_RA, SUN_DEC = 142.107, 14.909
GR = 1.7512

U, rx, ry, R = build_union(('0p6s', '1p2s'))
n = len(U)
cra0 = np.degrees(cat.get_ra()); cdec0 = np.degrees(cat.get_dec())
ura, udec = cra0[U.cat_i.values], cdec0[U.cat_i.values]

def solve(nuis_deg, with_scale):
    xs, ys = (U.px.values-NX/2)/W_NORM, (U.py.values-NY/2)/W_NORM
    ux, uy = rx/R, ry/R
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(U.py.values-NY/2)*PS]
    cols_y = [Z, np.ones(n), (U.px.values-NX/2)*PS]
    labels = ['N1','N2','Th']
    if with_scale:
        cols_x.append((U.px.values-NX/2)*PS); cols_y.append((U.py.values-NY/2)*PS)
        labels.append('S')
    cols_x.append(ux*R_SUN_AS/R); cols_y.append(uy*R_SUN_AS/R); labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0: continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([U.dx.values, U.dy.values])
    c, res_, rank_, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    dof = len(b) - len(c)
    sig2 = float(resid @ resid) / dof
    cov = sig2 * np.linalg.inv(A.T @ A)
    # radial deflection as the L column sees it: data minus everything EXCEPT L
    iL = labels.index('L')
    other = A@c - np.outer(A[:, iL], [1]).ravel()*c[iL]
    clean = b - other
    ddx, ddy = clean[:n], clean[n:]
    defl_rad = ddx*(rx/R) + ddy*(ry/R)
    return c, cov, labels, defl_rad

cM1, covM1, labM1, radM1 = solve(2, False)
cM2, covM2, labM2, radM2 = solve(2, True)
L1, L2 = cM1[labM1.index('L')], cM2[labM2.index('L')]
sL1 = np.sqrt(covM1[labM1.index('L'), labM1.index('L')])
iL2, iS2 = labM2.index('L'), labM2.index('S')
sL2 = np.sqrt(covM2[iL2, iL2])

# Method 1 covariance with the imported scale: eq 23 at the CORRECTED units
dS = 25e-6
h_fl = 1/np.mean((R_SUN_AS/R)**2)
pc2 = h_fl * R_SUN_AS * dS                       # arcsec of L per the full dS
C1 = np.array([[sL1**2 + pc2**2, -pc2*dS*PS], [-pc2*dS*PS, (dS*PS)**2]])
mu1 = np.array([L1, PS])
# Method 2: convert the S coefficient (relative) to plate scale
C2 = np.array([[covM2[iL2, iL2], covM2[iL2, iS2]*PS], [covM2[iS2, iL2]*PS, covM2[iS2, iS2]*PS**2]])
mu2 = np.array([L2, PS*(1 + cM2[iS2])])

print(f"union 0.6+1.2 with anchor, N={n}, h={h_fl:.1f} Rsun^2")
print(f"Method 1: L = {L1:.3f} +- {np.sqrt(C1[0,0]):.3f} arcsec  (fit {sL1:.3f} + eq23 {pc2:.3f})")
print(f"Method 2: L = {L2:.3f} +- {sL2:.3f} arcsec; scale {mu2[1]:.7f} ({1e6*(mu2[1]/PS-1):+.1f} ppm vs CAL)")

# ---- 1. the field
fig, axf = plt.subplots(figsize=(9.5, 7.5))
axf.scatter(ura, udec, s=28, color='tab:blue', label='catalog', zorder=3)
axf.scatter(ura, udec, marker='+', s=46, color='orange', label='observation (used)', zorder=4)
for i in range(n):
    if U.mag.values[i] <= 8.6:
        axf.annotate(f" mag={U.mag.values[i]:.1f}", (ura[i], udec[i]), fontsize=7)
axf.add_patch(Circle((SUN_RA, SUN_DEC), R_SUN_AS/3600, color='black', zorder=5))
axf.set_xlabel('RA (degrees)', fontsize=13); axf.set_ylabel('DEC (degrees)', fontsize=13)
axf.set_title(f'Leon 2026 eclipse field: S2 union (0.6+1.2 s), {n} stars, '
              f'V {U.mag.min():.1f} to {U.mag.max():.1f}')
axf.legend(); axf.set_aspect(1/np.cos(np.radians(SUN_DEC)))
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'field_radec.png'), dpi=140); plt.close(fig)

# ---- 2. covariance ellipses
def draw(ax, cov, mu, color, label):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for k, ls in ((1, '-'), (2, ':')):
        ax.add_patch(Ellipse(mu, 2*k*np.sqrt(vals[1]), 2*k*np.sqrt(vals[0]), angle=ang,
                             fill=False, color=color, linestyle=ls,
                             label=f'{k}$\\sigma$ ({label})'))
fig, axc = plt.subplots(figsize=(9, 7))
draw(axc, C1, mu1, 'darkred', 'method 1')
draw(axc, C2, mu2, 'magenta', 'method 2')
axc.scatter(*mu1, marker='+', s=90, color='darkred')
axc.scatter(*mu2, marker='+', s=90, color='magenta')
axc.annotate(f"M1: {mu1[0]:.3f}$\\pm${np.sqrt(C1[0,0]):.3f}, {mu1[1]:.6f}", mu1,
             textcoords='offset points', xytext=(10, 10), fontsize=10)
axc.annotate(f"M2: {mu2[0]:.3f}$\\pm${sL2:.3f}, {mu2[1]:.6f}", mu2,
             textcoords='offset points', xytext=(10, -16), fontsize=10)
axc.axvline(GR, color='green', lw=1, alpha=0.6)
axc.annotate('GR 1.751', (GR, axc.get_ylim()[0]), color='green', fontsize=9,
             textcoords='offset points', xytext=(3, 12))
axc.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
axc.set_ylabel('Plate Scale (arcsec / pixel)', fontsize=13)
axc.set_title('Covariance of Deflection Constant and Plate Scale — S2 union, nuisance on')
axc.legend(loc='lower left', fontsize=9)
axc.autoscale_view()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'covariance.png'), dpi=140); plt.close(fig)

# ---- 3. deflection vs radius, both methods
for tag, Lv, radial in (('method 1', L1, radM1), ('method 2', L2, radM2)):
    fig, axd = plt.subplots(figsize=(9.5, 6.5))
    axd.axhline(0, color='black', lw=1)
    axd.scatter(R/R_SUN_AS, radial, s=26)
    xx = np.linspace((R/R_SUN_AS).min()-0.3, (R/R_SUN_AS).max()+0.3, 100)
    axd.plot(xx, Lv/xx, color='black')
    axd.plot(xx, GR/xx, color='green', lw=1, alpha=0.7)
    axd.annotate(f'fit L = {Lv:.3f} arcsec', (xx[8], Lv/xx[8]),
                 textcoords='offset points', xytext=(6, 8), fontsize=11)
    axd.annotate('GR', (xx[-25], GR/xx[-25]), color='green', fontsize=10,
                 textcoords='offset points', xytext=(4, 4))
    axd.set_xlabel('radial position (solar radii)', fontsize=13)
    axd.set_ylabel('radial deflection (arcsec)', fontsize=13)
    axd.set_title(f'Deflections, S2 union (0.6+1.2 s, anchor in, nuisance removed) — {tag}')
    fname = f"deflection_{tag.replace(' ', '')}.png"
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=140); plt.close(fig)
print('plots written to', OUT, flush=True)
