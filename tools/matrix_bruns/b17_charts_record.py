"""The summary chart set for cell 1's reduction of record (L = 1.777).

Sixth revision, from Douglas' third chart review (2026-09-01). This round:

  * titles lose "Bruns' method" everywhere; the magnitude cut (G <= 11) is stated
    instead, and the G <= 13 variant is styled identically to the main chart (Method-1
    label, Einstein line, shaded full-error band) so the two are comparable at a glance;
  * a second deflection chart for the 14-star link variant;
  * the covariance chart: y-axis "Plate scale (ppm difference from imported value)",
    the hidden atmosphere footnote deleted, and both method annotations moved clear of
    the ellipses;
  * the field chart is drawn in RA/DEC (converted through the master's own matched-table
    affine) rather than sensor pixels; the two bars (1 arcsec, per-star scatter, same
    colour) live in a box in the top-right; and the cryptic "(pointing + rotation
    removed)" is spelled out: each arrow is the measured displacement of one star minus
    the fitted pointing offset and field rotation, so deflection plus that star's
    measurement noise is what remains;
  * a labelled image of the 0.09 s master: the program's own CentroidsStackGood for E2
    shows no identified stars because E2's own plate solve fails (by design -- it rides
    the long master), so the stars are drawn here from the reduction instead: the two
    close-in stars and the 14 link stars it shares with the 0.62 s master.
"""
import glob, json, os, zipfile
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
SCALE_PPM, ATM_ERR = 10.3e-6, 0.150

# the master's sky affine, for RA/DEC axes and the Sun position
zh = zipfile.ZipFile(glob.glob(os.path.join(OUT, 'master062', 'stage2',
                                            'distortion_data*.zip'))[0])
dh = pd.read_csv(zh.open([m for m in zh.namelist()
                          if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
ra0, de0 = dh['RA(catalog)'].mean(), dh['DEC(catalog)'].mean()
Xa = (dh['RA(catalog)'].values-ra0)*np.cos(np.radians(de0))
Ya = dh['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax_, *_ = np.linalg.lstsq(Aa, dh['px'].values, rcond=None)
ay_, *_ = np.linalg.lstsq(Aa, dh['py'].values, rcond=None)
M = np.array([[ax_[0], ax_[1]], [ay_[0], ay_[1]]])       # sky(deg) -> px
Minv = np.linalg.inv(M)                                   # px -> sky(deg)


def px_to_sky(px, py):
    v = Minv @ np.vstack([px - ax_[2], py - ay_[2]])
    return ra0 + v[0]/np.cos(np.radians(de0)), de0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    """A displacement in sensor-axis arcsec -> (dRA*cos, dDec) arcsec."""
    v = Minv @ np.vstack([dx_as/PS, dy_as/PS])            # -> degrees-of-affine input
    return v[0]*3600*PS, v[1]*3600*PS                     # scale-preserving rotation


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


def deflection_chart(table_csv, fname, title, se_stat):
    t, rx_, ry_, R_ = load(table_csv)
    linked = (t.src == 'E2-linked').values
    c, l, cov, dxc, dyc = solve(t, rx_, ry_, R_)
    L = c[l.index('L')]
    rad = dxc*(rx_/R_) + dyc*(ry_/R_)
    h = 1/np.mean((R_SUN_AS/R_)**2)
    scale_err = h*R_SUN_AS*SCALE_PPM
    tot = float(np.hypot(np.hypot(se_stat, scale_err), ATM_ERR))
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axhline(0, color='black', lw=1)
    faint = t.mag.values > 11.0
    ax.scatter(R_[~linked & ~faint]/R_SUN_AS, rad[~linked & ~faint], s=38,
               color='tab:blue', zorder=4,
               label='0.62 s master (%d stars)' % int((~linked & ~faint).sum()))
    if faint.any():
        ax.scatter(R_[faint]/R_SUN_AS, rad[faint], s=38, marker='s', color='tab:orange',
                   zorder=4, label='G 11\u201313 (%d stars)' % int(faint.sum()))
    ax.scatter(R_[linked]/R_SUN_AS, rad[linked], s=70, marker='D', color='tab:red',
               zorder=5, label='close-in pair, 0.09 s master')
    resid = rad - L*R_SUN_AS/R_
    rms = float(np.sqrt(np.mean(resid**2)))
    for k in np.where(np.abs(resid) > 3*rms)[0]:
        ax.annotate('  G %.2f (%+.1f$\\sigma$)' % (t.mag.values[k], resid[k]/rms),
                    (R_[k]/R_SUN_AS, rad[k]), fontsize=7.5, color='crimson')
    xx = np.linspace(1.35, (R_/R_SUN_AS).max()+0.3, 300)
    ax.fill_between(xx, (L-tot)/xx, (L+tot)/xx, color='black', alpha=0.10,
                    label='total $\\pm$%.2f" (stat %.3f + scale %.3f + atm %.3f)'
                          % (tot, se_stat, scale_err, ATM_ERR))
    ax.plot(xx, L/xx, color='black', lw=2.2, label='Method 1 fit:  L = %.3f"' % L)
    ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
    ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
    ax.set_xlabel('radial position (solar radii)', fontsize=13)
    ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8.5, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)
    print('%s: N=%d L=%.3f tot=%.3f' % (fname, len(t), L, tot))
    return t, rx_, ry_, R_, c, l, cov, dxc, dyc, L, tot


# ---- 1. deflection, record (7-star link) + the 14-star-link variant + G13
rec = deflection_chart('bruns_method_star_table.csv', 'record_deflection.png',
                       'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 11', 0.064)
deflection_chart('bruns_method_star_table_link14.csv', 'record_deflection_link14.png',
                 'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 11, 14-star link', 0.060)
deflection_chart('bruns_method_star_table_mag13.csv', 'record_deflection_g13.png',
                 'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 13 variant', 0.088)

tab, rx, ry, R, c1, l1, cov1, dxc, dyc, L1, tot = rec
linked = (tab.src == 'E2-linked').values
n = len(tab)
sL1 = float(np.sqrt(cov1[l1.index('L'), l1.index('L')]))
c2, l2, cov2, _, _ = solve(tab, rx, ry, R, with_scale=True)
L2 = c2[l2.index('L')]
h = 1/np.mean((R_SUN_AS/R)**2)
rad = dxc*(rx/R) + dyc*(ry/R)
tanc = -dxc*(ry/R) + dyc*(rx/R)
star_rms = float(np.sqrt(np.mean((rad - L1*R_SUN_AS/R)**2 + tanc**2)))

# ---- 2. the field, in RA/DEC
fig, ax = plt.subplots(figsize=(10.5, 8.2))
sra, sdec = px_to_sky(tab.px.values, tab.py.values)
vra, vdec = sensor_vec_to_sky(dxc, dyc)                    # arcsec on the sky
ARROW_DEG = 0.17                                           # degrees of arrow per arcsec
for k in range(n):
    col = 'tab:red' if linked[k] else 'tab:blue'
    x0, y0 = sra[k], sdec[k]
    x1 = x0 + vra[k]*ARROW_DEG/np.cos(np.radians(de0))
    y1 = y0 + vdec[k]*ARROW_DEG
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                color=col, lw=1.5, shrinkA=0, shrinkB=0))
    ax.annotate(' %.1f' % tab.mag.values[k], (x0, y0), fontsize=6.5,
                color=('tab:red' if linked[k] else 'black'))
ax.scatter(sra[~linked], sdec[~linked], s=22, color='tab:blue', zorder=5,
           label='0.62 s master (%d stars)' % int((~linked).sum()))
ax.scatter(sra[linked], sdec[linked], s=70, marker='D', color='tab:red', zorder=5,
           label='0.09 s master, linked (2)')
sun_ra, sun_dec = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
ax.add_patch(Circle((sun_ra[0], sun_dec[0]), R_SUN_AS/3600/np.cos(np.radians(de0))*0 +
                    R_SUN_AS/3600, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
# the two bars, boxed, top right, one colour
cx0, cy0 = sra.max()+0.28, sdec.max()+0.10
bar_as = ARROW_DEG/np.cos(np.radians(de0))
ax.plot([cx0, cx0 + 1.0*bar_as], [cy0, cy0], color='black', lw=2.5)
ax.annotate('1 arcsec of displacement', (cx0, cy0 + 0.030), fontsize=8)
ax.plot([cx0, cx0 + star_rms*bar_as], [cy0 - 0.075, cy0 - 0.075], color='black', lw=4)
ax.annotate('per-star scatter (%.2f")' % star_rms, (cx0, cy0 - 0.045), fontsize=8)
from matplotlib.patches import FancyBboxPatch
ax.add_patch(FancyBboxPatch((cx0 - 0.03, cy0 - 0.12), 1.0*bar_as + 0.11, 0.20,
                            boxstyle='round,pad=0.01', fill=False, color='gray', lw=0.8))
pad_ra, pad_de = 0.30, 0.22
ax.set_xlim(sra.min()-pad_ra, sra.max()+pad_ra+0.45)
ax.set_ylim(sdec.min()-pad_de, sdec.max()+pad_de)
ax.set_aspect(1/np.cos(np.radians(de0)))
ax.set_xlabel('RA (degrees)', fontsize=12)
ax.set_ylabel('DEC (degrees)', fontsize=12)
ax.set_title('Displacement vectors \u2014 Bruns 2017, G $\\leq$ 11 '
             '(fitted pointing offset and field rotation subtracted)', fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_field.png'), dpi=140)
plt.close(fig)

# ---- 3. L and plate scale
fig, ax = plt.subplots(figsize=(9.5, 7))
pc = h*R_SUN_AS*SCALE_PPM
C1 = np.array([[sL1**2 + pc**2, -pc*SCALE_PPM*1e6], [-pc*SCALE_PPM*1e6, (SCALE_PPM*1e6)**2]])
mu1 = np.array([L1, 0.0])
iL2, iS2 = l2.index('L'), l2.index('S')
C2 = np.array([[cov2[iL2, iL2], cov2[iL2, iS2]*1e6],
               [cov2[iS2, iL2]*1e6, cov2[iS2, iS2]*1e12]])
mu2 = np.array([L2, 1e6*c2[iS2]])


def draw(cov, mu, color, name):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), angle=ang,
                         fill=False, color=color, lw=1.6, label='1$\\sigma$ \u2014 %s' % name))
    ax.scatter(*mu, marker='+', s=110, color=color, zorder=5)


draw(C1, mu1, 'darkred', 'Method 1 (scale imported)')
draw(C2, mu2, 'tab:blue', 'Method 2 (scale free)')
ax.annotate('Method 1:  L = %.3f $\\pm$ %.3f" (stat+scale)' % (L1, np.sqrt(C1[0, 0])),
            (L1 + 1.35*np.sqrt(C1[0, 0]), mu1[1] + 2.0), fontsize=9, color='darkred')
ax.annotate('Method 2:  L = %.3f $\\pm$ %.3f"\n scale %+.1f ppm from imported'
            % (L2, np.sqrt(C2[0, 0]), mu2[1]),
            (L2 - 1.6*np.sqrt(C2[0, 0]), mu2[1] - 2.6*np.sqrt(C2[1, 1])), fontsize=9,
            color='tab:blue', ha='left', va='top')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('Plate scale (ppm difference from imported value)', fontsize=12)
ax.set_title('L and plate scale \u2014 Bruns 2017, G $\\leq$ 11', fontsize=12)
ax.legend(fontsize=9, loc='lower left')
ax.autoscale_view()
ax.margins(0.18)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=140)
plt.close(fig)

# ---- 4. the 0.09 s master, annotated from the reduction
sfits = sorted(glob.glob(os.path.join(OUT, 'master009', '**', 'STACKED2*.fit'),
                         recursive=True))
if sfits:
    from astropy.io import fits as pyfits
    img = pyfits.getdata(sfits[-1]).astype(np.float64)
    det62 = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(
        OUT, 'master062', 'centroid_data*.zip'))[0]).open('STACKED_CENTROIDS_DATA.csv'))
    detE2 = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(
        OUT, 'master009', 'centroid_data*.zip'))[0]).open('STACKED_CENTROIDS_DATA.csv'))
    detE2 = detE2.sort_values('flux (noise-normed)', ascending=False).reset_index(drop=True)
    common = []
    for _, r in detE2.iterrows():
        dd = np.hypot(det62['px'].values - r.px, det62['py'].values - r.py)
        if dd.min() < 25.0:
            common.append((r.px, r.py))
        if len(common) == 14:
            break
    INNER = ((1179.0, 2314.0), (2102.0, 1241.0))
    lo, hi = np.percentile(img, [5, 99.5])
    disp = np.arcsinh((np.clip(img, lo, hi) - lo)/max(hi - lo, 1)*30)
    fig, ax = plt.subplots(figsize=(11, 8.3))
    ax.imshow(disp, cmap='gray', origin='lower', interpolation='nearest')
    for x0, y0 in common:
        ax.add_patch(Circle((x0, y0), 28, fill=False, color='yellow', lw=1.3))
    for x0, y0 in INNER:
        dd = np.hypot(detE2['px'].values - x0, detE2['py'].values - y0)
        k = int(np.argmin(dd))
        ax.add_patch(Circle((detE2['px'].values[k], detE2['py'].values[k]), 34,
                            fill=False, color='red', lw=1.8))
    ax.plot([], [], color='yellow', label='the 14 stars shared with the 0.62 s master '
                                          '(7 brightest set the link)')
    ax.plot([], [], color='red', label='the two close-in stars (G 7.09, G 7.52)')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.set_title('The 0.09 s master (E2, 11 frames): its plate solve fails by design '
                 '\u2014 it rides the 0.62 s master through the link', fontsize=10.5)
    ax.set_xlabel('px'); ax.set_ylabel('py')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'master009_annotated.png'), dpi=140)
    plt.close(fig)
    print('master009_annotated.png written (%d common stars drawn)' % len(common))
print('charts ->', OUT)
