"""The summary chart set for cell 1's reduction of record (L = 1.777).

Eighth revision (Douglas' fifth chart review, 2026-09-01). This round: the record
deflection chart moves to G <= 10.5; the covariance, field and G13 charts move to the
14-star link; the master images get their legends below the frame; and the layout notes
below. Earlier revisions are archived under chart_versions/revNN_*.

Seventh revision items retained:

  * every produced chart is ALSO archived under chart_versions/rev07_* -- and older
    revisions are regenerated from git history into the same folder, because deleting
    superseded versions turned out to destroy useful context. Nothing gets overwritten
    into oblivion again;
  * deflection charts name their link in the title (the G13 variant is a 7-star link,
    which was not stated and read as if it were the 14); outliers are annotated from
    2.5 sigma so the G 10.64 note is back;
  * the field chart: sensor footprint drawn as a polygon in RA/Dec (the axes span the
    star positions, not the sensor -- the footprint makes the sensor's true extent
    visible); legend and both bars moved outside the plot; arrows asserted inside the
    axes so the clipping bug cannot return; the subtitle spells out what was removed:
    "each arrow = the star's measured shift after subtracting the camera's pointing
    offset and rotation; deflection + that star's measurement noise remain";
  * the covariance annotations are pinned in axes-fraction coordinates, so they can
    neither leave the box nor collide with the ellipses;
  * the two master images are drawn from the RAW frame means with the masked-blur
    coronal subtraction and NO painted disk (natural view), y increasing downward as
    the project has always displayed frames, legends outside, and: master009 shows the
    close-in pair and the 14 shared stars; master062 shows every identified star of
    the reduction and none of the spurious detections.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch

OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
VER = os.path.join(OUT, 'chart_versions')
os.makedirs(VER, exist_ok=True)
REV = 'rev09'
RAWDIR = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
PS, NX, NY = 2.0868004, 3296, 2472
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
GR, NEWTON = 1.7512, 0.8756
SCALE_PPM, ATM_ERR = 10.3e-6, 0.150


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140)
    fig.savefig(os.path.join(VER, REV + '_' + name), dpi=140)
    plt.close(fig)


# ---- shared machinery
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
Minv = np.linalg.inv(np.array([[ax_[0], ax_[1]], [ay_[0], ay_[1]]]))


def px_to_sky(px, py):
    v = Minv @ np.vstack([np.asarray(px, float) - ax_[2], np.asarray(py, float) - ay_[2]])
    return ra0 + v[0]/np.cos(np.radians(de0)), de0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    v = Minv @ np.vstack([dx_as/PS, dy_as/PS])
    return v[0]*3600*PS, v[1]*3600*PS


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
    for k in np.where(np.abs(resid) > 2.5*rms)[0]:
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
    fig.tight_layout()
    if fname.startswith('_scratch'):
        plt.close(fig)
    else:
        save(fig, fname)
    print('%s: N=%d L=%.3f tot=%.3f' % (fname, len(t), L, tot))
    return t, rx_, ry_, R_, c, l, cov, dxc, dyc, L, tot


deflection_chart('bruns_method_star_table_mag10.5_link14.csv', 'record_deflection.png',
                 'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 10.5, 14-star link', 0.062)
deflection_chart('bruns_method_star_table_link14.csv', 'record_deflection_link14.png',
                 'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 11, 14-star link', 0.060)
deflection_chart('bruns_method_star_table_mag13_link14.csv', 'record_deflection_g13.png',
                 'Deflection vs radius \u2014 Bruns 2017, G $\\leq$ 13, 14-star link', 0.086)
# the covariance, field and master charts below all run on the 14-star link
rec = deflection_chart('bruns_method_star_table_link14.csv', '_scratch.png', 'scratch',
                       0.060)

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

# ---- the field, RA/Dec, sensor footprint drawn, nothing clipped, furniture outside
fig, ax = plt.subplots(figsize=(11.5, 8))
sra, sdec = px_to_sky(tab.px.values, tab.py.values)
vra, vdec = sensor_vec_to_sky(dxc, dyc)
ARROW_DEG = 0.17
corners = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
ax.add_patch(Polygon(np.c_[corners[0], corners[1]], fill=False, color='gray', lw=1.2,
                     label='sensor footprint'))
ends_ra, ends_de = [], []
for k in range(n):
    col = 'tab:red' if linked[k] else 'tab:blue'
    x0, y0 = sra[k], sdec[k]
    x1 = x0 + vra[k]*ARROW_DEG/np.cos(np.radians(de0))
    y1 = y0 + vdec[k]*ARROW_DEG
    ends_ra.append(x1); ends_de.append(y1)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                color=col, lw=1.5, shrinkA=0, shrinkB=0))
    # the linked pair's markers are large diamonds, so their labels need clearing
    dx_lab = 0.030 if linked[k] else 0.008
    ax.annotate('%.1f' % tab.mag.values[k], (x0 + dx_lab, y0), fontsize=6.5,
                color=('tab:red' if linked[k] else 'black'))
ax.scatter(sra[~linked], sdec[~linked], s=22, color='tab:blue', zorder=5,
           label='0.62 s master (%d stars)' % int((~linked).sum()))
ax.scatter(sra[linked], sdec[linked], s=70, marker='D', color='tab:red', zorder=5,
           label='0.09 s master, linked (2)')
sun_ra, sun_dec = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
ax.add_patch(Circle((float(sun_ra[0]), float(sun_dec[0])), R_SUN_AS/3600, color='black',
                    zorder=3, label='the Sun, 1 R$_\\odot$ to scale'))
lo_ra = min(min(corners[0]), min(ends_ra)) - 0.06
hi_ra = max(max(corners[0]), max(ends_ra)) + 0.06
lo_de = min(min(corners[1]), min(ends_de)) - 0.05
hi_de = max(max(corners[1]), max(ends_de)) + 0.05
for x1, y1 in zip(ends_ra, ends_de):
    assert lo_ra < x1 < hi_ra and lo_de < y1 < hi_de, 'an arrow leaves the axes'
ax.set_xlim(lo_ra, hi_ra); ax.set_ylim(lo_de, hi_de)
ax.set_aspect(1/np.cos(np.radians(de0)))
ax.set_xlabel('RA (degrees)', fontsize=12)
ax.set_ylabel('DEC (degrees)', fontsize=12)
ax.set_title('Displacement vectors \u2014 Bruns 2017, G $\\leq$ 11, 14-star link', fontsize=12)
fig.text(0.06, 0.020, 'each arrow = the star\u2019s measured shift after subtracting the '
         'camera\u2019s pointing offset and rotation; deflection + measurement noise remain',
         fontsize=9)
ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.75))
bar_deg = ARROW_DEG/np.cos(np.radians(de0))
from matplotlib.patches import FancyBboxPatch
for y_fr, ln, txt in ((0.44, 1.0, '1 arcsec of displacement'),
                      (0.34, star_rms, 'per-star scatter (%.2f")' % star_rms)):
    xa, ya = 1.04, y_fr
    ax.annotate('', xy=(xa + ln*bar_deg/(hi_ra-lo_ra), ya), xytext=(xa, ya),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color='black', lw=3))
    ax.annotate(txt, (xa, ya + 0.028), xycoords='axes fraction', fontsize=8)
ax.add_patch(FancyBboxPatch((1.02, 0.29), 0.30, 0.22, boxstyle='round,pad=0.012',
                            transform=ax.transAxes, fill=False, color='gray', lw=0.9,
                            clip_on=False))
fig.subplots_adjust(right=0.74, bottom=0.13)
save(fig, 'record_field.png')

# ---- L and plate scale, annotations pinned in axes coordinates
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
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.add_patch(FancyBboxPatch((0.020, 0.030), 0.545, 0.165, boxstyle='round,pad=0.010',
                            transform=ax.transAxes, fill=True, facecolor='white',
                            edgecolor='gray', lw=0.9, zorder=6))
ax.text(0.038, 0.150, 'Method 1:  L = %.3f $\\pm$ %.3f" (stat+scale)'
        % (L1, np.sqrt(C1[0, 0])), transform=ax.transAxes, fontsize=9.5,
        color='darkred', zorder=7)
ax.text(0.038, 0.100, 'Method 2:  L = %.3f $\\pm$ %.3f",  scale %+.1f ppm'
        % (L2, np.sqrt(C2[0, 0]), mu2[1]), transform=ax.transAxes, fontsize=9.5,
        color='tab:blue', zorder=7)
ax.text(0.038, 0.050, 'Imported plate scale: %.7f "/px' % PS, transform=ax.transAxes,
        fontsize=9.5, color='black', zorder=7)
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('Plate scale (ppm difference from imported value)', fontsize=12)
ax.set_title('L and plate scale \u2014 Bruns 2017, G $\\leq$ 11, 14-star link', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.autoscale_view()
ax.margins(0.15)
save(fig, 'record_covariance.png')

# ---- the two master images: natural backdrop, no painted disk, y down, legend outside
from astropy.io import fits as pyfits
from scipy.ndimage import gaussian_filter


def natural_master(patterns, cache):
    """Unshifted mean of the RAW frames, masked-blur coronal subtraction, no painting."""
    cpath = os.path.join(OUT, cache)
    if os.path.exists(cpath):
        return np.load(cpath)
    files = []
    for p in patterns:
        files += sorted(glob.glob(os.path.join(RAWDIR, p)))
    acc = None
    for f in files:
        d = pyfits.getdata(f).astype(np.float64)
        acc = d if acc is None else acc + d
    mean = acc/len(files)
    valid = mean < 65535
    num = gaussian_filter(np.where(valid, mean, 0.0), 10.0)
    den = gaussian_filter(valid.astype(np.float64), 10.0)
    model = np.where(den > 0.05, num/np.maximum(den, 1e-9), 65535.0)
    img = mean - model
    np.save(cpath, img.astype(np.float32))
    return img


def master_figure(img, circles, fname, title):
    lo, hi = np.percentile(img, [5, 99.5])
    disp = np.arcsinh((np.clip(img, lo, hi) - lo)/max(hi - lo, 1)*30)
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
    handles = []
    for xs, ys, colr, size, lab in circles:
        for x0, y0 in zip(xs, ys):
            ax.add_patch(Circle((x0, y0), size, fill=False, color=colr, lw=1.4))
        handles.append(plt.Line2D([], [], color=colr, label=lab))
    ax.legend(handles=handles, fontsize=9, loc='upper left', bbox_to_anchor=(0.0, -0.09),
              borderaxespad=0, frameon=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('px'); ax.set_ylabel('py')
    fig.subplots_adjust(bottom=0.20)
    save(fig, fname)


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
INNER = []
for x0, y0 in ((1179.0, 2314.0), (2102.0, 1241.0)):
    dd = np.hypot(detE2['px'].values - x0, detE2['py'].values - y0)
    k = int(np.argmin(dd))
    INNER.append((detE2['px'].values[k], detE2['py'].values[k]))

img009 = natural_master(['E2_*.fit'], 'master009_natural.npy')
master_figure(img009,
              [([p[0] for p in common], [p[1] for p in common], 'yellow', 28,
                'the 14 stars shared with the 0.62 s master'),
               ([p[0] for p in INNER], [p[1] for p in INNER], 'red', 34,
                'the two close-in stars (G 7.09, G 7.52)')],
              'master009_annotated.png',
              'The 0.09 s master (E2, 11 frames) \u2014 Bruns 2017')

img062 = natural_master(['EA_*.fit', 'EB_*.fit'], 'master062_natural.npy')
used = tab[~linked]
master_figure(img062,
              [(used.px.values, used.py.values, 'yellow', 28,
                'the %d identified stars used in the reduction (G \u2264 11)' % len(used))],
              'master062_annotated.png',
              'The 0.62 s master (EA+EB, 34 frames) \u2014 Bruns 2017')
print('charts ->', OUT)
