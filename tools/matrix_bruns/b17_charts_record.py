"""The summary chart set for cell 1's reduction of record (L = 1.777).

Twelfth revision (2026-09-02): the scale term moves to the bracket HC3 of the pair the
reduction of record uses, 9.23 ppm rather than the windowed pair's 10.3 -- see SCALE_PPM.

Eleventh revision (2026-09-02): the atmospheric term becomes the BRACKETED null, 0.059
rather than 0.150 -- see the ATM_ERR comment below -- so every band and total on these
charts narrows. No fitted value changes. The covariance box is also packed around its own
text and now quotes the total including the atmosphere.

Tenth revision (2026-09-01, while building the Leon copy): the field chart's arrows
were drawn 2.087x too long relative to their own scale bar -- sensor_vec_to_sky
multiplied arcseconds by the plate scale. Fixed and asserted at runtime; nothing
else changes, and no number changes (the arrows are drawing, not fitting).

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
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.record_charts import (ChartWriter, SkyFrame, arcsinh_stretch, bar_frame,
                                 covariance_chart, field_chart as draw_field, reference_curves,
                                 scale_bars)

OUT = r"F:/MEE_output/bruns2017/matrix_bruns2017_brunsmethod"
VER = os.path.join(OUT, 'chart_versions')
RECORD = r"F:/MEE_output/RECORD/bruns2017"
os.makedirs(VER, exist_ok=True)
REV = 'rev12'
RAWDIR = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
PS, NX, NY = 2.0868004, 3296, 2472
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
GR, NEWTON = 1.7512, 0.8756
# The bracket's HC3, computed on the pair the reduction of record actually uses --
# Gaussian background + footprint moments, 117 and 119 stars, HC3 12.64 and 13.46 ppm,
# mean/sqrt(2) = 9.23. The published 10.3 came from the WINDOWED pair (105/110 stars,
# HC3 14.43/14.70), which is a different convention from the science fit it was being
# applied to; carrying it here mixed the two. tools/matrix_bruns/b17_scale_error_audit.py
# reproduces the published windowed figures exactly with the same estimator, so this is
# a change of input, not of method.
SCALE_PPM = 9.23e-6
# The atmospheric term is the null measured with the construction Bruns' eclipse fit
# actually used: the eclipse pointing against the MEAN of the RIGHT and LEFT pointings
# either side of it, which his night rehearsal repeats on a two-minute R-E-L cadence
# (tools/matrix_bruns/b17_lr_bracket_null.py, 8 triplets). The one-sided nulls of
# b17_atmosphere2.py give 0.150, and that is what these charts carried through revision
# 10; but a one-sided figure charges this design for the drift its bracket cancels. The
# one-sided means against R alone and L alone are -0.039 and +0.036 arcsec -- equal and
# opposite, a gradient across the sky -- and the bracketed mean is -0.001.
ATM_ERR = 0.059


_writer = ChartWriter(OUT, REV, ver=VER)
save = _writer.save


# ---- shared machinery
zh = zipfile.ZipFile(glob.glob(os.path.join(OUT, 'master062', 'stage2',
                                            'distortion_data*.zip'))[0])
dh = pd.read_csv(zh.open([m for m in zh.namelist()
                          if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
# the sky frame: the shared construction (tools/record_charts.py), fitted on the matched
# stars of the 0.62 s master; a unit sensor displacement is asserted to round-trip there
SF = SkyFrame.from_stars(dh['RA(catalog)'].values, dh['DEC(catalog)'].values,
                         dh['px'].values, dh['py'].values, PS)
px_to_sky, sensor_vec_to_sky, de0 = SF.px_to_sky, SF.sensor_vec_to_sky, SF.de0


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
    reference_curves(ax, xx, L, tot, 'Method 1 fit:  L = %.3f"' % L,
                     'total $\\pm$%.2f" (stat %.3f + scale %.3f + atm %.3f)'
                     % (tot, se_stat, scale_err, ATM_ERR))
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
sL1_analytic = float(np.sqrt(cov1[l1.index('L'), l1.index('L')]))
# The record quotes the 300-sample bootstrap (0.060) for this variant, not the analytic
# fit sigma (0.0645). Revision 10 built the covariance ellipse from the analytic value
# while the deflection band used the bootstrap, so the two charts of one record set
# disagreed by 0.002 on the total. The bootstrap is what the record carries, so it is
# what both use; the analytic value is kept and printed for comparison.
sL1 = 0.060
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
sun_ra, sun_dec = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
# the linked pair's markers are large diamonds, so their labels need clearing
lo_ra, hi_ra, _, _ = draw_field(
    ax, sra, sdec, vra, vdec, SF.corners(NX, NY),
    (float(sun_ra[0]), float(sun_dec[0]), R_SUN_AS/3600), ARROW_DEG, SF.cos0,
    groups=[(~linked, dict(s=22, color='tab:blue',
                           label='0.62 s master (%d stars)' % int((~linked).sum()))),
            (linked, dict(s=70, marker='D', color='tab:red', label='0.09 s master, linked (2)'))],
    arrow_color=['tab:red' if l else 'tab:blue' for l in linked], arrow_lw=1.5,
    point_labels=(['%.1f' % m for m in tab.mag.values], np.where(linked, 0.030, 0.008),
                  ['tab:red' if l else 'black' for l in linked], 6.5),
    pad=(0.06, 0.05), title='Displacement vectors \u2014 Bruns 2017, G $\\leq$ 11, 14-star link')
fig.text(0.06, 0.020, 'each arrow = the star\u2019s measured shift after subtracting the '
         'camera\u2019s pointing offset and rotation; deflection + measurement noise remain',
         fontsize=9)
ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.75))
scale_bars(ax, ((0.44, 1.0, '1 arcsec of displacement'),
                (0.34, star_rms, 'per-star scatter (%.2f")' % star_rms)),
           ARROW_DEG, SF.cos0, hi_ra - lo_ra)
bar_frame(ax, (1.02, 0.29, 0.30, 0.22))
fig.subplots_adjust(right=0.74, bottom=0.13)
save(fig, 'record_field.png')

# ---- L and plate scale, annotations pinned in axes coordinates
pc = h*R_SUN_AS*SCALE_PPM
C1 = np.array([[sL1**2 + pc**2, -pc*SCALE_PPM*1e6], [-pc*SCALE_PPM*1e6, (SCALE_PPM*1e6)**2]])
mu1 = np.array([L1, 0.0])
iL2, iS2 = l2.index('L'), l2.index('S')
C2 = np.array([[cov2[iL2, iL2], cov2[iL2, iS2]*1e6],
               [cov2[iS2, iL2]*1e6, cov2[iS2, iS2]*1e12]])
mu2 = np.array([L2, 1e6*c2[iS2]])
# the box is packed around its own text rather than hand-sized, as the Leon chart's is
_tot1 = float(np.hypot(np.sqrt(C1[0, 0]), ATM_ERR))
_lines = [('Method 1:  L = %.3f $\\pm$ %.3f" (stat + scale)' % (L1, np.sqrt(C1[0, 0])), 'darkred'),
          ('      $\\pm$ %.3f" with the atmosphere term %.3f' % (_tot1, ATM_ERR), 'darkred'),
          ('Method 2:  L = %.3f $\\pm$ %.3f"' % (L2, np.sqrt(C2[0, 0])), 'tab:blue'),
          ('      scale %+.1f ppm from imported' % mu2[1], 'tab:blue'),
          ('Imported plate scale: %.7f "/px' % PS, 'black'),
          ('      (the L+R8 bracket, $\\pm$%.1f ppm HC3-class)' % (SCALE_PPM*1e6), 'black')]
fig, ax = covariance_chart(C1, mu1, C2, mu2, _lines,
                           'L and plate scale \u2014 Bruns 2017, G $\\leq$ 11, 14-star link',
                           newton=False, box_alpha=1.0)
print('covariance: M1 stat %.4f (bootstrap, quoted) vs %.4f (analytic fit); scale %.4f; '
      'tot with atmosphere %.3f' % (sL1, sL1_analytic, pc, _tot1))
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
    disp = arcsinh_stretch(img)
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
# ---- the record copy, the same construction the Leon and Mexico tools use
# Added 2026-09-04 (Douglas): RECORD/bruns2017 was being filled by hand, so a regenerated
# chart overwrote its predecessor and the older version survived only under chart_versions/
# here. Now the copy is part of the run and anything it would overwrite is moved into a
# dated superseded/ folder first, as in RECORD/leon2026.
if os.environ.get('B17_COPY_RECORD') == '1':
    import filecmp
    import shutil
    os.makedirs(RECORD, exist_ok=True)
    produced = ['record_deflection.png', 'record_deflection_link14.png',
                'record_deflection_g13.png', 'record_field.png', 'record_covariance.png',
                'master009_annotated.png', 'master062_annotated.png',
                'bruns_method_star_table.csv']
    src = {f: os.path.join(OUT, f) for f in produced}
    old_files = [f for f in os.listdir(RECORD)
                 if f in src and os.path.exists(src[f])
                 and not filecmp.cmp(src[f], os.path.join(RECORD, f), shallow=False)]
    if old_files:
        sup = os.path.join(RECORD, 'superseded_' + pd.Timestamp.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(sup, exist_ok=True)
        for f in old_files:
            shutil.move(os.path.join(RECORD, f), os.path.join(sup, f))
        print('superseded %d file(s) -> %s' % (len(old_files), sup))
    for f, q in src.items():
        if os.path.exists(q):
            shutil.copy2(q, os.path.join(RECORD, f))
    print('record set ->', RECORD)
print('charts ->', OUT)
