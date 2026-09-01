"""The summary chart set for Leon 2026's reduction of record -- the Leon copy of
tools/matrix_bruns/b17_charts_record.py, same four charts, same construction, same rules
(docs/STEP3_CHARTS_AND_SETTINGS.md section 1).

Inputs are the tables written by tools/step3_record_table.py (the headline union: 0.6+1.2 s
tiers, G <= 11, R > 2 R_sun, below-Sun anchor in) and the nulls from
tools/step3_atmosphere.py. The atmosphere night maps are drawn by
tools/step3_atmosphere_maps.py; this script produces the other three and their variants.

What differs from the Bruns copy, and why:

  * the estimator carries the vertical-deg-2 nuisance, because that is how Leon's L was
    measured (S1 gate); every chart says so in its title, and a no-nuisance variant is
    drawn beside the record so the nuisance's effect is visible rather than hidden;
  * the field chart comes in two views: the L-view (pointing, rotation AND nuisance
    removed -- what the L column sees) and the raw view (pointing and rotation only --
    where the vertically polarised atmosphere is still in the arrows);
  * the scale term is MEASURED on this field's geometry by injecting a uniform 1-ppm
    plate-scale error into the star positions and reading the fitted L, rather than the
    naive eq-23 h*R_sun*dS. Measured: 0.0278"/ppm with the vertical nuisance on, 0.0209
    without, against the naive 0.0257 -- so on this ONE-SIDED field the free offsets and
    rotation do NOT suppress a uniform scale error (the ~6x suppression the M5 rehearsal
    measured was for its residual's compression-like content, not for an isotropic
    scale). At CAL_piLeo's 25 ppm that is +-0.70" of L, the largest term in the budget,
    and it was absent from the quoted headline (1.98 +- 0.60 stat +- 0.33 atm) -- the
    S2 plots carried it in their covariance chart but the quote dropped it. The legend
    quotes the measured and the naive value side by side;
  * the atmosphere term is the quoted +-0.33 (the S1 gate's max over three night
    windows); the cell-1 statistic (rms over windows) is read from atmosphere_nulls.csv
    and stated beside it.

Rules kept from the Bruns chart reviews: arrows are vectors, not radial projections; a
unit sensor displacement is asserted to round-trip to one arcsec of sky (the check the
Bruns chart lacked until revision 10); every arrow end is asserted inside the axes;
positions, the Sun and the vectors share one frame (the 0.6 s tier's sky->sensor affine,
inverted); the variant is in every title; every chart is also written under
chart_versions/<rev>_*, and superseded revisions are never deleted.

Set L26_COPY_RECORD=1 to also copy the record set into RECORD/leon2026/ (older charts
there are moved into a dated superseded/ folder, not deleted).
"""
import glob, json, os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyBboxPatch

OUT = r"D:/MEE2024 output/MEE_output/step3_record"
VER = os.path.join(OUT, 'chart_versions')
os.makedirs(VER, exist_ok=True)
REV = os.environ.get('L26_REV', 'rev01')
RECORD = r"D:/MEE2024 output/MEE_output/RECORD/leon2026"

meta = json.load(open(os.path.join(OUT, 'leon_union_meta.json'), encoding='utf-8'))
PS, NX, NY, W_NORM = meta['PS'], meta['NX'], meta['NY'], meta['W_NORM']
SUNPX, SUNPY, R_SUN_AS = meta['SUNPX'], meta['SUNPY'], meta['R_SUN_AS']
GR, NEWTON = 1.7512, 0.8756
SCALE_PPM = 25.0                 # CAL_piLeo, HC3-class (docs/CAL_PILEO_STEP2.md)
ATM_ERR = 0.33                   # quoted: the S1 gate's max over the three night windows
try:
    _nulls = pd.read_csv(os.path.join(OUT, 'atmosphere_nulls.csv'))
    _v = _nulls[_nulls.kind == 'median'].Lv.values
    ATM_RMS, ATM_MAX = float(np.sqrt(np.mean(_v**2))), float(np.abs(_v).max())
except Exception:
    ATM_RMS, ATM_MAX = float('nan'), float('nan')

ax_ = np.array(meta['affine_ax']); ay_ = np.array(meta['affine_ay'])
ra0, de0 = meta['affine_ra0'], meta['affine_de0']
Minv = np.linalg.inv(np.array([[ax_[0], ax_[1]], [ay_[0], ay_[1]]]))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140)
    fig.savefig(os.path.join(VER, REV + '_' + name), dpi=140)
    plt.close(fig)


def px_to_sky(px, py):
    v = Minv @ np.vstack([np.asarray(px, float) - ax_[2], np.asarray(py, float) - ay_[2]])
    return ra0 + v[0]/np.cos(np.radians(de0)), de0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    """Sensor-axis displacement (arcsec) -> (arcsec of RA*cos(dec), arcsec of Dec)."""
    v = Minv @ np.vstack([np.asarray(dx_as, float)/PS, np.asarray(dy_as, float)/PS])
    return v[0]*3600, v[1]*3600


_rt = np.hypot(*sensor_vec_to_sky(np.array([1.0]), np.array([0.0])))[0]
assert abs(_rt - 1.0) < 0.01, 'a 1 arcsec sensor displacement maps to %.3f arcsec' % _rt


def load(name):
    t = pd.read_csv(os.path.join(OUT, name))
    rx_, ry_ = (t.px.values-SUNPX)*PS, (t.py.values-SUNPY)*PS
    return t, rx_, ry_, np.hypot(rx_, ry_)


def design(px, py, rx_, ry_, R_, nuis_deg=2, with_scale=False):
    """The union's design matrix (tools/step3_s2_union.py), plus the optional scale column."""
    xs, ys = (px-NX/2)/W_NORM, (py-NY/2)/W_NORM
    ux, uy = rx_/R_, ry_/R_
    m = len(px); Z = np.zeros(m)
    cols_x = [np.ones(m), Z, -(py-NY/2)*PS]
    cols_y = [Z, np.ones(m), (px-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((px-NX/2)*PS); cols_y.append((py-NY/2)*PS); labels.append('S')
    cols_x.append(ux*R_SUN_AS/R_); cols_y.append(uy*R_SUN_AS/R_); labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def solve(t, rx_, ry_, R_, nuis_deg=2, with_scale=False):
    m = len(t)
    A, labels = design(t.px.values, t.py.values, rx_, ry_, R_, nuis_deg, with_scale)
    b = np.concatenate([t.dx.values, t.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    iL = labels.index('L')
    # the L-view: data minus everything the fit attributes to the non-L columns
    other = A@c - A[:, iL]*c[iL]
    clean = b - other
    # the raw view: only pointing and rotation removed
    keep = [labels.index(k) for k in ('N1', 'N2', 'Th')]
    raw = b - sum(A[:, k]*c[k] for k in keep)
    return c, labels, cov, clean[:m], clean[m:], raw[:m], raw[m:]


def boot_se(t, rx_, ry_, R_, nuis_deg=2, nb=1000):
    rng = np.random.default_rng(3)
    out = []
    for _ in range(nb):
        k = rng.integers(0, len(t), len(t))
        try:
            A, labels = design(t.px.values[k], t.py.values[k], rx_[k], ry_[k], R_[k], nuis_deg)
            c, *_ = np.linalg.lstsq(A, np.concatenate([t.dx.values[k], t.dy.values[k]]),
                                    rcond=None)
            out.append(c[labels.index('L')])
        except Exception:
            pass
    return float(np.std(out, ddof=1))


def scale_leverage(t, rx_, ry_, R_, nuis_deg=2):
    """Fitted L (arcsec) per ppm of uniform plate-scale error, on THIS geometry, with the
    estimator actually used. Signed."""
    A, labels = design(t.px.values, t.py.values, rx_, ry_, R_, nuis_deg)
    dx = 1e-6*(t.px.values-NX/2)*PS; dy = 1e-6*(t.py.values-NY/2)*PS
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return float(c[labels.index('L')])


def deflection_chart(table_csv, fname, title, nuis_deg=2):
    t, rx_, ry_, R_ = load(table_csv)
    anchor = t.is_anchor.values.astype(bool)
    c, l, cov, dxc, dyc, _, _ = solve(t, rx_, ry_, R_, nuis_deg)
    L = c[l.index('L')]
    se_stat = boot_se(t, rx_, ry_, R_, nuis_deg)
    g = scale_leverage(t, rx_, ry_, R_, nuis_deg)
    h = 1/np.mean((R_SUN_AS/R_)**2)
    scale_err = abs(g)*SCALE_PPM
    naive = h*R_SUN_AS*SCALE_PPM*1e-6
    tot = float(np.hypot(np.hypot(se_stat, scale_err), ATM_ERR))
    rad = dxc*(rx_/R_) + dyc*(ry_/R_)
    tanc = -dxc*(ry_/R_) + dyc*(rx_/R_)
    resid = rad - L*R_SUN_AS/R_
    rms = float(np.sqrt(np.mean(resid**2)))
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.axhline(0, color='black', lw=1)
    ax.scatter(R_[~anchor]/R_SUN_AS, rad[~anchor], s=38, color='tab:blue', zorder=4,
               label='0.6+1.2 s union (%d stars)' % int((~anchor).sum()))
    if anchor.any():
        ax.scatter(R_[anchor]/R_SUN_AS, rad[anchor], s=80, marker='D', color='tab:red',
                   zorder=5, label='below-Sun anchor, G 7.71 at %.2f R$_\\odot$'
                   % (R_[anchor][0]/R_SUN_AS))
    for k in np.where(np.abs(resid) > 2.5*rms)[0]:
        ax.annotate('  G %.2f (%+.1f$\\sigma$)' % (t.mag.values[k], resid[k]/rms),
                    (R_[k]/R_SUN_AS, rad[k]), fontsize=7.5, color='crimson')
    xx = np.linspace(1.9, (R_/R_SUN_AS).max()+0.3, 300)
    ax.fill_between(xx, (L-tot)/xx, (L+tot)/xx, color='black', alpha=0.10,
                    label='total $\\pm$%.2f" (stat %.2f + scale %.2f + atm %.2f)'
                          % (tot, se_stat, scale_err, ATM_ERR))
    ax.plot(xx, L/xx, color='black', lw=2.2, label='Method 1 fit:  L = %.3f"' % L)
    ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
    ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
    ax.set_xlabel('radial position (solar radii)', fontsize=13)
    ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.text(0.06, 0.012, 'scale term = %.0f ppm (CAL_piLeo, HC3-class) $\\times$ the measured '
             'leverage %.4f"/ppm on this geometry with this estimator;\nthe naive eq-23 bound '
             'h$\\cdot$R$_\\odot\\cdot\\delta$S would be %.2f" (h = %.1f R$_\\odot^2$)'
             % (SCALE_PPM, g, naive, h), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if not fname.startswith('_scratch'):
        save(fig, fname)
    else:
        plt.close(fig)
    print('%-38s N=%d h=%.1f L=%+.3f stat %.3f scale %.3f (naive %.2f; g=%.4f"/ppm) atm %.2f '
          'tot %.3f | rad-about-fit rms %.3f tan rms %.3f'
          % (fname, len(t), h, L, se_stat, scale_err, naive, g, ATM_ERR, tot, rms,
             float(np.sqrt(np.mean(tanc**2)))), flush=True)
    return dict(t=t, rx=rx_, ry=ry_, R=R_, c=c, l=l, cov=cov, dxc=dxc, dyc=dyc, L=L,
                se=se_stat, g=g, h=h, scale_err=scale_err, tot=tot, rms=rms, anchor=anchor,
                star_rms=float(np.sqrt(np.mean(resid**2 + tanc**2))))


TITLE = 'Leon 2026, 0.6+1.2 s union, G $\\leq$ 11, anchor in, vertical-deg-2 nuisance'
rec = deflection_chart('leon_union_star_table.csv', 'record_deflection.png',
                       'Deflection vs radius \u2014 ' + TITLE)
deflection_chart('leon_union_star_table_sans_anchor.csv', 'record_deflection_sans_anchor.png',
                 'Deflection vs radius \u2014 Leon 2026, 0.6+1.2 s union, G $\\leq$ 11, '
                 'anchor OUT, vertical-deg-2 nuisance')
deflection_chart('leon_union_star_table.csv', 'record_deflection_no_nuisance.png',
                 'Deflection vs radius \u2014 Leon 2026, 0.6+1.2 s union, G $\\leq$ 11, '
                 'anchor in, NO nuisance (Method 1 base)', nuis_deg=None)
deflection_chart('leon_union_star_table_full4.csv', 'record_deflection_full4.png',
                 'Deflection vs radius \u2014 Leon 2026, FULL 4-tier union (cross-check, not '
                 'the record), G $\\leq$ 11, anchor in, vertical-deg-2 nuisance')

# ---- the field: displacement vectors in RA/Dec, two views
tab, rx, ry, R = rec['t'], rec['rx'], rec['ry'], rec['R']
anchor = rec['anchor']
n = len(tab)
_, _, _, dxc_L, dyc_L, dxc_raw, dyc_raw = solve(tab, rx, ry, R, 2)
sra, sdec = px_to_sky(tab.px.values, tab.py.values)
corners = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
sun_ra, sun_dec = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
ARROW_DEG = 0.34          # degrees of chart per arcsec of displacement: twice the Bruns
                          # chart's 0.17, because this frame is twice as wide (3.8 vs 1.9
                          # deg) -- the arrows keep the same size relative to the frame


def field_chart(dxv, dyv, fname, title, caption, scatter_label):
    vra, vdec = sensor_vec_to_sky(dxv, dyv)
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.add_patch(Polygon(np.c_[corners[0], corners[1]], fill=False, color='gray', lw=1.2,
                         label='sensor footprint'))
    ends_ra, ends_de = [], []
    for k in range(n):
        col = 'tab:red' if anchor[k] else 'tab:blue'
        x0, y0 = sra[k], sdec[k]
        x1 = x0 + vra[k]*ARROW_DEG/np.cos(np.radians(de0))
        y1 = y0 + vdec[k]*ARROW_DEG
        ends_ra.append(x1); ends_de.append(y1)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                    color=col, lw=1.5, shrinkA=0, shrinkB=0))
        ax.annotate('%.1f' % tab.mag.values[k], (x0 + (0.030 if anchor[k] else 0.010), y0),
                    fontsize=6.5, color=('tab:red' if anchor[k] else 'black'))
    ax.scatter(sra[~anchor], sdec[~anchor], s=22, color='tab:blue', zorder=5,
               label='0.6+1.2 s union (%d stars)' % int((~anchor).sum()))
    ax.scatter(sra[anchor], sdec[anchor], s=70, marker='D', color='tab:red', zorder=5,
               label='below-Sun anchor, G 7.71')
    ax.add_patch(Ellipse((float(sun_ra[0]), float(sun_dec[0])),
                         2*R_SUN_AS/3600/np.cos(np.radians(de0)), 2*R_SUN_AS/3600,
                         color='black', zorder=3, label='the Sun, 1 R$_\\odot$ to scale'))
    lo_ra = min(min(corners[0]), min(ends_ra)) - 0.08
    hi_ra = max(max(corners[0]), max(ends_ra)) + 0.08
    lo_de = min(min(corners[1]), min(ends_de)) - 0.06
    hi_de = max(max(corners[1]), max(ends_de)) + 0.06
    for x1, y1 in zip(ends_ra, ends_de):
        assert lo_ra < x1 < hi_ra and lo_de < y1 < hi_de, 'an arrow leaves the axes'
    ax.set_xlim(lo_ra, hi_ra); ax.set_ylim(lo_de, hi_de)
    ax.set_aspect(1/np.cos(np.radians(de0)))
    ax.set_xlabel('RA (degrees)', fontsize=12)
    ax.set_ylabel('DEC (degrees)', fontsize=12)
    ax.set_title(title, fontsize=12)
    fig.text(0.06, 0.020, caption, fontsize=9)
    ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.75))
    bar_deg = ARROW_DEG/np.cos(np.radians(de0))
    sc = float(np.sqrt(np.mean(dxv**2 + dyv**2)))
    for y_fr, ln, txt in ((0.44, 1.0, '1 arcsec of displacement'),
                          (0.34, sc, '%s (%.2f")' % (scatter_label, sc))):
        xa, ya = 1.04, y_fr
        ax.annotate('', xy=(xa + ln*bar_deg/(hi_ra-lo_ra), ya), xytext=(xa, ya),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', color='black', lw=3))
        ax.annotate(txt, (xa, ya + 0.028), xycoords='axes fraction', fontsize=8)
    ax.add_patch(FancyBboxPatch((1.02, 0.29), 0.34, 0.22, boxstyle='round,pad=0.012',
                                transform=ax.transAxes, fill=False, color='gray', lw=0.9,
                                clip_on=False))
    fig.subplots_adjust(right=0.72, bottom=0.13)
    save(fig, fname)
    print('%s: arrows %.2f-%.2f", rms vector %.3f"' % (fname, float(np.hypot(vra, vdec).min()),
                                                        float(np.hypot(vra, vdec).max()), sc))


field_chart(dxc_L, dyc_L, 'record_field.png', 'Displacement vectors \u2014 ' + TITLE,
            'each arrow = the star\u2019s measured shift after subtracting the fitted pointing '
            'offset, rotation AND the vertical-deg-2 nuisance field;\ndeflection + measurement '
            'noise remain (the L-view)', 'rms vector')
field_chart(dxc_raw, dyc_raw, 'record_field_raw.png',
            'Displacement vectors, nuisance left IN \u2014 Leon 2026, 0.6+1.2 s union, '
            'G $\\leq$ 11, anchor in',
            'each arrow = the star\u2019s measured shift after subtracting only the fitted '
            'pointing offset and rotation;\ndeflection + the vertically polarised atmosphere + '
            'noise remain', 'rms vector')

# ---- L and plate scale: Method 1 (imported scale, measured leverage) and Method 2 (scale free)
L1, se1, g = rec['L'], rec['se'], rec['g']
c2, l2, cov2, _, _, _, _ = solve(tab, rx, ry, R, 2, with_scale=True)
iL2, iS2 = l2.index('L'), l2.index('S')
L2 = c2[iL2]
pc = g*SCALE_PPM                                     # arcsec of L per 1 sigma of scale (signed)
# a true scale LARGER than the imported one makes the model's angles too small, so the
# displacements carry -e*(px-centre)*S and L moves by -g*e: L and the scale are
# anti-correlated (Method 2's own covariance shows the same tilt)
C1 = np.array([[se1**2 + pc**2, -pc*SCALE_PPM], [-pc*SCALE_PPM, SCALE_PPM**2]])
mu1 = np.array([L1, 0.0])
C2 = np.array([[cov2[iL2, iL2], cov2[iL2, iS2]*1e6], [cov2[iS2, iL2]*1e6, cov2[iS2, iS2]*1e12]])
mu2 = np.array([L2, 1e6*c2[iS2]])
tot1 = float(np.hypot(np.sqrt(C1[0, 0]), ATM_ERR))
fig, ax = plt.subplots(figsize=(9.5, 7))


def draw(cov, mu, color, name):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), angle=ang, fill=False,
                         color=color, lw=1.6, label='1$\\sigma$ \u2014 %s' % name))
    ax.scatter(*mu, marker='+', s=110, color=color, zorder=5)


draw(C1, mu1, 'darkred', 'Method 1 (scale imported; stat + scale)')
draw(C2, mu2, 'tab:blue', 'Method 2 (scale free)')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.axvline(NEWTON, color='orange', lw=1.2, ls='--', label='Newton 0.876"')
ax.add_patch(FancyBboxPatch((0.020, 0.030), 0.60, 0.215, boxstyle='round,pad=0.010',
                            transform=ax.transAxes, fill=True, facecolor='white',
                            edgecolor='gray', lw=0.9, zorder=6))
ax.text(0.038, 0.205, 'Method 1:  L = %.3f $\\pm$ %.3f" (stat %.3f + scale %.3f); '
        '$\\pm$ %.2f" with atmosphere %.2f' % (L1, np.sqrt(C1[0, 0]), se1, abs(pc), tot1, ATM_ERR),
        transform=ax.transAxes, fontsize=9.5, color='darkred', zorder=7)
ax.text(0.038, 0.150, 'Method 2:  L = %.3f $\\pm$ %.3f",  scale %+.1f ppm from imported'
        % (L2, np.sqrt(C2[0, 0]), mu2[1]), transform=ax.transAxes, fontsize=9.5,
        color='tab:blue', zorder=7)
ax.text(0.038, 0.100, 'Imported plate scale: %.7f "/px (CAL_piLeo, $\\pm$%.0f ppm HC3-class)'
        % (PS, SCALE_PPM), transform=ax.transAxes, fontsize=9.5, color='black', zorder=7)
ax.text(0.038, 0.050, 'scale leverage measured by injection: %+.4f "/ppm (naive eq-23: '
        '%.4f "/ppm)' % (g, rec['h']*R_SUN_AS*1e-6), transform=ax.transAxes, fontsize=9.0,
        color='black', zorder=7)
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('Plate scale (ppm difference from imported value)', fontsize=12)
ax.set_title('L and plate scale \u2014 ' + TITLE, fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.autoscale_view()
ax.margins(0.15)
save(fig, 'record_covariance.png')
print('covariance: M1 L=%.3f +- %.3f (stat+scale), tot %.3f; M2 L=%.3f +- %.3f, scale %+.1f ppm'
      % (L1, np.sqrt(C1[0, 0]), tot1, L2, np.sqrt(C2[0, 0]), mu2[1]), flush=True)
print('atmosphere: quoted +-%.2f (S1 gate max); cell-1 statistic rms %.3f, max %.3f'
      % (ATM_ERR, ATM_RMS, ATM_MAX))

summary = dict(rev=REV, N=int(n), h=rec['h'], L_method1=L1, stat=se1, scale=abs(pc),
               scale_leverage_as_per_ppm=g, scale_ppm=SCALE_PPM, atmosphere=ATM_ERR,
               atmosphere_rms_over_windows=ATM_RMS, atmosphere_max_over_windows=ATM_MAX,
               total=rec['tot'], L_method2=L2, method2_stat=float(np.sqrt(C2[0, 0])),
               method2_scale_ppm=float(mu2[1]), radial_about_fit_rms=rec['rms'],
               star_rms=rec['star_rms'])
json.dump(summary, open(os.path.join(OUT, 'record_summary.json'), 'w'), indent=1)
print('charts ->', OUT)

if os.environ.get('L26_COPY_RECORD') == '1':
    os.makedirs(RECORD, exist_ok=True)
    old = [f for f in os.listdir(RECORD) if f.endswith(('.png', '.csv', '.json'))]
    if old:
        sup = os.path.join(RECORD, 'superseded_' + pd.Timestamp.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(sup, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(RECORD, f), os.path.join(sup, f))
    for f in ('record_deflection.png', 'record_deflection_sans_anchor.png',
              'record_deflection_no_nuisance.png', 'record_deflection_full4.png',
              'record_field.png', 'record_field_raw.png', 'record_covariance.png',
              'atmosphere_night_maps.png', 'leon_union_star_table.csv',
              'leon_union_star_table_sans_anchor.csv', 'leon_union_star_table_full4.csv',
              'leon_union_meta.json', 'atmosphere_nulls.csv', 'atmosphere_maps_stats.csv',
              'record_summary.json'):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RECORD, f))
    print('record set ->', RECORD)
