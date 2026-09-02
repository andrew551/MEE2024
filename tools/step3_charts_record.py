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

Revision 2 (Douglas' review, 2026-09-02):
  * the field charts move to ALT/AZ. The sensor sits within a few degrees of that frame,
    so the footprint is nearly axis-aligned and the vertical axis is the direction the
    atmosphere is polarised along -- which is what the raw view exists to show;
  * the below-Sun star G 7.71 loses its red diamond and becomes an ordinary member. It is
    not the analogue of Bruns' close-in pair: those sat at 1.49 and 1.62 R_sun and had to
    be carried from a different exposure tier by a measured link, while this one is in the
    same two tiers as everything else at 2.17 R_sun. The sans-anchor chart survives as a
    leverage test, retitled;
  * the covariance text box is packed around its own text (VPacker) instead of being given
    a hand-guessed size, and every entry wraps onto a second line;
  * outlier labels now name the number of tiers the star was seen in, and a two-witness
    variant chart is drawn: the rule that admits only stars detected in BOTH tiers, so the
    cross-tier consistency vet can act on every member. That is what removes the G 10.00
    outlier -- a single-tier, 2-px-footprint detection the vet never had a second witness
    for. It is a detection-quality rule, not a rule about the answer, and it moves L by
    -0.06 " (a tenth of the statistical error).

Revision 3 (2026-09-02, same evening):
  * **the two-witness rule becomes the record** on Douglas' instruction, for the deflection,
    covariance and field charts alike. Every chart here is now the 36-star set unless its
    title says otherwise; `record_deflection_all_matches.png` keeps the superseded 42-star
    admission as a variant. Nothing else about the reduction changes;
  * the two exposure tiers get annotated master images, the analogue of cell 1's
    master009/master062: yellow for the stars both exposures found, red for the six only
    one of them did -- which is a picture of the rule itself. They are drawn from the
    aligned raw stacks written by `tools/step3_tier_stacks.py`.
"""
import glob, json, os, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

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


def deflection_chart(table_csv, fname, title, nuis_deg=2, two_witness=True):
    t, rx_, ry_, R_ = load(table_csv)
    if two_witness:
        keep = t.ntier.values >= 2
        t, rx_, ry_, R_ = t[keep].reset_index(drop=True), rx_[keep], ry_[keep], R_[keep]
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
    # every star carries the same marker: unlike Bruns' close-in pair, which came from a
    # different exposure tier and had to be linked in, Leon's innermost star is an ordinary
    # member of the same two tiers as the rest (Douglas, 2026-09-02)
    ax.scatter(R_/R_SUN_AS, rad, s=38, color='tab:blue', zorder=4,
               label='0.6+1.2 s union (%d stars)' % len(t))
    for k in np.where(np.abs(resid) > 2.5*rms)[0]:
        nt = int(t.ntier.values[k])
        ax.annotate('  G %.2f (%+.1f$\\sigma$, %d tier%s)'
                    % (t.mag.values[k], resid[k]/rms, nt, '' if nt == 1 else 's'),
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


TITLE = ('Leon 2026, 0.6+1.2 s union, G $\\leq$ 11, two-witness rule, 36 stars, '
         'vertical-deg-2 nuisance')
rec = deflection_chart('leon_union_star_table.csv', 'record_deflection.png',
                       'Deflection vs radius \u2014 ' + TITLE)
deflection_chart('leon_union_star_table_sans_anchor.csv', 'record_deflection_sans_anchor.png',
                 'Deflection vs radius \u2014 Leon 2026, two-witness union, G $\\leq$ 11, the '
                 'innermost star (G 7.71 at 2.17 R$_\\odot$) dropped \u2014 leverage test')
deflection_chart('leon_union_star_table.csv', 'record_deflection_no_nuisance.png',
                 'Deflection vs radius \u2014 Leon 2026, two-witness union, G $\\leq$ 11, '
                 'NO nuisance (Method 1 base)', nuis_deg=None)
deflection_chart('leon_union_star_table_full4.csv', 'record_deflection_full4.png',
                 'Deflection vs radius \u2014 Leon 2026, FULL 4-tier union (cross-check, not '
                 'the record), G $\\leq$ 11, two-witness, vertical-deg-2 nuisance')
# the superseded admission rule, kept as the variant it became: every catalogue match
# admitted, single-witness stars included. It is what the record was until 2026-09-02.
deflection_chart('leon_union_star_table.csv', 'record_deflection_all_matches.png',
                 'Deflection vs radius \u2014 Leon 2026, 0.6+1.2 s union, G $\\leq$ 11, EVERY '
                 'match admitted (42 stars, superseded), vertical-deg-2 nuisance',
                 two_witness=False)

# ---- the field: displacement vectors in ALT/AZ, two views
#
# Douglas, 2026-09-02: put the field chart into alt/az. The sensor is within a few degrees
# of that frame already (ROLL 315.5 deg against a parallactic angle that nearly cancels it),
# so the footprint comes out almost axis-aligned -- and the vertical axis is then the
# direction the atmosphere is polarised along, which is the whole point of the raw view.
# Positions come from each star's own catalogue position transformed to alt/az; the vectors
# are projected onto the alt and az directions measured at the field centre. Azimuth
# increases to the right and altitude upward, which is the sky as the observer sees it.
tab, rx, ry, R = rec['t'], rec['rx'], rec['ry'], rec['R']
n = len(tab)
_, _, _, dxc_L, dyc_L, dxc_raw, dyc_raw = solve(tab, rx, ry, R, 2)
T_SCI = Time(meta['OPTS']['observation_date'] + 'T' + meta['MIDT']['0p6s'], scale='utc')
SITE = EarthLocation(lat=meta['OPTS']['observation_lat']*u.deg,
                     lon=meta['OPTS']['observation_long']*u.deg,
                     height=meta['OPTS']['observation_height']*u.m)


def sky_to_altaz(ra, dec):
    """Geometric alt/az (no refraction: pressure defaults to zero) at the science mid-time."""
    aa = SkyCoord(np.atleast_1d(ra)*u.deg, np.atleast_1d(dec)*u.deg).transform_to(
        AltAz(obstime=T_SCI, location=SITE))
    return np.asarray(aa.az.deg), np.asarray(aa.alt.deg)


def altaz_basis():
    """Unit vectors of increasing altitude and azimuth, in sensor pixel axes."""
    ra_c, dec_c = px_to_sky(np.array([NX/2.0]), np.array([NY/2.0]))
    fc = SkyCoord(float(ra_c[0])*u.deg, float(dec_c[0])*u.deg)
    aa = fc.transform_to(AltAz(obstime=T_SCI, location=SITE))
    out = {}
    for key, off in (('alt', dict(alt=aa.alt + 0.05*u.deg, az=aa.az)),
                     ('az', dict(alt=aa.alt, az=aa.az + 0.05*u.deg/np.cos(aa.alt)))):
        p = SkyCoord(AltAz(obstime=T_SCI, location=SITE, **off)).icrs
        dv = np.array([(p.ra.deg - fc.ra.deg)*np.cos(np.radians(de0)), p.dec.deg - fc.dec.deg])
        v = np.array([dv @ ax_[:2], dv @ ay_[:2]])          # sky degrees -> sensor pixels
        out[key] = v/np.linalg.norm(v)
    return out['alt'], out['az'], float(aa.alt.deg), float(aa.az.deg)


E_ALT, E_AZ, ALT_C, AZ_C = altaz_basis()
TILT = np.degrees(np.arctan2(E_ALT[0], -E_ALT[1]))   # sensor -y against the local vertical
saz, salt = sky_to_altaz(tab.ra_cat.values, tab.dec_cat.values)
corner_ra, corner_dec = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
caz, calt = sky_to_altaz(corner_ra, corner_dec)
sun_ra, sun_dec = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
sunaz, sunalt = sky_to_altaz(sun_ra, sun_dec)
COSA = np.cos(np.radians(ALT_C))
ARROW_DEG = 0.34          # degrees of chart per arcsec of displacement: twice the Bruns
                          # chart's 0.17, because this frame is twice as wide (3.8 vs 1.9
                          # deg) -- the arrows keep the same size relative to the frame
print('field centre alt %.3f deg az %.3f deg; sensor -y sits %.1f deg from the local '
      'vertical' % (ALT_C, AZ_C, TILT))


def field_chart(dxv, dyv, fname, title, caption, scatter_label):
    v_alt = dxv*E_ALT[0] + dyv*E_ALT[1]
    v_az = dxv*E_AZ[0] + dyv*E_AZ[1]
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.add_patch(Polygon(np.c_[caz, calt], fill=False, color='gray', lw=1.2,
                         label='sensor footprint'))
    ends_az, ends_alt = [], []
    for k in range(n):
        x0, y0 = saz[k], salt[k]
        x1 = x0 + v_az[k]*ARROW_DEG/COSA
        y1 = y0 + v_alt[k]*ARROW_DEG
        ends_az.append(x1); ends_alt.append(y1)
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                    color='tab:blue', lw=1.5, shrinkA=0, shrinkB=0))
        ax.annotate('%.1f' % tab.mag.values[k], (x0 + 0.010, y0), fontsize=6.5, color='black')
    ax.scatter(saz, salt, s=22, color='tab:blue', zorder=5,
               label='0.6+1.2 s union (%d stars)' % n)
    ax.add_patch(Ellipse((float(sunaz[0]), float(sunalt[0])), 2*R_SUN_AS/3600/COSA,
                         2*R_SUN_AS/3600, color='black', zorder=3,
                         label='the Sun, 1 R$_\\odot$ to scale'))
    lo_az = min(caz.min(), min(ends_az)) - 0.10
    hi_az = max(caz.max(), max(ends_az)) + 0.10
    lo_alt = min(calt.min(), min(ends_alt)) - 0.08
    hi_alt = max(calt.max(), max(ends_alt)) + 0.08
    for x1, y1 in zip(ends_az, ends_alt):
        assert lo_az < x1 < hi_az and lo_alt < y1 < hi_alt, 'an arrow leaves the axes'
    ax.set_xlim(lo_az, hi_az); ax.set_ylim(lo_alt, hi_alt)
    ax.set_aspect(1/COSA)
    ax.set_xlabel('azimuth (degrees, increasing to the right as seen by the observer)',
                  fontsize=12)
    ax.set_ylabel('altitude (degrees)', fontsize=12)
    ax.set_title(title, fontsize=12)
    fig.text(0.06, 0.020, caption, fontsize=9)
    ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.75))
    bar_deg = ARROW_DEG/COSA
    sc = float(np.sqrt(np.mean(dxv**2 + dyv**2)))
    for y_fr, ln, txt in ((0.44, 1.0, '1 arcsec of displacement'),
                          (0.34, sc, '%s (%.2f")' % (scatter_label, sc))):
        xa, ya = 1.04, y_fr
        ax.annotate('', xy=(xa + ln*bar_deg/(hi_az-lo_az), ya), xytext=(xa, ya),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', color='black', lw=3))
        ax.annotate(txt, (xa, ya + 0.028), xycoords='axes fraction', fontsize=8)
    ax.add_patch(FancyBboxPatch((1.02, 0.29), 0.34, 0.22, boxstyle='round,pad=0.012',
                                transform=ax.transAxes, fill=False, color='gray', lw=0.9,
                                clip_on=False))
    fig.subplots_adjust(right=0.72, bottom=0.13)
    save(fig, fname)
    print('%s: vertical rms %.3f", horizontal rms %.3f" (V/H %.1f), arrows %.2f-%.2f"'
          % (fname, float(np.sqrt(np.mean(v_alt**2))), float(np.sqrt(np.mean(v_az**2))),
             float(np.sqrt(np.mean(v_alt**2))/np.sqrt(np.mean(v_az**2))),
             float(np.hypot(v_alt, v_az).min()), float(np.hypot(v_alt, v_az).max())))


field_chart(dxc_L, dyc_L, 'record_field.png', 'Displacement vectors \u2014 ' + TITLE,
            'each arrow = the star\u2019s measured shift after subtracting the fitted pointing '
            'offset, rotation AND the vertical-deg-2 nuisance field;\ndeflection + measurement '
            'noise remain (the L-view)', 'rms vector')
field_chart(dxc_raw, dyc_raw, 'record_field_raw.png',
            'Displacement vectors, nuisance left IN \u2014 Leon 2026, 0.6+1.2 s union, '
            'G $\\leq$ 11, 42 stars',
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
# the text box: each entry wraps onto a second, indented line, and the frame is packed
# around the text rather than given a hand-guessed size -- the hand-sized box of revision 1
# was too small for its own contents (Douglas, 2026-09-02)
BOXLINES = [
    ('Method 1:  L = %.3f $\\pm$ %.3f"' % (L1, np.sqrt(C1[0, 0])), 'darkred'),
    ('      (stat %.3f + scale %.3f);  $\\pm$ %.2f" with atmosphere %.2f'
     % (se1, abs(pc), tot1, ATM_ERR), 'darkred'),
    ('Method 2:  L = %.3f $\\pm$ %.3f"' % (L2, np.sqrt(C2[0, 0])), 'tab:blue'),
    ('      scale %+.1f ppm from imported' % mu2[1], 'tab:blue'),
    ('Imported plate scale: %.7f "/px' % PS, 'black'),
    ('      (CAL_piLeo, $\\pm$%.0f ppm HC3-class)' % SCALE_PPM, 'black'),
    ('Scale leverage measured by injection:', 'black'),
    ('      %+.4f "/ppm  (naive eq-23: %.4f "/ppm)' % (g, rec['h']*R_SUN_AS*1e-6), 'black'),
]
_pack = VPacker(children=[TextArea(t, textprops=dict(color=c, size=9.5))
                          for t, c in BOXLINES], pad=0, sep=3, align='left')
_box = AnchoredOffsetbox(loc='lower left', child=_pack, pad=0.45, borderpad=0.6,
                         frameon=True, bbox_to_anchor=(0.0, 0.0),
                         bbox_transform=ax.transAxes)
_box.patch.set(facecolor='white', edgecolor='gray', linewidth=0.9, alpha=1.0)
_box.set_zorder(6)
ax.add_artist(_box)
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

# ---- the two tier stacks, annotated: what each exposure actually delivered
#
# The cell-1 analogue is master009_annotated / master062_annotated, and the recipe is the
# same: a masked-blur coronal subtraction for display only, NO painted disk (the natural
# view), y increasing downward as this project has always displayed frames, legend below
# the frame. Two improvements on the Bruns version: the image is the ALIGNED raw stack
# (tools/step3_tier_stacks.py) rather than an unshifted mean, so the stars are not smeared
# by the two pixels of dither; and the blur is masked at saturation, which is the fix for
# the coronal trench the cell-1 preprocessing still carries.
#
# The circles answer the question directly (Douglas, 2026-09-02): yellow where both
# exposures found the star, red where only this one did. The red circles are exactly the
# stars the two-witness rule drops.
from astropy.io import fits as pyfits
from scipy.ndimage import gaussian_filter

STACKS = r"D:/MEE2024 output/MEE_output/SCI_tier_stacks"
allmatch = pd.read_csv(os.path.join(OUT, 'leon_union_star_table.csv'))


def tier_figure(tier, exp_label):
    p = os.path.join(STACKS, 'SCI_%s_mean.fits' % tier)
    if not os.path.exists(p):
        print('no aligned stack for %s -- run tools/step3_tier_stacks.py first' % tier)
        return
    hdr = pyfits.getheader(p)
    img = pyfits.getdata(p).astype(np.float64)
    valid = img < 65535
    num = gaussian_filter(np.where(valid, img, 0.0), 10.0)
    den = gaussian_filter(valid.astype(np.float64), 10.0)
    model = np.where(den > 0.05, num/np.maximum(den, 1e-9), 65535.0)
    sub = img - model
    lo, hi = np.percentile(sub, [5, 99.5])
    disp = np.arcsinh((np.clip(sub, lo, hi) - lo)/max(hi - lo, 1)*30)

    inmine = allmatch.tiers.str.contains(tier).values
    both = inmine & (allmatch.ntier.values >= 2)
    only = inmine & (allmatch.ntier.values == 1)
    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
    handles = []
    for sel, colr, lab in (
            (both, 'yellow', 'in both exposures (%d stars) — the two-witness set' % both.sum()),
            (only, 'red', 'in this exposure only (%d) — dropped by the two-witness rule'
             % only.sum())):
        for x0, y0 in zip(allmatch.px.values[sel], allmatch.py.values[sel]):
            ax.add_patch(Circle((x0, y0), 55, fill=False, color=colr, lw=1.4))
        handles.append(plt.Line2D([], [], color=colr, label=lab))
    ax.legend(handles=handles, fontsize=9, loc='upper left', bbox_to_anchor=(0.0, -0.07),
              borderaxespad=0, frameon=True)
    ax.set_title('The %s master (%d frames, %.1f s total) — Leon 2026 SCI_ladder\n'
                 'aligned raw stack, coronal model subtracted for display, no painted disk'
                 % (exp_label, hdr['NFRAMES'], hdr['EXPTOTAL']), fontsize=11)
    ax.set_xlabel('px'); ax.set_ylabel('py')
    fig.subplots_adjust(bottom=0.18)
    save(fig, 'master_%s_annotated.png' % tier)
    print('master_%s_annotated.png: %d in both exposures, %d in this one only'
          % (tier, both.sum(), only.sum()), flush=True)


tier_figure('0p6s', '0.6 s')
tier_figure('1p2s', '1.2 s')

summary = dict(rev=REV, N=int(n), h=rec['h'], L_method1=L1, stat=se1, scale=abs(pc),
               admission_rule='two-witness (detected in both tiers)',
               scale_leverage_as_per_ppm=g, scale_ppm=SCALE_PPM, atmosphere=ATM_ERR,
               atmosphere_rms_over_windows=ATM_RMS, atmosphere_max_over_windows=ATM_MAX,
               total=rec['tot'], L_method2=L2, method2_stat=float(np.sqrt(C2[0, 0])),
               method2_scale_ppm=float(mu2[1]), radial_about_fit_rms=rec['rms'],
               star_rms=rec['star_rms'])
json.dump(summary, open(os.path.join(OUT, 'record_summary.json'), 'w'), indent=1)
print('charts ->', OUT)

if os.environ.get('L26_COPY_RECORD') == '1':
    os.makedirs(RECORD, exist_ok=True)
    import filecmp
    # only files that actually differ from what is about to be copied are superseded;
    # re-running with the same output must not bury identical copies under a new date
    old = [f for f in os.listdir(RECORD) if f.endswith(('.png', '.csv', '.json'))
           and not (os.path.exists(os.path.join(OUT, f))
                    and filecmp.cmp(os.path.join(OUT, f), os.path.join(RECORD, f), shallow=False))]
    if old:
        sup = os.path.join(RECORD, 'superseded_' + pd.Timestamp.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(sup, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(RECORD, f), os.path.join(sup, f))
    for f in ('record_deflection.png', 'record_deflection_sans_anchor.png',
              'record_deflection_no_nuisance.png', 'record_deflection_full4.png',
              'record_deflection_all_matches.png', 'master_0p6s_annotated.png',
              'master_1p2s_annotated.png', 'zenith_floor.csv', 'zenith_nulls.csv',
              'record_field.png', 'record_field_raw.png', 'record_covariance.png',
              'atmosphere_night_maps.png', 'leon_union_star_table.csv',
              'leon_union_star_table_sans_anchor.csv', 'leon_union_star_table_full4.csv',
              'leon_union_meta.json', 'atmosphere_nulls.csv', 'atmosphere_maps_stats.csv',
              'atmosphere_floor_table.csv', 'record_summary.json'):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RECORD, f))
    print('record set ->', RECORD)
