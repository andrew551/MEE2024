"""The summary chart set for cell 2's reduction of record — Mexico 2024, Station 1.

The cell-1 and cell-3 chart sets in the same style (`tools/matrix_bruns/b17_charts_record.py`,
`tools/step3_charts_record.py`), built on this cell's estimator: the pooled Method-2 fit over
every observation of the four totality blocks (`s1_pooled_fit.py --ref twopass`).

What differs from the other two cells, and why the charts differ:

  * **there is no Method 1 here.** Method 1 imports a plate scale; the daytime refocus put the
    only calibration ~600 ppm from the eclipse, so there is nothing that can legitimately be
    imported. Every chart is Method 2, and the covariance chart carries one ellipse, not two;
  * **the estimator is a pooled fit, not a union.** Each of the four blocks carries its own
    offset, rotation and scale and they share one L, so a star seen in four blocks is four
    rows. The deflection chart therefore plots observations; a per-star version is drawn
    beside it for legibility;
  * **the plate scale is a measurement, not an input.** The covariance chart's vertical axis is
    the fitted scale against the published one (Dittrich et al. 2025, 1.847363 +- 0.000013),
    and what is compared is the JOINT scale -- stage 2's corrected by the S the fit finds
    alongside L. A scale fitted without L has the deflection absorbed into it and is ~46 ppm
    low.

Rules kept from the Bruns and Leon chart reviews: arrows are vectors, not radial projections;
a unit sensor displacement is asserted to round-trip to one arcsec of sky; every arrow end is
asserted inside the axes; the variant is named in every title; every chart is also written
under chart_versions/<rev>_*, and superseded revisions are never deleted.

Set MX24_COPY_RECORD=1 to also copy the record set into RECORD/mexico2024/ (older charts there
are moved into a dated superseded/ folder, not deleted).
"""
import glob, json, os, shutil, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REV = 'rev01'
REC = r"D:/MEE2024 output/MEE_output/station1_record"
OUT = os.path.join(REC, 'charts')
VER = os.path.join(OUT, 'chart_versions')
RECORD = r"D:/MEE2024 output/MEE_output/RECORD/mexico2024"
POOLED = os.path.join(REC, 'pooled_fit', 'twopass')
os.makedirs(VER, exist_ok=True)

NX, NY, PS = 9576, 6388, 1.84847
GR, NEWTON = 1.7512, 0.8756
# the budget beside the statistical term (docs/STEP3_2026.md, "Cell 2's error terms")
ATM_ERR, MODEL_ORDER = 0.11, 0.03
# Dittrich et al. 2025, the published analysis of these frames
PUB_L, PUB_L_ERR = 1.839, 0.239
PUB_SCALE, PUB_SCALE_ERR = 1.847363, 1.3e-5
BLOCK_LABEL = {'0p25s_1810': '0.25 s, 18:11:12', '0p3s_1811': '0.3 s, 18:11:58',
               '0p4s_1812': '0.4 s, 18:13:00', '0p3s_1813': '0.3 s, 18:14:02'}
BLOCK_COLOR = {'0p25s_1810': 'tab:blue', '0p3s_1811': 'tab:orange',
               '0p4s_1812': 'tab:green', '0p3s_1813': 'tab:purple'}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140)
    fig.savefig(os.path.join(VER, REV + '_' + name), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- the rows and the fit
t = pd.read_csv(os.path.join(POOLED, 'pooled_rows.csv'))
summary = json.load(open(os.path.join(POOLED, 'pooled_summary.json')))
blocks = [b for b in BLOCK_LABEL if b in set(t.block)]
L, SE_STAT = summary['L'], summary['sigma_bootstrap']
TOT = float(np.sqrt(SE_STAT**2 + ATM_ERR**2 + MODEL_ORDER**2))
nblk = t.groupby('key').block.nunique()
t['nblk'] = t.key.map(nblk)
ux, uy = t.rx.values/t.R.values, t.ry.values/t.R.values
# the deflection each observation actually shows: its residual about the fit, plus the fit's
# own deflection at that radius. The per-block offset, rotation and scale are already out --
# res_x/res_y are the residuals of the pooled solve.
t['rad'] = (t.res_x.values*ux + t.res_y.values*uy) + L*t.RS.values/t.R.values
t['tan'] = -t.res_x.values*uy + t.res_y.values*ux
t['vx'] = t.res_x.values + L*t.RS.values/t.R.values*ux
t['vy'] = t.res_y.values + L*t.RS.values/t.R.values*uy
print('%d observations of %d stars; L = %.3f +- %.3f (stat), total +-%.3f'
      % (len(t), t.key.nunique(), L, SE_STAT, TOT))

# ---------------------------------------------------------------- sky frame
ra0, de0 = t.ra.mean(), t.dec.mean()
Xa = (t.ra.values-ra0)*np.cos(np.radians(de0))
Ya = t.dec.values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax_, *_ = np.linalg.lstsq(Aa, t.px.values, rcond=None)
ay_, *_ = np.linalg.lstsq(Aa, t.py.values, rcond=None)
Minv = np.linalg.inv(np.array([[ax_[0], ax_[1]], [ay_[0], ay_[1]]]))


def px_to_sky(px, py):
    v = Minv @ np.vstack([np.asarray(px, float) - ax_[2], np.asarray(py, float) - ay_[2]])
    return ra0 + v[0]/np.cos(np.radians(de0)), de0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    """Sensor-axis displacement (arcsec) -> sky displacement (arcsec of RA*cos(dec), Dec)."""
    v = Minv @ np.vstack([np.asarray(dx_as, float)/PS, np.asarray(dy_as, float)/PS])
    return v[0]*3600, v[1]*3600


_rt = np.hypot(*sensor_vec_to_sky(np.array([1.0]), np.array([0.0])))[0]
assert abs(_rt - 1.0) < 0.01, 'a 1 arcsec sensor displacement maps to %.3f arcsec' % _rt


# ---------------------------------------------------------------- 1. deflection vs radius
def deflection_chart(d, fname, title, note, unit='obs'):
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axhline(0, color='black', lw=1)
    for b in blocks:
        k = d.block.values == b
        if not k.any():
            continue
        ax.scatter(d.Rsun.values[k], d.rad.values[k], s=26, alpha=0.85,
                   color=BLOCK_COLOR[b], zorder=4,
                   label='%s (%d %s)' % (BLOCK_LABEL[b], int(k.sum()), unit))
    resid = d.rad.values - L*d.RS.values/d.R.values
    rms = float(np.sqrt(np.mean(resid**2)))
    for k in np.where(np.abs(resid) > 3.0*rms)[0]:
        ax.annotate('G %.1f (%+.1f$\\sigma$)' % (d.magV.values[k], resid[k]/rms),
                    (d.Rsun.values[k], d.rad.values[k]),
                    xytext=(6, -11), textcoords='offset points', fontsize=7.5, color='crimson')
    xx = np.linspace(1.85, d.Rsun.max()+0.4, 300)
    ax.fill_between(xx, (L-TOT)/xx, (L+TOT)/xx, color='black', alpha=0.10,
                    label='total $\\pm$%.2f" (stat %.3f + atm %.2f + order %.2f)'
                          % (TOT, SE_STAT, ATM_ERR, MODEL_ORDER))
    ax.plot(xx, L/xx, color='black', lw=2.2, label='pooled Method 2:  L = %.3f"' % L)
    ax.plot(xx, GR/xx, color='green', lw=1.5, label='Einstein  1.751"')
    ax.plot(xx, NEWTON/xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
    ax.plot(xx, PUB_L/xx, color='gray', lw=1.2, ls=':',
            label='published 1.839" (Dittrich et al. 2025)')
    ax.set_xlabel('radial position (solar radii)', fontsize=13)
    ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
    ax.set_title(title, fontsize=12)
    lo, hi = np.percentile(d.rad.values, [0.5, 99.5])
    ax.set_ylim(min(lo, -0.35) - 0.15, max(hi, L/1.9) + 0.35)
    ax.legend(fontsize=8.5, loc='upper right')
    fig.text(0.06, 0.015, note, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, fname)
    print('%s: N=%d rms %.3f"' % (fname, len(d), rms))


deflection_chart(t, 'record_deflection.png',
                 'Deflection vs radius \u2014 Mexico 2024 Station 1, pooled over every '
                 'observation, G $\\leq$ 13',
                 'one point per observation: a star seen in four blocks appears four times, '
                 'which is what the pooled fit weighs. Per-block offset, rotation and scale removed.')

per = t.groupby('key').agg(Rsun=('Rsun', 'mean'), rad=('rad', 'mean'), RS=('RS', 'mean'),
                           R=('R', 'mean'), magV=('magV', 'mean'), nblk=('nblk', 'first'),
                           block=('block', 'first')).reset_index()
deflection_chart(per, 'record_deflection_per_star.png',
                 'Deflection vs radius \u2014 Mexico 2024 Station 1, one point per star '
                 '(legibility copy of the record)',
                 'the same fit; each star\u2019s observations averaged for display only. '
                 'Colour is the first block that saw the star, so the counts are stars.',
                 unit='stars')

deflection_chart(t[t.nblk == 4], 'record_deflection_all4.png',
                 'Deflection vs radius \u2014 Mexico 2024 Station 1, stars seen in ALL FOUR '
                 'blocks (cross-check, not the record)',
                 'the old union-style admission rule, kept as a cross-check: L moves +0.03" '
                 'and 51 stars are lost. The record admits every observation.')

# ---------------------------------------------------------------- 2. the field
u = t.groupby('key').agg(px=('px', 'mean'), py=('py', 'mean'), vx=('vx', 'mean'),
                         vy=('vy', 'mean'), magV=('magV', 'mean'), nblk=('nblk', 'first'),
                         Rsun=('Rsun', 'mean')).reset_index()
fig, ax = plt.subplots(figsize=(11.5, 8))
sra, sdec = px_to_sky(u.px.values, u.py.values)
vra, vdec = sensor_vec_to_sky(u.vx.values, u.vy.values)
ARROW_DEG = 0.40
corners = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
ax.add_patch(Polygon(np.c_[corners[0], corners[1]], fill=False, color='gray', lw=1.2,
                     label='sensor footprint (ASI6200MM, 9576$\\times$6388)'))
ends_ra, ends_de = [], []
for k in range(len(u)):
    x0, y0 = sra[k], sdec[k]
    x1 = x0 + vra[k]*ARROW_DEG/np.cos(np.radians(de0))
    y1 = y0 + vdec[k]*ARROW_DEG
    ends_ra.append(x1); ends_de.append(y1)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>,head_width=0.20,head_length=0.40',
                                color=('tab:red' if u.nblk.values[k] < 4 else 'tab:blue'),
                                lw=1.2, shrinkA=0, shrinkB=0))
full = u.nblk.values == 4
ax.scatter(sra[full], sdec[full], s=18, color='tab:blue', zorder=5,
           label='seen in all four blocks (%d stars)' % int(full.sum()))
ax.scatter(sra[~full], sdec[~full], s=18, color='tab:red', zorder=5,
           label='seen in one to three blocks (%d)' % int((~full).sum()))
sun_ra, sun_dec = px_to_sky(np.array([t.sun_px.mean()]), np.array([t.sun_py.mean()]))
ax.add_patch(Circle((float(sun_ra[0]), float(sun_dec[0])), t.RS.mean()/3600, color='black',
                    zorder=3, label='the Sun, 1 R$_\\odot$ to scale'))
ax.add_patch(Circle((float(sun_ra[0]), float(sun_dec[0])), 2*t.RS.mean()/3600, fill=False,
                    color='gray', ls='--', lw=1.0, zorder=3, label='2 R$_\\odot$, the inner cut'))
lo_ra = min(min(corners[0]), min(ends_ra)) - 0.05
hi_ra = max(max(corners[0]), max(ends_ra)) + 0.05
lo_de = min(min(corners[1]), min(ends_de)) - 0.05
hi_de = max(max(corners[1]), max(ends_de)) + 0.05
for x1, y1 in zip(ends_ra, ends_de):
    assert lo_ra < x1 < hi_ra and lo_de < y1 < hi_de, 'an arrow leaves the axes'
ax.set_xlim(lo_ra, hi_ra); ax.set_ylim(lo_de, hi_de)
ax.set_aspect(1/np.cos(np.radians(de0)))
ax.set_xlabel('RA (degrees)', fontsize=12)
ax.set_ylabel('DEC (degrees)', fontsize=12)
ax.set_title('Displacement vectors \u2014 Mexico 2024 Station 1, pooled fit, G $\\leq$ 13',
             fontsize=12)
fig.text(0.06, 0.020, 'each arrow = the star\u2019s measured shift after subtracting that '
         'block\u2019s pointing offset, rotation and plate scale; deflection + measurement '
         'noise remain', fontsize=9)
ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.72))
star_rms = float(np.sqrt(np.mean(t.res.values**2)))
bar_deg = ARROW_DEG/np.cos(np.radians(de0))
for y_fr, ln, txt in ((0.42, 1.0, '1 arcsec of displacement'),
                      (0.32, star_rms, 'per-observation scatter (%.2f")' % star_rms)):
    ax.annotate('', xy=(1.04 + ln*bar_deg/(hi_ra-lo_ra), y_fr), xytext=(1.04, y_fr),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color='black', lw=3))
    ax.annotate(txt, (1.04, y_fr + 0.028), xycoords='axes fraction', fontsize=8)
fig.subplots_adjust(left=0.07, right=0.76, top=0.94, bottom=0.10)
save(fig, 'record_field.png')

# ---------------------------------------------------------------- 3. L and the plate scale
# One shared scale instead of four, so the ellipse is two-dimensional. The four per-block
# scales agree to 3 ppm, so this costs nothing and makes the covariance readable.
n = len(t); Z = np.zeros(n)
xs, ys = (t.px.values-NX/2)*PS, (t.py.values-NY/2)*PS
cx, cy, names = [], [], []
for b in blocks:
    m = (t.block.values == b).astype(float)
    cx += [m, Z, -ys*m]; cy += [Z, m, xs*m]; names += [b+':x0', b+':y0', b+':th']
cx += [xs, ux*t.RS.values/t.R.values]; cy += [ys, uy*t.RS.values/t.R.values]
names += ['S', 'L']
M = np.vstack([np.column_stack(cx), np.column_stack(cy)])
sc = np.sqrt((M**2).mean(0)); Mn = M/sc
b_ = np.concatenate([t.dx.values, t.dy.values])
c, *_ = np.linalg.lstsq(Mn, b_, rcond=None)
res = b_ - Mn@c
cov = ((res@res)/(len(b_)-Mn.shape[1]))*np.linalg.pinv(Mn.T@Mn)/np.outer(sc, sc)
c = c/sc
iS, iL = names.index('S'), names.index('L')
# the JOINT plate scale: what stage 2 used, corrected by the S fitted alongside L
ps2 = {}
for b in blocks:
    z = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', b, 'stage2_twopass', '**',
                                      'distortion_data*.zip'), recursive=True))[-1]
    ps2[b] = float(json.load(zipfile.ZipFile(z).open('distortion_results.txt'))['platescale (arcseconds/pixel)'])
joint = float(np.mean([ps2[b] for b in blocks])) - c[iS]*PS
joint_ppm = 1e6*(joint/PUB_SCALE - 1)
sS_ppm = 1e6*np.sqrt(cov[iS, iS])
C = np.array([[cov[iL, iL], cov[iL, iS]*1e6*PS/PUB_SCALE],
              [cov[iS, iL]*1e6*PS/PUB_SCALE, cov[iS, iS]*1e12*(PS/PUB_SCALE)**2]])
_infl = SE_STAT/np.sqrt(C[0, 0])
C = C*np.array([[_infl**2, _infl], [_infl, 1.0]])
fig, ax = plt.subplots(figsize=(9.5, 7))
vals, vecs = np.linalg.eigh(C)
ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
mu = np.array([c[iL], joint_ppm])
for k, ls in ((1, '-'), (2, '--')):
    ax.add_patch(Ellipse(mu, 2*k*np.sqrt(vals[1]), 2*k*np.sqrt(vals[0]), angle=ang, ls=ls,
                         fill=False, color='tab:blue', lw=1.6,
                         label='%d$\\sigma$ \u2014 Method 2 (scale fitted with L)' % k))
ax.scatter(*mu, marker='+', s=140, color='tab:blue', zorder=5)
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.axvline(NEWTON, color='orange', lw=1.5, ls='--', label='Newton 0.876"')
ax.axhline(0, color='gray', lw=1.0)
ax.axhspan(-1e6*PUB_SCALE_ERR/PUB_SCALE, 1e6*PUB_SCALE_ERR/PUB_SCALE, color='gray', alpha=0.18,
           label='published plate scale $\\pm$%.0f ppm (Dittrich et al. 2025)'
                 % (1e6*PUB_SCALE_ERR/PUB_SCALE))
_lines = [('Pooled Method 2:  L = %.3f $\\pm$ %.3f" (stat)' % (L, SE_STAT), 'tab:blue'),
          ('      $\\pm$ %.3f" with atmosphere %.2f and model order %.2f' % (TOT, ATM_ERR, MODEL_ORDER), 'tab:blue'),
          ('      L / L$_{Newton}$ = %.2f $\\pm$ %.2f' % (L/NEWTON, TOT/NEWTON), 'tab:blue'),
          ('Joint plate scale: %.6f "/px (%+.0f $\\pm$ %.0f ppm)' % (joint, joint_ppm, sS_ppm), 'black'),
          ('      (stage 2\u2019s scale corrected by the S fitted with L)', 'black'),
          ('Published: L = %.3f $\\pm$ %.3f", scale %.6f' % (PUB_L, PUB_L_ERR, PUB_SCALE), 'gray'),
          ('There is no Method 1 for this cell: the refocus left no', 'black'),
          ('      same-day scale that could be imported.', 'black')]
_pack = VPacker(children=[TextArea(x, textprops=dict(color=col, size=9.5)) for x, col in _lines],
                pad=0, sep=3, align='left')
_box = AnchoredOffsetbox(loc='lower left', child=_pack, pad=0.45, borderpad=0.6, frameon=True,
                         bbox_to_anchor=(0.0, 0.0), bbox_transform=ax.transAxes)
_box.patch.set(facecolor='white', edgecolor='gray', linewidth=0.9)
_box.set_zorder(6)
ax.add_artist(_box)
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('fitted plate scale (ppm from the published value)', fontsize=12)
ax.set_title('L and plate scale \u2014 Mexico 2024 Station 1, pooled fit, one shared scale',
             fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.autoscale_view(); ax.margins(0.22)
save(fig, 'record_covariance.png')
print('covariance: L %.3f +- %.3f (fit), joint scale %+.1f +- %.1f ppm from published'
      % (c[iL], np.sqrt(cov[iL, iL]), joint_ppm, sS_ppm))

# ---------------------------------------------------------------- 4. the annotated master
stk = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', '0p4s_1812', 'CENTROID_OUTPUT*',
                                    'STACKED_FLOAT*.fit')))
if stk:
    from astropy.io import fits as pyfits
    img = pyfits.getdata(stk[-1]).astype(np.float32)
    lo, hi = np.percentile(img, [5, 99.5])
    disp = np.arcsinh((np.clip(img, lo, hi) - lo)/max(hi - lo, 1)*30)
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
    d4 = t[t.block == '0p4s_1812']
    handles = []
    for k, (colr, lab) in ((d4.nblk.values == 4, ('yellow', 'also seen in the other three blocks')),
                           (d4.nblk.values < 4, ('red', 'seen in fewer than four blocks'))):
        for x0, y0 in zip(d4.px.values[k], d4.py.values[k]):
            ax.add_patch(Circle((x0, y0), 60, fill=False, color=colr, lw=1.2))
        handles.append(plt.Line2D([], [], color=colr, label='%s (%d)' % (lab, int(k.sum()))))
    ax.add_patch(Circle((d4.sun_px.mean(), d4.sun_py.mean()), 2*d4.RS.mean()/PS, fill=False,
                        color='cyan', lw=1.4, ls='--'))
    handles.append(plt.Line2D([], [], color='cyan', ls='--', label='2 R$_\\odot$, the inner cut'))
    ax.legend(handles=handles, fontsize=9, loc='upper left', bbox_to_anchor=(0.0, -0.09),
              borderaxespad=0, frameon=True)
    ax.set_title('The 0.4 s block (123 frames, occulted and coronal-subtracted) \u2014 '
                 'Mexico 2024 Station 1', fontsize=11)
    ax.set_xlabel('px'); ax.set_ylabel('py')
    fig.subplots_adjust(bottom=0.20)
    save(fig, 'master_0p4s_annotated.png')

# ---------------------------------------------------------------- 5. the tables
t.to_csv(os.path.join(OUT, 'station1_star_table.csv'), index=False)
rec = dict(cell='Mexico 2024 Station 1', estimator='pooled Method 2, every observation',
           reference='quintic, seventeen zenith fields, windowed', stage2='two-pass, scale fitted',
           calibration='dark + flat (the published 2024 choice)', magnitude_cut=13.0,
           radius_cut=[2.0, 9.0], observations=int(len(t)), stars=int(t.key.nunique()),
           L=L, sigma_stat=SE_STAT, sigma_atmosphere=ATM_ERR, sigma_model_order=MODEL_ORDER,
           sigma_total=TOT, scale_share_of_stat=summary['scale_share'],
           joint_platescale=joint, joint_platescale_ppm_from_published=joint_ppm,
           GR=GR, NEWTON=NEWTON, L_over_Newton=L/NEWTON, L_over_Newton_err=TOT/NEWTON,
           sigma_from_GR=abs(L-GR)/TOT, sigma_from_Newton=abs(L-NEWTON)/TOT,
           published=dict(source='Dittrich et al. 2025', L=PUB_L, L_err=PUB_L_ERR,
                          platescale=PUB_SCALE, platescale_err=PUB_SCALE_ERR),
           per_block={b: summary['blocks'][b] for b in blocks})
json.dump(rec, open(os.path.join(OUT, 'record_summary.json'), 'w'), indent=1)
print('L/L_Newton = %.2f +- %.2f; GR at %.2f sigma, Newton at %.2f sigma'
      % (L/NEWTON, TOT/NEWTON, abs(L-GR)/TOT, abs(L-NEWTON)/TOT))

# ---------------------------------------------------------------- 6. the record copy
if os.environ.get('MX24_COPY_RECORD') == '1':
    import filecmp
    os.makedirs(RECORD, exist_ok=True)
    extra = {'atmosphere_night_maps.png': os.path.join(REC, 'atmosphere_night_maps.png'),
             'atmosphere_maps_stats.csv': os.path.join(REC, 'atmosphere_maps_stats.csv'),
             'zenith_floor.csv': os.path.join(REC, 'zenith_floor.csv'),
             'zenith_nulls.csv': os.path.join(REC, 'zenith_nulls.csv'),
             'admission_magnitude_vet_grid.csv': os.path.join(REC, 'pooled_fit', 'grid_twopass.csv'),
             'calibration_arms.csv': os.path.join(REC, 'eclipse_caldecomp', 'caldecomp_arms.csv'),
             'flat_mechanism.csv': os.path.join(REC, 'eclipse_caldecomp', 'flat_mechanism.csv')}
    produced = ['record_deflection.png', 'record_deflection_per_star.png',
                'record_deflection_all4.png', 'record_field.png', 'record_covariance.png',
                'master_0p4s_annotated.png', 'station1_star_table.csv', 'record_summary.json']
    src = {f: os.path.join(OUT, f) for f in produced}
    src.update(extra)
    # only files that actually differ are superseded; re-running with the same output must
    # not bury identical copies under a new date
    old = [f for f in os.listdir(RECORD)
           if f in src and os.path.exists(src[f])
           and not filecmp.cmp(src[f], os.path.join(RECORD, f), shallow=False)]
    if old:
        sup = os.path.join(RECORD, 'superseded_' + pd.Timestamp.now().strftime('%Y-%m-%d_%H%M'))
        os.makedirs(sup, exist_ok=True)
        for f in old:
            shutil.move(os.path.join(RECORD, f), os.path.join(sup, f))
        print('superseded %d file(s) -> %s' % (len(old), sup))
    for f, p in src.items():
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RECORD, f))
    print('record set ->', RECORD)
print('charts ->', OUT)
