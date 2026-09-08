"""The record chart set for Mexico 2024 Station 2, in the cell-2 house style.

Douglas asked for a reduced set: the deflection chart, the field chart, the covariance chart
carrying BOTH methods, and one annotated master per tier. Written to
`D:\\MEE2024 output\\MEE_output\\RECORD\\mexico2024st2`.

Station 2 is the station cell 2 could not be: it has a real eclipse-day bracket, right at
18:10:32-18:11:21 and left at 18:14:05-18:14:57, so unlike Station 1 it can be reduced BOTH ways.

  Method 1  the eclipse field against the L/R bracket mean, scale imported (Bruns' construction)
  Method 2  against the 15-field cubic zenith reference, scale fitted alongside L

Both tiers are pooled as cell 2 pools its four: every observation is a row, each tier carries its
own offset, rotation and scale, and one L is shared.

  .venv/Scripts/python.exe tools/matrix_station2/s2_charts_record.py
  MX24ST2_COPY_RECORD=1 .venv/Scripts/python.exe tools/matrix_station2/s2_charts_record.py
"""
import glob
import json
import os
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Circle, Ellipse, Polygon

REV = 'rev01'
OUT = r"D:/MEE2024 output/MEE_output/station2_transfer"
CHARTS = os.path.join(OUT, 'charts')
VER = os.path.join(CHARTS, 'chart_versions')
RECORD = r"D:/MEE2024 output/MEE_output/RECORD/mexico2024st2"
os.makedirs(CHARTS, exist_ok=True); os.makedirs(VER, exist_ok=True)

NX, NY, PS = 4656, 3520, 1.8672511
GR, NEWTON = 1.7512, 0.8756
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2
MAGCUT, RMIN, RMAX = 13.0, 2.0, 5.5
ATM_ERR = 0.12                       # Station 2's own zenith null, Method 2
TIERS = (('100ms', '0.100 s, 18:11:30-18:12:44', 'tab:blue'),
         ('075ms', '0.075 s, 18:12:45-18:13:55', 'tab:orange'))
SUB = {'m2': {'100ms': 'eclipse/100ms/stage2', '075ms': 'eclipse/075ms/stage2_rmt36'},
       'm1': {'100ms': 'eclipse/100ms/stage2_method1', '075ms': 'eclipse/075ms/stage2_method1'}}


def save(fig, name):
    fig.savefig(os.path.join(CHARTS, name), dpi=140)
    fig.savefig(os.path.join(VER, REV + '_' + name), dpi=140)
    plt.close(fig)


def load(method):
    frames = []
    for tag, _, _ in TIERS:
        f = glob.glob(os.path.join(OUT, SUB[method][tag], '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        if not f:
            continue
        d = pd.read_csv(f[0])
        d = d[d['magV'] <= MAGCUT].copy()
        d['tier'] = tag
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D['rx'] = (D['px'] - SUNPX) * PS
    D['ry'] = (D['py'] - SUNPY) * PS
    D['R'] = np.hypot(D['rx'], D['ry'])
    D['Rsun'] = D['R'] / R_SUN_AS
    return D[(D['Rsun'] >= RMIN) & (D['Rsun'] <= RMAX)].reset_index(drop=True)


def design(d, with_scale):
    n = len(d); Z = np.zeros(n)
    px, py = d['px'].values, d['py'].values
    ux, uy = d['rx'].values / d['R'].values, d['ry'].values / d['R'].values
    cx, cy, lab = [], [], []
    for tag, _, _ in TIERS:
        m = (d['tier'] == tag).values.astype(float)
        cx += [m, Z, -m * (py - NY / 2) * PS]
        cy += [Z, m, m * (px - NX / 2) * PS]
        lab += ['N1_' + tag, 'N2_' + tag, 'Th_' + tag]
        if with_scale:
            cx.append(m * (px - NX / 2) * PS); cy.append(m * (py - NY / 2) * PS)
            lab.append('S_' + tag)
    cx.append(ux * R_SUN_AS / d['R'].values); cy.append(uy * R_SUN_AS / d['R'].values)
    lab.append('L')
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), lab


def solve(d, with_scale):
    A, lab = design(d, with_scale)
    y = np.concatenate([d['dx_arcsec'].values, d['dy_arcsec'].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    n = len(d)
    return c, lab, resid[:n], resid[n:], A


def bootstrap(d, with_scale, draws=600, seed=7):
    rng = np.random.default_rng(seed)
    ids = d['ID'].unique(); out = []
    for _ in range(draws):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([d[d['ID'] == i] for i in pick], ignore_index=True)
        if s['tier'].nunique() < len(TIERS):
            continue
        try:
            c, lab, *_ = solve(s, with_scale)
            out.append(c[lab.index('L')])
        except Exception:
            pass
    return float(np.std(out, ddof=1)), len(out)


# ---------------------------------------------------------------- the two reductions
D2 = load('m2')
c2, lab2, rx2, ry2, A2 = solve(D2, True)
L2 = c2[lab2.index('L')]
S2, n2 = bootstrap(D2, True)

D1 = load('m1')
c1, lab1, rx1, ry1, A1 = solve(D1, False)
L1 = c1[lab1.index('L')]
S1, n1 = bootstrap(D1, False)

IMPORTED = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, 'bracket_cubic', n, '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for n in ('right', 'left')]))
BRK_SPREAD = abs(json.load(open(glob.glob(os.path.join(OUT, 'bracket_cubic', 'left', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)']
                 - json.load(open(glob.glob(os.path.join(OUT, 'bracket_cubic', 'right', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)'])

TOT2 = float(np.hypot(S2, ATM_ERR))
print('Method 2: L = %+.3f +- %.3f (stat, %d draws), %d obs of %d stars'
      % (L2, S2, n2, len(D2), D2.ID.nunique()))
print('Method 1: L = %+.3f +- %.3f (stat, %d draws), imported scale %.7f "/px (L-R %.1f ppm)'
      % (L1, S1, n1, IMPORTED, 1e6 * BRK_SPREAD / IMPORTED))

# per-observation radial deflection, with the nuisances removed
ux, uy = D2.rx.values / D2.R.values, D2.ry.values / D2.R.values
D2['rad'] = (rx2 * ux + ry2 * uy) + L2 * R_SUN_AS / D2.R.values
D2['vx'] = rx2 + L2 * R_SUN_AS / D2.R.values * ux
D2['vy'] = ry2 + L2 * R_SUN_AS / D2.R.values * uy
D2['res'] = np.hypot(rx2, ry2)

# ---------------------------------------------------------------- 1. deflection vs radius
fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
dots = []
for tag, lab, colr in TIERS:
    k = (D2.tier == tag).values
    dots.append(ax.scatter(D2.Rsun.values[k], D2.rad.values[k], s=34, alpha=0.85, color=colr,
                           zorder=4, label='%s (%d obs)' % (lab, int(k.sum()))))
xx = np.linspace(RMIN + 0.9, D2.Rsun.max() + 0.4, 300)
band = ax.fill_between(xx, (L2 - TOT2) / xx, (L2 + TOT2) / xx, color='black', alpha=0.10,
                       label='total $\\pm$%.2f" (stat %.2f + atmosphere %.2f)' % (TOT2, S2, ATM_ERR))
ln1, = ax.plot(xx, L2 / xx, color='black', lw=2.2, label='pooled Method 2:  L = %.2f"' % L2)
ln2, = ax.plot(xx, GR / xx, color='green', lw=1.5, label='Einstein  1.751"')
ln3, = ax.plot(xx, NEWTON / xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
ax.set_title('Deflection vs radius \u2014 Mexico 2024 Station 2, pooled over both tiers, '
             'G $\\leq$ 13', fontsize=12)
lo, hi = np.percentile(D2.rad.values, [1, 99])
ax.set_ylim(min(lo, -0.6) - 0.2, max(hi, 1.0) + 0.3)
first = ax.legend(handles=dots, fontsize=9, loc='lower left', title='exposure tiers (%d observations)' % len(D2),
                  title_fontsize=9)
ax.add_artist(first)
ax.legend(handles=[ln1, ln2, ln3, band], fontsize=9, loc='upper right')
fig.text(0.06, 0.015, 'one point per observation: each star appears twice, once per tier. The '
         'saturated blob and its exclusion ring remove everything inside %.1f R$_\\odot$, which '
         'is where the deflection is largest.' % D2.Rsun.min(), fontsize=8.5)
fig.tight_layout(rect=(0, 0.035, 1, 1))
save(fig, 'record_deflection.png')

# ---------------------------------------------------------------- 2. the field
u = D2.groupby('ID').agg(px=('px', 'mean'), py=('py', 'mean'), vx=('vx', 'mean'),
                         vy=('vy', 'mean'), Rsun=('Rsun', 'mean')).reset_index()
fig, ax = plt.subplots(figsize=(11.5, 8))
ARROW = 900.0                     # px drawn per arcsec of displacement
ax.add_patch(Polygon(np.c_[[0, NX, NX, 0], [0, 0, NY, NY]], fill=False, color='gray', lw=1.2,
                     label='sensor footprint'))
for k in range(len(u)):
    ax.annotate('', xy=(u.px.values[k] + u.vx.values[k] * ARROW / PS,
                        u.py.values[k] + u.vy.values[k] * ARROW / PS),
                xytext=(u.px.values[k], u.py.values[k]),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                color='tab:blue', lw=1.3, shrinkA=0, shrinkB=0))
ax.scatter(u.px, u.py, s=26, color='tab:blue', zorder=5, label='%d stars (both tiers)' % len(u))
ax.add_patch(Circle((SUNPX, SUNPY), R_SUN_AS / PS, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
ax.add_patch(Circle((SUNPX, SUNPY), 2 * R_SUN_AS / PS, fill=False, color='gray', ls='--', lw=1.0,
                    zorder=3, label='2 R$_\\odot$'))
ax.set_xlim(-400, NX + 400); ax.set_ylim(NY + 400, -400)
ax.set_aspect('equal')
ax.set_xlabel('px', fontsize=12); ax.set_ylabel('py', fontsize=12)
ax.set_title('Displacement vectors (%d stars) \u2014 Mexico 2024 Station 2, pooled fit, '
             'G $\\leq$ 13' % len(u), fontsize=12)
star_rms = float(np.sqrt(np.mean(D2.res.values ** 2)))
for y_fr, ln, txt in ((0.40, 1.0, '1 arcsec of displacement'),
                      (0.30, star_rms, 'per-observation scatter (%.2f")' % star_rms)):
    ax.annotate('', xy=(1.04 + ln * ARROW / PS / (NX + 800), y_fr), xytext=(1.04, y_fr),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color='black', lw=3))
    ax.annotate(txt, (1.04, y_fr + 0.03), xycoords='axes fraction', fontsize=8)
ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.70))
fig.text(0.06, 0.02, 'each arrow = the star\u2019s measured shift after subtracting that tier\u2019s '
         'pointing offset, rotation and plate scale; deflection + measurement noise remain',
         fontsize=9)
fig.subplots_adjust(left=0.07, right=0.76, top=0.94, bottom=0.10)
save(fig, 'record_field.png')

# ---------------------------------------------------------------- 3. covariance, BOTH methods
sig2 = float(np.sqrt(np.mean(np.concatenate([rx2, ry2]) ** 2)))
cov2 = sig2 ** 2 * np.linalg.pinv(A2.T @ A2)
iL, iS = lab2.index('L'), lab2.index('S_100ms')
joint = PS * (1 + c2[iS])
C = np.array([[cov2[iL, iL], cov2[iL, iS] * PS], [cov2[iS, iL] * PS, cov2[iS, iS] * PS ** 2]])
C[0, 0] = S2 ** 2                                   # carry the bootstrap sigma the record quotes
RHO = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
fig, ax = plt.subplots(figsize=(9.8, 7))
vals, vecs = np.linalg.eigh(C)
ax.add_patch(Ellipse((c2[iL], joint), 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]),
                     angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])), fill=False,
                     color='tab:blue', lw=1.8, label='1$\\sigma$ \u2014 Method 2 (scale fitted with L)'))
ax.scatter(c2[iL], joint, marker='+', s=150, color='tab:blue', zorder=5)
ax.errorbar([L1], [IMPORTED], xerr=[S1], yerr=[BRK_SPREAD / 2], fmt='s', color='tab:red',
            ms=7, capsize=4, lw=1.6, zorder=5,
            label='Method 1 \u2014 scale imported from the L/R bracket')
ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
ax.axvline(NEWTON, color='orange', lw=1.5, ls='--', label='Newton 0.876"')
_lines = [('Method 2:  L = %+.2f $\\pm$ %.2f" (stat), $\\pm$%.2f" with atmosphere %.2f'
           % (L2, S2, TOT2, ATM_ERR), 'tab:blue'),
          ('      fitted plate scale %.6f "/px' % joint, 'tab:blue'),
          ('      correlation L vs plate scale = %+.2f' % RHO, 'tab:blue'),
          ('Method 1:  L = %+.2f $\\pm$ %.2f" (stat)' % (L1, S1), 'tab:red'),
          ('      imported %.6f "/px, the L/R mean (L\u2212R %.0f ppm)'
           % (IMPORTED, 1e6 * BRK_SPREAD / IMPORTED), 'tab:red'),
          ('from %d observations of %d stars, both tiers pooled'
           % (len(D2), D2.ID.nunique()), 'black')]
_box = AnchoredOffsetbox(loc='lower left', pad=0.45, borderpad=0.6, frameon=True,
                         child=VPacker(children=[TextArea(x, textprops=dict(color=col, size=9.5))
                                                 for x, col in _lines], pad=0, sep=3, align='left'),
                         bbox_to_anchor=(0.0, 0.0), bbox_transform=ax.transAxes)
_box.patch.set(facecolor='white', edgecolor='gray', linewidth=0.9); _box.set_zorder(6)
ax.add_artist(_box)
ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
ax.set_ylabel('plate scale (arcsec per pixel)', fontsize=12)
ax.ticklabel_format(axis='y', useOffset=False, style='plain')
ax.set_title('L and plate scale, both methods \u2014 Mexico 2024 Station 2', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.autoscale_view(); ax.margins(0.30)
save(fig, 'record_covariance.png')
print('covariance: Method 2 rho %+.2f, joint scale %.7f; Method 1 imported %.7f' % (RHO, joint, IMPORTED))

# ---------------------------------------------------------------- 4. the annotated masters
from astropy.io import fits as pyfits
for tag, lab, _ in TIERS:
    stk = sorted(glob.glob(os.path.join(OUT, 'eclipse', tag, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit')))
    if not stk:
        stk = sorted(glob.glob(os.path.join(OUT, 'eclipse', tag, 'CENTROID_OUTPUT*', 'STACKED2*.fit')))
    if not stk:
        print('%s: no stacked image' % tag); continue
    zz = glob.glob(os.path.join(OUT, 'eclipse', tag, 'centroid_data*.zip'))
    nfr = json.load(zipfile.ZipFile(zz[0]).open('results.txt')).get('#frames stacked') if zz else '?'
    img = pyfits.getdata(stk[-1]).astype(np.float32)
    lo, hi = np.percentile(img, [5, 99.5])
    disp = np.arcsinh((np.clip(img, lo, hi) - lo) / max(hi - lo, 1) * 30)
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ax.imshow(disp, cmap='gray', origin='upper', interpolation='nearest')
    dt = D2[D2.tier == tag]
    for x0, y0 in zip(dt.px.values, dt.py.values):
        ax.add_patch(Circle((x0, y0), 55, fill=False, color='yellow', lw=1.3))
    ax.add_patch(Circle((SUNPX, SUNPY), 2 * R_SUN_AS / PS, fill=False, color='cyan', lw=1.4, ls='--'))
    ax.legend(handles=[plt.Line2D([], [], color='yellow', label='matched and fitted (%d)' % len(dt)),
                       plt.Line2D([], [], color='cyan', ls='--', label='2 R$_\\odot$')],
              fontsize=9, loc='upper left', bbox_to_anchor=(0.0, -0.09), borderaxespad=0, frameon=True)
    ax.set_title('The %s tier (%s frames, occulted and coronal-subtracted) \u2014 Mexico 2024 Station 2'
                 % (lab, nfr), fontsize=11)
    ax.set_xlabel('px'); ax.set_ylabel('py')
    fig.subplots_adjust(bottom=0.20)
    save(fig, 'master_%s_annotated.png' % tag)

# ---------------------------------------------------------------- 5. summary + record copy
rec = dict(cell='Mexico 2024 Station 2', estimator='pooled over both tiers, every observation',
           reference='cubic, fifteen zenith fields, moments+annular, 0.5 arcsec gate',
           bracket='right 18:10:32-18:11:21, left 18:14:05-18:14:57, ends of totality trimmed',
           observations=int(len(D2)), stars=int(D2.ID.nunique()),
           method2=dict(L=L2, sigma_stat=S2, sigma_atmosphere=ATM_ERR, sigma_total=TOT2,
                        joint_platescale=joint, corr_L_platescale=float(RHO)),
           method1=dict(L=L1, sigma_stat=S1, imported_platescale=IMPORTED,
                        bracket_LR_ppm=1e6 * BRK_SPREAD / IMPORTED),
           radius_range=[float(D2.Rsun.min()), float(D2.Rsun.max())],
           GR=GR, NEWTON=NEWTON,
           sigma_from_GR_method2=abs(L2 - GR) / TOT2, sigma_from_Newton_method2=abs(L2 - NEWTON) / TOT2)
json.dump(rec, open(os.path.join(CHARTS, 'record_summary.json'), 'w'), indent=1)
D2.to_csv(os.path.join(CHARTS, 'station2_star_table.csv'), index=False)

if os.environ.get('MX24ST2_COPY_RECORD') == '1':
    os.makedirs(RECORD, exist_ok=True)
    for f in ('record_deflection.png', 'record_field.png', 'record_covariance.png',
              'master_100ms_annotated.png', 'master_075ms_annotated.png',
              'record_summary.json', 'station2_star_table.csv'):
        p = os.path.join(CHARTS, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RECORD, f))
    print('record set ->', RECORD)
print('charts ->', CHARTS)
