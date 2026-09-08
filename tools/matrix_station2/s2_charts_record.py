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
import datetime
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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.analysis_window import WINDOWS
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Circle, Ellipse, Polygon

REV = 'rev03'
OUT = r"D:/MEE2024 output/MEE_output/station2_transfer"
CHARTS = os.path.join(OUT, 'charts')
VER = os.path.join(CHARTS, 'chart_versions')
RECORD = r"D:/MEE2024 output/MEE_output/RECORD/mexico2024st2"
os.makedirs(CHARTS, exist_ok=True); os.makedirs(VER, exist_ok=True)

NX, NY, PS = 4656, 3520, 1.8672511
GR, NEWTON = 1.7512, 0.8756
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2
_W = WINDOWS['mexico2024_station2']          # 2-10 R_sun, G <= 13; see the registry
MAGCUT, RMIN, RMAX = _W.mag, _W.rmin, _W.rmax
ATM_ERR = 0.12                       # Station 2's own zenith null, Method 2
TIERS = (('100ms', '0.100 s, 18:11:30-18:12:44', 'tab:blue'),
         ('075ms', '0.075 s, 18:12:45-18:13:55', 'tab:orange'))
SUB = {'m2': {'100ms': 'eclipse/100ms/stage2', '075ms': 'eclipse/075ms/stage2_rmt36'},
       'm1': {'100ms': 'eclipse/100ms/stage2_method1_quadfree',
       '075ms': 'eclipse/075ms/stage2_method1_quadfree'}}


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
        # the residual table carries no sky coordinates; take them from the run's own matched
        # catalogue, so the field charts can be drawn in RA/DEC as cell 2's are
        c = glob.glob(os.path.join(OUT, SUB[method][tag], '**', 'CATALOGUE_MATCHED_ERRORS.csv'),
                      recursive=True)
        cat = pd.read_csv(c[0])[['ID', 'RA(catalog)', 'DEC(catalog)']].rename(
            columns={'RA(catalog)': 'ra', 'DEC(catalog)': 'dec'})
        d = d.merge(cat.drop_duplicates('ID'), on='ID', how='left')
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
    OUT, 'bracket_quadfree', n, '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for n in ('right', 'left')]))
BRK_SPREAD = abs(json.load(open(glob.glob(os.path.join(OUT, 'bracket_quadfree', 'left', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)']
                 - json.load(open(glob.glob(os.path.join(OUT, 'bracket_quadfree', 'right', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)'])

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
# The sky frame, exactly as cell 2 builds it (s1_charts_record.py): an affine from catalogue
# RA/DEC to pixels, fitted on the matched stars and inverted, so a sensor-axis displacement in
# arcsec can be drawn on RA/DEC axes.
RA0, DE0 = float(D2.ra.mean()), float(D2['dec'].mean())
_Xa = (D2.ra.values - RA0) * np.cos(np.radians(DE0))
_Ya = D2['dec'].values - DE0
_Aa = np.c_[_Xa, _Ya, np.ones_like(_Xa)]
_axc, *_ = np.linalg.lstsq(_Aa, D2.px.values, rcond=None)
_ayc, *_ = np.linalg.lstsq(_Aa, D2.py.values, rcond=None)
MINV = np.linalg.inv(np.array([[_axc[0], _axc[1]], [_ayc[0], _ayc[1]]]))


def px_to_sky(px, py):
    v = MINV @ np.vstack([np.asarray(px, float) - _axc[2], np.asarray(py, float) - _ayc[2]])
    return RA0 + v[0] / np.cos(np.radians(DE0)), DE0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    """Sensor-axis displacement (arcsec) -> sky displacement (arcsec of RA*cos(dec), Dec)."""
    v = MINV @ np.vstack([np.asarray(dx_as, float) / PS, np.asarray(dy_as, float) / PS])
    return v[0] * 3600, v[1] * 3600


def field_chart(u, fname, title, note, star_rms, colour='tab:blue', label=None):
    """u: one row per star with px, py, vx, vy (arcsec, nuisances removed). RA/DEC axes."""
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ARROW_DEG = 0.40                  # degrees drawn per arcsec of displacement
    sra, sdec = px_to_sky(u.px.values, u.py.values)
    vra, vdec = sensor_vec_to_sky(u.vx.values, u.vy.values)
    cor_ra, cor_de = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
    sun_ra, sun_de = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
    sun_ra, sun_de = float(sun_ra[0]), float(sun_de[0])
    ax.add_patch(Polygon(np.c_[cor_ra, cor_de], fill=False, color='gray', lw=1.2,
                         label='sensor footprint'))
    ends_ra, ends_de = [], []
    for k in range(len(u)):
        x1 = sra[k] + vra[k] * ARROW_DEG / np.cos(np.radians(DE0))
        y1 = sdec[k] + vdec[k] * ARROW_DEG
        ends_ra.append(x1); ends_de.append(y1)
        ax.annotate('', xy=(x1, y1), xytext=(sra[k], sdec[k]),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.45',
                                    color=colour, lw=1.3, shrinkA=0, shrinkB=0))
    ax.scatter(sra, sdec, s=26, color=colour, zorder=5, label=label or '%d stars' % len(u))
    ax.add_patch(Circle((sun_ra, sun_de), R_SUN_AS / 3600, color='black', zorder=3,
                        label='the Sun, 1 R$_\\odot$ to scale'))
    ax.add_patch(Circle((sun_ra, sun_de), 2 * R_SUN_AS / 3600, fill=False, color='gray', ls='--',
                        lw=1.0, zorder=3, label='2 R$_\\odot$'))
    lo_ra = min(cor_ra.min(), min(ends_ra), sun_ra - 2 * R_SUN_AS / 3600) - 0.05
    hi_ra = max(cor_ra.max(), max(ends_ra), sun_ra + 2 * R_SUN_AS / 3600) + 0.05
    lo_de = min(cor_de.min(), min(ends_de), sun_de - 2 * R_SUN_AS / 3600) - 0.05
    hi_de = max(cor_de.max(), max(ends_de), sun_de + 2 * R_SUN_AS / 3600) + 0.05
    for x1, y1 in zip(ends_ra, ends_de):
        assert lo_ra < x1 < hi_ra and lo_de < y1 < hi_de, 'an arrow leaves the axes'
    ax.set_xlim(hi_ra, lo_ra)          # RA increases to the left, as cell 2 draws it
    ax.set_ylim(lo_de, hi_de)
    ax.set_aspect(1 / np.cos(np.radians(DE0)))
    ax.set_xlabel('RA (degrees)', fontsize=12); ax.set_ylabel('DEC (degrees)', fontsize=12)
    ax.set_title(title, fontsize=12)
    bar_deg = ARROW_DEG / np.cos(np.radians(DE0))
    for y_fr, ln, txt in ((0.40, 1.0, '1 arcsec of displacement'),
                          (0.30, star_rms, 'scatter (%.2f")' % star_rms)):
        ax.annotate('', xy=(1.04 + ln * bar_deg / (hi_ra - lo_ra), y_fr), xytext=(1.04, y_fr),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', color='black', lw=3))
        ax.annotate(txt, (1.04, y_fr + 0.03), xycoords='axes fraction', fontsize=8)
    ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.70))
    fig.text(0.06, 0.02, note, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.76, top=0.94, bottom=0.10)
    save(fig, fname)


NOTE_TIER = ('each arrow = the star\u2019s measured shift after subtracting that tier\u2019s '
             'pointing offset, rotation and plate scale; deflection + measurement noise remain')
u = D2.groupby('ID').agg(px=('px', 'mean'), py=('py', 'mean'), vx=('vx', 'mean'),
                         vy=('vy', 'mean'), Rsun=('Rsun', 'mean'), n=('tier', 'nunique')).reset_index()
field_chart(u, 'record_field.png',
            'Displacement vectors (%d stars) \u2014 Mexico 2024 Station 2, pooled fit, G $\\leq$ 13' % len(u),
            NOTE_TIER, float(np.sqrt(np.mean(D2.res.values ** 2))), label='%d stars (both tiers pooled)' % len(u))
for tag, lab, colr in TIERS:
    dt = D2[D2.tier == tag]
    ut = dt.groupby('ID').agg(px=('px', 'mean'), py=('py', 'mean'), vx=('vx', 'mean'),
                              vy=('vy', 'mean'), Rsun=('Rsun', 'mean')).reset_index()
    field_chart(ut, 'record_field_%s.png' % tag,
                'Displacement vectors, the %s tier (%d stars) \u2014 Mexico 2024 Station 2' % (lab, len(ut)),
                NOTE_TIER, float(np.sqrt(np.mean(dt.res.values ** 2))), colour=colr,
                label='%d stars, %s' % (len(ut), lab))

# ---------------------------------------------------------------- 2b. both tiers, AVERAGED
# Stars seen in both tiers only; each star's two nuisance-removed vectors (vx, vy) averaged into
# one point; L refitted on those points with offset, rotation and scale free again.
both = u[u.n == 2].copy()
def design_single(d):
    n = len(d); Z = np.zeros(n)
    px, py = d['px'].values, d['py'].values
    rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS; R = np.hypot(rx, ry)
    cx = [np.ones(n), Z, -(py - NY / 2) * PS, (px - NX / 2) * PS, rx / R * R_SUN_AS / R]
    cy = [Z, np.ones(n), (px - NX / 2) * PS, (py - NY / 2) * PS, ry / R * R_SUN_AS / R]
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), R
def solve_single(d):
    A, R = design_single(d)
    y = np.concatenate([d['vx'].values, d['vy'].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c; n = len(d)
    return c[-1], resid[:n], resid[n:], R
L_avg, rxa, rya, R_avg = solve_single(both)
rng = np.random.default_rng(11)
bl = []
for _ in range(600):
    pick = rng.choice(len(both), size=len(both), replace=True)
    try:
        bl.append(solve_single(both.iloc[pick])[0])
    except Exception:
        pass
S_avg = float(np.std(bl, ddof=1)); TOT_avg = float(np.hypot(S_avg, ATM_ERR))
uxa, uya = (both.px.values - SUNPX) * PS / R_avg, (both.py.values - SUNPY) * PS / R_avg
both['rad'] = (rxa * uxa + rya * uya) + L_avg * R_SUN_AS / R_avg
both['Rsun'] = R_avg / R_SUN_AS
print('Both tiers, averaged: L = %+.3f +- %.3f (stat, %d draws), %d stars'
      % (L_avg, S_avg, len(bl), len(both)))

fig, ax = plt.subplots(figsize=(10, 6.8))
ax.axhline(0, color='black', lw=1)
dot = ax.scatter(both.Rsun.values, both.rad.values, s=40, color='tab:purple', zorder=4,
                 label='%d stars, each the mean of its two tiers' % len(both))
xx = np.linspace(RMIN + 0.9, both.Rsun.max() + 0.4, 300)
band = ax.fill_between(xx, (L_avg - TOT_avg) / xx, (L_avg + TOT_avg) / xx, color='black', alpha=0.10,
                       label='total $\\pm$%.2f" (stat %.2f + atmosphere %.2f)' % (TOT_avg, S_avg, ATM_ERR))
ln1, = ax.plot(xx, L_avg / xx, color='black', lw=2.2, label='averaged fit:  L = %.2f"' % L_avg)
ln0, = ax.plot(xx, L2 / xx, color='tab:blue', lw=1.2, ls=':', label='pooled fit:  L = %.2f"' % L2)
ln2, = ax.plot(xx, GR / xx, color='green', lw=1.5, label='Einstein  1.751"')
ln3, = ax.plot(xx, NEWTON / xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
ax.set_xlabel('radial position (solar radii)', fontsize=13)
ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
ax.set_title('Deflection vs radius \u2014 Mexico 2024 Station 2, stars in BOTH tiers, positions averaged',
             fontsize=12)
lo, hi = np.percentile(both.rad.values, [1, 99])
ax.set_ylim(min(lo, -0.6) - 0.2, max(hi, 1.0) + 0.3)
first = ax.legend(handles=[dot], fontsize=9, loc='lower left'); ax.add_artist(first)
ax.legend(handles=[ln1, ln0, ln2, ln3, band], fontsize=9, loc='upper right')
fig.text(0.06, 0.012, 'one point per star: its two tier observations, each with that tier’s offset, '
         'rotation and scale removed, averaged before the fit.' + chr(10) + 'Stars seen in one tier only are excluded '
         '(two here: one per tier). L is refitted on the averaged points with offset, rotation and scale free.',
         fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'record_deflection_both.png')
both['vx'] = rxa + L_avg * R_SUN_AS / R_avg * uxa
both['vy'] = rya + L_avg * R_SUN_AS / R_avg * uya
field_chart(both, 'record_field_both.png',
            'Displacement vectors, stars in BOTH tiers, positions averaged (%d stars) \u2014 Mexico 2024 Station 2' % len(both),
            'each arrow = the mean of the star\u2019s two tier shifts (each tier\u2019s offset, rotation and '
            'scale removed); deflection + measurement noise remain',
            float(np.sqrt(np.mean(rxa ** 2 + rya ** 2))), colour='tab:purple',
            label='%d stars in both tiers, averaged' % len(both))

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
rec = dict(rev=REV, cell='Mexico 2024 Station 2', estimator='pooled over both tiers, every observation',
           reference='cubic, fifteen zenith fields, moments+annular, 0.5 arcsec gate',
           bracket='right 18:10:32-18:11:21, left 18:14:05-18:14:57, ends of totality trimmed',
           observations=int(len(D2)), stars=int(D2.ID.nunique()),
           method2=dict(L=L2, sigma_stat=S2, sigma_atmosphere=ATM_ERR, sigma_total=TOT2,
                        joint_platescale=joint, corr_L_platescale=float(RHO)),
           method1=dict(L=L1, sigma_stat=S1, imported_platescale=IMPORTED,
                        bracket_LR_ppm=1e6 * BRK_SPREAD / IMPORTED),
           both_tiers_averaged=dict(L=L_avg, sigma_stat=S_avg, sigma_total=TOT_avg, stars=int(len(both))),
           radius_range=[float(D2.Rsun.min()), float(D2.Rsun.max())],
           GR=GR, NEWTON=NEWTON,
           sigma_from_GR_method2=abs(L2 - GR) / TOT2, sigma_from_Newton_method2=abs(L2 - NEWTON) / TOT2)
json.dump(rec, open(os.path.join(CHARTS, 'record_summary.json'), 'w'), indent=1)
D2.to_csv(os.path.join(CHARTS, 'station2_star_table.csv'), index=False)

if os.environ.get('MX24ST2_COPY_RECORD') == '1':
    os.makedirs(RECORD, exist_ok=True)
    # never overwrite a record revision: park whatever is there in a dated superseded_* folder
    stale = [f for f in os.listdir(RECORD) if os.path.isfile(os.path.join(RECORD, f))]
    old_rev = None
    if os.path.exists(os.path.join(RECORD, 'record_summary.json')):
        old_rev = json.load(open(os.path.join(RECORD, 'record_summary.json'))).get('rev')
    if stale and old_rev == REV:
        stale = []                     # a rerun of the same revision overwrites itself
    if stale:
        park = os.path.join(RECORD, 'superseded_%s_before_%s' % (datetime.date.today().isoformat(), REV))
        k = 2
        while os.path.exists(park):
            park = os.path.join(RECORD, 'superseded_%s_before_%s_%d' % (datetime.date.today().isoformat(), REV, k)); k += 1
        os.makedirs(park, exist_ok=True)
        for f in stale:
            shutil.move(os.path.join(RECORD, f), os.path.join(park, f))
        print('previous record files ->', park)
    for f in ('record_deflection.png', 'record_deflection_both.png',
              'record_field.png', 'record_field_100ms.png', 'record_field_075ms.png', 'record_field_both.png',
              'record_covariance.png', 'master_100ms_annotated.png', 'master_075ms_annotated.png',
              'record_summary.json', 'station2_star_table.csv'):
        p = os.path.join(CHARTS, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RECORD, f))
    print('record set ->', RECORD)
print('charts ->', CHARTS)
