"""The record chart set for Mexico 2024 Station 2, in the cell-2 house style.

Douglas asked for a reduced set: the deflection chart, the field chart, the covariance chart
carrying BOTH methods, and one annotated master per tier. Written to
`F:\\MEE_output\\RECORD\\mexico2024st2`.

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
from tools.record_charts import (ChartWriter, SkyFrame, arcsinh_stretch, covariance_chart,
                                 field_chart as draw_field, joint_plate_scale, ppm_from,
                                 reference_curves, scale_bars)
from matplotlib.patches import Circle

REV = 'rev07'
OUT = r"F:/MEE_output/station2_transfer"
CHARTS = os.path.join(OUT, 'charts')
VER = os.path.join(CHARTS, 'chart_versions')
RECORD = r"F:/MEE_output/RECORD/mexico2024st2"
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
       '075ms': 'eclipse/075ms/stage2_method1_quadfree'},
       # Method 3 (2026-09-14): quadratic and cubic from the quadratic-free bracket, constant
       # and linear refitted on the eclipse field, scale fitted -- tools/matrix_station2/
       # s2_method3.py.  Same reference files and tolerances as m1; only the rung differs.
       'm3': {'100ms': 'eclipse/100ms/stage2_method3',
       '075ms': 'eclipse/075ms/stage2_method3'}}


_writer = ChartWriter(CHARTS, REV, ver=VER)
save = _writer.save


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


def design(d, with_scale, shared_scale=True):
    """Per-tier offset and rotation, one L, and the plate scale either SHARED across the tiers
    (the record estimator from rev05) or free per tier (kept for comparison).

    Douglas, 2026-09-09: fitted separately, the two tiers' scales came out 96 ppm apart -- not
    physical for one optic 1.3 minutes apart, and at 0.0171 " of L per ppm it is the whole of
    the 2.50 vs 0.55 " split between them. Each tier's own scale is uncertain by ~85 ppm because
    it is 80 % degenerate with L on 17 stars in a narrow annulus. Cell 2 shares one scale across
    its four blocks for the same reason (they agree to 3 ppm); this does the same.
    """
    n = len(d); Z = np.zeros(n)
    px, py = d['px'].values, d['py'].values
    ux, uy = d['rx'].values / d['R'].values, d['ry'].values / d['R'].values
    xs, ys = (px - NX / 2) * PS, (py - NY / 2) * PS
    cx, cy, lab = [], [], []
    for tag, _, _ in TIERS:
        m = (d['tier'] == tag).values.astype(float)
        cx += [m, Z, -m * ys]
        cy += [Z, m, m * xs]
        lab += ['N1_' + tag, 'N2_' + tag, 'Th_' + tag]
        if with_scale and not shared_scale:
            cx.append(m * xs); cy.append(m * ys)
            lab.append('S_' + tag)
    if with_scale and shared_scale:
        cx.append(xs); cy.append(ys); lab.append('S')
    cx.append(ux * R_SUN_AS / d['R'].values); cy.append(uy * R_SUN_AS / d['R'].values)
    lab.append('L')
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), lab


def solve(d, with_scale, shared_scale=True):
    A, lab = design(d, with_scale, shared_scale)
    y = np.concatenate([d['dx_arcsec'].values, d['dy_arcsec'].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    n = len(d)
    return c, lab, resid[:n], resid[n:], A


def bootstrap(d, with_scale, draws=600, seed=7, shared_scale=True):
    rng = np.random.default_rng(seed)
    ids = d['ID'].unique(); out = []
    for _ in range(draws):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([d[d['ID'] == i] for i in pick], ignore_index=True)
        if s['tier'].nunique() < len(TIERS):
            continue
        try:
            c, lab, *_ = solve(s, with_scale, shared_scale)
            out.append(c[lab.index('L')])
        except Exception:
            pass
    return float(np.std(out, ddof=1)), len(out)


# ---------------------------------------------------------------- the two reductions
D2 = load('m2')
c2, lab2, rx2, ry2, A2 = solve(D2, True)                     # one scale shared by both tiers
L2 = c2[lab2.index('L')]
S2, n2 = bootstrap(D2, True)
# the per-tier-scale fit, reported beside the record estimator rather than as it
c2t, lab2t, *_ = solve(D2, True, shared_scale=False)
L2_TIERS = c2t[lab2t.index('L')]
S2_TIERS, _ = bootstrap(D2, True, shared_scale=False)
TIER_SCALE_PPM = {tag: 1e6 * c2t[lab2t.index('S_' + tag)] for tag, _, _ in TIERS}

# Method 3: solved exactly as Method 2 is -- both tiers pooled, one shared scale, the scale
# free -- because Method 3's scale IS fitted (the linear terms are free, so
# distortion_polynomial never substitutes the reference's).  Only the stage-2 rung differs.
D3 = load('m3')
c3, lab3, rx3, ry3, A3 = solve(D3, True)
L3 = c3[lab3.index('L')]
S3, n3 = bootstrap(D3, True)

D1 = load('m1')
c1, lab1, rx1, ry1, A1 = solve(D1, False)
L1 = c1[lab1.index('L')]
S1, n1 = bootstrap(D1, False)

IMPORTED = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, 'bracket_quadfree', n, '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for n in ('right', 'left')]))
BRK_SPREAD = abs(json.load(open(glob.glob(os.path.join(OUT, 'bracket_quadfree', 'left', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)']
                 - json.load(open(glob.glob(os.path.join(OUT, 'bracket_quadfree', 'right', '**', 'distortion_results.txt'), recursive=True)[0], encoding='utf-8'))['platescale (arcseconds/pixel)'])

# The bracket's own fitted scale uncertainty, which Method 1 imports along with the scale --
# and which the Method 1 bar did not carry until 2026-09-08. Each side reports ~30 ppm even on
# 82 stars; the mean of two carries that over root 2. At this field's leverage that exceeds the
# statistical error on L, and it is what puts GR inside the Method 1 bar.
BRK_SIG = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, 'bracket_quadfree', n, '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale_relative_uncertainty'] for n in ('right', 'left')])) / np.sqrt(2)
LEVERAGE = 0.0171                    # arcsec of L per ppm, measured on this field
S1_SCALE = 1e6 * BRK_SIG * LEVERAGE  # the imported scale's contribution to L, in arcsec
TOT2 = float(np.hypot(S2, ATM_ERR))
print('Method 2, one shared scale: L = %+.3f +- %.3f (stat, %d draws), %d obs of %d observations'
      % (L2, S2, n2, len(D2), D2.ID.nunique()))
print('Method 2, a scale per tier:  L = %+.3f +- %.3f  (tier scales %s ppm from the reference)'
      % (L2_TIERS, S2_TIERS, ', '.join('%s %+.1f' % (k, v) for k, v in TIER_SCALE_PPM.items())))
print('Method 1: L = %+.3f +- %.3f (stat) +- %.3f (imported scale, %.1f ppm), scale %.7f "/px'
      % (L1, S1, S1_SCALE, 1e6 * BRK_SIG, IMPORTED))

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
band, ln1, ln2, ln3 = reference_curves(
    ax, xx, L2, TOT2, 'pooled Method 2:  L = %.2f"' % L2,
    'total $\\pm$%.2f" (stat %.2f + atmosphere %.2f)' % (TOT2, S2, ATM_ERR))
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
# The sky frame and the field chart are the shared constructions (tools/record_charts.py):
# an affine from catalogue RA/DEC to pixels fitted on the matched stars and inverted, RA
# ascending to the right, every arrow end asserted inside the axes.
SF = SkyFrame.from_stars(D2.ra.values, D2['dec'].values, D2.px.values, D2.py.values, PS)
px_to_sky, sensor_vec_to_sky = SF.px_to_sky, SF.sensor_vec_to_sky


def field_chart(u, fname, title, note, star_rms, colour='tab:blue', label=None):
    """u: one row per star with px, py, vx, vy (arcsec, nuisances removed). RA/DEC axes."""
    fig, ax = plt.subplots(figsize=(11.5, 8))
    ARROW_DEG = 0.40                  # degrees drawn per arcsec of displacement
    sra, sdec = px_to_sky(u.px.values, u.py.values)
    vra, vdec = sensor_vec_to_sky(u.vx.values, u.vy.values)
    sun_ra, sun_de = px_to_sky(np.array([SUNPX]), np.array([SUNPY]))
    lo_ra, hi_ra, _, _ = draw_field(
        ax, sra, sdec, vra, vdec, SF.corners(NX, NY),
        (float(sun_ra[0]), float(sun_de[0]), R_SUN_AS / 3600), ARROW_DEG, SF.cos0,
        groups=[(np.ones(len(u), bool), dict(s=26, color=colour, label=label or '%d observations' % len(u)))],
        arrow_color=colour, arrow_lw=1.3, sun_ring2=True, include_sun_in_limits=True,
        pad=(0.05, 0.05), title=title)
    scale_bars(ax, ((0.40, 1.0, '1 arcsec of displacement'),
                    (0.30, star_rms, 'scatter (%.2f")' % star_rms)),
               ARROW_DEG, SF.cos0, hi_ra - lo_ra, text_dy=0.03)
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
print('Both tiers, averaged: L = %+.3f +- %.3f (stat, %d draws), %d observations'
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
# Drawn as cells 1 and 3 draw it (b17_charts_record.py, step3_charts_record.py): the vertical
# axis is the plate scale in ppm from the IMPORTED value, both methods are 1-sigma ellipses, and
# Method 1's ellipse carries its imported-scale term. rev05 and earlier plotted the absolute scale
# with Method 1 as a point -- and put Method 2's scale on the wrong base: PS*(1+S) with PS the
# ZENITH reference, where the stage-2 model the residuals are measured against carries its own
# free scale. Cell 2's construction, joint = stage-2 scale - S*PS, reproduces its recorded joint
# scales to seven digits; on Station 2 it moves the Method 2 scale from +665 ppm to -54 ppm from
# the bracket, which is the whole reason the rev05 chart looked nothing like the other two.
sig2 = float(np.sqrt(np.mean(np.concatenate([rx2, ry2]) ** 2)))
cov2 = sig2 ** 2 * np.linalg.pinv(A2.T @ A2)
iL, iS = lab2.index('L'), lab2.index('S')
STAGE2_SCALE = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, SUB['m2'][tag], '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for tag, _, _ in TIERS]))
joint = joint_plate_scale(STAGE2_SCALE, c2[iS], PS)  # S < 0 means a larger scale
JOINT_PPM = ppm_from(joint, IMPORTED)                # physical: + means more arcsec per pixel
# Cells 1 and 3 plot S itself, whose sign is the residual convention: their "+ppm" is a scale
# that is physically SMALLER. To sit in that set the same quantity is plotted here, and the box
# states both readings so the sign cannot be misread.
Y2 = -JOINT_PPM
C2 = np.array([[cov2[iL, iL], cov2[iL, iS] * 1e6], [cov2[iS, iL] * 1e6, cov2[iS, iS] * 1e12]])
C2[0, 0] = S2 ** 2                                  # carry the bootstrap sigma the record quotes
RHO = C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1])
SCALE_PPM = 1e6 * BRK_SIG                           # the imported scale's own uncertainty
C1 = np.array([[S1 ** 2 + S1_SCALE ** 2, -S1_SCALE * SCALE_PPM], [-S1_SCALE * SCALE_PPM, SCALE_PPM ** 2]])
# Method 3's ellipse, built by the same construction as Method 2's: its own stage-2 scale
# carried through joint_plate_scale, and the bootstrap sigma on the diagonal.
sig3 = float(np.sqrt(np.mean(np.concatenate([rx3, ry3]) ** 2)))
cov3 = sig3 ** 2 * np.linalg.pinv(A3.T @ A3)
iL3, iS3 = lab3.index('L'), lab3.index('S')
STAGE2_SCALE3 = float(np.mean([json.load(open(glob.glob(os.path.join(
    OUT, SUB['m3'][tag], '**', 'distortion_results.txt'), recursive=True)[0],
    encoding='utf-8'))['platescale (arcseconds/pixel)'] for tag, _, _ in TIERS]))
joint3 = joint_plate_scale(STAGE2_SCALE3, c3[iS3], PS)
JOINT_PPM3 = ppm_from(joint3, IMPORTED)
Y3 = -JOINT_PPM3
C3 = np.array([[cov3[iL3, iL3], cov3[iL3, iS3] * 1e6], [cov3[iS3, iL3] * 1e6, cov3[iS3, iS3] * 1e12]])
C3[0, 0] = S3 ** 2
TOT3 = float(np.hypot(S3, ATM_ERR))
_tot1 = float(np.hypot(np.sqrt(C1[0, 0]), ATM_ERR))
_lines = [('Method 1:  L = %.3f $\\pm$ %.3f" (stat %.3f + scale %.3f)' % (L1, np.sqrt(C1[0, 0]), S1, S1_SCALE), 'darkred'),
          ('      $\\pm$ %.3f" with the atmosphere term %.2f' % (_tot1, ATM_ERR), 'darkred'),
          ('Method 2:  L = %.3f $\\pm$ %.3f" (stat), $\\pm$%.3f" with atmosphere' % (L2, S2, TOT2), 'tab:blue'),
          ('      scale %+.1f ppm from imported (%.7f "/px, %+.0f ppm in "/px)' % (Y2, joint, JOINT_PPM), 'tab:blue'),
          ('      correlation L vs scale = %+.2f' % RHO, 'tab:blue'),
          ('Imported plate scale: %.7f "/px' % IMPORTED, 'black'),
          ('      (the L/R bracket mean, $\\pm$%.1f ppm from its own fits)' % SCALE_PPM, 'black'),
          ('Method 3:  L = %.3f $\\pm$ %.3f" (stat), $\\pm$%.3f" with atmosphere' % (L3, S3, TOT3), 'tab:purple'),
          ('      quadratic + cubic imported from the bracket, linear refitted here', 'tab:purple'),
          ('      scale %+.1f ppm from imported (%.7f "/px)' % (Y3, joint3), 'tab:purple'),
          ('from %d observations of %d stars, both tiers pooled, one shared scale' % (len(D2), D2.ID.nunique()), 'black')]
fig, ax = covariance_chart(C1, (L1, 0.0), C2, (L2, Y2), _lines,
                           'L and plate scale — Mexico 2024 Station 2, G $\\leq$ 13, both tiers, one scale',
                           newton=True, newton_lw=1.5,
                           name1='Method 1 (scale imported; stat + scale)', name2='Method 2 (scale free)',
                           C3=C3, mu3=(L3, Y3),
                           name3='Method 3 (quadratic imported, scale free)')
save(fig, 'record_covariance.png')
print('covariance: Method 2 rho %+.2f, joint scale %.7f; Method 1 imported %.7f' % (RHO, joint, IMPORTED))
print('            Method 3 L = %.3f +- %.3f, joint scale %.7f (%+.1f ppm), %d observations'
      % (L3, S3, joint3, Y3, len(D3)))

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
    disp = arcsinh_stretch(img)
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
rec = dict(rev=REV, cell='Mexico 2024 Station 2',
           estimator='pooled over both tiers, every observation, ONE plate scale shared by the tiers',
           reference='cubic, fifteen zenith fields, moments+annular, 0.5 arcsec gate',
           bracket='right 18:10:32-18:11:21, left 18:14:05-18:14:57, ends of totality trimmed',
           observations=int(len(D2)), stars=int(D2.ID.nunique()),
           method2=dict(L=L2, sigma_stat=S2, sigma_atmosphere=ATM_ERR, sigma_total=TOT2,
                        joint_platescale=joint, joint_ppm_from_imported=float(JOINT_PPM),
                        stage2_platescale=STAGE2_SCALE, corr_L_platescale=float(RHO)),
           method2_scale_per_tier=dict(L=L2_TIERS, sigma_stat=S2_TIERS,
                                       tier_scale_ppm_from_reference=TIER_SCALE_PPM),
           double_star_removal='none: F31 inoperative, and the 10 arcsec cut is not trusted (Douglas, 2026-09-09)',
           method1=dict(L=L1, sigma_stat=S1, imported_platescale=IMPORTED,
                        bracket_LR_ppm=1e6 * BRK_SPREAD / IMPORTED,
                        sigma_scale_ppm=1e6 * BRK_SIG, sigma_L_from_scale=S1_SCALE,
                        sigma_total=float(np.hypot(S1, S1_SCALE))),
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
