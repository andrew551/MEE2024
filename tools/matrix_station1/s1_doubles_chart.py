"""Cell 2's fitted stars that have a close catalogue companion, and what dropping them does.

F31 (`docs/ROADMAP.md`): the stage-2 double-star cut has never removed anything on an offline
reduction -- `GaiaOfflineProvider.lookup_neighbours` returns the flagged input stars where the
caller expects their companions -- so `remove_double_tab2` was on for cell 2 and did nothing.
Andrew is investigating the defect; nothing here changes the pipeline or the record.

This tool measures the consequence and draws it: the pooled Method 2 fit with and without the
stars that have a Gaia companion inside the configured 10 " cut, and a copy of the record's
field chart with those stars ringed in yellow.

The cut, the window and the estimator are the record's own, taken from the run and the registry,
not chosen here: G <= 13, 2-10 R_sun (`tools/analysis_window.py`), `double_star_cutoff` 10.0 "
read from the stage-1 archive the record was built from.

  .venv/Scripts/python.exe tools/matrix_station1/s1_doubles_chart.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from mee2024.MEE2024util import get_data_root
from tools.analysis_window import WINDOWS

REC = r"F:/MEE_output/mexico2024/station1_record"
OUT = os.path.join(REC, 'charts')
VER = os.path.join(OUT, 'chart_versions')
RECORD = r"F:/MEE_output/RECORD/mexico2024"
POOLED = os.path.join(REC, 'pooled_fit', 'twopass')
NX, NY, PS = 9576, 6388, 1.84847
GR = 1.7512
ATM_ERR = 0.11
W = WINDOWS['mexico2024_station1']
CUT_AS = 10.0                     # double_star_cutoff, the pipeline default the record ran with
DRAWS = 600

t = pd.read_csv(os.path.join(POOLED, 'pooled_rows.csv'))
summary = json.load(open(os.path.join(POOLED, 'pooled_summary.json')))
blocks = sorted(set(t.block))
print('record: %d observations of %d stars, L = %.3f +- %.3f (bootstrap)'
      % (len(t), t.key.nunique(), summary['L'], summary['sigma_bootstrap']))

# ---------------------------------------------------------------- which stars have a companion
C = os.path.join(get_data_root(), 'catalogues', 'gaia_dr3_g15')
ra_c = np.degrees(np.asarray(np.load(os.path.join(C, 'ra.npy'), mmap_mode='r')))
dec_c = np.degrees(np.asarray(np.load(os.path.join(C, 'dec.npy'), mmap_mode='r')))
mag_c = np.asarray(np.load(os.path.join(C, 'mag.npy'), mmap_mode='r'))
sid_c = np.asarray(np.load(os.path.join(C, 'source_id.npy'), mmap_mode='r'))


def companion(ra0, dec0, own_id):
    """Nearest catalogue neighbour inside CUT_AS, excluding the star itself. (sep, mag) or None."""
    d = CUT_AS / 3600.0
    box = d / max(np.cos(np.radians(dec0)), 1e-6)
    idx = np.nonzero((np.abs(dec_c - dec0) < d * 1.5) & (np.abs(ra_c - ra0) < box * 1.5))[0]
    if not len(idx):
        return None
    sep = np.hypot((ra_c[idx] - ra0) * np.cos(np.radians(dec0)), dec_c[idx] - dec0) * 3600.0
    keep = (sep > 1e-4) & (sid_c[idx] != own_id) & (sep < CUT_AS)
    if not keep.any():
        return None
    j = np.argmin(sep[keep])
    return float(sep[keep][j]), float(mag_c[idx][keep][j])


stars = t.drop_duplicates('key')[['key', 'ra', 'dec', 'magV', 'Rsun']]
doubles = {}
for _, r in stars.iterrows():
    s = str(r.key)
    own = np.uint64(int(s[5:])) if s.startswith('gaia:') else None
    c = companion(float(r.ra), float(r['dec']), own)
    if c:
        doubles[r.key] = c
print('\n%d of %d fitted stars have a Gaia companion inside %.0f " '
      '(the cut the record ran with, which F31 stopped from acting):'
      % (len(doubles), len(stars), CUT_AS))
for k, (sep, mg) in sorted(doubles.items(), key=lambda kv: kv[1][0]):
    r = stars[stars.key == k].iloc[0]
    print('   G %5.2f at %5.2f R_sun   companion %5.2f " away, G %5.2f   (%d observations)'
          % (r.magV, r.Rsun, sep, mg, int((t.key == k).sum())))

# ---------------------------------------------------------------- the pooled fit, both ways
def solve(d):
    """The record's estimator: per-block offset and rotation, one shared scale, one L."""
    n = len(d); Z = np.zeros(n)
    xs, ys = (d.px.values - NX / 2) * PS, (d.py.values - NY / 2) * PS
    ux, uy = d.rx.values / d.R.values, d.ry.values / d.R.values
    cx, cy = [], []
    for b in blocks:
        m = (d.block.values == b).astype(float)
        cx += [m, Z, -ys * m]; cy += [Z, m, xs * m]
    cx += [xs, ux * d.RS.values / d.R.values]
    cy += [ys, uy * d.RS.values / d.R.values]
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    y = np.concatenate([d.dx.values, d.dy.values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c[-1]


def boot(d, seed=3):
    rng = np.random.default_rng(seed); ids = d.key.unique(); out = []
    for _ in range(DRAWS):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([d[d.key == i] for i in pick], ignore_index=True)
        if len(set(s.block)) < len(blocks):
            continue
        try:
            out.append(solve(s))
        except Exception:
            pass
    return float(np.std(out, ddof=1)), len(out)


L_all = solve(t)
S_all, _ = boot(t)
kept = t[~t.key.isin(doubles)]
L_cut = solve(kept)
S_cut, _ = boot(kept)
print('\n%-34s %8s %8s %10s %8s' % ('', 'stars', 'obs', 'L (arcsec)', 'sigma'))
print('%-34s %8d %8d %10.3f %8.3f' % ('as the record stands', t.key.nunique(), len(t), L_all, S_all))
print('%-34s %8d %8d %10.3f %8.3f' % ('doubles removed', kept.key.nunique(), len(kept), L_cut, S_cut))
print('%-34s %8s %8s %+10.3f' % ('difference', '', '', L_cut - L_all))
print('total with the atmosphere term %.2f ": record %.3f +- %.3f, doubles removed %.3f +- %.3f'
      % (ATM_ERR, L_all, np.hypot(S_all, ATM_ERR), L_cut, np.hypot(S_cut, ATM_ERR)))

# ---------------------------------------------------------------- the chart
# The record's field chart, verbatim in construction (s1_charts_record.py), with the doubles ringed.
ux, uy = t.rx.values / t.R.values, t.ry.values / t.R.values
t['vx'] = t.res_x.values + L_all * t.RS.values / t.R.values * ux
t['vy'] = t.res_y.values + L_all * t.RS.values / t.R.values * uy
u = t.groupby('key').agg(px=('px', 'mean'), py=('py', 'mean'), vx=('vx', 'mean'),
                         vy=('vy', 'mean')).reset_index()

ra0, de0 = t.ra.mean(), t['dec'].mean()
Xa = (t.ra.values - ra0) * np.cos(np.radians(de0))
Ya = t['dec'].values - de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
axc, *_ = np.linalg.lstsq(Aa, t.px.values, rcond=None)
ayc, *_ = np.linalg.lstsq(Aa, t.py.values, rcond=None)
Minv = np.linalg.inv(np.array([[axc[0], axc[1]], [ayc[0], ayc[1]]]))


def px_to_sky(px, py):
    v = Minv @ np.vstack([np.asarray(px, float) - axc[2], np.asarray(py, float) - ayc[2]])
    return ra0 + v[0] / np.cos(np.radians(de0)), de0 + v[1]


def sensor_vec_to_sky(dx_as, dy_as):
    v = Minv @ np.vstack([np.asarray(dx_as, float) / PS, np.asarray(dy_as, float) / PS])
    return v[0] * 3600, v[1] * 3600


fig, ax = plt.subplots(figsize=(11.5, 8))
ARROW_DEG = 0.40
sra, sdec = px_to_sky(u.px.values, u.py.values)
vra, vdec = sensor_vec_to_sky(u.vx.values, u.vy.values)
cor_ra, cor_de = px_to_sky(np.array([0, NX, NX, 0]), np.array([0, 0, NY, NY]))
ax.add_patch(Polygon(np.c_[cor_ra, cor_de], fill=False, color='gray', lw=1.2,
                     label='sensor footprint'))
ends_ra, ends_de = [], []
is_dbl = u.key.isin(doubles).values
for k in range(len(u)):
    x1 = sra[k] + vra[k] * ARROW_DEG / np.cos(np.radians(de0))
    y1 = sdec[k] + vdec[k] * ARROW_DEG
    ends_ra.append(x1); ends_de.append(y1)
    ax.annotate('', xy=(x1, y1), xytext=(sra[k], sdec[k]),
                arrowprops=dict(arrowstyle='-|>,head_width=0.20,head_length=0.40',
                                color='tab:blue', lw=1.1, shrinkA=0, shrinkB=0))
ax.scatter(sra[~is_dbl], sdec[~is_dbl], s=18, color='tab:blue', zorder=5,
           label='%d with no companion' % int((~is_dbl).sum()))
ax.scatter(sra[is_dbl], sdec[is_dbl], s=90, facecolors='none', edgecolors='gold', linewidths=2.2,
           zorder=6, label='%d with one inside %.0f " (F31)' % (int(is_dbl.sum()), CUT_AS))
ax.scatter(sra[is_dbl], sdec[is_dbl], s=18, color='tab:blue', zorder=7)
sun_ra, sun_dec = px_to_sky(np.array([t.sun_px.mean()]), np.array([t.sun_py.mean()]))
sun_ra, sun_dec = float(sun_ra[0]), float(sun_dec[0])
ax.add_patch(Circle((sun_ra, sun_dec), t.RS.mean() / 3600, color='black', zorder=3,
                    label='the Sun, 1 R$_\\odot$ to scale'))
ax.add_patch(Circle((sun_ra, sun_dec), 2 * t.RS.mean() / 3600, fill=False, color='gray', ls='--',
                    lw=1.0, zorder=3, label='2 R$_\\odot$'))
lo_ra = min(cor_ra.min(), min(ends_ra)) - 0.05
hi_ra = max(cor_ra.max(), max(ends_ra)) + 0.05
lo_de = min(cor_de.min(), min(ends_de)) - 0.05
hi_de = max(cor_de.max(), max(ends_de)) + 0.05
for x1, y1 in zip(ends_ra, ends_de):
    assert lo_ra < x1 < hi_ra and lo_de < y1 < hi_de, 'an arrow leaves the axes'
# RA ascending to the right, as s1_charts_record.py line 272 draws it
ax.set_xlim(lo_ra, hi_ra); ax.set_ylim(lo_de, hi_de)
ax.set_aspect(1 / np.cos(np.radians(de0)))
ax.set_xlabel('RA (degrees)', fontsize=12); ax.set_ylabel('DEC (degrees)', fontsize=12)
ax.set_title('Displacement vectors (%d stars), the close doubles ringed \u2014 Mexico 2024 Station 1, '
             'pooled fit, G $\\leq$ 13' % len(u), fontsize=12)
star_rms = float(np.sqrt(np.mean(t.res.values ** 2)))
bar_deg = ARROW_DEG / np.cos(np.radians(de0))
for y_fr, ln, txt in ((0.38, 1.0, '1 arcsec of displacement'),
                      (0.28, star_rms, 'per-observation scatter (%.2f")' % star_rms)):
    ax.annotate('', xy=(1.04 + ln * bar_deg / (hi_ra - lo_ra), y_fr), xytext=(1.04, y_fr),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color='black', lw=3))
    ax.annotate(txt, (1.04, y_fr + 0.028), xycoords='axes fraction', fontsize=8)
ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.70))
_note = ('ringed: a Gaia source lies within %.0f " of the star, so the stage-2 cut should have '
         'removed it (F31).%sRemoving all %d moves L from %.3f to %.3f " (%+.3f "), inside the '
         '±%.3f " statistical bar. Not applied: the record stands at %.3f ".'
         % (CUT_AS, chr(10), len(doubles), L_all, L_cut, L_cut - L_all, S_all, L_all))
fig.text(0.055, 0.020, _note, fontsize=8.5)
fig.subplots_adjust(left=0.07, right=0.78, top=0.94, bottom=0.14)
os.makedirs(VER, exist_ok=True)
fig.savefig(os.path.join(OUT, 'record_field_doubles.png'), dpi=140)
fig.savefig(os.path.join(VER, 'rev02_record_field_doubles.png'), dpi=140)
plt.close(fig)

tab = pd.DataFrame([{'key': k, 'magV': float(stars[stars.key == k].magV.iloc[0]),
                     'Rsun': float(stars[stars.key == k].Rsun.iloc[0]),
                     'companion_sep_arcsec': v[0], 'companion_magG': v[1],
                     'observations': int((t.key == k).sum())} for k, v in doubles.items()])
tab.sort_values('companion_sep_arcsec').to_csv(os.path.join(OUT, 'station1_double_stars.csv'),
                                               index=False)
print('\ncharts -> %s' % OUT)
if os.environ.get('MX24_COPY_RECORD') == '1':
    import shutil
    for f in ('record_field_doubles.png', 'station1_double_stars.csv'):
        shutil.copy2(os.path.join(OUT, f), os.path.join(RECORD, f))
    print('added to %s (new files; nothing existing is touched)' % RECORD)
