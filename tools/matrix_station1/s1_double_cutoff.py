"""What should `double_star_cutoff` be? Measured on cell 2, not assumed. (Douglas, 2026-09-08)

The default is 10 ", which is a round number in a config file, not a measurement. Cell 2 has 192
fitted stars in 639 observations and an eclipse-field PSF of sigma 3.2 px = 13.9 " FWHM, so every
companion inside ~14 " is blended into its primary's footprint rather than resolved beside it.
That is the regime the cut exists for, and it is testable.

The test is directional, which is what makes it sharp. A blend pulls the measured centroid TOWARD
the companion, by roughly

    shift = separation x f2/(f1+f2),   f2/f1 = 10^(-0.4 dG)

while the two are unresolved. So for every star with a catalogue neighbour, project that star's
mean residual about the pooled fit onto the unit vector pointing at the neighbour. Scatter is
symmetric about zero; blending is not. Where the projection stops being positive is where the cut
belongs.

Two datasets. Cell 2's eclipse field has 192 stars and answers the question weakly; Station 1's
SIXTEEN zenith fields have 1712 stars measured 6 times each at a residual rms of 0.167 ", and
answer it decisively. The zenith residual used is obs-minus-catalogue from the RA/DEC columns --
NOT px - px_dist, which is the distortion correction, not a residual.

Companions come from the same catalogue the pipeline uses (gaia_dr3_g15, so G <= 15 -- fainter
neighbours are invisible to the cut whatever its radius, which is a separate limitation worth
knowing). The search runs to 60 " to have a control population well outside any plausible cut.

L is reported against cutoff at the end as a SENSITIVITY only. The cut must be chosen on the
blend evidence; choosing it on L would be choosing the answer.

  .venv/Scripts/python.exe tools/matrix_station1/s1_double_cutoff.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from mee2024.MEE2024util import get_data_root
from tools.analysis_window import WINDOWS

REC = r"D:/MEE2024 output/MEE_output/station1_record"
OUT = os.path.join(REC, 'charts')
POOLED = os.path.join(REC, 'pooled_fit', 'twopass')
NX, NY, PS = 9576, 6388, 1.84847
PSF_FWHM_AS = 2.3548 * 3.2 * PS          # sigma 3.2 px on the eclipse stacks, docs/STEP3_2026.md
SEARCH_AS = 60.0                         # well outside any plausible cut, to give a control set
CAT_LIMIT_G = 15.0                       # gaia_dr3_g15: the pipeline cannot see fainter neighbours
W = WINDOWS['mexico2024_station1']

t = pd.read_csv(os.path.join(POOLED, 'pooled_rows.csv'))
summary = json.load(open(os.path.join(POOLED, 'pooled_summary.json')))
blocks = sorted(set(t.block))
print('cell 2: %d observations of %d stars, L = %.3f, per-observation scatter %.3f "'
      % (len(t), t.key.nunique(), summary['L'], float(np.sqrt(np.mean(t.res.values ** 2)))))
print('eclipse-field PSF: sigma 3.2 px = %.1f " FWHM, so a companion inside ~%.0f " is blended, '
      'not resolved' % (PSF_FWHM_AS, PSF_FWHM_AS))

# ---------------------------------------------------------------- catalogue neighbours
C = os.path.join(get_data_root(), 'catalogues', 'gaia_dr3_g15')
ra_c = np.degrees(np.asarray(np.load(os.path.join(C, 'ra.npy'), mmap_mode='r')))
dec_c = np.degrees(np.asarray(np.load(os.path.join(C, 'dec.npy'), mmap_mode='r')))
mag_c = np.asarray(np.load(os.path.join(C, 'mag.npy'), mmap_mode='r'))
sid_c = np.asarray(np.load(os.path.join(C, 'source_id.npy'), mmap_mode='r'))


def neighbour(ra0, dec0, own_id):
    """Nearest catalogue neighbour inside SEARCH_AS. (sep ", dRA*cos, dDec in ", its G) or None."""
    d = SEARCH_AS / 3600.0
    box = d / max(np.cos(np.radians(dec0)), 1e-6)
    idx = np.nonzero((np.abs(dec_c - dec0) < d * 1.5) & (np.abs(ra_c - ra0) < box * 1.5))[0]
    if not len(idx):
        return None
    dx = (ra_c[idx] - ra0) * np.cos(np.radians(dec0)) * 3600.0
    dy = (dec_c[idx] - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    keep = (sep > 1e-3) & (sid_c[idx] != own_id) & (sep < SEARCH_AS)
    if not keep.any():
        return None
    j = np.argmin(sep[keep])
    return float(sep[keep][j]), float(dx[keep][j]), float(dy[keep][j]), float(mag_c[idx][keep][j])


# ---------------------------------------------------------------- per-star residual, sky frame
ra0, de0 = t.ra.mean(), t['dec'].mean()
Aa = np.c_[(t.ra.values - ra0) * np.cos(np.radians(de0)), t['dec'].values - de0,
           np.ones(len(t))]
axc, *_ = np.linalg.lstsq(Aa, t.px.values, rcond=None)
ayc, *_ = np.linalg.lstsq(Aa, t.py.values, rcond=None)
M = np.array([[axc[0], axc[1]], [ayc[0], ayc[1]]])       # degrees of sky -> pixels


def sky_dir_to_sensor(dra_as, ddec_as):
    """A sky offset in arcsec -> the same direction on the sensor axes, unit length."""
    v = M @ np.array([dra_as / 3600.0, ddec_as / 3600.0])
    n = np.hypot(*v)
    return v / n if n else v


star = t.groupby('key').agg(res_x=('res_x', 'mean'), res_y=('res_y', 'mean'),
                            magV=('magV', 'first'), Rsun=('Rsun', 'mean'),
                            ra=('ra', 'first'), dec=('dec', 'first'),
                            nobs=('res', 'size')).reset_index()
rows = []
for _, r in star.iterrows():
    s = str(r.key)
    own = np.uint64(int(s[5:])) if s.startswith('gaia:') else None
    nb = neighbour(float(r.ra), float(r['dec']), own)
    if not nb:
        continue
    sep, dra, ddec, gm = nb
    u = sky_dir_to_sensor(dra, ddec)
    proj = float(r.res_x * u[0] + r.res_y * u[1])          # + = pulled toward the companion
    perp = float(-r.res_x * u[1] + r.res_y * u[0])         # the control direction
    dG = gm - r.magV
    f = 10 ** (-0.4 * dG)
    rows.append(dict(key=r.key, magV=r.magV, Rsun=r.Rsun, nobs=r.nobs, sep=sep, comp_G=gm,
                     dG=dG, predicted=sep * f / (1 + f), proj=proj, perp=perp,
                     res=float(np.hypot(r.res_x, r.res_y))))
d = pd.DataFrame(rows)
print('\n%d of %d stars have a catalogue neighbour inside %.0f "' % (len(d), len(star), SEARCH_AS))

# ---------------------------------------------------------------- the ten, individually
print('\nthe stars inside the 10 " default, each with its residual projected onto the companion:')
print('%-6s %6s %7s %7s %9s %9s %9s %5s' % ('G', 'R_sun', 'sep "', 'comp G', 'predicted',
                                            'toward "', 'across "', 'obs'))
for _, r in d[d.sep < 10].sort_values('sep').iterrows():
    print('%-6.2f %6.2f %7.2f %7.2f %9.3f %+9.3f %+9.3f %5d'
          % (r.magV, r.Rsun, r.sep, r.comp_G, r.predicted, r.proj, r.perp, r.nobs))

# ---------------------------------------------------------------- projection against separation
print('\nresidual projected onto the companion direction, binned by separation:')
print('%-14s %5s %11s %11s %10s' % ('separation', 'n', 'toward "', 'across "', 'toward/sigma'))
edges = [0, 5, 10, 15, 20, 30, 45, 60]
for a, b in zip(edges[:-1], edges[1:]):
    m = (d.sep >= a) & (d.sep < b)
    if m.sum() < 2:
        print('%-14s %5d  (too few)' % ('%g-%g "' % (a, b), int(m.sum()))); continue
    v, w = d.proj[m].values, d.perp[m].values
    se = np.std(v, ddof=1) / np.sqrt(len(v))
    print('%-14s %5d %+7.3f\u00b1%.3f %+7.3f\u00b1%.3f %10.1f'
          % ('%g-%g "' % (a, b), int(m.sum()), v.mean(), se, w.mean(),
             np.std(w, ddof=1) / np.sqrt(len(w)), v.mean() / se if se else np.nan))

# a weighted line through the predicted-shift relation
m = d.sep < 30
if m.sum() > 3:
    k, c = np.polyfit(d.predicted[m], d.proj[m], 1)
    print('\nprojection vs predicted blend shift (stars inside 30 "): slope %.2f, intercept %+.3f "'
          % (k, c))
    print('  slope 1 = the blend model exactly; 0 = no blend signal at all')

# ---------------------------------------------------------------- L against the cutoff (sensitivity)
def solve(dd):
    n = len(dd); Z = np.zeros(n)
    xs, ys = (dd.px.values - NX / 2) * PS, (dd.py.values - NY / 2) * PS
    ux, uy = dd.rx.values / dd.R.values, dd.ry.values / dd.R.values
    cx, cy = [], []
    for b in blocks:
        mm = (dd.block.values == b).astype(float)
        cx += [mm, Z, -ys * mm]; cy += [Z, mm, xs * mm]
    cx += [xs, ux * dd.RS.values / dd.R.values]
    cy += [ys, uy * dd.RS.values / dd.R.values]
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    y = np.concatenate([dd.dx.values, dd.dy.values])
    return np.linalg.lstsq(A, y, rcond=None)[0][-1]


print('\nSENSITIVITY ONLY -- the cut is chosen on the blend evidence above, never on L:')
print('%-10s %7s %7s %9s' % ('cutoff "', 'stars', 'obs', 'L (arcsec)'))
for cut in (0, 2, 5, 10, 15, 20, 30):
    drop = set(d.key[d.sep < cut]) if cut else set()
    kept = t[~t.key.isin(drop)]
    print('%-10s %7d %7d %9.3f' % ('%g' % cut if cut else 'none', kept.key.nunique(), len(kept),
                                   solve(kept)))

# ---------------------------------------------------------------- the zenith fields: the real test
ZEN = (r"D:/MEE2024 output/MEE_output/station1_record/zenith_nulls_corr/**/"
       r"CATALOGUE_MATCHED_ERRORS.csv")
zf = sorted(__import__('glob').glob(ZEN, recursive=True))
frames = []
for f in zf:
    z = pd.read_csv(f)
    cd = np.cos(np.radians(z['DEC(catalog)'].values))
    frames.append(pd.DataFrame(dict(
        ID=z.ID.values, magV=z.magV.values, ra=z['RA(catalog)'].values, dec=z['DEC(catalog)'].values,
        rx=(z['RA(obs)'].values - z['RA(catalog)'].values) * cd * 3600.0,
        ry=(z['DEC(obs)'].values - z['DEC(catalog)'].values) * 3600.0)))
Z = pd.concat(frames, ignore_index=True)
print()
print('=== the zenith fields: %d stacks, %d rows, %d stars, residual rms %.3f "'
      % (len(zf), len(Z), Z.ID.nunique(), float(np.sqrt(np.mean(Z.rx ** 2 + Z.ry ** 2)))))
zi = {}
for _, r in Z.drop_duplicates('ID')[['ID', 'ra', 'dec']].iterrows():
    ss = str(r.ID)
    own = np.uint64(int(ss[5:])) if ss.startswith('gaia:') else None
    n = neighbour(float(r.ra), float(r['dec']), own)
    if n:
        zi[r.ID] = n
Z = Z[Z.ID.isin(zi)].copy()
zsep = np.array([zi[i][0] for i in Z.ID]); zdra = np.array([zi[i][1] for i in Z.ID])
zdd = np.array([zi[i][2] for i in Z.ID]); zcg = np.array([zi[i][3] for i in Z.ID])
nn = np.hypot(zdra, zdd)
Z['sep'] = zsep; Z['dG'] = zcg - Z.magV.values
Z['proj'] = Z.rx.values * zdra / nn + Z.ry.values * zdd / nn
Z['perp'] = -Z.rx.values * zdd / nn + Z.ry.values * zdra / nn
zg = Z.groupby('ID').agg(sep=('sep', 'first'), dG=('dG', 'first'), magV=('magV', 'first'),
                         proj=('proj', 'mean'), perp=('perp', 'mean'),
                         n=('proj', 'size')).reset_index()
print('%d stars have a neighbour inside %.0f "; each measured in %d stacks; PSF ~3.7 " FWHM'
      % (len(zg), SEARCH_AS, int(zg.n.median())))
print('%-13s %5s %16s %16s %7s' % ('separation', 'stars', 'toward "', 'across "', 'sigma'))
ZEDGES = [0, 2, 4, 6, 10, 14, 20, 30, 45, 60]
for a, b in zip(ZEDGES[:-1], ZEDGES[1:]):
    m = (zg.sep >= a) & (zg.sep < b)
    if m.sum() < 3:
        print('%-13s %5d  (too few)' % ('%g-%g "' % (a, b), int(m.sum()))); continue
    v, w = zg.proj[m].values, zg.perp[m].values
    se = np.std(v, ddof=1) / np.sqrt(len(v))
    print('%-13s %5d %+9.4f±%.4f %+9.4f±%.4f %7.1f'
          % ('%g-%g "' % (a, b), int(m.sum()), v.mean(), se, w.mean(),
             np.std(w, ddof=1) / np.sqrt(len(w)), v.mean() / se))
print()
print('the flux-weighted blend model, fitted inside a growing radius:')
for lim in (4, 6, 8, 10, 14):
    m = zg.sep < lim
    if m.sum() > 3:
        f = 10 ** (-0.4 * zg.dG[m]); pred = zg.sep[m] * f / (1 + f)
        k2, c2 = np.polyfit(pred, zg.proj[m], 1)
        print('   inside %2d " : slope %+.2f, intercept %+.3f " (n=%d)' % (lim, k2, c2, int(m.sum())))

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].axhline(0, color='black', lw=1)
ax[0].axvline(PSF_FWHM_AS, color='tab:red', ls='--', lw=1.2,
              label='PSF FWHM %.1f "' % PSF_FWHM_AS)
ax[0].axvline(10, color='tab:green', ls=':', lw=1.2, label='the 10 " default')
ax[0].scatter(d.sep, d.proj, s=18, color='tab:blue', label='toward the companion')
ax[0].scatter(d.sep, d.perp, s=14, color='lightgray', zorder=1, label='across it (control)')
for a, b in zip(edges[:-1], edges[1:]):
    m = (d.sep >= a) & (d.sep < b)
    if m.sum() >= 2:
        ax[0].errorbar([(a + b) / 2], [d.proj[m].mean()],
                       yerr=[np.std(d.proj[m], ddof=1) / np.sqrt(m.sum())],
                       fmt='s', color='black', ms=5, capsize=3, zorder=5)
ax[0].set_xlabel('separation of the nearest catalogue neighbour (arcsec)')
ax[0].set_ylabel('mean residual projected (arcsec)')
ax[0].set_title('Is the centroid pulled toward the companion?', fontsize=11)
ax[0].legend(fontsize=8)
lim = max(0.6, float(np.abs(d.proj).max()) * 1.1)
ax[0].set_ylim(-lim, lim)

ax[1].axhline(0, color='black', lw=1)
mm = d.sep < 30
ax[1].scatter(d.predicted[mm], d.proj[mm], s=22, color='tab:blue')
xx = np.linspace(0, float(d.predicted[mm].max()) * 1.05, 50)
ax[1].plot(xx, xx, color='tab:red', ls='--', lw=1.2, label='the blend model, slope 1')
if mm.sum() > 3:
    ax[1].plot(xx, k * xx + c, color='black', lw=1.6, label='fitted slope %.2f' % k)
ax[1].set_xlabel('predicted blend shift, separation $\\times$ f$_2$/(f$_1$+f$_2$)  (arcsec)')
ax[1].set_ylabel('measured shift toward the companion (arcsec)')
ax[1].set_title('Does it follow the flux-weighted prediction?', fontsize=11)
ax[1].legend(fontsize=8)
fig.suptitle('Choosing double_star_cutoff on cell 2 \u2014 Mexico 2024 Station 1, %d stars with a '
             'neighbour inside %.0f "' % (len(d), SEARCH_AS), fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'double_cutoff_test.png'), dpi=140)
plt.close(fig)
d.sort_values('sep').to_csv(os.path.join(OUT, 'station1_neighbour_test.csv'), index=False)
print('\ncharts -> %s' % OUT)
