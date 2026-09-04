"""Station 1: WHY does flat-fielding move the eclipse centroids by ~250 mas?

`s1_caldecomp_fit.py` measured the effect: on the corona-subtracted 0.4 s block, flat-fielding
moves every centroid ~250 mas relative to the un-flatted stack (the dark moves them 37 mas) and
L by -0.29" on identical stars. PSF injection puts the flat's own pixel-response effect at
2.4-3.3 mas, so the pixel response is not what acts. This is the cheap test that separates the
candidates, before anything is re-stacked.

  A. THE PIXEL RESPONSE, computed rather than injected. A smooth multiplicative gradient across
     a PSF shifts a windowed centroid by about sigma_psf^2 * d(ln F)/dx. Sampled at every star
     from the master flat, this is the prediction the injection made, in mas.
  B. WHAT THE SHIFT TRACKS. Per star, the flat-minus-neither shift against: the local flat value
     and gradient (a mechanism through the corona model or the pixel response tracks these),
     the star's brightness (a mechanism through detection or noise tracks this), and the
     distance from the Sun (a mechanism through the coronal background tracks this).
  C. IS IT THE ALIGNER? The four arms are aligned independently, so a per-frame alignment
     difference is a candidate. A whole-frame shift is absorbed by the fit's offset, so what
     matters is the SPREAD of the per-frame shifts -- if the flat arm's alignment is noisier,
     the stack is blurred differently and every star's centroid moves.
  D. IS IT REPEATABLE OR IS IT NOISE? The x and y components of the shift are compared with the
     stage-2 per-star residual of each arm. A 250 mas systematic and a 250 mas noise term look
     the same in a mean; they differ in whether the shift correlates with anything at all.

Writes station1_record/eclipse_caldecomp/flat_mechanism.csv and prints the table.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_caldecomp_fit import table, NX, NY, PS, CD, DF, TMID, MAGCUT, RCUT, RMAX  # noqa: E402

ARMS = {'neither': os.path.join(CD, 'neither'),
        'dark only': os.path.join(CD, 'darkonly'),
        'flat only': os.path.join(CD, 'flatonly'),
        'dark + flat': os.path.join(r"D:/MEE2024 output/MEE_output/station1_record",
                                    'eclipse_corona', '0p4s_1812')}
PSF_SIGMA_PX = 3.2      # measured on these stacks (FWHM 7.5 px at centre)


def latest(root):
    hit = sorted(glob.glob(os.path.join(root, 'stage2', '**', 'distortion_data*.zip'), recursive=True))
    return hit[-1] if hit else None


def stack_meta(root):
    z = glob.glob(os.path.join(root, 'centroid_data*.zip'))
    if not z:
        return None, None
    zf = zipfile.ZipFile(z[0])
    r = json.load(zf.open('results.txt'))
    c = pd.read_csv(zf.open('STACKED_CENTROIDS_DATA.csv')); c.columns = [k.strip() for k in c.columns]
    return r, c


# ---------------------------------------------------------------- the master flat
flat = fits.getdata(os.path.join(DF, 'master_flat.fits')).astype(np.float32)
flat = flat / np.median(flat[flat > 0.2])
sm = gaussian_filter(np.where(flat > 0.2, flat, 1.0), 8.0)     # the smooth part, dust and vignetting
gy, gx = np.gradient(sm)


def sample(a, px, py):
    return a[np.clip(np.round(py).astype(int), 0, a.shape[0]-1),
             np.clip(np.round(px).astype(int), 0, a.shape[1]-1)]


# ---------------------------------------------------------------- the tables
tabs = {}
for name, root in ARMS.items():
    zp = latest(root)
    if zp:
        d = table(zp, TMID)
        tabs[name] = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)].set_index('key')
common = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
base = tabs['neither'].loc[common]
print('=== %d stars in the science set of all four arms ===' % len(common))

# ---------------------------------------------------------------- A. the pixel response, computed
fv = sample(sm, base.px.values, base.py.values)
gxs, gys = sample(gx, base.px.values, base.py.values), sample(gy, base.px.values, base.py.values)
grad = np.hypot(gxs, gys)/fv                                    # d(ln F)/dpixel
pred_mas = 1000*PSF_SIGMA_PX**2 * grad * PS
print('\nA. the pixel response, computed from the master flat at each star')
print('   local flat value %.3f-%.3f; |grad ln F| median %.2e /px, max %.2e' % (fv.min(), fv.max(), np.median(grad), grad.max()))
print('   predicted centroid shift sigma_psf^2 * grad(ln F): median %.1f mas, max %.1f mas'
      % (np.median(pred_mas), pred_mas.max()))
print('   (the injection test said 2.4-3.3 mas; the MEASURED flat-minus-neither shift is ~250 mas)')

# ---------------------------------------------------------------- the measured shifts
rows = pd.DataFrame(index=base.index)
rows['px'], rows['py'], rows['magV'], rows['Rsun'] = base.px, base.py, base.magV, base.Rsun
rows['flat'], rows['grad_lnF'], rows['pred_mas'] = fv, grad, pred_mas
for name in ('dark only', 'flat only', 'dark + flat'):
    a = tabs[name].loc[common]
    sx, sy = a.dx.values-base.dx.values, a.dy.values-base.dy.values
    n = len(sx); Z = np.zeros(n)
    xs, ys = (base.px.values-NX/2)*PS, (base.py.values-NY/2)*PS
    M = np.vstack([np.column_stack([np.ones(n), Z, -ys, xs]), np.column_stack([Z, np.ones(n), xs, ys])])
    c, *_ = np.linalg.lstsq(M, np.concatenate([sx, sy]), rcond=None)
    res = np.concatenate([sx, sy]) - M@c
    key = name.replace(' + ', '').replace(' ', '_')
    rows[key+'_dx'], rows[key+'_dy'] = res[:n], res[n:]
    rows[key+'_mag'] = np.hypot(res[:n], res[n:])

# ---------------------------------------------------------------- B. what the shift tracks
print('\nB. what the flat-minus-neither shift tracks (Spearman rho over %d stars)' % len(common))
sh = rows.flat_only_mag.values


def rho(a, b):
    return float(pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)), method='spearman'))


for label, v in (('local flat value', rows.flat), ('|grad ln F| at the star', rows.grad_lnF),
                 ('predicted pixel-response shift', rows.pred_mas), ('G magnitude (fainter = larger)', rows.magV),
                 ('distance from the Sun', rows.Rsun), ('distance from the frame centre',
                  np.hypot(rows.px-NX/2, rows.py-NY/2))):
    print('   |shift| vs %-34s rho = %+.2f' % (label, rho(sh, v)))
print('   for contrast, the DARK-minus-neither shift:')
for label, v in (('|grad ln F| at the star', rows.grad_lnF), ('G magnitude (fainter = larger)', rows.magV)):
    print('   |shift| vs %-34s rho = %+.2f' % (label, rho(rows.dark_only_mag.values, v)))
print('   by brightness quartile (flat only):')
q = pd.qcut(rows.magV, 4, labels=False)
for i in range(4):
    k = q == i
    print('      G %.1f-%.1f  n=%3d  |shift| %5.0f mas' % (rows.magV[k].min(), rows.magV[k].max(), k.sum(), 1000*rows.flat_only_mag[k].mean()))

# ---------------------------------------------------------------- C. the aligner
print('\nC. the per-frame alignment of each arm (shifts_px is (y, x))')
for name, root in ARMS.items():
    r, _ = stack_meta(root)
    if not r:
        continue
    s = np.array(r.get('alignment', {}).get('shifts_px', [[0, 0]]), float)
    print('   %-12s %3d frames, spread (rms about the mean) y %.3f px, x %.3f px; span %.2f px'
          % (name, len(s), s[:, 0].std(), s[:, 1].std(),
             np.max(np.hypot(*(s[:, None, :]-s[None, :, :]).T)) if len(s) > 1 else 0.0))

# ---------------------------------------------------------------- D. signal or noise
print('\nD. is the shift structure or scatter?')
for name in ('dark only', 'flat only', 'dark + flat'):
    key = name.replace(' + ', '').replace(' ', '_')
    dx, dy = rows[key+'_dx'].values, rows[key+'_dy'].values
    # a systematic field pattern is smooth: neighbouring stars move together. Nearest-neighbour
    # correlation of the shift is the test that needs no model of the mechanism.
    p = np.c_[rows.px.values, rows.py.values]
    d2 = ((p[:, None, :]-p[None, :, :])**2).sum(-1); np.fill_diagonal(d2, np.inf)
    nn = np.argmin(d2, 1)
    cc = (np.corrcoef(dx, dx[nn])[0, 1] + np.corrcoef(dy, dy[nn])[0, 1])/2
    print('   %-12s |shift| median %5.0f mas; nearest-neighbour correlation of the shift %+.2f (median separation %.0f px)'
          % (name, 1000*np.median(rows[key+'_mag']), cc, np.median(np.sqrt(d2[np.arange(len(nn)), nn]))))

# ---------------------------------------------------------------- E. is dL bigger than the shift implies?
# The arms share their photons, so their L estimates are correlated and the DIFFERENCE in L is
# driven by the difference in centroids -- the per-star shift measured above. If that shift is
# per-star measurement noise (isotropic, no preferred direction on the sky), dL has a
# predictable spread: keep each star's shift AMPLITUDE, randomise its direction, add it to the
# base arm and refit. A measured dL inside that distribution is not a systematic; one outside
# it is. This is the test that decides whether the flat's "0.3 arcsec lever" is a bias or a
# redrawn measurement error.
print('\nE. is the change in L larger than the per-star shift implies?')
from s1_caldecomp_fit import fit as fit_L
rng = np.random.default_rng(23)
base_full = tabs['neither'].loc[common].reset_index()
L0 = fit_L(base_full, vet=False)[0]['L']
mc = {}
for name in ('dark only', 'flat only', 'dark + flat'):
    key = name.replace(' + ', '').replace(' ', '_')
    amp = rows[key + '_mag'].values
    L1 = fit_L(tabs[name].loc[common].reset_index(), vet=False)[0]['L']
    draws = []
    for _ in range(400):
        th = rng.uniform(0, 2*np.pi, len(amp))
        d2 = base_full.copy()
        d2['dx'] = d2.dx.values + amp*np.cos(th)
        d2['dy'] = d2.dy.values + amp*np.sin(th)
        draws.append(fit_L(d2, vet=False)[0]['L'])
    sd = float(np.std(draws, ddof=1))
    mc[name] = (L1 - L0, sd)
    print('   %-12s measured dL = %+.3f"; isotropic-noise prediction +-%.3f"  ->  %.1f sigma'
          % (name, L1 - L0, sd, abs(L1 - L0)/sd))
print('   (the arms share their photons, so this is the right null: same stars, same sky, a')
print('    redrawn per-star measurement error of the amplitude actually observed)')

rows.to_csv(os.path.join(CD, 'flat_mechanism.csv'))
print('\n->', os.path.join(CD, 'flat_mechanism.csv'))
