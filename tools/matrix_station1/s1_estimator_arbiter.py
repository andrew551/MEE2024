"""Cell 2: which centroid estimator is right, rather than splitting the difference.

(Superseded in scope by `magnitude_independence_all_cells.py`, which runs the same test on
all three eclipse datasets. Kept because it also reports the magnitude-bin cross-check and
the averaging arithmetic.)

Douglas, 2026-09-03, quoting Bruns 2018 section 2.3: "Astrometrica determines centroids by
fitting a Gaussian curve to the pixel values across a radial center while MaxIm DL is based
on a moment calculation", and 2.6, where L = 1.7338 (Astrometrica) and 1.7658 (MaxIm DL)
were averaged, weighted by the mean centroid rms, to the reported 1.7520. "Does it make any
sense for us to average the old and new method when considering the final result for the
Mexico 2024 dataset (as Bruns did for his 2017 data)?"

Averaging is the right move when you cannot tell which estimator is better. This asks whether
we can, on Station 1's own eclipse field, by the one diagnostic that separates a biased
estimator from an unbiased one without knowing the answer in advance.

**The test.** The deflection depends only on the light ray's impact parameter, so a G 8 star
and a G 12 star at the same angular distance from the Sun are bent by the same angle.

(Briefly called "the magnitude-independence test" when first written. **That term is retired as
deceptive**: these campaigns shoot through a red filter and are essentially monochromatic, so
there is no chromatic leverage in the data at all. What varies is apparent MAGNITUDE. The name
is the magnitude-independence test -- see `magnitude_independence_all_cells.py`.) So fit an extra column,

    dx = ... + [L + Lmag*(G - Gref)] * (R_sun/R) * u_x

and read `Lmag`, the arcseconds of apparent deflection per magnitude. For a clean estimator
it is zero. For one whose centroid error grows toward the faint end -- which is what a
footprint moment does on a steep coronal background, because the residual after an annular
subtraction scales inversely with the star's own flux -- it is not, and its sign and size say
how much of the fitted L is estimator rather than gravity.

This is a stronger test than fitting L in magnitude bins, because it uses every star and
costs one parameter rather than splitting 150 stars into subsets of 50. Both are reported.

Run against whichever reference both stacks share, so the only difference between the two
columns of the table is the estimator.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

EC = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_convention"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'
MAGCUT, RCUT, RMAX, GREF = 12.0, 2.0, 9.0, 10.0
sun = get_sun(Time(T_MID, scale='utc'))
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)


def load(est, ref='A_2024_17field_moments'):
    z = glob.glob(os.path.join(EC, est, ref, '**', 'distortion_data*.zip'), recursive=True)
    if not z:
        return None
    zf = zipfile.ZipFile(z[0])
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    d = d[d['flag_is_outlier'] == False].copy()
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cxm = np.c_[X, Y, np.ones(len(d))]
    d['dx'] = (ox@ax - cxm@ax)*PS; d['dy'] = (ox@ay - cxm@ay)*PS
    d['rx'] = (d.px.values-SPX)*PS; d['ry'] = (d.py.values-SPY)*PS
    d['R'] = np.hypot(d.rx, d.ry); d['Rsun'] = d.R/RS
    return d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)].reset_index(drop=True)


def fit(d, magterm=False, nuis=0):
    p, q, r = d.px.values, d.py.values, d.R.values
    ux, uy = d.rx.values/r, d.ry.values/r
    dx = d.dx.values - np.median(d.dx.values); dy = d.dy.values - np.median(d.dy.values)
    mag = d.magV.values
    n = len(d); Z = np.zeros(n)
    xs, ys = (p-NX/2)*PS, (q-NY/2)*PS; xn, yn = (p-NX/2)/(NX/2), (q-NY/2)/(NX/2)
    cxl = [np.ones(n), Z, -ys, xs, ux*RS/r]; cyl = [Z, np.ones(n), xs, ys, uy*RS/r]
    lab = ['N1', 'N2', 'Th', 'S', 'L']
    if magterm:
        cxl.append(ux*RS/r*(mag-GREF)); cyl.append(uy*RS/r*(mag-GREF)); lab.append('Lmag')
    if nuis:
        for i in range(nuis+1):
            for jj in range(nuis+1-i):
                if i == 0 and jj == 0:
                    continue
                cxl.append(Z); cyl.append(xn**i*yn**jj); lab.append('v')
    M = np.vstack([np.column_stack(cxl), np.column_stack(cyl)])
    sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
    c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
    res = b - Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
    e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn))))
    c, e = c/sc, e/sc
    out = dict(n=n, rms=np.sqrt(s2))
    for k in ('L', 'Lmag', 'S'):
        if k in lab:
            out[k] = c[lab.index(k)]; out['e'+k] = e[lab.index(k)]
    return out


tabs = {k: load(k) for k in ('windowed', 'moments')}
tabs = {k: v for k, v in tabs.items() if v is not None}
if not tabs:
    raise SystemExit('no eclipse stacks found')

print('=== the magnitude-independence test: apparent deflection per magnitude ===')
print('the deflection is magnitude-independent, so Lmag should be zero for a clean estimator\n')
print('%-12s %5s | %-24s | %-28s | %8s' % ('estimator', 'stars', 'L, no mag term', 'L and Lmag fitted together', 'rms'))
res = {}
for est, d in tabs.items():
    a = fit(d, magterm=False)
    b = fit(d, magterm=True)
    res[est] = (a, b)
    print('%-12s %5d | %+7.3f +- %.3f"        | L %+7.3f +- %.3f", Lmag %+6.3f +- %.3f"/mag | %.3f"'
          % (est, a['n'], a['L'], a['eL'], b['L'], b['eL'], b['Lmag'], b['eLmag'], a['rms']))
for est, (a, b) in res.items():
    z = b['Lmag']/b['eLmag']
    print('   %-10s Lmag is %.1f sigma from zero%s' % (est, abs(z), '' if abs(z) < 2 else '  <-- significant'))

print('\n=== the same thing by independent magnitude bins (a cross-check, not the test) ===')
BINS = [(5, 9.5), (9.5, 10.5), (10.5, 12)]
print('%-12s' % 'estimator' + ''.join('%22s' % ('G %.1f-%.1f' % bb) for bb in BINS))
for est, d in tabs.items():
    row = ''
    for lo, hi in BINS:
        t = d[(d.magV >= lo) & (d.magV < hi)]
        if len(t) < 25:
            row += '%22s' % ('n=%d, too few' % len(t)); continue
        f = fit(t)
        row += '%22s' % ('%+.2f +- %.2f (%d)' % (f['L'], f['eL'], f['n']))
    print('%-12s' % est + row)

print('\n=== what this means for averaging the two estimators ===')
if len(res) == 2:
    lw, lm = res['windowed'][0]['L'], res['moments'][0]['L']
    ew, em = res['windowed'][0]['rms'], res['moments'][0]['rms']
    print('  windowed  L = %.3f", per-star rms %.3f"' % (lw, ew))
    print('  moments   L = %.3f", per-star rms %.3f"' % (lm, em))
    print('  simple mean                %.3f"' % ((lw+lm)/2))
    print('  Bruns\' 1/rms weighting     %.3f"' % ((lw/ew + lm/em)/(1/ew + 1/em)))
    print('  inverse-variance weighting %.3f"' % ((lw/ew**2 + lm/em**2)/(1/ew**2 + 1/em**2)))
    print('  the spread between them, %.3f", is %.0f %% of L -- Bruns\' two programs differed'
          ' by 0.032" = 1.8 %%.' % (abs(lm-lw), 100*abs(lm-lw)/((lw+lm)/2)))
print('\nGR = 1.751".')
