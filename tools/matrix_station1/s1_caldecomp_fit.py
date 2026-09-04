"""Station 1: dark against flat on the corona-subtracted 0.4 s eclipse block.

Four re-stacks of the same raw block, all with the disk occulter and the per-frame coronal
subtraction on, differing only in calibration: neither, dark only, flat only, dark + flat.
All four were fitted by `s1_refit_driver.py caldecomp` (and `s1_eclipse_corona.py` for the
dark + flat arm) against the same quintic seventeen-field reference, constant-only. This tool
reads those stage-2 tables and answers three things:

  1. each arm on its own stars -- science set, vet, L -- and how many of its stacked
     detections sit on a hot pixel of the master dark (the `s1_hotpixel_risk.py` test);
  2. each arm on the stars common to all four, so the L differences are on identical stars;
  3. WHERE the flat acts: the per-star shift of each arm relative to `neither`, with the
     offset, rotation and scale a science fit would absorb removed, decomposed into radial
     and tangential components about the Sun and binned in R_sun.

The first run of this decomposition lived in an inline script and its numbers survived only
as a printout; this file exists so that the record has an artifact. It writes
station1_record/eclipse_caldecomp/caldecomp_{arms,common,radial}.csv.

`table` and `fit` are copied from `s1_eclipse_corona.py`, which cannot be imported without
running its stacks; `coincides` from `s1_hotpixel_risk.py`.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

REC = r"D:/MEE2024 output/MEE_output/station1_record"
CD = os.path.join(REC, 'eclipse_caldecomp')
DF = os.path.join(REC, 'darks_flats')
NX, NY, PS = 9576, 6388, 1.84847
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0
HOT_ADU = 200
TMID = '18:13:00'
ARMS = [('neither', os.path.join(CD, 'neither')),
        ('dark only', os.path.join(CD, 'darkonly')),
        ('flat only', os.path.join(CD, 'flatonly')),
        ('dark + flat', os.path.join(REC, 'eclipse_corona', '0p4s_1812'))]
BINS = [(2, 3), (3, 4), (4, 6), (6, 9)]


def table(zp, tmid):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    d = d[d['flag_is_outlier'] == False].copy()
    sun = get_sun(Time('2024-04-08T' + tmid + (':00' if len(tmid) == 5 else ''), scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cm = np.c_[X, Y, np.ones(len(d))]
    d['dx'] = (ox@ax - cm@ax)*PS; d['dy'] = (ox@ay - cm@ay)*PS
    d['rx'] = (d.px.values-SPX)*PS; d['ry'] = (d.py.values-SPY)*PS
    d['R'] = np.hypot(d.rx, d.ry); d['Rsun'] = d.R/RS; d['RS'] = RS
    # a key that is the same star in every arm: the catalogue position, to 0.1 mas
    d['key'] = [('%.7f_%.7f' % (a, b)) for a, b in zip(d['RA(catalog)'], d['DEC(catalog)'])]
    return d


def fit(d, vet=True):
    """Method 2, constant-only distortion already applied upstream: offset, rotation, scale
    and L. Returns the fit and the rows that survived the vet."""
    d = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)].copy()
    for _ in range(4 if vet else 1):
        p, q, r = d.px.values, d.py.values, d.R.values
        ux, uy = d.rx.values/r, d.ry.values/r; RS = d.RS.values
        dx = d.dx.values-np.median(d.dx.values); dy = d.dy.values-np.median(d.dy.values)
        n = len(d); Z = np.zeros(n); xs, ys = (p-NX/2)*PS, (q-NY/2)*PS
        M = np.vstack([np.column_stack([np.ones(n), Z, -ys, xs, ux*RS/r]),
                       np.column_stack([Z, np.ones(n), xs, ys, uy*RS/r])])
        sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
        res = b - Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn)))); c, e = c/sc, e/sc
        per = np.hypot(res[:n], res[n:])
        lim = max(4.0*1.4826*np.median(np.abs(per-np.median(per))) + np.median(per), 0.6)
        if not vet or (per < lim).all():
            break
        d = d[per < lim]
    return dict(n=n, L=c[4], eL=e[4], rms=np.sqrt(s2), rin=d.Rsun.min()), d


def coincides(px, py, used, hotset):
    out = np.zeros(len(px), bool)
    for dx, dy in used:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                xs = np.round(px).astype(int) - dx + ox
                ys = np.round(py).astype(int) - dy + oy
                out |= np.array([(int(b), int(a)) in hotset for a, b in zip(xs, ys)])
    return out


def hot_count(root, hotset):
    z = glob.glob(os.path.join(root, 'centroid_data*.zip'))
    if not z:
        return None, None
    zf = zipfile.ZipFile(z[0])
    r = json.load(zf.open('results.txt'))
    c = pd.read_csv(zf.open('STACKED_CENTROIDS_DATA.csv')); c.columns = [k.strip() for k in c.columns]
    sh = np.array(r.get('alignment', {}).get('shifts_px', [[0, 0]]))
    used = sorted(set(map(tuple, np.round(sh).astype(int))))
    return len(c), int(coincides(c.px.values, c.py.values, used, hotset).sum())


def latest_zip(root):
    hit = sorted(glob.glob(os.path.join(root, 'stage2', '**', 'distortion_data*.zip'), recursive=True))
    return hit[-1] if hit else None


# ---------------------------------------------------------------- hot pixels of the master dark
bias = fits.getdata(os.path.join(DF, 'master_bias.fits')).astype(np.float32)
dark = fits.getdata(os.path.join(DF, 'master_dark-400ms.fits')).astype(np.float32)
hy, hx = np.nonzero((dark - bias) > HOT_ADU)
hotset = set(zip(hy.tolist(), hx.tolist()))
del bias, dark

# ---------------------------------------------------------------- 1. each arm on its own stars
print('=== 1. each arm on its own stars (quintic 17-field reference, constant-only, vetted) ===')
tabs, rows = {}, []
for name, root in ARMS:
    zp = latest_zip(root)
    if not zp:
        print('  %-12s no stage 2' % name); continue
    d = table(zp, TMID)
    f, kept = fit(d)
    nc, nh = hot_count(root, hotset)
    tabs[name] = (d, kept)
    rows.append(dict(arm=name, centroids=nc, on_hot_pixel=nh,
                     hot_pct=(100.0*nh/nc if nc else np.nan), matched=len(d), stars=f['n'],
                     L=f['L'], eL=f['eL'], residual=f['rms'], innermost_Rsun=f['rin']))
    print('  %-12s %4d centroids, %3d on a hot pixel (%.1f %%); %3d matched, %3d in the science set;'
          ' L = %.3f +- %.3f", residual %.3f", innermost %.2f R_sun'
          % (name, nc, nh, 100.0*nh/nc, len(d), f['n'], f['L'], f['eL'], f['rms'], f['rin']))
pd.DataFrame(rows).to_csv(os.path.join(CD, 'caldecomp_arms.csv'), index=False)

# ---------------------------------------------------------------- 2. the common stars
common = None
for name, (d, kept) in tabs.items():
    common = set(kept.key) if common is None else common & set(kept.key)
print('\n=== 2. the %d stars in every arm\'s vetted science set, refitted without a vet ===' % len(common))
rows = []
for name, (d, kept) in tabs.items():
    f, _ = fit(d[d.key.isin(common)], vet=False)
    rows.append(dict(arm=name, stars=f['n'], L=f['L'], eL=f['eL'], residual=f['rms']))
    print('  %-12s %3d stars  L = %.3f +- %.3f"  residual %.3f"' % (name, f['n'], f['L'], f['eL'], f['rms']))
pd.DataFrame(rows).to_csv(os.path.join(CD, 'caldecomp_common.csv'), index=False)

# ---------------------------------------------------------------- 3. where each arm moves the stars
print('\n=== 3. per-star shift relative to "neither", offset/rotation/scale removed, on the common stars ===')
print('    radial: + is away from the Sun.  mas.')
base = tabs['neither'][0].set_index('key').loc[sorted(common)]
rows = []
for name, (d, kept) in tabs.items():
    if name == 'neither':
        continue
    a = d.set_index('key').loc[base.index]
    sx, sy = (a.dx.values - base.dx.values), (a.dy.values - base.dy.values)
    p, q = base.px.values, base.py.values
    n = len(p); Z = np.zeros(n); xs, ys = (p-NX/2)*PS, (q-NY/2)*PS
    M = np.vstack([np.column_stack([np.ones(n), Z, -ys, xs]), np.column_stack([Z, np.ones(n), xs, ys])])
    c, *_ = np.linalg.lstsq(M, np.concatenate([sx, sy]), rcond=None)
    res = np.concatenate([sx, sy]) - M@c
    rx, ry = res[:n], res[n:]
    ux, uy = base.rx.values/base.R.values, base.ry.values/base.R.values
    rad, tan = rx*ux + ry*uy, -rx*uy + ry*ux
    mag = np.hypot(rx, ry)
    print('  --- %s minus neither: median |shift| %.0f mas over %d stars' % (name, 1000*np.median(mag), n))
    for lo, hi in BINS:
        k = (base.Rsun.values >= lo) & (base.Rsun.values < hi)
        if k.sum() == 0:
            continue
        rows.append(dict(arm=name, Rsun_lo=lo, Rsun_hi=hi, n=int(k.sum()),
                         radial_mas=1000*rad[k].mean(), tangential_mas=1000*tan[k].mean(),
                         abs_shift_mas=1000*mag[k].mean()))
        print('      %d-%d R_sun  n=%3d  radial %+5.0f  tangential %+5.0f  |shift| %4.0f'
              % (lo, hi, k.sum(), 1000*rad[k].mean(), 1000*tan[k].mean(), 1000*mag[k].mean()))
pd.DataFrame(rows).to_csv(os.path.join(CD, 'caldecomp_radial.csv'), index=False)
print('\n->', CD)
