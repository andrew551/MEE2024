"""Does the estimator difference depend on SKY BRIGHTNESS? The transfer-mismatch question.

Douglas, 2026-09-03: "If we moved both the day-time and night-time data to the new method,
isn't there going to be a slight mismatch caused by the two different sky brightness? Do we
have to worry about such a mismatch with the Mexico data also?"

The concern is exact and it is the right one to have. A distortion model is fitted on a
calibration field and applied to an eclipse field. If an estimator carries a bias that depends
on the sky behind the star, then the calibration absorbs `b(sky_cal)` into the fitted
polynomial and the eclipse field contributes `b(sky_ecl)`, and what reaches L is the
**difference**. Two fields each individually well behaved can still transfer badly if their
skies differ, and eclipse fields differ from every calibration field by construction: the
corona is the brightest sky any of these campaigns ever works against.

The Mexico eclipse field is the ideal place to measure this, because it contains its own range
of sky brightness. The local background at the matched stars spans a factor of eight, from a
few hundred ADU/s far out to a few thousand near the Sun, all in one frame with one estimator
pair on one stack. So the question "does the estimator difference grow with sky?" can be asked
without comparing fields at all, and therefore without confounding it with exposure, star
density, seeing or epoch.

Method: take the stars matched in BOTH the windowed and the moments re-centroiding of the same
2024 calibrated stack, measure each star's local sky from that stack with an annulus, and bin
the windowed-minus-moments centroid difference by local sky. The difference is decomposed into
the radial and tangential directions about the Sun, because only the radial part can masquerade
as deflection.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

TIERS = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_tiers"
STACK = r"D:/MEE2024 output/Station 1/eclipse fields/CENTROID_OUTPUT20240416232626/STACKED20240416232626.fit"
NX, NY, PS = 9576, 6388, 1.84847
REF = 'stage2_F_17field_windowed'
sun = get_sun(Time('2024-04-08T18:13:00', scale='utc'))
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)


def table(est):
    z = glob.glob(os.path.join(TIERS, '0p4s_1812_%s' % est, REF, '**', 'distortion_data*.zip'), recursive=True)
    if not z:
        return None
    zf = zipfile.ZipFile(z[0])
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    return d[d['flag_is_outlier'] == False][['ID', 'px', 'py', 'magV']].copy()


W, M = table('windowed'), table('moments')
if W is None or M is None:
    raise SystemExit('need both re-centroidings of the 0.4 s block')
m = W.merge(M, on='ID', suffixes=('_w', '_m'))
print('%d stars matched in both estimators on the same stacked image' % len(m))

img = fits.getdata(STACK).astype(np.float32)
B = 16
yy, xx = np.mgrid[-B:B+1, -B:B+1]
rr = np.hypot(xx, yy)
ann = (rr > 10) & (rr <= 16)
sky, peak = [], []
for _, r in m.iterrows():
    iy, ix = int(round(r.py_w)), int(round(r.px_w))
    if iy < B or ix < B or iy >= NY-B or ix >= NX-B:
        sky.append(np.nan); peak.append(np.nan); continue
    sub = img[iy-B:iy+B+1, ix-B:ix+B+1]
    bg = float(np.median(sub[ann]))
    sky.append(bg)
    peak.append(float(sub[rr <= 3].max()) - bg)
m['sky'] = sky
m['peak'] = peak
m = m[np.isfinite(m.sky) & np.isfinite(m.peak) & (m.peak > 0)]

# Sun position from the windowed fit's own geometry, via the catalogue-matched affine
zw = glob.glob(os.path.join(TIERS, '0p4s_1812_windowed', REF, '**', 'distortion_data*.zip'), recursive=True)[0]
zf = zipfile.ZipFile(zw)
full = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
full.columns = [c.strip() for c in full.columns]
ra0, de0 = full['RA(catalog)'].mean(), full['DEC(catalog)'].mean()
X = (full['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = full['DEC(catalog)'].values-de0
A = np.c_[X, Y, np.ones_like(X)]
ax, *_ = np.linalg.lstsq(A, full.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, full.py.values, rcond=None)
sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)

dx = (m.px_w.values - m.px_m.values)*PS
dy = (m.py_w.values - m.py_m.values)*PS
rx, ry = (m.px_w.values-SPX)*PS, (m.py_w.values-SPY)*PS
R = np.hypot(rx, ry)
m['Rsun'] = R/RS
m['rad'] = (dx*rx/R + dy*ry/R)*1000
m['tan'] = (-dx*ry/R + dy*rx/R)*1000
m['snr'] = m.peak/np.sqrt(np.clip(m.sky, 1, None))

print('local sky at the stars: %.0f to %.0f ADU (a factor of %.1f), all in one frame'
      % (m.sky.min(), m.sky.max(), m.sky.max()/max(m.sky.min(), 1)))
print('\n=== windowed minus moments, binned by LOCAL SKY ===')
print('%-18s %6s %10s %10s %10s %10s' % ('local sky (ADU)', 'n', 'radial', 'tangential', 'median R', 'median G'))
qs = np.quantile(m.sky, [0, 0.25, 0.5, 0.75, 1.0])
for lo, hi in zip(qs[:-1], qs[1:]):
    k = (m.sky >= lo) & (m.sky <= hi)
    if k.sum() < 5:
        continue
    print('%-18s %6d %+9.0f  %+9.0f  %9.2f  %9.1f'
          % ('%.0f-%.0f' % (lo, hi), k.sum(), m.rad[k].mean(), m['tan'][k].mean(),
             m.Rsun[k].median(), m.magV_w[k].median()))
print('   (radial and tangential in mas; the tangential column is the control -- an estimator')
print('    difference driven by the sky should show in radial about the Sun, not tangential)')

print('\n=== the same, binned by SIGNAL TO NOISE, which is what a sky-driven bias really tracks ===')
print('%-18s %6s %10s %10s %10s' % ('peak/sqrt(sky)', 'n', 'radial', 'tangential', 'median G'))
qs = np.quantile(m.snr, [0, 0.25, 0.5, 0.75, 1.0])
for lo, hi in zip(qs[:-1], qs[1:]):
    k = (m.snr >= lo) & (m.snr <= hi)
    if k.sum() < 5:
        continue
    print('%-18s %6d %+9.0f  %+9.0f  %9.1f' % ('%.0f-%.0f' % (lo, hi), k.sum(), m.rad[k].mean(), m['tan'][k].mean(), m.magV_w[k].median()))

good = np.isfinite(m.rad) & np.isfinite(m.sky)
c1 = np.corrcoef(np.log10(m.sky[good]), m.rad[good])[0, 1]
c2 = np.corrcoef(np.log10(m.snr[good]), m.rad[good])[0, 1]
c3 = np.corrcoef(m.magV_w[good], m.rad[good])[0, 1]
print('\ncorrelations of the radial windowed-minus-moments difference with:')
print('   log local sky   r = %+.3f' % c1)
print('   log S/N         r = %+.3f' % c2)
print('   magnitude       r = %+.3f' % c3)
print('\nIf the sky drives it, the sky and S/N correlations lead. If it is the PSF alone, they')
print('do not, and only magnitude and field position matter. That distinction is what decides')
print('whether a calibration field at a different sky brightness transfers cleanly.')
m.to_csv(os.path.join(TIERS, 'estimator_vs_sky.csv'), index=False)
print('\n->', os.path.join(TIERS, 'estimator_vs_sky.csv'))
