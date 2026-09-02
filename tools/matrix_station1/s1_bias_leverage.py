"""Cell 2, Station 1: what the moment convention's magnitude-dependent bias is worth in L,
propagated through Station 1's OWN eclipse geometry, under Method 1 and Method 2.

The companion diagnostic (s1_magbias_diagnostic.py) measures the bias on the 2024-era zenith
fits: purely radial about the frame centre, growing with field radius, +3 / +12 / +22 / +31 /
+10 mas per magnitude in the five radius bins, same sign in 17 of 17 fields (faint stars
outward, bright inward -- the opposite sign to Leon, and about a fifth of its size).

That number decides the convention question only once it is expressed in L. So: take the
eclipse field's actual matched stars (the 2024 constant-only stage 2, 182 stars), the Sun's
actual frame position from the ephemeris at 18:12 UTC through the field's own affine, the
science cuts (G <= 12, R > 2 R_sun), and inject the measured bias field as the ONLY
displacement -- a pure systematic with true L = 0 -- then fit:

  * Method 1 (scale imported), which is how the bias would enter a Bruns-style chain;
  * Method 2 (scale free), which is how Station 1 is actually reduced.

Why the two differ: the bias is radial about the FRAME centre and the Sun sits only ~480 px
from it, so the two radial patterns are nearly concentric. Method 1 has only the 1/r column
to absorb a radial systematic and takes it as deflection; Method 2 has a scale column that
absorbs the bulk of any frame-centred radial field, and L takes only the remainder.

Also reported: the same bias with the sign flipped, and at five times the amplitude (Leon's
class), so the reader can see where a different optic would land.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

S1 = r"D:/MEE2024 output/Station 1"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'
MAGCUT, RCUT, GREF = 12.0, 2.0, 10.0
# the measured bias, mas per magnitude, by distance from the frame centre (s1_magbias_diagnostic.py)
SLOPES = [((0, 1500), 3.1), ((1500, 2500), 11.8), ((2500, 3500), 21.7), ((3500, 4500), 31.1), ((4500, 6000), 10.4)]

# the eclipse field's matched stars: the largest constant-only 2024 stage 2
cands = []
for z in glob.glob(os.path.join(S1, 'distortion_data*.zip')):
    zf = zipfile.ZipFile(z)
    rn = [n for n in zf.namelist() if n.endswith('distortion_results.txt')]
    if not rn:
        continue
    j = json.load(zf.open(rn[0]))
    if j.get('fixed distortion order') == 'constant':
        cands.append((j['#stars used'], z, j))
n, z, j = sorted(cands)[-1]
zf = zipfile.ZipFile(z)
d = pd.read_csv(zf.open([m for m in zf.namelist() if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
d.columns = [c.strip() for c in d.columns]
print('eclipse field: %s, %d matched stars, imported ps %.7f' % (os.path.basename(z)[15:33], n, j['platescale (arcseconds/pixel)']))

ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
A = np.c_[X, Y, np.ones_like(X)]
ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None)
ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
t = Time(T_MID, scale='utc'); sun = get_sun(t)
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
SUNPX, SUNPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
px, py, mag = d.px.values, d.py.values, d.magV.values
rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
R = np.hypot(rx, ry)
keep = (R > RCUT*RS) & (mag <= MAGCUT)
px, py, mag, rx, ry, R = (a[keep] for a in (px, py, mag, rx, ry, R))
h = 1/np.mean((RS/R)**2)
print('Sun at px (%.0f, %.0f), %.0f px from the frame centre; R_sun %.1f" = %.0f px' % (SUNPX, SUNPY, np.hypot(SUNPX-NX/2, SUNPY-NY/2), RS, RS/PS))
print('science set: %d stars G<=%g outside %g R_sun, R %.1f-%.1f R_sun, h = %.1f R_sun^2' % (len(px), MAGCUT, RCUT, R.min()/RS, R.max()/RS, h))

rc = np.hypot(px-NX/2, py-NY/2)
slope = np.zeros(len(px))
for (lo, hi), s in SLOPES:
    slope[(rc >= lo) & (rc < hi)] = s
bias = slope*(mag-GREF)/1000.0                  # arcsec, outward from the frame centre for faint stars
ucx, ucy = (px-NX/2)/rc, (py-NY/2)/rc
DX, DY = bias*ucx, bias*ucy


def fit(dx, dy, with_scale, nd=2):
    xs, ys = (px-NX/2)/(NX/2), (py-NY/2)/(NX/2)
    ux, uy = rx/R, ry/R
    m = len(px); Z = np.zeros(m)
    cx = [np.ones(m), Z, -(py-NY/2)*PS]; cy = [Z, np.ones(m), (px-NX/2)*PS]; lab = ['N1', 'N2', 'Th']
    if with_scale:
        cx.append((px-NX/2)*PS); cy.append((py-NY/2)*PS); lab.append('S')
    cx.append(ux*RS/R); cy.append(uy*RS/R); lab.append('L')
    if nd:
        for i in range(nd+1):
            for jj in range(nd+1-i):
                if i == 0 and jj == 0:
                    continue
                cx.append(Z); cy.append(xs**i*ys**jj)
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[lab.index('L')], (1e6*c[lab.index('S')] if with_scale else np.nan)


print('\nthe measured bias field injected as the only displacement (true L = 0):')
print('  %-26s %14s %14s %10s' % ('', 'Method 1 dL', 'Method 2 dL', 'M2 dS'))
for nd, nm in ((None, 'no nuisance'), (2, 'vertical-deg-2 nuisance')):
    L1, _ = fit(DX, DY, False, nd); L2, S2 = fit(DX, DY, True, nd)
    print('  %-26s %+13.3f" %+13.3f" %+8.1f ppm' % (nm, L1, L2, S2))
for tag, k in (('sign flipped', -1.0), ('5x amplitude (Leon class)', 5.0)):
    L1, _ = fit(k*DX, k*DY, False, 2); L2, S2 = fit(k*DX, k*DY, True, 2)
    print('  %-26s %+13.3f" %+13.3f" %+8.1f ppm' % (tag, L1, L2, S2))
L2v, _ = fit(DX, DY, True, 2)
print('\nfor scale: Station 1\'s 2024 Method-2 result was L = 1.854 on 74 stars (naive 4.7 %); the record')
print('quotes sigma_L ~13 %% = 0.24". The estimator axis under Method 2 is worth %.3f" = %.0f %% of that.'
      % (abs(L2v), 100*abs(L2v)/0.24))
