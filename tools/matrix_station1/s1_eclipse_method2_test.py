"""Cell 2, Station 1: is Method 2's single isotropic scale column enough on the eclipse field?

The zenith calibration was taken at 05:32-06:15 UTC on eclipse day at 10 C, totality was at
18:12 at 15 C after a daytime refocus, and the eclipse field's fitted plate scale is ~600 ppm
below the zenith mean (1.84728 against 1.84847 "/px). Method 1 is therefore impossible here
whatever the centroids do -- the imported scale reads as +16" of "deflection" -- and Station 1
is reduced by Method 2, which fits one isotropic scale S beside L.

But a refocus of a reducer system changes more than the isotropic scale, and the zenith null
test caught that in the act: the one +45 ppm scale event in the calibration session cost the
scale-free estimator 0.19" on a null pair while its neighbours gave 0.02-0.11". So this asks,
on the 2024 eclipse fit itself (constant-only stage 2 against the frozen zenith quintic,
archive 20240426155036, 182 stars):

  * Method 1 with the imported scale (for the record of what the 600 ppm is worth in L);
  * Method 2 with isotropic S, with and without the vertical-deg-2 nuisance;
  * Method 2 with an ANISOTROPIC linear block -- Sx, Sy and a skew term -- which is what a
    tilt or an astigmatic focus change would need.

If the anisotropic fit moves L by less than its error and Sy - Sx is consistent with zero, the
pipeline's isotropic S is sufficient and the refocus is a pure magnification change at the
level the eclipse field can see.

Residuals are rebuilt from the archive's RA/DEC(obs) - RA/DEC(catalog) through the field's
own affine, so they are in the sensor frame in arcsec. Cuts: G <= 12, 2 < R < 9 R_sun (the
2024 stage-3 selection), stars the 2024 fit flagged as outliers dropped.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

S1 = r"D:/MEE2024 output/Station 1"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'

z = glob.glob(os.path.join(S1, 'distortion_data20240426155036*.zip'))[0]
zf = zipfile.ZipFile(z)
j = json.load(zf.open([m for m in zf.namelist() if m.endswith('distortion_results.txt')][0]))
print('archive 155036: fixed=%s  grav=%s refr=%s aberr=%s  n=%d  stage-2 rms %.3f"  imported ps %.7f'
      % (j['fixed distortion order'], j['gravitational correction enabled?'], j['refraction correction enabled?'],
         j['aberration/parallax correction enabled?'], j['#stars used'], j['final rms error (arcseconds)'],
         j['platescale (arcseconds/pixel)']))
d = pd.read_csv(zf.open([m for m in zf.namelist() if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
d.columns = [c.strip() for c in d.columns]
ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
A = np.c_[X, Y, np.ones_like(X)]
ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
t = Time(T_MID, scale='utc'); sun = get_sun(t)
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
SUNPX, SUNPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
cx = np.c_[X, Y, np.ones(len(d))]
DX, DY = (ox@ax - cx@ax)*PS, (ox@ay - cx@ay)*PS          # arcsec, sensor frame, observed minus catalogue
px, py, mag = d.px.values, d.py.values, d.magV.values
rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry)
print('Sun at px (%.0f, %.0f), R_sun %.1f"; raw residual rms/axis %.3f"' % (SUNPX, SUNPY, RS, np.sqrt((DX**2+DY**2).mean()/2)))


def run(MAGCUT, RMAX, label):
    keep = (R > 2*RS) & (mag <= MAGCUT) & (R < RMAX*RS) & (d['flag_is_outlier'].values == False)
    p, q, r = px[keep], py[keep], R[keep]; ux, uy = rx[keep]/r, ry[keep]/r
    dx, dy = DX[keep]-np.median(DX[keep]), DY[keep]-np.median(DY[keep])
    n = len(p); Z = np.zeros(n)
    xs, ys = (p-NX/2)*PS, (q-NY/2)*PS; xn, yn = (p-NX/2)/(NX/2), (q-NY/2)/(NX/2)

    def solve(cols_x, cols_y, lab, nuis):
        cx_, cy_, L = list(cols_x), list(cols_y), list(lab)
        if nuis:
            for i in range(nuis+1):
                for jj in range(nuis+1-i):
                    if i == 0 and jj == 0:
                        continue
                    cx_.append(Z); cy_.append(xn**i*yn**jj); L.append('v')
        Am = np.vstack([np.column_stack(cx_), np.column_stack(cy_)])
        sc = np.sqrt((Am**2).mean(0)); An = Am/sc            # unit-rms columns: arcsec-scale columns beside unit ones
        b = np.concatenate([dx, dy]); c, *_ = np.linalg.lstsq(An, b, rcond=None)
        res = b-An@c; dof = len(b)-An.shape[1]; s2 = (res**2).sum()/dof
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(An.T@An))))
        return c/sc, e/sc, np.sqrt(s2), L

    print('\n=== %s: %d stars, h = %.1f R_sun^2 ===' % (label, n, 1/np.mean((RS/r)**2)))
    bx = [np.ones(n), Z, -ys, ux*RS/r]; by = [Z, np.ones(n), xs, uy*RS/r]; lab = ['N1', 'N2', 'Th', 'L']
    out = {}
    for nm, ex, ey, el, nu in (('M1 scale imported, base', [], [], [], 0),
                               ('M1 scale imported, v-deg2', [], [], [], 2),
                               ('M2 isotropic S, base', [xs], [ys], ['S'], 0),
                               ('M2 isotropic S, v-deg2', [xs], [ys], ['S'], 2),
                               ('M2 anisotropic Sx,Sy,skew, base', [xs, Z, ys], [Z, ys, xs], ['Sx', 'Sy', 'K'], 0)):
        c, e, s, L = solve(bx+ex, by+ey, lab+el, nu)
        line = '  %-34s L = %+.3f +- %.3f"  rms %.3f"' % (nm, c[3], e[3], s)
        for k in el:
            i = L.index(k); line += '  %s %+.0f+-%.0f ppm' % (k, 1e6*c[i], 1e6*e[i])
        print(line); out[nm] = (c, e, L)
    c, e, L = out['M2 anisotropic Sx,Sy,skew, base']
    i, k = L.index('Sx'), L.index('Sy')
    print('  anisotropy Sy - Sx = %+.0f +- %.0f ppm; L(aniso) - L(iso, base) = %+.3f"'
          % (1e6*(c[k]-c[i]), 1e6*np.hypot(e[i], e[k]), c[3]-out['M2 isotropic S, base'][0][3]))


run(12, 9, 'eclipse field, G<=12, 2<R<9 R_sun (the 2024 stage-3 cuts)')
run(12, 99, 'eclipse field, G<=12, R>2 R_sun, no outer cutoff')
print('\nGR = 1.751"; the 2024 stage 3 of record: L = 1.854 on 74 stars, Method 2 isotropic, its own vetting.')
