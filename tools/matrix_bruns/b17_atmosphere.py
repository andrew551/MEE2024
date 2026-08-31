"""Cell 1's atmospheric systematic, measured on Bruns' own night fields.

Leon quotes L = 1.98 +- 0.60 (stat) +- 0.33 (atmosphere). That second term is not a
formula: it is an empirical null. The estimator is run on real night fields from the same
campaign -- real atmosphere, ZERO true deflection -- with the Sun placed where it sat on
the eclipse frame, and whatever L it reports is the fake deflection that this atmosphere
plus this estimator manufacture. Leon's worst such null, after the vertical-deg-2 nuisance
had absorbed what it could, was 0.32-0.33 arcsec, so 0.33 is quoted as a systematic
(docs/STEP3_2026.md, the S1 gate).

Cell 1 was quoted WITHOUT such a term -- 1.720 +- 0.069 (stat) +- 0.105 (scale) -- which
understated its error, because Bruns' atmosphere is smaller but not zero. This supplies
it, from the 29 re-reduced night fields in bruns2017_nights/ (Aug 19-20 2017, same
instrument, same site).

Method, mirroring the Leon gate's second version (the one without the circular smoothing
step): per-star residuals as measured, pointing removed by the median, the eclipse Sun
pixel imposed, the eclipse radial cut and magnitude cut applied, then the same estimator.
Each field's own local vertical is used for the nuisance direction where its results file
gives RA/DEC/time, since the atmosphere is polarised along the vertical of the field it
was measured in.

The honest caveat, the same one Leon's number carries: these are NIGHT fields at their own
altitudes, not the eclipse field at 54 degrees in daylight. They bound the estimator's
response to a real atmosphere of this instrument; they are not the eclipse-day atmosphere.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS, L_REF = 948.7, 1.7512
SUNPX, SUNPY = 1645.0, 1741.0          # the eclipse field's measured Sun pixel
RCUT, MAGCUT = 2.0, 11.0
SITE = EarthLocation(lat=42.7363889*u.deg, lon=-106.3180556*u.deg, height=2400*u.m)
VX_ECL, VY_ECL = 0.447, -0.895         # the eclipse field's local vertical, sensor axes

def design(x_px, y_px, rx, ry, R, vx, vy, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ur*R_SUN_AS/R]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, vr*R_SUN_AS/R]
    labels = ['N1','N2','Th','L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0: continue
                cols_x.append(vx*xs**i*ys**j); cols_y.append(vy*xs**i*ys**j)
                labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels

def fit_L(dx, dy, x_px, y_px, rx, ry, R, vx, vy, nuis_deg=None):
    A, labels = design(x_px, y_px, rx, ry, R, vx, vy, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]

def field_vertical(resfile):
    """The field's own local vertical in sensor axes, or the eclipse one if unknown."""
    try:
        j = json.load(open(resfile, encoding='utf-8'))
        ra, dec = j['RA'], j['DEC']
        date = j['observation_date']; tm = j.get('observation_time (UTC)') or '08:00'
        t = Time(f'{date}T{tm if len(tm) > 5 else tm + ":00"}', scale='utc')
        fc = SkyCoord(ra*u.deg, dec*u.deg)
        aa = fc.transform_to(AltAz(obstime=t, location=SITE))
        up = SkyCoord(AltAz(alt=aa.alt+0.1*u.deg, az=aa.az, obstime=t, location=SITE)).icrs
        # sensor axes via the field's own roll: +y is north rotated by ROLL
        roll = np.radians(j.get('ROLL', 0.0))
        dra = (up.ra.deg-ra)*np.cos(np.radians(dec)); ddec = up.dec.deg-dec
        n = np.hypot(dra, ddec)
        if n == 0: return VX_ECL, VY_ECL, aa.alt.deg
        e, nn = dra/n, ddec/n
        return (e*np.cos(roll) - nn*np.sin(roll),
                -(e*np.sin(roll) + nn*np.cos(roll)), aa.alt.deg)
    except Exception:
        return VX_ECL, VY_ECL, float('nan')

rows = []
rng = np.random.default_rng(11)
for f in sorted(glob.glob(os.path.join(NIGHTS, '*', 'stage2', '**', 'TWOD_RESIDUALS.csv'),
                          recursive=True)):
    field = f.split(os.sep)[-5] if os.sep in f else f.split('/')[-5]
    d = pd.read_csv(f)
    d = d[d['magV'] <= MAGCUT]
    if len(d) < 40:
        continue
    resfile = os.path.join(os.path.dirname(f), 'distortion_results.txt')
    vx, vy, alt = field_vertical(resfile)
    px, py = d['px'].values, d['py'].values
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
    dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
    R = np.hypot(rx, ry)
    keep = R > RCUT*R_SUN_AS
    if keep.sum() < 30:
        continue
    px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
    err = d['error_arcsec'].values[keep]
    Lb = fit_L(dx, dy, px, py, rx, ry, R, vx, vy)
    Lv = fit_L(dx, dy, px, py, rx, ry, R, vx, vy, nuis_deg=2)
    boots = [fit_L(dx + rng.normal(0, err/np.sqrt(2)), dy + rng.normal(0, err/np.sqrt(2)),
                   px, py, rx, ry, R, vx, vy, nuis_deg=2) for _ in range(60)]
    rows.append(dict(field=field, n=int(keep.sum()), alt=alt, Lb=Lb, Lv=Lv,
                     floor=float(np.std(boots, ddof=1)),
                     rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2))))

R_ = pd.DataFrame(rows)
print(f'{"field":6} {"N":>5} {"alt":>6} {"rms(as/ax)":>10} {"L base":>8} {"L v-deg2":>9} {"floor":>7}')
for _, r in R_.iterrows():
    print(f'{r.field:6} {int(r.n):5d} {r.alt:6.1f} {r.rms:10.3f} {r.Lb:+8.3f} {r.Lv:+9.3f} {r.floor:7.3f}')
print()
print(f'{len(R_)} night fields, true L = 0 in every one')
for tag, col in (('L base', 'Lb'), ('L v-deg2', 'Lv')):
    v = R_[col].values
    print(f'  {tag:9}: mean {v.mean():+.3f}  rms {np.sqrt((v**2).mean()):.3f}  '
          f'max |dL| {np.abs(v).max():.3f}  (bootstrap floor ~{R_.floor.median():.3f})')
print()
print('  Leon for comparison: v-deg2 nulls 0.19 / 0.32 / 0.19 arcsec against a floor of '
      '0.03-0.06 -> +-0.33 quoted')
v = R_['Lv'].values
print(f'\nCELL-1 ATMOSPHERIC SYSTEMATIC (rms of the v-deg2 nulls): +-{np.sqrt((v**2).mean()):.3f} arcsec')
