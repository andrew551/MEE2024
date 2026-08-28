# SUPERSEDED 2026-08-27: the parallactic rotation below is 90 deg off (verified
# against an empirical alt-az affine; see refraction/analysis/m3_maps.py and
# CAL_PILEO_STEP2.md section 7 correction). Kept for the record; do not reuse.
"""Which way does the CAL_piLeo residual point?

sigma_y runs 1.3-2.0x sigma_x and the residual is flat with magnitude, which is not what
photon-limited centroids look like. Three candidate axes, each with a different owner:

  local VERTICAL      -> atmospheric: refraction model, or chromatic refraction at 9.9 deg
  the stage-1 DRIFT   -> trailing: the field moves 9.2 px during the 33.7 s stack
  a SENSOR axis       -> instrumental

Working on the sky avoids guessing the ROLL sign convention, which distortion_fitter.py
itself flags as dodgy. CATALOGUE_MATCHED_ERRORS.csv holds every matched star including the
ones the tolerance rejected, so it is filtered to the fitted set first.
"""
import numpy as np, pandas as pd, json, zipfile
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

SITE = EarthLocation(lat=42.740470*u.deg, lon=-5.613780*u.deg, height=1101*u.m)
WHEN = Time('2026-08-12T18:29:34', scale='utc')

def parallactic(ra, dec):
    c = SkyCoord(ra*u.deg, dec*u.deg).transform_to(AltAz(obstime=WHEN, location=SITE))
    alt, az = c.alt.rad, c.az.rad
    phi, d = np.radians(42.740470), np.radians(dec)
    return np.arctan2(np.sin(az)*np.cos(phi)/np.cos(d),
                      (np.sin(phi) - np.sin(d)*np.sin(alt))/(np.cos(d)*np.cos(alt))), np.degrees(alt)

def spread_along(dn, de, ang):
    """rms of the residual projected on a direction at position angle `ang` (N through E)."""
    return (dn*np.cos(ang) + de*np.sin(ang)).std(ddof=0)

def run(path, label, drift_px):
    d = pd.read_csv(path)
    d = d[~d.flag_is_outlier].reset_index(drop=True)
    ra_c, dec_c = d['RA(catalog)'].values, d['DEC(catalog)'].values
    dn = (d['DEC(obs)'].values - dec_c) * 3600.0
    de = (d['RA(obs)'].values - ra_c) * 3600.0 * np.cos(np.radians(dec_c))
    q, alt = parallactic(ra_c, dec_c)
    n = len(d)
    print(f'\n=== {label}  N={n} ===')

    # empirical sensor -> sky map, so the drift can be quoted as a sky position angle
    A = np.column_stack([d.px - 3124, d.py - 2088, np.ones(n)])
    kx = np.linalg.lstsq(A, de, rcond=None)[0]
    ky = np.linalg.lstsq(A, dn, rcond=None)[0]
    de_drift = kx[0]*drift_px[1] + kx[1]*drift_px[0]
    dn_drift = ky[0]*drift_px[1] + ky[1]*drift_px[0]
    pa_drift = np.arctan2(de_drift, dn_drift)

    # principal axis of the residual itself
    C = np.cov(np.vstack([dn, de]))
    w, v = np.linalg.eigh(C)
    pa_major = np.arctan2(v[1, -1], v[0, -1])

    axes = [('local vertical', q.mean()),
            ('stage-1 drift ', pa_drift),
            ('residual major', pa_major)]
    print(f'  {"axis":16s} {"PA":>8s}   {"rms along":>9s} {"rms across":>10s}   ratio')
    for name, ang in axes:
        a, b = spread_along(dn, de, ang), spread_along(dn, de, ang + np.pi/2)
        print(f'  {name:16s} {np.degrees(ang) % 180:8.1f}   {a:8.4f}" {b:9.4f}"   {a/b:5.2f}')
    rng = np.random.default_rng(11)
    idx = rng.integers(0, n, (2000, n))
    r = (dn[idx]*np.cos(pa_major) + de[idx]*np.sin(pa_major)).std(axis=1) / \
        (dn[idx]*np.cos(pa_major+np.pi/2) + de[idx]*np.sin(pa_major+np.pi/2)).std(axis=1)
    print(f'  major-axis elongation, bootstrap 68%: {np.percentile(r,16):.2f} - {np.percentile(r,84):.2f}')
    print(f'  angle between residual major axis and drift: '
          f'{abs((np.degrees(pa_major-pa_drift)+90) % 180 - 90):.1f} deg;  '
          f'and vertical: {abs((np.degrees(pa_major-q.mean())+90) % 180 - 90):.1f} deg')
    print(f'  mean vertical residual {(dn*np.cos(q)+de*np.sin(q)).mean():+.4f}" '
          f'(se {spread_along(dn,de,q.mean())/np.sqrt(n):.4f})')

z = zipfile.ZipFile(r'D:\MEE2024 output\MEE_output\step2_ladder\A_none\centroid_data20260825033557.zip')
drift = json.load(z.open('results.txt'))['alignment']['shifts_px'][-1]
print(f'stage-1 total drift over the stack: {drift[0]:.2f} px (py), {drift[1]:.2f} px (px)')

base = r"D:/MEE2024 output/MEE_output/step2_ladder"
for tag, r_ in [("tol 0.5", "tolsweep_0.5/DISTORTION_OUTPUT20260825143416__20260825033557"),
                ("tol 1.0", "tolsweep_1.0/DISTORTION_OUTPUT20260825143422__20260825033557"),
                ("tol 5.0", "tolsweep_5.0/DISTORTION_OUTPUT20260825143434__20260825033557")]:
    run(f"{base}/{r_}/distortion/CATALOGUE_MATCHED_ERRORS.csv", tag, drift)
