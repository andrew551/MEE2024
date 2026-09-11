"""Is the gain-0 / gain-125 disagreement caused by the GAIN, or by the 39 s between them?

*** SUPERSEDED BY hu_block_diff.py -- READ THIS FIRST (2026-09-11, late). ***
This tool compares RAW DETECTED PIXEL POSITIONS with an ISOTROPIC similarity removed.  At
8.6 deg altitude that is the wrong nuisance model: the differential refraction across the
field changes by a few hundred ppm between two epochs 39 s apart, as an ANISOTROPIC compression
plus a shear from the sensor's 14.9 deg tilt to the vertical, and a similarity removes only the
isotropic half.  Projected radially about a Sun near the field centre through eleven inner
stars lopsided in azimuth, the remainder looks Sun-centred and monotonic in radius -- which is
every "finding" below.  hu_block_diff.py repeats the comparison on refraction-corrected
displacements with a full affine removed: the blocks then agree to -6 +- 10 ppm in scale and
~10 ppm in every affine term, and no 1/r term reproduces on held-out stars.  Kept so the
record's section 3q-3r can be reproduced; do not build on it.


Douglas, 2026-09-11: "We did not see such a problem in the Leon 2026 exposure tiers where the
gain was held constant and the exposure was changed.  Is changing the gain while holding the
exposure constant fundamentally different in terms of the corona subtraction?"

Two facts frame the test.  Leon's tiers were INTERLEAVED -- the folder timestamps put 0.1 s
at 18:28:13-45, 0.3 s at 18:28:21-41, 0.6 s at 18:28:26-37 and 1.2 s at 18:28:29, all inside
one 32 s window -- so every tier saw the same atmosphere and Leon could not have detected a
time-driven block difference even if one existed.  Husillos' blocks are SEQUENTIAL, 39 s
apart.  So "Leon did not see it" does not by itself say the gain is the cause.

The four half-blocks (hu_halves.py) separate the two variables, because pairs can be formed
at the same gain ~20 s apart and across the gain boundary ~21 s apart:

    g125_A  18:29:09.9   gain 125        same gain:   g125_A-g125_B  19.9 s
    g125_B  18:29:29.8   gain 125                     g0_A-g0_B      15.7 s
    g0_A    18:29:51.2   gain 0          cross gain:  g125_B-g0_A    21.4 s   <-- decisive
    g0_B    18:30:06.9   gain 0                       g125_A-g0_B    57.0 s

For each pair the star-to-star displacement is fitted ABOUT THE SUN with translation +
rotation + scale, then with a 1/r deflection-shaped term added, and split into inner and
outer halves by solar radius.  A pure scale is the same in both halves by definition; a
Sun-centred 1/r structure is not.  If that structure follows the GAIN boundary it appears in
the cross-gain rows only; if it follows TIME it grows with dt regardless of gain.

The same machinery, applied to the two full blocks, also splits their difference by
MAGNITUDE (saturation and brightness-dependent centroiding would make it depend on
brightness) and by RADIUS.

    .venv/Scripts/python.exe tools/husillos2026/hu_gain_or_time.py
"""
import glob
import os
import zipfile

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
import astropy.units as u

HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
PS, R_SUN_AS = 2.2028, 947.1
SITE = EarthLocation(lat=42.09293 * u.deg, lon=-4.52702 * u.deg, height=743 * u.m)

HALF_MID = {'g125_A': '18:29:09.9', 'g125_B': '18:29:29.8',
            'g0_A': '18:29:51.2', 'g0_B': '18:30:06.9'}
PAIRS = [('g125_A', 'g125_B', 'same gain 125'), ('g0_A', 'g0_B', 'same gain 0'),
         ('g125_B', 'g0_A', 'CROSS gain'), ('g125_A', 'g0_B', 'CROSS gain')]


def matched(path_glob):
    z = glob.glob(path_glob, recursive=True)
    if not z:
        raise SystemExit('no stage-2 output matching ' + path_glob)
    zf = zipfile.ZipFile(z[0])
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    return t.set_index('ID')


def sun_px(a, tmid):
    """The Sun on the sensor, forward-projected through the block's own star affine."""
    T = Time('2026-08-12 ' + tmid)
    sun = get_body('sun', T, SITE)
    ra, dec = a['RA(catalog)'].values, a['DEC(catalog)'].values
    ra0, de0 = ra.mean(), dec.mean()
    M = np.c_[(ra - ra0) * np.cos(np.radians(de0)), dec - de0, np.ones(len(ra))]
    cx, *_ = np.linalg.lstsq(M, a['px'].values, rcond=None)
    cy, *_ = np.linalg.lstsq(M, a['py'].values, rcond=None)
    sx, sy = (sun.ra.deg - ra0) * np.cos(np.radians(de0)), sun.dec.deg - de0
    return cx[0] * sx + cx[1] * sy + cx[2], cy[0] * sx + cy[1] * sy + cy[2]


def fit(a, b, spx, spy, with_defl, keep=None):
    """Displacement b - a about the Sun: translation + rotation + scale [+ 1/r], 3-sigma clipped.

    The scale term is in ppm of the SENSOR frame: positive means the field is larger on the
    sensor in b, i.e. FEWER arcsec per pixel.  The 1/r term is in arcsec of L.  Without the
    1/r term a Sun-centred structure is reported as a scale, which is exactly what the
    inner/outer split exposes.
    """
    if keep is not None:
        a, b = a[keep], b[keep]
    n = len(a)
    rx, ry = a['px'].values - spx, a['py'].values - spy
    R = np.hypot(rx, ry)
    Rsun = R * PS / R_SUN_AS
    d = np.concatenate([b['px'].values - a['px'].values, b['py'].values - a['py'].values])
    Z, O = np.zeros(n), np.ones(n)
    defl = 1.0 / (Rsun * PS)                       # px per arcsec of L
    cx = [O, Z, -ry, rx] + ([rx / R * defl] if with_defl else [])
    cy = [Z, O, rx, ry] + ([ry / R * defl] if with_defl else [])
    M = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    k = np.ones(2 * n, bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(M[k], d[k], rcond=None)
        r = d - M @ c
        nk = np.abs(r) < 3 * np.std(r[k])
        if nk.sum() == k.sum() or nk.sum() < 12:
            break
        k = nk
    c, *_ = np.linalg.lstsq(M[k], d[k], rcond=None)
    res = d[k] - M[k] @ c
    cov = (res @ res) / max(k.sum() - M.shape[1], 1) * np.linalg.inv(M[k].T @ M[k])
    out = dict(scale=c[3] * 1e6, scale_e=np.sqrt(cov[3, 3]) * 1e6, n=n, resid=float(np.std(res)))
    if with_defl:
        out.update(dL=c[4], dL_e=np.sqrt(cov[4, 4]))
    return out, Rsun


def full_blocks():
    print('THE TWO FULL BLOCKS (gain 125 first, gain 0 second; displacement = gain0 - gain125)')
    A = matched(os.path.join(HUS, 'step3', 'eclipse_gain125', '**', 'distortion_data*.zip'))
    B = matched(os.path.join(HUS, 'step3', 'eclipse_gain0', '**', 'distortion_data*.zip'))
    both = sorted(A.index.intersection(B.index))
    a, b = A.loc[both], B.loc[both]
    spx, spy = sun_px(a, '18:29:20')
    s0, Rsun = fit(a, b, spx, spy, False)
    s1, _ = fit(a, b, spx, spy, True)
    print('   %d shared stars, %.2f to %.2f R_sun' % (len(both), Rsun.min(), Rsun.max()))
    print('   scale only            %+7.1f +- %4.1f ppm                      resid %.3f px'
          % (s0['scale'], s0['scale_e'], s0['resid']))
    print('   scale + 1/r           %+7.1f +- %4.1f ppm   1/r %+.2f +- %.2f "  resid %.3f px'
          % (s1['scale'], s1['scale_e'], s1['dL'], s1['dL_e'], s1['resid']))
    mag = a['magV'].values
    medm = np.median(mag)
    for lbl, keep in (('brighter half (G <= %.2f)' % medm, mag <= medm),
                      ('fainter  half (G  > %.2f)' % medm, mag > medm)):
        s, _ = fit(a, b, spx, spy, False, keep)
        print('   by magnitude, %-26s %+7.1f +- %4.1f ppm  (%d stars)'
              % (lbl, s['scale'], s['scale_e'], s['n']))
    edges = np.percentile(Rsun, [0, 33.3, 66.7, 100])
    for i in range(3):
        keep = (Rsun >= edges[i]) & (Rsun <= edges[i + 1])
        s, _ = fit(a, b, spx, spy, False, keep)
        print('   by radius, %4.1f-%4.1f R_sun            %+7.1f +- %4.1f ppm  (%d stars)'
              % (edges[i], edges[i + 1], s['scale'], s['scale_e'], s['n']))


def half_pairs():
    print()
    print('THE HALF-BLOCK PAIRS (displacement = later - earlier)')
    tabs = {t: matched(os.path.join(HUS, 'halves', 's2_' + t, '**', 'distortion_data*.zip'))
            for t in HALF_MID}
    print('   %-15s %-14s %5s %6s %16s %16s %14s %14s'
          % ('pair', 'kind', 'dt s', 'shared', 'scale (ppm)', '1/r term (")',
             'inner (ppm)', 'outer (ppm)'))
    for lo, hi, kind in PAIRS:
        A, B = tabs[lo], tabs[hi]
        both = sorted(A.index.intersection(B.index))
        a, b = A.loc[both], B.loc[both]
        spx, spy = sun_px(a, HALF_MID[lo])
        dt = (Time('2026-08-12 ' + HALF_MID[hi]) - Time('2026-08-12 ' + HALF_MID[lo])).sec
        s0, Rsun = fit(a, b, spx, spy, False)
        s1, _ = fit(a, b, spx, spy, True)
        med = np.median(Rsun)
        si, _ = fit(a, b, spx, spy, False, Rsun <= med)
        so, _ = fit(a, b, spx, spy, False, Rsun > med)
        asym = (si['scale'] - so['scale']) / np.hypot(si['scale_e'], so['scale_e'])
        print('   %-15s %-14s %5.1f %6d %+7.1f +- %-5.1f %+7.2f +- %-5.2f %+6.0f +- %-4.0f %+6.0f +- %-4.0f  inner-outer %.1f sigma'
              % (lo + '-' + hi, kind, dt, len(both), s0['scale'], s0['scale_e'],
                 s1['dL'], s1['dL_e'], si['scale'], si['scale_e'], so['scale'], so['scale_e'],
                 asym))
    print()
    print('   A Sun-centred structure that follows the GAIN shows in the CROSS rows only;')
    print('   one that follows TIME grows with dt regardless of gain.')


if __name__ == '__main__':
    full_blocks()
    half_pairs()
