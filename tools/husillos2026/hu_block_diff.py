"""The two-block difference done properly: refraction-corrected displacements, full affine.

Everything in sections 3q's "Sun-centred structure" rested on hu_gain_or_time.py, which
compared the two blocks' RAW DETECTED PIXEL POSITIONS with an isotropic similarity
(translation + rotation + one scale) removed.  hu_field_shape.py then showed that the residual
field between the blocks is dominated by stable LINEAR terms -- an anisotropic scale and a shear
of order 60-90 ppm -- that a similarity cannot absorb.  That is the signature of DIFFERENTIAL
REFRACTION changing between the two block epochs: at z = 81.4 deg, d2R/dz2 is ~11 arcsec per
deg^2, the field sets ~0.11 deg in the 39 s between the blocks, and the vertical compression
across the +-2.9 deg field changes by a few hundred ppm -- of which only the isotropic half
was being removed.  Projected radially about a Sun near the field centre, through 11 inner
stars lopsided in azimuth, a vertical anisotropy looks Sun-centred.  The "Sun-centred 1/r
structure that follows the gain" may have been that projection.

So this repeats the comparison on what stage 2 actually hands to stage 3: the displacement
observation - catalogue in sky arcsec, with the catalogue REFRACTION-CORRECTED at each block's
own mid-time, and removes a FULL AFFINE (translation, two scales, two shears -- 6 parameters)
instead of a similarity.  Then it asks whether any 1/r Sun-centred term survives, with its
error, on the inner/outer split, and on held-out stars.  The raw-pixel similarity version is
run alongside so the two can be seen side by side.

If nothing Sun-centred survives, the whole thread from section 3q -- "follows the gain", the
-227/-135/-60 ppm profile, the 10-30 s "atmosphere" reading of the sub-stacks -- was an
artefact of comparing uncorrected pixels with too few nuisance terms, and the two blocks'
L difference is what section 3f said it was: per-star noise (r = 0.484) plus a low-order
difference.  If a 1/r term survives at several sigma, it is real and still unexplained.

    .venv/Scripts/python.exe tools/husillos2026/hu_block_diff.py
"""
import glob
import os
import zipfile

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
import astropy.units as u

HUS = r'F:\MEE_output\husillos2026'
R_SUN_AS = 947.1
SITE = EarthLocation(lat=42.09293 * u.deg, lon=-4.52702 * u.deg, height=743 * u.m)
T_MID = '2026-08-12 18:29:40'          # between the two block mid-times


def matched(d):
    z = glob.glob(os.path.join(HUS, 'step3', d, '**', 'distortion_data*.zip'), recursive=True)
    zf = zipfile.ZipFile(z[0])
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    return t.set_index('ID')


def disp(t):
    """observation - catalogue, arcsec on the sky, RA scaled by cos(dec).  Catalogue is the
    refraction- and aberration-corrected one stage 2 wrote."""
    c = np.cos(np.radians(t['DEC(catalog)'].values))
    return (np.column_stack([(t['RA(obs)'].values - t['RA(catalog)'].values) * 3600 * c,
                             (t['DEC(obs)'].values - t['DEC(catalog)'].values) * 3600]))


def fit(D, X, Y, Rsun, model, keep=None):
    """Fit `model` to the 2-vector field D (n x 2) at positions X, Y (arcsec from the Sun).

    model: 'sim' (translation + rotation + isotropic scale, 4 params),
           'aff' (translation + full linear, 6 params),
           'aff+1/r' (affine plus a Sun-centred deflection-shaped term, 7 params).
    Returns coefficient vector, residual (2n), covariance, design.
    """
    if keep is not None:
        D, X, Y, Rsun = D[keep], X[keep], Y[keep], Rsun[keep]
    n = len(X)
    Z, O = np.zeros(n), np.ones(n)
    R = np.hypot(X, Y)
    if model == 'sim':
        cx, cy = [O, Z, -Y, X], [Z, O, X, Y]
    else:
        cx, cy = [O, Z, X, Y, Z, Z], [Z, O, Z, Z, X, Y]
    if model == 'aff+1/r':
        cx = cx + [X / R / Rsun]
        cy = cy + [Y / R / Rsun]
    M = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    d = np.concatenate([D[:, 0], D[:, 1]])
    k = np.ones(2 * n, bool)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(M[k], d[k], rcond=None)
        r = d - M @ c
        nk = np.abs(r) < 3 * np.std(r[k])
        if nk.sum() == k.sum() or nk.sum() < 12:
            break
        k = nk
    c, *_ = np.linalg.lstsq(M[k], d[k], rcond=None)
    res = d - M @ c
    cov = (res[k] @ res[k]) / max(k.sum() - M.shape[1], 1) * np.linalg.inv(M[k].T @ M[k])
    return c, res, cov, M, k


def heldout_1r(D, X, Y, Rsun, rng, k=300):
    """Does a 1/r term on top of the affine reproduce on held-out stars?"""
    n = len(X)
    gains = []
    for _ in range(k):
        m = rng.permutation(n) < n // 2
        ca, ra, _, Ma, _ = fit(D, X, Y, Rsun, 'aff', m)
        cb, rb, _, Mb, _ = fit(D, X, Y, Rsun, 'aff+1/r', m)
        # predict the held-out half with both models, score the 1/r's marginal gain
        _, _, _, Mh_a, _ = fit(D, X, Y, Rsun, 'aff', ~m)
        _, _, _, Mh_b, _ = fit(D, X, Y, Rsun, 'aff+1/r', ~m)
        dh = np.concatenate([D[~m, 0], D[~m, 1]])
        ea = np.std(dh - Mh_a @ ca)
        eb = np.std(dh - Mh_b @ cb)
        gains.append(1 - eb / ea)
    return 100 * np.mean(gains)


def report(title, D, X, Y, Rsun, unit):
    print(title)
    n = len(X)
    for model in ('sim', 'aff', 'aff+1/r'):
        c, res, cov, M, k = fit(D, X, Y, Rsun, model)
        rms = float(np.std(res[k]))
        line = '   %-8s resid %.3f %s' % (model, rms, unit)
        if model == 'sim':
            line += '   scale %+7.1f +- %4.1f ppm' % (c[3] * 1e6, np.sqrt(cov[3, 3]) * 1e6)
        else:
            sx, sy = c[2] * 1e6, c[5] * 1e6
            line += ('   scale x %+6.1f  y %+6.1f  (aniso %+6.1f)  shear %+6.1f %+6.1f ppm'
                     % (sx, sy, sx - sy, c[3] * 1e6, c[4] * 1e6))
        if model == 'aff+1/r':
            line += '   1/r %+.3f +- %.3f %s' % (c[6], np.sqrt(cov[6, 6]), unit)
        print(line)
    # inner/outer on the AFFINE residual: a Sun-centred term would leave a radial split
    c, res, cov, M, k = fit(D, X, Y, Rsun, 'aff')
    rx, ry = res[:n], res[n:]
    R = np.hypot(X, Y)
    radial = (rx * X + ry * Y) / R
    med = np.median(Rsun)
    ri, ro = radial[Rsun <= med], radial[Rsun > med]
    print('   affine residual, radial: inner %+.3f +- %.3f   outer %+.3f +- %.3f %s   '
          'difference %.1f sigma'
          % (ri.mean(), ri.std(ddof=1) / np.sqrt(len(ri)), ro.mean(),
             ro.std(ddof=1) / np.sqrt(len(ro)), unit,
             abs(ri.mean() - ro.mean()) / np.hypot(ri.std(ddof=1) / np.sqrt(len(ri)),
                                                   ro.std(ddof=1) / np.sqrt(len(ro)))))
    g = heldout_1r(D, X, Y, Rsun, np.random.default_rng(5))
    print('   1/r term on top of the affine, held-out gain: %+.1f %%  '
          '(a real Sun-centred term reproduces; noise scores <= 0)' % g)
    print()


def main():
    A, B = matched('eclipse_gain125'), matched('eclipse_gain0')     # earlier, later
    both = sorted(A.index.intersection(B.index))
    a, b = A.loc[both], B.loc[both]
    n = len(both)
    T = Time(T_MID)
    sun = get_body('sun', T, SITE)
    ra, dec = a['RA(catalog)'].values, a['DEC(catalog)'].values
    X = (ra - sun.ra.deg) * np.cos(np.radians(dec)) * 3600.0
    Y = (dec - sun.dec.deg) * 3600.0
    Rsun = np.hypot(X, Y) / R_SUN_AS
    print('%d shared two-witness stars, %.2f-%.2f R_sun; displacement = gain 0 - gain 125 '
          '(later - earlier)\n' % (n, Rsun.min(), Rsun.max()))

    # 1. what section 3q did: raw detected pixels
    Dpx = np.column_stack([b['px'].values - a['px'].values, b['py'].values - a['py'].values])
    Xpx, Ypx = a['px'].values - 5043.0, a['py'].values - 3386.0
    Rpx = np.hypot(Xpx, Ypx) * 2.2028 / R_SUN_AS
    report('1. RAW PIXEL POSITIONS (what hu_gain_or_time.py compared), NO refraction correction',
           Dpx, Xpx, Ypx, Rpx, 'px')

    # 2. what stage 3 sees: refraction-corrected sky displacements
    Dsky = disp(b) - disp(a)
    report('2. REFRACTION-CORRECTED SKY DISPLACEMENTS (what stage 3 fits), obs - catalogue',
           Dsky, X, Y, Rsun, '"')

    print('READING: if the raw-pixel comparison shows anisotropy/shear and a 1/r term while the')
    print('corrected one shows neither, the "Sun-centred structure" was differential refraction')
    print('between the block epochs, projected radially through an isotropic-only fit.')


if __name__ == '__main__':
    main()
