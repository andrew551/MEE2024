"""Can Husillos carry Leon's vertical nuisance? Two measurements that have to come first.

Douglas, 2026-09-10: "The Leon 2026 analysis also used a vertical nuisance filter. Are we
able to do that here or do we need to do further analysis first?"

Leon's estimator adds a degree-2 polynomial surface in field position to the VERTICAL
component of the displacement only (`tools/step3_s2_union.py:design`, the `v{i}{j}` columns),
and it is the difference between L base and "L v-deg2" in every Leon table.  Two facts had to
be true for Leon before it was legitimate, and NEITHER transfers by assumption:

  1. IT IS APPLIED ALONG THE SENSOR y AXIS.  That is only the vertical because Leon measured
     it: "the sensor's -y axis sits 3.6 deg from the local vertical" (docs/STEP3_2026.md,
     chart revision 2).  Husillos is a different pointing on a different mount with a
     different roll, so the angle has to be measured, not inherited.  A nuisance applied
     along the wrong axis absorbs the wrong component.
  2. THERE HAS TO BE SOMETHING VERTICAL TO ABSORB.  Leon justified the filter with a measured
     polarisation: its union's displacements had vertical rms 0.898 " against horizontal
     0.363 ", V/H = 2.5, matching the 2.4 its night maps measured on other fields on other
     nights.  If Husillos' V/H is ~1 the atmosphere is not polarised in this field and a
     deg-2 vertical surface would be five free parameters absorbing noise -- which moves L
     without justifying it.  That is precisely the "never choose an analysis parameter at the
     keyboard" trap in CLAUDE.md.

This tool measures both and reports; it does not fit L.  The alt/az basis follows
`tools/step3_charts_record.py:altaz_basis` -- the same convention, through the shared
`record_charts.SkyFrame`, rather than a private copy.

    python tools/husillos2026/hu_vertical.py
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from record_charts import SkyFrame  # noqa: E402

NX, NY = 9576, 6388  # Zeus 455M PRO (IMX455), from the SER header
HUS = r'F:\MEE_output\husillos2026'
UNI = os.path.join(HUS, 'step3', 'union', 'host_gain125')

#: Husillos site, from `G:\Joe Izen Spain 2026\Husillos Spain.JPG` (docs/HUSILLOS2026_*.md).
LAT, LON, HEIGHT = 42.09293, -4.52702, 743.0
#: mid-time of the host block (gain 125), the frame the union is expressed in
DATE, MIDT = '2026-08-12', '18:29:20'


def _host_results():
    z = glob.glob(os.path.join(HUS, 'step3', 'eclipse_gain125', '**',
                               'distortion_data*.zip'), recursive=True)[0]
    zf = zipfile.ZipFile(z)
    return json.load(zf.open([n for n in zf.namelist()
                              if n.endswith('distortion_results.txt')][0]))


def main():
    U = pd.read_csv(os.path.join(UNI, 'husillos_union_star_table.csv'),
                    dtype={'ID': str})
    U = U[U.nblock == 2].reset_index(drop=True)      # the two-witness union of record
    res = _host_results()
    ps = res['platescale (arcseconds/pixel)']
    SF = SkyFrame.from_stars(U['RA_cat'].values, U['DEC_cat'].values,
                             U['px'].values, U['py'].values, ps)

    T = Time(DATE + 'T' + MIDT, scale='utc')
    SITE = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=HEIGHT * u.m)

    # --- 1. where is the local vertical on this sensor?
    ra_c, dec_c = U['RA_cat'].mean(), U['DEC_cat'].mean()
    fc = SkyCoord(ra_c * u.deg, dec_c * u.deg)
    aa = fc.transform_to(AltAz(obstime=T, location=SITE))
    basis = {}
    for key, off in (('alt', dict(alt=aa.alt + 0.05 * u.deg, az=aa.az)),
                     ('az', dict(alt=aa.alt, az=aa.az + 0.05 * u.deg / np.cos(aa.alt)))):
        p = SkyCoord(AltAz(obstime=T, location=SITE, **off)).icrs
        dv = np.array([(p.ra.deg - fc.ra.deg) * np.cos(np.radians(dec_c)),
                       p.dec.deg - fc.dec.deg])
        v = SF.sky_to_px_dir(dv[0], dv[1])
        basis[key] = v / np.linalg.norm(v)
    e_alt, e_az = basis['alt'], basis['az']
    tilt = np.degrees(np.arctan2(e_alt[0], -e_alt[1]))   # sensor -y against the vertical

    print('field centre at the host mid-time %s UTC' % MIDT)
    print('   altitude %.2f deg, azimuth %.2f deg' % (aa.alt.deg, aa.az.deg))
    print('   Leon for comparison: +9.9 deg at C2 (I:\\Leon location and weather data'
          '\\actual leon site.JPG)')
    print()
    print('1. the local vertical on the sensor')
    print('   e_alt in sensor pixels  (%+.4f, %+.4f)' % (e_alt[0], e_alt[1]))
    print('   sensor -y is %.1f deg from the local vertical   (Leon: 3.6 deg)' % tilt)

    # --- 2. is anything vertically polarised?
    # displacements are stored in sky arcsec (RA*cos(dec), Dec); project onto alt/az
    d_sky = np.column_stack([U['dx'].values, U['dy'].values])
    # unit sky-direction vectors for alt and az, from the same 0.05 deg offsets
    def sky_dir(key, off):
        p = SkyCoord(AltAz(obstime=T, location=SITE, **off)).icrs
        v = np.array([(p.ra.deg - fc.ra.deg) * np.cos(np.radians(dec_c)),
                      p.dec.deg - fc.dec.deg])
        return v / np.linalg.norm(v)
    u_alt = sky_dir('alt', dict(alt=aa.alt + 0.05 * u.deg, az=aa.az))
    u_az = sky_dir('az', dict(alt=aa.alt, az=aa.az + 0.05 * u.deg / np.cos(aa.alt)))
    v_comp = d_sky @ u_alt
    h_comp = d_sky @ u_az
    V, H = np.std(v_comp), np.std(h_comp)
    print()
    print('2. polarisation of the %d two-witness union displacements' % len(U))
    print('   vertical   rms %.3f "' % V)
    print('   horizontal rms %.3f "' % H)
    print('   V/H %.2f        (Leon science field 2.5 before the nuisance, 2.1 after;'
          ' its night maps 2.4)' % (V / H))

    # the same split on the raw sensor axes, as a check that the rotation is what changed it
    print('   for reference, on the raw sensor axes: x %.3f ", y %.3f ", y/x %.2f'
          % (np.std(U['dx']), np.std(U['dy']), np.std(U['dy']) / np.std(U['dx'])))

    # --- what a smooth vertical surface could actually take out
    #
    # Against the PURE-NOISE EXPECTATION, not against zero: a k-parameter least-squares fit
    # on n points removes sqrt(k/n) of the rms even from white noise, so "the rms went down"
    # is not evidence of structure.  A surface that removes LESS than that has found nothing.
    x = (U['px'].values - U['px'].mean()) / (NX / 2.0)
    y = (U['py'].values - U['py'].mean()) / (NX / 2.0)
    print()
    print('3. what a smooth vertical surface would remove')
    print('   %-8s %-11s %-9s %-11s %s' % ('degree', 'params', 'rms after', 'pure noise',
                                           'held-out gain'))
    rng = np.random.default_rng(7)
    for deg in (1, 2, 3):
        M = np.column_stack([x ** i * y ** j
                             for i in range(deg + 1) for j in range(deg + 1 - i)])
        k = M.shape[1]
        c, *_ = np.linalg.lstsq(M, v_comp, rcond=None)
        after = np.std(v_comp - M @ c)
        exp = V * np.sqrt(max(1 - k / len(U), 0.0))
        # star-split cross-validation: fit on half the stars, score on the other half.
        # This is the test that settled the zenith field's polynomial order
        # (hu_order_stability.crossval) and it is the only one that cannot be fooled by
        # extra parameters.
        gains = []
        for _ in range(200):
            m = rng.permutation(len(U)) < len(U) // 2
            try:
                cc, *_ = np.linalg.lstsq(M[m], v_comp[m], rcond=None)
            except np.linalg.LinAlgError:
                continue
            held = v_comp[~m]
            gains.append(1 - np.std(held - M[~m] @ cc) / np.std(held))
        g = 100 * np.mean(gains)
        print('   %-8d %-11d %-9.4f %-11.4f %+.1f %%%s'
              % (deg, k, after, exp, g, '' if g > 0 else '   (worse than not fitting)'))
    print('   Leon deg-2: 0.898 " -> 0.763 ", against a pure-noise expectation of ~0.85 ".')


if __name__ == '__main__':
    main()
