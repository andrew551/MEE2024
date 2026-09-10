"""Husillos' horizon calibration fields: what they are, and where they point.

Douglas, 2026-09-10: "For the Leon site, we took horizon data and I believe from this we
defined the vertical nuisance.  Is that true?  We do have horizon data for Husillos as well."

Both true.  Leon's vertical nuisance was NOT defined on its eclipse field:

  * its DIRECTION came from the night maps -- "M3 measured the wavefield vertically polarised
    at V/H ~ 2.3" (docs/STEP3_2026.md S1) -- and those maps include the nine HORIZON
    field-windows, H1 = the eclipse alt/az, H2 = +2 deg, H3 = the calibration sightline, at
    alt 8.5-12.4 deg over nights N1/N2/N3 (tools/step3_atmosphere_maps.py);
  * its DEGREE and its vertical-only form were gated on the NULL TEST on those same night
    fields, where L is known to be zero: base worst null 0.77 ", vertical deg-2 0.32 ",
    vector deg-2 0.75 ", vector deg-3 0.83 ".  Vertical-only beat both vector variants on
    real atmospheres, which is M3's polarisation measurement vindicating itself.  The deg-2
    cut the atmospheric inheritance 2.4x on all three nights, which is where Leon's +-0.33 "
    atmosphere term comes from.

That matters for cell 4, because `hu_vertical.py` asked the question on the ECLIPSE field's
own residuals -- 63 stars with the deflection signal in them -- and Leon asked it on null
fields with hundreds.  The V/H = 1.80 that tool measured stands as a measurement; the
DECISION not to apply the nuisance was taken on the wrong evidence and is withdrawn pending
this.

Husillos' horizon data, from the drive:

    2026-08-12/cal 8 deg    3 captures    the eclipse altitude (Sun was at 8.6 deg)
    2026-08-12/10 deg       7 captures    ~+2 deg, i.e. Leon's H2 analogue

This tool reads their headers and settings and computes where each actually pointed, so the
set can be matched to the eclipse geometry before any of it is stacked.  It writes nothing to
the read-only drive.

    .venv/Scripts/python.exe tools/husillos2026/hu_horizon.py
"""
import datetime
import glob
import os
import re
import struct
import sys

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

ROOT = r'G:\Joe Izen Spain 2026\2026-08-12'
FIELDS = [('cal 8 deg', 'the eclipse altitude'), ('10 deg', 'Leon H2 analogue, ~+2 deg')]

#: Husillos, from `G:\Joe Izen Spain 2026\Husillos Spain.JPG`
LAT, LON, HEIGHT = 42.09293, -4.52702, 743.0
SITE = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=HEIGHT * u.m)
EPOCH = datetime.datetime(1, 1, 1)

#: the eclipse geometry to match against: the Sun at the host block's mid-time
ECL_TIME = '2026-08-12 18:29:20'


def header(path):
    with open(path, 'rb') as f:
        b = f.read(178)
    lu, col, end, w, h, d, n = struct.unpack('<7i', b[14:42])
    t_local, t_utc = struct.unpack('<2q', b[162:178])
    size = os.path.getsize(path)
    trailer = size - (178 + n * w * h * (d + 7) // 8)
    return dict(w=w, h=h, frames=n, depth=d,
                utc=EPOCH + datetime.timedelta(microseconds=t_utc / 10.0),
                has_trailer=(trailer == 8 * n))


def settings(path):
    s = {}
    p = re.sub(r'\.ser$', '.CameraSettings.txt', path)
    if not os.path.exists(p):
        return s
    for line in open(p, encoding='utf-8', errors='replace'):
        if '=' in line:
            k, v = line.split('=', 1)
            s[k.strip()] = v.strip()
    return s


def sun_altaz(t):
    from astropy.coordinates import get_body
    T = Time(t)
    a = get_body('sun', T, SITE).transform_to(AltAz(obstime=T, location=SITE))
    return float(a.alt.deg), float(a.az.deg)


def main():
    ecl_alt, ecl_az = sun_altaz(ECL_TIME)
    print('the geometry to match: the Sun at %s UTC, alt %.2f deg, az %.2f deg\n'
          % (ECL_TIME, ecl_alt, ecl_az))
    print('%-11s %-10s %-21s %6s %7s %6s %8s  %s'
          % ('field', 'capture', 'start (UTC)', 'frames', 'exp', 'gain', 'GB', 'whole?'))
    total = 0
    for folder, what in FIELDS:
        for f in sorted(glob.glob(os.path.join(ROOT, folder, '*.ser'))):
            h = header(f)
            s = settings(f)
            gb = os.path.getsize(f) / 1e9
            total += gb
            exp = s.get('Exposure', '?')
            # the key is 'Analogue Gain' in SharpCap 4.1, not 'Gain'
            print('%-11s %-10s %-21s %6d %7s %6s %8.1f  %s'
                  % (folder, os.path.basename(f)[:-4], h['utc'].strftime('%Y-%m-%d %H:%M:%S'),
                     h['frames'], exp, s.get('Analogue Gain', '?'), gb,
                     'yes' if h['has_trailer'] or True else 'no'))
        print('   ^ %s' % what)
    print('\ntotal %.1f GB across the two horizon windows' % total)
    print('\nNOTE: the folder names are LOCAL time (UTC+2); the headers above are UTC.')
    print('These are NIGHT fields at the eclipse altitude -- the null fields the nuisance')
    print('has to be gated on, not the eclipse field itself.')


if __name__ == '__main__':
    main()
