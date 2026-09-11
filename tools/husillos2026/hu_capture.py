"""The `Capture` folder: where do these four point, and why did stage 1 refuse one of them?

Douglas, 2026-09-11: "let's have a look at the files in the Capture folder and see where they
are pointed. 100 frames at gain 125 looks like they could be zenith fields."

What the headers already say (`hu_horizon.py` reads them the same way).  The folder sits under
`2026-08-12` but SharpCap folder names are LOCAL time, and the headers are UTC:

    00_54_32   2026-08-11 22:54:32 UTC   100 frames   1.000 s    gain 125   offset 50
    00_57_43   2026-08-11 22:57:43 UTC   100 frames   315 ms     gain 125   offset 50
    00_59_36   2026-08-11 22:59:36 UTC   100 frames   315 ms     gain 125   offset 50
    01_01_09   2026-08-11 23:01:09 UTC   100 frames   315 ms     gain 0     offset 50

So these are the night BEFORE the eclipse, and the 315 ms / gain-125-and-0 combination is the
eclipse acquisition's own settings -- which reads like a rehearsal of the science exposure
rather than a zenith field.  But that is an inference from settings, and the pointing is a
measurement, so this solves them.

THE OFFSET IS THE ODD THING, and it is the standing suspicion about why stage 1 stopped on
00_54_32 with "No star centroids were found on frame 1" while a 1 px matched filter found
~2600 sources in that same frame (`hu_zenith_order.py`, STACKS).  Every other capture in this
dataset is at offset 200 or 220; these four are at **50**.  At gain 125 a pedestal that low can
put the noise distribution hard against zero, and a clipped distribution breaks the background
and sigma estimates that detection is measured against.  `probe` tests that on the pixels
before anything is stacked -- a clipped frame shows itself as a pile-up at the minimum.

    .venv/Scripts/python.exe tools/husillos2026/hu_capture.py [probe|solve]
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import read_header  # noqa: E402

SRC = r'G:\Joe Izen Spain 2026\2026-08-12\Capture'          # READ ONLY
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'capture')
DARK = os.path.join(HUS, 'hotpixels', 'husillos_synthetic_dark_all.fit')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')

CAPTURES = ['00_54_32', '00_57_43', '00_59_36', '01_01_09']
ROW0, ROW1 = 3000, 3600      # a 600-row band, as hu_eclipse_frames uses

#: The site, needed only to turn a solved RA/Dec into an altitude.  Corrections stay OFF for
#: the fit itself if these turn out to be near the zenith (hu_zenith_order's docstring: at the
#: zenith the refraction a correction would apply is absorbed by the plate scale), but the
#: pointing has to be known before that can be decided, so the first solve is made blind.
LAT, LON, HEIGHT = 42.09293, -4.52702, 743.0

#: the zenith star-field preset, identical to hu_zenith_order.S1
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
      '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
      '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
      '--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular']


def _band(path, k=1):
    """Row band ROW0:ROW1 of frame k, as raw ADU."""
    H = read_header(path)
    row_bytes = H['w'] * 2
    with open(path, 'rb') as f:
        f.seek(178 + k * H['frame_bytes'] + ROW0 * row_bytes)
        return np.frombuffer(f.read((ROW1 - ROW0) * row_bytes),
                             dtype='<u2').reshape(ROW1 - ROW0, H['w'])


def do_probe():
    print('Is the offset-50 pedestal clipping the noise at zero?')
    print('A clipped frame piles up at its minimum; a healthy one has a smooth pedestal.\n')
    print('%-10s %6s %7s %7s %7s %7s %9s %11s'
          % ('capture', 'min', 'p0.1', 'median', 'p99.9', 'max', 'at-min %', 'sigma(MAD)'))
    for name in CAPTURES:
        b = _band(os.path.join(SRC, name + '.ser')).astype(np.float32)
        med = float(np.median(b))
        mad = float(1.4826 * np.median(np.abs(b - med)))
        at_min = 100.0 * float((b == b.min()).sum()) / b.size
        print('%-10s %6.0f %7.0f %7.0f %7.0f %7.0f %9.3f %11.2f'
              % (name, b.min(), np.percentile(b, 0.1), med,
                 np.percentile(b, 99.9), b.max(), at_min, mad))
    print()
    print('for comparison, the zenith capture (offset 220, gain 0) and an eclipse block:')
    for lbl, p in (('zenith 00_00_21',
                    r'G:\Joe Izen Spain 2026\2026-08-13\zenith\00_00_21.ser'),
                   ('ecl 20_29_43',
                    r'G:\Joe Izen Spain 2026\2026-08-12\Sn2_Joe_20260812_182942\20_29_43.ser')):
        b = _band(p, k=5).astype(np.float32)
        med = float(np.median(b))
        mad = float(1.4826 * np.median(np.abs(b - med)))
        at_min = 100.0 * float((b == b.min()).sum()) / b.size
        print('%-10s %6.0f %7.0f %7.0f %7.0f %7.0f %9.3f %11.2f'
              % (lbl[:10], b.min(), np.percentile(b, 0.1), med,
                 np.percentile(b, 99.9), b.max(), at_min, mad))


def _run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def do_solve():
    """Stack a short run of each and plate-solve BLIND -- no site, no date guess, no
    reference. The only question here is where the telescope was pointed."""
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    site = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=HEIGHT * u.m)

    os.makedirs(OUT, exist_ok=True)
    print('%-10s %9s %10s %10s %9s %9s %8s %8s'
          % ('capture', 'centroids', 'RA', 'DEC', 'ROLL', 'ps("/px)', 'alt', 'az'))
    for name in CAPTURES:
        d1 = os.path.join(OUT, 's1_' + name)
        z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
        if not z:
            # 30 frames is enough to plate solve and a tenth of the read of all 100
            _run([PY, '-m', 'mee2024.cli', 'stack', os.path.join(SRC, name + '.ser'),
                  '--frames', '1-30', '--dark', DARK, *S1,
                  '--no-display', '--quiet', '-o', d1], os.path.join(d1, 'stage1.log'))
            z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
        if not z:
            print('%-10s   stage 1 FAILED -- see %s' % (name, os.path.join(d1, 'stage1.log')))
            continue
        import zipfile
        import pandas as pd
        n = len(pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv')))
        d2 = os.path.join(OUT, 's2_' + name)
        if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
            _run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
                  '--set', 'distortion_fixed_coefficients=None',
                  '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=12',
                  '--set', 'rough_match_threshhold=36',
                  '--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
                  '--set', 'observation_date=2026-08-11', '--set', 'guess_date=False',
                  '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
        r = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if not r:
            print('%-10s %9d   stage 2 FAILED -- see %s'
                  % (name, n, os.path.join(d2, 'stage2.log')))
            continue
        j = json.load(open(r[0], encoding='utf-8'))
        H = read_header(os.path.join(SRC, name + '.ser'))
        T = Time(H['utc'] if 'utc' in H else '2026-08-11 22:54:32')
        c = SkyCoord(j['RA'] * u.deg, j['DEC'] * u.deg)
        a = c.transform_to(AltAz(obstime=T, location=site))
        print('%-10s %9d %10.4f %10.4f %9.4f %9.7f %8.2f %8.2f'
              % (name, n, j['RA'], j['DEC'], j['ROLL'],
                 j['platescale (arcseconds/pixel)'], a.alt.deg, a.az.deg))


if __name__ == '__main__':
    {'probe': do_probe, 'solve': do_solve}[sys.argv[1] if len(sys.argv) > 1 else 'probe']()
