"""Reduce Husillos' horizon fields, so the vertical nuisance can be gated the way Leon gated it.

Leon defined its vertical nuisance on HORIZON NIGHT FIELDS, not on its eclipse field:
the direction from the night maps' polarisation measurement (V/H ~ 2.3) and the degree from a
NULL TEST on fields where L is known to be zero (docs/STEP3_2026.md, "S1 -- the nuisance
estimator, built and gated").  `hu_vertical.py` asked the same question of the eclipse field's
own 63 residuals, which is the wrong object: the deflection signal is in them, there is no
known-zero truth to score against, and 63 stars is a tenth of what a night field gives.

Husillos has the equivalent data, on the SAME NIGHT as the eclipse and at both eclipse gains
(`hu_horizon.py`):

    cal 8 deg   3 captures, 20:53-20:59 UTC   gains 0, 0, 125    the eclipse altitude
    10 deg      7 captures, 21:31-21:44 UTC   gains 0, 125, ...  ~+2 deg, Leon's H2 analogue

This tool stacks them and fits them against the same zenith quintic reference the eclipse
blocks use, at the same rung.  Corrections are ON with the site: at 8-10 degrees altitude
nothing is refraction-safe (hu_zenith_order's docstring), which is the whole reason these
fields are the right analogue of the eclipse geometry in the first place.

Stage 1 uses the ZENITH star-field preset, not the eclipse one: there is no Sun in these
frames, so no occulter, no coronal subtraction and no saturated blob.  The synthetic
hot-pixel dark IS applied -- these captures are where it was built from, and leaving the hot
pixels in cost the eclipse field its plate solve (section 3c of the eclipse record).

    .venv/Scripts/python.exe tools/husillos2026/hu_horizon_reduce.py [probe|all]

`probe` does one capture per window and reports whether they solve and how deep; `all` does
every capture.  Each stage 1 reads a 12 GB SER, so `all` is an hours-long job.
"""
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')

SRC = r'G:\Joe Izen Spain 2026\2026-08-12'          # READ ONLY
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'horizon')
DARK = os.path.join(HUS, 'hotpixels', 'husillos_synthetic_dark_all.fit')
REF = os.path.join(HUS, 'step3', 'ref')

#: (tag, folder, file, gain, first, last).  Frame 0 is skipped everywhere: the zenith work
#: measured it as the frame most likely to carry a defect, and one bad frame ruins a stack
#: (docs/HUSILLOS2026_ZENITH.md).
CAPTURES = [
    ('h8_g0_a',   'cal 8 deg', '22_53_15',   0, 1, 99),
    ('h8_g0_b',   'cal 8 deg', '22_56_41',   0, 1, 99),
    ('h8_g125',   'cal 8 deg', '22_59_14', 125, 1, 99),
    ('h10_g0',    '10 deg',    '23_31_59',   0, 1, 99),
    ('h10_g125a', '10 deg',    '23_34_38', 125, 1, 99),
    ('h10_g125b', '10 deg',    '23_37_17', 125, 1, 99),
    ('h10_g0_c',  '10 deg',    '23_41_01',   0, 1, 50),
    ('h10_g125c', '10 deg',    '23_42_43', 125, 1, 49),
    ('h10_g125d', '10 deg',    '23_44_06', 125, 1, 49),
    # 23_40_27 has 14 frames; too short to stack against the others, left out deliberately
]
PROBE = ('h8_g125', 'h10_g125a')

#: the zenith star-field preset, identical to hu_zenith_order.S1
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
      '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
      '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
      '--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular']

#: corrections ON: at 8-10 deg altitude nothing is refraction-safe
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0',
        '--set', 'observation_temp=25.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_humidity=0.35']

def midtime(folder, name):
    """Mid-capture UTC, READ from the capture's own CameraSettings.txt.

    An earlier version of this file carried a hard-coded table of these, computed by adding
    half the duration to the start time. That is the same species of mistake as the invented
    Leon altitude: the number is written down in the data and there is no reason to derive it.
    """
    p = os.path.join(SRC, folder, name + '.CameraSettings.txt')
    for line in open(p, encoding='utf-8', errors='replace'):
        if line.startswith('MidCapture='):
            return line.split('=', 1)[1].strip()[11:19]   # 2026-08-12T20:54:20.7Z -> 20:54:20
    raise SystemExit('no MidCapture in ' + p)


def run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def czip(d):
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return z[0] if z else None


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def refzip():
    r = glob.glob(os.path.join(REF, '**', 'distortion_data*.zip'), recursive=True)
    if not r:
        raise SystemExit('build the zenith reference first: hu_step3.py ref')
    return r[0]


def stage1(tag, folder, name, first, last):
    d = os.path.join(OUT, 's1_' + tag)
    if czip(d):
        return d
    ser = os.path.join(SRC, folder, name + '.ser')
    # the container plus --frames: a `path.ser#N` on the command line globs to nothing
    run([PY, '-m', 'mee2024.cli', 'stack', ser,
         '--frames', '%d-%d' % (first, last), '--dark', DARK,
         *S1, '--no-display', '--quiet', '-o', d],
        os.path.join(d, 'stage1.log'))
    return d


def stage2(tag, d1, tmid):
    d = os.path.join(OUT, 's2_' + tag)
    if results(d):
        return d
    z = czip(d1)
    if not z:
        return None
    # same rung as the eclipse blocks: cubic and higher frozen from the zenith reference
    run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', 'quintic',
         '--set', 'distortion_reference_files=' + refzip(),
         '--set', 'distortion_fixed_coefficients=quadratic',
         '--set', 'distortion_free_scale=True',
         '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
         '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
         *SITE, '--set', 'observation_time=' + tmid,
         '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    return d


def main(which):
    os.makedirs(OUT, exist_ok=True)
    todo = [c for c in CAPTURES if which == 'all' or c[0] in PROBE]
    print('reducing %d horizon captures\n' % len(todo))
    print('%-11s %5s %7s %9s %10s %12s' % ('tag', 'gain', 'stars', 'rms (")', 'ps ("/px)',
                                           'centroids'))
    for tag, folder, name, gain, first, last in todo:
        d1 = stage1(tag, folder, name, first, last)
        z = czip(d1)
        if not z:
            print('%-11s %5d   stage 1 FAILED (see %s)' % (tag, gain, d1))
            continue
        d2 = stage2(tag, d1, midtime(folder, name))
        j = results(d2) if d2 else None
        import zipfile
        import pandas as pd
        n_cent = len(pd.read_csv(zipfile.ZipFile(z).open('STACKED_CENTROIDS_DATA.csv')))
        if not j:
            print('%-11s %5d %7s %9s %10s %12d'
                  % (tag, gain, 'FAILED', '-', '-', n_cent))
            continue
        print('%-11s %5d %7d %9.4f %10.7f %12d'
              % (tag, gain, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)'], n_cent))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'probe')
