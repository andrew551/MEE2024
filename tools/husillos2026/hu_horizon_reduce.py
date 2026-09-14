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
    10 deg      7 captures, 21:31-21:44 UTC   gains 0, 125, ...  THREE pointings (see main)

This tool stacks them and fits them against the same zenith quintic reference the eclipse
blocks use, at the same rung.  Corrections are ON with the site: at 8-10 degrees altitude
nothing is refraction-safe (hu_zenith_order's docstring), which is the whole reason these
fields are the right analogue of the eclipse geometry in the first place.

Stage 1 uses the ZENITH star-field preset, not the eclipse one: there is no Sun in these
frames, so no occulter, no coronal subtraction and no saturated blob.  The synthetic
hot-pixel dark IS applied -- these captures are where it was built from, and leaving the hot
pixels in cost the eclipse field its plate solve (section 3c of the eclipse record).

    .venv/Scripts/python.exe tools/husillos2026/hu_horizon_reduce.py [probe|all]

`probe` does one capture per window; `h10` does the dark-sky `10 deg` window only; `deep` re-runs the three `cal 8 deg` captures with the
eclipse blocks' own detection settings (see DEEP below); `deep10` does the same for the `10 deg` window; `all` does every capture.
Each stage 1 reads a 12 GB SER, so `all` is an hours-long job.  Outputs go to s1_/s2_<tag> (zenith preset) and s1d_/s2d_<tag> (DEEP).

The `10 deg` folder is NOT one tracked field -- see main(): three pointings at 10.0, 5.7 and 15.0 deg altitude.
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
HUS = r'F:\MEE_output\husillos2026'
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
    # 23_42_43: stage 1 cannot match frame 48 to frame 0 -- the mount slewed from the 5.7 deg
    # pointing to the 15 deg one INSIDE this capture, so only the frames before the slew stack
    ('h10_g125c_pre', '10 deg', '23_42_43', 125, 1, 47),
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

#: The `cal 8 deg` window needs this, and it is a borrowed convention rather than a new number.
#:
#: The probe found `cal 8 deg` 6x shallower than `10 deg` -- 27 centroids against 165, and stage
#: 2 stopped at 20 matched stars where a quintic needs 21 -- and the cause is physical, not a
#: setting: it was shot with the Sun 15.6 to 16.5 deg below the horizon, i.e. still inside
#: ASTRONOMICAL TWILIGHT, looking due west (az 272.7) at alt 8.5 into the residual glow.  The
#: `10 deg` window is 35 minutes later at Sun -21 to -22.5 deg, in full night.
#:
#: So the shallow field is given the DETECTION SETTINGS THE ECLIPSE BLOCKS THEMSELVES USE
#: (min_area 2, Gaussian-subtracted detection at 4.0 sigma, sigma_subtract 0) -- cell 2's
#: eclipse convention, `tools/matrix_station1/s1_eclipse_corona.py`, already the convention of
#: record for this cell's science frames.  What is NOT taken from there is the coronal
#: subtraction and the occulter: there is no Sun in these frames.
DEEP = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
        '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
        '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
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


def stage1(tag, folder, name, first, last, deep=False):
    d = os.path.join(OUT, ('s1d_' if deep else 's1_') + tag)
    if czip(d):
        return d
    ser = os.path.join(SRC, folder, name + '.ser')
    # the container plus --frames: a `path.ser#N` on the command line globs to nothing
    run([PY, '-m', 'mee2024.cli', 'stack', ser,
         '--frames', '%d-%d' % (first, last), '--dark', DARK,
         *(DEEP if deep else S1), '--no-display', '--quiet', '-o', d],
        os.path.join(d, 'stage1.log'))
    return d


def stage2(tag, d1, tmid, deep=False):
    d = os.path.join(OUT, ('s2d_' if deep else 's2_') + tag)
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
    deep = which == 'deep'
    if which in ('h10', 'deep10'):
        # The dark-sky window in time order.  An earlier comment here called it "seven
        # consecutive captures of one tracked field 2.5 min apart at 10.4 -> 8.5 deg" -- that
        # was one plate solve propagated over the whole folder.  Solving each capture
        # (2026-09-12) gives THREE pointings: 23_31_59 and 23_34_38 at alt 10.0 deg (re-pointed
        # 0.66 deg in RA between them, i.e. Joe held 10 deg), 23_41_01 at 5.7 deg, and 23_44_06
        # at 15.0 deg, with the slew between the last two caught inside 23_42_43.  Only
        # consecutive captures of ONE pointing make a null pair (hu_atmosphere.py checks).
        #
        # `deep10` repeats the window with the eclipse blocks' detection settings (DEEP): at
        # gain 0 and 1 s through 5.6 air masses the zenith preset found 71 stars, and a null
        # pair is only as good as its shallower member.
        deep = which == 'deep10'
        todo = [c for c in CAPTURES if c[1] == '10 deg']
    elif deep:
        # gain 125 FIRST: it is the decisive one. The probe left it at 20 matched stars
        # against the 21 a quintic needs, so it is the capture that decides whether the
        # eclipse-altitude window is usable at all; the two gain-0 captures are shallower
        # and were shot even earlier in twilight. Deep detection costs ~45 min per capture
        # against ~7 for the zenith preset (min_area 2 on a twilight-bright field makes a
        # great many candidates), so running the decisive one first is worth an hour.
        order = {'h8_g125': 0, 'h8_g0_b': 1, 'h8_g0_a': 2}
        todo = sorted([c for c in CAPTURES if c[1] == 'cal 8 deg'],
                      key=lambda c: order[c[0]])
    else:
        todo = [c for c in CAPTURES if which == 'all' or c[0] in PROBE]
    print('reducing %d horizon captures%s\n' % (len(todo), ' (deep detection)' if deep else ''))
    print('%-11s %5s %7s %9s %10s %12s' % ('tag', 'gain', 'stars', 'rms (")', 'ps ("/px)',
                                           'centroids'))
    for tag, folder, name, gain, first, last in todo:
        d1 = stage1(tag, folder, name, first, last, deep)
        z = czip(d1)
        if not z:
            print('%-11s %5d   stage 1 FAILED (see %s)' % (tag, gain, d1))
            continue
        d2 = stage2(tag, d1, midtime(folder, name), deep)
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
