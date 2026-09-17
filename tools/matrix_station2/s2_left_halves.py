"""Does the tail of the left bracket distort its plate scale? (Douglas, 2026-09-08)

The left calibration field is shot immediately after C3, while the sky is brightening fast: its
level rises 29.7 % across the 47 frames already kept, and six more were trimmed at the end
because the rise was steeper still. Douglas' question is whether frames at the end are dragging
the fit -- the left scale is the one that pushes Method 1 to 2.49 " where the right gives 2.04 ".

So split the kept series in half and fit each half on its own, plus a shorter tail-trimmed
version. If the tail is the problem the first half and the trimmed set agree with each other and
disagree with the whole.

The right field is split the same way as the control: its sky is FALLING, so if the halves of the
right agree while the halves of the left do not, the brightening is the mechanism.

  .venv/Scripts/python.exe tools/matrix_station2/s2_left_halves.py
"""
import glob
import json
import os
import subprocess

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
SRC = r"I:/Station2Export/EclipseImages/eclipse"
OUT = r"F:/MEE_output/mexico2024/station2"
DEST = os.path.join(OUT, "lr_halves")

# The eclipse-convention stage-1 settings the bracket was stacked with (s2_stage2.py's LR set).
ECLIPSE = ['sensitive_mode_stack=True', 'centroid_gaussian_subtract=True',
           'centroid_gaussian_thresh=4.0', 'min_area=2', 'centroid_refine_window=False',
           'background_subtraction_mode=annular', 'delete_saturated_blob=False',
           'centroid_gap_blob=30', 'eclipse_mask_mode=disk', 'eclipse_disk_margin_px=10',
           'max_star_mag_dist=12.0', 'rough_match_threshhold=36.0']
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
        '--set', 'observation_wavelength=0.633', '--set', 'observation_temp=15.2',
        '--set', 'observation_humidity=0.24']

# (label, folder, --frames range as the CLI counts them, mid-time)
# the kept sets are right 7-49 and left 0-46, from the trimming study; halves split those.
RUNS = [('right_all', 'Right Field', '7-49', '18:10:55'),
        ('right_first', 'Right Field', '7-28', '18:10:45'),
        ('right_second', 'Right Field', '29-49', '18:11:07'),
        ('left_all', 'Left Field', '0-46', '18:14:30'),
        ('left_first', 'Left Field', '0-23', '18:14:17'),
        ('left_second', 'Left Field', '24-46', '18:14:44'),
        ('left_early30', 'Left Field', '0-29', '18:14:20')]


def results_of(d):
    g = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(g[0], encoding='utf-8')) if g else None


refs = sorted(glob.glob(os.path.join(OUT, 'ref_cubic', '*', '**', 'distortion_results.txt'),
                        recursive=True))
print('cubic zenith reference: %d fields\n' % len(refs))
print('%-14s %6s %6s %8s %12s %10s' % ('set', 'frames', 'stars', 'rms (")', 'scale ("/px)', 'vs whole'))
base = {}
for label, folder, rng, tm in RUNS:
    d = os.path.join(DEST, label)
    os.makedirs(d, exist_ok=True)
    if not glob.glob(os.path.join(d, 'centroid_data*.zip')):
        cmd = [PY, '-m', 'mee2024.cli', 'stack', os.path.join(SRC, folder, '*.FIT'),
               '-o', d, '--no-display', '--quiet', '--no-config', '--frames', rng]
        for kv in ECLIPSE:
            cmd += ['--set', kv]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print('%-14s STAGE 1 FAILED: %s' % (label, (r.stderr or r.stdout).strip()[-200:]))
            continue
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('%-14s no archive' % label); continue
    s2 = os.path.join(d, 'stage2')
    os.makedirs(s2, exist_ok=True)
    if not results_of(s2):
        cmd = [PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
               '--fix-distortion', *refs,
               '--set', 'distortion_fixed_coefficients=quadratic',
               '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
               '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
               *SITE, '--set', 'observation_time=' + tm,
               '--no-display', '--quiet', '-o', s2]
        with open(os.path.join(s2, 'stage2.log'), 'w') as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    j = results_of(s2)
    if not j:
        print('%-14s stage 2 found nothing' % label); continue
    ps = j['platescale (arcseconds/pixel)']
    side = label.split('_')[0]
    if label.endswith('_all'):
        base[side] = ps
    delta = '' if side not in base else '%+7.1f ppm' % (1e6 * (ps - base[side]) / base[side])
    n = int(rng.split('-')[1]) - int(rng.split('-')[0]) + 1
    print('%-14s %6d %6d %8.4f %12.7f %10s'
          % (label, n, j['#stars used'], j['final rms error (arcseconds)'], ps, delta))
