"""Station 2's bracket refitted in the convention Bruns 2017 and Leon 2026 actually used.

Douglas, 2026-09-08: how did the other two cells carry the night-time zenith model into their
DAYTIME calibration field -- were the quadratic coefficients allowed to move, or frozen to the
night? Read back from the runs themselves (`fixed distortion order` in distortion_results.txt):

  Bruns 2017  L and R8 against night reference EC06 ............ quadratic
  Leon 2026   CAL_piLeo against the six 08-12 zenith fields .... quadratic
  Station 2   the L/R bracket against the fifteen zenith fields . CONSTANT  <-- the odd one out

The option names the order ABOVE which coefficients are frozen, so `quadratic` lets the
constant, linear and quadratic terms re-fit in daylight and freezes only the cubic, while
`constant` freezes linear, quadratic and cubic alike and leaves only the pointing offsets
(the Station 2 runs added `distortion_free_scale` to get a scale out at all).

That matters here because a frozen quadratic cannot absorb a day-night change in the optic's
low-order distortion; whatever changed is pushed into the one free parameter, the scale -- and
on this field 47 ppm of scale is 0.77 " of deflection constant. So the bracket is refitted the
published way and Method 1 rebuilt on top of it.

  .venv/Scripts/python.exe tools/matrix_station2/s2_bracket_convention.py
"""
import glob
import json
import os
import subprocess
import sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/mexico2024/station2_transfer"
LR = os.path.join(OUT, "lr_trimmed")
ECL = os.path.join(OUT, "eclipse")
TIERS = (("100ms", "18:12:07"), ("075ms", "18:13:20"))
BRACKET_TIMES = (("right", "18:10:55"), ("left", "18:14:30"))


def site(temp, hum):
    return ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
            '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
            '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
            '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
            '--set', 'observation_wavelength=0.633', '--set', 'observation_temp=%.1f' % temp,
            '--set', 'observation_humidity=%.2f' % hum]


SITE = site(15.2, 0.24)


def results_of(d):
    g = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(g[0], encoding='utf-8')) if g else None


def path_of(d):
    return glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)[0]


def run(cmd, log):
    with open(log, 'w') as fh:
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------- 1. the bracket, quadratic-free
refs = sorted(glob.glob(os.path.join(OUT, 'ref_cubic', '*', '**', 'distortion_results.txt'), recursive=True))
print('cubic zenith reference: %d fields' % len(refs))
brk = {}
for name, tm in BRACKET_TIMES:
    z = glob.glob(os.path.join(LR, name, 'centroid_data*.zip'))
    if not z:
        print('  %s: no stage-1 archive' % name); sys.exit(1)
    d = os.path.join(OUT, 'bracket_quadfree', name)
    os.makedirs(d, exist_ok=True)
    if not results_of(d):
        # no distortion_free_scale: with the linear terms free the scale is already a free parameter,
        # which is how Bruns' L/R8 and Leon's CAL_piLeo were fitted.
        run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
             '--fix-distortion', *refs,
             '--set', 'distortion_fixed_coefficients=quadratic',
             '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
             *SITE, '--set', 'observation_time=' + tm,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results_of(d)
    if not j:
        print('  %s: FAILED\n    %s' % (name, open(os.path.join(d, 'stage2.log')).read()[-500:])); sys.exit(1)
    brk[name] = j
    print('  %-5s %4d stars  rms %.4f"  ps %.7f  (fixed above %s)'
          % (name, j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)'], j['fixed distortion order']))

OLD = 1.8659016566709439                      # constant-only bracket mean, the rev02 Method 1 scale
NEW = 0.5 * (brk['left']['platescale (arcseconds/pixel)'] + brk['right']['platescale (arcseconds/pixel)'])
SPREAD = abs(brk['left']['platescale (arcseconds/pixel)'] - brk['right']['platescale (arcseconds/pixel)'])
print('\nimported scale, quadratic-free: %.7f "/px  (was %.7f, %+.1f ppm; L-R spread %.1f ppm, was 31.3)'
      % (NEW, OLD, 1e6 * (NEW - OLD) / OLD, 1e6 * SPREAD / NEW))

# ---------------------------------------------------------------- 2. Method 1 on the new scale
for tag, tm in TIERS:
    z = glob.glob(os.path.join(ECL, tag, 'centroid_data*.zip'))
    d = os.path.join(ECL, tag, 'stage2_method1_quadfree')
    os.makedirs(d, exist_ok=True)
    if not results_of(d):
        run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
             '--fix-distortion', path_of(os.path.join(OUT, 'bracket_quadfree', 'right')),
             path_of(os.path.join(OUT, 'bracket_quadfree', 'left')),
             '--set', 'distortion_fixed_coefficients=constant',
             '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=' + ('36' if tag == '075ms' else '100'),
             *SITE, '--set', 'observation_time=' + tm,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results_of(d)
    print('  %s Method 1: %s' % (tag, ('%d stars, rms %.4f", ps %.7f'
          % (j['#stars used'], j['final rms error (arcseconds)'], j['platescale (arcseconds/pixel)']))
          if j else 'FAILED'))
