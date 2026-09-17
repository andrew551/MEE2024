"""Method 3 on Station 2: the bracket's quadratic imported, the linear refitted on the eclipse field.

Douglas, 2026-09-14, defining the pathway: "using the quadratic terms from the external field
and the linear term from the eclipse field itself. The last method I would like to start
calling Method 3 from now on" -- then: "record_covariance.png ... can we create a new version
that includes the new Method 3 (import quadratic coefficients)."

WHAT THE SETTING IS.  `distortion_fixed_coefficients` names the highest order left FREE, so
"quadratic and above from the bracket, linear free" is `linear`.  Station 2's three pathways
then differ by one word each:

    constant    Method 1  only the constant free; linear, quadratic, cubic from the bracket,
                          and the bracket's plate scale imported with them
    linear      METHOD 3  constant and linear free on the eclipse field; quadratic and cubic
                          from the bracket; the scale fitted, not imported
    (zenith)    Method 2  against the 15-field cubic zenith reference, scale fitted

THE SCALE TAKES CARE OF ITSELF, and the reason is worth stating because it is the difference
between Method 1 and Method 3.  `distortion_polynomial` replaces the fitted scale with the
reference's ONLY at order_free == 0 -- the plate scale is the isotropic part of the linear map,
so once the linear terms are free the scale is fitted on the eclipse field whatever
`distortion_free_scale` says.  Method 3 therefore cannot import a scale, which is exactly why
it belongs beside Method 2 on the covariance chart rather than beside Method 1.

Everything else -- reference files, tolerances, magnitude cut, match threshold per tier, site
and time -- is copied from the Method 1 run in `s2_bracket_convention.py` so that the only
difference between the two outputs is the rung.

    .venv/Scripts/python.exe tools/matrix_station2/s2_method3.py
"""
import glob
import json
import os
import subprocess
import sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/mexico2024/station2"
ECL = os.path.join(OUT, "eclipse")
TIERS = (("100ms", "18:12:07"), ("075ms", "18:13:20"))

#: the eclipse-day site card, as every Station 2 tool states it
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
        '--set', 'observation_wavelength=0.633', '--set', 'observation_temp=15.2',
        '--set', 'observation_humidity=0.24']


def results_of(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def path_of(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return r[0] if r else None


def run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    right = path_of(os.path.join(OUT, 'bracket_quadfree', 'right'))
    left = path_of(os.path.join(OUT, 'bracket_quadfree', 'left'))
    if not (right and left):
        raise SystemExit('run tools/matrix_station2/s2_bracket_convention.py first')
    print('quadratic imported from the quadratic-free bracket:')
    for nm, p in (('right', right), ('left', left)):
        j = json.load(open(p, encoding='utf-8'))
        print('   %-5s %3d stars, rms %.4f ", ps %.7f "/px, free to %s'
              % (nm, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)'], j['fixed distortion order']))
    print()
    for tag, tm in TIERS:
        z = glob.glob(os.path.join(ECL, tag, 'centroid_data*.zip'))
        d = os.path.join(ECL, tag, 'stage2_method3')
        os.makedirs(d, exist_ok=True)
        if not results_of(d):
            run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
                 '--fix-distortion', right, left,
                 '--set', 'distortion_fixed_coefficients=linear',
                 '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                 '--set', 'max_star_mag_dist=13',
                 '--set', 'rough_match_threshhold=' + ('36' if tag == '075ms' else '100'),
                 *SITE, '--set', 'observation_time=' + tm,
                 '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
        j = results_of(d)
        if not j:
            print('  %s Method 3: FAILED\n%s' % (tag, open(os.path.join(d, 'stage2.log')).read()[-600:]))
            sys.exit(1)
        # CLAUDE.md: read the rung back from the run's own results rather than trusting the call
        print('  %-6s Method 3: %4d stars, rms %.4f ", ps %.7f "/px  | fixed above %s, scale %s'
              % (tag, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)'], j.get('fixed distortion order'),
                 j.get('plate scale source')))


if __name__ == '__main__':
    main()
