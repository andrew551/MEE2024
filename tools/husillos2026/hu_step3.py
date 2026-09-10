"""Cell 4's first Method 2: the eclipse field against the zenith reference, and a deflection.

Douglas, 2026-09-10: *"Let's do a quick Method 2 calculation with what we have so far."*  This
does it, and the word to hold on to is **quick**.  Four things are missing that every closed cell
in the matrix has, and each of them is a term in the budget rather than a rounding:

  * **one zenith field, not seventeen.**  Cell 2's reference is seventeen fields averaged; this
    is one.  The reference's own field-to-field scatter -- the number `s1_reference_tolerance.py`
    calls (b), 27-32 mas at the corners for Leon -- cannot be measured from a single field, so it
    is missing from the budget entirely rather than merely uncertain.
  * **no atmospheric data.**  Joe took none.  Cells 1-3 take their atmosphere term from zenith
    nulls -- sixteen of them for cell 2 -- and one zenith field yields none, so the +-0.11 to
    +-0.33 " that every other cell carries has no counterpart here.
  * **the weather is assumed.**  926.5 hPa is the standard atmosphere at 743 m; 25 C and 35 % are
    ordinary August evening values.  At 8.6 deg altitude refraction is 384 " and its second
    derivative across the field is 19 ", so this is not a small assumption -- see the sensitivity
    `hu_eclipse_overlap.py` reports, where turning refraction on at all moved the matched-star
    count from 17 to 71.
  * **no darks and no flats**, so the flat term cells 1 and 2 measure is absent.

The pathway is cell 2's, rung for rung (`docs/V1_4_0_TESTING.md` section 5):

    zenith reference   free, quintic, gate 0.5 "        distortion_fixed_coefficients=None
    eclipse field      constant + distortion_free_scale, gates 20 " then 3 "

There is no daytime calibration field on the middle rung here yet -- CalibS exists and is not
reduced -- so this is the Station-1 shape: an eclipse field fitted straight against a zenith
reference with its scale free.  That is Method 2 by construction, and Method 1 is not available
until CalibS is reduced.

The admitted-star window is `analysis_window.WINDOWS['husillos2026']`, registered with its
citation before this tool was written, as CLAUDE.md requires.  It is cell 2's window inherited,
and the outer bound has NOT been decided on cell 4's own data.

  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py ref
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py stage2
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py stage3
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py report
"""
import glob
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analysis_window import WINDOWS  # noqa: E402

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
HUS = r"D:/MEE2024 output/MEE_output/husillos2026"
OUT = os.path.join(HUS, 'step3')
ZEN = os.path.join(HUS, 'zenith_order', 's1_with_f0')
ECL = os.path.join(HUS, 'eclipse', 's1_sn2_darkall')

WIN = WINDOWS['husillos2026']
R_SUN_AS = 958.2

#: Husillos, from the site card. The weather is ASSUMED -- see the module docstring.
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_temp=25.0', '--set', 'observation_humidity=0.35',
        '--set', 'observation_wavelength=0.55']
ZEN_TIME = '22:00:53'      # the zenith capture's mid-time, 12 Aug
ECL_TIME = '18:29:59'      # the trimmed Sn2's mid-time


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


def refpath():
    r = glob.glob(os.path.join(OUT, 'ref', '**', 'distortion_results.txt'), recursive=True)
    return r[0] if r else None


def do_ref():
    """The zenith reference: free QUINTIC, the order docs/HUSILLOS2026_ZENITH.md section 3.7 fixed."""
    d = os.path.join(OUT, 'ref')
    if refpath():
        print('reference already built')
    else:
        run([PY, '-m', 'mee2024.cli', 'distortion', czip(ZEN), '--order', 'quintic',
             '--set', 'distortion_fixed_coefficients=None',
             '--set', 'distortion_fit_tol=0.5', '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=36', *SITE,
             '--set', 'observation_time=' + ZEN_TIME,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results(d)
    print('zenith reference: %s' % ('%d stars, rms %.4f ", plate scale %.7f "/px'
                                    % (j['#stars used'], j['final rms error (arcseconds)'],
                                       j['platescale (arcseconds/pixel)']) if j else 'FAILED'))


def do_stage2():
    """The eclipse field on the ladder's eclipse rung: constant + free scale, 20 " then 3 "."""
    ref = refpath()
    if not ref:
        print('build the reference first')
        return
    d = os.path.join(OUT, 'eclipse_twopass')
    if not results(d):
        run([PY, '-m', 'mee2024.cli', 'distortion', czip(ECL), '--order', 'quintic',
             '--fix-distortion', ref,
             '--set', 'distortion_fixed_coefficients=constant',
             '--set', 'distortion_free_scale=True',
             '--set', 'distortion_fit_tol_initial=20.0',
             '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *SITE,
             '--set', 'observation_time=' + ECL_TIME,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results(d)
    print('eclipse field vs the reference: %s'
          % ('%d stars, rms %.4f ", plate scale %.7f "/px (fitted, free scale)'
             % (j['#stars used'], j['final rms error (arcseconds)'],
                j['platescale (arcseconds/pixel)']) if j else 'FAILED'))
    if j:
        # CLAUDE.md: read the rung back from the run's own results before believing any claim
        print('   fixed distortion order: %s   |  plate scale source: %s  |  free scale: %s'
              % (j.get('fixed distortion order', '?'), j.get('plate scale source', '?'),
                 j.get('distortion_free_scale', '?')))


def dzip():
    z = glob.glob(os.path.join(OUT, 'eclipse_twopass', '**', 'distortion_data*.zip'),
                  recursive=True)
    return z[0] if z else None


def do_stage3():
    """Method 2, through the registered window."""
    z = dzip()
    if not z:
        print('no stage-2 output; run stage2 first')
        return
    d = os.path.join(OUT, 'method2')
    run([PY, '-m', 'mee2024.cli', 'eclipse', z,
         '--set', 'eclipse_method=Method 2',
         '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
         '--set', 'limit_radial_sun_radii=True',
         '--set', 'limit_radial_sun_radii_value=%.1f' % WIN.rmax,
         '--set', 'remove_double_stars_eclipse=False',
         '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage3.log'))
    for f in sorted(glob.glob(os.path.join(d, '**', '*.txt'), recursive=True)):
        txt = open(f, encoding='utf-8', errors='replace').read()
        if 'Method 2' in txt or 'L=' in txt:
            print('--- %s' % os.path.basename(f))
            # the results file carries mu and +-; the Windows console is cp1252, so anything
            # it cannot encode is transliterated rather than crashing the run
            safe = txt.replace('±', '+-').replace('μ', 'mu').replace('�', '?')
            print(safe.strip()[:2500].encode('ascii', 'replace').decode('ascii'))


def do_report():
    print('=' * 96)
    print('CELL 4, A FIRST METHOD 2 -- preliminary, see the caveats')
    print('=' * 96)
    j = results(os.path.join(OUT, 'ref'))
    if j:
        print('zenith reference : 1 field, free quintic, gate 0.5 "  -> %d stars, rms %.4f ", '
              'ps %.7f' % (j['#stars used'], j['final rms error (arcseconds)'],
                           j['platescale (arcseconds/pixel)']))
    k = results(os.path.join(OUT, 'eclipse_twopass'))
    if k:
        print('eclipse field    : constant + free scale, 20 " then 3 " -> %d stars, rms %.4f ", '
              'ps %.7f' % (k['#stars used'], k['final rms error (arcseconds)'],
                           k['platescale (arcseconds/pixel)']))
        if j:
            print('                   scale moved %+.0f ppm from the zenith reference'
                  % (1e6 * (k['platescale (arcseconds/pixel)']
                            / j['platescale (arcseconds/pixel)'] - 1)))
    z = dzip()
    if z:
        zf = zipfile.ZipFile(z)
        n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
        if n:
            t = pd.read_csv(zf.open(n[0]), dtype={'ID': str})
            t.columns = [c.strip() for c in t.columns]
            t = t[~t['flag_is_outlier'].astype(bool)]
            print()
            print('the window: G <= %.1f, %.1f-%.1f R_sun  (%s)'
                  % (WIN.mag, WIN.rmin, WIN.rmax, 'analysis_window.WINDOWS[husillos2026]'))
            print('   %d fitted stars, G %.2f-%.2f' % (len(t), t.magV.min(), t.magV.max()))
    for tag, d in (('Method 2 only', 'method2'), ('Methods 1 and 2', 'both_methods')):
        fs = sorted(glob.glob(os.path.join(OUT, d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True))
        if not fs:
            continue
        print()
        print('--- %s' % tag)
        txt = io.open(fs[0], encoding='utf-8', errors='replace').read()
        for line in txt.splitlines():
            l = line.strip()
            if any(k in l for k in ('Method 1 results', 'Method 2 results', 'number of stars',
                                    'deflected star position rms', 'deflection constant')):
                # the results file carries a plus-minus sign and the Windows console is
                # cp1252, so anything it cannot encode is transliterated, never fatal
                print('   ' + l.replace(chr(0x00b1), ' +- ')
                      .encode('ascii', 'replace').decode('ascii')[:150])
    print()
    print('  MISSING FROM THE BUDGET, not merely uncertain:')
    print('   * one zenith field, not seventeen -- no reference field-to-field term at all')
    print('   * no atmospheric data -- no zenith-null atmosphere term (cells 1-3 carry '
          '0.11-0.33 ")')
    print('   * the weather is assumed at 926.5 hPa / 25 C / 35 % RH')
    print('   * no darks and no flats')
    print('   * the outer radial bound is inherited, not decided on this cell\'s data')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    os.makedirs(OUT, exist_ok=True)
    {'ref': do_ref, 'stage2': do_stage2, 'stage3': do_stage3, 'report': do_report}[cmd]()
