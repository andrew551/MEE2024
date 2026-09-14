"""Cell 4's first Method 2: the eclipse field against the zenith reference, and a deflection.

Douglas, 2026-09-10: *"Let's do a quick Method 2 calculation with what we have so far."*  This
does it, and the word to hold on to is **quick**.  Four things are missing that every closed cell
in the matrix has, and each of them is a term in the budget rather than a rounding:

  * **one zenith field, not seventeen.**  Cell 2's reference is seventeen fields averaged; this
    is one.  (There ARE two science blocks -- the same field at two gains -- so that is not a
    limitation; the reference is.)  The reference's own field-to-field scatter -- the number `s1_reference_tolerance.py`
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

The pathway (Douglas, 2026-09-10): **take the cubic and higher coefficients from the zenith
field and apply them to the eclipse field**, and let the eclipse field fit its own constant,
linear and quadratic.

    zenith reference   free, quintic, gate 0.5 "     distortion_fixed_coefficients=None
    eclipse field      distortion_fixed_coefficients=QUADRATIC, gates 20 " then 3 "

`distortion_fixed_coefficients` names the highest order left FREE and freezes everything above it
(`distortion_polynomial.py`, `order_free = mapping[...]`), so `quadratic` is exactly "cubic and
higher from the reference".  An earlier version of this tool used `constant`, which also freezes
the linear and quadratic and left the fit with an 0.8767 " residual -- wrong for a field 3.5
hours and 73 degrees of altitude away from its reference, where the low orders have moved.

**Method 2 only.**  Method 1 imports a plate scale from a calibration field; Husillos has no
reduced calibration field yet, so there is nothing to import and nothing to report.

**The two-witness rule** (Douglas, 2026-09-10; adopted matrix-wide 2026-09-02,
`docs/MATRIX_2026.md`): admit only stars seen in BOTH blocks.  Leon adopted it because a
single-witness star cannot be arbitrated by the cross-tier consistency vet -- its one bad
detection has nothing to contradict it -- and it cost Leon six stars, +-0.04 " of statistical
error and -0.06 " of L.  Husillos' two witnesses are the same field at two gains, 57 s apart.

It is applied as a FILTER ON THE STAGE-2 OUTPUT, not as a re-implemented fit: stage 3 reads
`CATALOGUE_MATCHED_ERRORS.csv` out of the distortion zip (`eclipse_analysis.py`), so a copy of
the zip with the single-witness rows removed runs through stage 3's own arithmetic unchanged.

**The field is NOT cropped radially** (Douglas, 2026-09-10).  `analysis_window.WINDOWS`
['husillos2026'] exists and is cited, and its outer bound is cell 2's inherited 10 R_sun which
has never been tested on cell 4's data; applying it here would drop stars on a borrowed number.
So every matched star is fitted and the window is recorded rather than enforced.

  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py ref
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py stage2
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py stage3
  .venv/Scripts/python.exe tools/husillos2026/hu_step3.py report
"""
import glob
import io
import re
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analysis_window import WINDOWS  # noqa: E402

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
HUS = r"F:/MEE_output/husillos2026"
OUT = os.path.join(HUS, 'step3')
ZEN = os.path.join(HUS, 'zenith_order', 's1_with_f0')

#: BOTH science acquisitions. Husillos does not have one science block: the same field was shot
#: twice within a minute at two different gains, and both plate-solve (docs/HUSILLOS2026_ECLIPSE
#: .md section 3b). They share 63 stars of 71 and 84.
BLOCKS = [('gain0', os.path.join(HUS, 'eclipse', 's1_sn2_darkall'), '18:29:59'),
          ('gain125', os.path.join(HUS, 'eclipse', 's1_sun_dark'), '18:29:20')]
ECL = BLOCKS[0][1]

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


def do_stage2(only=None):
    """Both science blocks against the zenith reference, cubic and higher frozen."""
    ref = refpath()
    if not ref:
        print('build the reference first')
        return
    for tag, src, tmid in BLOCKS:
        if only and tag != only:
            continue
        _stage2_one(ref, tag, src, tmid)


def _stage2_one(ref, tag, src, tmid):
    d = os.path.join(OUT, 'eclipse_%s' % tag)
    if not results(d):
        run([PY, '-m', 'mee2024.cli', 'distortion', czip(src), '--order', 'quintic',
             '--fix-distortion', ref,
             # cubic and higher from the zenith; constant, linear and quadratic re-fitted
             '--set', 'distortion_fixed_coefficients=quadratic',
             '--set', 'distortion_free_scale=True',
             '--set', 'distortion_fit_tol_initial=20.0',
             '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *SITE,
             '--set', 'observation_time=' + tmid,
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    j = results(d)
    print('%-8s vs the reference: %s' % (tag,
          '%d stars, rms %.4f ", plate scale %.7f "/px (fitted, free scale)'
          % (j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)']) if j else 'FAILED'))
    if j:
        # CLAUDE.md: read the rung back from the run's own results before believing any claim
        print('   fixed distortion order: %s   |  plate scale source: %s  |  free scale: %s'
              % (j.get('fixed distortion order', '?'), j.get('plate scale source', '?'),
                 j.get('distortion_free_scale', '?')))


def dzip(tag):
    z = glob.glob(os.path.join(OUT, 'eclipse_%s' % tag, '**', 'distortion_data*.zip'),
                  recursive=True)
    return z[0] if z else None


def do_stage3(only=None):
    """Method 2 on each block, uncropped."""
    for tag, _src, _t in BLOCKS:
        if only and tag != only:
            continue
        _stage3_one(tag)


def _stage3_one(tag):
    z = dzip(tag)
    if not z:
        print('%s: no stage-2 output' % tag)
        return
    print('=== %s' % tag)
    d = os.path.join(OUT, 'method2_%s' % tag)
    run([PY, '-m', 'mee2024.cli', 'eclipse', z,
         '--set', 'eclipse_method=Method 2',
         '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
         # NOT cropped radially: the outer bound in the registry is cell 2's, inherited and
         # never tested on this cell, so enforcing it would drop stars on a borrowed number
         '--set', 'limit_radial_sun_radii=False',
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


def _matched(tag):
    z = dzip(tag)
    if not z:
        return None
    zf = zipfile.ZipFile(z)
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
    if not n:
        return None
    t = pd.read_csv(zf.open(n[0]), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    return t


def do_witness():
    """The two-witness rule: keep only the stars both blocks matched, then run stage 3 as usual."""
    tabs = {tag: _matched(tag) for tag, _s, _t in BLOCKS}
    if any(v is None for v in tabs.values()):
        print('both blocks must be fitted first')
        return
    ids = [set(v['ID']) for v in tabs.values()]
    both = set.intersection(*ids)
    print('two-witness rule: %s' % ' , '.join('%s %d stars' % (k, len(v))
                                              for k, v in tabs.items()))
    print('   seen in both: %d   (single-witness dropped: %s)'
          % (len(both), ', '.join('%s %d' % (k, len(set(v['ID']) - both))
                                  for k, v in tabs.items())))
    for tag, _src, _t in BLOCKS:
        t = tabs[tag]
        keep = t['ID'].isin(both)
        # what the rule removes, in the currency that matters
        gone = t[~keep]
        if len(gone):
            print('   %-8s drops %d: G %s' % (tag, len(gone),
                                              ' '.join('%.2f' % g for g in sorted(gone.magV))))
        src = dzip(tag)
        dst = os.path.join(OUT, 'witness_%s.zip' % tag)
        zin = zipfile.ZipFile(src)
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                    buf = io.StringIO()
                    t[keep].to_csv(buf, index=False)
                    data = buf.getvalue().encode('utf-8')
                zout.writestr(item, data)
        d = os.path.join(OUT, 'method2_%s_witness' % tag)
        run([PY, '-m', 'mee2024.cli', 'eclipse', dst,
             '--set', 'eclipse_method=Method 2',
             '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
             '--set', 'limit_radial_sun_radii=False',
             '--set', 'remove_double_stars_eclipse=False',
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage3.log'))
        fs = sorted(glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True))
        if not fs:
            print('   %-8s stage 3 FAILED' % tag)
            continue
        for line in io.open(fs[0], encoding='utf-8', errors='replace').read().splitlines():
            l = line.strip()
            if any(k in l for k in ('Method 2 results', 'number of stars',
                                    'deflected star position rms')):
                print('   %-8s %s' % (tag, l.replace(chr(0x00b1), ' +- ')
                                      .encode('ascii', 'replace').decode('ascii')[:130]))


RSUN_ARCSEC = 947.1  # docs/STAGE3_THEORY.md section 3


def _stage3_arrays(d):
    """Radii (R_sun), deflections ("), plate scale, and both L values, out of a stage-3 report."""
    f = glob.glob(os.path.join(OUT, d, 'ECLIPSE_OUTPUT*.txt'))
    if not f:
        return None
    t = io.open(f[0], encoding='utf-8', errors='replace').read()

    def arr(key):
        m = re.search(re.escape(key) + r':\s*\[(.*?)\]', t, re.S)
        return np.array([float(x) for x in m.group(1).split()])
    return dict(r=arr('radial distances'), d=arr('deflection (arcsec)'),
                ps=float(re.search(r'platescale=([\d.]+)', t).group(1)),
                fixed=float(re.search(r'deflection constant = ([\d.]+)', t).group(1)),
                m2=float(re.search(r'L=([\d.]+)', t).group(1)),
                sig=float(re.search(r'L=[\d.]+.([\d.]+)', t).group(1)))


def do_compare():
    """Why the two blocks disagree: the plate scale, or per-star noise?

    Under the two-witness rule the blocks read L = 1.596 +- 0.507 and 2.782 +- 0.540 " on
    IDENTICAL stars.  Two stories, and both are testable rather than arguable:

      (a) the plate scale.  The fitted scales differ by 87 ppm and Method 2 fits L and the
          scale together.  dL = dS * h * R_sun (docs/STAGE3_THEORY.md s4) predicts a size AND
          a sign, and h is a property of THIS star field -- Leon's is 19.8 R_sun^2 and Bruns'
          8.2, so borrowing either would have been inventing a number.
      (b) per-star noise.  The same 64 stars are measured twice, so correlate them directly.

    Stars are paired between the blocks by radius.  Nearest-radius is NOT a bijection here --
    two stars sit 0.001 R_sun apart and a greedy match pairs one of them twice -- so this is a
    minimum-cost assignment and the worst residual is printed as the check.
    """
    A = _stage3_arrays('method2_gain0_witness')
    B = _stage3_arrays('method2_gain125_witness')
    if not A or not B:
        print('run witness first')
        return
    i, j = linear_sum_assignment(np.abs(A['r'][:, None] - B['r'][None, :]))
    assert (i == np.arange(len(A['r']))).all()
    b = dict(B, r=B['r'][j], d=B['d'][j])
    print('paired %d stars, worst radius mismatch %.4f R_sun'
          % (len(A['r']), np.abs(A['r'] - b['r']).max()))

    h = 1.0 / np.mean(1.0 / A['r'] ** 2)
    lever = h * RSUN_ARCSEC * 1e-6
    dS = (A['ps'] - b['ps']) / b['ps'] * 1e6
    print('\n(a) the plate scale')
    print('    mean radius             %.2f R_sun  (%.2f to %.2f)'
          % (A['r'].mean(), A['r'].min(), A['r'].max()))
    print('    h = 1/mean(1/r^2)       %.2f R_sun^2   (Leon 19.8, Bruns 8.2)' % h)
    print('    leverage                %.4f " of L per ppm  (naive h, not by injection)' % lever)
    print('    scales differ           %+.1f ppm' % dS)
    print('    predicted dL            %+.3f "' % (dS * lever))
    print('    observed  dL            %+.3f "   <- wrong sign, %.1fx too large'
          % (A['m2'] - b['m2'], abs(dS * lever / (A['m2'] - b['m2']))))

    rho = float(np.corrcoef(A['d'], b['d'])[0, 1])
    diff = np.std(A['d'] - b['d'])
    sd = (A['sig'] ** 2 + b['sig'] ** 2 - 2 * rho * A['sig'] * b['sig']) ** 0.5
    print('\n(b) per-star noise, same 64 stars')
    print('    correlation             %.3f' % rho)
    print('    per-star difference rms %.3f "  -> %.3f " independent noise per block'
          % (diff, diff / 2 ** 0.5))
    print('    each block own scatter  %.3f / %.3f "' % (np.std(A['d']), np.std(b['d'])))
    print('    sigma of the difference %.3f "  at that correlation' % sd)
    print('    the blocks differ by    %.3f " = %.1f sigma'
          % (b['m2'] - A['m2'], (b['m2'] - A['m2']) / sd))

    print('\n(c) the scale is not the cause, but it is an amplifier')
    for t, x in (('gain0', A), ('gain125', b)):
        print('    %-8s scale held %.3f  ->  Method 2 %.3f  (%+.3f ")'
              % (t, x['fixed'], x['m2'], x['m2'] - x['fixed']))
    print('    gap: %.3f " with the scale held, %.3f " with it free'
          % (abs(A['fixed'] - b['fixed']), abs(A['m2'] - b['m2'])))


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
    {'ref': do_ref, 'stage2': do_stage2, 'stage3': do_stage3,
     'witness': do_witness, 'compare': do_compare, 'report': do_report}[cmd]()
