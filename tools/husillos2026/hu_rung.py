"""The eclipse field against CalibS at all three rungs, so the ladder is one table.

Douglas, 2026-09-12: "What is the value of L in a Method 1 calculation where we import only
the quadratic terms from CalibS but not the linear term?"

WHAT THAT SETTING IS.  `distortion_fixed_coefficients` names the highest order left FREE, so
"quadratic and above from CalibS, linear free" is `linear`.  Three rungs against one
calibration field:

    constant    only the constant free; linear, quadratic and above from CalibS   <- Method 1
    linear      constant and linear free; quadratic and above from CalibS         <- the ask
    quadratic   constant, linear, quadratic free; cubic and above from CalibS     <- record rung
                (CalibS' cubic+ are the zenith's, bit-identical, so this reproduces the
                reduction of record almost exactly and is included as the closing check)

AND WHY THE ASK CANNOT BE METHOD 1.  The plate scale IS the isotropic part of the linear
term.  `distortion_polynomial.py:298-312` shows it: only at order_free == 0 does the fitter
run a linear fit, discard the stretch and skew, and then REPLACE the scale with the
reference's `fix_platescale` -- and even that only when `distortion_free_scale` is off.  At
`linear` or above the linear coefficients are fitted on the eclipse field, so the scale is
fitted with them; `distortion_fitter.py:538` reports `plate scale source: fitted on this
field` and this tool asserts on that readback rather than on the arguments passed.

So the ask is METHOD 2 WITH CALIBS' QUADRATIC, not Method 1.  Stage 3 still prints a
"Method 1" line, but at this rung it means "hold the scale this same field just fitted",
which uses the data twice; the honest number at `linear` and `quadratic` is Method 2, and at
`constant` it is Method 1.  Both are printed at every rung so the degeneracy is visible.

WHAT ELSE MOVES WITH THE RUNG.  Each freed order absorbs part of a 1/r deflection before
stage 3 sees it (hu_absorption.py, confirmed by injection in hu_inject.py):

    constant   f = 1.000      linear   f = 0.911      quadratic   f = 0.860   (union of record)

so L at different rungs is not directly comparable until each is divided by its own f.  This
tool prints both.

    .venv/Scripts/python.exe tools/husillos2026/hu_rung.py [stage2|report|all]
"""
import glob
import io
import json
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from analysis_window import WINDOWS  # noqa: E402
from hu_method1 import BLOCKS, CALIBS, PY, SITE, calibs_zip, czip, results, run  # noqa: E402

WIN = WINDOWS['husillos2026']
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'step3')

#: (rung, output prefix, expected `plate scale source`, absorption fraction for the union).
#: `constant` reuses the settled-CalibS Method 1 runs already on disk (hu_method1.py, TAGP
#: m1s_) rather than repeating them under a second name.
RUNGS = [('constant', 'm1s_', 'imported from the reference files', 1.000),
         ('linear', 'lin_', 'fitted on this field', 0.911),
         ('quadratic', 'quad_', 'fitted on this field', 0.860)]


def dzip(prefix, tag):
    z = glob.glob(os.path.join(OUT, prefix + tag, '**', 'distortion_data*.zip'), recursive=True)
    return z[0] if z else None


def matched(prefix, tag):
    zf = zipfile.ZipFile(dzip(prefix, tag))
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    assert t['ID'].str.contains(r'\.', regex=True).sum() == 0, 'a float-shaped id got through'
    return t


def do_stage2():
    cal = calibs_zip()
    j = results(CALIBS)
    print('CalibS %s: %d stars, rms %.4f ", ps %.7f "/px +- %.1f ppm'
          % (os.path.basename(CALIBS), j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)'],
             j['platescale_relative_uncertainty'] * 1e6))
    print()
    for rung, prefix, want, _f in RUNGS:
        for tag, src, tmid in BLOCKS:
            d = os.path.join(OUT, prefix + tag)
            if not results(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', czip(src), '--order', 'quintic',
                     '--set', 'distortion_reference_files=' + cal,
                     '--set', 'distortion_fixed_coefficients=' + rung,
                     # only meaningful at `constant`; off everywhere so the readback below
                     # is the single source of truth about what the fit actually did
                     '--set', 'distortion_free_scale=False',
                     '--set', 'distortion_fit_tol_initial=20.0',
                     '--set', 'distortion_fit_tol=3.0',
                     '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
                     *SITE, '--set', 'observation_time=' + tmid,
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            r = results(d)
            if not r:
                print('%-10s %-9s stage 2 FAILED -- %s' % (rung, tag, os.path.join(d, 'stage2.log')))
                continue
            ok = r['fixed distortion order'] == rung and r['plate scale source'] == want
            print('%-10s %-9s %3d stars, rms %.4f ", ps %.7f "/px | order %s | scale %s  %s'
                  % (rung, tag, r['#stars used'], r['final rms error (arcseconds)'],
                     r['platescale (arcseconds/pixel)'], r['fixed distortion order'],
                     r['plate scale source'], 'OK' if ok else '<-- NOT THE RUNG ASKED FOR'))
            if not ok:
                raise SystemExit('refusing to continue on a rung that is not what was asked')


def _stage3(zpath, outdir):
    fs = glob.glob(os.path.join(outdir, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True)
    if not fs:
        run([PY, '-m', 'mee2024.cli', 'eclipse', zpath,
             '--set', 'eclipse_method=Method 1 & 2',
             '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
             # NOT cropped radially: Douglas, 2026-09-10
             '--set', 'limit_radial_sun_radii=False',
             '--set', 'remove_double_stars_eclipse=False',
             '--no-display', '--quiet', '-o', outdir], os.path.join(outdir, 'stage3.log'))
        fs = glob.glob(os.path.join(outdir, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True)
    if not fs:
        return {}
    txt = io.open(fs[0], encoding='utf-8', errors='replace').read()
    out = {}
    for m in re.finditer(r'Method (\d) results: L=([-\d.]+)\D+?([\d.]+), platescale=([\d.]+)',
                         txt):
        out['M' + m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def two_witness(prefix):
    """The two-witness subset of each block, written back as a stage-3 input.

    Mirrors hu_method1.do_stage3: the rule is matrix-wide (2026-09-02) and the union applies
    the same admission, so per-block and union numbers are on one star set.
    """
    tabs = {tag: matched(prefix, tag) for tag, _s, _t in BLOCKS}
    both = set(tabs[BLOCKS[0][0]]['ID']).intersection(tabs[BLOCKS[1][0]]['ID'])
    made = {}
    for tag, _s, _t in BLOCKS:
        t = tabs[tag]
        dst = os.path.join(OUT, prefix + 'witness_%s.zip' % tag)
        zin = zipfile.ZipFile(dzip(prefix, tag))
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                    buf = io.StringIO()
                    t[t['ID'].isin(both)].to_csv(buf, index=False)
                    data = buf.getvalue().encode('utf-8')
                zout.writestr(item, data)
        made[tag] = dst
    return made, len(both)


def do_report():
    print('%-10s %-9s %4s %15s %15s %11s %9s'
          % ('rung', 'block', 'N', 'Method 1 L (")', 'Method 2 L (")', 'ps ("/px)', 'f'))
    for rung, prefix, want, f in RUNGS:
        if not dzip(prefix, BLOCKS[0][0]):
            print('%-10s stage 2 not run' % rung)
            continue
        made, n = two_witness(prefix)
        for tag, _s, _t in BLOCKS:
            r = _stage3(made[tag], os.path.join(OUT, prefix + 'rung3_%s' % tag))
            if not r:
                print('%-10s %-9s stage 3 FAILED' % (rung, tag))
                continue
            m1, m2 = r.get('M1'), r.get('M2')
            print('%-10s %-9s %4d %6.3f +- %.3f %6.3f +- %.3f %11.6f %9.3f'
                  % (rung, tag, n, m1[0], m1[1], m2[0], m2[1], m2[2], f))
        env = dict(os.environ, HU_STAGE2_PREFIX=prefix, HU_ECLIPSE_METHOD='Method 1 & 2',
                   HU_UNION_HOST='gain125')
        p = subprocess.run([PY, os.path.join(REPO, 'tools', 'husillos2026', 'hu_union.py'),
                            'stage3'], cwd=REPO, env=env, capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if 'union_2witness' in line and 'Method' in line:
                s = line.strip().replace(u'\u00b1', ' +- ')
                print('%-10s %s' % ('  union', s[:110]))
        print()
    print('f is the fraction of a 1/r deflection that reaches stage 3 at that rung')
    print('(hu_absorption.py, verified by injection in hu_inject.py).  Divide L by f to')
    print('compare rungs; at `constant` the honest column is Method 1, elsewhere Method 2.')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(OUT, exist_ok=True)
    if cmd in ('stage2', 'all'):
        do_stage2()
        print()
    if cmd in ('report', 'all'):
        do_report()
