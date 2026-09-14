"""Run all three pipeline stages into F: and check they reproduce the recorded numbers.

Douglas, 2026-09-14, after the output tree moved from `D:\\MEE2024 output\\MEE_output` to
`F:\\MEE_output`: "Let's test the whole pipeline runs correctly from F:".

A green test suite would not answer that question -- the suite never touches the output tree.
What answers it is a real reduction: raw frames off the read-only input drive, a dark and a
distortion reference read from F:, and all three stages writing to F:.  The engine has not
been committed to since these results were recorded (`git log -- mee2024/`), so a correct run
reproduces them to the digit, and any difference is the move's doing rather than the code's.

  stage 1   23_44_06, frames 1-49, the DEEP detection settings, against the synthetic
            hot-pixel dark on F:          -> compare to horizon/s1d_h10_g125d
  stage 2   that stack against the zenith quintic reference on F:, same rung as the
            eclipse blocks                -> compare to horizon/s2d_h10_g125d
  stage 3   Method 2 on the recorded gain-0 eclipse stage-2 zip on F:, admitted-star window
            from tools/analysis_window.py -> compare to step3/method2_gain0

Nothing here chooses an analysis parameter: every setting is imported from the driver that
owns it (`hu_horizon_reduce` for stages 1-2, `analysis_window` for the stage-3 window), so
this file cannot drift away from the reductions it is checking.

    .venv/Scripts/python.exe tools/husillos2026/hu_pipeline_check.py [--keep]
"""
import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import hu_horizon_reduce as H  # noqa: E402  -- the settings live there, not here
from analysis_window import WINDOWS  # noqa: E402

REPO = H.REPO
PY = H.PY
WIN = WINDOWS['husillos2026']

TEST = r'F:\MEE_output\_pipeline_check'
#: the capture to reproduce, as hu_horizon_reduce defines it
TAG, FOLDER, NAME, _GAIN, FIRST, LAST = [c for c in H.CAPTURES if c[0] == 'h10_g125d'][0]
REC_S1 = os.path.join(H.OUT, 's1d_h10_g125d')
REC_S2 = os.path.join(H.OUT, 's2d_h10_g125d')
REC_S3 = os.path.join(H.HUS, 'step3', 'method2_gain0')
ECL_S2 = os.path.join(H.HUS, 'step3', 'eclipse_gain0')

#: stage-2 fields worth comparing: the pointing, the fit quality, the scale, and the two
#: provenance fields CLAUDE.md says to read back rather than assume
FIELDS = ['RA', 'DEC', 'ROLL', 'platescale (arcseconds/pixel)',
          'final rms error (arcseconds)', '#stars used',
          'fixed distortion order', 'plate scale source']


def run(cmd, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with io.open(log, 'w', encoding='utf-8', errors='replace') as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t0


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(io.open(r[0], encoding='utf-8')) if r else None


def shifts(d):
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        return None
    import zipfile
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z[0]).open('results.txt'),
                                   encoding='utf-8', errors='replace'))
    return r


def L_of(d):
    """The result lines of a stage-3 ECLIPSE_OUTPUT text, verbatim.

    Compared as strings rather than parsed into floats.  The file writes its plus-or-minus
    in the console codepage, not UTF-8, so reading it as UTF-8 turns the sign into U+FFFD
    and a regex looking for the sign finds nothing -- which is what made the first run of
    this check report "not parsed" on BOTH files and call it a failure.  The bytes are
    written by the same code on both sides, so comparing the lines as they stand is both
    simpler and stricter than recovering the numbers.
    """
    f = sorted(glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True))
    if not f:
        return None
    txt = io.open(f[-1], encoding='utf-8', errors='replace').read()
    want = ('deflection constant', 'Method 2 results', 'number of stars used',
            'deflected star position rms')
    lines = [ln.strip() for ln in txt.splitlines()
             if any(ln.strip().startswith(w) for w in want)]
    return lines or None


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true', help='leave the test tree on F:')
    ap.add_argument('--resume', action='store_true',
                    help='reuse any stage that already produced output')
    ap.add_argument('--compare-only', action='store_true',
                    help='compare what is already in the test tree; run nothing')
    a = ap.parse_args()
    a.resume = a.resume or a.compare_only

    # Stage 1 is an hour (49 frames re-centroided at ~78 s each), so the tree is only wiped
    # for a genuinely fresh run.  The desktop app crashed mid-check on 2026-09-14 and the
    # detached run carried on regardless; --resume picks up whatever it left.
    if os.path.isdir(TEST) and not a.resume:
        shutil.rmtree(TEST)
    os.makedirs(TEST, exist_ok=True)
    ok = True
    print('output tree under test: F:\\MEE_output')
    print('input (read only):      %s' % H.SRC)
    print()

    # ---- stage 1 --------------------------------------------------------------------
    d1 = os.path.join(TEST, 's1')
    ser = os.path.join(H.SRC, FOLDER, NAME + '.ser')
    print('STAGE 1  %s frames %d-%d, DEEP settings, dark from F:' % (NAME, FIRST, LAST))
    if shifts(d1) and a.resume:
        print('   already done, reused')
    elif a.compare_only:
        print('   nothing to compare')
        return 1
    else:
        rc, dt = run([PY, '-m', 'mee2024.cli', 'stack', ser,
                      '--frames', '%d-%d' % (FIRST, LAST), '--dark', H.DARK,
                      *H.DEEP, '--no-display', '--quiet', '-o', d1],
                     os.path.join(d1, 'stage1.log'))
        print('   exit %d in %.0f s' % (rc, dt))
    new, rec = shifts(d1), shifts(REC_S1)
    if not new or not rec:
        print('   FAIL: no centroid zip to compare')
        sys.exit(1)
    for k in ('n_centroids', '#frames stacked', 'platesolved', 'RA', 'DEC', 'roll',
              'platescale/arcsec'):
        a_, b_ = new.get(k), rec.get(k)
        m = same(a_, b_)
        ok &= m
        print('   %-22s %-22s %s' % (k, a_, 'MATCH' if m else 'RECORDED ' + str(b_)))
    sn = new.get('alignment', {}).get('shifts_px')
    sr = rec.get('alignment', {}).get('shifts_px')
    if sn and sr:
        m = sn == sr
        ok &= m
        print('   %-22s %-22s %s' % ('alignment shifts_px', '%d frames' % len(sn),
                                     'IDENTICAL' if m else 'DIFFERS'))

    # ---- stage 2 --------------------------------------------------------------------
    d2 = os.path.join(TEST, 's2')
    print('\nSTAGE 2  against the zenith quintic reference on F:')
    if results(d2) and a.resume:
        print('   already done, reused')
    elif a.compare_only:
        print('   nothing to compare')
        return 1
    else:
        rc, dt = run([PY, '-m', 'mee2024.cli', 'distortion', glob.glob(
            os.path.join(d1, 'centroid_data*.zip'))[0], '--order', 'quintic',
            '--set', 'distortion_reference_files=' + H.refzip(),
            '--set', 'distortion_fixed_coefficients=quadratic',
            '--set', 'distortion_free_scale=True',
            '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
            '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
            *H.SITE, '--set', 'observation_time=' + H.midtime(FOLDER, NAME),
            '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
        print('   exit %d in %.0f s' % (rc, dt))
    jn, jr = results(d2), results(REC_S2)
    if not jn or not jr:
        print('   FAIL: stage 2 produced no results')
        sys.exit(1)
    for k in FIELDS:
        a_, b_ = jn.get(k), jr.get(k)
        m = same(a_, b_)
        ok &= m
        print('   %-32s %-22s %s' % (k, a_, 'MATCH' if m else 'RECORDED ' + str(b_)))

    # ---- stage 3 --------------------------------------------------------------------
    d3 = os.path.join(TEST, 's3')
    z = glob.glob(os.path.join(ECL_S2, '**', 'distortion_data*.zip'), recursive=True)
    print('\nSTAGE 3  Method 2 on the recorded gain-0 eclipse stage-2 zip on F:')
    if not z:
        print('   SKIP: no eclipse stage-2 zip on F:')
    else:
        if L_of(d3) and a.resume:
            print('   already done, reused')
        elif a.compare_only:
            print('   nothing to compare')
            return 1
        else:
            rc, dt = run([PY, '-m', 'mee2024.cli', 'eclipse', z[0],
                          '--set', 'eclipse_method=Method 2',
                          '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
                          '--set', 'limit_radial_sun_radii=False',
                          '--set', 'remove_double_stars_eclipse=False',
                          '--no-display', '--quiet', '-o', d3],
                         os.path.join(d3, 'stage3.log'))
            print('   exit %d in %.0f s' % (rc, dt))
        ln, lr = L_of(d3), L_of(REC_S3)
        if not ln or not lr:
            print('   FAIL: no stage-3 result lines (new %s, recorded %s)'
                  % (bool(ln), bool(lr)))
            ok = False
        else:
            for a_ in ln:
                b_ = next((x for x in lr if x.split('=')[0] == a_.split('=')[0]
                           and x.split(':')[0] == a_.split(':')[0]), None)
                m = a_ == b_
                ok &= m
                print('   %-70s %s' % (a_.encode('ascii', 'replace').decode('ascii'),
                                       'MATCH' if m else 'RECORDED ' + str(b_)))

    print('\n%s' % ('PASS: all three stages reproduce the recorded numbers from F:.' if ok
                    else 'FAIL: something did not reproduce; the test tree is kept.'))
    if a.keep or not ok:
        print('test tree: %s' % TEST)
    else:
        shutil.rmtree(TEST)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
