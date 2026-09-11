"""Does the plate scale drift during totality? Split BOTH eclipse blocks in half and refit.

Douglas, 2026-09-11.  The three fields that exist read, on one rung and directly comparable:

    eclipse gain 125   18:29:20   2.2027459 +- 21.0 ppm       0 ppm
    eclipse gain 0     18:29:59   2.2029009 +- 19.0 ppm    + 70.3 ppm
    CalibS             18:30:31   2.2029895 +- 25.2 ppm    +110.6 ppm

Monotonic in time, in the direction a cooling tube gives (a cold tube reads a LARGER scale;
totality removes the heating, so the scale comes back up -- docs/STEP3_2026.md, "The plate
scale across totality: Bruns falls, Station 2 rises").  But THREE VARIABLES MOVE TOGETHER
across those three fields -- time, gain (125 against 0) and pointing (CalibS is 10.17 deg from
the Sun) -- and the only pair at fixed gain is 1.3 sigma.  So the trend is suggestive and not
measured.

SPLITTING EACH BLOCK IN HALF REMOVES ALL THREE CONFOUNDS AT ONCE: the two halves of one block
share a gain, share a pointing, and differ only in time.  Two blocks give two independent
estimates of the same drift rate, which is the check a single split could not provide.
Station 2 did exactly this for the same reason (`tools/matrix_station2/s2_left_halves.py`).

The prediction to beat.  The full-span rate is +1.56 ppm/s.  Over a ~20 s half-block separation
that is ~31 ppm between halves, which both blocks should show, at the same rate, independently.
A null result -- halves agreeing inside their errors -- says the three-field trend is pointing
or gain, not time, and that CalibS' scale can be imported at face value.

Stage 1 settings are `hu_eclipse_stars.S1` verbatim (cell 2's eclipse convention) with the same
`husillos_synthetic_dark_all.fit`, and stage 2 is the same rung as the three fields above --
quadratic, zenith reference, free scale.  Anything else and the halves would not be comparable
to the wholes.

    .venv/Scripts/python.exe tools/husillos2026/hu_halves.py [stage1|stage2|report]
"""
import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')

G = r'G:\Joe Izen Spain 2026\2026-08-12'                 # READ ONLY
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'halves')
DARK = os.path.join(HUS, 'hotpixels', 'husillos_synthetic_dark_all.fit')
REF = os.path.join(HUS, 'step3', 'ref')

SUN = os.path.join(G, 'SunJoe_20260812_182845', '20_28_45.ser')
SN2 = os.path.join(G, 'Sn2_Joe_20260812_182942', '20_29_43.ser')

#: (tag, container, first, last, gain).  The record blocks are Sun 46-171 and Sn2 2-102
#: (read back from their own stage-1 `source_files`), split as evenly as the counts allow.
HALVES = [
    ('g125_A', SUN, 46, 108, 125),
    ('g125_B', SUN, 109, 171, 125),
    ('g0_A', SN2, 2, 52, 0),
    ('g0_B', SN2, 53, 102, 0),
]

#: cell 2's eclipse stage 1, identical to hu_eclipse_stars.S1 -- imported would be better, but
#: that module runs work at import time, so it is repeated here and pinned by a test-free
#: assertion: any edit must be made in both places or the halves stop being comparable.
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
      '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=10.0',
      '--set', 'coronal_pedestal_adu=2000.0']

SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0',
        '--set', 'observation_temp=25.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_humidity=0.35']


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


def midtime(container, first, last):
    """Mid-time of the frames stacked.

    Prefers the SER's own per-frame timestamp trailer.  The SUN capture does not have one that
    can be used: its trailer is present and the right size (1440 bytes for 180 frames) but was
    NEVER WRITTEN -- `ser_track.read_timestamps` returns None for exactly this, "the space is
    there and unwritten (an aborted capture)", which is the same capture whose last frames
    SharpCap never delivered.  Sn2's trailer is written and is used.

    The fallback is the capture's own CameraSettings: StartCapture plus frame index over
    ActualFrameRate.  Both are read from the file rather than assumed -- the two captures run
    at 3.1705 and 3.1704 fps, not at the nominal rate implied by the 315 ms exposure.
    """
    import datetime
    import re
    from ser_track import read_header, read_timestamps
    H = read_header(container)
    ts = read_timestamps(container, H)
    mid = (first + last) // 2
    if ts:
        return ts[mid]
    cfg = re.sub(r'\.ser$', '.CameraSettings.txt', container)
    kv = {}
    for line in open(cfg, encoding='utf-8', errors='replace'):
        if '=' in line:
            k, v = line.split('=', 1)
            kv[k.strip()] = v.strip()
    t0 = datetime.datetime.strptime(kv['StartCapture'][:23], '%Y-%m-%dT%H:%M:%S.%f')
    fps = float(kv['ActualFrameRate'].replace('fps', ''))
    return t0 + datetime.timedelta(seconds=mid / fps)


def do_stage1():
    os.makedirs(OUT, exist_ok=True)
    for tag, cont, a, b, gain in HALVES:
        d = os.path.join(OUT, 's1_' + tag)
        if czip(d):
            print('%-8s already stacked' % tag, flush=True)
            continue
        print('%-8s stacking frames %d-%d of %s' % (tag, a, b, os.path.basename(cont)),
              flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', cont, '--frames', '%d-%d' % (a, b),
             '--dark', DARK, *S1, '--no-display', '--quiet', '-o', d],
            os.path.join(d, 'stage1.log'))


def do_stage2():
    ref = refzip()
    for tag, cont, a, b, gain in HALVES:
        z = czip(os.path.join(OUT, 's1_' + tag))
        if not z:
            print('%-8s no stage 1' % tag)
            continue
        d = os.path.join(OUT, 's2_' + tag)
        if not results(d):
            t = midtime(cont, a, b).strftime('%H:%M:%S')
            run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', 'quintic',
                 '--set', 'distortion_reference_files=' + ref,
                 '--set', 'distortion_fixed_coefficients=quadratic',
                 '--set', 'distortion_free_scale=True',
                 '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                 '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
                 *SITE, '--set', 'observation_time=' + t,
                 '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
    do_report()


def do_report():
    import numpy as np
    print('%-8s %5s %-10s %7s %9s %12s %8s'
          % ('half', 'gain', 'mid UTC', 'stars', 'rms(")', 'ps("/px)', '+-ppm'))
    got = {}
    for tag, cont, a, b, gain in HALVES:
        j = results(os.path.join(OUT, 's2_' + tag))
        t = midtime(cont, a, b)
        if not j:
            print('%-8s %5d %-10s   FAILED' % (tag, gain, t.strftime('%H:%M:%S')))
            continue
        got[tag] = (t, j['platescale (arcseconds/pixel)'],
                    j['platescale_relative_uncertainty'] * 1e6)
        print('%-8s %5d %-10s %7d %9.4f %12.7f %8.1f'
              % (tag, gain, t.strftime('%H:%M:%S'), j['#stars used'],
                 j['final rms error (arcseconds)'], j['platescale (arcseconds/pixel)'],
                 j['platescale_relative_uncertainty'] * 1e6))
    print()
    print('THE TEST -- each block against itself, fixed gain and fixed pointing:')
    for lo, hi, gain in (('g125_A', 'g125_B', 125), ('g0_A', 'g0_B', 0)):
        if lo not in got or hi not in got:
            continue
        (t1, p1, u1), (t2, p2, u2) = got[lo], got[hi]
        dt = (t2 - t1).total_seconds()
        d_ppm = (p2 - p1) / p1 * 1e6
        sig = float(np.hypot(u1, u2))
        print('   gain %-3d  %+6.1f ppm over %4.1f s  =  %+.3f ppm/s   (%.1f sigma on %.1f ppm)'
              % (gain, d_ppm, dt, d_ppm / dt, abs(d_ppm) / sig, sig))
    print()
    print('   for comparison: the three-field trend is +1.558 ppm/s,')
    print('                   Station 2 across totality +0.157, Bruns -0.358 ppm/s')


def do_diff():
    """The differential scale between two halves, measured on the stars they SHARE.

    This is the sensitive version of the test.  A difference between two fits of the SAME
    field is far better determined than either fit's absolute value: the catalogue positions,
    the frozen zenith cubic-and-above, the refraction model and the pointing are identical
    between halves and cancel exactly, leaving only centroid noise.  The absolute comparison
    in `report` carries ~+-30 ppm per half against a predicted ~31 ppm signal and is
    underpowered by construction; this one need not be.

    THE ESTIMATOR MUST BE A FULL SIMILARITY, and the first version of this function got that
    wrong.  Each half is stacked and aligned against ITS OWN first frame, so the two stacks'
    pixel grids differ by an arbitrary whole-pixel translation (and, in principle, a small
    rotation).  Regressing radial displacement on radius with no intercept, as the first
    version did, feeds that translation straight into the slope: it returned -618 +- 500 ppm
    on the gain-0 pair, which is not a measurement of anything.  So four parameters are fitted
    together --

        dx = tx - theta * ry + s * rx
        dy = ty + theta * rx + s * ry

    -- and `s`, the isotropic scale term, is the answer.  tx/ty absorb the stack offset and
    theta any roll between them.
    """
    import numpy as np
    import pandas as pd
    import zipfile

    def matched(tag):
        d = os.path.join(OUT, 's2_' + tag)
        z = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
        if not z:
            return None
        zf = zipfile.ZipFile(z[0])
        n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
        t = pd.read_csv(zf.open(n), dtype={'ID': str})
        t.columns = [c.strip() for c in t.columns]
        t['ID'] = t['ID'].astype(str).str.strip()
        return t.set_index('ID')

    print()
    print('DIFFERENTIAL SCALE on shared stars (similarity fit: translation + rotation + scale)')
    print('%-10s %7s %8s %16s %18s' % ('pair', 'shared', 'dt (s)', 'dscale (ppm)', 'ppm/s'))
    for lo, hi, gain in (('g125_A', 'g125_B', 125), ('g0_A', 'g0_B', 0)):
        A, B = matched(lo), matched(hi)
        if A is None or B is None:
            print('%-10s   stage 2 missing' % (lo + '/' + hi))
            continue
        both = A.index.intersection(B.index)
        a, b = A.loc[both], B.loc[both]
        n = len(both)
        if n < 8:
            print('%-10s %7d   too few shared stars to fit' % ('gain %d' % gain, n))
            continue
        cx, cy = a['px'].mean(), a['py'].mean()
        rx, ry = a['px'].values - cx, a['py'].values - cy
        dx = b['px'].values - a['px'].values
        dy = b['py'].values - a['py'].values
        Z, O = np.zeros(n), np.ones(n)
        M = np.vstack([np.column_stack([O, Z, -ry, rx]),
                       np.column_stack([Z, O, rx, ry])])
        d = np.concatenate([dx, dy])
        # A SIGMA CLIP IS NOT OPTIONAL HERE. The gain-0 pair first fitted with a 4.8 px rms
        # residual against the gain-125 pair's 0.29 -- a handful of bad centroids in the
        # shallower halves, and on 27 stars they dominate a 4-parameter fit completely
        # (-619 +- 359 ppm, which is not a measurement). Three passes at 3 sigma, with the
        # count of survivors reported so a heavy cut cannot pass unnoticed.
        keep = np.ones(2 * n, bool)
        for _ in range(3):
            c, *_ = np.linalg.lstsq(M[keep], d[keep], rcond=None)
            r = d - M @ c
            sd = float(np.std(r[keep]))
            new_keep = np.abs(r) < 3 * sd
            if new_keep.sum() == keep.sum() or new_keep.sum() < 12:
                break
            keep = new_keep
        M, d, n_clip = M[keep], d[keep], int((~keep).sum())
        c, *_ = np.linalg.lstsq(M, d, rcond=None)
        resid = d - M @ c
        dof = max(len(d) - 4, 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.inv(M.T @ M)
        scale_ppm = c[3] * 1e6
        se_ppm = float(np.sqrt(cov[3, 3])) * 1e6
        dd = {t: (cc, x, y) for t, cc, x, y, _ in HALVES}
        dt = (midtime(*dd[hi]) - midtime(*dd[lo])).total_seconds()
        print('%-10s %7d %8.1f %7.1f +- %-5.1f %8.3f +- %.3f'
              % ('gain %d' % gain, n, dt, scale_ppm, se_ppm, scale_ppm / dt, se_ppm / dt))
        print('%-10s          residual %.3f px rms, %d of %d components clipped, '
              'translation (%+.2f, %+.2f) px, rotation %+.1f arcsec'
              % ('', float(np.std(resid)), n_clip, 2 * n, c[0], c[1],
                 np.degrees(c[2]) * 3600))
    print()
    print('   predicted by the three-field trend: +1.558 ppm/s')
    print('   Station 2 across totality +0.157 ppm/s, Bruns -0.358 ppm/s')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    os.makedirs(OUT, exist_ok=True)
    {'stage1': do_stage1, 'stage2': do_stage2, 'report': do_report,
     'diff': do_diff}[cmd]()
