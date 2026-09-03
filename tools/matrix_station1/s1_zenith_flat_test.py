"""Cell 2: does flat-fielding the zenith fields actually move the distortion fit?

Douglas, 2026-09-03: "'Worth doing for tidiness, but it will not move anything': have you
already tested this on a few fields to verify how much it moves?"

No. That claim was a prediction from the PSF-injection measurement in `s1_darks_flats.py` --
the flat moves a centroid 2.4-3.3 mas, and a quintic in position absorbs almost none of it
because it is nearly all pixel-scale response variation rather than vignetting -- and a
prediction is not a test. This is the test.

It matters because the two sides of cell 2's transfer are not treated alike: the 2024 eclipse
stacks were dark- and flat-calibrated and the seventeen zenith stacks were not, so any part of
the flat that a quintic DOES absorb enters the frozen model on one side only.

Three of the four zenith blocks whose raw frames Douglas found are re-stacked from raw with
`--flat` and nothing else changed, against the no-flat versions already in
`zenith_raw_ab/windowed_annular/`. Doing it from raw rather than dividing the stacked image is
the honest route: the flat multiplies each frame before alignment, and the stacked image has
already had a background removed, so dividing it afterwards is not the same operation.

No dark, deliberately -- the 2024 zenith reductions used none, and the question here is the
flat alone. The flat is the ZENITH SESSION's own (06:28 UTC, same night and focus), not the
post-eclipse one: an earlier run of this tool used the post-eclipse flat by mistake and is
superseded by this one.

Reported: stars, fit rms, plate scale, and the thing that actually transfers -- the fitted
distortion evaluated over the frame, as an rms displacement difference in mas between the
flat and no-flat models. That last number is the mismatch the eclipse field would inherit.
"""
import glob, json, os, subprocess, zipfile
import numpy as np

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAW = r"I:/Mexico 2024/Station 1 Zenith"
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
ZFLAT = r"I:/Mexico 2024/Station 1 Zenith/Flats/2024-04-08_06_28_18Z"
AB = r"D:/MEE2024 output/MEE_output/station1_record/zenith_raw_ab/windowed_annular"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/zenith_flat_test2"
NX, NY = 9576, 6388
BLOCKS = [('f1', '2024-04-08_05_32_53Z', '05:33:30'),
          ('f2', '2024-04-08_05_35_48Z', '05:36:25'),
          ('f3', '2024-04-08_05_38_32Z', '05:39:10')]
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
      '--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular']
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=760.0',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def build(tag, block, tmid):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        frames = [f for f in sorted(glob.glob(os.path.join(RAW, block, '*.FIT')))
                  if not f.endswith('_0000.FIT')]
        print('  %s: stacking %d frames WITH the flat...' % (tag, len(frames)), flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', *frames,
             # the ZENITH SESSION's own flat (06:28 UTC, same night, same focus), not the
             # post-eclipse one on G: which was shot at 19:21 after the daytime refocus.
             # The first run of this tool used the G: flat by mistake; the two agree to
             # 0.2 % in vignetting profile (s1_flat_stability.py) but the matched one is
             # the only defensible choice for these fields.
             '--flat', os.path.join(ZFLAT, '*.FIT'),
             *S1, '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        return None
    d2 = os.path.join(d, 'stage2_free')
    os.makedirs(d2, exist_ok=True)
    hit = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not hit:
        run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'quintic',
             '--set', 'distortion_fit_tol=0.3', '--set', 'max_star_mag_dist=15',
             '--set', 'rough_match_threshhold=36', *SITE,
             '--set', 'observation_time=' + tmid, '--no-display', '--quiet', '-o', d2],
            os.path.join(d2, 'stage2.log'))
        hit = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    return hit[0] if hit else None


def coeffs(resfile):
    j = json.load(open(resfile, encoding='utf-8'))
    kx = ky = None
    for k, v in j.items():
        if isinstance(v, dict) and 'x^5' in v:
            if kx is None:
                kx = v
            elif ky is None:
                ky = v
    return j, kx, ky


def model_field(kx, ky, ps):
    """Evaluate the fitted distortion over a grid, in PIXELS (the basis is normalised by w = NX/2; the coefficients carry pixels)."""
    gy, gx = np.mgrid[0:NY:64, 0:NX:64]
    x = (gx - NX/2)/(NX/2); y = (gy - NY/2)/(NX/2)
    def ev(k):
        t = np.zeros_like(x)
        for term, c in k.items():
            if term == '1':
                t += c; continue
            p = 0; q = 0
            for part in term.split('*'):
                part = part.strip()
                if part.startswith('x'):
                    p = int(part[2:]) if '^' in part else 1
                elif part.startswith('y'):
                    q = int(part[2:]) if '^' in part else 1
            t += c*(x**p)*(y**q)
        return t
    return ev(kx), ev(ky)


print('=== flat vs no flat, three zenith fields, everything else identical ===\n')
print('%-4s %-26s %6s %9s %13s' % ('', 'variant', 'stars', 'rms (")', 'plate scale'))
rows = []
for tag, block, tmid in BLOCKS:
    new = build(tag, block, tmid)
    old = glob.glob(os.path.join(AB, tag, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
    if not new or not old:
        print('%-4s could not build both variants' % tag); continue
    jn, kxn, kyn = coeffs(new)
    jo, kxo, kyo = coeffs(old[0])
    for lab, j in (('no flat (the record)', jo), ('WITH the flat', jn)):
        print('%-4s %-26s %6d %9.4f %13.7f'
              % (tag if lab.startswith('no') else '', lab, j['#stars used'],
                 j['final rms error (arcseconds)'], j['platescale (arcseconds/pixel)']))
    dps = 1e6*(jn['platescale (arcseconds/pixel)']-jo['platescale (arcseconds/pixel)'])/jo['platescale (arcseconds/pixel)']
    row = dict(tag=tag, dps=dps, drms=jn['final rms error (arcseconds)']-jo['final rms error (arcseconds)'])
    if kxn and kxo:
        ax, ay = model_field(kxn, kyn, 1.0)
        bx, by = model_field(kxo, kyo, 1.0)
        # The basis is (x/w)^p (y/w)^q with w = max(img_shape)/2 = NX/2, which model_field
        # already uses, and the SUM is in PIXELS -- distortion_polynomial labels its own
        # coefficient plot "distortion coefficient (pixels)". So the only conversion to mas
        # is the plate scale. An earlier version multiplied by w as well and reported the
        # model difference as 97 arcsec instead of 20 mas.
        diff = np.hypot(ax-bx, ay-by)*1.84847*1000
        # remove the mean offset and a tilt: those are absorbed by the constant-only science fit
        gy, gx = np.mgrid[0:NY:64, 0:NX:64]
        A = np.column_stack([np.ones(gx.size), ((gx-NX/2)/(NX/2)).ravel(), ((gy-NY/2)/(NX/2)).ravel()])
        res = []
        for f in ((ax-bx).ravel(), (ay-by).ravel()):
            c, *_ = np.linalg.lstsq(A, f, rcond=None)
            res.append((f - A@c))
        resid = np.hypot(res[0], res[1])*1.84847*1000
        row['dmodel'] = float(np.sqrt((diff**2).mean()))
        row['dmodel_res'] = float(np.sqrt((resid**2).mean()))
        print('%-4s %-26s scale %+.1f ppm | fitted model differs by %.1f mas rms over the frame,'
              ' %.1f mas after the offset and tilt a constant-only science fit removes'
              % ('', '-> difference', dps, row['dmodel'], row['dmodel_res']))
        print('%-4s %-26s (different star sets, %d against %d, so some of this is sampling'
              ' noise rather than the flat)' % ('', '', jn['#stars used'], jo['#stars used']))
    else:
        print('%-4s %-26s scale %+.1f ppm (coefficients not both readable)' % ('', '-> difference', dps))
    rows.append(row)
    print()

if rows:
    print('=== summary over %d fields ===' % len(rows))
    print('  plate scale, flat minus no flat: %s ppm (mean %+.1f)'
          % (', '.join('%+.1f' % r['dps'] for r in rows), np.mean([r['dps'] for r in rows])))
    print('  fit rms change: %s "' % ', '.join('%+.4f' % r['drms'] for r in rows))
    if 'dmodel_res' in rows[0]:
        m = np.mean([r['dmodel_res'] for r in rows])
        print('  fitted model difference after offset+tilt: %s mas (mean %.1f)'
              % (', '.join('%.1f' % r['dmodel_res'] for r in rows), m))
        print('\n  That last figure is the mismatch the eclipse field inherits, because the 2024')
        print('  eclipse stacks WERE flat-fielded and these zenith stacks were not. Against a')
        print('  per-star fit residual of ~120 mas on the zenith fields and ~140 mas on the')
        print('  eclipse union, and a Method-2 sigma_L of 108 mas.')
print('\n->', OUT)
