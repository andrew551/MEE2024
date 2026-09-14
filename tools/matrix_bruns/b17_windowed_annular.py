"""Cell 1 in the Station 1 convention: Bruns' own procedure with windowed + annular centroids.

Douglas, 2026-09-05: what would cell 1's Method 1 / Method 2 table look like under the same
centroiding as the Mexico 2024 reduction (the fixed Gaussian window, annular background --
from today also the program's default)?

The record's Bruns reduction (`b17_bruns_method.py`) is footprint moments + Gaussian
background, his own convention, chosen because it reproduces Bruns 2018 end to end. Its
calibration side -- L and R8 stacked from raw, then a cubic with quadratic-and-below fixed from
the fifteen night fields -- is rebuilt here with the two estimator flags flipped, E2 is
re-stacked likewise, and the record tool is then run with B17M_ESTIMATOR=windowed against this
tree so that the 0.62 s master, the constant-only stage 2, the seven-star link and the star
table are all built by exactly the record's procedure in the other convention. The fifteen
night references keep their moment-convention cubic terms (only those are imported); the
question asked is about the eclipse and L/R side, and that is what changes.

An earlier windowed reduction of this field exists (matrix_bruns2017, the S0 tree of
2026-08-29) and must not be used for this: its union was superseded and its per-star residual
is 0.28"/axis against 0.10 for every other reduction of these frames.

Writes matrix_bruns2017_windowed_annular/{L,R8,E2}/ and matrix_bruns2017_brunsmethod_windowed/.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
M = r"F:/MEE_output"
MAIN = os.path.join(M, 'matrix_bruns2017')                    # preprocessed EA/EB/E2 frames
RAWCAL = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images"
WSIG = os.environ.get('B17M_WINDOW_SIGMA', '2.0')      # the window, px; see b17_bruns_method.py
SUF = '' if WSIG == '2.0' else '_w' + WSIG.replace('.', 'p')
CONV = os.path.join(M, 'matrix_bruns2017_windowed_annular' + SUF)
OUT = os.path.join(M, 'matrix_bruns2017_brunsmethod_windowed' + SUF)
NIGHTREFS = json.load(open(glob.glob(os.path.join(M, 'bruns2017_lr', 'L', 'stage2', 'DISTORTION_OUTPUT*', 'distortion',
                                                  'distortion_results.txt'))[0], encoding='utf-8'))['fixed distortion reference files'].split(';')
SITE = ['--set', 'observation_lat=42 44 11 N', '--set', 'observation_long=106 19 05 W',
        '--set', 'observation_height=2400', '--set', 'observation_temp=13.0',
        '--set', 'observation_pressure=770.0', '--set', 'observation_humidity=0.4',
        '--set', 'observation_wavelength=0.625']
# the record tool's stage-1 set with the estimator flags flipped
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_refine_window=True',
      '--set', 'centroid_window_sigma=' + WSIG, '--set', 'background_subtraction_mode=annular']


def run(cmd, log, env=None):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def stack(name, frames):
    d = os.path.join(CONV, name); os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  stacking %s from %d frames, windowed + annular' % (name, len(frames)), flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', *frames, *S1, '--no-scan', '--no-display', '--quiet', '-o', d],
            os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  %s: STAGE 1 FAILED' % name, flush=True); return None
    print('  %s: %d centroids' % (name, json.load(__import__('zipfile').ZipFile(z[0]).open('results.txt'))['n_centroids']), flush=True)
    return z[0]


def stage2(name, cz, refs, fixed, obstime, tol='0.5'):
    d2 = os.path.join(CONV, name, 'stage2'); os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'cubic', '--date-from-header',
             '--fix-distortion', *refs, '--set', 'distortion_fixed_coefficients=%s' % fixed,
             '--set', 'distortion_fit_tol=%s' % tol, '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=True',
             '--set', 'enable_corrections_ref=True', *SITE, '--set', 'observation_time=' + obstime,
             '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
    r = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not r:
        print('  %s: stage 2 FAILED' % name, flush=True); return None
    j = json.load(open(r[0], encoding='utf-8'))
    print('  %s: %d matched, rms %.4f", plate scale %.7f' % (name, j['#stars used'], j['final rms error (arcseconds)'],
                                                             j['platescale (arcseconds/pixel)']), flush=True)
    return r[0]


print('=== the calibration side, windowed + annular (the record: %d night references keep their cubic terms) ===' % len(NIGHTREFS), flush=True)
lz = stack('L', sorted(glob.glob(os.path.join(RAWCAL, 'left', 'L_[1-7]_*.fit'))))
rz = stack('R8', sorted(glob.glob(os.path.join(RAWCAL, 'right', 'R_[1-8]_*.fit'))))
refL = stage2('L', lz, NIGHTREFS, 'quadratic', '17:44') if lz else None
refR = stage2('R8', rz, NIGHTREFS, 'quadratic', '17:44') if rz else None
ez = stack('E2', sorted(glob.glob(os.path.join(MAIN, 'E2', 'preprocessed', '*.fits'))))
if not (refL and refR and ez):
    raise SystemExit('the convention tree is incomplete; see the logs under ' + CONV)

print('\n=== Bruns\' procedure in this convention (b17_bruns_method.py, B17M_ESTIMATOR=windowed) ===', flush=True)
env = dict(os.environ, B17M_ESTIMATOR='windowed', B17M_CONV=CONV, B17M_OUT=OUT, B17M_WINDOW_SIGMA=WSIG)
os.makedirs(OUT, exist_ok=True)
rc = run([PY, os.path.join(REPO, 'tools', 'matrix_bruns', 'b17_bruns_method.py')], os.path.join(OUT, 'bruns_method.log'), env=env)
print('  rc %d; log: %s' % (rc, os.path.join(OUT, 'bruns_method.log')), flush=True)
print(''.join(open(os.path.join(OUT, 'bruns_method.log'), encoding='utf-8', errors='ignore').readlines()[-25:]))

# ---------------------------------------------------------------- Method 1 and Method 2, both conventions
PS, NX, NY = 2.0868004, 3296, 2472; R_SUN_AS = 948.7; SUNPX, SUNPY = 1645.0, 1741.0


def fit_both(t):
    rx, ry = (t.px.values-SUNPX)*PS, (t.py.values-SUNPY)*PS; R = np.hypot(rx, ry)
    m = len(t); Z = np.zeros(m); xs, ys = (t.px.values-NX/2)*PS, (t.py.values-NY/2)*PS
    b = np.concatenate([t.dx.values, t.dy.values]); out = []
    for ws in (False, True):
        cx = [np.ones(m), Z, -ys] + ([xs] if ws else []) + [rx/R*R_SUN_AS/R]
        cy = [Z, np.ones(m), xs] + ([ys] if ws else []) + [ry/R*R_SUN_AS/R]
        A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
        c, *_ = np.linalg.lstsq(A, b, rcond=None); r = b - A@c
        cov = (r@r)/(len(b)-len(c))*np.linalg.inv(A.T@A)
        out.append(dict(L=c[-1], eL=np.sqrt(cov[-1, -1]), S=(1e6*c[3] if ws else np.nan), eS=(1e6*np.sqrt(cov[3, 3]) if ws else np.nan),
                        rms=np.sqrt((r@r)/len(b))))
    return out


print('\n=== Method 1 (imported scale) and Method 2 (scale free) on the two conventions\' star tables ===')
for label, tree in (('moments + Gaussian (record)', os.path.join(M, 'matrix_bruns2017_brunsmethod')),
                    ('windowed + annular', OUT)):
    for name in ('bruns_method_star_table.csv',):      # the 7-star-link table both trees produce by default
        p = os.path.join(tree, name)
        if not os.path.exists(p):
            print('  %-28s %-44s missing' % (label, name)); continue
        t = pd.read_csv(p); m1, m2 = fit_both(t)
        print('  %-28s %-44s %2d stars  M1 %.3f +- %.3f   M2 %.3f +- %.3f   S %+5.1f +- %4.1f ppm   rms/axis %.3f"'
              % (label, name.replace('bruns_method_star_table_', '').replace('.csv', ''), len(t), m1['L'], m1['eL'], m2['L'], m2['eL'], m2['S'], m2['eS'], m1['rms']))
print('->', OUT)
