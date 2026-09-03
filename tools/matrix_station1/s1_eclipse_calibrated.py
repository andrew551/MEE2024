"""Cell 2: re-stack the eclipse tier WITH the dark and flat the 2024 reduction actually used.

Found while inventorying the folders on 2026-09-03: the 2024 eclipse reduction of record
(`eclipse fields/CENTROID_OUTPUT20240416232626`) kept a `DARK_STACK` (median 502 ADU, max
21483) and a `FLAT_STACK` (median 22417) beside its results, both matching the masters built
from the `G:` calibration sets exactly. So the 2024 eclipse WAS dark- and flat-calibrated,
while none of the seventeen zenith reductions was.

That matters twice over. The convention re-stacks of `s1_eclipse_convention.py` were run
without calibration, which is why 11 % of their detections land on a dark-flagged hot pixel
(`s1_hotpixel_risk.py`) and the 2024 result does not; and it means the absolute L from those
re-stacks is not strictly like-for-like with the 2024 number, because two things differ
rather than one.

This closes that. The same 123 frames of the 0.4 s tier, the same windowed + annular
convention, the same Sun masking -- with `--dark` and `--flat` added. Reported against the
uncalibrated windowed stack, so the calibration axis is isolated:

  * how many hot pixels the dark excludes, and how the detection count changes;
  * the fitted L under Method 1 and Method 2, against each available reference;
  * the per-star fit residual, which is where a flat would show up if it mattered
    (it should not: `s1_darks_flats.py` puts flat-fielding at 2.4-3.3 mas).

References: the 2024 seventeen-field moments reference, and the seventeen-field WINDOWED
reference from `s1_zenith_recentroid.py` once it exists.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
TIER = os.path.join(G, 'CapObj', '2024-04-08_18_12_30Z')
Z24 = r"D:/MEE2024 output/Station 1/zenith calibrations"
RECEN = r"D:/MEE2024 output/MEE_output/station1_record/zenith_recentroid"
UNCAL = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_convention/windowed"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_calibrated"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0

S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True']
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_time=18:12', '--set', 'observation_long=105 16 22.1 W',
       '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=15.0',
       '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
       '--set', 'observation_height=2400.0']

REFS = {'A_2024_17field_moments': sorted(glob.glob(os.path.join(Z24, '*', '**', 'distortion_results.txt'), recursive=True)),
        'F_17field_windowed': sorted(glob.glob(os.path.join(RECEN, '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))}
os.makedirs(OUT, exist_ok=True)
t = Time(T_MID, scale='utc'); sun = get_sun(t)
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stack():
    z = glob.glob(os.path.join(OUT, 'centroid_data*.zip'))
    if z:
        return z[0]
    print('  stacking 123 frames with dark + flat...', flush=True)
    # globs rather than 203 explicit paths: resolve_input_files expands them, and Windows
    # has a command-line length limit that 203 full paths would come close to
    run([PY, '-m', 'mee2024.cli', 'stack', os.path.join(TIER, '*.FIT'),
         # the sets are <name>/CapObj/<timestamp>/*.FIT, one level deeper than they look
         '--dark', os.path.join(G, 'dark-400ms', 'CapObj', '*', '*.FIT'),
         '--flat', os.path.join(G, 'flat', 'CapObj', '2024-04-08*', '*.FIT'),
         *S1, '--no-scan', '--no-display', '--quiet', '-o', OUT],
        os.path.join(OUT, 'stage1.log'))
    z = glob.glob(os.path.join(OUT, 'centroid_data*.zip'))
    return z[0] if z else None


def fit(cz, refname, refs, root):
    d = os.path.join(root, refname)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    if not hit:
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic',
             '--fix-distortion', *refs, '--set', 'distortion_fixed_coefficients=constant',
             '--set', 'distortion_fit_tol=20.0', '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=100', *MET, '--no-display', '--quiet', '-o', d],
            os.path.join(d, 'stage2.log'))
        hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    return hit[0] if hit else None


def method_fits(zp, common=None):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    dd = d[d['flag_is_outlier'] == False]
    if common is not None:
        dd = dd[dd.ID.isin(common)]
    ra0, de0 = dd['RA(catalog)'].mean(), dd['DEC(catalog)'].mean()
    X = (dd['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = dd['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, dd.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, dd.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(dd['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), dd['DEC(obs)'].values-de0, np.ones(len(dd))]
    cxm = np.c_[X, Y, np.ones(len(dd))]
    DX, DY = (ox@ax-cxm@ax)*PS, (ox@ay-cxm@ay)*PS
    px, py, mag = dd.px.values, dd.py.values, dd.magV.values
    rx, ry = (px-SPX)*PS, (py-SPY)*PS; R = np.hypot(rx, ry)
    k = (R > RCUT*RS) & (R < RMAX*RS) & (mag <= MAGCUT)
    p, q, r = px[k], py[k], R[k]; ux, uy = rx[k]/r, ry[k]/r
    dx, dy = DX[k]-np.median(DX[k]), DY[k]-np.median(DY[k])
    n = len(p); Z = np.zeros(n)
    xs, ys = (p-NX/2)*PS, (q-NY/2)*PS; xn, yn = (p-NX/2)/(NX/2), (q-NY/2)/(NX/2)

    def solve(with_scale, nuis):
        cxl = [np.ones(n), Z, -ys]; cyl = [Z, np.ones(n), xs]; lab = ['N1', 'N2', 'Th']
        if with_scale:
            cxl.append(xs); cyl.append(ys); lab.append('S')
        cxl.append(ux*RS/r); cyl.append(uy*RS/r); lab.append('L')
        if nuis:
            for i in range(nuis+1):
                for jj in range(nuis+1-i):
                    if i == 0 and jj == 0:
                        continue
                    cxl.append(Z); cyl.append(xn**i*yn**jj); lab.append('v')
        M = np.vstack([np.column_stack(cxl), np.column_stack(cyl)])
        sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
        res = b - Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn))))
        c, e = c/sc, e/sc; i = lab.index('L')
        return c[i], e[i], (1e6*c[lab.index('S')] if with_scale else np.nan), np.sqrt(s2)
    return dict(n=n, ids=set(dd.ID), m2b=solve(True, 0), m2v=solve(True, 2), m1=solve(False, 2))


print('=== calibrated eclipse stack ===', flush=True)
cz = stack()
if not cz:
    raise SystemExit('stage 1 failed -- see %s' % os.path.join(OUT, 'stage1.log'))
r = json.load(zipfile.ZipFile(cz).open('results.txt'))
print('  %d centroids from %d frames (estimator %s)' % (r['n_centroids'], r.get('#frames stacked'), r.get('centroid estimator')))
log = open(os.path.join(OUT, 'stage1.log'), encoding='utf-8', errors='replace').read()
for line in log.splitlines():
    if 'hot pixel' in line.lower() or 'dark' in line.lower()[:30] or 'flat normalis' in line.lower():
        print('  %s' % line.strip())
uz = glob.glob(os.path.join(UNCAL, 'centroid_data*.zip'))
if uz:
    ru = json.load(zipfile.ZipFile(uz[0]).open('results.txt'))
    print('  uncalibrated windowed stack, for comparison: %d centroids' % ru['n_centroids'])

print('\n=== L, calibrated against uncalibrated ===', flush=True)
print('%-26s %-24s %5s | %-22s | %-24s' % ('eclipse stack', 'reference', 'stars', 'Method 2 base', 'Method 2 v-deg2'))
for nm, root, czz in (('calibrated (dark+flat)', OUT, cz), ('uncalibrated', UNCAL, uz[0] if uz else None)):
    if czz is None:
        continue
    for refname, refs in REFS.items():
        if not refs:
            continue
        zp = fit(czz, refname, refs, root)
        if not zp:
            print('%-26s %-24s stage 2 FAILED' % (nm, refname)); continue
        m = method_fits(zp)
        print('%-26s %-24s %5d | %+7.3f +- %.3f" S %+5.0f | %+7.3f +- %.3f" rms %.3f"'
              % (nm, refname, m['n'], m['m2b'][0], m['m2b'][1], m['m2b'][2], m['m2v'][0], m['m2v'][1], m['m2v'][3]))
print('\nGR = 1.751"; the 2024 record (moments, dark+flat) quoted 1.854".')
print('->', OUT)
