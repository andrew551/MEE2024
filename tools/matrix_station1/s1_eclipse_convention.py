"""Cell 2: the field side of the convention question -- re-stack the eclipse tier in both
estimators and read L against both references.

`s1_reference_convention_test.py` varied the REFERENCE against a fixed eclipse field and
found the convention worth 0.02-0.03" under Method 2. This varies the FIELD. Together they
close Douglas' question of 2026-09-03: the seventeen 2024 zenith fits are all footprint
moments and the missing raw frames cannot be re-centroided, so does the eclipse field have
to stay in that convention too?

The 2024 eclipse stage 1 of record (`eclipse fields/centroid_data20240416232626.zip`) is
already in the modern matrix settings in every respect except the estimator: sensitive
stacking on, annular background, 4.0 sigma detection, min_area 2, sigma_subtract 0, with the
Sun masked by `delete_saturated_blob` at level 95, `blob_radius_extra` 200 and
`centroid_gap_blob` 100. So re-stacking those same 123 frames with only
`centroid_refine_window` flipped is a clean single-axis test -- and it is also step one of
the eclipse re-stack the reduction needs anyway.

Frames: `G:\\Mexico April 2024\\Station-1-Eclipse-Data\\CapObj\\2024-04-08_18_12_30Z`, the
0.4 s tier at totality, 123 frames (the archive of record used _0000 through _0122).

Each stack is then fitted constant-only against both the 2024 seventeen-field moments
reference and the two-field windowed reference, and L is read under Method 1 and Method 2 on
the common star set. Four cells: eclipse convention x reference convention. If L is flat
across the whole 2 x 2 to well inside the 0.10" Method-2 null floor, the conventions are
free and the missing zenith frames do not block the reduction.

Also reported per stack: centroid count, matched stars, stage-2 rms, and the fitted scale.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
TIER = r"G:/Mexico April 2024/Station-1-Eclipse-Data/CapObj/2024-04-08_18_12_30Z"
Z24 = r"D:/MEE2024 output/Station 1/zenith calibrations"
AB = r"D:/MEE2024 output/MEE_output/station1_record/zenith_raw_ab"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_convention"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0

# the 2024 eclipse stage 1, verbatim except for the estimator under test
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'centroid_window_sigma=2.0']
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_time=18:12', '--set', 'observation_long=105 16 22.1 W',
       '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=15.0',
       '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
       '--set', 'observation_height=2400.0']
ESTIM = {'windowed': ['--set', 'centroid_refine_window=True'],
         'moments': ['--set', 'centroid_refine_window=False']}


def free(v, tag):
    hit = glob.glob(os.path.join(AB, v, tag, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
    return hit[0] if hit else None


REFS = {
    'A_2024_17field_moments': sorted(glob.glob(os.path.join(Z24, '*', '**', 'distortion_results.txt'), recursive=True)),
    'C_2field_windowed': [p for p in (free('windowed_annular', 'f1'), free('windowed_annular', 'f2')) if p],
}
os.makedirs(OUT, exist_ok=True)
FRAMES = sorted(glob.glob(os.path.join(TIER, '*.FIT')))
print('eclipse 0.4 s tier: %d frames, %s .. %s' % (len(FRAMES), os.path.basename(FRAMES[0]), os.path.basename(FRAMES[-1])), flush=True)

t = Time(T_MID, scale='utc'); sun = get_sun(t)
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stack(est):
    d = os.path.join(OUT, est)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  stacking %s (123 frames, this takes a while)...' % est, flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', *FRAMES, *S1, *ESTIM[est],
             '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  %s: STAGE 1 FAILED' % est, flush=True); return None
    r = json.load(zipfile.ZipFile(z[0]).open('results.txt'))
    print('  %-9s %d centroids (estimator %s, background %s), %d frames stacked'
          % (est, r['n_centroids'], r.get('centroid estimator'), r.get('background stubtraction mode'), r.get('#frames stacked')), flush=True)
    return z[0]


def fit(est, cz, refname, refs):
    d = os.path.join(OUT, est, refname)
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
    j = json.load(zf.open([n for n in zf.namelist() if n.endswith('distortion_results.txt')][0]))
    dd = d[d['flag_is_outlier'] == False]
    if common is not None:
        dd = dd[dd.ID.isin(common)]
    ra0, de0 = dd['RA(catalog)'].mean(), dd['DEC(catalog)'].mean()
    X = (dd['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = dd['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, dd.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, dd.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SUNPX, SUNPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(dd['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), dd['DEC(obs)'].values-de0, np.ones(len(dd))]
    cx = np.c_[X, Y, np.ones(len(dd))]
    DX, DY = (ox@ax - cx@ax)*PS, (ox@ay - cx@ay)*PS
    px, py, mag = dd.px.values, dd.py.values, dd.magV.values
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry)
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
    return dict(n=n, j=j, ids=set(dd.ID), m1=solve(False, 2), m2b=solve(True, 0), m2v=solve(True, 2))


cells, zips = {}, {}
for est in ('windowed', 'moments'):
    cz = stack(est)
    if not cz:
        continue
    for refname, refs in REFS.items():
        if not refs:
            continue
        zp = fit(est, cz, refname, refs)
        if not zp:
            print('  %s / %s: stage 2 FAILED' % (est, refname), flush=True); continue
        zips[(est, refname)] = zp
        r = method_fits(zp)
        print('  %-9s vs %-24s %3d stars in the fit, stage-2 rms %.3f"' % (est, refname, r['n'], r['j']['final rms error (arcseconds)']), flush=True)

common = None
for k, zp in zips.items():
    ids = method_fits(zp)['ids']
    common = ids if common is None else (common & ids)
print('\ncommon non-outlier stars across all %d cells: %d' % (len(zips), len(common or [])))
print('\n%-9s %-24s %5s | %-20s | %-30s | %-30s' % ('eclipse', 'reference', 'stars', 'Method 1 (v-deg2)', 'Method 2 base', 'Method 2 v-deg2'))
for (est, refname), zp in zips.items():
    r = method_fits(zp, common); cells[(est, refname)] = r
    print('%-9s %-24s %5d | %+7.3f +- %.3f"    | %+7.3f +- %.3f"  S %+6.0f ppm | %+7.3f +- %.3f"  S %+6.0f ppm'
          % (est, refname, r['n'], r['m1'][0], r['m1'][1], r['m2b'][0], r['m2b'][1], r['m2b'][2],
             r['m2v'][0], r['m2v'][1], r['m2v'][2]))

print('\n=== the 2 x 2 ===')
for refname in REFS:
    a, b = cells.get(('windowed', refname)), cells.get(('moments', refname))
    if a and b:
        print('  FIELD convention, moments -> windowed, against %-24s: M1 %+.3f"  M2 base %+.3f"  M2 v-deg2 %+.3f"'
              % (refname, a['m1'][0]-b['m1'][0], a['m2b'][0]-b['m2b'][0], a['m2v'][0]-b['m2v'][0]))
for est in ('windowed', 'moments'):
    a, b = cells.get((est, 'C_2field_windowed')), cells.get((est, 'A_2024_17field_moments'))
    if a and b:
        print('  REFERENCE, 2024 17-field moments -> 2-field windowed, %-9s field: M1 %+.3f"  M2 base %+.3f"  M2 v-deg2 %+.3f"'
              % (est, a['m1'][0]-b['m1'][0], a['m2b'][0]-b['m2b'][0], a['m2v'][0]-b['m2v'][0]))
vals = [c['m2v'][0] for c in cells.values()]
if vals:
    print('\n  Method 2 (v-deg2) across the whole 2 x 2: %.3f to %.3f", spread %.3f" against a 0.10" null floor'
          ' and sigma_L ~ %.2f"' % (min(vals), max(vals), max(vals)-min(vals), list(cells.values())[0]['m2v'][1]))
print('\n->', OUT)
