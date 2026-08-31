"""The convention rollback: moment-mode centroids per the original design criterion.

Douglas' ruling (2026-08-31): MEE2024's founding design constraint is that it must
reproduce Bruns 2018 on the Bruns 2017 data. The windowed-centroid chain gives
L = 1.556 +- 0.135 -- outside that constraint -- while the moment-based convention
(v0.4.0, and Bruns' own tools) gives 1.74. The convention is therefore rolled back
BY OPTIONS, not by code: centroid_gaussian_subtract=False, centroid_refine_window=
False, sensitive_mode_stack=False emulate the moment centroider inside v1.4.0-dev,
which keeps the rollback testable and reversible.

Chain, each step verified before the next:
  1. moment-mode stage 1 on the raw L / R8 calibration frames;
  2. star-by-star affine against the 2024 centroid lists (dataL/dataR.zip): the
     convention collapse check -- the -32/-37 ppm brightness-dependent scale must
     shrink toward zero;
  3. moment-mode stage 2 (usual settings, 15-night refs): scales vs the 2024 fits;
  4. moment-mode restack of the science tiers (SAME blur-sub + forbidden-disk
     preprocessed frames -- background handling is not the centroid convention, and
     its path bias was measured <= 0.014 arcsec), constant-only stage 2 against the
     MOMENT L/R bracket, and the union estimator -> L.

Everything lands under matrix_bruns2017_moment/ mirroring the main tree.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAWCAL = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images"
MAIN = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_moment"
NIGHTREFS = json.load(open(glob.glob(r'D:/MEE2024 output/MEE_output/bruns2017_lr/L/stage2/DISTORTION_OUTPUT*/distortion/distortion_results.txt')[0], encoding='utf-8'))['fixed distortion reference files'].split(';')

S1_MOMENT = ['--set','sensitive_mode_stack=False','--set','centroid_gaussian_subtract=False',
             '--set','centroid_refine_window=False','--set','min_area=2',
             '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
             '--set','remove_edgy_centroids=True']
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def stack(name, frames, extra=()):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY,'-m','mee2024.cli','stack',*frames,*S1_MOMENT,*extra,
             '--no-scan','--no-display','--quiet','-o',d], os.path.join(d,'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print(f'{name}: STAGE 1 FAILED', flush=True); return None
    n = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
    print(f'{name}: {n} centroids (moment mode)', flush=True)
    return z[0]

def stage2(name, czip, refs, fixed, obstime, tol='0.5'):
    d2 = os.path.join(OUT, name, 'stage2')
    os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        run([PY,'-m','mee2024.cli','distortion',czip,'--order','cubic','--date-from-header',
             '--fix-distortion',*refs,
             '--set',f'distortion_fixed_coefficients={fixed}',
             '--set',f'distortion_fit_tol={tol}','--set','max_star_mag_dist=13',
             '--set','rough_match_threshhold=36',
             '--set','enable_corrections=True','--set','enable_corrections_ref=True',*SITE,
             '--set','observation_time='+obstime,'--no-display','--quiet','-o',d2],
            os.path.join(d2,'stage2.log'))
    r = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not r:
        print(f'{name}: stage 2 FAILED', flush=True); return None
    j = json.load(open(r[0], encoding='utf-8'))
    print(f"{name}: {j['#stars used']} matched, rms {j['final rms error (arcseconds)']:.4f}, "
          f"ps {j['platescale (arcseconds/pixel)']:.7f}", flush=True)
    return r[0]

def affine_check(czip, zip24, tag):
    def load(zp, inner):
        z = zipfile.ZipFile(zp)
        n = [m for m in z.namelist() if m.endswith('STACKED_CENTROIDS_DATA.csv')][0]
        df = pd.read_csv(z.open(n)); df.columns = [c.strip() for c in df.columns]
        return df
    a, b = load(zip24, True), load(czip, False)
    pairs = []
    for _, r in a.iterrows():
        d = np.hypot(b.px.values-r.px, b.py.values-r.py)
        j = int(np.argmin(d))
        if d[j] < 3.0:
            pairs.append((r.px, r.py, b.px.values[j], b.py.values[j],
                          r.get('flux (noise-normed)', np.nan)))
    P = np.array(pairs)
    x0, y0 = 3296/2, 2472/2
    X = np.c_[P[:,0]-x0, P[:,1]-y0, np.ones(len(P))]
    cx, *_ = np.linalg.lstsq(X, P[:,2]-x0, rcond=None)
    cy, *_ = np.linalg.lstsq(X, P[:,3]-y0, rcond=None)
    scale = ((cx[0]-1)+(cy[1]-1))/2
    fl = P[:,4]; bright = fl > np.nanmedian(fl)
    out = {}
    for nm, sel in (('all', np.ones(len(P), bool)), ('bright', bright), ('faint', ~bright)):
        cxs, *_ = np.linalg.lstsq(X[sel], P[sel,2]-x0, rcond=None)
        cys, *_ = np.linalg.lstsq(X[sel], P[sel,3]-y0, rcond=None)
        out[nm] = 1e6*((cxs[0]-1)+(cys[1]-1))/2
    print(f'{tag}: affine 2024->moment2026, {len(P)} pairs: scale all {out["all"]:+.1f} ppm, '
          f'bright {out["bright"]:+.1f}, faint {out["faint"]:+.1f} '
          f'(windowed chain was ~-32/-37 all, -19 bright, -49/-53 faint)', flush=True)

# ---- 1-3: calibration fields in moment mode
lz = stack('L', sorted(glob.glob(os.path.join(RAWCAL, 'left', 'L_[1-7]_*.fit'))))
rz = stack('R8', sorted(glob.glob(os.path.join(RAWCAL, 'right', 'R_[1-8]_*.fit'))))
if lz: affine_check(lz, r'I:/2017 eclipse data analysis/left right calibration/dataL.zip', 'L')
if rz: affine_check(rz, r'I:/2017 eclipse data analysis/left right calibration/dataR.zip', 'R8')
refL = stage2('L', lz, NIGHTREFS, 'quadratic', '17:44') if lz else None
refR = stage2('R8', rz, NIGHTREFS, 'quadratic', '17:44') if rz else None
if refL and refR:
    pl = json.load(open(refL, encoding='utf-8'))['platescale (arcseconds/pixel)']
    pr = json.load(open(refR, encoding='utf-8'))['platescale (arcseconds/pixel)']
    print(f'moment bracket mean: {(pl+pr)/2:.7f}  (2024 chain: 2.0867322; windowed 2026: 2.0868004)', flush=True)

    # ---- 4: science tiers in moment mode (same preprocessed frames)
    MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}
    for t in ('EA', 'E2', 'EB'):
        pframes = sorted(glob.glob(os.path.join(MAIN, t, 'preprocessed', '*.fits')))
        cz = stack(t, pframes)
        if cz:
            stage2(t, cz, [refL, refR], 'constant', MIDT[t], tol='2.0')
print('done', flush=True)
