"""Rollback attempt 2: footprint moments INSIDE the sensitive path (the MaxIm DL shape).

Attempt 1 (b17_moment.py) was mis-designed. Turning centroid_gaussian_subtract OFF drops
to simple_get_centroids -- the Tetra path, which returns before any background mode is
consulted, thresholds globally at 2 sigma, and therefore drowns in coronal structure
(1173-1527 detections, every solve failed). That is not "moment centroiding"; it is a
different detector.

Douglas' question -- can moment centroiding coexist with the background subtract? -- has
the answer the code actually gives: YES, and it is the config default. With
centroid_gaussian_subtract=True the detector computes `sub` (Gaussian or annular
background), thresholds on local SNR, and takes FLUX-WEIGHTED MOMENTS over each detected
footprint. `centroid_refine_window` is what replaces those moments with the fixed-Gaussian
windowed estimator. So the honest moment convention is:

    centroid_gaussian_subtract = True      (sensitive detection, as always)
    background_subtraction_mode = annular  (a ring background around each star)
    centroid_refine_window     = False     <-- the ONLY change from our 2026 standard

which is also the shape of MaxIm DL's method as Bruns describes it in section 2.3: a
moment calculation over an inner circle with an annulus for the background -- the
convention his 1.752 arcsec was measured in.

Checkpoints, in order: (1) star-by-star affine vs the 2024 centroid lists must collapse
toward zero and go brightness-flat; (2) the L/R bracket must move toward 2.0867322;
(3) the science tiers must still solve (attempt 1 failed here); (4) L.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAWCAL = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images"
MAIN = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_moment2"
NIGHTREFS = json.load(open(glob.glob(r'D:/MEE2024 output/MEE_output/bruns2017_lr/L/stage2/DISTORTION_OUTPUT*/distortion/distortion_results.txt')[0], encoding='utf-8'))['fixed distortion reference files'].split(';')

# our eclipse-day standard, with the estimator switched to footprint moments
S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
      '--set','remove_edgy_centroids=True','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=annular']
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def stack(name, frames):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY,'-m','mee2024.cli','stack',*frames,*S1,
             '--no-scan','--no-display','--quiet','-o',d], os.path.join(d,'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print(f'{name}: STAGE 1 FAILED', flush=True); return None
    n = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
    print(f'{name}: {n} centroids (footprint moments + annular)', flush=True)
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
    def load(zp):
        z = zipfile.ZipFile(zp)
        n = [m for m in z.namelist() if m.endswith('STACKED_CENTROIDS_DATA.csv')][0]
        df = pd.read_csv(z.open(n)); df.columns = [c.strip() for c in df.columns]
        return df
    a, b = load(zip24), load(czip)
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
    fl = P[:,4]; bright = fl > np.nanmedian(fl)
    out = {}
    for nm, sel in (('all', np.ones(len(P), bool)), ('bright', bright), ('faint', ~bright)):
        cxs, *_ = np.linalg.lstsq(X[sel], P[sel,2]-x0, rcond=None)
        cys, *_ = np.linalg.lstsq(X[sel], P[sel,3]-y0, rcond=None)
        out[nm] = 1e6*((cxs[0]-1)+(cys[1]-1))/2
    cx, *_ = np.linalg.lstsq(X, P[:,2]-x0, rcond=None)
    cy, *_ = np.linalg.lstsq(X, P[:,3]-y0, rcond=None)
    rx = P[:,2]-x0 - X@cx; ry = P[:,3]-y0 - X@cy
    print(f'{tag}: affine 2024->moments2026, {len(P)} pairs: all {out["all"]:+.1f} ppm, '
          f'bright {out["bright"]:+.1f}, faint {out["faint"]:+.1f}; scatter '
          f'({rx.std(ddof=1):.3f},{ry.std(ddof=1):.3f}) px', flush=True)
    print(f'   [windowed chain was: all -32/-37, bright -19, faint -49/-53 ppm]', flush=True)

lz = stack('L', sorted(glob.glob(os.path.join(RAWCAL, 'left', 'L_[1-7]_*.fit'))))
rz = stack('R8', sorted(glob.glob(os.path.join(RAWCAL, 'right', 'R_[1-8]_*.fit'))))
if lz: affine_check(lz, r'I:/2017 eclipse data analysis/left right calibration/dataL.zip', 'L')
if rz: affine_check(rz, r'I:/2017 eclipse data analysis/left right calibration/dataR.zip', 'R8')
refL = stage2('L', lz, NIGHTREFS, 'quadratic', '17:44') if lz else None
refR = stage2('R8', rz, NIGHTREFS, 'quadratic', '17:44') if rz else None
if refL and refR:
    pl = json.load(open(refL, encoding='utf-8'))['platescale (arcseconds/pixel)']
    pr = json.load(open(refR, encoding='utf-8'))['platescale (arcseconds/pixel)']
    print(f'moments bracket mean: {(pl+pr)/2:.7f}   [2024 chain 2.0867322; windowed 2026 '
          f'2.0868004; attempt-1 tetra 2.0868062]', flush=True)
    MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}
    for t in ('EA', 'E2', 'EB'):
        cz = stack(t, sorted(glob.glob(os.path.join(MAIN, t, 'preprocessed', '*.fits'))))
        if cz:
            stage2(t, cz, [refL, refR], 'constant', MIDT[t], tol='2.0')
print('done', flush=True)
