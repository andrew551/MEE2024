"""The full 2024-convention reduction: Gaussian background + footprint moments, end to end.

Why this configuration. Three A/Bs on the Bruns calibration fields decomposed the
2024-vs-2026 plate-scale gap (~+33 ppm) into its parts, and the answer was not the one
the 2026-08-31 record first proposed:

    windowed + annular (the 2026 standard)      2.0868004   (reference)
    footprint moments + annular                 2.0867970   -1.6 ppm   <- the ESTIMATOR
    windowed + Gaussian (R6, frame-identical)              -19.1 ppm   <- the BACKGROUND
    footprint moments + Gaussian                2.0867533  -22.6 ppm   (both together)
    the 2024 chain                              2.0867322   (+10.1 ppm still unexplained)

So the era gap is mostly the **background-subtraction mode inside detection**, not the
windowed-versus-moment estimator, which is worth under 2 ppm. Both matter through the
same physics -- how much of an asymmetric off-axis PSF's wings each star's measurement
sees -- but the 17 px annular ring reaches further into the coma than the Gaussian
kernel does, and that is where the radial scale difference lives. The Gaussian+moments
configuration also gives the best residuals of any tried (L rms 0.2087 vs 0.2170
windowed+annular and 0.2326 moments+annular, on more stars).

This script reduces the SCIENCE field in the same convention and refits L. That is the
test that matters: changing the convention on the calibration alone is the cross-mix
artifact (worth +0.227 arcsec by the imported scale, and meaningless), because a uniform
centroid convention largely cancels in the calibration -> science transfer. Only a
consistent end-to-end reduction says what the convention is really worth to L.

Preprocessed frames are shared with the main tree (blur subtraction and forbidden disk
are not part of the centroid convention; their bias was measured <= 0.014 arcsec).
"""
import glob, json, os, subprocess, zipfile

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
MAIN = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024"
MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']
S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
      '--set','remove_edgy_centroids=True','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian']
REF_L = glob.glob(os.path.join(OUT, 'L', 'stage2', '**', 'distortion_results.txt'), recursive=True)[0]
REF_R = glob.glob(os.path.join(OUT, 'R8', 'stage2', '**', 'distortion_results.txt'), recursive=True)[0]

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

for t in ('EA', 'E2', 'EB'):
    d = os.path.join(OUT, t)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        frames = sorted(glob.glob(os.path.join(MAIN, t, 'preprocessed', '*.fits')))
        run([PY,'-m','mee2024.cli','stack',*frames,*S1,'--no-scan','--no-display','--quiet',
             '-o',d], os.path.join(d,'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print(f'{t}: STAGE 1 FAILED', flush=True); continue
    n = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
    d2 = os.path.join(d, 'stage2')
    os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True):
        run([PY,'-m','mee2024.cli','distortion',z[0],'--order','cubic','--date-from-header',
             '--fix-distortion',REF_L,REF_R,'--set','distortion_fixed_coefficients=constant',
             '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
             '--set','rough_match_threshhold=36','--set','enable_corrections=True',
             '--set','enable_corrections_ref=True',*SITE,'--set','observation_time='+MIDT[t],
             '--no-display','--quiet','-o',d2], os.path.join(d2,'stage2.log'))
    r = glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True)
    if r:
        j = json.load(open(r[0], encoding='utf-8'))
        print(f"{t}: {n} centroids -> {j['#stars used']} matched, rms "
              f"{j['final rms error (arcseconds)']:.4f}, imported ps "
              f"{j['platescale (arcseconds/pixel)']:.7f}, estimator={j.get('centroid estimator')}",
              flush=True)
    else:
        print(f'{t}: {n} centroids -> stage 2 FAILED (expected for E2)', flush=True)

print('\n--- the union estimator on the like-2024 tree ---', flush=True)
src = open(os.path.join(REPO, 'tools', 'matrix_bruns', 'b17_union.py'), encoding='utf-8').read()
src = src.replace('matrix_bruns2017', 'matrix_bruns2017_like2024').replace("'stage2_constant'", "'stage2'")
exec(compile(src, 'b17_union_like2024', 'exec'))
