"""Leon 2026 re-reduced in the Bruns-compatible convention, so the matrix cells compare.

Cell 1 established (docs/MATRIX_2026.md) that the centroid convention is worth ~0.2
arcsec of L on Bruns' optics, and that the lever is the BACKGROUND MODE rather than the
windowed-versus-moment estimator: annular vs Gaussian moved the fitted plate scale by
19.1 ppm on frame-identical stacks, the estimator by 1.6. Bruns 2018 is reproduced by
`background_subtraction_mode=Gaussian` + `centroid_refine_window=False`.

Leon's headline -- L = 1.98 +- 0.60 (stat) +- 0.33 (atm) -- was reduced windowed+annular,
so it is quoted in a different convention from cell 1 and the two cannot be compared as
they stand. This re-runs Leon end to end in cell 1's convention.

What changes and what deliberately does not:

  * CHANGED, at both levels that see stars: CAL_piLeo (the imported plate scale) and the
    four science tiers are re-stacked with Gaussian background + footprint moments.
    Changing only one level is the cross-mix that produced a meaningless +0.24 arcsec on
    the Bruns data; a convention has to be applied to the calibration and the science
    field together or it does not cancel where it should.
  * UNCHANGED: the six 08-12 zenith cubic references stay as they are, reduced windowed.
    Not an oversight -- the Bruns cell froze its 15 night references the same way, and
    swapping night references there moved the fitted scale by 1.0-1.1 ppm. Holding them
    fixed keeps the two cells' treatment identical, which is the point of the exercise.
  * UNCHANGED: the FROZEN step3_s0_v4 preprocessed frames (coronal subtraction +
    forbidden disk) are reused as they are. The disk margin was cut 20 -> 10 px on
    2026-08-31, but Leon's innermost admitted star sits 2.07 R_sun out, ~570 px clear of
    its tier's disk, so the margin cannot reach it. Reusing the frames keeps this a
    single-variable comparison against the headline.

Outputs to step3_bruns_convention/. The union/estimator is the same machinery as the
headline (tools/step3_s2_union.py) with its paths redirected, so the two numbers differ
only by the convention.
"""
import glob, json, os, subprocess, sys, zipfile

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
OUT = r"D:/MEE2024 output/MEE_output/step3_bruns_convention"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
FRAMES = [l.strip() for l in open(os.path.join(REPO, "calibration", "cal_pileo_frames.txt"),
                                  encoding="utf-8") if l.strip()]
assert len(REFS) == 6 and len(FRAMES) == 16

# the convention under test: Gaussian background, footprint moments. Everything else is
# the eclipse-day standard of docs/FIELD_PRESETS.md.
S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
      '--set','remove_edgy_centroids=True','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian']
SITE = ['--set','observation_lat=42.740470','--set','observation_long=-5.613780',
        '--set','observation_height=1101','--set','observation_humidity=0.208',
        '--set','observation_wavelength=0.62']
CAL_MET = ['--set','observation_temp=30.5','--set','observation_pressure=896.6']
SCI_MET = ['--set','observation_temp=29.2','--set','observation_pressure=896.7']
MIDT = {'0p1s':'18:28:32','0p3s':'18:28:34','0p6s':'18:28:33','1p2s':'18:28:32'}

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def stage1(name, frames):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY,'-m','mee2024.cli','stack',*frames,*S1,'--no-scan','--no-display','--quiet',
             '-o',d], os.path.join(d,'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print(f'{name}: STAGE 1 FAILED', flush=True); return None
    n = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
    print(f'{name}: {n} centroids', flush=True)
    return z[0]

# ---- 1. CAL_piLeo, the imported plate scale. ORDER is part of the definition (F23).
cz = stage1('cal_pileo', FRAMES)
d2 = os.path.join(OUT, 'cal_pileo', 'stage2')
os.makedirs(d2, exist_ok=True)
if cz and not glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True):
    run([PY,'-m','mee2024.cli','distortion',cz,'--order','cubic','--date-from-header',
         '--fix-distortion',*REFS,'--set','distortion_fixed_coefficients=quadratic',
         '--set','distortion_fit_tol=1.0','--set','max_star_mag_dist=13',
         '--set','rough_match_threshhold=36','--set','enable_corrections=True',
         '--set','enable_corrections_ref=True',*SITE,*CAL_MET,
         '--set','observation_time=18:29:35','--no-display','--quiet','-o',d2],
        os.path.join(d2,'stage2.log'))
calres = glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True)
if not calres:
    print('CAL_piLeo stage 2 FAILED -- stopping', flush=True); sys.exit(1)
j = json.load(open(calres[0], encoding='utf-8'))
PS_NEW = j['platescale (arcseconds/pixel)']
print(f"CAL_piLeo: {j['#stars used']} stars, rms {j['final rms error (arcseconds)']:.4f}, "
      f"ps {PS_NEW:.7f} (windowed+annular canonical was 2.2054043: "
      f"{1e6*(PS_NEW-2.2054043)/2.2054043:+.1f} ppm)", flush=True)

# ---- 2. the science tiers, constant-only against the new CAL
for tier in ('0p1s','0p3s','0p6s','1p2s'):
    frames = sorted(glob.glob(os.path.join(V4, tier, 'preprocessed', '*.fits')))
    tz = stage1(tier, frames)
    if not tz: continue
    td = os.path.join(OUT, tier, 'stage2_constant')
    os.makedirs(td, exist_ok=True)
    if not glob.glob(os.path.join(td,'**','distortion_results.txt'), recursive=True):
        run([PY,'-m','mee2024.cli','distortion',tz,'--order','cubic','--date-from-header',
             '--fix-distortion',calres[0],'--set','distortion_fixed_coefficients=constant',
             '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
             '--set','rough_match_threshhold=36','--set','enable_corrections=True',
             '--set','enable_corrections_ref=True',*SITE,*SCI_MET,
             '--set','observation_time='+MIDT[tier],'--no-display','--quiet','-o',td],
            os.path.join(td,'stage2.log'))
    r = glob.glob(os.path.join(td,'**','distortion_results.txt'), recursive=True)
    if r:
        jj = json.load(open(r[0], encoding='utf-8'))
        print(f"  {tier}: {jj['#stars used']} matched, rms "
              f"{jj['final rms error (arcseconds)']:.4f}, estimator={jj.get('centroid estimator')}",
              flush=True)
    else:
        print(f'  {tier}: stage 2 FAILED', flush=True)

# ---- 3. the union and the estimator, same machinery as the headline
print('\n--- union + estimator (Bruns convention) ---', flush=True)
src = open(os.path.join(REPO, 'tools', 'step3_s2_union.py'), encoding='utf-8').read()
src = src.replace(r"BL = r\"D:/MEE2024 output/MEE_output/step3_prelim_L\"",
                  r"BL = r\"D:/MEE2024 output/MEE_output/step3_bruns_convention\"")
src = src.replace('D:/MEE2024 output/MEE_output/step3_prelim_L',
                  'D:/MEE2024 output/MEE_output/step3_bruns_convention')
src = src.replace('D:/MEE2024 output/MEE_output/step3_s0_v4',
                  'D:/MEE2024 output/MEE_output/step3_bruns_convention')
src = src.replace("PS, NX, NY, W_NORM = 2.2054043",
                  f"PS, NX, NY, W_NORM = {PS_NEW:.7f}")
exec(compile(src, 'step3_s2_union_bruns_convention', 'exec'))
