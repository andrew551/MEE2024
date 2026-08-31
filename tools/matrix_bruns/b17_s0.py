"""Matrix cell 1 (Bruns 2017): S0 + constant-only stage 2 + first per-tier stage 3.

Held constant from Leon step 3 (tools/step3_s0_v4.py, step3_prelim_L.py):
  - coronal subtraction: unshifted per-tier mean, 10 px Gaussian blur, subtract per
    frame, +2000 ADU pedestal (Bruns' own 2017 method, returned to his data);
  - forbidden disk: painted flat at the pedestal, radius max(1.25 R_sun, measured
    99th-pct saturation radius + 10 px). The margin was 20 px until 2026-08-31: on this
    dataset G 7.52 at 1.49 R_sun landed only 11 px outside E2's disk edge, close enough
    to being masked that Douglas cut the margin in half. 10 px still clears the
    saturation shoulder (the 99th-percentile radius already sits outside the solidly
    clipped core) while giving an inner-annulus star twice the room; residual saturated tips dilated 10 px;
  - stage-1 flags identical to Leon V4 (sensitive stack, gaussian-subtract 4 sigma,
    min_area 2, no hull blob);
  - stage-2: constant-only (distortion_fixed_coefficients=constant), cubic, tol 2.0,
    mag 13 matching, corrections ON, site/wavelength exactly as the canonical L/R
    reductions stored them.
Held constant from THIS dataset's own calibration chain (bruns2017_lr, 2026-08-25):
  - the frozen model is the MEAN of the L and R8 canonical fits (both passed as
    --fix-distortion references) -> imported plate scale 2.0868004 arcsec/px. This is
    the bracket working as designed: the linear atmospheric differential between the
    two sides cancels in the average, and the 45.0 ppm L-R split bounds what remains.
  - no darks, no flats (the L/R chain used none).

Differences from Leon, stated: the disk centre is the per-tier saturated-pixel
centroid (streamer bias here is small -- the three tiers agree to 14 px; the
post-solve Sun ephemeris is printed as the check, and the run must be redone if it
lands > 30 px from the centroid used).

Tiers: EA 17 x 0.62 s (mid 17:43:22 UT), E2 11 x 0.09 s (17:43:47), EB 17 x 0.62 s
(17:44:13). E2 is the inner-annulus tier (saturation 1.42 R_sun vs the deep tiers'
1.98) -- Bruns' scripted "2-star series", his design's answer to near-Sun stars.
"""
import glob, json, os, subprocess, zipfile
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, binary_dilation

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
LR = r"D:/MEE2024 output/MEE_output/bruns2017_lr"
REF_L = glob.glob(os.path.join(LR, 'L', 'stage2', 'DISTORTION_OUTPUT*', 'distortion', 'distortion_results.txt'))[0]
REF_R = glob.glob(os.path.join(LR, 'R8', 'stage2', 'DISTORTION_OUTPUT*', 'distortion', 'distortion_results.txt'))[0]

PS, PED = 2.0868004, 2000.0
R_SUN_AS = 948.0                       # 2017-08-21 apparent; refined in the union tool
RSUN_PX = R_SUN_AS/PS                  # ~454 px
RSAT_PX = {'EA': 901, 'E2': 646, 'EB': 902}    # measured, b17_inventory.py
DISK_MARGIN_PX = 10                            # was 20; see the module docstring
MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}

SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']
STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']
S2 = ['--order','cubic','--date-from-header','--fix-distortion',REF_L,REF_R,
      '--set','distortion_fixed_coefficients=constant',
      '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
      '--set','rough_match_threshhold=36',
      '--set','enable_corrections=True','--set','enable_corrections_ref=True',*SITE]

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

for tier in ('EA', 'E2', 'EB'):
    src = sorted(glob.glob(os.path.join(RAW, f'{tier}_*.fit')))
    pre = os.path.join(OUT, tier, 'preprocessed')
    os.makedirs(pre, exist_ok=True)
    if len(glob.glob(os.path.join(pre, '*.fits'))) != len(src):
        acc = None
        for f in src:
            d = fits.getdata(f).astype(np.float64)
            acc = d if acc is None else acc + d
        mean = acc/len(src)
        model = gaussian_filter(mean, 10.0)
        sat_any = None
        for f in src:
            s = fits.getdata(f) >= 65535
            sat_any = s if sat_any is None else (sat_any | s)
        yy, xx = np.nonzero(sat_any)
        cx, cy = float(xx.mean()), float(yy.mean())
        radius = max(1.25*RSUN_PX, RSAT_PX[tier] + DISK_MARGIN_PX)
        print(f'{tier}: subtracting blurred mean; disk r={radius:.0f} px '
              f'({radius*PS/R_SUN_AS:.2f} R_sun) at sat centroid ({cx:.0f},{cy:.0f})', flush=True)
        ny, nx = mean.shape
        gy, gx = np.mgrid[0:ny, 0:nx]
        disk = np.hypot(gx-cx, gy-cy) <= radius
        for f in src:
            with fits.open(f) as hd:
                img = hd[0].data.astype(np.float64); hdr = hd[0].header.copy()
            sub = np.clip(img - model + PED, 0, 65535)
            sub[binary_dilation(img >= 65535, iterations=10)] = PED
            sub[disk] = PED
            hdr['HISTORY'] = (f'b17 S0: tier-mean blur-10px subtracted, +{PED:.0f} ADU; '
                              f'disk r={radius:.0f}px at ({cx:.0f},{cy:.0f}); sat tips 10px')
            fits.writeto(os.path.join(pre, os.path.basename(f).replace('.fit', '.fits')),
                         sub.astype(np.uint16), hdr, overwrite=True)
    pframes = sorted(glob.glob(os.path.join(pre, '*.fits')))

    zips = glob.glob(os.path.join(OUT, tier, 'centroid_data*.zip'))
    if not zips:
        run([PY,'-m','mee2024.cli','stack',*pframes,*STAGE1,
             '--no-scan','--no-display','--quiet','-o',os.path.join(OUT, tier)],
            os.path.join(OUT, tier, 'stage1.log'))
        zips = glob.glob(os.path.join(OUT, tier, 'centroid_data*.zip'))
    if not zips:
        print(f'{tier}: STAGE 1 FAILED', flush=True); continue
    n = json.load(zipfile.ZipFile(zips[0]).open('results.txt'))['n_centroids']

    d2 = os.path.join(OUT, tier, 'stage2_constant')
    os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        run([PY,'-m','mee2024.cli','distortion',zips[0],*S2,
             '--set','observation_time='+MIDT[tier],'--no-display','--quiet','-o',d2],
            os.path.join(d2, 'stage2.log'))
    res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not res:
        print(f'{tier}: {n} centroids -> constant-only stage 2 FAILED', flush=True); continue
    j = json.load(open(res[0], encoding='utf-8'))
    print(f"{tier}: {n} centroids -> {j['#stars used']} matched, "
          f"rms {j['final rms error (arcseconds)']:.4f} arcsec, "
          f"imported ps {j['platescale (arcseconds/pixel)']:.7f} arcsec/px", flush=True)

    dzip = glob.glob(os.path.join(d2, 'distortion_data*.zip'))[0]
    d3 = os.path.join(OUT, tier, 'stage3')
    os.makedirs(d3, exist_ok=True)
    if not glob.glob(os.path.join(d3, '*.log')):
        run([PY,'-m','mee2024.cli','eclipse',dzip,
             '--set','enable_corrections=True','--set','enable_corrections_ref=True',*SITE,
             '--set','observation_time='+MIDT[tier],'--no-display','--quiet','-o',d3],
            os.path.join(d3, 'stage3.log'))
    for line in open(os.path.join(d3, 'stage3.log'), encoding='utf-8', errors='replace'):
        low = line.lower()
        if any(k in low for k in ('deflection', 'final cov')) and len(line) < 200:
            print('   ', line.rstrip()[:150], flush=True)
print('done', flush=True)
