"""The first preliminary L from Leon 2026 -- with every caveat stated up front.

Chain (STEP3_PLAN): zenith cubic -> CAL_piLeo low orders -> constant-only fit on the
science field, corrections ON, then stage 3 (Methods 1 and 2). The CAL_piLeo reference is
the canonical 16-frame/night-2-references result (2.2054043 arcsec/px), which embeds the
frozen zenith cubic; `distortion_fixed_coefficients=constant` leaves only the pointing
offsets free, so the imported plate scale is what Method 1 leans on.

PRELIMINARY means: per-tier stacks only (26-43 stars, not the 71-star union); no per-star
MAD medians; no F16-per-frame; the below-Sun G7.71 anchor still excluded by the edge trim;
no nuisance estimator (S1) -- so the M5-forecast inheritance systematic of 0.1-0.6 arcsec
is UNCORRECTED and expected to dominate; and tol 2.0 rather than the plan's 999, which is
mismatch-safe (M3 measured zero at 2.0) but can clip a large deflection+scatter excursion
and slightly attenuate L. Two tiers give two estimates; their spread is part of the answer.

The printed sigma_L is knowingly understated: v1.4.0-dev still carries the eq.-23 units
bug (PR #6 pending), so the plate-scale term is reported at 1/2.2 of its true size. The
honest term is computed here by hand instead: dL = h * dS * R_sun(arcsec) per the
corrected form, with dS = 25 ppm (HC3-class).
"""
import glob, json, os, subprocess
import numpy as np

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
OUT = r"D:/MEE2024 output/MEE_output/step3_prelim_L"
CAL = glob.glob(r"D:/MEE2024 output/MEE_output/cal_pileo_step2/canonical_16f_night2refs/DISTORTION_OUTPUT*/distortion/distortion_results.txt")[0]
MIDT = {'0p6s': '18:28:33', '1p2s': '18:28:32'}

COMMON = ['--set','enable_corrections=True','--set','enable_corrections_ref=True',
          '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
          '--set','observation_height=1101','--set','observation_temp=29.2',
          '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
          '--set','observation_wavelength=0.62']

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

for tier in ('0p6s', '1p2s'):
    czip = glob.glob(os.path.join(V4, tier, 'centroid_data*.zip'))[0]
    d2 = os.path.join(OUT, tier, 'stage2_constant')
    os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        rc = run([PY,'-m','mee2024.cli','distortion',czip,
                  '--order','cubic','--date-from-header','--fix-distortion',CAL,
                  '--set','distortion_fixed_coefficients=constant',
                  '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
                  '--set','rough_match_threshhold=36',*COMMON,
                  '--set','observation_time='+MIDT[tier],
                  '--no-display','--quiet','-o',d2], os.path.join(d2,'stage2.log'))
    res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not res:
        print(f'{tier}: constant-only stage 2 FAILED', flush=True); continue
    j = json.load(open(res[0], encoding='utf-8'))
    print(f"{tier}: constant-only vs CAL: {j['#stars used']} stars, "
          f"rms {j['final rms error (arcseconds)']:.4f} arcsec "
          f"(imported ps {j['platescale (arcseconds/pixel)']:.7f})", flush=True)
    dzip = glob.glob(os.path.join(d2, 'distortion_data*.zip'))[0]
    d3 = os.path.join(OUT, tier, 'stage3')
    os.makedirs(d3, exist_ok=True)
    run([PY,'-m','mee2024.cli','eclipse',dzip,*COMMON,
         '--set','observation_time='+MIDT[tier],
         '--no-display','--quiet','-o',d3], os.path.join(d3,'stage3.log'))
    print(f'{tier}: stage 3 done -> {d3}', flush=True)
    for line in open(os.path.join(d3,'stage3.log'), encoding='utf-8', errors='replace'):
        low = line.lower()
        if any(k in low for k in ('deflection', ' l =', 'method', 'final')) and len(line) < 200:
            print('   ', line.rstrip(), flush=True)
print('done', flush=True)
