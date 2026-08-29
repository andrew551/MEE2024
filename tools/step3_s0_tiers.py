"""S0: stack each SCI_ladder tier with the daytime centroid regime, and see what it reaches.

Douglas' preliminary run used the ZENITH stage-1 regime -- gaussian subtraction off,
threshold 5.0, min_area 4, sigma_subtract 3.0 -- which is tuned for thousands of stars on a
dark sky. On a bright coronal background the CAL_piLeo regime is the analogue: gaussian
subtraction ON to remove the smooth gradient, threshold 4.0, min_area 2, no sigma cut. That
run reached 13 matched stars at V <= 8.94, none inside 4.3 R_sun; the field is meant to hold
132 to G <= 11 from 2.0 R_sun out.

Per TIER, not per block: the folder is the truth about exposure (verified by sky level), and
each tier's two blocks bracket the 1.2 s one, so combining them doubles the depth at that
tier's saturation radius. Integration per tier is 4.6 / 6.9 / 7.2 / 6.0 s -- comparable by
design, at r_sat 1.38 / 1.57 / 1.71 / 1.86 R_sun.

delete_saturated_blob stays TRUE, unlike the CAL_piLeo recipe this regime otherwise copies.
That recipe was written for a field with no Sun in it. Here the corona saturates 265 k pixels
even at 0.1 s, and with the blob left in place its ragged rim is detected as hundreds of
"stars": a first attempt found 773 centroids of which 724 lay inside 2 R_sun, every one of the
brightest fifteen at r = 1.2-1.5 R_sun with areas of 130-1024 px against a real star's few.
The solver works from the brightest centroids, so it was fed nothing but coronal debris and
the plate solve failed outright.

Stage 2 imports the zenith cubic rather than fitting one: Douglas' preliminary fitted a FREE
cubic on 12 stars, ~20 parameters against 24 equations, so its 0.227 arcsec rms is
overfitting and not a field-quality figure. CAL_piLeo is deliberately NOT used -- this asks
what the detector sees, not what the deflection is, and the calibration belongs to S2/S3.
Tolerance 2.0 arcsec is diagnostic (M3 measured zero persistent mismatches there); the final
reduction's 999 needs F16 and MAD-clipped per-star medians, which do not exist yet.
"""
import datetime as dt, glob, json, os, subprocess, sys
from astropy.io import fits

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
SRC = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"        # G: only
OUT = r"D:/MEE2024 output/MEE_output/step3_s0"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
assert len(REFS) == 6, REFS

STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=True',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0',
          # Pin the blob geometry EXPLICITLY. The CLI loads the user's interactive
          # MEE_config.txt underneath --set overrides, and the first S0 run silently
          # inherited blob_radius_extra=500 / centroid_gap_blob=150 from it -- a 650 px
          # exclusion margin beyond the convex hull, ~1.5 R_sun, which walled off the
          # entire 2-4 R_sun annulus (detected 2 of its 36 catalogue stars) and put the
          # blob edge just inside the innermost mag 9.2 star (Douglas' observation).
          # Any parameter not pinned here is whatever the last interactive session was.
          '--set','blob_radius_extra=100','--set','centroid_gap_blob=30',
          '--set','blob_saturation_level=100']

def midtime(frames):
    """Exposure-weighted mid-point. The folder is the truth about exposure, not EXPTIME."""
    ts, ws = [], []
    for f in frames:
        h = fits.getheader(f)
        t = dt.datetime.fromisoformat(h['DATE-OBS'])
        e = float(os.path.basename(os.path.dirname(os.path.dirname(f))).rstrip('s'))
        ts.append(t + dt.timedelta(seconds=e/2)); ws.append(e)
    t0 = min(ts)
    off = sum(w*(t-t0).total_seconds() for t, w in zip(ts, ws))/sum(ws)
    return (t0 + dt.timedelta(seconds=off)).strftime('%H:%M:%S')

for tier in ('0.1s','0.3s','0.6s','1.2s'):
    frames = sorted(glob.glob(os.path.join(SRC, tier, '*', '*.fits')))
    if not frames:
        print(f'{tier}: no frames'); continue
    out = os.path.join(OUT, tier.replace('.','p'))
    os.makedirs(out, exist_ok=True)
    tmid = midtime(frames)
    zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        print(f'{tier}: stage 1 on {len(frames)} frames '
              f'({len(frames)*float(tier.rstrip("s")):.1f} s integration), mid {tmid}', flush=True)
        subprocess.run([PY,'-m','mee2024.cli','stack',*frames,*STAGE1,
                        '--no-scan','--no-display','--quiet','-o',out], cwd=REPO,
                       stdout=open(os.path.join(out,'stage1.log'),'w'), stderr=subprocess.STDOUT)
        zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        print(f'{tier}: STAGE 1 FAILED'); continue
    n = json.load(__import__('zipfile').ZipFile(zips[0]).open('results.txt'))['n_centroids']
    print(f'{tier}: {n} centroids detected', flush=True)

    d2 = os.path.join(out, 'stage2')
    if not glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True):
        os.makedirs(d2, exist_ok=True)
        subprocess.run([PY,'-m','mee2024.cli','distortion',zips[0],
                        '--order','cubic','--date-from-header','--fix-distortion',*REFS,
                        '--set','distortion_fixed_coefficients=quadratic',
                        '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
                        '--set','rough_match_threshhold=36',
                        '--set','enable_corrections=True','--set','enable_corrections_ref=True',
                        '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
                        '--set','observation_height=1101','--set','observation_temp=29.2',
                        '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
                        '--set','observation_wavelength=0.62','--set',f'observation_time={tmid}',
                        '--no-display','--quiet','-o',d2], cwd=REPO,
                       stdout=open(os.path.join(d2,'stage2.log'),'w'), stderr=subprocess.STDOUT)
    r = glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True)
    if r:
        j = json.load(open(r[0], encoding='utf-8'))
        print(f"   -> {j['#stars used']} stars matched, rms {j['final rms error (arcseconds)']:.4f} arcsec, "
              f"ps {j['platescale (arcseconds/pixel)']:.7f}", flush=True)
    else:
        print(f'   -> STAGE 2 FAILED (see {d2}/stage2.log)', flush=True)
print('done', flush=True)
