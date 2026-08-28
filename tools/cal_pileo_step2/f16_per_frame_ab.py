"""CAL_piLeo: what stack-based F16 rejects, against what per-frame F16 should reject.

Section 8 of the step-2 record measured the per-frame answer -- 1 star of 73, clipped in 6 of
the 8 x 2 s frames -- but on the SUPERSEDED reduction: 17 frames, twelve references, tol 1.0.
The canonical is 16 frames against the six 08-12 references, 74 stars. This redoes the
measurement there, and puts it beside what the merged (stack-based) F16 actually does.

The prediction being tested: on a mixed 1 s + 2 s stack the stacked peak dilutes a clip that
only the long exposures carry, so stack-based F16 finds nothing while per-frame finds real
clipping. If that holds, it is the measured case for the per-frame mask at step 3, where the
SCI_ladder is a ladder of exposures and the sky rises through the sequence as well.

Stage 1 has to be re-run because `peak (adu)` did not exist when the canonical archive was
written; stage 2 then runs twice off that one archive, so rejection is the only variable.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"D:/MEE2024 output/MEE_output/f16_cal_pileo_test2"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
FRAMES = [l.strip() for l in open(r"D:/MEE2024 output/MEE_output/f16_cal_pileo_test/canonical_order.txt")
          if l.strip()]
SAT, NEAR, BOX = 65535, 60000, 4

STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']
STAGE2 = ['--order','cubic','--date-from-header','--fix-distortion',*REFS,
          '--set','distortion_fixed_coefficients=quadratic',
          '--set','distortion_fit_tol=1.0','--set','max_star_mag_dist=13',
          '--set','rough_match_threshhold=36',
          '--set','enable_corrections=True','--set','enable_corrections_ref=True',
          '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
          '--set','observation_height=1101','--set','observation_temp=30.5',
          '--set','observation_pressure=896.6','--set','observation_humidity=0.208',
          '--set','observation_wavelength=0.62','--set','observation_time=18:29:35']

assert len(FRAMES) == 16, f"expected the canonical 16 frames, got {len(FRAMES)}"
assert len(REFS) == 6, f"expected six 08-12 references, got {len(REFS)}"
os.makedirs(OUT, exist_ok=True)

def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def results_of(d):
    r = glob.glob(os.path.join(d, "**", "distortion_results.txt"), recursive=True)
    return json.load(open(r[0], encoding="utf-8")) if r else None

zips = glob.glob(os.path.join(OUT, "centroid_data*.zip"))
if not zips:
    print(f"stage 1 on the canonical {len(FRAMES)} frames", flush=True)
    run([PY,"-m","mee2024.cli","stack",*FRAMES,*STAGE1,
         "--no-scan","--no-display","--quiet","-o",OUT], os.path.join(OUT,"stage1.log"))
    zips = glob.glob(os.path.join(OUT, "centroid_data*.zip"))
    if not zips:
        print("STAGE 1 FAILED - see stage1.log"); sys.exit(1)

arms = {}
for arm, flag in (("off","False"), ("on","True")):
    d2 = os.path.join(OUT, f"stage2_{arm}")
    if not results_of(d2):
        os.makedirs(d2, exist_ok=True)
        run([PY,"-m","mee2024.cli","distortion",zips[0],*STAGE2,
             "--set",f"reject_saturated_stars={flag}","--no-display","--quiet","-o",d2],
            os.path.join(d2,"stage2.log"))
    arms[arm] = results_of(d2)
    if arms[arm] is None:
        print(f"STAGE 2 ({arm}) FAILED - see {d2}/stage2.log"); sys.exit(1)

a, b = arms["off"], arms["on"]
print("\n=== what the merged, STACK-BASED F16 does ===")
print(f"  stars used      {a['#stars used']} -> {b['#stars used']}")
print(f"  plate scale     {a['platescale (arcseconds/pixel)']:.7f} -> {b['platescale (arcseconds/pixel)']:.7f} "
      f"({1e6*(b['platescale (arcseconds/pixel)']/a['platescale (arcseconds/pixel)']-1):+.2f} ppm)")
print(f"  rms             {a['final rms error (arcseconds)']:.4f} -> {b['final rms error (arcseconds)']:.4f} arcsec")
print(f"  outcome         {b.get('saturation outcome','(key absent)')}")

# ---- per-frame truth, the section-8 method on the canonical archive
z = zipfile.ZipFile(zips[0])
meta = json.load(z.open('results.txt'))
shifts = [(round(s[0]), round(s[1])) for s in meta['alignment']['shifts_px']]
files = meta['source_files'] if isinstance(meta['source_files'], list) else eval(meta['source_files'])
stars = pd.read_csv(glob.glob(os.path.join(OUT,"stage2_off","**","TWOD_RESIDUALS.csv"),
                              recursive=True)[0])
peak = np.zeros((len(stars), len(files))); exptime = []
for k, (f, sh) in enumerate(zip(files, shifts)):
    with fits.open(f) as hd:
        img = hd[0].data; exptime.append(float(hd[0].header['EXPTIME']))
    ny, nx = img.shape
    for i, s in stars.iterrows():
        r, c = int(round(s.py)) - sh[0], int(round(s.px)) - sh[1]
        if BOX <= r < ny-BOX and BOX <= c < nx-BOX:
            peak[i,k] = img[r-BOX:r+BOX+1, c-BOX:c+BOX+1].max()

clipped = (peak >= SAT).any(axis=1); near = (peak >= NEAR).any(axis=1)
print("\n=== what PER-FRAME F16 finds, on the same 74-star fit ===")
print(f"  stars in the fit                {len(stars)}")
print(f"  clipped at {SAT} in >=1 raw frame  {int(clipped.sum())}")
print(f"  above {NEAR} in >=1 raw frame      {int(near.sum())}")
print(f"  worst peak on the STACK          {stars.get('peak (adu)', pd.Series([np.nan])).max()}")
for i in np.where(clipped)[0]:
    n = int((peak[i] >= SAT).sum())
    tiers = sorted({exptime[k] for k in range(len(files)) if peak[i,k] >= SAT})
    print(f"    star {i}: clipped in {n} of {len(files)} frames, exposures {tiers}, "
          f"max raw {peak[i].max():.0f}, max in unclipped frames {peak[i][peak[i]<SAT].max():.0f}")
json.dump({"n_stars":int(len(stars)),"clipped":int(clipped.sum()),"near":int(near.sum()),
           "stack_rejected":int(a['#stars used']-b['#stars used'])},
          open(os.path.join(OUT,"f16_cal_pileo_summary.json"),"w"), indent=1)
print("\ndone", flush=True)
