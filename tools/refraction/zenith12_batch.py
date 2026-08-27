"""All 12 Leon zenith fields from G: (authoritative), exact handoff settings
(zenith stage-1 regime; stage 2 cubic, tol 0.2 arcsec, mag 13, corrections OFF),
so HC3/jackknife can be computed from their residual files. Resumable."""
import glob, json, os, subprocess, sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"D:/MEE2024 output/MEE_output/refraction/zenith12"
STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=False',
          '--set','centroid_gaussian_thresh=5.0','--set','min_area=4',
          '--set','sigma_subtract=3.0','--set','delete_saturated_blob=True',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']

for night in ("2026-08-11", "2026-08-12"):
    zdir = rf"G:/Leon Aug 2026/{night}/Zenith"
    for fld in sorted(os.listdir(zdir)):
        tag = f"{night[5:]}_{fld}"
        out = os.path.join(OUT, tag)
        if glob.glob(os.path.join(out, "stage2", "**", "distortion_results.txt"), recursive=True):
            continue
        os.makedirs(out, exist_ok=True)
        frames = sorted(glob.glob(os.path.join(zdir, fld, "*", "*.fits")))
        zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
        if not zips:
            rc = subprocess.run([PY, "-m", "mee2024.cli", "stack", *frames, *STAGE1,
                                 "--no-scan", "--no-display", "--quiet", "-o", out],
                                cwd=REPO, stdout=open(os.path.join(out, "stage1.log"), "w"),
                                stderr=subprocess.STDOUT).returncode
            zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
            if rc != 0 or not zips:
                print(f"{tag}: STAGE1 FAILED", flush=True); continue
        d2 = os.path.join(out, "stage2")
        os.makedirs(d2, exist_ok=True)
        subprocess.run([PY, "-m", "mee2024.cli", "distortion", zips[0],
                        "--order", "cubic", "--date-from-header",
                        "--set", "distortion_fit_tol=0.2", "--set", "max_star_mag_dist=13",
                        "--set", "rough_match_threshhold=36",
                        "--set", "enable_corrections=False",
                        "--set", "enable_corrections_ref=False",
                        "--no-display", "--quiet", "-o", d2], cwd=REPO,
                       stdout=open(os.path.join(d2, "stage2.log"), "w"),
                       stderr=subprocess.STDOUT)
        r = glob.glob(os.path.join(d2, "**", "distortion_results.txt"), recursive=True)
        if r:
            d = json.load(open(r[0]))
            print(f"{tag}: stars={d['#stars used']} ps={d['platescale (arcseconds/pixel)']:.7f} "
                  f"HC0={d['platescale_relative_uncertainty']*1e6:.2f} ppm", flush=True)
        else:
            print(f"{tag}: STAGE2 FAILED", flush=True)
print("done: zenith12 complete", flush=True)
