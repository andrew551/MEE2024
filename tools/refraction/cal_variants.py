"""Q1 variants: does frame selection by photometric criteria change CAL_piLeo?

B20 = the settled 17 + the three photometrically-usable post-C3 frames Douglas names
      (1.0s/18_29_51 f4, f5 and 2.0s/18_29_57 f1 -- elevated sky 1.4-2.2x, gradients
      2-3x, but unsaturated with detectable stars).
A16 = the settled 17 minus the accidental true-0.3 s frame (pure 1 s + 2 s tiers).
Both from G: (authoritative), CAL stage-1 regime, stage 2 identical to the baseline but
at each variant's own true-exposure weighted mid-time.
"""
import datetime, glob, json, os, subprocess, sys
from astropy.io import fits

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RD = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"
REFS = sorted(glob.glob(r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed/08-1[12]_Z*.txt"))
assert len(REFS) == 12
G = r"G:\Leon Aug 2026\2026-08-12\Eclipse\CAL_piLeo"

base17 = [l.strip() for l in open(os.path.join(os.path.dirname(RD), "cal_pileo_step2_frames.txt")) if l.strip()]
extra = [os.path.join(G, "1.0s", "18_29_51", "CAL_piLeo_00004.fits"),
         os.path.join(G, "1.0s", "18_29_51", "CAL_piLeo_00005.fits"),
         os.path.join(G, "2.0s", "18_29_57", "CAL_piLeo_00001.fits")]
pure16 = [f for f in base17 if "0.3s" not in f]
assert len(pure16) == 16

STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']

def true_exp(path):
    for e in ("0.3s", "1.0s", "2.0s"):
        if os.sep + e + os.sep in path or "/" + e + "/" in path.replace("\\","/"):
            return float(e[:-1])
    return float(fits.getheader(path)["EXPTIME"])

def wmid(frames):
    rows = []
    for f in frames:
        h = fits.getheader(f)
        rows.append((datetime.datetime.fromisoformat(h["DATE-OBS"]), true_exp(f)))
    t0 = rows[0][0]
    num = sum(((t + datetime.timedelta(seconds=e/2)) - t0).total_seconds() * e for t, e in rows)
    den = sum(e for _, e in rows)
    return (t0 + datetime.timedelta(seconds=num/den)).strftime("%H:%M:%S")

def run(name, frames):
    out = os.path.join(RD, name)
    os.makedirs(out, exist_ok=True)
    zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
    if not zips:
        rc = subprocess.run([PY, "-m", "mee2024.cli", "stack", *frames, *STAGE1,
                             "--no-scan", "--no-exposure-check", "--no-display", "--quiet",
                             "-o", out], cwd=REPO,
                            stdout=open(os.path.join(out, "stage1.log"), "w"),
                            stderr=subprocess.STDOUT).returncode
        zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
        if rc != 0 or not zips:
            print(f"{name}: STAGE1 FAILED"); return
    t = wmid(frames)
    d2 = os.path.join(out, "stage2")
    os.makedirs(d2, exist_ok=True)
    subprocess.run([PY, "-m", "mee2024.cli", "distortion", zips[0],
        "--order", "cubic", "--date-from-header", "--fix-distortion", *REFS,
        "--set", "distortion_fixed_coefficients=quadratic",
        "--set", "distortion_fit_tol=1.0", "--set", "max_star_mag_dist=13",
        "--set", "enable_corrections=True", "--set", "enable_corrections_ref=True",
        "--set", "enable_gravitational_def=False",
        "--set", f"observation_time={t}",
        "--set", "observation_lat=42.740470", "--set", "observation_long=-5.613780",
        "--set", "observation_height=1101", "--set", "observation_temp=30.5",
        "--set", "observation_pressure=896.6", "--set", "observation_humidity=0.208",
        "--set", "observation_wavelength=0.62",
        "--no-display", "--quiet", "-o", d2], cwd=REPO,
        stdout=open(os.path.join(d2, "stage2.log"), "w"), stderr=subprocess.STDOUT)
    r = glob.glob(os.path.join(d2, "**", "distortion_results.txt"), recursive=True)
    if r:
        d = json.load(open(r[0]))
        print(f"{name}: n_frames={len(frames)} t_mid={t} stars={d['#stars used']} "
              f"rms={d['final rms error (arcseconds)']:.4f} arcsec "
              f"ps={d['platescale (arcseconds/pixel)']:.7f} arcsec/px "
              f"HC0={d['platescale_relative_uncertainty']*1e6:.2f} ppm", flush=True)
    else:
        print(f"{name}: STAGE2 FAILED", flush=True)

print("baseline 17: ps=2.2054456 arcsec/px, 73 stars, rms 0.5292 arcsec, t=18:29:34 (for reference)")
run("variant_A16_pure_tiers", pure16)
run("variant_B20_plus_postC3", base17 + extra)
print("done", flush=True)
