"""Free-cubic stage-2 reruns of the 27-field mosaic stability band (alt 50-80 deg,
unflipped, FOCUSPOS 17041 = the zenith-2 set's focus). Zenith conventions: corrections
OFF, cubic fully free; tol 0.5 rather than the zenith 0.2 because 5-frame stacks carry
~3x the per-star noise (tolerance moves the cubic <1 %, instrument-comparison sec 9).
Reuses the M4 stage-1 zips -- stage 2 only."""
import glob, json, os, subprocess, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RD = r"D:/MEE2024 output/MEE_output/refraction"
ks = pd.read_csv(os.path.join(RD, "band_fields.csv")).k.tolist()
flds = sorted(d for d in os.listdir(os.path.join(RD, "mosaic"))
              if d.startswith("REFR_M") and int(d[6:8]) in ks)
print(f"{len(flds)} band fields", flush=True)
for i, fld in enumerate(flds, 1):
    out = os.path.join(RD, "mosaic", fld, "freecubic")
    if glob.glob(os.path.join(out, "**", "distortion_results.txt"), recursive=True):
        continue
    os.makedirs(out, exist_ok=True)
    z = glob.glob(os.path.join(RD, "mosaic", fld, "centroid_data*.zip"))[0]
    subprocess.run([sys.executable, "-m", "mee2024.cli", "distortion", z,
        "--order", "cubic", "--date-from-header",
        "--set", "distortion_fit_tol=0.5", "--set", "max_star_mag_dist=13",
        "--set", "enable_corrections=False", "--set", "enable_corrections_ref=False",
        "--set", "enable_gravitational_def=False",
        "--no-display", "--quiet", "-o", out], cwd=REPO,
        stdout=open(os.path.join(out, "stage2.log"), "w"), stderr=subprocess.STDOUT)
    ok = bool(glob.glob(os.path.join(out, "**", "distortion_results.txt"), recursive=True))
    print(f"[{i}/{len(flds)}] {fld} {'ok' if ok else 'FAILED'}", flush=True)
print("done: band free-cubic complete", flush=True)
