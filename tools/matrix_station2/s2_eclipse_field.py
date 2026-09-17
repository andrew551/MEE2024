"""Are there stars in Station 2's eclipse field?

The 2025 paper says stations 2-5 "seem not to have detected many stars near the Sun", and none of
them was analysed in 2024. Station 2 now has what it needs to try: a 15-field zenith reference and
an eclipse-day bracket, both reduced (docs/STEP3_2026.md, "Station 2: an external plate scale").

Two tiers, both with the Sun in frame at about (x 2485, y 771) behind a saturated blob ~1.1 R_sun
across:

    Eclipse 100ms   739 frames, 0.100 s, gain 0, 18:11:30-18:12:44
    Eclipse  75ms   749 frames, 0.075 s, gain 0, 18:12:45-18:13:55

Stage 1 uses Station 1's eclipse convention exactly (docs/STEP3_2026.md): sensitive stacking,
Gaussian subtract on, 4 sigma, min_area 2, sigma_subtract 0, annular background, footprint
moments, the saturated blob deleted with the disk mask, and per-frame coronal subtraction at
blur sigma 10 px on a 2000 ADU pedestal. No dark or flat: every Station 2 field dithers far
enough that hot pixels smear rather than stack.

Stage 2 follows the Bruns 2017 correction set -- refraction and aberration on at the site and
the tier mid-time, the frozen zenith cubic, the plate scale free -- because the bracket showed
the distortion changes between epochs and the eclipse field is a third epoch again.

  .venv/Scripts/python.exe tools/matrix_station2/s2_eclipse_field.py stack
  .venv/Scripts/python.exe tools/matrix_station2/s2_eclipse_field.py solve
"""
import glob
import json
import os
import subprocess
import sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
SRC = r"I:/Station2Export/EclipseImages/eclipse"
OUT = r"F:/MEE_output/mexico2024/station2_transfer/eclipse"

# (tag, folder, stage-2 mid-time, frame range or None)
#
# The 100 ms tier opens 9 s after a slew and is still settling: its first surviving frame, 0031,
# carries a saturated blob 1.5 %% larger than the settled median -- it is smeared -- and the blob
# moves 1.53 px/frame over frames 1-9 against 0.68 from frame 10 on. Because the stacker takes
# frames[0] as its alignment reference (F23), the worst frame in the set would otherwise define
# the geometry for all 739. Drop the first ten (Douglas, 2026-09-08; the README warns to "treat
# the first 10-20 images in this set carefully").
#
# The 75 ms tier needs no trim: it follows the 100 ms tier after 1 s at the same pointing, with
# no slew in between.
TIERS = (("100ms", "Eclipse 100ms", "18:12:07", "10-738"),
         ("075ms", "Eclipse 75ms", "18:13:20", None))

# Station 1's eclipse field, verbatim
ECLIPSE = ["sensitive_mode_stack=True", "centroid_gaussian_subtract=True",
           "centroid_gaussian_thresh=4.0", "min_area=2", "sigma_subtract=0.0",
           "background_subtraction_mode=annular", "centroid_refine_window=False",
           "delete_saturated_blob=True", "blob_saturation_level=95",
           "blob_radius_extra=200", "centroid_gap_blob=100", "eclipse_mask_mode=disk",
           "coronal_subtract=True", "coronal_subtract_sigma_px=10.0",
           "coronal_pedestal_adu=2000.0", "remove_edgy_centroids=True"]

SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
        '--set', 'observation_wavelength=0.633', '--set', 'observation_temp=15.2',
        '--set', 'observation_humidity=0.24']


def do_stack():
    for tag, folder, _, rng in TIERS:
        pattern = os.path.join(SRC, folder, "*.FIT")
        frames = sorted(glob.glob(pattern))
        dest = os.path.join(OUT, tag)
        os.makedirs(dest, exist_ok=True)
        if glob.glob(os.path.join(dest, "centroid_data*.zip")):
            print(f"{tag}: already stacked"); continue
        # 739 expanded paths blow past Windows' 32 kB command line, so hand the CLI the
        # pattern and let it expand -- which is what its "lights" argument documents.
        cmd = [PY, "-m", "mee2024.cli", "stack", pattern, "-o", dest,
               "--no-display", "--quiet", "--no-config"]
        if rng:
            cmd += ["--frames", rng]
        for kv in ECLIPSE:
            cmd += ["--set", kv]
        print(f"{tag}: {len(frames)} frames{' (' + rng + ')' if rng else ''} -> {dest}", flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print("  FAILED\n  " + "\n  ".join((r.stderr or r.stdout).strip().splitlines()[-12:]))
            continue
        for line in (r.stdout or "").splitlines():
            if any(k in line for k in ("n centroids", "platescale/arcsec", "blob", "coronal")):
                print("  " + line.strip()[:170])


def do_solve():
    refs = sorted(glob.glob(os.path.join(OUT, "..", "ref_cubic", "*", "**", "distortion_results.txt"),
                            recursive=True))
    print(f"cubic zenith reference: {len(refs)} fields")
    for tag, _, tm, _rng in TIERS:
        z = glob.glob(os.path.join(OUT, tag, "centroid_data*.zip"))
        if not z:
            print(f"  {tag}: not stacked yet"); continue
        d = os.path.join(OUT, tag, "stage2")
        os.makedirs(d, exist_ok=True)
        got = glob.glob(os.path.join(d, "**", "distortion_results.txt"), recursive=True)
        if not got:
            cmd = [PY, "-m", "mee2024.cli", "distortion", z[0], "--order", "cubic",
                   "--fix-distortion", *refs,
                   "--set", "distortion_fixed_coefficients=constant",
                   "--set", "distortion_free_scale=True",
                   "--set", "distortion_fit_tol_initial=20.0", "--set", "distortion_fit_tol=3.0",
                   "--set", "max_star_mag_dist=13", "--set", "rough_match_threshhold=100",
                   *SITE, "--set", "observation_time=" + tm,
                   "--no-display", "--quiet", "-o", d]
            with open(os.path.join(d, "stage2.log"), "w") as fh:
                subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
            got = glob.glob(os.path.join(d, "**", "distortion_results.txt"), recursive=True)
        if got:
            j = json.load(open(got[0], encoding="utf-8"))
            print(f"  {tag}: {j['#stars used']} stars matched, rms {j['final rms error (arcseconds)']:.4f}\", "
                  f"ps {j['platescale (arcseconds/pixel)']:.7f}")
        else:
            print(f"  {tag}: stage 2 found nothing")
            print("    " + open(os.path.join(d, "stage2.log")).read()[-400:].replace("\n", "\n    "))


{"stack": do_stack, "solve": do_solve}[sys.argv[1] if len(sys.argv) > 1 else "stack"]()
