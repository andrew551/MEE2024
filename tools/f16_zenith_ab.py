"""F16 A/B on the six 08-12 zenith fields: does rejecting clipped stars move the frozen cubic?

The question this answers. `reject_saturated_stars` ships False; the case for defaulting it
on is that nothing else at any stage tests a peak value, so at a loose tolerance a clipped
star reaches the fit unchallenged. Flipping it moves d(3000), which every downstream number
is pinned to -- so the size of that move has to be measured, not assumed.

The design. Stage 1 runs ONCE per field, because `peak (adu)` is written unconditionally
now; stage 2 then runs TWICE off the same centroid archive, with rejection off and on. Same
frames, same centroids, same catalogue, same frozen settings -- the rejection is the only
difference between the arms, so any shift is attributable.

Why stage 1 must be re-run at all: the existing archives under refraction/zenith12 predate
the peak column, and `_drop_saturated` declines to reject when it is absent (correctly --
"not known" is not grounds for rejection). Reusing them would produce a null result that
means nothing.

Settings are the handoff's, exactly (zenith stage-1 regime; stage 2 cubic, tol 0.2, mag 13,
corrections OFF), so the control arm is comparable with the twelve reference files in
calibration/zenith_cubic/.

Note the stack-based limitation: `peak_value` measures the stacked image, so this is a fair
test only because the zenith frames share one exposure -- a star clipped in all 30 sits at
the clip level in the mean. On a mixed-exposure stack like CAL_piLeo it would be inert
(CAL_PILEO_STEP2.md section 8), which is why that field is not tested here.
"""
import glob, json, os, subprocess, sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
SRC = r"G:/Leon Aug 2026/2026-08-12/Zenith"      # G: only -- the authoritative tree
OUT = r"D:/MEE2024 output/MEE_output/f16_zenith_test"

STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=False',
          '--set','centroid_gaussian_thresh=5.0','--set','min_area=4',
          '--set','sigma_subtract=3.0','--set','delete_saturated_blob=True',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']
STAGE2 = ['--order','cubic','--date-from-header',
          '--set','distortion_fit_tol=0.2','--set','max_star_mag_dist=13',
          '--set','rough_match_threshhold=36',
          '--set','enable_corrections=False','--set','enable_corrections_ref=False']

sys.path.insert(0, os.path.join(REPO, "tools", "refraction"))
from band_stability import d3000          # validated against the handoff

def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def results_of(d):
    r = glob.glob(os.path.join(d, "**", "distortion_results.txt"), recursive=True)
    return json.load(open(r[0])) if r else None

rows = []
for fld in sorted(os.listdir(SRC)):
    out = os.path.join(OUT, fld)
    os.makedirs(out, exist_ok=True)

    zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
    if not zips:
        frames = sorted(glob.glob(os.path.join(SRC, fld, "*", "*.fits")))
        print(f"{fld}: stage 1 on {len(frames)} frames", flush=True)
        run([PY,"-m","mee2024.cli","stack",*frames,*STAGE1,
             "--no-scan","--no-display","--quiet","-o",out],
            os.path.join(out, "stage1.log"))
        zips = glob.glob(os.path.join(out, "centroid_data*.zip"))
        if not zips:
            print(f"{fld}: STAGE 1 FAILED", flush=True); continue

    arms = {}
    for arm, flag in (("off", "False"), ("on", "True")):
        d2 = os.path.join(out, f"stage2_{arm}")
        if not results_of(d2):
            os.makedirs(d2, exist_ok=True)
            run([PY,"-m","mee2024.cli","distortion",zips[0],*STAGE2,
                 "--set",f"reject_saturated_stars={flag}",
                 "--no-display","--quiet","-o",d2],
                os.path.join(d2, "stage2.log"))
        arms[arm] = results_of(d2)

    if not (arms["off"] and arms["on"]):
        print(f"{fld}: STAGE 2 FAILED", flush=True); continue
    a, b = arms["off"], arms["on"]
    rows.append(dict(field=fld,
                     d_off=abs(d3000(a)), d_on=abs(d3000(b)),
                     n_off=a["#stars used"], n_on=b["#stars used"],
                     ps_off=a["platescale (arcseconds/pixel)"],
                     ps_on=b["platescale (arcseconds/pixel)"],
                     note=b.get("saturation outcome", "(key absent)")))
    r = rows[-1]
    print(f"{fld}: d3000 {r['d_off']:.4f} -> {r['d_on']:.4f} arcsec "
          f"({1e2*(r['d_on']/r['d_off']-1):+.2f} %), stars {r['n_off']} -> {r['n_on']}; {r['note']}",
          flush=True)

if rows:
    import statistics as st
    mo = st.mean(r["d_off"] for r in rows); mn = st.mean(r["d_on"] for r in rows)
    po = st.mean(r["ps_off"] for r in rows); pn = st.mean(r["ps_on"] for r in rows)
    print("\n=== six 08-12 zenith fields ===")
    print(f"d(3000) mean   : {mo:.4f} -> {mn:.4f} arcsec   ({1e2*(mn/mo-1):+.3f} %)")
    print(f"  per-field sd : {1e2*st.pstdev([r['d_off'] for r in rows])/mo:.2f} % (off arm)")
    print(f"plate scale    : {po:.7f} -> {pn:.7f} ({1e6*(pn/po-1):+.2f} ppm)")
    print(f"stars rejected : {sum(r['n_off']-r['n_on'] for r in rows)} of {sum(r['n_off'] for r in rows)}")
    json.dump(rows, open(os.path.join(OUT, "f16_ab_summary.json"), "w"), indent=1)
print("done", flush=True)
