"""Station 2's atmospheric null, from its night zenith series.

Douglas, 2026-09-08: what atmosphere term does Station 2's zenith series give, and is it the same
as Station 1's? It should be -- same site, same night, same mount, same telescope and reducer, and
the two sessions overlap in time (Station 2 04:08-04:41 UTC, Station 1 05:32-06:15).

Built exactly as `tools/matrix_station1/s1_zenith_floor.py` builds Station 1's, so the two numbers
are comparable rather than merely similar: each field is refitted constant-only against the
PREVIOUS field in capture order, the eclipse field's own Sun position is imposed on it, the
science cuts are applied, and L is fitted both ways -- Method 1 with the vertical-deg-2 nuisance,
and Method 2 with the scale free, which is how a scale-free cell is actually reduced. A pair of
zenith fields contains no deflection, so every L is a null and their scatter is the floor.

Station 2 differs from Station 1 in three ways that matter:
  * 15 fields not 17, so 14 pairs not 16;
  * gaps of 28-381 s (median 41) against Station 1's ~2.5 min cadence -- a shorter baseline,
    which should if anything give a SMALLER null;
  * a quarter of the sensor area, so h (the deflection geometry factor) is weaker and the same
    positional scatter buys a larger L.

  .venv/Scripts/python.exe tools/matrix_station2/s2_zenith_null.py
"""
import glob
import json
import os
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/mexico2024/station2_transfer"
NULLS = os.path.join(OUT, "zenith_nulls")

# The zenith fields are near the pole of refraction -- alt 86-90 deg -- so the correction is
# 0.2-2.8 arcsec and a 10 K error in the assumed temperature moves it 0.10 arcsec at most
# (measured 2026-09-07). 10 C is Station 1's own assumption, carried over for comparability.
CORR = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=762.6',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0',
        '--set', 'observation_wavelength=0.633']

NX, NY, PS = 4656, 3520, 1.8672511          # ASI1600MM, the cubic reference mean
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2   # the eclipse field's own Sun, measured
MAGCUT, RCUT = 12.0, 2.0                    # Station 1's cuts, verbatim


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def fields():
    out = []
    for z in sorted(glob.glob(os.path.join(OUT, "zenith", "*", "centroid_data*.zip"))):
        stamp = os.path.basename(os.path.dirname(z))
        m = re.search(r"(\d{2})_(\d{2})_(\d{2})Z", stamp)
        minute = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 60
        ref = glob.glob(os.path.join(OUT, "ref_cubic", stamp, "**", "distortion_results.txt"),
                        recursive=True)
        if not ref:
            continue
        j = json.load(open(ref[0], encoding="utf-8"))
        out.append(dict(stamp=stamp, zip=z, ref=ref[0], minute=minute,
                        ps=j["platescale (arcseconds/pixel)"], n=j["#stars used"],
                        rms=j["final rms error (arcseconds)"]))
    return sorted(out, key=lambda x: x["minute"])


def refit(tag, zip_path, refs, obs_time):
    d = os.path.join(NULLS, tag)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, "**", "TWOD_RESIDUALS.csv"), recursive=True)
    if hit:
        return hit[0]
    run([PY, "-m", "mee2024.cli", "distortion", zip_path, "--order", "cubic",
         "--fix-distortion", *refs, "--set", "distortion_fixed_coefficients=constant",
         "--set", "distortion_fit_tol=2.0", "--set", "max_star_mag_dist=13",
         "--set", "rough_match_threshhold=36", *CORR,
         "--set", "observation_time=" + obs_time,   # each field's own; without it stage 2
         "--no-display", "--quiet", "-o", d],       # cannot build a Time and dies
        os.path.join(d, "stage2.log"))
    hit = glob.glob(os.path.join(d, "**", "TWOD_RESIDUALS.csv"), recursive=True)
    return hit[0] if hit else None


def design(px, py, rx, ry, R, nuis_deg=None, with_scale=False):
    W = NX / 2.0
    xs, ys = (px - NX / 2) / W, (py - NY / 2) / W
    ux, uy = rx / R, ry / R
    n = len(px); Zc = np.zeros(n)
    cx = [np.ones(n), Zc, -(py - NY / 2) * PS]
    cy = [Zc, np.ones(n), (px - NX / 2) * PS]
    lab = ["N1", "N2", "Th"]
    if with_scale:
        cx.append((px - NX / 2) * PS); cy.append((py - NY / 2) * PS); lab.append("S")
    cx.append(ux * R_SUN_AS / R); cy.append(uy * R_SUN_AS / R); lab.append("L")
    if nuis_deg:
        for i in range(nuis_deg + 1):
            for j in range(nuis_deg + 1 - i):
                if i == 0 and j == 0:
                    continue
                cx.append(Zc); cy.append(xs ** i * ys ** j); lab.append(f"v{i}{j}")
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), lab


def fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=None, with_scale=False):
    A, lab = design(px, py, rx, ry, R, nuis_deg, with_scale)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[lab.index("L")], (1e6 * c[lab.index("S")] if with_scale else np.nan)


FL = fields()
print(f"{len(FL)} zenith fields in capture order:")
for f in FL:
    print(f"  {f['stamp']}  {int(f['minute']//60):02d}:{int(f['minute']%60):02d}  "
          f"ps {f['ps']:.7f}  n={f['n']:4d}  rms {f['rms']:.4f}")
ps = np.array([f["ps"] for f in FL])
print(f"  plate scale over the session: mean {ps.mean():.7f}, rms {1e6*ps.std(ddof=1)/ps.mean():.1f} ppm, "
      f"first-to-last {1e6*(ps[-1]-ps[0])/ps[0]:+.1f} ppm")
print()

rng = np.random.default_rng(11)
rows = []
print(f"  {'pair':<22} {'gap':>6} {'N':>4} {'h':>6} {'rms':>7} {'M1 base':>9} {'M1 v2':>9} {'M2 v2':>9} {'S ppm':>8} {'floor':>7}")
for prev, cur in zip(FL, FL[1:]):
    tm = re.search(r"(\d{2})_(\d{2})_(\d{2})Z", cur["stamp"])
    path = refit(cur["stamp"] + "_vs_" + prev["stamp"], cur["zip"], [prev["ref"]],
                 f"{tm.group(1)}:{tm.group(2)}:{tm.group(3)}")
    if path is None:
        print(f"  {cur['stamp']}: null refit failed"); continue
    d = pd.read_csv(path)
    d = d[d["magV"] <= MAGCUT]
    px, py = d["px"].values, d["py"].values
    dx = d["dx_arcsec"].values - np.median(d["dx_arcsec"])
    dy = d["dy_arcsec"].values - np.median(d["dy_arcsec"])
    err = d["error_arcsec"].values
    rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS
    R = np.hypot(rx, ry)
    keep = R > RCUT * R_SUN_AS
    px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
    if len(px) < 20:
        print(f"  {cur['stamp']}: only {len(px)} stars beyond {RCUT} R_sun"); continue
    Lb, _ = fit_L(dx, dy, px, py, rx, ry, R)
    Lv, _ = fit_L(dx, dy, px, py, rx, ry, R, 2)
    L2, S2 = fit_L(dx, dy, px, py, rx, ry, R, 2, True)
    floor = float(np.std([fit_L(dx + rng.normal(0, err / np.sqrt(2)),
                                dy + rng.normal(0, err / np.sqrt(2)),
                                px, py, rx, ry, R, 2)[0] for _ in range(40)], ddof=1))
    h = 1 / np.mean((R_SUN_AS / R) ** 2)
    gap = cur["minute"] - prev["minute"]
    rms = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
    rows.append(dict(pair=cur["stamp"][-9:] + "/" + prev["stamp"][-9:], gap_min=gap, n=len(px),
                     h=h, rms=rms, L1=Lb, L1v=Lv, L2=L2, S2=S2, floor=floor))
    print(f"  {rows[-1]['pair']:<22} {gap:>6.1f} {len(px):>4} {h:>6.1f} {rms:>7.3f} "
          f"{Lb:>9.3f} {Lv:>9.3f} {L2:>9.3f} {S2:>8.1f} {floor:>7.3f}", flush=True)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "zenith_nulls.csv"), index=False)
    print()
    print("=" * 92)
    print("STATION 2 ZENITH NULL")
    print("=" * 92)
    for lab, col in (("Method 1, base", "L1"), ("Method 1, vertical-deg-2", "L1v"),
                     ("Method 2, scale free", "L2")):
        v = df[col].values
        print(f"  {lab:<26} mean {v.mean():+7.3f}\"  rms {np.sqrt(np.mean(v**2)):.3f}\"  "
              f"sd {v.std(ddof=1):.3f}\"  |max| {np.abs(v).max():.3f}\"")
    print(f"  photon floor (bootstrap)   {df['floor'].mean():.3f}\"")
    print(f"  h (geometry factor)        {df['h'].mean():.1f}   stars/pair {df['n'].mean():.0f}")
    print()
    print("  Station 1, same construction (docs/MATRIX_2026.md):")
    print("    Method 2 null +-0.101-0.109\", photon floor 0.026\", h 25.4, 16 pairs")
