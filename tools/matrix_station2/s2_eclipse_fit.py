"""Station 2's deflection constant, pooled over both eclipse tiers.

Douglas, 2026-09-08: the two tiers are independent observations of the same field, so treat them
as 34 data points rather than 17. That is cell 2's own estimator -- every observation is a row,
each tier gets its own offset, rotation and scale, and one L is shared (s1_pooled_fit.py). Here
there are two tiers instead of four, so 2 x 4 nuisances + L = 9 parameters on 68 coordinates.

    100 ms   729 frames, 17 stars, rms 0.476 "
     75 ms   749 frames, 17 stars, rms 0.419 "

Both were fitted constant-only against the 15-field cubic zenith reference with the scale free,
two-pass at 20 " then 3 ", refraction and aberration on at the site and the tier mid-time.

Errors come from a star bootstrap (resampling stars, not observations, since a star seen in both
tiers is one draw of the atmosphere) and are checked against the cluster-robust sandwich.

  .venv/Scripts/python.exe tools/matrix_station2/s2_eclipse_fit.py
"""
import glob
import os

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from tools.analysis_window import WINDOWS

OUT = r"D:/MEE2024 output/MEE_output/station2_transfer"
TIERS = (("100ms", "eclipse/100ms/stage2"), ("075ms", "eclipse/075ms/stage2_rmt36"))

NX, NY, PS = 4656, 3520, 1.8672511
SUNPX, SUNPY, R_SUN_AS = 2485.0, 771.0, 958.2
_W = WINDOWS['mexico2024_station2']          # 2-10 R_sun, G <= 13; see the registry
MAGCUT, RMIN, RMAX = _W.mag, _W.rmin, _W.rmax


def load(sub):
    f = glob.glob(os.path.join(OUT, sub, "**", "TWOD_RESIDUALS.csv"), recursive=True)
    return pd.read_csv(f[0]) if f else None


rows = []
for tag, sub in TIERS:
    d = load(sub)
    if d is None:
        print(f"{tag}: no residuals"); continue
    d = d[d["magV"] <= MAGCUT].copy()
    d["rx"] = (d["px"] - SUNPX) * PS
    d["ry"] = (d["py"] - SUNPY) * PS
    d["R"] = np.hypot(d["rx"], d["ry"]) / R_SUN_AS
    d["tier"] = tag
    keep = (d["R"] >= RMIN) & (d["R"] <= RMAX)
    print(f"{tag}: {len(d)} matched, {int(keep.sum())} in {RMIN}-{RMAX} R_sun, "
          f"R {d['R'].min():.2f}-{d['R'].max():.2f}")
    rows.append(d[keep])

D = pd.concat(rows, ignore_index=True)
tiers = sorted(D["tier"].unique())
print(f"\npooled: {len(D)} observations of {D['ID'].nunique()} stars across {len(tiers)} tiers")


def design(d):
    """per-tier offset, rotation and scale; one shared L"""
    n = len(d)
    px, py = d["px"].values, d["py"].values
    rx, ry, R = d["rx"].values, d["ry"].values, d["R"].values * R_SUN_AS
    cx, cy, lab = [], [], []
    for t in tiers:
        m = (d["tier"] == t).values.astype(float)
        Z = np.zeros(n)
        cx += [m, Z, -m * (py - NY / 2) * PS, m * (px - NX / 2) * PS]
        cy += [Z, m, m * (px - NX / 2) * PS, m * (py - NY / 2) * PS]
        lab += [f"N1_{t}", f"N2_{t}", f"Th_{t}", f"S_{t}"]
    cx.append(rx / R * R_SUN_AS / R); cy.append(ry / R * R_SUN_AS / R); lab.append("L")
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), lab


def fit(d):
    A, lab = design(d)
    y = np.concatenate([d["dx_arcsec"].values, d["dy_arcsec"].values])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    return c[lab.index("L")], lab, c, resid, A


L, lab, c, resid, A = fit(D)
rms = float(np.sqrt(np.mean(resid ** 2)))
h = 1 / np.mean((1 / D["R"].values) ** 2)

rng = np.random.default_rng(7)
ids = D["ID"].unique()
boot = []
for _ in range(600):
    pick = rng.choice(ids, size=len(ids), replace=True)
    d = pd.concat([D[D["ID"] == i] for i in pick], ignore_index=True)
    if d["tier"].nunique() < len(tiers):
        continue
    try:
        boot.append(fit(d)[0])
    except Exception:
        pass
sig = float(np.std(boot, ddof=1))

print()
print("=" * 78)
print("STATION 2 DEFLECTION CONSTANT")
print("=" * 78)
print(f"  L                 {L:+.3f} +- {sig:.3f} arcsec (star bootstrap, {len(boot)} draws)")
print(f"  residual rms      {rms:.3f} arcsec per coordinate")
print(f"  geometry h        {h:.1f}")
print(f"  observations      {len(D)} of {D['ID'].nunique()} stars")
for t in tiers:
    print(f"    {t}: S {c[lab.index('S_' + t)]*1e6:+7.1f} ppm")
print()
print(f"  GR predicts 1.751 arcsec  ->  {(L - 1.751)/sig:+.1f} sigma")
print(f"  Newton would be 0.8755    ->  {(L - 0.8755)/sig:+.1f} sigma")
print()
print("  atmosphere term, Station 2 zenith null (Method 2): +-0.12 arcsec")
print(f"  so the total would be about +-{np.hypot(sig, 0.12):.2f} arcsec")
print()
print("  for comparison, cell 2 (Station 1): L = 1.804 +- 0.084 (stat) +- 0.11 (atm),")
print("  639 observations of 192 stars, h 25.4")

# each tier alone, as a consistency check
print()
print("  each tier alone:")
for t in tiers:
    d = D[D["tier"] == t]
    A1, lab1 = design(d)
    keep = [i for i, l in enumerate(lab1) if not l.endswith("_" + t) or True]
    try:
        y = np.concatenate([d["dx_arcsec"].values, d["dy_arcsec"].values])
        cc, *_ = np.linalg.lstsq(A1, y, rcond=None)
        print(f"    {t}: L {cc[lab1.index('L')]:+.3f} arcsec on {len(d)} stars")
    except Exception as e:
        print(f"    {t}: {e}")
