"""Station 2's atmosphere term built the way its eclipse fit is actually bracketed.

Douglas, 2026-09-08. Station 2's eclipse field sits between two calibration fields -- right at
18:10:32-18:11:21, the eclipse at 18:11:30-18:13:55, left at 18:14:05-18:14:57 -- which is Bruns'
design. Cell 1 established that a null must be built the way the science fit was built or it
charges the wrong thing: his one-sided night null gave +-0.150 ", the same data taken against the
MEAN of the fields either side gave +-0.059 ", because a bracket cancels anything linear in time
across the gap (docs/STEP3_CHARTS_AND_SETTINGS.md section 2).

The one-sided Station 2 null is +-0.123 " Method 2 (s2_zenith_null.py). This builds the bracketed
one from the same night series: for each consecutive TRIPLET of zenith fields, the middle field is
refitted constant-only against the average of its two neighbours -- `--fix-distortion` over two
references averages them, which is exactly how Bruns' L/R bracket was frozen. 15 fields give 13
triplets.

  .venv/Scripts/python.exe tools/matrix_station2/s2_bracket_null.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

import s2_zenith_null as Z          # reuse fields(), refit(), design(), fit_L() unchanged

OUT = Z.OUT
Z.NULLS = os.path.join(OUT, "zenith_nulls_bracketed")

FL = Z.fields()
print(f"{len(FL)} zenith fields; {max(0, len(FL) - 2)} bracketed triplets")
print()
print(f"  {'middle field vs its neighbours':<34} {'span':>6} {'N':>4} {'rms':>7} "
      f"{'M1 v2':>9} {'M2 v2':>9} {'S ppm':>8}")

rng = np.random.default_rng(11)
rows = []
for prev, cur, nxt in zip(FL, FL[1:], FL[2:]):
    tm = re.search(r"(\d{2})_(\d{2})_(\d{2})Z", cur["stamp"])
    path = Z.refit(cur["stamp"] + "_bracketed", cur["zip"], [prev["ref"], nxt["ref"]],
                   f"{tm.group(1)}:{tm.group(2)}:{tm.group(3)}")
    if path is None:
        print(f"  {cur['stamp']}: refit failed"); continue
    d = pd.read_csv(path)
    d = d[d["magV"] <= Z.MAGCUT]
    px, py = d["px"].values, d["py"].values
    dx = d["dx_arcsec"].values - np.median(d["dx_arcsec"])
    dy = d["dy_arcsec"].values - np.median(d["dy_arcsec"])
    err = d["error_arcsec"].values
    rx, ry = (px - Z.SUNPX) * Z.PS, (py - Z.SUNPY) * Z.PS
    R = np.hypot(rx, ry)
    keep = R > Z.RCUT * Z.R_SUN_AS
    px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
    if len(px) < 20:
        print(f"  {cur['stamp']}: only {len(px)} stars"); continue
    Lv, _ = Z.fit_L(dx, dy, px, py, rx, ry, R, 2)
    L2, S2 = Z.fit_L(dx, dy, px, py, rx, ry, R, 2, True)
    span = nxt["minute"] - prev["minute"]
    rms = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
    rows.append(dict(field=cur["stamp"][-9:], span_min=span, n=len(px), rms=rms, L1v=Lv, L2=L2, S2=S2))
    print(f"  {cur['stamp'][-9:] + ' vs ' + prev['stamp'][-9:] + '+' + nxt['stamp'][-9:]:<34} "
          f"{span:>6.1f} {len(px):>4} {rms:>7.3f} {Lv:>9.3f} {L2:>9.3f} {S2:>8.1f}", flush=True)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "zenith_nulls_bracketed.csv"), index=False)
    print()
    print("=" * 88)
    print("STATION 2 ATMOSPHERE TERM: ONE-SIDED AGAINST BRACKETED")
    print("=" * 88)
    print(f"  {'construction':<34} {'n':>3} {'mean':>9} {'rms':>9} {'sd':>9}")
    one = pd.read_csv(os.path.join(OUT, "zenith_nulls.csv"))
    for lab, v in (("one-sided, Method 1 v-deg-2", one["L1v"].values),
                   ("one-sided, Method 2", one["L2"].values),
                   ("BRACKETED, Method 1 v-deg-2", df["L1v"].values),
                   ("BRACKETED, Method 2", df["L2"].values)):
        print(f"  {lab:<34} {len(v):>3} {v.mean():>+9.3f} {np.sqrt(np.mean(v**2)):>9.3f} {v.std(ddof=1):>9.3f}")
    r = np.sqrt(np.mean(df["L2"].values ** 2)) / np.sqrt(np.mean(one["L2"].values ** 2))
    print()
    print(f"  the bracket reduces the Method 2 null by a factor {1/r:.2f}")
    print("  Bruns 2017, same comparison: +-0.150 \" one-sided -> +-0.059 \" bracketed (2.5x)")
