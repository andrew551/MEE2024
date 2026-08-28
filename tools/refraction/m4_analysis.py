"""M4 analysis: the meridian mosaic -- model-validity curve, matched pairs, focus step.

Harvests the 80 per-field reductions (corrections ON and OFF), then:
  1. plate scale vs solved altitude, both correction states -- the corrections-ON curve
     should be flat; departures map the standard model's validity boundary;
  2. matched pairs (field k vs 81-k: equal zenith distance, opposite azimuth) -- pair
     differences at k = 19..40 are pure azimuthal asymmetry (same pier side, same
     FOCUSPOS); pairs k = 1..18 add the pier flip + 4-step focus change, so the offset
     between the two pair populations calibrates the focus step;
  3. rms and star count vs altitude (seeing/extinction byproduct).
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = r"D:/MEE2024 output/MEE_output/refraction"
OUT = os.path.join(RD, "m4_mosaic")
os.makedirs(OUT, exist_ok=True)

rows = []
for fld in sorted(glob.glob(os.path.join(RD, "mosaic", "REFR_M*"))):
    name = os.path.basename(fld)
    m = re.match(r"REFR_M(\d+)_az(\d+)_alt([\d.]+)", name)
    k, az_nom, alt_nom = int(m.group(1)), int(m.group(2)), float(m.group(3))
    rec = dict(k=k, az_nom_deg=az_nom, alt_nom_deg=alt_nom,
               flipped=k >= 63, focuspos_steps=17037 if k >= 63 else 17041)
    okboth = True
    for corr in ("on", "off"):
        r = glob.glob(os.path.join(fld, f"corr_{corr}", "**", "distortion_results.txt"),
                      recursive=True)
        if not r:
            okboth = False
            continue
        d = json.load(open(r[0]))
        rec[f"ps_{corr}"] = d["platescale (arcseconds/pixel)"]
        rec[f"rms_{corr}_as"] = d["final rms error (arcseconds)"]
        rec[f"stars_{corr}"] = d["#stars used"]
        rec[f"roll_{corr}_deg"] = d["ROLL"]
        if corr == "on":
            rec["alt_deg"] = d.get("observation alt (degrees)")
            rec["az_deg"] = d.get("observation az (degrees)")
    if okboth:
        rows.append(rec)
df = pd.DataFrame(rows).sort_values("k")
df.to_csv(os.path.join(OUT, "m4_fields.csv"), index=False)
print(f"{len(df)} fields with both correction states")

ZREF_ON = 2.2068874     # arcsec/px, the 08-12 zenith corrections-ON scale (FOCUSPOS 17041)
df["dev_on_ppm"] = (df.ps_on / ZREF_ON - 1) * 1e6
df["corr_ppm"] = (df.ps_on / df.ps_off - 1) * 1e6

alt = df.alt_deg.values
print(f"\nsolved alt range {np.nanmin(alt):.1f}-{np.nanmax(alt):.1f} deg; "
      f"pointing offset (solved - nominal): median "
      f"{np.nanmedian(df.alt_deg - df.alt_nom_deg):+.2f} deg")

print("\ncorrections-ON deviation from zenith reference (ppm), by altitude band:")
bands = [(5, 10), (10, 15), (15, 25), (25, 40), (40, 60), (60, 90)]
for a, b in bands:
    s = df[(df.alt_deg >= a) & (df.alt_deg < b) & ~df.flipped]
    f = df[(df.alt_deg >= a) & (df.alt_deg < b) & df.flipped]
    txt = f"  {a:2d}-{b:2d} deg: "
    if len(s):
        txt += f"unflipped n={len(s):2d} {s.dev_on_ppm.mean():+7.1f} +/- {s.dev_on_ppm.std(ddof=1) if len(s)>1 else 0:5.1f}"
    if len(f):
        txt += f"   flipped n={len(f):2d} {f.dev_on_ppm.mean():+7.1f} +/- {f.dev_on_ppm.std(ddof=1) if len(f)>1 else 0:5.1f}"
    print(txt)

# matched pairs
print("\nmatched pairs (k vs 81-k, equal zenith distance, opposite azimuth), ps_on diff:")
clean, conf = [], []
for k in range(1, 41):
    a_ = df[df.k == k]
    b_ = df[df.k == 81 - k]
    if len(a_) and len(b_):
        dppm = float((b_.ps_on.iloc[0] / a_.ps_on.iloc[0] - 1) * 1e6)
        (conf if 81 - k >= 63 else clean).append((k, a_.alt_nom_deg.iloc[0], dppm))
if clean:
    c = np.array([x[2] for x in clean])
    print(f"  clean pairs (k=19..40, same pier side/focus): n={len(c)}, "
          f"mean {c.mean():+.1f} ppm, sd {c.std(ddof=1):.1f} ppm  <- azimuthal asymmetry")
if conf:
    c2 = np.array([x[2] for x in conf])
    print(f"  flip+focus pairs (k=1..18): n={len(c2)}, mean {c2.mean():+.1f} ppm, "
          f"sd {c2.std(ddof=1):.1f} ppm")
    if clean:
        print(f"  -> focus-step + flip estimate: {c2.mean() - c.mean():+.1f} ppm "
              f"(4 EAF steps => {(c2.mean() - c.mean())/4:+.1f} ppm/step)")

# figure
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for flip, mk, lab in ((False, "o", "unflipped (FOC 17041)"), (True, "s", "flipped (FOC 17037)")):
    s = df[df.flipped == flip]
    ax[0].plot(s.alt_deg, s.dev_on_ppm, mk, ms=4, label=lab)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_ylabel("corrections-ON scale vs zenith ref (ppm)")
ax[0].legend(fontsize=8)
ax[0].set_title("M4: the model-validity curve -- flat means the standard model holds")
ax[1].semilogy(df.alt_deg, df.rms_on_as, "o", ms=4, label="rms (arcsec)")
ax[1].semilogy(df.alt_deg, df.stars_on / 1000, "s", ms=4, label="stars used (thousands)")
ax[1].set_xlabel("solved altitude (deg)")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "m4_curves.png"), dpi=110)
print(f"\nfigure -> {OUT}/m4_curves.png")
