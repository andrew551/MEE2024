"""Instrument stability from the mosaic band (alt 50-80 deg, unflipped, FOCUSPOS 17041).

Part A: plate scale -- 27 corrections-ON fields, raw and drift-detrended, against the
zenith-2 set taken the same night at the same focus.
Part B (after band_cubic.py): per-field free-cubic d(3000 px) -- the isotropic radial
cubic displacement at r = 3000 px, the handoff's own metric -- validated here against the
handoff files before use.
"""
import datetime, glob, json, os
import numpy as np, pandas as pd

RD = r"D:/MEE2024 output/MEE_output/refraction"
W = 3124.0

def d3000(res_json):
    """Isotropic radial cubic displacement at r=3000 px, arcsec, from the coeff dicts."""
    j = res_json
    cx = j["distortion coeffs x"]; cy = j["distortion coeffs y"]
    ps = j["platescale (arcseconds/pixel)"]
    th = np.linspace(0, 2*np.pi, 720, endpoint=False)
    r = 3000.0 / W
    x, y = r*np.cos(th), r*np.sin(th)
    cub_x = (cx["x^3"]*x**3 + cx["x^2 * y"]*x*x*y + cx["x * y^2"]*x*y*y + cx["y^3"]*y**3)
    cub_y = (cy["x^3"]*x**3 + cy["x^2 * y"]*x*x*y + cy["x * y^2"]*x*y*y + cy["y^3"]*y**3)
    rad = cub_x*np.cos(th) + cub_y*np.sin(th)      # coeffs act on (x-basis -> x etc.)
    return float(np.mean(rad)) * ps                 # px (w-normalised basis) -> arcsec

# validate against the handoff (published: 08-11 mean 3.1799", 08-12 mean 3.0297")
H = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
for night in ("08-11", "08-12"):
    vals = [abs(d3000(json.load(open(f)))) for f in sorted(glob.glob(f"{H}/{night}_Z*.txt"))]
    print(f"validator: {night} handoff d(3000) mean {np.mean(vals):.4f} arcsec "
          f"(published {'3.1799' if night=='08-11' else '3.0297'}), "
          f"per-field sd {np.std(vals, ddof=1)/np.mean(vals)*100:.2f} %")

# Part A: plate-scale stability
df = pd.read_csv(os.path.join(RD, "m4_mosaic", "m4_fields.csv"))
ks = set(pd.read_csv(os.path.join(RD, "band_fields.csv")).k)
band = df[df.k.isin(ks)].sort_values("k").copy()
tm = []
for _, r in band.iterrows():
    f = glob.glob(os.path.join(RD, "mosaic",
                  f"REFR_M{int(r.k):02d}_az{int(r.az_nom_deg):03d}_alt{r.alt_nom_deg:05.2f}",
                  "corr_on", "**", "distortion_results.txt"), recursive=True)[0]
    j = json.load(open(f))
    hh, mm, ss = j["observation_time (UTC)"].split(":")
    t = int(hh)*60 + int(mm) + int(ss)/60
    tm.append(t if t > 600 else t + 1440)          # past-midnight wrap
band["t_min"] = tm
ps = band.ps_on.values
mean = ps.mean()
raw_sd = ps.std(ddof=1)/mean*1e6
A = np.column_stack([np.ones(len(band)), band.t_min - band.t_min.mean()])
c, *_ = np.linalg.lstsq(A, (ps/mean - 1)*1e6, rcond=None)
det = (ps/mean - 1)*1e6 - A @ c
print(f"\nPart A -- plate scale, 27 band fields (corrections ON):")
print(f"  mean {mean:.7f} arcsec/px; raw f2f sd {raw_sd:.2f} ppm, se {raw_sd/np.sqrt(len(ps)):.2f} ppm")
print(f"  drift within band: {c[1]:+.2f} ppm/min; detrended sd {det.std(ddof=1):.2f} ppm, "
      f"se {det.std(ddof=1)/np.sqrt(len(ps)):.2f} ppm")
print(f"  vs zenith-2 same night/focus: mean 2.2068874 arcsec/px (corr ON), "
      f"f2f sd 4.56 ppm over 6 fields")
print(f"  band mean minus zenith-2 ON: {(mean/2.2068874 - 1)*1e6:+.1f} ppm "
      f"(the ~45-90 min of +1.4 ppm/min drift between the sets predicts +60-130 ppm)")

# Part B if ready
fc = sorted(glob.glob(os.path.join(RD, "mosaic", "REFR_M*", "freecubic", "**",
                                   "distortion_results.txt"), recursive=True))
if len(fc) >= 20:
    rows = []
    for f in fc:
        j = json.load(open(f))
        k = int(f.replace("\\", "/").split("REFR_M")[1][:2])
        rows.append((k, abs(d3000(j)), j["#stars used"], j["platescale (arcseconds/pixel)"]))
    b = pd.DataFrame(rows, columns=["k", "d3000_as", "stars", "ps_off"]).merge(
        band[["k", "t_min", "alt_deg"]], on="k")
    m = b.d3000_as.mean()
    print(f"\nPart B -- cubic stability, {len(b)} band fields (free cubic, corrections off):")
    print(f"  d(3000) mean {m:.4f} arcsec; f2f sd {b.d3000_as.std(ddof=1)/m*100:.2f} %; "
          f"se of mean {b.d3000_as.std(ddof=1)/m/np.sqrt(len(b))*100:.3f} %")
    print(f"  vs zenith-2 (same night, same focus): 3.0297 arcsec +/- 1.30 % f2f over 6 fields")
    print(f"  band mean vs zenith-2 mean: {(m/3.0297 - 1)*100:+.2f} %")
    Ac = np.column_stack([np.ones(len(b)), b.t_min - b.t_min.mean()])
    cc, *_ = np.linalg.lstsq(Ac, (b.d3000_as/m - 1)*100, rcond=None)
    print(f"  cubic drift within band: {cc[1]*60:+.3f} %/h; "
          f"alt-correlation r = {np.corrcoef(b.alt_deg, b.d3000_as)[0,1]:+.2f}")
    b.to_csv(os.path.join(RD, "band_cubic_results.csv"), index=False)
else:
    print(f"\nPart B pending: {len(fc)}/27 free-cubic fields done")
