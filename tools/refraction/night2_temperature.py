"""Night 2: plate scale and cubic against logged temperature (Douglas' hypothesis).

All corrections-ON plate scales (refraction removed with each point's own logger weather,
so what remains is the optics), FOCUSPOS 17041 unless marked. Cubic: free-cubic d(3000).
Predictions from two independently measured couplings: the train moves 5-9 focuser
steps/K (sec 12.4) and the cubic ~0.64 %/step (sec 9.2) -> 3.2-5.8 %/K of cubic; the
plate scale ~11 ppm/step (sec 12.2 class) -> 55-100 ppm/K, at a FIXED focuser.
"""
import datetime, glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools/refraction")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_mosaic import load_logger
from band_stability import d3000
from astropy.io import fits

RD = r"D:/MEE2024 output/MEE_output/refraction"
logger = load_logger()

def T_at(t_utc):
    return float(np.interp(t_utc.replace(tzinfo=datetime.timezone.utc).timestamp(),
                           logger[:, 0], logger[:, 1]))

def tmin(t):  # minutes past 22:00 UTC on 08-12
    m = (t - datetime.datetime(2026, 8, 12, 22, 0)).total_seconds() / 60
    return m

pts = []   # (t_min, T, ps_on, kind)
# zenith corr-ON (6)
for fld in ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right",
            "Z5_mid_right", "Z6_bottom_right"):
    if fld == "Z1_base":
        r = glob.glob(os.path.join(r"D:/MEE2024 output/MEE_output/cal_pileo_step2",
                                   "zenith_0812_Z1", "corr_on", "**",
                                   "distortion_results.txt"), recursive=True)
    else:
        r = glob.glob(os.path.join(RD, "zenith12", f"08-12_{fld}", "corr_on", "**",
                                   "distortion_results.txt"), recursive=True)
    j = json.load(open(r[0]))
    fs = sorted(glob.glob(rf"G:/Leon Aug 2026/2026-08-12/Zenith/{fld}/*/*.fits"))
    ts = [datetime.datetime.fromisoformat(fits.getheader(f)["DATE-OBS"]) for f in (fs[0], fs[-1])]
    t = ts[0] + (ts[1] - ts[0]) / 2
    pts.append((tmin(t), T_at(t), j["platescale (arcseconds/pixel)"], "zenith"))
# mosaic corr-ON, alt >= 30
df = pd.read_csv(os.path.join(RD, "m4_mosaic", "m4_fields.csv"))
for _, r in df[df.alt_deg >= 30].iterrows():
    fld = f"REFR_M{int(r.k):02d}_az{int(r.az_nom_deg):03d}_alt{r.alt_nom_deg:05.2f}"
    res = glob.glob(os.path.join(RD, "mosaic", fld, "corr_on", "**",
                                 "distortion_results.txt"), recursive=True)
    j = json.load(open(res[0]))
    hh, mm, ss = j["observation_time (UTC)"].split(":")
    t = datetime.datetime(2026, 8, 12 if int(hh) >= 22 else 13, int(hh), int(mm), int(ss))
    pts.append((tmin(t), T_at(t), r.ps_on, "flipped" if r.flipped else "mosaic"))
P = pd.DataFrame(pts, columns=["t_min", "T_C", "ps", "kind"])
P["dev_ppm"] = (P.ps / 2.2068874 - 1) * 1e6

fit = P[P.kind.isin(("zenith", "mosaic"))]
cT = np.polyfit(fit.T_C, fit.dev_ppm, 1)
rT = np.corrcoef(fit.T_C, fit.dev_ppm)[0, 1]
ct = np.polyfit(fit.t_min, fit.dev_ppm, 1)
rt = np.corrcoef(fit.t_min, fit.dev_ppm)[0, 1]
# joint (T + time) -- the bumps in T(t) give the leverage to separate them
A = np.column_stack([np.ones(len(fit)), fit.T_C - fit.T_C.mean(),
                     fit.t_min - fit.t_min.mean()])
cj, res_, *_ = np.linalg.lstsq(A, fit.dev_ppm.values, rcond=None)
sig = np.sqrt(float(res_[0]) / (len(fit) - 3))
cov = np.linalg.inv(A.T @ A) * sig**2
print(f"PLATE SCALE, {len(fit)} points (6 zenith + {len(fit)-6} unflipped mosaic alt>=30):")
print(f"  vs T alone:    {cT[0]:+7.1f} ppm/K   (r = {rT:+.2f})")
print(f"  vs time alone: {ct[0]*60:+7.1f} ppm/h  (r = {rt:+.2f})")
print(f"  joint:  dps/dT = {cj[1]:+.1f} +/- {np.sqrt(cov[1,1]):.1f} ppm/K ;  "
      f"dps/dt = {cj[2]*60:+.1f} +/- {np.sqrt(cov[2,2])*60:.1f} ppm/h ;  scatter {sig:.1f} ppm")
print(f"  prediction (5-9 steps/K x ~11 ppm/step, fixed focuser): -55 to -100 ppm/K")

# cubic vs T
cb = pd.read_csv(os.path.join(RD, "band_cubic_results.csv"))
cb["T_C"] = [float(np.interp((datetime.datetime(2026, 8, 12, 22, 0)
             + datetime.timedelta(minutes=m - 1440 if m > 1300 else m)
             ).replace(tzinfo=datetime.timezone.utc).timestamp()
             + (86400 if m > 1300 else 0), logger[:, 0], logger[:, 1]))
             for m in cb.t_min]
zen = []
H = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
for fld in ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right",
            "Z5_mid_right", "Z6_bottom_right"):
    j = json.load(open(os.path.join(H, f"08-12_{fld}.txt")))
    fs = sorted(glob.glob(rf"G:/Leon Aug 2026/2026-08-12/Zenith/{fld}/*/*.fits"))
    ts = [datetime.datetime.fromisoformat(fits.getheader(f)["DATE-OBS"]) for f in (fs[0], fs[-1])]
    t = ts[0] + (ts[1] - ts[0]) / 2
    zen.append((T_at(t), abs(d3000(j))))
zen = pd.DataFrame(zen, columns=["T_C", "d3000_as"])
allc = pd.concat([zen.assign(kind="zenith"),
                  cb[["T_C", "d3000_as"]].assign(kind="band")])
cc = np.polyfit(allc.T_C, allc.d3000_as / allc.d3000_as.mean() * 100, 1)
rc = np.corrcoef(allc.T_C, allc.d3000_as)[0, 1]
print(f"\nCUBIC d(3000), {len(allc)} points (6 zenith + {len(cb)} band):")
print(f"  vs T: {cc[0]:+.2f} %/K   (r = {rc:+.2f})")
print(f"  prediction (5-9 steps/K x 0.64 %/step, fixed focuser): +3.2 to +5.8 %/K")

# figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
tt = np.linspace(30, 160, 400)
ax[0].plot((logger[:, 0] - datetime.datetime(2026, 8, 12, 22, 0,
           tzinfo=datetime.timezone.utc).timestamp()) / 60, logger[:, 1], "-", lw=1)
ax[0].set_xlim(20, 165); ax[0].set_ylim(20.4, 24.2)
ax[0].set_xlabel("minutes past 22:00 UTC, 2026-08-12"); ax[0].set_ylabel("logger T (deg C)")
for lo, hi, lab in ((48, 70, "zenith set"), (70, 140, "mosaic"), (142, 157, "N3 horizon")):
    ax[0].axvspan(lo, hi, alpha=0.12)
    ax[0].text((lo+hi)/2, 24.0, lab, ha="center", fontsize=8)
ax[0].set_title("the night-2 temperature curve (spreader, free air)")
for kind, mk, col in (("zenith", "o", "crimson"), ("mosaic", "s", "#1f77b4"),
                      ("flipped", "^", "#999999")):
    s = P[P.kind == kind]
    ax[1].plot(s.T_C, s.dev_ppm, mk, ms=5, color=col, label=kind)
xs = np.linspace(P.T_C.min(), P.T_C.max(), 10)
ax[1].plot(xs, np.polyval(cT, xs), "k--", lw=1,
           label=f"{cT[0]:+.0f} ppm/K (r={rT:+.2f})")
ax[1].set_xlabel("logger T (deg C)"); ax[1].set_ylabel("plate scale vs zenith ref (ppm)")
ax[1].legend(fontsize=8); ax[1].set_title("corrections-ON plate scale vs temperature")
for kind, mk, col in (("zenith", "o", "crimson"), ("band", "s", "#1f77b4")):
    s = allc[allc.kind == kind]
    ax[2].plot(s.T_C, s.d3000_as, mk, ms=5, color=col, label=kind)
xs = np.linspace(allc.T_C.min(), allc.T_C.max(), 10)
m0 = allc.d3000_as.mean()
ax[2].plot(xs, m0 * (1 + cc[0]/100 * (xs - allc.T_C.mean())), "k--", lw=1,
           label=f"{cc[0]:+.2f} %/K (r={rc:+.2f})")
ax[2].set_xlabel("logger T (deg C)"); ax[2].set_ylabel("free-cubic d(3000) (arcsec)")
ax[2].legend(fontsize=8); ax[2].set_title("cubic vs temperature")
fig.tight_layout()
fig.savefig(os.path.join(RD, "night2_temperature.png"), dpi=115)
print(f"\nfigure -> {RD}/night2_temperature.png")
