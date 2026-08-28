"""(1) Redrawn plate-scale-vs-T figure: no flipped fields, night-1 points added, tight
scale. (2) Corrected cubic-vs-T figure (the earlier one had a time-wrap bug putting band
points at ~29 C). (3) The two decomposition tests: can temperature explain the night-1
plate scale, and the CAL_piLeo daytime plate scale?"""
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
REF = 2.2068828     # arcsec/px, night-2 zenith corrections-ON mean (6 fields)

def T_at(t):
    return float(np.interp(t.replace(tzinfo=datetime.timezone.utc).timestamp(),
                           logger[:, 0], logger[:, 1]))

def zen_points(night, tag):
    out = []
    for fld in ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right",
                "Z5_mid_right", "Z6_bottom_right"):
        if night == "08-12" and fld == "Z1_base":
            r = glob.glob(os.path.join(r"D:/MEE2024 output/MEE_output/cal_pileo_step2",
                          "zenith_0812_Z1", "corr_on", "**", "distortion_results.txt"),
                          recursive=True)
        else:
            r = glob.glob(os.path.join(RD, "zenith12", f"{night}_{fld}", "corr_on", "**",
                                       "distortion_results.txt"), recursive=True)
        j = json.load(open(r[0]))
        fs = sorted(glob.glob(rf"G:/Leon Aug 2026/2026-{night}/Zenith/{fld}/*/*.fits"))
        ts = [datetime.datetime.fromisoformat(fits.getheader(f)["DATE-OBS"])
              for f in (fs[0], fs[-1])]
        t = ts[0] + (ts[1] - ts[0]) / 2
        out.append((T_at(t), j["platescale (arcseconds/pixel)"], tag))
    return out

pts = zen_points("08-12", "zenith night 2") + zen_points("08-11", "zenith night 1 (box T)")
df = pd.read_csv(os.path.join(RD, "m4_mosaic", "m4_fields.csv"))
for _, r in df[(df.alt_deg >= 30) & (~df.flipped)].iterrows():
    fld = f"REFR_M{int(r.k):02d}_az{int(r.az_nom_deg):03d}_alt{r.alt_nom_deg:05.2f}"
    j = json.load(open(glob.glob(os.path.join(RD, "mosaic", fld, "corr_on", "**",
                      "distortion_results.txt"), recursive=True)[0]))
    hh, mm, ss = j["observation_time (UTC)"].split(":")
    t = datetime.datetime(2026, 8, 12 if int(hh) >= 22 else 13, int(hh), int(mm), int(ss))
    pts.append((T_at(t), r.ps_on, "mosaic (unflipped, alt >= 30)"))
P = pd.DataFrame(pts, columns=["T_C", "ps", "kind"])
P["dev_ppm"] = (P.ps / REF - 1) * 1e6

n2 = P[P.kind != "zenith night 1 (box T)"]
c = np.polyfit(n2.T_C, n2.dev_ppm, 1)
fig, ax = plt.subplots(figsize=(9, 6))
for kind, mk, col, ms in (("zenith night 2", "o", "crimson", 8),
                          ("mosaic (unflipped, alt >= 30)", "s", "#1f77b4", 5),
                          ("zenith night 1 (box T)", "D", "#2ca02c", 8)):
    s = P[P.kind == kind]
    ax.plot(s.T_C, s.dev_ppm, mk, ms=ms, color=col, label=f"{kind} (n={len(s)})",
            alpha=0.85)
xs = np.linspace(P.T_C.min() - 0.1, P.T_C.max() + 0.1, 10)
ax.plot(xs, np.polyval(c, xs), "k--", lw=1.2,
        label=f"night-2 fit: {c[0]:+.1f} ppm/K")
ax.set_xlabel("logger temperature (deg C)")
ax.set_ylabel("corrections-ON plate scale vs night-2 zenith mean (ppm)")
ax.set_title("plate scale vs temperature -- nights 1 and 2, no flipped fields")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(RD, "platescale_vs_temperature.png"), dpi=120)

# corrected cubic figure
cb = pd.read_csv(os.path.join(RD, "band_cubic_results.csv"))
cb["m22"] = [(m - 1320) if m >= 1320 else (m + 120) for m in cb.t_min]
cb["T_C"] = [T_at(datetime.datetime(2026, 8, 12, 22, 0) + datetime.timedelta(minutes=float(m)))
             for m in cb.m22]
H = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
zc = []
for night, Tz in (("08-12", None), ("08-11", None)):
    for fld in ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right",
                "Z5_mid_right", "Z6_bottom_right"):
        j = json.load(open(os.path.join(H, f"{night}_{fld}.txt")))
        fs = sorted(glob.glob(rf"G:/Leon Aug 2026/2026-{night}/Zenith/{fld}/*/*.fits"))
        ts = [datetime.datetime.fromisoformat(fits.getheader(f)["DATE-OBS"])
              for f in (fs[0], fs[-1])]
        t = ts[0] + (ts[1] - ts[0]) / 2
        zc.append((T_at(t), abs(d3000(j)), night))
zc = pd.DataFrame(zc, columns=["T_C", "d3000_as", "night"])
fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(cb.T_C, cb.d3000_as, "s", ms=5, color="#1f77b4",
         label=f"mosaic band 50-80 deg (n={len(cb)}), night 2")
s = zc[zc.night == "08-12"]
ax2.plot(s.T_C, s.d3000_as, "o", ms=8, color="crimson", label="zenith night 2 (n=6)")
s = zc[zc.night == "08-11"]
ax2.plot(s.T_C, s.d3000_as, "D", ms=8, color="#2ca02c", label="zenith night 1 (box T, n=6)")
ax2.set_xlabel("logger temperature (deg C)")
ax2.set_ylabel("free-cubic d(3000) (arcsec)")
ax2.set_title("cubic vs temperature -- corrected time axis (the earlier figure's band\n"
              "points sat at ~29 deg C from a time-wrap bug; these are the true values)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(os.path.join(RD, "cubic_vs_temperature.png"), dpi=120)

# decomposition tests
m1 = P[P.kind == "zenith night 1 (box T)"].ps.mean()
gap = (m1 / REF - 1) * 1e6
print(f"night-1 corr-ON mean: {m1:.7f} arcsec/px -> gap vs night-2: {gap:+.1f} ppm")
for lab, T1 in (("box reading trusted (24.25 C)", 24.25),
                ("box runs warm per sec 19.3 (air ~22.15 C)", 22.15)):
    dT = T1 - 23.5
    pred = c[0] * dT
    print(f"  {lab}: dT = {dT:+.2f} K -> thermal prediction {pred:+.1f} ppm; "
          f"unexplained {gap - pred:+.1f} ppm")
print(f"  (+8 focuser steps also separate the nights, inside the ~15-step backlash)")

CAL, T_CAL, DSTEP = 2.2054197, 30.5, 129
dcal = (CAL / REF - 1) * 1e6
for lab, Td in (("assumed 30.5 C", 30.5), ("FOCTEMP-corrected ~28.5 C", 28.5)):
    tshare = c[0] * (Td - 23.5)
    rest = dcal - tshare
    print(f"\nCAL_piLeo vs night-2 zenith: {dcal:+.1f} ppm total; day T {lab}:")
    print(f"  temperature share {tshare:+.1f} ppm ({tshare/dcal*100:.0f} %); "
          f"residual {rest:+.1f} ppm over +129 steps = {rest/DSTEP:+.2f} ppm/step focus coupling")
print(f"\nEFL language: {c[0]:+.1f} ppm/K of plate scale = {-c[0]:+.1f} ppm/K of EFL "
      f"= {-c[0]*363.5e3/1e6:+.1f} um/K on the 363.5 mm train "
      f"(aluminium tube CTE alone: ~+8.4 um/K)")
