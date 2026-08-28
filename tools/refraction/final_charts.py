"""The canonical charts, current state (16.5): plate scale and cubic, each against BOTH
thermometers -- air logger (amplitude anchor) and FOCTEMP (phase anchor). Night-1 points
shown as the transport-step demonstration, excluded from fits."""
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
REF = 2.2068828

def airT(t):
    return float(np.interp(t.replace(tzinfo=datetime.timezone.utc).timestamp(),
                           logger[:, 0], logger[:, 1]))

def zrow(night, fld):
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
    ft = float(fits.getheader(fs[0])["FOCTEMP"])
    return (airT(ts[0] + (ts[1]-ts[0])/2), ft,
            (j["platescale (arcseconds/pixel)"]/REF - 1)*1e6)

Z = ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right", "Z5_mid_right",
     "Z6_bottom_right")
n2 = [zrow("08-12", f) for f in Z]
n1 = [zrow("08-11", f) for f in Z]
df = pd.read_csv(os.path.join(RD, "m4_mosaic", "m4_fields.csv"))
mo = []
for _, r in df[(df.alt_deg >= 30) & (~df.flipped)].iterrows():
    j = json.load(open(glob.glob(os.path.join(RD, "mosaic",
        f"REFR_M{int(r.k):02d}_az{int(r.az_nom_deg):03d}_alt{r.alt_nom_deg:05.2f}",
        "corr_on", "**", "distortion_results.txt"), recursive=True)[0]))
    hh, mm, ss = j["observation_time (UTC)"].split(":")
    t = datetime.datetime(2026, 8, 12 if int(hh) >= 22 else 13, int(hh), int(mm), int(ss))
    f0 = glob.glob(rf"G:/Leon Aug 2026/2026-08-1[23]/Refraction mosaic/"
                   rf"REFR_M{int(r.k):02d}_*/*/*00001.fits")[0]
    mo.append((airT(t), float(fits.getheader(f0)["FOCTEMP"]),
               (r.ps_on/REF - 1)*1e6))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for col, xl, idx in ((0, "air logger temperature (deg C)", 0),
                     (1, "FOCTEMP (deg C)", 1)):
    a = ax[col]
    fitpts = np.array([(p[idx], p[2]) for p in n2 + mo])
    c = np.polyfit(fitpts[:, 0], fitpts[:, 1], 1)
    rr = np.corrcoef(fitpts[:, 0], fitpts[:, 1])[0, 1]
    n = len(fitpts)
    se = np.sqrt(np.sum((fitpts[:, 1] - np.polyval(c, fitpts[:, 0]))**2)/(n-2)
                 / np.sum((fitpts[:, 0] - fitpts[:, 0].mean())**2))
    a.plot([p[idx] for p in mo], [p[2] for p in mo], "s", ms=5, color="#1f77b4",
           label=f"mosaic, unflipped alt>=30 (n={len(mo)})")
    a.plot([p[idx] for p in n2], [p[2] for p in n2], "o", ms=8, color="crimson",
           label="zenith night 2 (n=6)")
    a.plot([p[idx] for p in n1], [p[2] for p in n1], "D", ms=7, color="#2ca02c",
           alpha=0.7, label="zenith night 1 (transport step; not fitted)")
    xs = np.linspace(fitpts[:, 0].min()-0.05, fitpts[:, 0].max()+0.05, 10)
    a.plot(xs, np.polyval(c, xs), "k--", lw=1.2,
           label=f"fit: {c[0]:+.1f} $\pm$ {se:.1f} ppm/K,  r = {rr:+.2f}")
    a.set_xlabel(xl); a.legend(fontsize=8); a.grid(alpha=0.25)
ax[0].set_ylabel("corrections-ON plate scale vs night-2 zenith mean (ppm)")
fig.suptitle("plate scale against the two thermometers -- air carries the amplitude "
             "(aluminium x 535 mm arithmetic), FOCTEMP the phase", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(RD, "platescale_vs_temperature.png"), dpi=120)

# cubic chart
cb = pd.read_csv(os.path.join(RD, "band_cubic_results.csv"))
cb["m22"] = [(m - 1320) if m >= 1320 else (m + 120) for m in cb.t_min]
cb["airT"] = [airT(datetime.datetime(2026, 8, 12, 22, 0)
              + datetime.timedelta(minutes=float(m))) for m in cb.m22]
fts = []
for _, r in cb.iterrows():
    f0 = glob.glob(rf"G:/Leon Aug 2026/2026-08-1[23]/Refraction mosaic/"
                   rf"REFR_M{int(r.k):02d}_*/*/*00001.fits")[0]
    fts.append(float(fits.getheader(f0)["FOCTEMP"]))
cb["ft"] = fts
H = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
zc = []
for night in ("08-12", "08-11"):
    for fld in Z:
        j = json.load(open(os.path.join(H, f"{night}_{fld}.txt")))
        fs = sorted(glob.glob(rf"G:/Leon Aug 2026/2026-{night}/Zenith/{fld}/*/*.fits"))
        ts = [datetime.datetime.fromisoformat(fits.getheader(f)["DATE-OBS"])
              for f in (fs[0], fs[-1])]
        zc.append((night, airT(ts[0]+(ts[1]-ts[0])/2),
                   float(fits.getheader(fs[0])["FOCTEMP"]), abs(d3000(j))))
zc = pd.DataFrame(zc, columns=["night", "airT", "ft", "d3000"])
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for col, xl, key in ((0, "air logger temperature (deg C)", "airT"),
                     (1, "FOCTEMP (deg C)", "ft")):
    a = ax2[col]
    a.plot(cb[key if key != "airT" else "airT"], cb.d3000_as, "s", ms=5,
           color="#1f77b4", label=f"mosaic band 50-80 deg (n={len(cb)}), night 2")
    s = zc[zc.night == "08-12"]
    a.plot(s[key], s.d3000, "o", ms=8, color="crimson", label="zenith night 2 (n=6)")
    s = zc[zc.night == "08-11"]
    a.plot(s[key], s.d3000, "D", ms=7, color="#2ca02c", alpha=0.7,
           label="zenith night 1 (pre-transport optic; context only)")
    rin = np.corrcoef(cb[key if key != "airT" else "airT"], cb.d3000_as)[0, 1]
    a.set_xlabel(xl); a.grid(alpha=0.25)
    a.annotate(f"within-band r = {rin:+.2f} (no tracking)\n"
               f"the -7.3 % step between sequences\ncarries the information",
               xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9)
    a.legend(fontsize=8)
ax2[0].set_ylabel("free-cubic d(3000) (arcsec)")
fig2.suptitle("cubic against the two thermometers -- the step, not a trend; thermal via "
              "the shared lever at ~0.27 %/step is leading but unproven (sec 16.5)",
              fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(RD, "cubic_vs_temperature.png"), dpi=120)
print("both charts regenerated")
