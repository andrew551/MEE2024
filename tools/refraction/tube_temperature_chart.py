"""The single canonical stability chart: plate scale and cubic against the normalised
tube temperature -- FOCTEMP minus its measured +7.74 K offset to the calibrated free-air
logger (Douglas' construction: the uncalibrated but telescope-coupled sensor's dynamics,
on the calibrated instrument's scale)."""
import datetime, glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools/refraction")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_stability import d3000
from astropy.io import fits

RD = r"D:/MEE2024 output/MEE_output/refraction"
REF = 2.2068828
OFF = 7.74            # K, FOCTEMP minus free-air logger, night-2 measured (+/- 0.67)

def ft_of(pattern):
    return float(fits.getheader(sorted(glob.glob(pattern))[0])["FOCTEMP"]) - OFF

Z = ("Z1_base", "Z2_mid_left", "Z3_top_left", "Z4_top_right", "Z5_mid_right",
     "Z6_bottom_right")
ps_pts, cu_pts = [], []
for night, tag in (("08-12", "n2"), ("08-11", "n1")):
    for fld in Z:
        if night == "08-12" and fld == "Z1_base":
            r = glob.glob(os.path.join(r"D:/MEE2024 output/MEE_output/cal_pileo_step2",
                          "zenith_0812_Z1", "corr_on", "**", "distortion_results.txt"),
                          recursive=True)
        else:
            r = glob.glob(os.path.join(RD, "zenith12", f"{night}_{fld}", "corr_on", "**",
                                       "distortion_results.txt"), recursive=True)
        j = json.load(open(r[0]))
        t = ft_of(rf"G:/Leon Aug 2026/2026-{night}/Zenith/{fld}/*/*00001.fits")
        ps_pts.append((t, (j["platescale (arcseconds/pixel)"]/REF - 1)*1e6, tag))
        jz = json.load(open(rf"D:/MEE2024 output/MEE_output/Claude Code/"
                            rf"HANDOFF_zenith_cubic/inpipeline_windowed/{night}_{fld}.txt"))
        cu_pts.append((t, abs(d3000(jz)), tag))
df = pd.read_csv(os.path.join(RD, "m4_mosaic", "m4_fields.csv"))
for _, r in df[(df.alt_deg >= 30) & (~df.flipped)].iterrows():
    j = json.load(open(glob.glob(os.path.join(RD, "mosaic",
        f"REFR_M{int(r.k):02d}_az{int(r.az_nom_deg):03d}_alt{r.alt_nom_deg:05.2f}",
        "corr_on", "**", "distortion_results.txt"), recursive=True)[0]))
    t = ft_of(rf"G:/Leon Aug 2026/2026-08-1[23]/Refraction mosaic/"
              rf"REFR_M{int(r.k):02d}_*/*/*00001.fits")
    ps_pts.append((t, (r.ps_on/REF - 1)*1e6, "mosaic"))
cb = pd.read_csv(os.path.join(RD, "band_cubic_results.csv"))
for _, r in cb.iterrows():
    t = ft_of(rf"G:/Leon Aug 2026/2026-08-1[23]/Refraction mosaic/"
              rf"REFR_M{int(r.k):02d}_*/*/*00001.fits")
    cu_pts.append((t, r.d3000_as, "band"))

fig, ax = plt.subplots(2, 1, figsize=(9.5, 10), sharex=True)
STYLE = {"mosaic": ("s", "#1f77b4", 5, "mosaic, unflipped alt>=30"),
         "band":   ("s", "#1f77b4", 5, "mosaic band 50-80 deg"),
         "n2":     ("o", "crimson", 8, "zenith night 2"),
         "n1":     ("D", "#2ca02c", 7, "zenith night 1 (transport step; not fitted)")}
P = pd.DataFrame(ps_pts, columns=["t", "y", "k"])
fit = P[P.k != "n1"]
c = np.polyfit(fit.t, fit.y, 1)
rr = np.corrcoef(fit.t, fit.y)[0, 1]
se = np.sqrt(np.sum((fit.y - np.polyval(c, fit.t))**2)/(len(fit)-2)
             / np.sum((fit.t - fit.t.mean())**2))
for k in ("mosaic", "n2", "n1"):
    s = P[P.k == k]; mk, col, ms, lab = STYLE[k]
    ax[0].plot(s.t, s.y, mk, ms=ms, color=col, alpha=0.85, label=f"{lab} (n={len(s)})")
xs = np.linspace(P.t.min()-0.05, P.t.max()+0.05, 10)
ax[0].plot(xs, np.polyval(c, xs), "k--", lw=1.2,
           label=f"fit: {c[0]:+.1f} $\pm$ {se:.1f} ppm/K,  r = {rr:+.2f}")
ax[0].set_ylabel("corrections-ON plate scale vs night-2 zenith mean (ppm)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
C = pd.DataFrame(cu_pts, columns=["t", "y", "k"])
for k in ("band", "n2", "n1"):
    s = C[C.k == k]; mk, col, ms, lab = STYLE[k]
    ax[1].plot(s.t, s.y, mk, ms=ms, color=col, alpha=0.85, label=f"{lab} (n={len(s)})")
band = C[C.k == "band"]
rin = np.corrcoef(band.t, band.y)[0, 1]
ax[1].annotate(f"within-band r = {rin:+.2f}; the -7.3 % step between the sequences\n"
               f"carries the information (thermal at ~0.27 %/step leading, unproven)",
               xy=(0.03, 0.05), xycoords="axes fraction", fontsize=9)
ax[1].set_ylabel("free-cubic d(3000) (arcsec)")
ax[1].set_xlabel("tube temperature: FOCTEMP $-$ 7.74 K "
                 "(air-equivalent; offset measured $\pm$ 0.67 K, night 2)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
fig.suptitle("Leon FRA500 + 0.7x train stability, night 2: plate scale and cubic vs the "
             "normalised tube temperature", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(RD, "stability_vs_tube_temperature.png"), dpi=120)
print(f"chart written; plate-scale fit {c[0]:+.1f} +/- {se:.1f} ppm/K, r = {rr:+.2f}; "
      f"x-axis span {P.t.min():.1f}-{P.t.max():.1f} C air-equivalent")
