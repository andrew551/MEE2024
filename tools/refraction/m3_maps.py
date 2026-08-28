"""M3: residual maps from the M2 per-frame ladder.

For each of the nine field-windows: per-star quasi-static residual (median over the ~45
frames; what stacking cannot remove) and per-frame jitter (scatter about it; what stacking
averages), rotated into the local alt-az frame via an empirical affine fitted from one
mid-block solve. Outputs: a 3x3 quiver-map figure, a vertical-profile figure, and a stats
table. Input residuals are from the corrections-ON quadratic-free fits, so the maps show
the structure the step-2-style fit could NOT absorb -- cubic-and-above model error plus
quasi-static atmosphere.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

RD = r"D:/MEE2024 output/MEE_output/refraction"
OUT = os.path.join(RD, "m3_maps")
os.makedirs(OUT, exist_ok=True)
SITE = EarthLocation(lat=42.740470 * u.deg, lon=-5.613780 * u.deg, height=1101 * u.m)
NX, NY = 6248, 4176
WINDOWS, FIELDS = ("N1", "N2", "N3"), ("H1", "H2", "H3")
DATE = {"N1": "2026-08-11", "N2": "2026-08-12", "N3": "2026-08-13"}

def altaz_basis(w, f):
    """Unit vectors (in sensor px basis) of increasing altitude and azimuth*cos(alt),
    plus alt(px,py) in degrees -- from one mid-block solved frame."""
    cal = sorted(glob.glob(os.path.join(RD, "perframe", w, f, "f20", "corr_on", "**",
                                        "CATALOGUE_MATCHED_ERRORS.csv"), recursive=True))
    resf = sorted(glob.glob(os.path.join(RD, "perframe", w, f, "f20", "corr_on", "**",
                                         "distortion_results.txt"), recursive=True))
    d = pd.read_csv(cal[0])
    j = json.load(open(resf[0]))
    t = Time(f"{DATE[w]}T{j['observation_time (UTC)']}", scale="utc")
    aa = SkyCoord(d["RA(catalog)"].values * u.deg,
                  d["DEC(catalog)"].values * u.deg).transform_to(
                      AltAz(obstime=t, location=SITE))
    alt = aa.alt.deg
    azc = aa.az.deg * np.cos(np.radians(alt.mean()))
    # affine (azc, alt) -> (px, py)
    A = np.column_stack([azc - azc.mean(), alt - alt.mean(), np.ones(len(d))])
    cx, *_ = np.linalg.lstsq(A, d.px.values, rcond=None)
    cy, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    v_az = np.array([cx[0], cy[0]]); v_az /= np.linalg.norm(v_az)
    v_alt = np.array([cx[1], cy[1]]); v_alt /= np.linalg.norm(v_alt)
    # inverse for alt(px,py)
    B = np.column_stack([d.px.values - NX / 2, d.py.values - NY / 2, np.ones(len(d))])
    ca, *_ = np.linalg.lstsq(B, alt, rcond=None)
    return v_alt, v_az, (lambda px, py: np.column_stack(
        [px - NX / 2, py - NY / 2, np.ones(len(px))]) @ ca)

stats, prof = [], {}
fig_q, axq = plt.subplots(3, 3, figsize=(16.5, 11), sharex=True, sharey=True)
for i, w in enumerate(WINDOWS):
    for k, f in enumerate(FIELDS):
        files = sorted(glob.glob(os.path.join(RD, "perframe", w, f, "f*", "corr_on", "**",
                                              "TWOD_RESIDUALS.csv"), recursive=True))
        acc = {}
        for fp in files:
            d = pd.read_csv(fp)
            for _, r in d.iterrows():
                acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec))
        rowsd = []
        for v in acc.values():
            if len(v) < 20:
                continue
            a = np.array(v)
            rowsd.append([np.median(a[:, 0]), np.median(a[:, 1]),
                          np.median(a[:, 2]), np.median(a[:, 3]),
                          1.4826 * np.median(np.abs(a[:, 2] - np.median(a[:, 2]))),
                          1.4826 * np.median(np.abs(a[:, 3] - np.median(a[:, 3]))),
                          len(a)])
        P = np.array(rowsd)
        px, py, qx, qy, jx, jy, nfr = P.T
        qx, qy = qx - np.median(qx), qy - np.median(qy)
        mag = np.hypot(qx, qy)
        lim = max(3 * 1.4826 * np.median(np.abs(mag - np.median(mag))) + np.median(mag), 2.5)
        good = mag < lim
        n_clip = int((~good).sum())
        px, py, qx, qy, jx, jy = (a[good] for a in (px, py, qx, qy, jx, jy))

        v_alt, v_az, alt_of = altaz_basis(w, f)
        q_alt = qx * v_alt[0] + qy * v_alt[1]
        q_az = qx * v_az[0] + qy * v_az[1]
        j_alt = np.median(np.abs(jx * v_alt[0] + jy * v_alt[1]))
        j_az = np.median(np.abs(jx * v_az[0] + jy * v_az[1]))
        alt_star = alt_of(px, py)

        qs_rms = np.sqrt(np.mean(qx**2 + qy**2))
        jit = np.median(np.hypot(jx, jy))
        stats.append(dict(window=w, field=f, n_stars=len(px), n_clipped=n_clip,
            alt_mean_deg=round(float(np.mean(alt_star)), 2),
            qs_rms_arcsec=round(float(qs_rms), 3),
            qs_rms_alt_arcsec=round(float(np.sqrt(np.mean(q_alt**2))), 3),
            qs_rms_az_arcsec=round(float(np.sqrt(np.mean(q_az**2))), 3),
            jitter_med_arcsec_per_frame=round(float(jit), 3),
            jitter_alt_arcsec=round(float(j_alt), 3),
            jitter_az_arcsec=round(float(j_az), 3),
            stack45_jitter_arcsec=round(float(jit / np.sqrt(45)), 3)))

        ax = axq[i, k]
        ax.quiver(px, py, qx, qy, angles="xy", scale_units="xy",
                  scale=0.0018, width=0.003, color="#1f77b4")
        ax.quiver([400], [3900], [1.0], [0.0], angles="xy", scale_units="xy",
                  scale=0.0018, width=0.004, color="crimson")
        ax.text(430, 3650, '1"', color="crimson", fontsize=9)
        va = v_alt * 600
        ax.annotate("", xy=(5600 + va[0], 3700 + va[1]), xytext=(5600, 3700),
                    arrowprops=dict(arrowstyle="->", color="green"))
        ax.text(5300, 3350, "up", color="green", fontsize=8)
        ax.set_title(f"{w}/{f}  alt {np.mean(alt_star):.1f} deg  "
                     f"qs rms {qs_rms:.2f}\"  clip {n_clip}", fontsize=10)
        ax.set_xlim(0, NX); ax.set_ylim(NY, 0)

        b = np.linspace(alt_star.min(), alt_star.max(), 9)
        ib = np.digitize(alt_star, b)
        xs = [alt_star[ib == m].mean() for m in range(1, 9) if (ib == m).sum() > 3]
        ys = [q_alt[ib == m].mean() for m in range(1, 9) if (ib == m).sum() > 3]
        es = [q_alt[ib == m].std(ddof=1) / np.sqrt((ib == m).sum())
              for m in range(1, 9) if (ib == m).sum() > 3]
        prof[(w, f)] = (xs, ys, es)

fig_q.suptitle("M3: quasi-static residual maps (median over ~45 frames, corrections ON, "
               "quadratic-free) -- sensor px axes, arrows in arcsec", fontsize=12)
fig_q.tight_layout()
fig_q.savefig(os.path.join(OUT, "m3_quiver_maps.png"), dpi=110)

fig_p, axp = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for i, w in enumerate(WINDOWS):
    for f, c in zip(FIELDS, ("#d62728", "#1f77b4", "#2ca02c")):
        xs, ys, es = prof[(w, f)]
        axp[i].errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1, color=c, label=f)
    axp[i].axhline(0, color="k", lw=0.5)
    axp[i].set_title(f"{w}"); axp[i].set_xlabel("star altitude (deg)")
    axp[i].legend(fontsize=8)
axp[0].set_ylabel("quasi-static residual, vertical component (arcsec)")
fig_p.suptitle("M3: vertical residual vs altitude within the frame -- the structure the "
               "quadratic-free fit could not absorb", fontsize=11)
fig_p.tight_layout()
fig_p.savefig(os.path.join(OUT, "m3_vertical_profiles.png"), dpi=110)

df = pd.DataFrame(stats)
df.to_csv(os.path.join(OUT, "m3_stats.csv"), index=False)
print(df.to_string(index=False))
print(f"\nCAL_piLeo daytime for comparison (stacked, step-2 doc): rms 0.53 arcsec, "
      f"sigma vertical 0.32 / horizontal 0.42 arcsec")
print(f"figures -> {OUT}")
