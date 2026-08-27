"""F19 coma test: radial FWHM growth, night 1 vs night 2, all 12 zenith stacks.
Adaptive Gaussian-weighted moments per star (the M3 method), binned by field radius."""
import glob, os, sys, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

RD = r"D:/MEE2024 output/MEE_output/refraction"
NX, NY, PS = 6248, 4176, 2.2054

def adaptive(cut, n=25):
    k = cut.shape[0] // 2
    yy, xx = np.mgrid[-k:k+1, -k:k+1]
    r = np.hypot(xx, yy)
    w0 = cut - np.median(cut[(r > k-3) & (r <= k)])
    mx = my = 0.0; vxx = vyy = 4.0; vxy = 0.0
    for _ in range(n):
        det = vxx*vyy - vxy**2
        if det <= 0: return None
        dx, dy = xx - mx, yy - my
        g = np.exp(-0.5*(vyy*dx**2 - 2*vxy*dx*dy + vxx*dy**2)/det)
        w = w0 * g; tot = w.sum()
        if tot <= 0: return None
        nmx, nmy = (w*xx).sum()/tot, (w*yy).sum()/tot
        nxx = 2*(w*(xx-nmx)**2).sum()/tot; nyy = 2*(w*(yy-nmy)**2).sum()/tot
        nxy = 2*(w*(xx-nmx)*(yy-nmy)).sum()/tot
        if nxx <= 0 or nyy <= 0 or nxx*nyy - nxy**2 <= 0: return None
        if abs(nxx-vxx) < 1e-4 and abs(nyy-vyy) < 1e-4:
            vxx, vyy, vxy = nxx, nyy, nxy; break
        mx, my, vxx, vyy, vxy = nmx, nmy, nxx, nyy, nxy
    return vxx, vyy

rows = []
for run_dir in sorted(glob.glob(os.path.join(RD, "zenith12", "08-1?_Z*"))):
    night = os.path.basename(run_dir)[:5]
    img = fits.getdata(glob.glob(os.path.join(run_dir, "CENTROID_OUTPUT*",
                                              "STACKED_FLOAT*.fit"))[0]).astype(float)
    z = zipfile.ZipFile(glob.glob(os.path.join(run_dir, "centroid_data*.zip"))[0])
    c = pd.read_csv(z.open("STACKED_CENTROIDS_DATA.csv"))
    c = c.sort_values("flux (noise-normed)", ascending=False).head(700)
    for _, s in c.iterrows():
        x0, y0 = int(round(s.px)), int(round(s.py))
        if not (8 < x0 < NX-9 and 8 < y0 < NY-9): continue
        m = adaptive(img[y0-8:y0+9, x0-8:x0+9])
        if m is None or not (0.4 < m[0] < 16 and 0.4 < m[1] < 16): continue
        rr = np.hypot((s.px-NX/2), (s.py-NY/2)) * PS / 3600
        fwhm = 2.355*np.sqrt(0.5*(m[0]+m[1]))*PS
        rows.append((night, os.path.basename(run_dir), rr, fwhm))
d = pd.DataFrame(rows, columns=["night", "field", "r_deg", "fwhm_as"])
bins = [0, 0.3, 0.6, 0.9, 1.2, 1.45]
print(f"{'annulus (deg)':>14s} {'night1 FWHM (as)':>17s} {'night2 FWHM (as)':>17s} {'ratio n2/n1':>12s}")
prof = {}
for night, g in d.groupby("night"):
    b = pd.cut(g.r_deg, bins)
    prof[night] = g.groupby(b, observed=True).fwhm_as.median()
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    f1 = prof["08-11"].iloc[i]; f2 = prof["08-12"].iloc[i]
    print(f"  {lo:.1f}-{hi:.2f}      {f1:17.3f} {f2:17.3f} {f2/f1:12.3f}")
g1 = prof["08-11"].iloc[4]/prof["08-11"].iloc[0]
g2 = prof["08-12"].iloc[4]/prof["08-12"].iloc[0]
print(f"\nradial growth (1.2-1.45 deg / 0-0.3 deg): night1 {g1:.3f}, night2 {g2:.3f}, "
      f"difference {g2-g1:+.3f}")
print(f"per-field scatter of the growth ratio:")
for night, g in d.groupby("night"):
    gr = []
    for f, gg in g.groupby("field"):
        b = pd.cut(gg.r_deg, bins)
        p = gg.groupby(b, observed=True).fwhm_as.median()
        if len(p) == 5 and p.iloc[0] > 0: gr.append(p.iloc[4]/p.iloc[0])
    print(f"  {night}: {np.mean(gr):.3f} +/- {np.std(gr, ddof=1):.3f} (n={len(gr)} fields)")
print("\nsensitivity context: Portland's conspicuous F19 coma (growth 1.32 vs Carrell's "
      "1.03 over a matched range) corresponds to ~1 mm of back-focus error; 32 um is "
      "~3 % of that, an expected growth change of ~0.01 -- so identity here BOUNDS the "
      "re-seat (< a few hundred um), it does not falsify it.")
