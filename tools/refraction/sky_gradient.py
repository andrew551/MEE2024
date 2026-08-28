"""Does the totality sky gradient explain CAL_piLeo's ~0.4 arcsec horizontal excess?

M3 found the daytime horizontal quasi-static residual is 3-6x the night value (0.42 vs
0.07-0.13 arcsec) while the vertical matches the night atmosphere exactly. Candidate: the
totality sky-brightness gradient (Sun ~11 deg west of the cal field) biasing windowed
centroids through the background. Mechanism arithmetic: after annular background
subtraction a linear gradient g (ADU/px) survives across the centroid window, shifting a
Gaussian-weighted centroid (sigma_w = 2 px) by

    delta = g * 2*pi*sigma_w^4 / F_w        (px; F_w = window-weighted star flux, ADU)

A UNIFORM shift is absorbed by the fit's free constant; what survives in the residuals is
the star-to-star spread, i.e. a 1/F-correlated scatter ALONG the gradient direction. So
the test is threefold: (1) measure g on the actual stacked image and per frame; (2) check
its direction against the Sun's; (3) test the per-star prediction delta_i = c * g/F_i
against the observed along-gradient residuals, star by star.
"""
import glob
import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"
SITE = EarthLocation(lat=42.740470 * u.deg, lon=-5.613780 * u.deg, height=1101 * u.m)
T0 = Time("2026-08-12T18:29:34", scale="utc")
NX, NY = 6248, 4176
PS = 2.2054           # arcsec/px
SIG_W = 2.0           # px, centroid_window_sigma
SIG_PSF = 1.17        # px, from the measured 6.1 arcsec FWHM


def plane_gradient(img, block=64, it=3):
    """Robust background plane: block medians, iterative clip, LS plane fit.
    Returns (gx, gy) in ADU/px and the block grid for curvature inspection."""
    ny, nx = img.shape
    by, bx = ny // block, nx // block
    g = np.median(img[:by * block, :bx * block]
                  .reshape(by, block, bx, block), axis=(1, 3))
    yy, xx = np.mgrid[0:by, 0:bx]
    xc, yc = (xx + 0.5) * block, (yy + 0.5) * block
    m = np.ones_like(g, bool)
    for _ in range(it):
        A = np.column_stack([np.ones(m.sum()), xc[m], yc[m]])
        c, *_ = np.linalg.lstsq(A, g[m], rcond=None)
        r = g - (c[0] + c[1] * xc + c[2] * yc)
        s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
        m = np.abs(r - np.median(r[m])) < 3 * s
    return c[1], c[2], g, float(np.median(g[m]))


# ---------------------------------------------------------------- direction reference
run = sorted(glob.glob(B + "/definitive_tol999/**/CATALOGUE_MATCHED_ERRORS.csv",
                       recursive=True))[0]
d = pd.read_csv(run)
d = d[~d.flag_is_outlier].reset_index(drop=True)
aa = SkyCoord(d["RA(catalog)"].values * u.deg, d["DEC(catalog)"].values * u.deg
              ).transform_to(AltAz(obstime=T0, location=SITE))
alt, azc = aa.alt.deg, aa.az.deg * np.cos(np.radians(aa.alt.deg.mean()))
A = np.column_stack([azc - azc.mean(), alt - alt.mean(), np.ones(len(d))])
cx, *_ = np.linalg.lstsq(A, d.px.values, rcond=None)
cy, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
v_az = np.array([cx[0], cy[0]]); v_az /= np.linalg.norm(v_az)     # +azimuth (sensor px)
v_alt = np.array([cx[1], cy[1]]); v_alt /= np.linalg.norm(v_alt)  # +altitude (sensor px)
sun_aa = get_sun(T0).transform_to(AltAz(obstime=T0, location=SITE))
d_az = (sun_aa.az.deg - aa.az.deg.mean()) * np.cos(np.radians(alt.mean()))
d_alt = sun_aa.alt.deg - alt.mean()
sun_dir = d_az * v_az + d_alt * v_alt
sun_dir /= np.linalg.norm(sun_dir)
print(f"field centre alt {alt.mean():.2f} az {aa.az.deg.mean():.2f} deg; Sun offset "
      f"{d_az:+.1f} deg az*cos(alt), {d_alt:+.1f} deg alt -> direction toward Sun in "
      f"sensor px: ({sun_dir[0]:+.3f}, {sun_dir[1]:+.3f})")

# ---------------------------------------------------------------- gradients
stack = fits.getdata(glob.glob(B + "/s1_combined17/CENTROID_OUTPUT*/STACKED_FLOAT*.fit")[0]
                     ).astype(float)
gx, gy, grid, sky = plane_gradient(stack)
gmag = np.hypot(gx, gy)
gdir = np.array([gx, gy]) / gmag
cosang = float(gdir @ sun_dir)
print(f"\nSTACK: sky {sky:.0f} ADU; gradient ({gx*1000:+.1f}, {gy*1000:+.1f}) ADU/kpx, "
      f"|g| {gmag*1000:.1f} ADU/kpx; cos(angle to Sun direction) = {cosang:+.2f}")
print(f"  components: along-azimuth {1000*(gx*v_az[0]+gy*v_az[1]):+.1f}, "
      f"along-altitude {1000*(gx*v_alt[0]+gy*v_alt[1]):+.1f} ADU/kpx")

print("\nper-frame gradients (ADU/kpx, sensor axes) and sky (ADU):")
for f in sorted(glob.glob(r"I:/Leon 2026/2026-08-12/Eclipse/CAL_piLeo/18_29_*/*.fits")):
    img = fits.getdata(f).astype(float)
    h = fits.getheader(f)
    fgx, fgy, _, fsky = plane_gradient(img, block=128, it=2)
    tag = f.replace("\\", "/").split("/")[-2] + "/" + f.split("_")[-1]
    print(f"  {tag:22s} exp {float(h['EXPTIME']):3.1f} s  sky {fsky:7.0f}  "
          f"g ({fgx*1000:+7.1f}, {fgy*1000:+7.1f})  |g| {np.hypot(fgx,fgy)*1000:6.1f}")

for f in sorted(glob.glob(r"G:/Leon Aug 2026/2026-08-12/Horizon/H3_calfield_sightline/*/*.fits"))[20:22]:
    img = fits.getdata(f).astype(float)
    fgx, fgy, _, fsky = plane_gradient(img, block=128, it=2)
    print(f"  NIGHT control H3 f{f[-7:-5]}: sky {fsky:6.0f} ADU, "
          f"g ({fgx*1000:+.1f}, {fgy*1000:+.1f}) ADU/kpx, |g| {np.hypot(fgx,fgy)*1000:.1f}")

# ---------------------------------------------------------------- per-star prediction
res = pd.read_csv(sorted(glob.glob(B + "/definitive_tol999/**/TWOD_RESIDUALS.csv",
                                   recursive=True))[0])
K = 2 * np.pi * SIG_W**4                       # px^4; delta = g*K/F_w
w_frac = SIG_W**2 / (SIG_W**2 + SIG_PSF**2)    # window-weighted fraction of star flux
yy, xx = np.mgrid[-8:9, -8:9]
wwin = np.exp(-(xx**2 + yy**2) / (2 * SIG_W**2))
rows = []
for _, s in res.iterrows():
    x0, y0 = int(round(s.px)), int(round(s.py))
    if not (8 < x0 < NX - 9 and 8 < y0 < NY - 9):
        continue
    cut = stack[y0 - 8:y0 + 9, x0 - 8:x0 + 9]
    rr = np.hypot(xx, yy)
    bg = np.median(cut[(rr > 6) & (rr <= 8)])
    F_w = float(((cut - bg) * wwin).sum())
    if F_w <= 0:
        continue
    pred = gmag * K / F_w * PS                 # arcsec, along the gradient direction
    obs_g = (s.dx_arcsec * gdir[0] + s.dy_arcsec * gdir[1])
    obs_p = (-s.dx_arcsec * gdir[1] + s.dy_arcsec * gdir[0])
    rows.append((s.magV, F_w, pred, obs_g, obs_p))
t = pd.DataFrame(rows, columns=["magV", "Fw_ADU", "pred_as", "obs_along_as", "obs_perp_as"])
t.to_csv(B + "/sky_gradient_perstar.csv", index=False)

print(f"\nper-star test ({len(t)} stars):")
print(f"  predicted along-gradient shift: median {t.pred_as.median():.4f} arcsec, "
      f"faintest quartile {t.nlargest(len(t)//4, 'magV').pred_as.median():.4f} arcsec, "
      f"max {t.pred_as.max():.4f} arcsec")
print(f"  observed along-gradient residual sd {t.obs_along_as.std(ddof=1):.3f} arcsec, "
      f"perpendicular sd {t.obs_perp_as.std(ddof=1):.3f} arcsec")
r = np.corrcoef(t.pred_as, t.obs_along_as)[0, 1]
rf = np.corrcoef(1 / t.Fw_ADU, t.obs_along_as)[0, 1]
print(f"  corr(pred, obs_along) = {r:+.3f};  corr(1/F, obs_along) = {rf:+.3f}  "
      f"(n = {len(t)}, |r| > {2/np.sqrt(len(t)):.2f} is a 2-sigma detection)")
q = pd.qcut(t.magV, 4)
print("  by magnitude quartile:  pred (as)   obs_along sd (as)  obs_perp sd (as)")
for k, g_ in t.groupby(q, observed=True):
    print(f"    {str(k):22s} {g_.pred_as.median():9.4f} {g_.obs_along_as.std(ddof=1):17.3f}"
          f" {g_.obs_perp_as.std(ddof=1):16.3f}")
