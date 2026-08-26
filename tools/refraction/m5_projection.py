"""M5 projection: what does the rehearsal residual field do to the deflection constant?

Inputs: the 45 constant-only rehearsal fits per window (H1 frames against the H3
reference -- the night analogue of step 3 against CAL_piLeo). Per-star mean residuals are
smoothed into a surface, sampled at the positions the real eclipse stars will occupy in
the frame, and fitted with the Method-1 model (pointing offsets + roll + L/R), giving a
forecast of the L bias this class of residual field produces.

Validation built in: (1) the sky->sensor mapping is fitted empirically from a solved
frame's own matched-star table, so no ROLL sign convention is assumed; (2) a synthetic
deflection field L = 1.000 arcsec is injected and must be recovered before the real
projection is trusted.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

RD = r"D:/MEE2024 output/MEE_output/refraction"
REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)

SITE = EarthLocation(lat=42.740470 * u.deg, lon=-5.613780 * u.deg, height=1101 * u.m)
T_MID = Time("2026-08-12T18:29:00", scale="utc")
PS = 2.2054456                      # arcsec/px, CAL_piLeo step-2 result
NX, NY = 6248, 4176
L_REF = 1.7512                      # arcsec, the deflection constant scale for % quotes
MAG_ECL = 11.0                      # step-3 max_star_mag_dist
RMIN_RSUN = 2.0                     # exclude stars inside 2 solar radii

# ---------------------------------------------------------------- eclipse-frame geometry
sun = get_sun(T_MID)
sun_aa = sun.transform_to(AltAz(obstime=T_MID, location=SITE))
R_SUN_AS = np.degrees(np.arcsin((696000 * u.km / sun.distance).decompose().value)) * 3600
centre_aa = AltAz(alt=sun_aa.alt + 0.74 * u.deg, az=sun_aa.az,
                  obstime=T_MID, location=SITE)
centre = SkyCoord(centre_aa).icrs
print(f"Sun alt {sun_aa.alt.deg:.2f} deg az {sun_aa.az.deg:.2f} deg, "
      f"R_sun {R_SUN_AS:.1f} arcsec; frame centre RA {centre.ra.deg:.4f} "
      f"Dec {centre.dec.deg:.4f} deg")

def gnomonic(sc, ctr):
    """Standard coordinates (xi, eta) in degrees about ctr: xi = +East, eta = +North."""
    ra, dec = np.radians(sc.ra.deg), np.radians(sc.dec.deg)
    ra0, dec0 = np.radians(ctr.ra.deg), np.radians(ctr.dec.deg)
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi), np.degrees(eta)

# Empirical sky->sensor affine from one solved frame (camera clamped at the eclipse
# rotation throughout, so the same rotation applies; only the centre moves).
cal = sorted(glob.glob(os.path.join(RD, "m5_rehearsal", "N2", "f20", "**",
                                    "CATALOGUE_MATCHED_ERRORS.csv"), recursive=True))
if not cal:
    cal = sorted(glob.glob(os.path.join(RD, "perframe", "N2", "H1", "f20", "corr_on", "**",
                                        "CATALOGUE_MATCHED_ERRORS.csv"), recursive=True))
dcal = pd.read_csv(cal[0])
res = sorted(glob.glob(os.path.join(os.path.dirname(cal[0]), "distortion_results.txt")))
jcal = json.load(open(res[0]))
fc = SkyCoord(jcal["RA"] * u.deg, jcal["DEC"] * u.deg)
xi, eta = gnomonic(SkyCoord(dcal["RA(catalog)"].values * u.deg,
                            dcal["DEC(catalog)"].values * u.deg), fc)
A = np.column_stack([xi, eta, np.ones_like(xi)])
mx, *_ = np.linalg.lstsq(A, dcal.px.values, rcond=None)
my, *_ = np.linalg.lstsq(A, dcal.py.values, rcond=None)
fitres = np.hypot(A @ mx - dcal.px.values, A @ my - dcal.py.values)
print(f"sky->sensor affine from {len(dcal)} stars, residual {np.median(fitres):.2f} px "
      f"(distortion-level, fine for geometry)")

def sky_to_px(sc):
    """Eclipse-frame sensor position: the affine's rotation/scale (the camera clamp is
    unchanged), re-centred so the eclipse frame centre sits at the sensor centre."""
    xi, eta = gnomonic(sc, centre)
    xi, eta = np.atleast_1d(xi), np.atleast_1d(eta)
    B = np.column_stack([xi, eta, np.zeros_like(xi)])
    return B @ mx + NX / 2, B @ my + NY / 2

# ---------------------------------------------------------------- eclipse star sample
from mee2024 import database_cache                                  # noqa: E402
dbs = database_cache.open_catalogue("gaia", gaia_limit=13)
half_ra = 2.4 / np.cos(np.radians(centre.dec.deg))
star = dbs.lookup_objects((centre.ra.deg - half_ra, centre.ra.deg + half_ra),
                          (centre.dec.deg - 2.4, centre.dec.deg + 2.4),
                          star_max_magnitude=MAG_ECL, time=2026.61)
svec = SkyCoord(np.degrees(star.get_ra()) * u.deg, np.degrees(star.get_dec()) * u.deg)
spx, spy = sky_to_px(svec)
sunpx, sunpy = sky_to_px(SkyCoord(sun.ra, sun.dec))
inframe = (spx > 60) & (spx < NX - 60) & (spy > 60) & (spy < NY - 60)
rx, ry = (spx - sunpx) * PS, (spy - sunpy) * PS          # arcsec from Sun, sensor axes
R = np.hypot(rx, ry)
keep = inframe & (R > RMIN_RSUN * R_SUN_AS)
spx, spy, rx, ry, R = spx[keep], spy[keep], rx[keep], ry[keep], R[keep]
print(f"eclipse star sample: {keep.sum()} stars G<={MAG_ECL:.0f} in frame outside "
      f"{RMIN_RSUN:.0f} R_sun; Sun at px ({sunpx[0]:.0f}, {sunpy[0]:.0f}); "
      f"R range {R.min()/R_SUN_AS:.1f}-{R.max()/R_SUN_AS:.1f} R_sun")
# F&L's h for THIS field: fixes which delta-L/delta-S sensitivity row applies (STAGE3 s4)
h_fl = 1.0 / np.mean((R_SUN_AS / R) ** 2)
print(f"F&L h = 1/mean(1/r^2) = {h_fl:.1f} R_sun^2  ->  dL/L per ppm of plate-scale error "
      f"= {h_fl * 946.8e-6 / L_REF * 100:.2f} %/ppm-ish; exact: dL = h * dS with S in "
      f"arcsec per R_sun: {h_fl:.1f} * {R_SUN_AS:.0f} as * 1e-6 = "
      f"{h_fl * R_SUN_AS * 1e-6:.5f} as/ppm = {h_fl * R_SUN_AS * 1e-6 / L_REF * 100:.2f} % of L per ppm")

# ---------------------------------------------------------------- Method-1 estimator
def fit_L(dx, dy, x_px, y_px, with_scale=False):
    """LS fit of [N1, N2, Theta(, S)] + L/R to a residual field (arcsec, sensor axes)."""
    xs, ys = (x_px - NX / 2) * PS, (y_px - NY / 2) * PS      # arcsec about frame centre
    ux, uy = rx / R, ry / R
    n = len(dx)
    Zx, Zy = np.zeros(n), np.zeros(n)
    cols_x = [np.ones(n), Zx, -ys, ux * R_SUN_AS / R]
    cols_y = [Zy, np.ones(n), xs, uy * R_SUN_AS / R]
    if with_scale:
        cols_x.insert(3, xs); cols_y.insert(3, ys)
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([dx, dy])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[-1]

# validation: inject L = 1.000 arcsec, must recover it
dx_inj = 1.0 * (R_SUN_AS / R) * (rx / R)
dy_inj = 1.0 * (R_SUN_AS / R) * (ry / R)
L_chk = fit_L(dx_inj, dy_inj, spx, spy)
L_chk2 = fit_L(dx_inj, dy_inj, spx, spy, with_scale=True)
print(f"estimator check: injected L=1.000 -> recovered {L_chk:.4f} (M1), "
      f"{L_chk2:.4f} (M2-style)  [must be ~1.000]")
assert abs(L_chk - 1) < 0.02

# ---------------------------------------------------------------- per-window projection
def surface(px, py, v, deg=3):
    """Least-squares polynomial surface over the sensor, evaluated later at (X, Y)."""
    x, y = (px - NX / 2) / 3124.0, (py - NY / 2) / 3124.0
    cols = [x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)]
    M = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(M, v, rcond=None)
    def ev(X, Y):
        xx, yy = (X - NX / 2) / 3124.0, (Y - NY / 2) / 3124.0
        return np.column_stack([xx**i * yy**j
                                for i in range(deg + 1)
                                for j in range(deg + 1 - i)]) @ c
    return ev, c

print(f"\n{'window':8s} {'stars':>6s} {'field rms (as)':>14s} {'lin part (ppm)':>14s} "
      f"{'dL M1 (as)':>11s} {'dL M1 (%L)':>11s} {'dL M2-style (as)':>16s} {'boot se (as)':>13s}")
rows = []
for w in ("N1", "N2", "N3"):
    files = sorted(glob.glob(os.path.join(RD, "m5_rehearsal", w, "f*", "**",
                                          "TWOD_RESIDUALS.csv"), recursive=True))
    if len(files) < 30:
        print(f"{w:8s}  only {len(files)} rehearsal fits, skipped"); continue
    acc = {}
    for f in files:
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec))
    ids = [k for k, v in acc.items() if len(v) >= 20]
    P = np.array([[np.median([q[0] for q in acc[i]]), np.median([q[1] for q in acc[i]]),
                   np.median([q[2] for q in acc[i]]), np.median([q[3] for q in acc[i]]),
                   np.std([q[2] for q in acc[i]], ddof=1) / np.sqrt(len(acc[i])),
                   np.std([q[3] for q in acc[i]], ddof=1) / np.sqrt(len(acc[i]))]
                  for i in ids])
    mpx, mpy, mdx, mdy, sdx, sdy = P.T
    # remove the median offset (the free constant of step 3 absorbs it)
    mdx, mdy = mdx - np.median(mdx), mdy - np.median(mdy)
    # Persistent catalogue mismatches survive per-star averaging (same wrong star every
    # frame) and tol 999 never rejects them -- N1 carried one at 64.5 arcsec. Clip by MAD:
    # this is the analysis-side analogue of the protection F16 will have to provide at
    # step 3, and the clipped count is itself a finding.
    mag = np.hypot(mdx, mdy)
    lim = max(3.0 * 1.4826 * np.median(np.abs(mag - np.median(mag))) + np.median(mag), 2.5)
    good = mag < lim
    n_clip = int((~good).sum())
    mpx, mpy, mdx, mdy, sdx, sdy = (a[good] for a in (mpx, mpy, mdx, mdy, sdx, sdy))
    frms = np.sqrt(np.mean(mdx**2 + mdy**2))
    evx, cx = surface(mpx, mpy, mdx)
    evy, cy = surface(mpx, mpy, mdy)
    # inherited linear (scale-like) content, for cross-check against the M2 differential
    xs, ys = (mpx - NX / 2) * PS, (mpy - NY / 2) * PS
    G = np.column_stack([np.ones_like(xs), xs, ys])
    sc = (np.linalg.lstsq(G, mdx, rcond=None)[0][1]
          + np.linalg.lstsq(G, mdy, rcond=None)[0][2]) / 2 * 1e6
    dxs, dys = evx(spx, spy), evy(spx, spy)
    L1 = fit_L(dxs, dys, spx, spy)
    L2 = fit_L(dxs, dys, spx, spy, with_scale=True)
    # bootstrap: resample the per-star means by their own SEs, rebuild surface, refit
    rng = np.random.default_rng(7)
    Ls = []
    for _ in range(200):
        ex, ey = (surface(mpx, mpy, mdx + rng.normal(0, sdx))[0],
                  surface(mpx, mpy, mdy + rng.normal(0, sdy))[0])
        Ls.append(fit_L(ex(spx, spy), ey(spx, spy), spx, spy))
    se = np.std(Ls, ddof=1)
    rows.append((w, len(mpx), frms, sc, L1, L2, se))
    print(f"{w:8s} {len(mpx):6d} {frms:14.3f} {sc:+14.1f} {L1:+11.4f} "
          f"{L1/L_REF*100:+10.1f}% {L2:+16.4f} {se:13.4f}   ({n_clip} star(s) clipped)")

print(f"""
Reading guide: 'lin part' is the mean-field's scale-like component (should track the M2
H3-H1 differentials of about -73/-73/-193 ppm... per window as measured); dL M1 is the
Method-1 bias the surface produces through the real estimator basis; dL M2-style frees a
scale term as eq. 12 would. Bootstrap se is star-noise only -- the window-to-window spread
is the honest systematic error bar.""")
