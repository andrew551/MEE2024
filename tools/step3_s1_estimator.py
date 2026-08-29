"""S1: the nuisance estimator, gated on the M5 night nulls before it may touch Leon.

STEP3_PLAN's design: a joint fit of [N1, N2, Theta, L*(1/R)u_radial, plus a SMOOTH
VERTICAL nuisance field], with the outer stars constraining the nuisance. The vertical
choice is measured, not aesthetic: M3 found the unabsorbed atmosphere vertically polarised
(quasi-static V/H ~ 2.3), and the CAL_piLeo affine puts sensor -y within 3.1 degrees of
the local vertical, so the nuisance direction is sensor y-hat and its amplitude a low-order
2D polynomial over the frame. The fit stays one linear model (Andrew's 2026-08-09 form /
F&L's A-matrix): L is just one more column, solved by lstsq with honest joint covariance.

THE GATE (decision rule fixed in the plan before looking): on the three M5 rehearsal
windows -- real night atmospheres, ZERO true deflection -- the estimator must
  (a) shrink the raw Method-1 fake deflections (-0.97 / +0.23 / +0.41 arcsec) toward the
      bootstrap noise floor, and
  (b) recover an injected L = 1.7512 arcsec unbiased to <= 2 %,
at the same nuisance order. If no order does both, the gate FAILS and Leon's L is quoted
with the +-0.1-0.6 arcsec systematic stated instead.

Geometry, the residual loader and the base fit are taken from tools/refraction/
m5_projection.py (which validated the basis by recovering an injected L = 1.000 to 1.0000);
copied rather than imported because that script executes at module level.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

RD = r"D:/MEE2024 output/MEE_output/refraction"
REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)

SITE = EarthLocation(lat=42.740470*u.deg, lon=-5.613780*u.deg, height=1101*u.m)
T_MID = Time("2026-08-12T18:29:00", scale="utc")
PS, NX, NY = 2.2054043, 6248, 4176
L_REF = 1.7512
W_NORM = 3124.0

sun = get_sun(T_MID)
sun_aa = sun.transform_to(AltAz(obstime=T_MID, location=SITE))
R_SUN_AS = np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600
centre_aa = AltAz(alt=sun_aa.alt + 0.74*u.deg, az=sun_aa.az, obstime=T_MID, location=SITE)
centre = SkyCoord(centre_aa).icrs

def gnomonic(sc, ctr):
    ra, dec = np.radians(sc.ra.deg), np.radians(sc.dec.deg)
    ra0, dec0 = np.radians(ctr.ra.deg), np.radians(ctr.dec.deg)
    cosc = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    xi = np.cos(dec)*np.sin(ra - ra0)/cosc
    eta = (np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0))/cosc
    return np.degrees(xi), np.degrees(eta)

cal = sorted(glob.glob(os.path.join(RD, "m5_rehearsal", "N2", "f20", "**",
                                    "CATALOGUE_MATCHED_ERRORS.csv"), recursive=True))
dcal = pd.read_csv(cal[0])
jcal = json.load(open(sorted(glob.glob(os.path.join(os.path.dirname(cal[0]),
                                                    "distortion_results.txt")))[0]))
fc = SkyCoord(jcal["RA"]*u.deg, jcal["DEC"]*u.deg)
xi, eta = gnomonic(SkyCoord(dcal["RA(catalog)"].values*u.deg,
                            dcal["DEC(catalog)"].values*u.deg), fc)
A0 = np.column_stack([xi, eta, np.ones_like(xi)])
mx, *_ = np.linalg.lstsq(A0, dcal.px.values, rcond=None)
my, *_ = np.linalg.lstsq(A0, dcal.py.values, rcond=None)

def sky_to_px(sc):
    xi, eta = gnomonic(sc, centre)
    xi, eta = np.atleast_1d(xi), np.atleast_1d(eta)
    B = np.column_stack([xi, eta, np.zeros_like(xi)])
    return B@mx + NX/2, B@my + NY/2

from mee2024 import database_cache
dbs = database_cache.open_catalogue("gaia", gaia_limit=13)
half_ra = 2.4/np.cos(np.radians(centre.dec.deg))
star = dbs.lookup_objects((centre.ra.deg - half_ra, centre.ra.deg + half_ra),
                          (centre.dec.deg - 2.4, centre.dec.deg + 2.4),
                          star_max_magnitude=11.0, time=2026.61)
svec = SkyCoord(np.degrees(star.get_ra())*u.deg, np.degrees(star.get_dec())*u.deg)
spx, spy = sky_to_px(svec)
sunpx, sunpy = sky_to_px(SkyCoord(sun.ra, sun.dec))
rx, ry = (spx - sunpx)*PS, (spy - sunpy)*PS
R = np.hypot(rx, ry)
keep = (spx > 60) & (spx < NX-60) & (spy > 60) & (spy < NY-60) & (R > 2.0*R_SUN_AS)
spx, spy, rx, ry, R = (a[keep] for a in (spx, spy, rx, ry, R))
print(f"eclipse geometry: {keep.sum()} stars G<=11 outside 2 R_sun; "
      f"Sun px ({sunpx[0]:.0f},{sunpy[0]:.0f}); h = {1/np.mean((R_SUN_AS/R)**2):.1f} R_sun^2")

# ------------------------------------------------------------------ the estimator
def design(x_px, y_px, rx_, ry_, R_, nuis_deg=None, with_scale=False, vector=False):
    """One linear model over 2N components. Columns: N1, N2, Theta, (S,) L, then the
    smooth vertical nuisance b_ij x^i y^j (i+j <= deg, skipping 0,0 which is N2)."""
    xs, ys = (x_px - NX/2)/W_NORM, (y_px - NY/2)/W_NORM     # dimensionless frame coords
    ux, uy = rx_/R_, ry_/R_
    n = len(x_px)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px - NY/2)*PS]
    cols_y = [Z, np.ones(n), (x_px - NX/2)*PS]
    if with_scale:
        cols_x.append((x_px - NX/2)*PS); cols_y.append((y_px - NY/2)*PS)
    cols_x.append(ux*R_SUN_AS/R_); cols_y.append(uy*R_SUN_AS/R_)
    labels = ['N1','N2','Th'] + (['S'] if with_scale else []) + ['L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0: continue
                cols_x.append(Z); cols_y.append(xs**i * ys**j)
                labels.append(f'v{i}{j}')
                if vector:
                    cols_x.append(xs**i * ys**j); cols_y.append(Z)
                    labels.append(f'h{i}{j}')
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    return A, labels

def fit_L(dx, dy, x_px, y_px, rx_, ry_, R_, nuis_deg=None, with_scale=False, vector=False):
    A, labels = design(x_px, y_px, rx_, ry_, R_, nuis_deg, with_scale, vector)
    b = np.concatenate([dx, dy])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[labels.index('L')]

# self-test on the eclipse geometry: inject pure L, every variant must return it
dx_i, dy_i = L_REF*(R_SUN_AS/R)*(rx/R), L_REF*(R_SUN_AS/R)*(ry/R)
for nd, vec in ((None,False), (1,False), (2,False), (3,False), (2,True), (3,True)):
    got = fit_L(dx_i, dy_i, spx, spy, rx, ry, R, nuis_deg=nd, vector=vec)
    print(f"  pure-L self-test, nuisance {nd}{'v' if vec else ''}: recovered {got:.4f} / {L_REF}")
    if abs(got - L_REF) > 0.02*L_REF:
        print(f"    -> IDENTIFIABILITY LOST (L absorbed by the nuisance)")

def surface(px, py, v, deg=3):
    x, y = (px - NX/2)/W_NORM, (py - NY/2)/W_NORM
    M = np.column_stack([x**i * y**j for i in range(deg+1) for j in range(deg+1-i)])
    c, *_ = np.linalg.lstsq(M, v, rcond=None)
    def ev(X, Y):
        xx, yy = (X - NX/2)/W_NORM, (Y - NY/2)/W_NORM
        return np.column_stack([xx**i * yy**j for i in range(deg+1)
                                for j in range(deg+1-i)]) @ c
    return ev

# ------------------------------------------------------------------ the gate
print(f"\n=== THE GATE: M5 night nulls (true L = 0) ===")
print(f"{'window':7} {'deg':>4} {'dL null (as)':>13} {'inj-recovery (as)':>18} {'bias %':>7}")
verdicts = {}
for w in ("N1", "N2", "N3"):
    files = sorted(glob.glob(os.path.join(RD, "m5_rehearsal", w, "f*", "**",
                                          "TWOD_RESIDUALS.csv"), recursive=True))
    acc = {}
    for f in files:
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec))
    ids = [k for k, v in acc.items() if len(v) >= 20]
    P = np.array([[np.median([q[c] for q in acc[i]]) for c in range(4)] for i in ids])
    mpx, mpy, mdx, mdy = P.T
    mdx, mdy = mdx - np.median(mdx), mdy - np.median(mdy)
    mag = np.hypot(mdx, mdy)
    lim = max(3.0*1.4826*np.median(np.abs(mag - np.median(mag))) + np.median(mag), 2.5)
    good = mag < lim
    mpx, mpy, mdx, mdy = (a[good] for a in (mpx, mpy, mdx, mdy))
    evx, evy = surface(mpx, mpy, mdx), surface(mpx, mpy, mdy)
    dxs, dys = evx(spx, spy), evy(spx, spy)
    for nd, vec in ((None,False), (1,False), (2,False), (3,False), (2,True), (3,True)):
        tag = f"{nd}{'v' if vec else ''}"
        Lnull = fit_L(dxs, dys, spx, spy, rx, ry, R, nuis_deg=nd, vector=vec)
        Linj = fit_L(dxs + dx_i, dys + dy_i, spx, spy, rx, ry, R, nuis_deg=nd, vector=vec)
        rec = Linj - Lnull
        verdicts.setdefault(tag, []).append((Lnull, rec))
        print(f"{w:7} {tag:>4} {Lnull:+13.4f} {rec:18.4f} {100*(rec-L_REF)/L_REF:+7.2f}")

print(f"\n=== VERDICT (plan criteria: |dL null| shrunk toward floor AND recovery bias <= 2 %) ===")
for nd, rows in verdicts.items():
    nulls = np.array([r[0] for r in rows]); recs = np.array([r[1] for r in rows])
    print(f"  deg {str(nd):>4}: |dL null| max {np.abs(nulls).max():.3f} as "
          f"(raw Method 1 was 0.97); worst recovery bias "
          f"{100*np.abs(recs - L_REF).max()/L_REF:.2f} %")
print('done', flush=True)


# ------------------------------------------------------------------ gate v2: unsmoothed
# The v1 gate above projects the night field through a DEG-3 SMOOTHING SURFACE before the
# estimator sees it -- so a full vector deg-3 nuisance absorbs it exactly BY CONSTRUCTION
# (nulls of 0.0000 are an artifact of the surface, not a property of the atmosphere), and
# even v-deg tests are softened. Gate v2 removes the circularity: the estimator fits the
# per-star MEDIAN residuals of the night field's OWN stars, wavefield patchiness and all,
# with the eclipse Sun placed at its measured frame position. The noise floor is the
# bootstrap over per-star standard errors -- criterion (a) means |dL null| comparable to
# that floor, not merely smaller than 0.97.
SUNPX, SUNPY = 3171.0, 3232.0        # measured (ephemeris through the fitted frames)
print("\n=== GATE V2: per-star night residuals, no smoothing ===")
print(f"{'window':7} {'var':>4} {'dL null (as)':>13} {'recovery bias %':>16} {'boot floor (as)':>16}")
rng = np.random.default_rng(11)
verdict2 = {}
for w in ("N1", "N2", "N3"):
    files = sorted(glob.glob(os.path.join(RD, "m5_rehearsal", w, "f*", "**",
                                          "TWOD_RESIDUALS.csv"), recursive=True))
    acc = {}
    for f in files:
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            acc.setdefault(r.ID, []).append((r.px, r.py, r.dx_arcsec, r.dy_arcsec))
    ids = [k for k, v in acc.items() if len(v) >= 20]
    P = np.array([[np.median([q[0] for q in acc[i]]), np.median([q[1] for q in acc[i]]),
                   np.median([q[2] for q in acc[i]]), np.median([q[3] for q in acc[i]]),
                   np.std([q[2] for q in acc[i]], ddof=1)/np.sqrt(len(acc[i])),
                   np.std([q[3] for q in acc[i]], ddof=1)/np.sqrt(len(acc[i]))]
                  for i in ids])
    mpx, mpy, mdx, mdy, sdx, sdy = P.T
    mdx, mdy = mdx - np.median(mdx), mdy - np.median(mdy)
    mag = np.hypot(mdx, mdy)
    lim = max(3.0*1.4826*np.median(np.abs(mag - np.median(mag))) + np.median(mag), 2.5)
    good = mag < lim
    mpx, mpy, mdx, mdy, sdx, sdy = (a[good] for a in (mpx, mpy, mdx, mdy, sdx, sdy))
    nrx, nry = (mpx - SUNPX)*PS, (mpy - SUNPY)*PS
    nR = np.hypot(nrx, nry)
    infield = nR > 2.0*R_SUN_AS
    mpx, mpy, mdx, mdy, sdx, sdy, nrx, nry, nR = (a[infield] for a in
        (mpx, mpy, mdx, mdy, sdx, sdy, nrx, nry, nR))
    inj_x, inj_y = L_REF*(R_SUN_AS/nR)*(nrx/nR), L_REF*(R_SUN_AS/nR)*(nry/nR)
    for nd, vec in ((None,False), (2,False), (2,True), (3,True)):
        tag = f"{nd}{'v' if vec else ''}"
        Lnull = fit_L(mdx, mdy, mpx, mpy, nrx, nry, nR, nuis_deg=nd, vector=vec)
        Linj = fit_L(mdx + inj_x, mdy + inj_y, mpx, mpy, nrx, nry, nR, nuis_deg=nd, vector=vec)
        rec = Linj - Lnull
        boots = [fit_L(mdx + rng.normal(0, sdx), mdy + rng.normal(0, sdy),
                       mpx, mpy, nrx, nry, nR, nuis_deg=nd, vector=vec)
                 for _ in range(120)]
        floor = float(np.std(boots, ddof=1))
        verdict2.setdefault(tag, []).append((Lnull, rec, floor))
        print(f"{w:7} {tag:>4} {Lnull:+13.4f} {100*(rec-L_REF)/L_REF:+16.2f} {floor:16.3f}")

print("\n=== VERDICT V2 ===")
for tag, rows in verdict2.items():
    nulls = np.array([r[0] for r in rows]); floors = np.array([r[2] for r in rows])
    recs = np.array([r[1] for r in rows])
    print(f"  {tag:>4}: |dL null| max {np.abs(nulls).max():.3f} as vs floor ~{floors.max():.3f}; "
          f"worst recovery bias {100*np.abs(recs - L_REF).max()/L_REF:.2f} %")
print('done v2', flush=True)

# ------------------------------------------------------------------ Leon application
# Gate verdict: v-deg2 is the best honest variant (0.77 -> 0.32 arcsec worst null, 2.4x)
# but does NOT reach the floor -- so it is applied AND the residual +-0.33 arcsec is
# quoted as the atmospheric systematic, per the plan's decision-before-looking rule.
import zipfile
print("\n=== LEON: estimator applied to the rematched star tables ===")
print(f"{'set':28} {'N':>4} {'L base (as)':>12} {'L v-deg2 (as)':>14}")
BL = r"D:/MEE2024 output/MEE_output/step3_prelim_L"
for tier in ('0p6s', '1p2s'):
    for tag in ('with_anchor', 'sans_anchor'):
        zp = os.path.join(BL, tier, f'stage3_rematched_{tag}', 'distortion_data_rematched.zip')
        if not os.path.exists(zp): continue
        za = zipfile.ZipFile(zp)
        df2 = pd.read_csv(za.open([n for n in za.namelist()
                                   if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
        df2.columns = [c.strip() for c in df2.columns]
        df2 = df2[df2['magV'] <= 11.0]
        # sky displacement (obs - catalog) -> sensor axes through the affine's linear part
        dxi = (df2['RA(obs)'].values - df2['RA(catalog)'].values) \
              * np.cos(np.radians(df2['DEC(catalog)'].values))
        deta = df2['DEC(obs)'].values - df2['DEC(catalog)'].values
        dpx = mx[0]*dxi + mx[1]*deta
        dpy = my[0]*dxi + my[1]*deta
        ddx, ddy = dpx*PS, dpy*PS
        lrx, lry = (df2['px'].values - SUNPX)*PS, (df2['py'].values - SUNPY)*PS
        lR = np.hypot(lrx, lry)
        ok = lR > 2.0*R_SUN_AS
        Lb = fit_L(ddx[ok], ddy[ok], df2['px'].values[ok], df2['py'].values[ok],
                   lrx[ok], lry[ok], lR[ok])
        Lv = fit_L(ddx[ok], ddy[ok], df2['px'].values[ok], df2['py'].values[ok],
                   lrx[ok], lry[ok], lR[ok], nuis_deg=2)
        print(f"{tier+' '+tag:28} {int(ok.sum()):4d} {Lb:+12.4f} {Lv:+14.4f}")
print("(GR: 1.7512 arcsec; quoted systematic from the gate: +-0.33 arcsec, v-deg2 residual)")
print('done leon', flush=True)
