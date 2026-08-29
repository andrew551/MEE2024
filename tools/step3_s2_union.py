"""S2: the union — every tier's stars in one catalogue-level table, one estimator fit.

The path (fixed by S1's findings): the estimator's Leon number is unstable at 24-25 stars
against 9 nuisance parameters, and stable at 41. The union brings every tier's detections
through the two-pass rematch machinery — the shallow tiers (0.1 s, 0.3 s), whose own plate
solves fail on fake stars, ride the 0.6 s tier's fitted model with a per-tier constant
pointing offset (the inter-tier drift is ~2 px; rotation differences over a 40 s window on
a clamped camera are negligible; the estimator's N1/N2 absorb what remains).

Per-star combination, per the plan's doctrine:
  - per tier: displacement = (obs - epoch-corrected catalogue), converted to sensor axes;
    the tier's MEDIAN displacement vector is subtracted (kills per-tier pointing constants
    so tiers can be mixed);
  - per star: the MEDIAN across tiers, with a cross-tier consistency vet — a star whose
    tiers disagree by more than 3x the field's cross-tier MAD is dropped (this is the
    automatic form of the vetting that caught V 9.10's corrupted centroid);
  - doubles dropped at the catalogue (the V 8.97 lesson), mag <= 11, R > 2 R_sun.

all87 is deliberately NOT a union member: its frames are the same photons as the tiers,
so including it would double-count, not deepen.

S2_LIMIT_MAG (env var, default 11.0) sets the catalogue limiting magnitude. Stars fainter
than 11.0 are admitted ONLY when detected in >= 2 tiers, so the cross-tier consistency vet
actually applies to them -- a single-tier faint match has no second witness and stays out.
Note a deeper catalogue also tightens the doubles and blend filters on the BRIGHT stars
(an 11-12 mag companion inside 10 arcsec now flags its primary), so the mag <= 11 subset
under a deeper catalogue is not guaranteed identical to the default run; measure the
drift, don't assume it.

Outputs: L (base and vertical-deg-2 nuisance) for the full union and for the 0.6+1.2
subset Douglas asked about, each with a star-resampling bootstrap for the statistical
error, with and without the below-Sun anchor.
"""
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

BL = r"D:/MEE2024 output/MEE_output/step3_prelim_L"
V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
PS, NX, NY, W_NORM = 2.2054043, 6248, 4176, 3124.0
SUNPX, SUNPY = 3171.0, 3232.0
R_SUN_AS, L_REF = 947.1, 1.7512
LIMIT_MAG = float(os.environ.get('S2_LIMIT_MAG', '11.0'))
MIDT = {'0p1s':'18:28:32','0p3s':'18:28:34','0p6s':'18:28:33','1p2s':'18:28:32'}
OPTS = dict(observation_date='2026-08-12', observation_lat=42.740470,
            observation_long=-5.613780, observation_height=1101.0,
            observation_temp=29.2, observation_pressure=896.7,
            observation_humidity=0.208, observation_wavelength=0.62,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            guess_date=False, distortionOrder='cubic')

# ---- the 0.6 s fitted model (host model for every tier)
src = glob.glob(os.path.join(BL, '0p6s', 'stage2_constant', 'distortion_data*.zip'))[0]
z6 = zipfile.ZipFile(src)
res = json.load(z6.open([n for n in z6.namelist() if n.endswith('distortion_results.txt')][0]))
q = (np.radians(res['platescale (arcseconds/pixel)']/3600.0),
     np.radians(res['RA']), np.radians(res['DEC']), np.radians(res['ROLL']))
cxd, cyd = res['distortion coeffs x'], res['distortion coeffs y']
names = list(cxd.keys())
CX = np.array([cxd[k] for k in names]); CY = np.array([cyd[k] for k in names])

def chain(det_pypx):
    """(py,px) -> (dec,ra) degrees through the 0.6 s model ('linear' variant: polynomial
    in the original frame, axis flip at the rotation -- validated to 0.0000 arcsec)."""
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q, plate, CX, CY, (NY, NX), OPTS)
    vec = transforms.linear_transform(q, plate_c, (NY, NX))
    return transforms.to_polar(vec)

# ---- catalogue, epoch-propagated and refraction/aberration-corrected per tier time
prov = providers.GaiaOfflineProvider.from_installed()
epoch = date_string_to_float(OPTS['observation_date'])
cat = prov.lookup((res['RA']-2.6, res['RA']+2.6), (res['DEC']-2.2, res['DEC']+2.2),
                  LIMIT_MAG, epoch=epoch)
not_dbl = ~np.asarray(cat.is_double(10.0))
corrs = {}
for t, tm in MIDT.items():
    c, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(
        cat, dict(OPTS, observation_time=tm))
    corrs[t] = (np.degrees(c.get_ra()), np.degrees(c.get_dec()))
cmag = np.asarray(cat.get_mags())

# sensor-axis conversion for sky displacements: the linear part of an affine fitted from
# the 0.6 s matched table (rotation/scale only -- exact enough for displacement rotation)
d6 = pd.read_csv(z6.open([n for n in z6.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
d6.columns = [c.strip() for c in d6.columns]
ra0, de0 = d6['RA(catalog)'].mean(), d6['DEC(catalog)'].mean()
Xa = (d6['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Ya = d6['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax, *_ = np.linalg.lstsq(Aa, d6['px'].values, rcond=None)
ay, *_ = np.linalg.lstsq(Aa, d6['py'].values, rcond=None)

def design(x_px, y_px, rx_, ry_, R_, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ux, uy = rx_/R_, ry_/R_
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ux*R_SUN_AS/R_]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, uy*R_SUN_AS/R_]
    labels = ['N1','N2','Th','L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0: continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels

# ---- per-tier star tables
tier_tabs = {}
for t in ('0p1s','0p3s','0p6s','1p2s'):
    det = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(V4, t, 'centroid_data*.zip'))[0])
                      .open('STACKED_CENTROIDS_DATA.csv'))
    sky = chain(det[['py','px']].values.astype(float))
    cra, cdec = corrs[t]
    # per-tier constant offset: median residual of provisional wide-gate matches
    rows = {}
    # gate 4.5: must exceed the largest PHYSICAL displacement (the anchor's
    # deflection-plus-edge-field offset is 2.4-2.9 arcsec); the cross-tier
    # consistency vet is the junk filter, not the gate
    for gate, collect in ((8.0, False), (4.5, True)):
        prov_d = []
        rows = {}
        for i in np.where(not_dbl)[0]:
            d = np.hypot((sky[:,1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:,0]-cdec[i])*3600
            j = int(np.argmin(d))
            if d[j] < gate:
                dxi = (sky[j,1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
                deta = (sky[j,0]-cdec[i])*3600
                prov_d.append((i, j, dxi, deta))
        if not collect:
            off = (np.median([p[2] for p in prov_d]), np.median([p[3] for p in prov_d]))
            sky[:,1] -= off[0]/3600/np.cos(np.radians(de0))
            sky[:,0] -= off[1]/3600
        else:
            seen = {}
            for i, j, dxi, deta in prov_d:
                seen.setdefault(j, []).append((i, dxi, deta))
            for j, cl in seen.items():
                if len(cl) > 1: continue                      # blend
                i, dxi, deta = cl[0]
                dpx = ax[0]*dxi/3600 + ax[1]*deta/3600        # sky delta -> sensor px
                dpy = ay[0]*dxi/3600 + ay[1]*deta/3600
                rows[i] = (det['px'][j], det['py'][j], dpx*PS, dpy*PS)
    tab = pd.DataFrame([dict(cat_i=i, px=v[0], py=v[1], dx=v[2], dy=v[3], mag=cmag[i])
                        for i, v in rows.items()])
    tab['dx'] -= tab['dx'].median(); tab['dy'] -= tab['dy'].median()
    tier_tabs[t] = tab
    print(f'{t}: {len(tab)} stars matched (offset-corrected, blends dropped)', flush=True)

# per-tier transparency: each tier's own v-deg2 L from its table alone
def _tier_L(tab):
    rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
    R = np.hypot(rx, ry)
    ok = (R > 2.0*R_SUN_AS) & (tab.mag.values <= 11.0)
    if ok.sum() < 15: return None, int(ok.sum())
    A, labels = design(tab.px.values[ok], tab.py.values[ok], rx[ok], ry[ok], R[ok], 2)
    c, *_ = np.linalg.lstsq(A, np.concatenate([tab.dx.values[ok], tab.dy.values[ok]]), rcond=None)
    return c[labels.index('L')], int(ok.sum())

# ---- the union: per-star median across tiers + cross-tier consistency vet
def build_union(tiers):
    per = {}
    for t in tiers:
        for _, r in tier_tabs[t].iterrows():
            per.setdefault(int(r.cat_i), []).append((r.px, r.py, r.dx, r.dy))
    recs, spreads = [], []
    for i, vs in per.items():
        vs = np.array(vs)
        recs.append((i, np.median(vs[:,0]), np.median(vs[:,1]),
                     np.median(vs[:,2]), np.median(vs[:,3]), len(vs),
                     np.hypot(vs[:,2].max()-vs[:,2].min(), vs[:,3].max()-vs[:,3].min())))
    U = pd.DataFrame(recs, columns=['cat_i','px','py','dx','dy','ntier','spread'])
    U['mag'] = cmag[U['cat_i'].values]
    # cross-tier consistency vet (only meaningful for ntier >= 2)
    sp = U.loc[U.ntier >= 2, 'spread']
    lim = 3*1.4826*np.median(np.abs(sp - sp.median())) + sp.median()
    bad = (U.ntier >= 2) & (U.spread > max(lim, 1.5))
    if bad.any():
        for _, r in U[bad].iterrows():
            print(f'  vetted OUT: G {r["mag"]:.2f} at ({r.px:.0f},{r.py:.0f}) '
                  f'cross-tier spread {r.spread:.2f} arcsec', flush=True)
    U = U[~bad]
    rx, ry = (U.px.values-SUNPX)*PS, (U.py.values-SUNPY)*PS
    R = np.hypot(rx, ry)
    # mag > 11 only with >= 2 tiers (the vet must apply); identical to mag <= 11 at default
    okmag = (U.mag.values <= 11.0) | ((U.mag.values <= LIMIT_MAG) & (U.ntier.values >= 2))
    ok = (R > 2.0*R_SUN_AS) & okmag
    return U[ok], rx[ok], ry[ok], R[ok]

def fit_L(U, rx, ry, R, nuis_deg=None):
    A, labels = design(U.px.values, U.py.values, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([U.dx.values, U.dy.values]), rcond=None)
    return c[labels.index('L')]

def report(name, tiers):
    U, rx, ry, R = build_union(tiers)
    anchor = np.hypot(U.px.values-3161, U.py.values-4163) < 6
    rng = np.random.default_rng(3)
    for tag, sel in (('with anchor', np.ones(len(U), bool)), ('sans anchor', ~anchor)):
        Us, rxs, rys, Rs = U[sel], rx[sel], ry[sel], R[sel]
        Lb = fit_L(Us, rxs, rys, Rs)
        Lv = fit_L(Us, rxs, rys, Rs, nuis_deg=2)
        boots = []
        n = len(Us)
        for _ in range(200):
            k = rng.integers(0, n, n)
            try: boots.append(fit_L(Us.iloc[k], rxs[k], rys[k], Rs[k], nuis_deg=2))
            except Exception: pass
        se = np.std(boots, ddof=1)
        h = 1/np.mean((R_SUN_AS/Rs)**2)
        print(f'{name:14} {tag:12} N={n:3d} h={h:5.1f}  L base {Lb:+.3f}  '
              f'L v-deg2 {Lv:+.3f} +- {se:.3f} (stat)  [GR {L_REF}]', flush=True)

print()
for t in ('0p1s','0p3s','0p6s','1p2s'):
    Lt, nt = _tier_L(tier_tabs[t])
    print(f'  {t} alone: N={nt}, L v-deg2 = {Lt if Lt is None else round(Lt,3)}')
print()
report('0.6+1.2', ('0p6s','1p2s'))
report('FULL UNION', ('0p1s','0p3s','0p6s','1p2s'))
print('done', flush=True)
