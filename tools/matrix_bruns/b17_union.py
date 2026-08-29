"""Matrix cell 1 (Bruns 2017): two-pass rematch + cross-tier union + the estimator.

The Leon S2 machinery (tools/step3_s2_union.py) with this dataset's geometry supplied:
  - host model: EA's constant-only fit (frozen L/R bracket, ps 2.0868004 arcsec/px);
    every tier's detections ride it with a per-tier constant offset (inter-tier pointing
    drift here is 7-14 px, translation only on a clamped mount over 51 s);
  - per-star MEDIAN across tiers after per-tier median-offset removal; cross-tier
    consistency vet at 3x MAD (floor 1.5 arcsec); doubles (is_double 10 arcsec) and
    blends dropped; mag <= 11; R > 2 R_sun default.
  - the estimator: one lstsq over [N1, N2, Theta, L*(1/R)u_radial + nuisance], where the
    nuisance is a deg-2 polynomial amplitude along the LOCAL VERTICAL, computed from the
    field's AltAz geometry rather than assumed to be sensor-y. (Leon's sensor-y choice
    WAS the local vertical to 3.1 degrees; this is the same physics, generalised.)

Also reported, because this field earns it: the 'inner' variant with the radial cut at
1.45 R_sun -- the E2 (0.09 s) tier saturates at 1.42 R_sun and exists precisely to reach
the inner annulus (Bruns' scripted 2-star series). Its stars carry the highest leverage
in the dataset; the cross-tier vet and the value table decide admission, per the Leon
anchor doctrine (measured, not assumed).

Statistical error: star-resampling bootstrap (200). Method-1 plate-scale term computed
by hand at the corrected eq-23 units: dL = h * R_sun(as) * dS, with dS = 10.3 ppm (the
L+R combined HC3) and, as the bracket bound, 22.5 ppm (half the measured L-R split).
"""
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

B = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
L_REF = 1.7512
MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}
HOST = 'EA'
OPTS = dict(observation_date='2017-08-21', observation_lat=42.7363889,
            observation_long=-106.3180556, observation_height=2400.0,
            observation_temp=13.0, observation_pressure=770.0,
            observation_humidity=0.4, observation_wavelength=0.625,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            guess_date=False, distortionOrder='cubic')
SITE = EarthLocation(lat=OPTS['observation_lat']*u.deg, lon=OPTS['observation_long']*u.deg,
                     height=OPTS['observation_height']*u.m)
T_MID = Time('2017-08-21T17:43:47', scale='utc')
sun = get_sun(T_MID)
R_SUN_AS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
print(f'R_sun (2017-08-21 17:43:47 UT) = {R_SUN_AS:.1f} arcsec', flush=True)

# ---- the host model (EA constant-only vs the frozen L/R bracket)
src = glob.glob(os.path.join(B, HOST, 'stage2_constant', 'distortion_data*.zip'))[0]
zh = zipfile.ZipFile(src)
res = json.load(zh.open([n for n in zh.namelist() if n.endswith('distortion_results.txt')][0]))
q = (np.radians(res['platescale (arcseconds/pixel)']/3600.0),
     np.radians(res['RA']), np.radians(res['DEC']), np.radians(res['ROLL']))
cxd, cyd = res['distortion coeffs x'], res['distortion coeffs y']
names = list(cxd.keys())
CX = np.array([cxd[k] for k in names]); CY = np.array([cyd[k] for k in names])

def chain(det_pypx):
    """(py,px) -> (dec,ra) degrees through the host model ('linear' variant, the one the
    Leon round-trip gate selected: polynomial in the original frame, flip at rotation)."""
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q, plate, CX, CY, (NY, NX), OPTS)
    vec = transforms.linear_transform(q, plate_c, (NY, NX))
    return transforms.to_polar(vec)

# round-trip gate on the host's own matched stars, 0.05 arcsec (the Leon rule)
dh = pd.read_csv(zh.open([n for n in zh.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
got = chain(dh[['py','px']].values.astype(float))
rt = float((np.hypot((got[:,1]-dh['RA(obs)'])*np.cos(np.radians(dh['DEC(obs)'])),
                     got[:,0]-dh['DEC(obs)'])*3600).max())
print(f'host round-trip max {rt:.4f} arcsec (gate 0.05)', flush=True)
assert rt < 0.05, 'reconstruction failed the round-trip gate'

# ---- catalogue, epoch-propagated, corrected per tier time
prov = providers.GaiaOfflineProvider.from_installed()
epoch = date_string_to_float(OPTS['observation_date'])
cat = prov.lookup((res['RA']-1.5, res['RA']+1.5), (res['DEC']-1.2, res['DEC']+1.2),
                  11.0, epoch=epoch)
not_dbl = ~np.asarray(cat.is_double(10.0))
corrs = {}
for t, tm in MIDT.items():
    c, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(
        cat, dict(OPTS, observation_time=tm))
    corrs[t] = (np.degrees(c.get_ra()), np.degrees(c.get_dec()))
cmag = np.asarray(cat.get_mags())
cra0 = np.degrees(cat.get_ra()); cdec0 = np.degrees(cat.get_dec())

# affines: corrected-sky -> sensor (displacement rotation), uncorrected-sky -> px (Sun)
ra0, de0 = dh['RA(catalog)'].mean(), dh['DEC(catalog)'].mean()
Xa = (dh['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Ya = dh['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax, *_ = np.linalg.lstsq(Aa, dh['px'].values, rcond=None)
ay, *_ = np.linalg.lstsq(Aa, dh['py'].values, rcond=None)

# Sun pixel: apparent geocentric Sun RA/Dec through the matched-table affine
# (RA(catalog) <-> px). Two measured traps live here: (1) NOT sun.icrs -- transforming
# get_sun's GCRS to ICRS re-centres on the solar-system BARYCENTRE and the direction
# becomes meaningless (it put the Sun at px 148535); GCRS ra/dec is the apparent
# geocentric direction. (2) no cross-matching against a shallower catalogue -- a mag-13
# table star grabbed a mag-11 neighbour 334 arcsec off and corrupted the affine. The
# affine lives in corrected (refracted+aberrated) sky and this Sun position is
# geometric: ~26 arcsec at alt 54 deg = ~12 px, inside the ~25 px mask-centre tolerance.
sx = (sun.ra.deg-ra0)*np.cos(np.radians(de0)); sy = sun.dec.deg-de0
SUNPX = float(np.array([sx, sy, 1.0])@ax); SUNPY = float(np.array([sx, sy, 1.0])@ay)
print(f'Sun ephemeris pixel: ({SUNPX:.0f}, {SUNPY:.0f}) '
      f'(sat centroids were EA (1637,1749) E2 (1651,1748) EB (1644,1748))', flush=True)

# local vertical in sensor coordinates, from the AltAz geometry at the field centre
fc = SkyCoord(res['RA']*u.deg, res['DEC']*u.deg)
aa = fc.transform_to(AltAz(obstime=T_MID, location=SITE))
up = SkyCoord(AltAz(alt=aa.alt+0.1*u.deg, az=aa.az, obstime=T_MID, location=SITE)).icrs
ux_ = (up.ra.deg-ra0)*np.cos(np.radians(de0)); uy_ = up.dec.deg-de0
fx = (fc.ra.deg-ra0)*np.cos(np.radians(de0)); fy = fc.dec.deg-de0
dvx = np.array([ux_-fx, uy_-fy, 0.0])
vpx = float(dvx@np.r_[ax[:2], 0]); vpy = float(dvx@np.r_[ay[:2], 0])
VX, VY = (v/np.hypot(vpx, vpy) for v in (vpx, vpy))
print(f'local vertical in sensor coords: ({VX:+.3f}, {VY:+.3f}) '
      f'({np.degrees(np.arctan2(VX, VY)):+.1f} deg from sensor +y); field alt {aa.alt.deg:.1f} deg',
      flush=True)

# ---- per-tier star tables (wide-gate offset pass 8.0, collect pass 4.5)
tier_tabs = {}
for t in ('EA', 'E2', 'EB'):
    det = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(B, t, 'centroid_data*.zip'))[0])
                      .open('STACKED_CENTROIDS_DATA.csv'))
    sky = chain(det[['py','px']].values.astype(float))
    cra, cdec = corrs[t]
    rows = {}
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
                if len(cl) > 1: continue                       # blend
                i, dxi, deta = cl[0]
                dpx = ax[0]*dxi/3600 + ax[1]*deta/3600
                dpy = ay[0]*dxi/3600 + ay[1]*deta/3600
                rows[i] = (det['px'][j], det['py'][j], dpx*PS, dpy*PS)
    tab = pd.DataFrame([dict(cat_i=i, px=v[0], py=v[1], dx=v[2], dy=v[3], mag=cmag[i])
                        for i, v in rows.items()])
    tab['dx'] -= tab['dx'].median(); tab['dy'] -= tab['dy'].median()
    tier_tabs[t] = tab
    print(f'{t}: {len(tab)} stars matched (offset-corrected, blends dropped)', flush=True)

def design(x_px, y_px, rx_, ry_, R_, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx_/R_, ry_/R_
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ur*R_SUN_AS/R_]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, vr*R_SUN_AS/R_]
    labels = ['N1','N2','Th','L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0: continue
                cols_x.append(VX*xs**i*ys**j); cols_y.append(VY*xs**i*ys**j)
                labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels

def build_union(tiers, rcut):
    per = {}
    for t in tiers:
        for _, r in tier_tabs[t].iterrows():
            per.setdefault(int(r.cat_i), []).append((r.px, r.py, r.dx, r.dy))
    recs = []
    for i, vs in per.items():
        vs = np.array(vs)
        recs.append((i, np.median(vs[:,0]), np.median(vs[:,1]),
                     np.median(vs[:,2]), np.median(vs[:,3]), len(vs),
                     np.hypot(vs[:,2].max()-vs[:,2].min(), vs[:,3].max()-vs[:,3].min())))
    U = pd.DataFrame(recs, columns=['cat_i','px','py','dx','dy','ntier','spread'])
    U['mag'] = cmag[U['cat_i'].values]
    sp = U.loc[U.ntier >= 2, 'spread']
    lim = 3*1.4826*np.median(np.abs(sp - sp.median())) + sp.median()
    bad = (U.ntier >= 2) & (U.spread > max(lim, 1.5))
    for _, r in U[bad].iterrows():
        print(f'  vetted OUT: G {r["mag"]:.2f} at ({r.px:.0f},{r.py:.0f}) '
              f'cross-tier spread {r.spread:.2f} arcsec', flush=True)
    U = U[~bad]
    rx, ry = (U.px.values-SUNPX)*PS, (U.py.values-SUNPY)*PS
    R = np.hypot(rx, ry)
    ok = (R > rcut*R_SUN_AS) & (U.mag.values <= 11.0)
    return U[ok], rx[ok], ry[ok], R[ok]

def fit_L(U, rx, ry, R, nuis_deg=None):
    A, labels = design(U.px.values, U.py.values, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([U.dx.values, U.dy.values]), rcond=None)
    return c[labels.index('L')]

def report(name, tiers, rcut):
    U, rx, ry, R = build_union(tiers, rcut)
    rng = np.random.default_rng(3)
    n = len(U)
    if n < 12:
        print(f'{name:24} N={n} -- too few stars', flush=True); return
    Lb = fit_L(U, rx, ry, R)
    Lv = fit_L(U, rx, ry, R, nuis_deg=2)
    boots_b, boots_v = [], []
    for _ in range(200):
        k = rng.integers(0, n, n)
        try:
            boots_b.append(fit_L(U.iloc[k], rx[k], ry[k], R[k]))
            boots_v.append(fit_L(U.iloc[k], rx[k], ry[k], R[k], nuis_deg=2))
        except Exception: pass
    h = 1/np.mean((R_SUN_AS/R)**2)
    eq23 = h*R_SUN_AS*10.3e-6
    eq23b = h*R_SUN_AS*22.5e-6
    print(f'{name:24} N={n:3d} h={h:5.1f} Rsun^2  '
          f'L base {Lb:+.3f} +- {np.std(boots_b, ddof=1):.3f}  '
          f'L v-deg2 {Lv:+.3f} +- {np.std(boots_v, ddof=1):.3f} (stat)  '
          f'[eq23: {eq23:.3f}" @10.3ppm, {eq23b:.3f}" @22.5ppm; GR {L_REF}]', flush=True)
    return U, rx, ry, R

print()
for t in ('EA', 'E2', 'EB'):
    tab = tier_tabs[t]
    rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
    R = np.hypot(rx, ry)
    ok = (R > 2.0*R_SUN_AS) & (tab.mag.values <= 11.0)
    if ok.sum() >= 12:
        A, labels = design(tab.px.values[ok], tab.py.values[ok], rx[ok], ry[ok], R[ok], 2)
        c, *_ = np.linalg.lstsq(A, np.concatenate([tab.dx.values[ok], tab.dy.values[ok]]),
                                rcond=None)
        print(f'  {t} alone (R>2): N={ok.sum()}, L v-deg2 = {c[labels.index("L")]:+.3f}')
    else:
        print(f'  {t} alone (R>2): N={ok.sum()} -- too few')
print()
report('0.62 union (R>2.0)', ('EA','EB'), 2.0)
report('FULL union (R>2.0)', ('EA','E2','EB'), 2.0)
out = report('FULL union (R>1.45)', ('EA','E2','EB'), 1.45)
if out is not None:
    U, rx, ry, R = out
    inner = U[R < 2.0*R_SUN_AS]
    if len(inner):
        print('  inner-annulus stars (R < 2 R_sun):')
        for _, r in inner.iterrows():
            rr = np.hypot((r.px-SUNPX), (r.py-SUNPY))*PS/R_SUN_AS
            print(f'    G {r.mag:.2f} at ({r.px:.0f},{r.py:.0f}) R={rr:.2f} Rsun '
                  f'ntier={int(r.ntier)} spread={r.spread:.2f}" '
                  f'dx={r.dx:+.2f}" dy={r.dy:+.2f}"', flush=True)
    U.to_csv(os.path.join(B, 'union_full_r145.csv'), index=False)
    print('union table ->', os.path.join(B, 'union_full_r145.csv'), flush=True)
print('done', flush=True)
