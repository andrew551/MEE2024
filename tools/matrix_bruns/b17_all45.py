"""Matrix cell 1: the all-45 stack -- every preprocessed frame in one stage-1 run.

This is how the first MEE analysis of this dataset was done (all 45 frames at once,
hull blob 20/10, 255 centroids, matched V 7.78 at 1.49 R_sun) and Douglas asked for it
through the current chain. Per the Leon all87 doctrine it is a DEPTH PROBE and
consistency check, not a union member -- its frames are the same photons as the tiers,
so admitting it to the union would double-count.

Mechanics: unweighted mean of all 45 preprocessed frames (matched-filter weights are
S/sigma^2 = exp/exp = 1 for sky-limited mixed exposures); each tier keeps its own
forbidden disk (EA/EB 921/922 px, E2 666 px), so the 666-921 px annulus carries E2
signal diluted 11/45 against painted pedestal. Frame order: EA, then EB, then E2 --
the stacker aligns to the FIRST frame (F23), and EA is the best-detected tier;
inter-tier drift is 7-14 px against pxl_tol=10, so the E2 frames' alignment is
checked in the log rather than assumed. Weighted mid-time = 17:43:47 UT (= E2's mid).

Stage 2: constant-only against the frozen L+R8 bracket, identical settings to the
tiers. Then the estimator on the all45 table itself (chained through its OWN fitted
model, G <= 11 corrected catalogue, wide/collect gates 8.0/4.5 arcsec, R > 2 R_sun
and an R > 1.45 variant), 200-resample bootstrap.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
sys.path.insert(0, REPO)
B = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
LR = r"D:/MEE2024 output/MEE_output/bruns2017_lr"
REF_L = glob.glob(os.path.join(LR, 'L', 'stage2', 'DISTORTION_OUTPUT*', 'distortion', 'distortion_results.txt'))[0]
REF_R = glob.glob(os.path.join(LR, 'R8', 'stage2', 'DISTORTION_OUTPUT*', 'distortion', 'distortion_results.txt'))[0]
MIDT = '17:43:47'
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']
STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

out = os.path.join(B, 'all45')
os.makedirs(out, exist_ok=True)
frames = (sorted(glob.glob(os.path.join(B, 'EA', 'preprocessed', '*.fits')))
          + sorted(glob.glob(os.path.join(B, 'EB', 'preprocessed', '*.fits')))
          + sorted(glob.glob(os.path.join(B, 'E2', 'preprocessed', '*.fits'))))
assert len(frames) == 45, len(frames)
zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
if not zips:
    run([PY,'-m','mee2024.cli','stack',*frames,*STAGE1,
         '--no-scan','--no-display','--quiet','-o',out], os.path.join(out,'stage1.log'))
    zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
n = json.load(zipfile.ZipFile(zips[0]).open('results.txt'))['n_centroids']
print(f'all45: {n} centroids (the original-era analysis had 255 with the hull blob)', flush=True)

d2 = os.path.join(out, 'stage2_constant')
os.makedirs(d2, exist_ok=True)
if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
    run([PY,'-m','mee2024.cli','distortion',zips[0],
         '--order','cubic','--date-from-header','--fix-distortion',REF_L,REF_R,
         '--set','distortion_fixed_coefficients=constant',
         '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
         '--set','rough_match_threshhold=36',
         '--set','enable_corrections=True','--set','enable_corrections_ref=True',*SITE,
         '--set','observation_time='+MIDT,'--no-display','--quiet','-o',d2],
        os.path.join(d2, 'stage2.log'))
res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
assert res, 'stage 2 failed'
j = json.load(open(res[0], encoding='utf-8'))
print(f"all45: {j['#stars used']} matched (to mag 13), rms "
      f"{j['final rms error (arcseconds)']:.4f} arcsec, imported ps "
      f"{j['platescale (arcseconds/pixel)']:.7f}", flush=True)

# ---- the estimator on the all45 table, through its OWN fitted model
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
L_REF = 1.7512
OPTS = dict(observation_date='2017-08-21', observation_lat=42.7363889,
            observation_long=-106.3180556, observation_height=2400.0,
            observation_temp=13.0, observation_pressure=770.0,
            observation_humidity=0.4, observation_wavelength=0.625,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            guess_date=False, distortionOrder='cubic', observation_time=MIDT)
zh = zipfile.ZipFile(glob.glob(os.path.join(d2, 'distortion_data*.zip'))[0])
resj = json.load(zh.open([m for m in zh.namelist() if m.endswith('distortion_results.txt')][0]))
q = (np.radians(resj['platescale (arcseconds/pixel)']/3600.0),
     np.radians(resj['RA']), np.radians(resj['DEC']), np.radians(resj['ROLL']))
cxd, cyd = resj['distortion coeffs x'], resj['distortion coeffs y']
names = list(cxd.keys())
CX = np.array([cxd[k] for k in names]); CY = np.array([cyd[k] for k in names])
def chain(det_pypx):
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q, plate, CX, CY, (NY, NX), OPTS)
    return transforms.to_polar(transforms.linear_transform(q, plate_c, (NY, NX)))
dh = pd.read_csv(zh.open([m for m in zh.namelist() if m.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
got = chain(dh[['py','px']].values.astype(float))
rt = float((np.hypot((got[:,1]-dh['RA(obs)'])*np.cos(np.radians(dh['DEC(obs)'])),
                     got[:,0]-dh['DEC(obs)'])*3600).max())
assert rt < 0.05, f'round-trip {rt}'

prov = providers.GaiaOfflineProvider.from_installed()
cat = prov.lookup((resj['RA']-1.5, resj['RA']+1.5), (resj['DEC']-1.2, resj['DEC']+1.2),
                  11.0, epoch=date_string_to_float('2017-08-21'))
not_dbl = ~np.asarray(cat.is_double(10.0))
cc, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(cat, OPTS)
cra, cdec = np.degrees(cc.get_ra()), np.degrees(cc.get_dec())
cmag = np.asarray(cat.get_mags())
ra0, de0 = dh['RA(catalog)'].mean(), dh['DEC(catalog)'].mean()
Xa = (dh['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Ya = dh['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax_, *_ = np.linalg.lstsq(Aa, dh['px'].values, rcond=None)
ay_, *_ = np.linalg.lstsq(Aa, dh['py'].values, rcond=None)
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
T = Time('2017-08-21T17:43:47', scale='utc')
sun = get_sun(T)
R_SUN_AS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
sx = (sun.ra.deg-ra0)*np.cos(np.radians(de0)); sy = sun.dec.deg-de0
SUNPX = float(np.array([sx, sy, 1.0])@ax_); SUNPY = float(np.array([sx, sy, 1.0])@ay_)
SITEC = EarthLocation(lat=42.7363889*u.deg, lon=-106.3180556*u.deg, height=2400*u.m)
fc = SkyCoord(resj['RA']*u.deg, resj['DEC']*u.deg)
aa = fc.transform_to(AltAz(obstime=T, location=SITEC))
up = SkyCoord(AltAz(alt=aa.alt+0.1*u.deg, az=aa.az, obstime=T, location=SITEC)).icrs
dv = np.array([(up.ra.deg-fc.ra.deg)*np.cos(np.radians(de0)), up.dec.deg-fc.dec.deg])
vpx = float(dv@ax_[:2]); vpy = float(dv@ay_[:2])
VX, VY = vpx/np.hypot(vpx, vpy), vpy/np.hypot(vpx, vpy)
print(f'Sun pixel ({SUNPX:.0f},{SUNPY:.0f}); vertical ({VX:+.3f},{VY:+.3f})', flush=True)

det = pd.read_csv(zipfile.ZipFile(zips[0]).open('STACKED_CENTROIDS_DATA.csv'))
sky = chain(det[['py','px']].values.astype(float))
rows = {}
for gate, collect in ((8.0, False), (4.5, True)):
    prov_d = []
    rows = {}
    for i in np.where(not_dbl)[0]:
        d = np.hypot((sky[:,1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:,0]-cdec[i])*3600
        jj = int(np.argmin(d))
        if d[jj] < gate:
            dxi = (sky[jj,1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
            deta = (sky[jj,0]-cdec[i])*3600
            prov_d.append((i, jj, dxi, deta))
    if not collect:
        off = (np.median([p[2] for p in prov_d]), np.median([p[3] for p in prov_d]))
        sky[:,1] -= off[0]/3600/np.cos(np.radians(de0))
        sky[:,0] -= off[1]/3600
    else:
        seen = {}
        for i, jj, dxi, deta in prov_d:
            seen.setdefault(jj, []).append((i, dxi, deta))
        for jj, cl in seen.items():
            if len(cl) > 1: continue
            i, dxi, deta = cl[0]
            rows[i] = (det['px'][jj], det['py'][jj],
                       (ax_[0]*dxi/3600 + ax_[1]*deta/3600)*PS,
                       (ay_[0]*dxi/3600 + ay_[1]*deta/3600)*PS)
tab = pd.DataFrame([dict(cat_i=i, px=v[0], py=v[1], dx=v[2], dy=v[3], mag=cmag[i])
                    for i, v in rows.items()])
tab['dx'] -= tab['dx'].median(); tab['dy'] -= tab['dy'].median()
print(f'all45 table: {len(tab)} stars matched at G <= 11', flush=True)

def design(x_px, y_px, rx_, ry_, R_, nd=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx_/R_, ry_/R_
    m = len(x_px); Z = np.zeros(m)
    cols_x = [np.ones(m), Z, -(y_px-NY/2)*PS, ur*R_SUN_AS/R_]
    cols_y = [Z, np.ones(m), (x_px-NX/2)*PS, vr*R_SUN_AS/R_]
    labels = ['N1','N2','Th','L']
    if nd:
        for i in range(nd+1):
            for jj in range(nd+1-i):
                if i == 0 and jj == 0: continue
                cols_x.append(VX*xs**i*ys**jj); cols_y.append(VY*xs**i*ys**jj)
                labels.append(f'v{i}{jj}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels

rng = np.random.default_rng(3)
for rcut in (2.0, 1.45):
    rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
    R = np.hypot(rx, ry)
    ok = (R > rcut*R_SUN_AS) & (tab.mag.values <= 11.0)
    Ut, rxs, rys, Rs = tab[ok], rx[ok], ry[ok], R[ok]
    m = len(Ut)
    def fitL(kk=None, nd=2):
        sel = slice(None) if kk is None else kk
        A, labels = design(Ut.px.values[sel], Ut.py.values[sel], rxs[sel], rys[sel], Rs[sel], nd)
        c, *_ = np.linalg.lstsq(A, np.concatenate([Ut.dx.values[sel], Ut.dy.values[sel]]), rcond=None)
        return c[labels.index('L')]
    bb, bv = [], []
    for _ in range(200):
        kk = rng.integers(0, m, m)
        try:
            bb.append(fitL(kk, None)); bv.append(fitL(kk, 2))
        except Exception: pass
    h = 1/np.mean((R_SUN_AS/Rs)**2)
    print(f'all45 R>{rcut}: N={m} h={h:.1f} Rsun^2  '
          f'L base {fitL(nd=None):+.3f} +- {np.std(bb, ddof=1):.3f}  '
          f'L v-deg2 {fitL(nd=2):+.3f} +- {np.std(bv, ddof=1):.3f} (stat)  [GR {L_REF}]', flush=True)
    inner = (Rs < 2.0*R_SUN_AS)
    for kk in np.where(inner)[0]:
        ur, vr = rxs[kk]/Rs[kk], rys[kk]/Rs[kk]
        print(f'  inner: G {Ut.mag.values[kk]:.2f} R={Rs[kk]/R_SUN_AS:.2f} Rsun radial '
              f'{Ut.dx.values[kk]*ur + Ut.dy.values[kk]*vr:+.3f} arcsec '
              f'(GR {L_REF*R_SUN_AS/Rs[kk]:+.3f})', flush=True)
print('done', flush=True)
