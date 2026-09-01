"""EC05, rescued without a plate solve -- the pointing implied from its siblings.

EC05 is the one night field neither solver cracks on any of its four centroid lists (the
v2 and classic solvers both fail, with and without darks -- the 2026-08-25 comparison
week hit the same wall, hence the bruns2017_nights_dark experiments). Douglas' point:
the pointing is not actually unknown. EC01-EC05 are five visits to the SAME field, seven
minutes apart, and EC04's fitted model is in hand. So:

  1. EC05's centroids are projected to the sky through EC04's fitted model (the same
     chain the union machinery uses, round-trip validated at 0.05 arcsec);
  2. matched to the corrected catalogue at EC05's own epoch (2017-08-19 ~04:26:20 UT,
     seven minutes after EC04, following the sequence spacing) with an 8 arcsec gate --
     wide enough for the mount's repointing scatter;
  3. the quadratic-free fit is then run by mee2024's OWN machinery
     (`distortion_polynomial._cubic_helper`, called three times exactly as
     `do_cubic_fit` calls it, with the same 15-field frozen cubic) -- so the residuals
     are the same construction as every other field on the map, not an imitation;
  4. a TWOD_RESIDUALS.csv and a minimal distortion_results.txt are written where the
     M3 map builder expects them.

The one thing this cannot do is discover a wildly wrong pointing -- but the match count
and rms say immediately if the seed was wrong, and both are printed.
"""
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections, _cubic_helper, _open_distortion_files
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUTM3 = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_m3"
NX, NY = 3296, 2472
DATE, TIME = '2017-08-19', '04:26:20'
NIGHTREFS = json.load(open(glob.glob(
    r'D:/MEE2024 output/MEE_output/bruns2017_lr/L/stage2/DISTORTION_OUTPUT*/distortion/'
    r'distortion_results.txt')[0], encoding='utf-8'))['fixed distortion reference files']

OPTS = dict(observation_date=DATE, observation_time=TIME,
            observation_lat=42.7363889, observation_long=-106.3180556,
            observation_height=2400.0, observation_temp=13.0,
            observation_pressure=770.0, observation_humidity=0.4,
            observation_wavelength=0.625, enable_corrections=True,
            enable_corrections_ref=True, enable_gravitational_def=False,
            gravity_sweep=False, guess_date=False, distortionOrder='cubic',
            distortion_fixed_coefficients='quadratic',
            distortion_reference_files=NIGHTREFS, no_plot=True)

# ---- 1. the seed: EC04's fitted model
seed = glob.glob(os.path.join(NIGHTS, 'EC04', 'stage2', '**', 'distortion_results.txt'),
                 recursive=True)[0]
js = json.load(open(seed, encoding='utf-8'))
q0 = (np.radians(js['platescale (arcseconds/pixel)']/3600.0),
      np.radians(js['RA']), np.radians(js['DEC']), np.radians(js['ROLL']))
CXs = np.array(list(js['distortion coeffs x'].values()))
CYs = np.array(list(js['distortion coeffs y'].values()))


def chain_seed(det_pypx):
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q0, plate, CXs, CYs, (NY, NX), OPTS)
    return transforms.to_polar(transforms.linear_transform(q0, plate_c, (NY, NX)))


det = pd.read_csv(zipfile.ZipFile(glob.glob(os.path.join(NIGHTS, 'EC05',
                                                         'centroid_data*.zip'))[0])
                  .open('STACKED_CENTROIDS_DATA.csv'))
sky = chain_seed(det[['py', 'px']].values.astype(float))

# ---- 2. match to the corrected catalogue at EC05's own epoch
prov = providers.GaiaOfflineProvider.from_installed()
cat = prov.lookup((js['RA']-1.5, js['RA']+1.5), (js['DEC']-1.2, js['DEC']+1.2), 13.0,
                  epoch=date_string_to_float(DATE))
cc, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(cat, OPTS)
cra, cdec = np.degrees(cc.get_ra()), np.degrees(cc.get_dec())
cmag = np.asarray(cat.get_mags())
# The seed lands the FIELD but not the exact pointing: EC05's offset from EC04 turned
# out to be larger than any tight gate (a direct 8 arcsec match found 10 stars at a
# median 5 arcsec -- the tail of a distribution centred elsewhere). So find the
# translation first: the 2-D histogram of ALL detection-to-catalogue position
# differences peaks at the true offset, junk detections contributing only a flat floor.
cosd = np.cos(np.radians(np.median(cdec)))
dra_all, dde_all = [], []
for i in range(len(cra)):
    dra = (sky[:, 1]-cra[i])*cosd*3600
    dde = (sky[:, 0]-cdec[i])*3600
    keep = (np.abs(dra) < 300) & (np.abs(dde) < 300)
    dra_all.append(dra[keep]); dde_all.append(dde[keep])
dra_all = np.concatenate(dra_all); dde_all = np.concatenate(dde_all)
H, xe, ye = np.histogram2d(dra_all, dde_all, bins=150, range=[[-300, 300], [-300, 300]])
pk = np.unravel_index(np.argmax(H), H.shape)
off_ra = 0.5*(xe[pk[0]] + xe[pk[0]+1]); off_de = 0.5*(ye[pk[1]] + ye[pk[1]+1])
print('translation search: peak offset (%.1f, %.1f) arcsec, %d pairs in the peak bin'
      % (off_ra, off_de, int(H[pk])))
sky[:, 1] -= off_ra/3600/cosd
sky[:, 0] -= off_de/3600
pairs = []
for i in range(len(cra)):
    d = np.hypot((sky[:, 1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:, 0]-cdec[i])*3600
    k = int(np.argmin(d))
    if d[k] < 8.0:
        pairs.append((k, i, d[k]))
# one-to-one: keep the closest catalogue claim per detection
best = {}
for k, i, dd in pairs:
    if k not in best or dd < best[k][1]:
        best[k] = (i, dd)
det_idx = np.array(sorted(best))
cat_idx = np.array([best[k][0] for k in det_idx])
print('EC05: %d matches through the EC04 seed (median sep %.2f arcsec)'
      % (len(det_idx), float(np.median([best[k][1] for k in det_idx]))))
assert len(det_idx) > 200, 'seed match failed'

# ---- 3. the quadratic-free fit, by the pipeline's own machinery
plate = det[['py', 'px']].values[det_idx].astype(float) - np.array([NY/2, NX/2])
target = transforms.icoord_to_vector(np.radians(np.c_[cdec[cat_idx], cra[cat_idx]]))
w = max(NY, NX)/2
m = 1
fix_x, fix_y, fix_ps, _ = _open_distortion_files(OPTS)
q = q0
for _ in range(3):
    q, plate_corr, coeff_x, coeff_y, basis, errors, reg_x, reg_y, ps_err = _cubic_helper(
        q, plate, target, w, m, fix_x, fix_y, OPTS)
ps_fit = np.degrees(q[0])*3600
# residuals in arcsec, sensor axes: observed minus model, from the helper's own errors
detrans = transforms.detransform_vectors(q, target)
resid_px = plate_corr - detrans          # (y, x) pixels, model-corrected obs vs predicted
dx_as = resid_px[:, 1]*ps_fit
dy_as = resid_px[:, 0]*ps_fit
rms = float(np.sqrt(np.mean(dx_as**2 + dy_as**2)/2))
print('EC05 rescued fit: %d stars, ps %.7f (siblings ~2.0878), rms %.4f arcsec/axis'
      % (len(det_idx), ps_fit, rms))

# ---- 4. write what the map builder expects
out = os.path.join(OUTM3, 'EC05', 'rescued', 'distortion')
os.makedirs(out, exist_ok=True)
pd.DataFrame(dict(px=det['px'].values[det_idx], py=det['py'].values[det_idx],
                  dx_px=dx_as/ps_fit, dy_px=dy_as/ps_fit,
                  dx_arcsec=dx_as, dy_arcsec=dy_as,
                  error_arcsec=np.hypot(dx_as, dy_as),
                  radius_px=np.hypot(det['px'].values[det_idx]-NX/2,
                                     det['py'].values[det_idx]-NY/2),
                  magV=cmag[cat_idx],
                  ID=['gaia:%d' % i for i in cat_idx])) \
  .to_csv(os.path.join(out, 'TWOD_RESIDUALS.csv'), index=False)
res = {'#stars used': int(len(det_idx)),
       'final rms error (arcseconds)': rms*np.sqrt(2),
       'platescale (arcseconds/pixel)': ps_fit,
       'RA': float(np.degrees(q[1])), 'DEC': float(np.degrees(q[2])),
       'ROLL': float(np.degrees(q[3])),
       'observation_date': DATE, 'observation_time (UTC)': TIME,
       'fixed distortion order': 'quadratic',
       'note': ('rescued: no plate solve succeeds on this field; pointing seeded from '
                'EC04 (same field, 7 min earlier) and the quadratic-free fit run by '
                'distortion_polynomial._cubic_helper with the same frozen cubic as '
                'every other field. tools/matrix_bruns/b17_rescue_ec05.py')}
json.dump(res, open(os.path.join(out, 'distortion_results.txt'), 'w'), indent=4)
print('written ->', out)
