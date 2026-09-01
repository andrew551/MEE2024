"""The structural question for Leon, measured: one 0.6+1.2 s master, or the per-star union?

Bruns' EA and EB are the same 0.62 s exposure 51 s apart, so one master is the natural
object and cell 1 built it. Leon's 0.6 s and 1.2 s are different exposures; the headline
used a per-star union across the two tiers (each tier stacked and fitted on its own, the
star's displacement the median over tiers, with a cross-tier consistency vet). The next
-session prompt asks that this be decided on its merits and the reason recorded, not
copied from cell 1. This builds the alternative and puts the two side by side.

The master: the 12 preprocessed 0.6 s frames and the 5 preprocessed 1.2 s frames of
step3_s0_v4 in ONE unweighted stack (the all-87 test showed an unweighted mean of mixed
exposures is photon-optimal: matched-filter weights are exposure/exposure = 1). One
adjustment is forced by the tiers' different saturation radii: the 1.2 s frames carry a
forbidden disk of 811 px, the 0.6 s frames 746 px, and a stack that mixes them would put
a 65 px ring of 12-real-plus-5-painted pixels around the disk. The 0.6 s frames are
therefore re-painted to the 1.2 s radius before stacking, so the master's inner boundary
is the deeper tier's (1.89 R_sun). The 0.6 s frames lead the list (the stacker aligns to
the first frame, F23; the 0.6 s tier is the better-detected one).

Then the headline chain: stage 1 with the headline's flags (windowed + annular), stage 2
constant-only against the canonical CAL_piLeo at the photon-weighted mid-time, and the
union tool's own matching (catalogue G <= 11, doubles at 10 arcsec dropped, gates 8 then
4.5 arcsec, blends dropped, median offset removed, R > 2 R_sun) applied to the master's
detections as a single tier. Same estimator, same bootstrap.

What the comparison cannot show and the record must say: a single master has ONE witness
per star, so the cross-tier vet -- the filter that removed the G 9.10 corrupted centroid
automatically -- has nothing to work with.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
OUT = r"D:/MEE2024 output/MEE_output/step3_record/master0612"
CAL = glob.glob(r"D:/MEE2024 output/MEE_output/cal_pileo_step2/canonical_16f_night2refs/"
                r"DISTORTION_OUTPUT*/distortion/distortion_results.txt")[0]
PS, NX, NY, W_NORM = 2.2054043, 6248, 4176, 3124.0
SUNPX, SUNPY = 3171.0, 3232.0
R_SUN_AS, L_REF = 947.1, 1.7512
SUN = (3171.0, 3232.0); PED = 2000.0
R_DISK_12 = max(1.25*R_SUN_AS/PS, 801 + 10)          # the 1.2 s tier's disk (step3_s0_v4)
MIDT = '18:28:33'
ANCHOR = (3161.0, 4163.0)
STAGE1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
          '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
          '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
          '--set', 'remove_edgy_centroids=True', '--set', 'centroid_refine_window=True',
          '--set', 'centroid_window_sigma=2.0', '--set', 'background_subtraction_mode=annular']
COMMON = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
          '--set', 'observation_lat=42.740470', '--set', 'observation_long=-5.613780',
          '--set', 'observation_height=1101', '--set', 'observation_temp=29.2',
          '--set', 'observation_pressure=896.7', '--set', 'observation_humidity=0.208',
          '--set', 'observation_wavelength=0.62']
OPTS = dict(observation_date='2026-08-12', observation_lat=42.740470,
            observation_long=-5.613780, observation_height=1101.0,
            observation_temp=29.2, observation_pressure=896.7,
            observation_humidity=0.208, observation_wavelength=0.62,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            guess_date=False, distortionOrder='cubic', observation_time=MIDT)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


# ---- 1. the master's frames: 0.6 s re-painted to the 1.2 s disk, then the 1.2 s as they are
pre = os.path.join(OUT, 'preprocessed')
os.makedirs(pre, exist_ok=True)
f06 = sorted(glob.glob(os.path.join(V4, '0p6s', 'preprocessed', '*.fits')))
f12 = sorted(glob.glob(os.path.join(V4, '1p2s', 'preprocessed', '*.fits')))
assert len(f06) == 12 and len(f12) == 5, (len(f06), len(f12))
frames = []
for k, f in enumerate(f06 + f12):
    dst = os.path.join(pre, '%02d_%s' % (k, os.path.basename(f)))
    if not os.path.exists(dst):
        with fits.open(f) as hd:
            img = hd[0].data.astype(np.float64); hdr = hd[0].header.copy()
        if k < len(f06):
            ny, nx = img.shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            img[np.hypot(xx - SUN[0], yy - SUN[1]) <= R_DISK_12] = PED
            hdr['HISTORY'] = 'master0612: forbidden disk enlarged to the 1.2 s radius %.0f px' % R_DISK_12
        fits.writeto(dst, img.astype(np.uint16), hdr, overwrite=True)
    frames.append(dst)
print('master0612: %d frames (12 x 0.6 s re-painted to r=%.0f px, 5 x 1.2 s)' % (len(frames), R_DISK_12),
      flush=True)

# ---- 2. stage 1 (headline flags) and stage 2 constant-only against the canonical CAL
z = glob.glob(os.path.join(OUT, 'centroid_data*.zip'))
if not z:
    run([PY, '-m', 'mee2024.cli', 'stack', *frames, *STAGE1, '--no-scan', '--no-display',
         '--quiet', '-o', OUT], os.path.join(OUT, 'stage1.log'))
    z = glob.glob(os.path.join(OUT, 'centroid_data*.zip'))
assert z, 'stage 1 failed'
n_det = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
d2 = os.path.join(OUT, 'stage2_constant')
os.makedirs(d2, exist_ok=True)
if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
    run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic', '--date-from-header',
         '--fix-distortion', CAL, '--set', 'distortion_fixed_coefficients=constant',
         '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
         '--set', 'rough_match_threshhold=36', *COMMON, '--set', 'observation_time=' + MIDT,
         '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
assert res, 'stage 2 failed'
j = json.load(open(res[0], encoding='utf-8'))
print('master0612: %d centroids -> %d matched, rms %.4f arcsec, imported ps %.7f'
      % (n_det, j['#stars used'], j['final rms error (arcseconds)'],
         j['platescale (arcseconds/pixel)']), flush=True)

# ---- 3. the union tool's matching, applied to the master as a single tier
q = (np.radians(j['platescale (arcseconds/pixel)']/3600.0), np.radians(j['RA']),
     np.radians(j['DEC']), np.radians(j['ROLL']))
cxd, cyd = j['distortion coeffs x'], j['distortion coeffs y']
names = list(cxd.keys())
CX = np.array([cxd[k] for k in names]); CY = np.array([cyd[k] for k in names])


def chain(det_pypx):
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q, plate, CX, CY, (NY, NX), OPTS)
    return transforms.to_polar(transforms.linear_transform(q, plate_c, (NY, NX)))


zh = zipfile.ZipFile(glob.glob(os.path.join(d2, 'distortion_data*.zip'))[0])
dh = pd.read_csv(zh.open([n for n in zh.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
got = chain(dh[['py', 'px']].values.astype(float))
rt = float((np.hypot((got[:, 1]-dh['RA(obs)'])*np.cos(np.radians(dh['DEC(obs)'])),
                     got[:, 0]-dh['DEC(obs)'])*3600).max())
assert rt < 0.05, 'round-trip gate failed: %.4f' % rt
ra0, de0 = dh['RA(catalog)'].mean(), dh['DEC(catalog)'].mean()
Xa = (dh['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Ya = dh['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax, *_ = np.linalg.lstsq(Aa, dh['px'].values, rcond=None)
ay, *_ = np.linalg.lstsq(Aa, dh['py'].values, rcond=None)

prov = providers.GaiaOfflineProvider.from_installed()
cat = prov.lookup((j['RA']-2.6, j['RA']+2.6), (j['DEC']-2.2, j['DEC']+2.2), 11.0,
                  epoch=date_string_to_float('2026-08-12'))
not_dbl = ~np.asarray(cat.is_double(10.0))
cc, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(cat, OPTS)
cra, cdec = np.degrees(cc.get_ra()), np.degrees(cc.get_dec())
cmag = np.asarray(cat.get_mags())
det = pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv'))
sky = chain(det[['py', 'px']].values.astype(float))
rows = {}
for gate, collect in ((8.0, False), (4.5, True)):
    prov_d = []; rows = {}
    for i in np.where(not_dbl)[0]:
        dd = np.hypot((sky[:, 1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:, 0]-cdec[i])*3600
        k = int(np.argmin(dd))
        if dd[k] < gate:
            prov_d.append((i, k, (sky[k, 1]-cra[i])*np.cos(np.radians(cdec[i]))*3600,
                           (sky[k, 0]-cdec[i])*3600))
    if not collect:
        off = (np.median([p[2] for p in prov_d]), np.median([p[3] for p in prov_d]))
        sky[:, 1] -= off[0]/3600/np.cos(np.radians(de0)); sky[:, 0] -= off[1]/3600
    else:
        seen = {}
        for i, k, dxi, deta in prov_d:
            seen.setdefault(k, []).append((i, dxi, deta))
        for k, cl in seen.items():
            if len(cl) > 1:
                continue
            i, dxi, deta = cl[0]
            rows[i] = (det['px'][k], det['py'][k], (ax[0]*dxi/3600 + ax[1]*deta/3600)*PS,
                       (ay[0]*dxi/3600 + ay[1]*deta/3600)*PS)
tab = pd.DataFrame([dict(cat_i=i, px=v[0], py=v[1], dx=v[2], dy=v[3], mag=cmag[i])
                    for i, v in rows.items()])
tab['dx'] -= tab['dx'].median(); tab['dy'] -= tab['dy'].median()
rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
keep = (R > 2.0*R_SUN_AS) & (tab.mag.values <= 11.0)
tab, rx, ry, R = tab[keep].reset_index(drop=True), rx[keep], ry[keep], R[keep]
tab['R_rsun'] = R/R_SUN_AS
tab['is_anchor'] = np.hypot(tab.px.values-ANCHOR[0], tab.py.values-ANCHOR[1]) < 6
tab.to_csv(os.path.join(OUT, 'master0612_star_table.csv'), index=False)
print('master0612 table: %d stars G<=11 outside 2 R_sun, anchor %s'
      % (len(tab), 'in' if tab.is_anchor.any() else 'OUT'), flush=True)

# membership against the union of record
U = pd.read_csv(r"D:/MEE2024 output/MEE_output/step3_record/leon_union_star_table.csv")
def key(t):
    return set(zip(np.round(t.px.values/8).astype(int), np.round(t.py.values/8).astype(int)))
only_master = key(tab) - key(U); only_union = key(U) - key(tab)
print('membership: %d in both, %d master-only, %d union-only'
      % (len(key(tab) & key(U)), len(only_master), len(only_union)), flush=True)


def design(px, py, rx_, ry_, R_, nuis_deg=None):
    xs, ys = (px-NX/2)/W_NORM, (py-NY/2)/W_NORM
    ux, uy = rx_/R_, ry_/R_
    n = len(px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(py-NY/2)*PS, ux*R_SUN_AS/R_]
    cols_y = [Z, np.ones(n), (px-NX/2)*PS, uy*R_SUN_AS/R_]
    labels = ['N1', 'N2', 'Th', 'L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for jj in range(nuis_deg+1-i):
                if i == 0 and jj == 0:
                    continue
                cols_x.append(Z); cols_y.append(xs**i*ys**jj); labels.append(f'v{i}{jj}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit(t, rx_, ry_, R_, nd=None):
    A, labels = design(t.px.values, t.py.values, rx_, ry_, R_, nd)
    b = np.concatenate([t.dx.values, t.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A@c
    return c[labels.index('L')], float(np.sqrt(np.mean(r**2)))


rng = np.random.default_rng(3)
lines = []
for tag, sel in (('with anchor', np.ones(len(tab), bool)), ('sans anchor', ~tab.is_anchor.values)):
    t = tab[sel].reset_index(drop=True); rxs, rys, Rs = rx[sel], ry[sel], R[sel]
    Lb, _ = fit(t, rxs, rys, Rs)
    Lv, rms = fit(t, rxs, rys, Rs, 2)
    boots = []
    for _ in range(200):
        k = rng.integers(0, len(t), len(t))
        try:
            boots.append(fit(t.iloc[k], rxs[k], rys[k], Rs[k], 2)[0])
        except Exception:
            pass
    se = float(np.std(boots, ddof=1))
    h = 1/np.mean((R_SUN_AS/Rs)**2)
    line = ('master0612 %-12s N=%3d h=%5.1f  L base %+.3f  L v-deg2 %+.3f +- %.3f (stat)  '
            'residual rms %.3f as/axis' % (tag, len(t), h, Lb, Lv, se, rms))
    lines.append(line); print(line, flush=True)
print('union of record (0.6+1.2, with anchor): N=42 h=27.1  L base +2.88  L v-deg2 +1.976 +- 0.582; '
      'sans anchor N=41 +1.50 +- 0.54', flush=True)
open(os.path.join(OUT, 'master_vs_union.txt'), 'w').write('\n'.join(lines) + '\n')
print('done', flush=True)
