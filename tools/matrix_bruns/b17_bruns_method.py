"""Cell 1 reduced BY BRUNS' OWN PROCEDURE -- the reduction the design criterion asks for.

Douglas, 2026-09-01, quoting Bruns 2018: the 0.62 s frames form ONE master image; the
0.09 s frames form a second; and the two close-in stars are carried from the short master
into the analysis by an offset computed from "the average centroid of the seven brightest
stars in the short-exposure master image ... compared to the same seven stars found in
the longer-exposure master image". Only those two centroids come from the short series;
everything else comes from the long master.

The previous record reduction departed from that in two ways, both inherited from Leon's
tier structure rather than decided for this dataset: EA and EB (the same 0.62 s exposure,
51 s apart) were stacked separately and combined per-star, and the two inner stars were
excluded because a single-tier detection cannot be cross-tier vetted. Douglas' ruling:
the design criterion is to repeat Bruns' calculation the way he did it, so the inner
stars MUST be in, linked his way.

So, exactly his procedure, in the cell's convention of record (Gaussian background,
footprint moments, the L+R8 bracket frozen):

  1. one 0.62 s master from all 34 EA+EB frames (their existing preprocessed frames --
     tier-mean coronal model, painted disk -- are reused; the EA and EB models are the
     same exposure 51 s apart and differ negligibly);
  2. constant-only stage 2 against the bracket -> the fitted model and the star table;
  3. the 0.09 s master (the existing E2 stack in the same convention);
  4. the offset between masters from the SEVEN BRIGHTEST stars of the short master that
     the long master also detects, as the mean difference of their centroids;
  5. the two close-in centroids shifted by that offset, pushed through the long master's
     fitted model, and added to the analysis;
  6. Method 1 (imported scale), with the vertical-deg-2 variant and Method 2 alongside.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
import sys
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

MAIN = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"          # preprocessed frames
CONV = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024"  # convention-of-record tree
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_brunsmethod"
REF_L = glob.glob(os.path.join(CONV, 'L', 'stage2', '**', 'distortion_results.txt'), recursive=True)[0]
REF_R = glob.glob(os.path.join(CONV, 'R8', 'stage2', '**', 'distortion_results.txt'), recursive=True)[0]
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
L_REF = 1.7512
MIDT = '17:43:47'      # photon-weighted mid of the 34 frames; equals the E2 mid
VX, VY = 0.447, -0.895
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']
S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
      '--set','remove_edgy_centroids=True','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian','--set','distortion_field_plot=True']


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


# ---- 1+2. the 0.62 s master and its constant-only fit
d = os.path.join(OUT, 'master062')
os.makedirs(d, exist_ok=True)
frames = (sorted(glob.glob(os.path.join(MAIN, 'EA', 'preprocessed', '*.fits')))
          + sorted(glob.glob(os.path.join(MAIN, 'EB', 'preprocessed', '*.fits'))))
assert len(frames) == 34, len(frames)
z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
if not z:
    run([PY,'-m','mee2024.cli','stack',*frames,*S1,'--no-scan','--quiet','-o',d],
        os.path.join(d, 'stage1.log'))
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
assert z, 'stage 1 failed'
d2 = os.path.join(d, 'stage2')
os.makedirs(d2, exist_ok=True)
if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
    run([PY,'-m','mee2024.cli','distortion',z[0],'--order','cubic','--date-from-header',
         '--fix-distortion',REF_L,REF_R,'--set','distortion_fixed_coefficients=constant',
         '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
         '--set','rough_match_threshhold=36','--set','enable_corrections=True',
         '--set','enable_corrections_ref=True','--set','distortion_field_plot=True',
         *SITE,'--set','observation_time=' + MIDT,'--quiet','-o',d2],
        os.path.join(d2, 'stage2.log'))
res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
assert res, 'stage 2 failed'
j = json.load(open(res[0], encoding='utf-8'))
print('master062: %d frames -> %d matched, rms %.4f, imported ps %.7f'
      % (len(frames), j['#stars used'], j['final rms error (arcseconds)'],
         j['platescale (arcseconds/pixel)']), flush=True)

zh = zipfile.ZipFile(glob.glob(os.path.join(d2, 'distortion_data*.zip'))[0])
q = (np.radians(j['platescale (arcseconds/pixel)']/3600.0),
     np.radians(j['RA']), np.radians(j['DEC']), np.radians(j['ROLL']))
cxd, cyd = j['distortion coeffs x'], j['distortion coeffs y']
names = list(cxd.keys())
CX = np.array([cxd[k] for k in names]); CY = np.array([cyd[k] for k in names])
OPTS = dict(observation_date='2017-08-21', observation_lat=42.7363889,
            observation_long=-106.3180556, observation_height=2400.0,
            observation_temp=13.0, observation_pressure=770.0,
            observation_humidity=0.4, observation_wavelength=0.625,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            guess_date=False, distortionOrder='cubic', observation_time=MIDT)


def chain(det_pypx):
    plate = det_pypx - np.array([NY/2, NX/2])
    plate_c = -apply_corrections(q, plate, CX, CY, (NY, NX), OPTS)
    return transforms.to_polar(transforms.linear_transform(q, plate_c, (NY, NX)))


dh = pd.read_csv(zh.open([n for n in zh.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
dh.columns = [c.strip() for c in dh.columns]
got = chain(dh[['py', 'px']].values.astype(float))
rt = float((np.hypot((got[:, 1]-dh['RA(obs)'])*np.cos(np.radians(dh['DEC(obs)'])),
                     got[:, 0]-dh['DEC(obs)'])*3600).max())
assert rt < 0.05, 'round-trip gate failed: %.4f' % rt

prov = providers.GaiaOfflineProvider.from_installed()
cat = prov.lookup((j['RA']-1.5, j['RA']+1.5), (j['DEC']-1.2, j['DEC']+1.2), 11.0,
                  epoch=date_string_to_float('2017-08-21'))
not_dbl = ~np.asarray(cat.is_double(10.0))
cc, _, _ = refraction_correction.AstroCorrect().correct_ra_dec(cat, OPTS)
cra, cdec = np.degrees(cc.get_ra()), np.degrees(cc.get_dec())
cmag = np.asarray(cat.get_mags())
ra0, de0 = dh['RA(catalog)'].mean(), dh['DEC(catalog)'].mean()
Xa = (dh['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Ya = dh['DEC(catalog)'].values-de0
Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
ax_, *_ = np.linalg.lstsq(Aa, dh['px'].values, rcond=None)
ay_, *_ = np.linalg.lstsq(Aa, dh['py'].values, rcond=None)
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u
sun = get_sun(Time('2017-08-21T' + MIDT, scale='utc'))
R_SUN_AS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
sx = (sun.ra.deg-ra0)*np.cos(np.radians(de0)); sy = sun.dec.deg-de0
SUNPX = float(np.array([sx, sy, 1.0])@ax_); SUNPY = float(np.array([sx, sy, 1.0])@ay_)
print('Sun pixel (%.0f, %.0f); R_sun %.1f arcsec' % (SUNPX, SUNPY, R_SUN_AS), flush=True)

det = pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv'))

# ---- 4. Bruns' offset: the seven brightest short-master stars found in both masters
e2z = glob.glob(os.path.join(CONV, 'E2', 'centroid_data*.zip'))[0]
detE2 = pd.read_csv(zipfile.ZipFile(e2z).open('STACKED_CENTROIDS_DATA.csv'))
detE2 = detE2.sort_values('flux (noise-normed)', ascending=False).reset_index(drop=True)
pairs = []
for _, r in detE2.iterrows():
    dd = np.hypot(det['px'].values - r.px, det['py'].values - r.py)
    k = int(np.argmin(dd))
    if dd[k] < 25.0:                        # inter-master pointing drift is ~7-14 px
        pairs.append((det['px'].values[k] - r.px, det['py'].values[k] - r.py))
    if len(pairs) == 7:
        break
assert len(pairs) == 7, 'only %d common stars found' % len(pairs)
offx = float(np.mean([p[0] for p in pairs])); offy = float(np.mean([p[1] for p in pairs]))
sx7 = float(np.std([p[0] for p in pairs], ddof=1)); sy7 = float(np.std([p[1] for p in pairs], ddof=1))
print('offset from the 7 brightest common stars: (%.2f, %.2f) px, scatter (%.2f, %.2f) px '
      '-> se of the link (%.2f, %.2f) arcsec'
      % (offx, offy, sx7, sy7, sx7*PS/np.sqrt(7), sy7*PS/np.sqrt(7)), flush=True)

# ---- 5. the two close-in stars, shifted into the long master's frame
INNER = ((1179.0, 2314.0), (2102.0, 1241.0))     # G 7.09 and G 7.52, E2 coordinates
inner_px = []
for x0, y0 in INNER:
    dd = np.hypot(detE2['px'].values - x0, detE2['py'].values - y0)
    k = int(np.argmin(dd))
    assert dd[k] < 6, 'inner star not found in the short master'
    inner_px.append((detE2['px'].values[k] + offx, detE2['py'].values[k] + offy))

# ---- 6. the star table: every long-master detection plus the two shifted centroids
rows = {}
sky = chain(det[['py', 'px']].values.astype(float))
for gate, collect in ((8.0, False), (4.5, True)):
    prov_d = []
    rows = {}
    for i in np.where(not_dbl)[0]:
        dd = np.hypot((sky[:, 1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:, 0]-cdec[i])*3600
        k = int(np.argmin(dd))
        if dd[k] < gate:
            dxi = (sky[k, 1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
            deta = (sky[k, 0]-cdec[i])*3600
            prov_d.append((i, k, dxi, deta))
    if not collect:
        off = (np.median([p[2] for p in prov_d]), np.median([p[3] for p in prov_d]))
        sky[:, 1] -= off[0]/3600/np.cos(np.radians(de0))
        sky[:, 0] -= off[1]/3600
    else:
        seen = {}
        for i, k, dxi, deta in prov_d:
            seen.setdefault(k, []).append((i, dxi, deta))
        for k, cl in seen.items():
            if len(cl) > 1:
                continue
            i, dxi, deta = cl[0]
            rows[i] = (det['px'][k], det['py'][k],
                       (ax_[0]*dxi/3600 + ax_[1]*deta/3600)*PS,
                       (ay_[0]*dxi/3600 + ay_[1]*deta/3600)*PS, 'master062')
# the shifted inner centroids go through the same (offset-corrected) chain
sky_in = chain(np.array([[p[1], p[0]] for p in inner_px], dtype=float))
sky_in[:, 1] -= off[0]/3600/np.cos(np.radians(de0))
sky_in[:, 0] -= off[1]/3600
for m, (x0, y0) in enumerate(inner_px):
    dd = np.hypot((cra-sky_in[m, 1])*np.cos(np.radians(cdec)), cdec-sky_in[m, 0])*3600
    i = int(np.argmin(dd))
    dxi = (sky_in[m, 1]-cra[i])*np.cos(np.radians(cdec[i]))*3600
    deta = (sky_in[m, 0]-cdec[i])*3600
    rows[i] = (x0, y0,
               (ax_[0]*dxi/3600 + ax_[1]*deta/3600)*PS,
               (ay_[0]*dxi/3600 + ay_[1]*deta/3600)*PS, 'E2-linked')
    print('inner star G %.2f linked in at sep %.2f arcsec' % (cmag[i], dd[i]), flush=True)

tab = pd.DataFrame([dict(cat_i=i, px=v[0], py=v[1], dx=v[2], dy=v[3], src=v[4],
                         mag=cmag[i]) for i, v in rows.items()])
tab['dx'] -= tab['dx'].median(); tab['dy'] -= tab['dy'].median()
rx, ry = (tab.px.values-SUNPX)*PS, (tab.py.values-SUNPY)*PS
R = np.hypot(rx, ry)
keep = (R > 1.45*R_SUN_AS) & (tab.mag.values <= 11.0)
tab, rx, ry, R = tab[keep].reset_index(drop=True), rx[keep], ry[keep], R[keep]
tab.to_csv(os.path.join(OUT, 'bruns_method_star_table.csv'), index=False)
print('final table: %d stars (%d from the long master, %d E2-linked)'
      % (len(tab), (tab.src == 'master062').sum(), (tab.src == 'E2-linked').sum()), flush=True)


def solve(with_scale=False, nuis_deg=None):
    xs, ys = (tab.px.values-NX/2)/W_NORM, (tab.py.values-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    n = len(tab)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(tab.py.values-NY/2)*PS]
    cols_y = [Z, np.ones(n), (tab.px.values-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((tab.px.values-NX/2)*PS)
        cols_y.append((tab.py.values-NY/2)*PS)
        labels.append('S')
    cols_x.append(ur*R_SUN_AS/R)
    cols_y.append(vr*R_SUN_AS/R)
    labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for jj in range(nuis_deg+1-i):
                if i == 0 and jj == 0:
                    continue
                cols_x.append(VX*xs**i*ys**jj)
                cols_y.append(VY*xs**i*ys**jj)
                labels.append('v%d%d' % (i, jj))
    A = np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)])
    b = np.concatenate([tab.dx.values, tab.dy.values])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A@c
    cov = (float(resid@resid)/(len(b)-len(c))) * np.linalg.inv(A.T@A)
    return c, labels, cov


rng = np.random.default_rng(3)
for tag, ws, nd in (('Method 1, base', False, None), ('Method 1, v-deg2', False, 2),
                    ('Method 2 (scale free)', True, None)):
    c, labels, cov = solve(ws, nd)
    iL = labels.index('L')
    boots = []
    for _ in range(300):
        k = rng.integers(0, len(tab), len(tab))
        tb, rb, yb, Rb = tab.iloc[k], rx[k], ry[k], R[k]
        try:
            xs, ys = (tb.px.values-NX/2)/W_NORM, (tb.py.values-NY/2)/W_NORM
            ur, vr = rb/Rb, yb/Rb
            n = len(tb)
            Z = np.zeros(n)
            colx = [np.ones(n), Z, -(tb.py.values-NY/2)*PS]
            coly = [Z, np.ones(n), (tb.px.values-NX/2)*PS]
            if ws:
                colx.append((tb.px.values-NX/2)*PS)
                coly.append((tb.py.values-NY/2)*PS)
            colx.append(ur*R_SUN_AS/Rb)
            coly.append(vr*R_SUN_AS/Rb)
            if nd:
                for i in range(nd+1):
                    for jj in range(nd+1-i):
                        if i == 0 and jj == 0:
                            continue
                        colx.append(VX*xs**i*ys**jj)
                        coly.append(VY*xs**i*ys**jj)
            A = np.vstack([np.column_stack(colx), np.column_stack(coly)])
            cb, *_ = np.linalg.lstsq(A, np.concatenate([tb.dx.values, tb.dy.values]),
                                     rcond=None)
            boots.append(cb[iL if not ws else labels.index('L')])
        except Exception:
            pass
    extra = ''
    if ws:
        extra = ', scale %+.1f ppm' % (1e6*c[labels.index('S')])
    print('%-22s L = %+.3f +- %.3f (boot)%s' % (tag, c[iL], np.std(boots, ddof=1), extra),
          flush=True)
h = 1/np.mean((R_SUN_AS/R)**2)
print('h = %.1f Rsun^2  [GR %.4f]' % (h, L_REF), flush=True)
