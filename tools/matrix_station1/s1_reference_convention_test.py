"""Cell 2: does the eclipse field have to be centroided in the same convention as the
reference whose distortion it imports?

Douglas, 2026-09-03: the other fifteen raw zenith blocks are not available; the seventeen
2024 fits are all footprint moments. "What was the convention there? Do we need to use the
same convention for the eclipse field?"

The question is answerable without the missing frames, because the two sides of the transfer
can be varied independently:

  * the REFERENCE side -- vary it now, with the two raw zenith fields Douglas did find
    (`s1_zenith_raw_ab.py` fitted a free quintic per field in each convention);
  * the FIELD side -- the eclipse stage-1 archive of record
    (`eclipse fields/centroid_data20240416232626.zip`, 123 frames of the 0.4 s tier at
    18:12:30, 276 centroids) is fixed in the 2024 moment convention until it is re-stacked.

So this varies the reference against the one fixed field and reads L. Three references, all
imported the same way (constant-only quintic, the 2024 eclipse settings: tol 20", G <= 13,
rough 100", refraction and aberration on at 18:12 UTC, 15 C / 760 mb / RH 0.25 / 2400 m):

    A  the 2024 seventeen-field average          -- the reduction of record
    B  moments + annular, the two raw fields     -- same frames as A's first two, modern code
    C  windowed + annular, the two raw fields    -- B with only the estimator changed

B against C is the convention effect with everything else held: same frames, same fields,
same code, same day. A against B is field count and code era. Each is read out under
Method 1 (for the record) and Method 2 with the isotropic scale (how Station 1 is reduced),
on the stars common to all three so the comparison is not a selection difference.

If B and C agree to well inside the 0.10" Method-2 null floor, the conventions need not
match: the distortion model is a smooth function of position and the estimator's
magnitude-dependent bias does not survive into it. If they disagree, the eclipse must be
re-stacked in whatever convention the reference was fitted in.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
ECL = r"D:/MEE2024 output/Station 1/eclipse fields/centroid_data20240416232626.zip"
Z24 = r"D:/MEE2024 output/Station 1/zenith calibrations"
AB = r"D:/MEE2024 output/MEE_output/station1_record/zenith_raw_ab"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/reference_convention"
NX, NY, PS = 9576, 6388, 1.84847
T_MID = '2024-04-08T18:12:30'
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_time=18:12', '--set', 'observation_long=105 16 22.1 W',
       '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=15.0',
       '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
       '--set', 'observation_height=2400.0']


def free(v, tag):
    hit = glob.glob(os.path.join(AB, v, tag, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
    return hit[0] if hit else None


REFS = {
    'A 2024 17-field moments': sorted(glob.glob(os.path.join(Z24, '*', '**', 'distortion_results.txt'), recursive=True)),
    'B 2-field moments+annular': [p for p in (free('moments_annular', 'f1'), free('moments_annular', 'f2')) if p],
    'C 2-field windowed+annular': [p for p in (free('windowed_annular', 'f1'), free('windowed_annular', 'f2')) if p],
}
os.makedirs(OUT, exist_ok=True)


def fit_eclipse(name, refs):
    d = os.path.join(OUT, name.split()[0])
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    if not hit:
        with open(os.path.join(d, 'stage2.log'), 'w') as fh:
            subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', ECL, '--order', 'quintic',
                            '--fix-distortion', *refs, '--set', 'distortion_fixed_coefficients=constant',
                            '--set', 'distortion_fit_tol=20.0', '--set', 'max_star_mag_dist=13',
                            '--set', 'rough_match_threshhold=100', *MET,
                            '--no-display', '--quiet', '-o', d],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    return hit[0] if hit else None


def table(zp):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    j = json.load(zf.open([n for n in zf.namelist() if n.endswith('distortion_results.txt')][0]))
    return d, j


t = Time(T_MID, scale='utc'); sun = get_sun(t)
RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)

loaded = {}
for name, refs in REFS.items():
    if not refs:
        print('%s: references missing' % name); continue
    zp = fit_eclipse(name, refs)
    if not zp:
        print('%s: stage 2 FAILED' % name); continue
    d, j = table(zp)
    loaded[name] = (d, j)
    print('%-28s %2d reference file(s); %d stars matched, stage-2 rms %.3f", imported ps %.7f'
          % (name, len(refs), j['#stars used'], j['final rms error (arcseconds)'], j['platescale (arcseconds/pixel)']), flush=True)

common = None
for name, (d, j) in loaded.items():
    ids = set(d.loc[d['flag_is_outlier'] == False, 'ID'])
    common = ids if common is None else (common & ids)
print('\ncommon non-outlier stars across the %d references: %d' % (len(loaded), len(common or [])))


def method_fits(d):
    d = d[d.ID.isin(common)].copy()
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SUNPX, SUNPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cx = np.c_[X, Y, np.ones(len(d))]
    DX, DY = (ox@ax - cx@ax)*PS, (ox@ay - cx@ay)*PS
    px, py, mag = d.px.values, d.py.values, d.magV.values
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry)
    k = (R > RCUT*RS) & (R < RMAX*RS) & (mag <= MAGCUT)
    p, q, r = px[k], py[k], R[k]; ux, uy = rx[k]/r, ry[k]/r
    dx, dy = DX[k]-np.median(DX[k]), DY[k]-np.median(DY[k])
    n = len(p); Z = np.zeros(n)
    xs, ys = (p-NX/2)*PS, (q-NY/2)*PS; xn, yn = (p-NX/2)/(NX/2), (q-NY/2)/(NX/2)

    def solve(with_scale, nuis):
        cxl = [np.ones(n), Z, -ys]; cyl = [Z, np.ones(n), xs]; lab = ['N1', 'N2', 'Th']
        if with_scale:
            cxl.append(xs); cyl.append(ys); lab.append('S')
        cxl.append(ux*RS/r); cyl.append(uy*RS/r); lab.append('L')
        if nuis:
            for i in range(nuis+1):
                for jj in range(nuis+1-i):
                    if i == 0 and jj == 0:
                        continue
                    cxl.append(Z); cyl.append(xn**i*yn**jj); lab.append('v')
        M = np.vstack([np.column_stack(cxl), np.column_stack(cyl)])
        sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
        res = b - Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn))))
        c, e = c/sc, e/sc
        i = lab.index('L')
        return c[i], e[i], (1e6*c[lab.index('S')] if with_scale else np.nan), np.sqrt(s2)
    return n, solve(False, 2), solve(True, 0), solve(True, 2)


print('\n%-28s %5s | %-22s | %-30s | %-30s' % ('reference', 'stars', 'Method 1 (v-deg2)', 'Method 2 base', 'Method 2 v-deg2'))
res = {}
for name, (d, j) in loaded.items():
    n, m1, m2b, m2v = method_fits(d)
    res[name] = (m1, m2b, m2v)
    print('%-28s %5d | %+7.3f +- %.3f"      | %+7.3f +- %.3f"  S %+6.0f ppm | %+7.3f +- %.3f"  S %+6.0f ppm'
          % (name, n, m1[0], m1[1], m2b[0], m2b[1], m2b[2], m2v[0], m2v[1], m2v[2]))

if 'B 2-field moments+annular' in res and 'C 2-field windowed+annular' in res:
    b, c = res['B 2-field moments+annular'], res['C 2-field windowed+annular']
    print('\nCONVENTION EFFECT ON THE REFERENCE (B -> C, same frames, only the estimator changed):')
    print('   Method 1  dL = %+.3f"      Method 2 base  dL = %+.3f"      Method 2 v-deg2  dL = %+.3f"'
          % (c[0][0]-b[0][0], c[1][0]-b[1][0], c[2][0]-b[2][0]))
    print('   against the Method-2 null floor of 0.10" and the eclipse field\'s own sigma_L of ~0.28".')
if 'A 2024 17-field moments' in res and 'B 2-field moments+annular' in res:
    a, b = res['A 2024 17-field moments'], res['B 2-field moments+annular']
    print('FIELD COUNT AND CODE ERA (A -> B, same convention, 17 fields -> 2, v0.4.5 -> v1.4.0-dev):')
    print('   Method 1  dL = %+.3f"      Method 2 base  dL = %+.3f"      Method 2 v-deg2  dL = %+.3f"'
          % (b[0][0]-a[0][0], b[1][0]-a[1][0], b[2][0]-a[2][0]))
print('\n->', OUT)
