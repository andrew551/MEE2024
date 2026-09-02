"""Does the null-test L grow with the time between the paired fields? Every pair of Leon
zenith fields within a night, not just the consecutive ones.

Douglas, 2026-09-02: "In theory, given enough stacking over a long enough time, shouldn't
all atmospheric disturbance average away? So the floor is also a time dependent figure."
The consecutive-pair nulls (tools/step3_zenith_floor.py) measure the differential at one
gap, 2 min 34 s. The six zenith fields of a night are 2.57 min apart in sequence, so the
fifteen pairs of a night span gaps of 2.6, 5.1, 7.7, 10.3 and 12.8 minutes -- the same
construction at five gaps, on the same night, at the same altitude. If the null is a
static model mismatch it will not care about the gap; if it is drift (thermal plate
scale, the tube cooling) or evolving atmosphere it will grow with it.

Same construction as step3_zenith_floor.py: field j fitted constant-only against field i's
own free cubic (i earlier than j), the eclipse Sun imposed, G <= 11, R > 2 R_sun, the
science estimator with the vertical-deg-2 nuisance, corrections off as at zenith. Also
recorded per pair: the free-fit plate-scale step between the two fields, so the drift
hypothesis can be tested directly (leverage x step against the measured null).

Writes step3_record/zenith_nulls_allpairs.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd
from astropy.io import fits

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
Z12 = r"D:/MEE2024 output/MEE_output/refraction/zenith12"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
NULLS = os.path.join(OUT, 'zenith_nulls_allpairs')
G = r"G:/Leon Aug 2026"
PS, NX, NY, W_NORM = 2.2054043, 6248, 4176, 3124.0
SUNPX, SUNPY, R_SUN_AS = 3171.0, 3232.0, 947.1
RCUT, MAGCUT = 2.0, 11.0
FIELDS = ('Z1_base', 'Z2_mid_left', 'Z3_top_left', 'Z4_top_right', 'Z5_mid_right',
          'Z6_bottom_right')
NIGHTS = {'08-11': '2026-08-11', '08-12': '2026-08-12'}


def mid_minutes(date, field):
    fr = sorted(glob.glob(os.path.join(G, date, 'Zenith', field, '*', '*.fits')))
    t = fits.getheader(fr[len(fr)//2])['DATE-OBS'].split('T')[1].split(':')
    return int(t[0])*60 + int(t[1]) + float(t[2])/60


def design(x_px, y_px, rx, ry, R, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ux, uy = rx/R, ry/R
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ux*R_SUN_AS/R]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, uy*R_SUN_AS/R]
    labels = ['N1', 'N2', 'Th', 'L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=None):
    A, labels = design(px, py, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]


def leverage(px, py, rx, ry, R):
    A, labels = design(px, py, rx, ry, R, 2)
    c, *_ = np.linalg.lstsq(A, np.concatenate([1e-6*(px-NX/2)*PS, 1e-6*(py-NY/2)*PS]), rcond=None)
    return float(c[labels.index('L')])


def refit(field, ref_field):
    d = os.path.join(NULLS, f'{field}_vs_{ref_field}')
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    ref = glob.glob(os.path.join(Z12, ref_field, 'stage2', '**', 'distortion_results.txt'), recursive=True)
    cz = glob.glob(os.path.join(Z12, field, 'centroid_data*.zip'))
    if not (ref and cz):
        return None
    with open(os.path.join(d, 'stage2.log'), 'w') as fh:
        subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', cz[0], '--order', 'cubic',
                        '--date-from-header', '--fix-distortion', ref[0],
                        '--set', 'distortion_fixed_coefficients=constant',
                        '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
                        '--set', 'rough_match_threshhold=36',
                        '--set', 'enable_corrections=False',
                        '--set', 'enable_corrections_ref=False',
                        '--no-display', '--quiet', '-o', d],
                       cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def scale_of(field):
    p = glob.glob(os.path.join(Z12, field, 'stage2', '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(p[0], encoding='utf-8'))['platescale (arcseconds/pixel)']


rng = np.random.default_rng(11)
rows = []
for night, date in NIGHTS.items():
    tm = {f: mid_minutes(date, f) for f in FIELDS}
    for i in range(len(FIELDS)):
        for j in range(i+1, len(FIELDS)):
            ref_field, field = f'{night}_{FIELDS[i]}', f'{night}_{FIELDS[j]}'
            path = refit(field, ref_field)
            if path is None:
                print(f'{field} vs {ref_field}: no residuals', flush=True)
                continue
            d = pd.read_csv(path)
            d = d[d['magV'] <= MAGCUT]
            px, py = d['px'].values, d['py'].values
            dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
            dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
            err = d['error_arcsec'].values
            rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
            R = np.hypot(rx, ry)
            keep = R > RCUT*R_SUN_AS
            px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
            Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
            Lb = fit_L(dx, dy, px, py, rx, ry, R)
            floor = float(np.std([fit_L(dx + rng.normal(0, err/np.sqrt(2)),
                                        dy + rng.normal(0, err/np.sqrt(2)),
                                        px, py, rx, ry, R, nuis_deg=2) for _ in range(40)], ddof=1))
            step = 1e6*(scale_of(field) - scale_of(ref_field))/scale_of(ref_field)
            g = leverage(px, py, rx, ry, R)
            gap = tm[FIELDS[j]] - tm[FIELDS[i]]
            rows.append(dict(night=date, field=field, ref=ref_field, gap_min=gap, n=int(keep.sum()),
                             Lb=Lb, Lv=Lv, floor=floor, step_ppm=step, leverage=g,
                             rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2))))
            print('  %-22s vs %-16s gap %5.1f min  N=%4d  L v-deg2 %+.3f  (floor %.3f)  '
                  'scale step %+6.1f ppm -> -lev*step %+.3f'
                  % (field, FIELDS[i], gap, keep.sum(), Lv, floor, step, -g*step), flush=True)

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'zenith_nulls_allpairs.csv'), index=False)
print('\n%d zenith null pairs; the null rms by gap:' % len(T))
T['gap_bin'] = np.round(T.gap_min/2.57)*2.57
for gb, sub in T.groupby('gap_bin'):
    v = sub.Lv.values
    print('  gap %5.1f min: %2d pairs  L v-deg2 rms %.3f  max %.3f  |scale step| rms %5.1f ppm'
          % (gb, len(sub), np.sqrt((v**2).mean()), np.abs(v).max(),
             np.sqrt((sub.step_ppm.values**2).mean())))
x, y = (-T.leverage*T.step_ppm).values, T.Lv.values
print('null vs -leverage*scale step over all pairs: r = %+.2f, slope %.2f' % (np.corrcoef(x, y)[0, 1], x@y/(x@x)))
print('->', os.path.join(OUT, 'zenith_nulls_allpairs.csv'))
