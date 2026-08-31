"""Cell 1's atmospheric systematic, done the way Leon's was (v2 -- v1 was invalid).

v1 (b17_atmosphere.py) read the night fields' existing TWOD_RESIDUALS and got +-0.018
arcsec. That number is meaningless and is withdrawn: those residuals come from a FREE
cubic fit at tolerance 0.2 (rms 0.059 arcsec), and a free cubic absorbs the smooth
atmospheric structure this test exists to measure. Leon's +-0.33 came from the opposite
construction -- M5's fields are fitted CONSTANT-ONLY against a frozen reference (rms 1.60
arcsec), which is what the eclipse field itself does, so those residuals still carry the
atmosphere the science fit inherits.

This rebuilds the Bruns nulls on that footing: each field is fitted CONSTANT-ONLY against
the previous field of the same group *on the same night*, exactly as the science tiers are
fitted constant-only against the L/R bracket. The residual field is handed to the
estimator with the eclipse Sun pixel imposed: true deflection is zero, so whatever L comes
back is what this atmosphere and this estimator manufacture between two epochs.

**Pairing consecutive same-night fields is the whole method, not a detail.** A first
attempt froze each group's field 01 as the reference for all nine others and reported
+-1.04 arcsec. That number is withdrawn: fields 06-10 of every group are from the FOLLOWING
night, so it was measuring the +85 ppm night-to-night plate-scale gap this project has
already documented (~0.85 arcsec of L at h = 10.6) rather than the atmosphere. Three
symptoms flagged it before the cause was found -- the null residuals were larger than the
eclipse field's own, they clustered by group instead of scattering, and the nuisance made
them worse rather than better. Consecutive same-night pairs are 6-7 minutes apart, which
is the configuration the science chain actually uses: CAL_piLeo and the eclipse field are
about two minutes apart.

What it measures, stated precisely: the field-to-field atmospheric differential over a few
minutes, inherited through a frozen distortion -- the same quantity Leon's gate measured,
and the one the eclipse field inherits from its calibration. It is not the eclipse-day
daytime atmosphere, which no night field can supply.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere2"
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
RCUT, MAGCUT = 2.0, 11.0
VX, VY = 0.447, -0.895                 # the eclipse field's local vertical, sensor axes
SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
        '--set','observation_height=2400','--set','observation_temp=13.0',
        '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
        '--set','observation_wavelength=0.625']


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def refit(field, reference):
    d = os.path.join(OUT, field)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    cz = glob.glob(os.path.join(NIGHTS, field, 'centroid_data*.zip'))
    if not cz:
        return None
    run([PY,'-m','mee2024.cli','distortion',cz[0],'--order','cubic','--date-from-header',
         '--fix-distortion',reference,'--set','distortion_fixed_coefficients=constant',
         '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
         '--set','rough_match_threshhold=36','--set','enable_corrections=True',
         '--set','enable_corrections_ref=True',*SITE,'--set','observation_time=08:00',
         '--no-display','--quiet','-o',d], os.path.join(d, 'stage2.log'))
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def design(x_px, y_px, rx, ry, R, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    n = len(x_px)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ur*R_SUN_AS/R]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, vr*R_SUN_AS/R]
    labels = ['N1','N2','Th','L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(VX*xs**i*ys**j)
                cols_y.append(VY*xs**i*ys**j)
                labels.append('v%d%d' % (i, j))
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, x_px, y_px, rx, ry, R, nuis_deg=None):
    A, labels = design(x_px, y_px, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]


def field_epoch(field):
    """(date, minutes) from the field's own fit, so pairs stay inside one night."""
    p = glob.glob(os.path.join(NIGHTS, field, 'stage2', '**', 'distortion_results.txt'),
                  recursive=True)
    if not p:
        return None
    j = json.load(open(p[0], encoding='utf-8'))
    t = (j.get('observation_time (UTC)') or '0:0:0').split(':')
    return j.get('observation_date'), int(t[0])*60 + int(t[1]) + float(t[2] if len(t) > 2 else 0)/60, p[0]


rows = []
rng = np.random.default_rng(11)
for group in ('EC', 'LC', 'RC'):
    epochs = []
    for i in range(1, 11):
        got = field_epoch('%s%02d' % (group, i))
        if got:
            epochs.append(('%s%02d' % (group, i), got[0], got[1], got[2]))
    by_night = {}
    for name, date, minute, res in epochs:
        by_night.setdefault(date, []).append((minute, name, res))
    for date in sorted(by_night):
        seq = sorted(by_night[date])
        print('%s %s: %d fields, %s' % (group, date, len(seq), ' '.join(n for _, n, _ in seq)),
              flush=True)
        for (m0, ref_name, ref_path), (m1, name, _) in zip(seq, seq[1:]):
            path = refit(name, ref_path)
            if path is None:
                print('  %s: no residuals' % name, flush=True)
                continue
            d = pd.read_csv(path)
            d = d[d['magV'] <= MAGCUT]
            px, py = d['px'].values, d['py'].values
            dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
            dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
            rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
            R = np.hypot(rx, ry)
            keep = R > RCUT*R_SUN_AS
            if keep.sum() < 25:
                print('  %s: only %d stars outside %.1f R_sun' % (name, keep.sum(), RCUT),
                      flush=True)
                continue
            px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
            err = d['error_arcsec'].values[keep]
            Lb = fit_L(dx, dy, px, py, rx, ry, R)
            Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
            boots = [fit_L(dx + rng.normal(0, err/np.sqrt(2)), dy + rng.normal(0, err/np.sqrt(2)),
                           px, py, rx, ry, R, nuis_deg=2) for _ in range(60)]
            floor = float(np.std(boots, ddof=1))
            rms = float(np.sqrt(np.mean(dx**2 + dy**2)/2))
            rows.append(dict(field=name, ref=ref_name, gap=m1-m0, n=int(keep.sum()),
                             Lb=Lb, Lv=Lv, floor=floor, rms=rms))
            print('  %s vs %s (%.0f min): N=%4d rms %.3f as/ax  L base %+.3f  '
                  'L v-deg2 %+.3f  (floor %.3f)'
                  % (name, ref_name, m1-m0, keep.sum(), rms, Lb, Lv, floor), flush=True)

R_ = pd.DataFrame(rows)
if len(R_):
    print('\n%d constant-only night nulls, true L = 0 in every one' % len(R_))
    print('  residual rms: %.3f - %.3f arcsec/axis (the eclipse field itself: 0.219)'
          % (R_.rms.min(), R_.rms.max()))
    for tag, col in (('L base', 'Lb'), ('L v-deg2', 'Lv')):
        v = R_[col].values
        print('  %-9s: mean %+.3f  rms %.3f  max |dL| %.3f'
              % (tag, v.mean(), np.sqrt((v**2).mean()), np.abs(v).max()))
    print('  bootstrap floor: %.3f' % R_.floor.median())
    v = R_['Lv'].values
    print('\nCELL-1 ATMOSPHERIC SYSTEMATIC = +-%.2f arcsec (rms of the v-deg2 nulls; '
          'Leon quotes +-0.33 by the same construction)' % np.sqrt((v**2).mean()))
