"""What survives Bruns' BRACKET: the night nulls redone with each field fitted against the
mean of the field before and the field after, the way his eclipse field was fitted against
L and R.

Douglas, 2026-09-02, on the null test: "Bruns seems to have folded in the atmospheric
uncertainty into positional uncertainty. Or did he assume the atmospheric term already
averaged away at the time scales he was measuring? Have we proved this assumption to be
false?"

The 22 nulls of b17_atmosphere2.py are ONE-SIDED: field k against field k-1, 6-7 minutes
earlier. That is the right analogue of Leon's one-sided CAL_piLeo, but it is not Bruns'
design. His eclipse field was fitted constant-only against the AVERAGE of the LEFT field
(before) and the RIGHT field (after) -- `_open_distortion_files` averages every
coefficient and the plate scale -- which cancels anything that drifts linearly across the
gap: the thermal plate-scale trend, a linear atmospheric trend, both. A one-sided null
carries that drift in full, so it can only over-state what his bracket left.

This repeats the null with his construction: field k against the mean of k-1 and k+1 of
the same night (both passed to --fix-distortion, exactly as L and R8 are), at the field's
own observation time, the eclipse Sun imposed, the science cuts, the science estimator with
the vertical-deg-2 nuisance along the local vertical. Three bracketed nulls per group-night
where five fields exist, fewer where they do not. The difference between the one-sided rms
(0.150") and the bracketed rms is what Bruns' design bought.

Writes matrix_bruns2017_atmosphere3/bracket_nulls.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3"
BR = os.path.join(OUT, 'bracket')
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
RCUT, MAGCUT = 2.0, 11.0
VX, VY = 0.447, -0.895
SITE = ['--set', 'observation_lat=42 44 11 N', '--set', 'observation_long=106 19 05 W',
        '--set', 'observation_height=2400', '--set', 'observation_temp=13.0',
        '--set', 'observation_pressure=770.0', '--set', 'observation_humidity=0.4',
        '--set', 'observation_wavelength=0.625']


def refit(field, refs, obstime):
    d = os.path.join(BR, field)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    cz = glob.glob(os.path.join(NIGHTS, field, 'centroid_data*.zip'))
    if not cz:
        return None
    with open(os.path.join(d, 'stage2.log'), 'w') as fh:
        subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', cz[0], '--order', 'cubic',
                        '--date-from-header', '--fix-distortion', *refs,
                        '--set', 'distortion_fixed_coefficients=constant',
                        '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
                        '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=True',
                        '--set', 'enable_corrections_ref=True', *SITE,
                        '--set', 'observation_time=' + obstime, '--no-display', '--quiet',
                        '-o', d], cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def design(x_px, y_px, rx, ry, R, nuis_deg=None):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS, ur*R_SUN_AS/R]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS, vr*R_SUN_AS/R]
    labels = ['N1', 'N2', 'Th', 'L']
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(VX*xs**i*ys**j); cols_y.append(VY*xs**i*ys**j)
                labels.append('v%d%d' % (i, j))
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, x_px, y_px, rx, ry, R, nuis_deg=None):
    A, labels = design(x_px, y_px, rx, ry, R, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]


def field_epoch(field):
    p = glob.glob(os.path.join(NIGHTS, field, 'stage2', '**', 'distortion_results.txt'), recursive=True)
    if not p:
        return None
    j = json.load(open(p[0], encoding='utf-8'))
    t = (j.get('observation_time (UTC)') or '0:0:0').split(':')
    return (j.get('observation_date'), int(t[0])*60 + int(t[1]) + float(t[2] if len(t) > 2 else 0)/60,
            p[0], j.get('observation_time (UTC)') or '08:00:00')


rows = []
rng = np.random.default_rng(11)
for group in ('EC', 'LC', 'RC'):
    epochs = []
    for i in range(1, 11):
        got = field_epoch('%s%02d' % (group, i))
        if got:
            epochs.append(('%s%02d' % (group, i), *got))
    by_night = {}
    for name, date, minute, res, tm in epochs:
        by_night.setdefault(date, []).append((minute, name, res, tm))
    for date in sorted(by_night):
        seq = sorted(by_night[date])
        print('%s %s: %s' % (group, date, ' '.join(n for _, n, _, _ in seq)), flush=True)
        for k in range(1, len(seq)-1):
            (m0, before, res0, _), (m1, name, _, tm), (m2, after, res2, _) = seq[k-1], seq[k], seq[k+1]
            path = refit(name, [res0, res2], tm)
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
                print('  %s: only %d stars' % (name, keep.sum()), flush=True)
                continue
            px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
            err = d['error_arcsec'].values[keep]
            Lb = fit_L(dx, dy, px, py, rx, ry, R)
            Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
            floor = float(np.std([fit_L(dx + rng.normal(0, err/np.sqrt(2)), dy + rng.normal(0, err/np.sqrt(2)),
                                        px, py, rx, ry, R, nuis_deg=2) for _ in range(40)], ddof=1))
            rows.append(dict(field=name, before=before, after=after, span_min=m2-m0, n=int(keep.sum()),
                             Lb=Lb, Lv=Lv, floor=floor, rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2))))
            print('  %s vs mean(%s, %s) spanning %.0f min: N=%4d rms %.3f  L base %+.3f  L v-deg2 %+.3f (floor %.3f)'
                  % (name, before, after, m2-m0, keep.sum(), rows[-1]['rms'], Lb, Lv, floor), flush=True)

T = pd.DataFrame(rows)
if len(T):
    T.to_csv(os.path.join(OUT, 'bracket_nulls.csv'), index=False)
    v = T.Lv.values
    print('\n%d BRACKETED night nulls (field against the mean of its neighbours before and after)' % len(T))
    print('  L base   : rms %.3f  max %.3f' % (np.sqrt((T.Lb.values**2).mean()), np.abs(T.Lb.values).max()))
    print('  L v-deg2 : rms %.3f  max %.3f  mean %+.3f  (bootstrap floor %.3f)'
          % (np.sqrt((v**2).mean()), np.abs(v).max(), v.mean(), T.floor.median()))
    print('  the one-sided construction on the same nights gave rms 0.150 (v-deg2), 22 pairs')
    print('->', os.path.join(OUT, 'bracket_nulls.csv'))
