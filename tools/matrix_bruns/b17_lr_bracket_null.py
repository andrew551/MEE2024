"""The null under Bruns' ACTUAL bracket: the eclipse pointing fitted against the mean of the
RIGHT and LEFT calibration pointings taken either side of it, minutes apart.

Douglas, 2026-09-02: "Is the atmospheric term reduced now by the fact that there were two
daytime fields used (L/R) or is that irrelevant?"

It is not irrelevant, and the night rehearsal can answer it exactly, because Bruns rehearsed
the eclipse-day *sequence* and not merely the pointings. Measured from the fits:

    2017-08-19  03:57 RC01  03:59 EC01  04:02 LC01   04:04 RC02  04:06 EC02  04:08 LC02 ...
    2017-08-20  03:47 RC06  03:49 EC06  03:51 LC06   03:53 RC07  03:55 EC07  03:58 LC07 ...

R, E, L, R, E, L on a 2-minute cadence, with the eclipse pointing **midway between the two
calibration pointings in azimuth** (RC 156.1 deg, EC 143.4, LC 131.0; the mean of R and L is
143.55, i.e. 0.15 deg from EC) and between them in altitude (54.3 / 54.5 / 53.6). That is
the eclipse-day geometry and the eclipse-day order, at night, where the true deflection is
zero.

So this is the null that matches his design, and the earlier two do not:

  * `b17_atmosphere2.py` fits each field against the PREVIOUS field of the same pointing,
    6-7 minutes earlier -- one-sided in time, single pointing. +-0.150 arcsec. That is the
    right analogue of Leon's one-sided CAL_piLeo, not of Bruns' bracket;
  * `b17_bracket_null.py` fits each field against the mean of its own-pointing neighbours
    before and after -- bracketed in time, single pointing. +-0.087. It cancels linear
    temporal drift only;
  * this tool fits EC against the mean of RC and LC either side -- bracketed in time AND in
    sky position, which is what averaging two calibration fields on opposite sides actually
    buys. Both references go to `--fix-distortion` together, so `_open_distortion_files`
    averages every coefficient and the plate scale, exactly as the eclipse fit does.

Reported alongside, from the same three fields: the one-sided cross-pointing nulls (EC
against RC alone, and against LC alone). The difference between those and the bracketed
value is the cancellation, measured rather than assumed -- and their spread is the night's
own L-R differential, the analogue of the 45 ppm eclipse-day L-R split.

Everything else is held to the science construction: the field's own observation time (the
refraction correction is applied at the altitude it implies), the eclipse Sun's frame
position imposed on the residuals, G <= 11, R > 2 R_sun, and the estimator with the
vertical-deg-2 nuisance along the local vertical.

Writes matrix_bruns2017_atmosphere3/lr_bracket_nulls.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_atmosphere3"
LR = os.path.join(OUT, 'lr_bracket')
PS, NX, NY, W_NORM = 2.0868004, 3296, 2472, 1648.0
R_SUN_AS = 948.7
SUNPX, SUNPY = 1645.0, 1741.0
RCUT, MAGCUT = 2.0, 11.0
VX, VY = 0.447, -0.895
SITE = ['--set', 'observation_lat=42 44 11 N', '--set', 'observation_long=106 19 05 W',
        '--set', 'observation_height=2400', '--set', 'observation_temp=13.0',
        '--set', 'observation_pressure=770.0', '--set', 'observation_humidity=0.4',
        '--set', 'observation_wavelength=0.625']


def result_of(field):
    p = glob.glob(os.path.join(NIGHTS, field, 'stage2', '**', 'distortion_results.txt'),
                  recursive=True)
    return p[0] if p else None


def meta_of(field):
    p = result_of(field)
    if not p:
        return None
    j = json.load(open(p, encoding='utf-8'))
    t = (j.get('observation_time (UTC)') or '0:0:0').split(':')
    return dict(path=p, date=j.get('observation_date'), tm=j.get('observation_time (UTC)'),
                minute=int(t[0])*60 + int(t[1]) + float(t[2] if len(t) > 2 else 0)/60,
                ps=j['platescale (arcseconds/pixel)'], alt=j.get('observation alt (degrees)'),
                az=j.get('observation az (degrees)'))


def refit(tag, field, refs, obstime):
    d = os.path.join(LR, tag)
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


def design(x_px, y_px, rx, ry, R, nuis_deg=None, with_scale=False):
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ur, vr = rx/R, ry/R
    n = len(x_px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(y_px-NY/2)*PS]
    cols_y = [Z, np.ones(n), (x_px-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((x_px-NX/2)*PS); cols_y.append((y_px-NY/2)*PS); labels.append('S')
    cols_x.append(ur*R_SUN_AS/R); cols_y.append(vr*R_SUN_AS/R); labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(VX*xs**i*ys**j); cols_y.append(VY*xs**i*ys**j)
                labels.append('v%d%d' % (i, j))
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def null_from(path, rng):
    d = pd.read_csv(path)
    d = d[d['magV'] <= MAGCUT]
    px, py = d['px'].values, d['py'].values
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
    dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS
    R = np.hypot(rx, ry)
    keep = R > RCUT*R_SUN_AS
    if keep.sum() < 25:
        return None
    px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
    err = d['error_arcsec'].values[keep]
    out = {}
    for tag, nd, ws in (('Lb', None, False), ('Lv', 2, False), ('Lv_freescale', 2, True)):
        A, labels = design(px, py, rx, ry, R, nd, ws)
        c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
        out[tag] = c[labels.index('L')]
        if ws:
            out['S_ppm'] = 1e6*c[labels.index('S')]
    A, labels = design(px, py, rx, ry, R, 2)
    boots = []
    for _ in range(40):
        c, *_ = np.linalg.lstsq(A, np.concatenate([dx + rng.normal(0, err/np.sqrt(2)),
                                                   dy + rng.normal(0, err/np.sqrt(2))]), rcond=None)
        boots.append(c[labels.index('L')])
    out['floor'] = float(np.std(boots, ddof=1))
    out['n'] = int(keep.sum())
    out['rms'] = float(np.sqrt(np.mean(dx**2 + dy**2)/2))
    return out


rng = np.random.default_rng(11)
rows = []
for k in range(1, 11):
    ec, rc, lc = 'EC%02d' % k, 'RC%02d' % k, 'LC%02d' % k
    me, mr, ml = meta_of(ec), meta_of(rc), meta_of(lc)
    if not (me and mr and ml):
        print('%s: incomplete R/E/L triplet, skipped' % ec, flush=True)
        continue
    if not (me['date'] == mr['date'] == ml['date']):
        print('%s: triplet spans nights, skipped' % ec, flush=True)
        continue
    split = 1e6*(ml['ps'] - mr['ps'])/mr['ps']
    print('%s  %s: R %s (az %.1f)  E %s (az %.1f)  L %s (az %.1f)   L-R split %+.1f ppm'
          % (ec, me['date'], mr['tm'], mr['az'], me['tm'], me['az'], ml['tm'], ml['az'], split),
          flush=True)
    variants = (('bracket', [mr['path'], ml['path']]),
                ('vsR', [mr['path']]),
                ('vsL', [ml['path']]))
    got = {}
    for name, refs in variants:
        path = refit('%s_%s' % (ec, name), ec, refs, me['tm'])
        if path is None:
            print('    %s: no residuals' % name, flush=True)
            continue
        got[name] = null_from(path, rng)
        if got[name] is None:
            print('    %s: too few stars' % name, flush=True)
            continue
        print('    %-8s N=%4d rms %.3f as/ax  L base %+.3f  L v-deg2 %+.3f  (scale-free %+.3f, '
              'fitted S %+.1f ppm; floor %.3f)'
              % (name, got[name]['n'], got[name]['rms'], got[name]['Lb'], got[name]['Lv'],
                 got[name]['Lv_freescale'], got[name]['S_ppm'], got[name]['floor']), flush=True)
    if 'bracket' in got:
        rows.append(dict(field=ec, date=me['date'], gap_min=ml['minute']-mr['minute'],
                         lr_split_ppm=split,
                         **{f'{v}_{key}': got[v][key] for v in got for key in
                            ('Lb', 'Lv', 'Lv_freescale', 'S_ppm', 'floor', 'n', 'rms')}))

T = pd.DataFrame(rows)
if len(T):
    T.to_csv(os.path.join(OUT, 'lr_bracket_nulls.csv'), index=False)
    print('\n%d R-E-L triplets, the eclipse pointing bracketed by the two calibration '
          'pointings %.1f min apart' % (len(T), T.gap_min.mean()))
    for v, name in (('bracket', 'BRACKETED (mean of R and L)'), ('vsR', 'one-sided, vs RIGHT'),
                    ('vsL', 'one-sided, vs LEFT')):
        col = f'{v}_Lv'
        if col not in T:
            continue
        x = T[col].values
        print('  %-28s L v-deg2 rms %.3f  max %.3f  mean %+.3f   (scale-free rms %.3f)'
              % (name, np.sqrt((x**2).mean()), np.abs(x).max(), x.mean(),
                 np.sqrt((T[f'{v}_Lv_freescale'].values**2).mean())))
    print('  night L-R plate-scale split: mean %+.1f ppm, rms %.1f ppm (eclipse day: 45.0 ppm)'
          % (T.lr_split_ppm.mean(), np.sqrt((T.lr_split_ppm.values**2).mean())))
    print('  bootstrap floor (median): %.3f' % T.bracket_floor.median())
    print('\nfor comparison, the same nights by the other two constructions:')
    print('  one-sided same-pointing, 6-7 min (b17_atmosphere2.py) : +-0.150')
    print('  bracketed same-pointing, 12-13 min (b17_bracket_null.py): +-0.087')
    print('->', os.path.join(OUT, 'lr_bracket_nulls.csv'))
