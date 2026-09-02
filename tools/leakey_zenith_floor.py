"""A second instrument's zenith floor: the Leakey 2024 Askar 65PHQ + ASI294MM sets.

Douglas, 2026-09-02: add the Leakey zenith data to the atmospheric-floor table. It answers
the caveat that table carries -- that its zenith row is measured on **Leon's** instrument,
so using it as the floor under Bruns' number assumes the two instruments' floors are
comparable. Leakey is a genuinely independent optic (Askar 65PHQ, 1.1506 arcsec/px on an
8288 x 5644 ASI294MM, against Leon's FRA500 + 0.7x at 2.2054 on 6248 x 4176), at a
different site, in a different year, reduced by the same chain.

Inputs are the existing reductions in `leakey_zenith/{zenith1,zenith2}/<block>/` (22 blocks,
all solved, free cubic at tol 0.2, corrections OFF as at zenith). This tool adds the two
things the floor table needs and those runs do not have:

  * **the quasi-static residual**, by the table's construction rather than a free cubic --
    each field re-fitted with the cubic-and-above FROZEN from the average of its own focus
    group and the quadratic free, so what remains is the structure a calibration fit cannot
    absorb. A free cubic absorbs exactly that and reports 0.052 arcsec, which is why it
    cannot be compared with Bruns' 0.100 or Leon's 0.067;
  * **the null-test L**, constant-only pairs within a focus group, with a Sun imposed at the
    same angular offset from frame centre as Leon's real one (2523 arcsec below, 104 to the
    side), the science cuts and the science estimator. Leakey never observed an eclipse
    field, so this is explicitly a forecast on that geometry, not a measurement of one.

**The focus groups are the whole reason this needs care.** The plate scale of zenith2 steps
+610 ppm between 03:01 and 03:12 and the fit rms improves from 0.09 to 0.04 arcsec -- a
refocus, not a drift. Pairing across it would measure the refocus. The groups, from the
fitted scales:

    zenith1  04:04-04:15  3 fields  ps 1.15057 +- 0.8 ppm      (08:52 field stands alone)
    zenith2  02:02-03:01 10 fields  ps 1.15026 +- 1.5 ppm
    zenith2  03:12-03:32  5 fields  ps 1.15095 +- 1.2 ppm      after the refocus
    zenith2  04:20-04:28  3 fields  ps 1.15101 +- 0.5 ppm

Within a group the scale is stable to ~1 ppm, against Leon's zenith 7.6 ppm between
consecutive fields -- so Leakey separates the drift and shape parts of the null more
cleanly than Leon can.

Site: Leakey, Texas, taken as 29.726 N, 99.764 W, 490 m (the town; the headers carry
`ORIGIN = Leakey TX` and no coordinates). Only the alt/az basis uses it, and a site error
of a few km moves a parallactic angle by far less than the measurement needs.

Writes step3_record/leakey_floor.csv and leakey_nulls.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.io import fits
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
LK = r"D:/MEE2024 output/MEE_output/leakey_zenith"
RAW = r"I:/Leakey 2024 data"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
QF = os.path.join(OUT, 'leakey_quadfree')
NULLS = os.path.join(OUT, 'leakey_nulls')
NX, NY = 8288, 5644
R_SUN_AS, MAGCUT, RCUT = 947.1, 11.0, 2.0
# the Sun imposed at Leon's own angular offset from frame centre, so the two rows describe
# the same geometry rather than the same pixel fractions
SUN_DX_AS, SUN_DY_AS = 104.0, 2523.0
SITE = EarthLocation(lat=29.726*u.deg, lon=-99.764*u.deg, height=490*u.m)
os.makedirs(QF, exist_ok=True)
os.makedirs(NULLS, exist_ok=True)


def blocks():
    out = []
    for zs, raw in (('zenith1', 'zenith 1'), ('zenith2', 'zenith 2')):
        for b in sorted(os.listdir(os.path.join(LK, zs))):
            r = glob.glob(os.path.join(LK, zs, b, 'stage2', '**', 'distortion_results.txt'),
                          recursive=True)
            cz = glob.glob(os.path.join(LK, zs, b, 'centroid_data*.zip'))
            if not (r and cz):
                continue
            j = json.load(open(r[0], encoding='utf-8'))
            fr = sorted(glob.glob(os.path.join(RAW, raw, b, '*.fit*')))
            t = fits.getheader(fr[len(fr)//2])['DATE-OBS'] if fr else None
            out.append(dict(set=zs, block=b, result=r[0], zip=cz[0], time=t,
                            ps=j['platescale (arcseconds/pixel)'], ra=j['RA'], dec=j['DEC'],
                            n=j['#stars used'], rms=j['final rms error (arcseconds)']))
    return out


B = blocks()
# focus groups: a new group whenever the scale steps by more than 50 ppm from the previous
groups, gid = {}, 0
prev = None
for b in B:
    if prev is None or b['set'] != prev['set'] or abs(1e6*(b['ps']-prev['ps'])/prev['ps']) > 50:
        gid += 1
    groups.setdefault(gid, []).append(b)
    prev = b
groups = {g: v for g, v in groups.items() if len(v) >= 3}
print('focus groups (>= 3 fields):', flush=True)
for g, v in groups.items():
    ps = np.array([x['ps'] for x in v])
    print('  group %d: %s  %s-%s  %2d fields  ps %.7f +- %.1f ppm'
          % (g, v[0]['set'], v[0]['time'][11:16], v[-1]['time'][11:16], len(v), ps.mean(),
             1e6*ps.std(ddof=1)/ps.mean()), flush=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def refit(outroot, tag, zip_path, refs, fixed):
    d = os.path.join(outroot, tag)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    run([PY, '-m', 'mee2024.cli', 'distortion', zip_path, '--order', 'cubic',
         '--date-from-header', '--fix-distortion', *refs,
         '--set', 'distortion_fixed_coefficients=' + fixed,
         '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
         '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=False',
         '--set', 'enable_corrections_ref=False', '--no-display', '--quiet', '-o', d],
        os.path.join(d, 'stage2.log'))
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def altaz_basis(b, resfile):
    """Unit vectors of increasing altitude and azimuth in sensor pixel axes."""
    j = json.load(open(resfile, encoding='utf-8'))
    mat = glob.glob(os.path.join(os.path.dirname(resfile), 'CATALOGUE_MATCHED_ERRORS.csv'))
    if not mat:
        return None, None, np.nan, np.nan
    d = pd.read_csv(mat[0])
    d.columns = [c.strip() for c in d.columns]
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0))
    Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d['px'].values, rcond=None)
    ay, *_ = np.linalg.lstsq(A, d['py'].values, rcond=None)
    t = Time(b['time'], scale='utc')
    fc = SkyCoord(j['RA']*u.deg, j['DEC']*u.deg)
    aa = fc.transform_to(AltAz(obstime=t, location=SITE))
    out = {}
    for key, off in (('alt', dict(alt=aa.alt+0.05*u.deg, az=aa.az)),
                     ('az', dict(alt=aa.alt, az=aa.az+0.05*u.deg/np.cos(aa.alt)))):
        p = SkyCoord(AltAz(obstime=t, location=SITE, **off)).icrs
        dv = np.array([(p.ra.deg-fc.ra.deg)*np.cos(np.radians(de0)), p.dec.deg-fc.dec.deg])
        v = np.array([dv @ ax[:2], dv @ ay[:2]])
        out[key] = v/np.linalg.norm(v)
    return out['alt'], out['az'], float(aa.alt.deg), float(aa.az.deg)


def design(px, py, rx, ry, R, PS, nuis_deg=None, with_scale=False):
    W = NX/2.0
    xs, ys = (px-NX/2)/W, (py-NY/2)/W
    ux, uy = rx/R, ry/R
    n = len(px); Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(py-NY/2)*PS]
    cols_y = [Z, np.ones(n), (px-NX/2)*PS]
    labels = ['N1', 'N2', 'Th']
    if with_scale:
        cols_x.append((px-NX/2)*PS); cols_y.append((py-NY/2)*PS); labels.append('S')
    cols_x.append(ux*R_SUN_AS/R); cols_y.append(uy*R_SUN_AS/R); labels.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(Z); cols_y.append(xs**i*ys**j); labels.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, px, py, rx, ry, R, PS, nuis_deg=None, with_scale=False):
    A, labels = design(px, py, rx, ry, R, PS, nuis_deg, with_scale)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')], (1e6*c[labels.index('S')] if with_scale else np.nan)


# ---------------------------------------------------------------- A. the quasi-static floor
rows = []
for g, v in groups.items():
    refs = [x['result'] for x in v]
    for b in v:
        tag = f'{b["set"]}_{b["block"]}'
        path = refit(QF, tag, b['zip'], refs, 'quadratic')
        if path is None:
            print(f'{tag}: no quadratic-free residuals', flush=True)
            continue
        res = glob.glob(os.path.join(QF, tag, '**', 'distortion_results.txt'), recursive=True)
        e_alt, e_az, alt, az = altaz_basis(b, res[0]) if res else (None, None, np.nan, np.nan)
        d = pd.read_csv(path)
        d = d[d['magV'] <= MAGCUT]
        dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
        dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
        m = np.hypot(dx, dy)
        lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
        k = m < lim
        dx, dy = dx[k], dy[k]
        va = dx*e_alt[0] + dy*e_alt[1] if e_alt is not None else np.full(len(dx), np.nan)
        vz = dx*e_az[0] + dy*e_az[1] if e_az is not None else np.full(len(dx), np.nan)
        rows.append(dict(group=g, set=b['set'], block=b['block'], time=b['time'], alt=alt, az=az,
                         n=int(k.sum()), qs_rms=float(np.sqrt(np.mean(dx**2 + dy**2))),
                         qs_alt=float(np.sqrt(np.mean(va**2))), qs_az=float(np.sqrt(np.mean(vz**2))),
                         qs_x=float(np.sqrt(np.mean(dx**2))), qs_y=float(np.sqrt(np.mean(dy**2)))))
        print('  %-24s alt %5.1f  N=%4d  rms %.3f (alt %.3f, az %.3f) V/H %.2f'
              % (tag, alt, k.sum(), rows[-1]['qs_rms'], rows[-1]['qs_alt'], rows[-1]['qs_az'],
                 rows[-1]['qs_alt']/max(rows[-1]['qs_az'], 1e-9)), flush=True)

F = pd.DataFrame(rows)
F.to_csv(os.path.join(OUT, 'leakey_floor.csv'), index=False)
print('\nLeakey zenith quadratic-free floor: rms %.3f, vertical %.3f, horizontal %.3f, V/H %.2f '
      '(alt %.1f-%.1f deg, %d fields)'
      % (F.qs_rms.mean(), F.qs_alt.mean(), F.qs_az.mean(), F.qs_alt.mean()/F.qs_az.mean(),
         F.alt.min(), F.alt.max(), len(F)), flush=True)
print('  sensor-axis anisotropy y/x = %.2f' % (F.qs_y.mean()/F.qs_x.mean()), flush=True)


# ---------------------------------------------------------------- B. the nulls
rng = np.random.default_rng(11)
nrows = []
for g, v in groups.items():
    for i in range(1, len(v)):
        b, ref = v[i], v[i-1]
        tag = f'{b["set"]}_{b["block"]}_vs_{ref["block"]}'
        path = refit(NULLS, tag, b['zip'], [ref['result']], 'constant')
        if path is None:
            print(f'{tag}: no residuals', flush=True)
            continue
        PS = ref['ps']
        sunpx, sunpy = NX/2 + SUN_DX_AS/PS, NY/2 + SUN_DY_AS/PS
        d = pd.read_csv(path)
        d = d[d['magV'] <= MAGCUT]
        px, py = d['px'].values, d['py'].values
        dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
        dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
        err = d['error_arcsec'].values
        rx, ry = (px-sunpx)*PS, (py-sunpy)*PS
        R = np.hypot(rx, ry)
        keep = R > RCUT*R_SUN_AS
        if keep.sum() < 25:
            print(f'  {tag}: only {keep.sum()} stars outside {RCUT} R_sun', flush=True)
            continue
        px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
        Lb, _ = fit_L(dx, dy, px, py, rx, ry, R, PS)
        Lv, _ = fit_L(dx, dy, px, py, rx, ry, R, PS, nuis_deg=2)
        Lf, S = fit_L(dx, dy, px, py, rx, ry, R, PS, nuis_deg=2, with_scale=True)
        floor = float(np.std([fit_L(dx + rng.normal(0, err/np.sqrt(2)),
                                    dy + rng.normal(0, err/np.sqrt(2)),
                                    px, py, rx, ry, R, PS, nuis_deg=2)[0]
                              for _ in range(40)], ddof=1))
        gap = (Time(b['time']) - Time(ref['time'])).to_value('min')
        step = 1e6*(b['ps']-ref['ps'])/ref['ps']
        h = 1/np.mean((R_SUN_AS/R)**2)
        nrows.append(dict(group=g, field=b['block'], ref=ref['block'], gap_min=gap,
                          step_ppm=step, n=int(keep.sum()), h=h, Lb=Lb, Lv=Lv,
                          Lv_freescale=Lf, S_ppm=S, floor=floor,
                          rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2))))
        print('  %-34s gap %5.1f min  N=%4d h=%4.1f  L v-deg2 %+.3f  (scale-free %+.3f)  '
              'step %+5.1f ppm  floor %.3f'
              % (f'{b["block"]} vs {ref["block"]}', gap, keep.sum(), h, Lv, Lf, step, floor),
              flush=True)

N = pd.DataFrame(nrows)
if len(N):
    N.to_csv(os.path.join(OUT, 'leakey_nulls.csv'), index=False)
    v, vf = N.Lv.values, N.Lv_freescale.values
    print('\n%d Leakey zenith nulls, within focus groups, %.1f-%.1f min apart, h = %.1f-%.1f R_sun^2'
          % (len(N), N.gap_min.min(), N.gap_min.max(), N.h.min(), N.h.max()))
    print('  residual rms %.3f-%.3f arcsec/axis' % (N.rms.min(), N.rms.max()))
    print('  L base   : rms %.3f  max %.3f' % (np.sqrt((N.Lb.values**2).mean()), np.abs(N.Lb.values).max()))
    print('  L v-deg2 : rms %.3f  max %.3f  (bootstrap floor %.3f)'
          % (np.sqrt((v**2).mean()), np.abs(v).max(), N.floor.median()))
    print('  scale free: rms %.3f  -> scale-like part %.3f' % (np.sqrt((vf**2).mean()),
          np.sqrt(max(np.mean(v**2) - np.mean(vf**2), 0))))
    print('  plate-scale steps within a group: rms %.1f ppm (Leon zenith: 7.6)'
          % np.sqrt((N.step_ppm.values**2).mean()))
    print('\nLEAKEY ZENITH NULL-TEST L = +-%.2f arcsec.  Leon zenith +-0.12, Bruns night +-0.15 '
          '(one-sided) / +-0.087 (bracketed).' % np.sqrt((v**2).mean()))
    print('->', os.path.join(OUT, 'leakey_nulls.csv'))
