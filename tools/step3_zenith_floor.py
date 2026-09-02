"""The zenith row of the atmospheric-floor table, filled in: the vertical/horizontal split
and the null-test L systematic at the instrument-and-model floor.

Douglas, 2026-09-02: "Why no vertical, horizontal, V/H and null-test for the Leon zenith
row? Couldn't at least some of them be filled in?" They can, and the row is the most
useful one in the table once they are, because it is the control:

  * the **alt/az split** was left out on the grounds that the vertical direction is
    degenerate overhead. That is wrong as stated. At alt 85-89 deg the field centre is
    still several degrees from the zenith, so the altitude direction is perfectly well
    defined -- what is weak is its *physical* significance, since refraction there is ~5"
    with almost no gradient. So the split should come out ISOTROPIC, and measuring it is
    the control that proves the V/H > 1 at low altitude is atmosphere and not an artefact
    of how the decomposition is built. A wrong assumed vertical would drive V/H toward 1,
    so this test can only fail in the safe direction;
  * the **null-test L** was left out because no one had run it. It is runnable, and the
    geometry is better than either existing null: the six zenith fields of a night are
    **2 min 34 s apart in sequence**, against Bruns' 6-7 minute pairs and the M5 windows'
    10-13 minutes, and the science chain's own CAL-to-eclipse gap is about two minutes.
    So this is the closest time match in the campaign to what the science field actually
    inherits -- measured where there is essentially no atmosphere to inherit.

Construction, identical to the other two campaigns (docs/STEP3_CHARTS_AND_SETTINGS.md
section 2): each field re-fitted CONSTANT-ONLY against the previous field of the same
night, the eclipse Sun's frame position imposed on the residuals, the science cuts applied
(G <= 11, R > 2 R_sun), and the science estimator run -- true L is zero, so whatever comes
back is what this floor and this estimator manufacture between two epochs. Corrections are
OFF, as they were in the zenith reductions of record (docs/LEON_2026-08-11.md 18.1: they
are not needed at zenith).

One difference from the eclipse field is stated rather than hidden: a zenith field carries
several hundred stars at G <= 11 against the eclipse union's 42, so its null has a much
smaller sampling error. The 42-star subsample column answers the like-for-like question --
what this residual field would manufacture at Leon's own star count.

Writes step3_record/zenith_floor.csv and zenith_nulls.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.io import fits
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
Z12 = r"D:/MEE2024 output/MEE_output/refraction/zenith12"
OUT = r"D:/MEE2024 output/MEE_output/step3_record"
QF = os.path.join(OUT, 'zenith_quadfree')
NULLS = os.path.join(OUT, 'zenith_nulls')
G = r"G:/Leon Aug 2026"
PS, NX, NY, W_NORM = 2.2054043, 6248, 4176, 3124.0
SUNPX, SUNPY, R_SUN_AS = 3171.0, 3232.0, 947.1
RCUT, MAGCUT = 2.0, 11.0
SITE = EarthLocation(lat=42.740470*u.deg, lon=-5.613780*u.deg, height=1101*u.m)
FIELDS = ('Z1_base', 'Z2_mid_left', 'Z3_top_left', 'Z4_top_right', 'Z5_mid_right',
          'Z6_bottom_right')
NIGHTS = {'08-11': '2026-08-11', '08-12': '2026-08-12'}


def mid_time(date, field):
    """The middle frame's DATE-OBS -- the zenith stage-2 results carry no observation time
    (corrections were off), so it comes from the frames themselves."""
    fr = sorted(glob.glob(os.path.join(G, date, 'Zenith', field, '*', '*.fits')))
    assert fr, (date, field)
    return fits.getheader(fr[len(fr)//2])['DATE-OBS']


def design(x_px, y_px, rx, ry, R, nuis_deg=None):
    """The science estimator's design matrix (tools/step3_s2_union.py), verbatim."""
    xs, ys = (x_px-NX/2)/W_NORM, (y_px-NY/2)/W_NORM
    ux, uy = rx/R, ry/R
    n = len(x_px)
    Z = np.zeros(n)
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


def altaz_basis(resfile, csvfile, date, tm):
    """Unit vectors of increasing altitude and azimuth in sensor pixel axes, from the
    field's own matched table -- the b17_m3_maps construction."""
    j = json.load(open(resfile, encoding='utf-8'))
    d = pd.read_csv(csvfile)
    d.columns = [c.strip() for c in d.columns]
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0))
    Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d['px'].values, rcond=None)
    ay, *_ = np.linalg.lstsq(A, d['py'].values, rcond=None)
    t = Time(date + 'T' + tm.split('T')[-1][:8], scale='utc')
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


# ---------------------------------------------------------------- A. the alt/az split
rows = []
for night, date in NIGHTS.items():
    for f in FIELDS:
        field = f'{night}_{f}'
        hit = glob.glob(os.path.join(QF, field, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        res = glob.glob(os.path.join(QF, field, '**', 'distortion_results.txt'), recursive=True)
        mat = glob.glob(os.path.join(QF, field, '**', 'CATALOGUE_MATCHED_ERRORS.csv'),
                        recursive=True)
        if not (hit and res and mat):
            print(f'{field}: no quadratic-free residuals, skipped', flush=True)
            continue
        tm = mid_time(date, f)
        e_alt, e_az, alt, az = altaz_basis(res[0], mat[0], date, tm)
        d = pd.read_csv(hit[0])
        d = d[d['magV'] <= MAGCUT]
        dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
        dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
        m = np.hypot(dx, dy)
        lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
        g = m < lim
        dx, dy = dx[g], dy[g]
        v_alt = dx*e_alt[0] + dy*e_alt[1]
        v_az = dx*e_az[0] + dy*e_az[1]
        rows.append(dict(field=field, date=date, time=tm, n=int(g.sum()), alt=alt, az=az,
                         qs_rms=float(np.sqrt(np.mean(dx**2 + dy**2))),
                         qs_alt=float(np.sqrt(np.mean(v_alt**2))),
                         qs_az=float(np.sqrt(np.mean(v_az**2)))))
        print('%-22s alt %5.2f az %6.2f  N=%4d  rms %.3f (alt %.3f, az %.3f) VH %.2f'
              % (field, alt, az, g.sum(), rows[-1]['qs_rms'], rows[-1]['qs_alt'],
                 rows[-1]['qs_az'], rows[-1]['qs_alt']/rows[-1]['qs_az']), flush=True)

F = pd.DataFrame(rows)
F.to_csv(os.path.join(OUT, 'zenith_floor.csv'), index=False)
print('\nzenith quadratic-free floor: rms %.3f, vertical %.3f, horizontal %.3f, V/H %.2f'
      % (F.qs_rms.mean(), F.qs_alt.mean(), F.qs_az.mean(), F.qs_alt.mean()/F.qs_az.mean()),
      flush=True)


# ------------------------------------------------- B. the constant-only nulls, same night
def refit(field, reference, night):
    d = os.path.join(NULLS, field)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    cz = glob.glob(os.path.join(Z12, field, 'centroid_data*.zip'))
    if not cz:
        return None
    with open(os.path.join(d, 'stage2.log'), 'w') as fh:
        subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', cz[0], '--order', 'cubic',
                        '--date-from-header', '--fix-distortion', reference,
                        '--set', 'distortion_fixed_coefficients=constant',
                        '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
                        '--set', 'rough_match_threshhold=36',
                        '--set', 'enable_corrections=False',
                        '--set', 'enable_corrections_ref=False',
                        '--no-display', '--quiet', '-o', d],
                       cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


rng = np.random.default_rng(11)
nrows = []
for night, date in NIGHTS.items():
    for k in range(1, len(FIELDS)):
        field = f'{night}_{FIELDS[k]}'
        ref_field = f'{night}_{FIELDS[k-1]}'
        ref = glob.glob(os.path.join(Z12, ref_field, 'stage2', '**', 'distortion_results.txt'),
                        recursive=True)
        if not ref:
            print(f'{field}: no reference fit for {ref_field}', flush=True)
            continue
        path = refit(field, ref[0], night)
        if path is None:
            print(f'{field}: constant-only refit produced no residuals', flush=True)
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
        if keep.sum() < 25:
            print(f'{field}: only {keep.sum()} stars outside {RCUT} R_sun', flush=True)
            continue
        px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
        Lb = fit_L(dx, dy, px, py, rx, ry, R)
        Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
        floor = float(np.std([fit_L(dx + rng.normal(0, err/np.sqrt(2)),
                                    dy + rng.normal(0, err/np.sqrt(2)),
                                    px, py, rx, ry, R, nuis_deg=2) for _ in range(60)], ddof=1))
        # like-for-like with the eclipse union's 42 stars
        sub = []
        for _ in range(200):
            i = rng.choice(len(px), min(42, len(px)), replace=False)
            try:
                sub.append(fit_L(dx[i], dy[i], px[i], py[i], rx[i], ry[i], R[i], nuis_deg=2))
            except Exception:
                pass
        nrows.append(dict(field=field, ref=ref_field, night=date, n=int(keep.sum()),
                          Lb=Lb, Lv=Lv, floor=floor,
                          sub42_rms=float(np.sqrt(np.mean(np.array(sub)**2))),
                          rms=float(np.sqrt(np.mean(dx**2 + dy**2)/2))))
        print('  %-22s vs %-14s N=%4d rms %.3f as/ax  L base %+.3f  L v-deg2 %+.3f  '
              '(floor %.3f, 42-star rms %.3f)'
              % (field, FIELDS[k-1], keep.sum(), nrows[-1]['rms'], Lb, Lv, floor,
                 nrows[-1]['sub42_rms']), flush=True)

N = pd.DataFrame(nrows)
if len(N):
    N.to_csv(os.path.join(OUT, 'zenith_nulls.csv'), index=False)
    v = N.Lv.values
    print('\n%d constant-only zenith nulls (consecutive same-night fields, 2 min 34 s apart), '
          'true L = 0 in every one' % len(N))
    print('  residual rms %.3f - %.3f arcsec/axis (the eclipse field itself: 0.75)'
          % (N.rms.min(), N.rms.max()))
    print('  L base   : rms %.3f  max %.3f' % (np.sqrt((N.Lb.values**2).mean()),
                                               np.abs(N.Lb.values).max()))
    print('  L v-deg2 : rms %.3f  max %.3f  (bootstrap floor %.3f)'
          % (np.sqrt((v**2).mean()), np.abs(v).max(), N.floor.median()))
    print('  at the eclipse union\'s 42 stars: rms %.3f' % N.sub42_rms.mean())
    print('\nZENITH NULL-TEST L SYSTEMATIC = +-%.2f arcsec on the full star sample, '
          '+-%.2f at 42 stars.' % (np.sqrt((v**2).mean()), N.sub42_rms.mean()))
    print('Bruns 2017 night: +-0.15; Leon horizon (M5): +-0.33 (max), +-0.22 (rms).')
    print('nulls ->', os.path.join(OUT, 'zenith_nulls.csv'))
