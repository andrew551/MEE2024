"""Cell 2: the other exposures -- a two-witness union, and how close in the short one reaches.

Douglas, 2026-09-03: "In the 2024 analysis we only used the 400 ms exposures. Would there be
an advantage to at least include the 300 ms data to do the two witness check as we did for
the Leon 2026 data. And then have a look at the 250 ms data to see if anything is closer in
to the corona, as we did for the Bruns 2017 data." And, correcting the count: "There are
really only three tiers as two are 300ms."

Three exposures in four blocks, which is a better hand than it first looks:

    18:10:26Z   0.25 s   124 frames   dark-250ms
    18:11:28Z   0.3  s   124 frames   dark-300ms
    18:12:30Z   0.4  s   123 frames   dark-400ms   <- the only one the 2024 analysis used
    18:13:31Z   0.3  s   124 frames   dark-300ms

The 0.3 s exposure was shot twice, before and after the 0.4 s, so those two blocks are
**independent captures at identical depth** -- the cleanest witness pair the campaign has,
because a star seen in both cannot owe its detection to the exposure. Leon's two-witness rule
had to compare 0.6 s against 1.2 s and live with the depth difference.

**All four blocks were already stacked in 2024, dark- and flat-calibrated, with identical
detection settings** (4.0 sigma, min_area 2, sigma_subtract 0, Sun masked), and every stacked
image survives. So this re-centroids those, exactly as `s1_zenith_recentroid.py` does for the
zenith fields -- which was validated there to reproduce a raw re-stack to 0.8 ppm of plate
scale. Re-stacking from raw would gain nothing and would lose the 2024 calibration, which is
worth +0.18 " in L on identical stars (`s1_eclipse_calibrated.py`).

    1810_4  0.25 s  20240417025605  109 frames (0015-0123; the operator dropped the first
                                    fifteen, which sit right at second contact)
    1811_4  0.3  s  20240417014008  124 frames
    1812_5  0.4  s  20240416232626  123 frames   <- the 2024 reduction of record
    1813_5  0.3  s  20240417035417  124 frames

Each is centroided BOTH ways, windowed and footprint moments, under one code version. The
stack, the calibration and the star field are then identical between the two, so the only
difference is the estimator -- the cleanest form of the comparison Bruns 2018 section 2.3
made between Astrometrica and MaxIm DL.

Each block gets its own mid-time, because the Sun moves about 2.4 " per minute against the
stars and the four blocks span four minutes: a single time would misplace the Sun by ~10 ",
which is 1 % of R_sun and matters most for the innermost stars where the deflection column is
steepest.

Reports:

  1. per block -- centroids, matched stars, fit residual, and **how close to the Sun the
     innermost matched star sits**, which is the Bruns question: a shorter exposure sees less
     coronal glare and can reach further in, and the innermost stars carry the most weight;
  2. the union -- every matched star with the number of blocks that saw it, and L fitted on
     the two-witness set (>= 2 blocks) against the 0.4 s block alone;
  3. what the two-witness rule costs and buys: stars, sigma_L, and the shift in L.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
Z24 = r"D:/MEE2024 output/Station 1/zenith calibrations"
RECEN = r"D:/MEE2024 output/MEE_output/station1_record/zenith_recentroid"
CAL04 = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_calibrated"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_tiers"
NX, NY, PS = 9576, 6388, 1.84847
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0

ECL24 = r"D:/MEE2024 output/Station 1/eclipse fields"
# tag, 2024 archive stamp, exposure, mid-time of the block
BLOCKS = [
    ('0p25s_1810', '20240417025605', '0.25 s', '18:11:12', '2024-04-08T18:11:12'),
    ('0p3s_1811',  '20240417014008', '0.3 s',  '18:11:58', '2024-04-08T18:11:58'),
    ('0p4s_1812',  '20240416232626', '0.4 s',  '18:13:00', '2024-04-08T18:13:00'),
    ('0p3s_1813',  '20240417035417', '0.3 s',  '18:14:02', '2024-04-08T18:14:02'),
]
ESTIM = {'windowed': ['--set', 'centroid_refine_window=True'],
         'moments':  ['--set', 'centroid_refine_window=False']}
# the 2024 settings, kept verbatim; the blob mask still applies because the Sun is present
# in the stacked image exactly as it was in the frames
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'centroid_window_sigma=2.0']


def met(tmid):
    return ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
            '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
            '--set', 'observation_time=' + tmid, '--set', 'observation_long=105 16 22.1 W',
            '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=15.0',
            '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
            '--set', 'observation_height=2400.0']


REF = sorted(glob.glob(os.path.join(RECEN, '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))
REFNAME = 'F_17field_windowed'
if not REF:
    REF = sorted(glob.glob(os.path.join(Z24, '*', '**', 'distortion_results.txt'), recursive=True))
    REFNAME = 'A_2024_17field_moments'
print('reference: %s, %d fields\n' % (REFNAME, len(REF)), flush=True)
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stack(tag, stamp, est):
    """Re-centroid the 2024 calibrated stack of this block, in one estimator."""
    img = os.path.join(ECL24, 'CENTROID_OUTPUT%s' % stamp, 'STACKED%s.fit' % stamp)
    if not os.path.exists(img):
        return None, None
    d = os.path.join(OUT, '%s_%s' % (tag, est))
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  re-centroiding %s (%s) as %s...' % (tag, stamp, est), flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', img, *S1, *ESTIM[est],
             '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return (z[0], d) if z else (None, d)


def stage2(root, cz, tmid):
    d = os.path.join(root, 'stage2_' + REFNAME)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    if not hit:
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic',
             '--fix-distortion', *REF, '--set', 'distortion_fixed_coefficients=constant',
             '--set', 'distortion_fit_tol=20.0', '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=100', *met(tmid),
             '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
        hit = glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True)
    return hit[0] if hit else None


def table(zp, tiso):
    """Matched stars with sensor-frame residuals and radius from the Sun at this block's time."""
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    j = json.load(zf.open([n for n in zf.namelist() if n.endswith('distortion_results.txt')][0]))
    d = d[d['flag_is_outlier'] == False].copy()
    sun = get_sun(Time(tiso, scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cxm = np.c_[X, Y, np.ones(len(d))]
    d['dx'] = (ox@ax - cxm@ax)*PS; d['dy'] = (ox@ay - cxm@ay)*PS
    d['rx'] = (d.px.values-SPX)*PS; d['ry'] = (d.py.values-SPY)*PS
    d['R'] = np.hypot(d.rx, d.ry); d['Rsun'] = d.R/RS
    d['RS'] = RS
    return d, j


def fit_L(d, nuis=0):
    k = (d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)
    t = d[k]
    if len(t) < 20:
        return None
    p, q, r = t.px.values, t.py.values, t.R.values
    ux, uy = t.rx.values/r, t.ry.values/r
    dx = t.dx.values - np.median(t.dx.values); dy = t.dy.values - np.median(t.dy.values)
    RS = t.RS.values
    n = len(t); Z = np.zeros(n)
    xs, ys = (p-NX/2)*PS, (q-NY/2)*PS; xn, yn = (p-NX/2)/(NX/2), (q-NY/2)/(NX/2)
    cxl = [np.ones(n), Z, -ys, xs, ux*RS/r]; cyl = [Z, np.ones(n), xs, ys, uy*RS/r]
    lab = ['N1', 'N2', 'Th', 'S', 'L']
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
    return dict(n=n, L=c[4], eL=e[4], S=1e6*c[3], rms=np.sqrt(s2),
                h=1/np.mean((RS/r)**2), rin=t.Rsun.min())


EST_LIST = ('windowed', 'moments')
tabs = {}
print('=== per block, each centroided both ways ===', flush=True)
EST_LIST = ('windowed', 'moments')
tabs = {}
print('%-12s %-7s %-9s %6s %8s %7s %9s %9s %9s' %
      ('block', 'expo', 'estimator', 'cen', 'matched', 'sci', 'innermost', 'rms', 'L'), flush=True)
for tag, stamp, expo, tmid, tiso in BLOCKS:
    for est in EST_LIST:
        cz, root = stack(tag, stamp, est)
        if not cz:
            print('  %s/%s: STAGE 1 FAILED' % (tag, est), flush=True)
            continue
        r = json.load(zipfile.ZipFile(cz).open('results.txt'))
        zp = stage2(root, cz, tmid)
        if not zp:
            print('  %s/%s: stage 2 FAILED' % (tag, est), flush=True)
            continue
        d, j = table(zp, tiso)
        tabs[(tag, est)] = d
        sci = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)]
        f = fit_L(d)
        print('%-12s %-7s %-9s %6d %8d %7d %9.2f %9.3f %9s'
              % (tag, expo, est, r['n_centroids'], len(d), len(sci),
                 sci.Rsun.min() if len(sci) else float('nan'),
                 j['final rms error (arcseconds)'],
                 ('%+.3f' % f['L']) if f else 'n/a'), flush=True)

if not tabs:
    raise SystemExit('nothing reduced')


def union_of(est):
    """One row per star: how many blocks saw it, residuals averaged over those blocks."""
    ids = {}
    for (tag, e), d in tabs.items():
        if e != est:
            continue
        for i_ in d.ID.values:
            ids.setdefault(i_, []).append(tag)
    rows = []
    for i_, tg in ids.items():
        sub = [tabs[(t, est)][tabs[(t, est)].ID == i_].iloc[0] for t in tg]
        rows.append(dict(ID=i_, ntier=len(tg), magV=sub[0].magV,
                         px=np.mean([x.px for x in sub]), py=np.mean([x.py for x in sub]),
                         dx=np.mean([x.dx for x in sub]), dy=np.mean([x.dy for x in sub]),
                         rx=np.mean([x.rx for x in sub]), ry=np.mean([x.ry for x in sub]),
                         R=np.mean([x.R for x in sub]), Rsun=np.mean([x.Rsun for x in sub]),
                         RS=np.mean([x.RS for x in sub]), tags=','.join(sorted(tg))))
    return pd.DataFrame(rows)


def show(nm, d):
    f = fit_L(d)
    if f:
        print('%-46s %5d %9.2f %10.3f %9.3f %8.3f' % (nm, f['n'], f['rin'], f['L'], f['eL'], f['rms']))
    return f


for est in EST_LIST:
    U = union_of(est)
    if not len(U):
        continue
    U.to_csv(os.path.join(OUT, 'union_%s.csv' % est), index=False)
    nw = U.ntier
    print('\n=== union, %s: %d distinct stars; seen in 1 block %d, 2 %d, 3 %d, 4 %d ==='
          % (est, len(U), (nw == 1).sum(), (nw == 2).sum(), (nw == 3).sum(), (nw == 4).sum()))
    print('%-46s %5s %9s %10s %9s %8s' % ('set', 'stars', 'innermost', 'L', '+-', 'rms'))
    show('the 0.4 s block alone (the 2024 choice)', tabs.get(('0p4s_1812', est), U))
    show('two-witness union (>= 2 blocks)', U[U.ntier >= 2])
    show('three-witness union (>= 3 blocks)', U[U.ntier >= 3])
    show('all matched stars, any single block', U)
    a, b = tabs.get(('0p3s_1811', est)), tabs.get(('0p3s_1813', est))
    if a is not None and b is not None:
        both = set(a.ID) & set(b.ID)
        show('the two 0.3 s blocks, seen in both', U[U.ID.isin(both)])

print('\n=== what the short exposure reaches (the Bruns question) ===')
EST = 'windowed'
for tag, stamp, expo, tmid, tiso in BLOCKS:
    d = tabs.get((tag, EST))
    if d is None:
        continue
    inner = d.nsmallest(3, 'Rsun')[['magV', 'Rsun']]
    print('  %-12s %-7s innermost three matched: %s'
          % (tag, expo, '  '.join('G %.1f at %.2f R_sun' % (r.magV, r.Rsun) for _, r in inner.iterrows())))
base = tabs.get(('0p4s_1812', EST))
if base is not None:
    U = union_of(EST)
    inner04 = base.Rsun.min()
    gained = U[(U.Rsun < inner04) & (~U.ID.isin(base.ID))]
    print('  the 0.4 s block reaches %.2f R_sun; the other blocks add %d matched star(s) inside that'
          % (inner04, len(gained)))
    for _, r in gained.sort_values('Rsun').iterrows():
        print('     G %.2f at %.2f R_sun, seen in %d block(s): %s' % (r.magV, r.Rsun, r.ntier, r.tags))
print('\n->', OUT)
