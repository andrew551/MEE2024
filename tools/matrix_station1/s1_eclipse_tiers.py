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

Every block is stacked windowed + annular with the Sun masked, and now **with its own matching
dark and the flat**, which is what the 2024 eclipse reduction did (`s1_eclipse_calibrated.py`)
and what keeps hot pixels out of the star list. The 0.4 s block is read from
`eclipse_calibrated/` rather than restacked.

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

BLOCKS = [
    ('0p25s_1810', '2024-04-08_18_10_26Z', 'dark-250ms', '18:10:57', '2024-04-08T18:10:57'),
    ('0p3s_1811',  '2024-04-08_18_11_28Z', 'dark-300ms', '18:11:58', '2024-04-08T18:11:58'),
    ('0p4s_1812',  '2024-04-08_18_12_30Z', 'dark-400ms', '18:13:00', '2024-04-08T18:13:00'),
    ('0p3s_1813',  '2024-04-08_18_13_31Z', 'dark-300ms', '18:14:02', '2024-04-08T18:14:02'),
]
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True']


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


def stack(tag, block, darkset):
    if tag == '0p4s_1812':
        z = glob.glob(os.path.join(CAL04, 'centroid_data*.zip'))
        if z:
            return z[0], CAL04
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('  stacking %s (%s) with %s + flat...' % (tag, block, darkset), flush=True)
        run([PY, '-m', 'mee2024.cli', 'stack', os.path.join(G, 'CapObj', block, '*.FIT'),
             '--dark', os.path.join(G, darkset, 'CapObj', '*', '*.FIT'),
             '--flat', os.path.join(G, 'flat', 'CapObj', '2024-04-08*', '*.FIT'),
             *S1, '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
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


print('=== per block ===', flush=True)
tabs = {}
for tag, block, darkset, tmid, tiso in BLOCKS:
    cz, root = stack(tag, block, darkset)
    if not cz:
        print('  %s: STAGE 1 FAILED' % tag, flush=True); continue
    r = json.load(zipfile.ZipFile(cz).open('results.txt'))
    zp = stage2(root, cz, tmid)
    if not zp:
        print('  %s: stage 2 FAILED' % tag, flush=True); continue
    d, j = table(zp, tiso)
    tabs[tag] = d
    sci = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)]
    f = fit_L(d)
    print('  %-12s %4d centroids, %4d matched, %3d in the science set; innermost matched star'
          ' %.2f R_sun (science %.2f); stage-2 rms %.3f"; L = %+.3f +- %.3f"'
          % (tag, r['n_centroids'], len(d), len(sci), d.Rsun.min(), sci.Rsun.min() if len(sci) else np.nan,
             j['final rms error (arcseconds)'], f['L'], f['eL']), flush=True)

if len(tabs) < 2:
    raise SystemExit('need at least two blocks')

# ---------------------------------------------------------------- the union
print('\n=== the union: how many blocks saw each star ===')
allids = {}
for tag, d in tabs.items():
    for i in d.ID.values:
        allids.setdefault(i, []).append(tag)
nwit = pd.Series({k: len(v) for k, v in allids.items()})
print('  %d distinct matched stars; seen in 1 block: %d, 2: %d, 3: %d, 4: %d'
      % (len(nwit), (nwit == 1).sum(), (nwit == 2).sum(), (nwit == 3).sum(), (nwit == 4).sum()))

# the union table: one row per star, residuals averaged over the blocks that saw it
rows = []
for i, tags in allids.items():
    sub = [tabs[t][tabs[t].ID == i].iloc[0] for t in tags]
    rows.append(dict(ID=i, ntier=len(tags), magV=sub[0].magV,
                     px=np.mean([s.px for s in sub]), py=np.mean([s.py for s in sub]),
                     dx=np.mean([s.dx for s in sub]), dy=np.mean([s.dy for s in sub]),
                     rx=np.mean([s.rx for s in sub]), ry=np.mean([s.ry for s in sub]),
                     R=np.mean([s.R for s in sub]), Rsun=np.mean([s.Rsun for s in sub]),
                     RS=np.mean([s.RS for s in sub]),
                     tags=','.join(sorted(tags))))
U = pd.DataFrame(rows)
U.to_csv(os.path.join(OUT, 'union_star_table.csv'), index=False)

print('\n=== L, Method 2 with the isotropic scale ===')
print('%-42s %5s %8s %10s %9s %8s' % ('set', 'stars', 'innermost', 'L', '+-', 'rms'))
def show(nm, d):
    f = fit_L(d)
    if f:
        print('%-42s %5d %8.2f %10.3f %9.3f %8.3f' % (nm, f['n'], f['rin'], f['L'], f['eL'], f['rms']))
    return f
r04 = show('0.4 s block alone (the 2024 choice)', tabs.get('0p4s_1812', U))
rtw = show('two-witness union (>= 2 blocks)', U[U.ntier >= 2])
ral = show('all matched stars, any single block', U)
r3w = show('three-witness union (>= 3 blocks)', U[U.ntier >= 3])
r33 = None
if '0p3s_1811' in tabs and '0p3s_1813' in tabs:
    both03 = set(tabs['0p3s_1811'].ID) & set(tabs['0p3s_1813'].ID)
    r33 = show('the two 0.3 s blocks, seen in both', U[U.ID.isin(both03)])

print('\n=== what the short exposure reaches (the Bruns question) ===')
for tag in ('0p25s_1810', '0p3s_1811', '0p4s_1812', '0p3s_1813'):
    if tag not in tabs:
        continue
    d = tabs[tag]
    sci = d[(d.magV <= MAGCUT) & (d.Rsun > RCUT)]
    inner = d.nsmallest(3, 'Rsun')[['ID', 'magV', 'Rsun']]
    print('  %-12s innermost three matched: %s'
          % (tag, '  '.join('G %.1f at %.2f R_sun' % (r.magV, r.Rsun) for _, r in inner.iterrows())))
base = tabs.get('0p4s_1812')
if base is not None:
    inner04 = base.Rsun.min()
    gained = U[(U.Rsun < inner04) & (~U.ID.isin(base.ID))]
    print('  the 0.4 s block reaches %.2f R_sun; the other blocks add %d matched star(s) inside that'
          % (inner04, len(gained)))
    for _, r in gained.sort_values('Rsun').iterrows():
        print('     G %.2f at %.2f R_sun, seen in %d block(s): %s' % (r.magV, r.Rsun, r.ntier, r.tags))
print('\n->', OUT)
