"""Cell 2: re-stack the eclipse blocks with the disk occulter and per-frame coronal subtraction.

Douglas, 2026-09-03: "Shall we now apply a gaussian blur/subtract to the eclipse fields along
with the more recently developed occulting routine?"

Yes, and it is the one remaining lever that can change the answer rather than refine it. The
2024 stacks — which everything in cell 2 so far has been built on, because re-centroiding them
is as good as re-stacking — used `blob` masking and **no coronal subtraction at all**. Three
of the open items point at the same fix:

  * the close-in star at 1.87 R_sun that all three short blocks see and the 0.4 s block misses
    is displaced -15.9 " against GR's +0.94, and sits exactly where a steep unsubtracted
    coronal gradient under an annular background would push it;
  * the usable inner limit is ~2.0 R_sun in every block, where Bruns reached further in --
    and `subtract_coronal_background` is documented as "the reason his inner-annulus stars
    were measurable at all";
  * the sky carries a **time-varying** gradient from cloud, tilting up to 380 ADU/s across the
    frame and rotating within a minute. A per-frame subtraction removes that with the corona;
    anything estimated from a stack cannot, because the stack has already averaged four
    different sky states together.

`subtract_coronal_background` runs inside the per-frame preprocessing (`stacker_implementation`
~1156), so unlike everything else in cell 2 this **must** be done from the raw frames. It is
Bruns' method: blur the frame heavily, subtract, add a pedestal -- with the blur estimated from
unsaturated pixels only, `blur(img*valid)/blur(valid)`, which is what stops the saturated
core's plateau cutting an over-subtracted ring through the inner annulus. Measured on the
Bruns 2017 frames: stacked sky sigma 2385 ADU the naive way against 66 this way.

Held to the 2024 settings in every other respect, with the two additions and the calibration
the 2024 run itself used:

    eclipse_mask_mode = disk (not blob), eclipse_disk_margin_px = 10
    coronal_subtract = True, sigma 10 px, pedestal 2000 ADU
    --dark <matching tier> --flat, as the 2024 reduction did
    windowed + annular, 4.0 sigma, min_area 2, sigma_subtract 0

The 0.25 s block starts at frame 0015, as the 2024 operator had it: the first fifteen sit at
second contact and re-stacking all 124 fails outright in the aligner.

Usage: s1_eclipse_corona.py [tag ...]   (default: the 0.4 s block alone, which is the
reduction of record and the right place to validate the settings before spending hours on
the other three).
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time
import astropy.units as u

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
RECEN = r"D:/MEE2024 output/MEE_output/station1_record/zenith_recentroid"
OLD = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_tiers"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/eclipse_corona"
NX, NY, PS = 9576, 6388, 1.84847
MAGCUT, RCUT, RMAX = 12.0, 2.0, 9.0

# tag, raw block, dark set, first frame to use, mid-time
BLOCKS = {
    '0p25s_1810': ('2024-04-08_18_10_26Z', 'dark-250ms', 15, '18:11:12'),
    '0p3s_1811':  ('2024-04-08_18_11_28Z', 'dark-300ms', 0,  '18:11:58'),
    '0p4s_1812':  ('2024-04-08_18_12_30Z', 'dark-400ms', 0,  '18:13:00'),
    '0p3s_1813':  ('2024-04-08_18_13_31Z', 'dark-300ms', 0,  '18:14:02'),
}
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      # the two additions under test
      '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
      '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=10.0',
      '--set', 'coronal_pedestal_adu=2000.0']


def met(tmid):
    return ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
            '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
            '--set', 'observation_time=' + tmid, '--set', 'observation_long=105 16 22.1 W',
            '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=15.0',
            '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
            '--set', 'observation_height=2400.0']


REF = sorted(glob.glob(os.path.join(RECEN, '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))
assert REF, 'the seventeen-field windowed reference must exist first (s1_zenith_recentroid.py)'
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stack(tag):
    block, darkset, first, tmid = BLOCKS[tag]
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if z:
        return z[0], d
    frames = sorted(glob.glob(os.path.join(G, 'CapObj', block, '*.FIT')))[first:]
    print('  %s: stacking %d frames from %s with %s + flat, disk occulter, coronal subtract'
          % (tag, len(frames), block, darkset), flush=True)
    run([PY, '-m', 'mee2024.cli', 'stack', *frames,
         '--dark', os.path.join(G, darkset, 'CapObj', '*', '*.FIT'),
         '--flat', os.path.join(G, 'flat', 'CapObj', '2024-04-08*', '*.FIT'),
         *S1, '--no-scan', '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return (z[0], d) if z else (None, d)


def stage2(root, cz, tmid):
    d = os.path.join(root, 'stage2')
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


def table(zp, tmid):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    j = json.load(zf.open([n for n in zf.namelist() if n.endswith('distortion_results.txt')][0]))
    d = d[d['flag_is_outlier'] == False].copy()
    sun = get_sun(Time('2024-04-08T' + tmid + ':00' if len(tmid) == 5 else '2024-04-08T' + tmid, scale='utc'))
    RS = float(np.degrees(np.arcsin((696000*u.km/sun.distance).decompose().value))*3600)
    ra0, de0 = d['RA(catalog)'].mean(), d['DEC(catalog)'].mean()
    X = (d['RA(catalog)'].values-ra0)*np.cos(np.radians(de0)); Y = d['DEC(catalog)'].values-de0
    A = np.c_[X, Y, np.ones_like(X)]
    ax, *_ = np.linalg.lstsq(A, d.px.values, rcond=None); ay, *_ = np.linalg.lstsq(A, d.py.values, rcond=None)
    sx, sy = (sun.ra.deg-ra0)*np.cos(np.radians(de0)), sun.dec.deg-de0
    SPX, SPY = float(np.array([sx, sy, 1])@ax), float(np.array([sx, sy, 1])@ay)
    ox = np.c_[(d['RA(obs)'].values-ra0)*np.cos(np.radians(de0)), d['DEC(obs)'].values-de0, np.ones(len(d))]
    cm = np.c_[X, Y, np.ones(len(d))]
    d['dx'] = (ox@ax - cm@ax)*PS; d['dy'] = (ox@ay - cm@ay)*PS
    d['rx'] = (d.px.values-SPX)*PS; d['ry'] = (d.py.values-SPY)*PS
    d['R'] = np.hypot(d.rx, d.ry); d['Rsun'] = d.R/RS; d['RS'] = RS
    return d, j


def fit(d, vet=True):
    d = d[(d.Rsun > RCUT) & (d.Rsun < RMAX) & (d.magV <= MAGCUT)].copy()
    for _ in range(4 if vet else 1):
        p, q, r = d.px.values, d.py.values, d.R.values
        ux, uy = d.rx.values/r, d.ry.values/r; RS = d.RS.values
        dx = d.dx.values-np.median(d.dx.values); dy = d.dy.values-np.median(d.dy.values)
        n = len(d); Z = np.zeros(n); xs, ys = (p-NX/2)*PS, (q-NY/2)*PS
        M = np.vstack([np.column_stack([np.ones(n), Z, -ys, xs, ux*RS/r]),
                       np.column_stack([Z, np.ones(n), xs, ys, uy*RS/r])])
        sc = np.sqrt((M**2).mean(0)); Mn = M/sc; b = np.concatenate([dx, dy])
        c, *_ = np.linalg.lstsq(Mn, b, rcond=None)
        res = b - Mn@c; s2 = (res**2).sum()/(len(b)-Mn.shape[1])
        e = np.sqrt(np.abs(np.diag(s2*np.linalg.pinv(Mn.T@Mn)))); c, e = c/sc, e/sc
        per = np.hypot(res[:n], res[n:])
        lim = max(4.0*1.4826*np.median(np.abs(per-np.median(per))) + np.median(per), 0.6)
        if not vet or (per < lim).all():
            break
        d = d[per < lim]
    return dict(n=n, L=c[4], eL=e[4], rms=np.sqrt(s2), rin=d.Rsun.min())


tags = sys.argv[1:] or ['0p4s_1812']
print('reference: seventeen-field windowed, %d files\n' % len(REF), flush=True)
for tag in tags:
    cz, root = stack(tag)
    if not cz:
        print('  %s: STAGE 1 FAILED -- see %s' % (tag, os.path.join(root, 'stage1.log')), flush=True)
        continue
    r = json.load(zipfile.ZipFile(cz).open('results.txt'))
    tmid = BLOCKS[tag][3]
    zp = stage2(root, cz, tmid)
    if not zp:
        print('  %s: stage 2 FAILED' % tag, flush=True); continue
    d, j = table(zp, tmid)
    f = fit(d)
    print('\n=== %s, with the disk occulter and coronal subtraction ===' % tag)
    print('  %d centroids, %d matched, %d in the science set' % (r['n_centroids'], len(d), f['n']))
    print('  innermost matched star %.2f R_sun (science set %.2f); stage-2 rms %.3f"'
          % (d.Rsun.min(), f['rin'], j['final rms error (arcseconds)']))
    print('  L = %+.3f +- %.3f", per-star residual %.3f"' % (f['L'], f['eL'], f['rms']))
    old = glob.glob(os.path.join(OLD, tag + '_windowed', 'centroid_data*.zip'))
    if old:
        ro = json.load(zipfile.ZipFile(old[0]).open('results.txt'))
        print('  the same block WITHOUT either (the 2024 stack, re-centroided windowed): %d centroids'
              % ro['n_centroids'])
    inner = d.nsmallest(6, 'Rsun')[['magV', 'Rsun']]
    print('  innermost six matched: %s'
          % '  '.join('G %.1f at %.2f' % (x.magV, x.Rsun) for _, x in inner.iterrows()))
print('\n->', OUT)
