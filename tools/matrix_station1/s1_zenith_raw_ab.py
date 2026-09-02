"""Cell 2, Station 1: the 2 x 2 convention test on the two raw zenith fields Douglas found.

`I:/Mexico 2024/Station 1 Zenith/` holds the source frames of the first two 2024 zenith
fields -- blocks 2024-04-08_05_32_53Z and 05_35_48Z, twenty 3 s frames each at gain 100,
full 9576 x 6388 frame, the 2024 archives 201719 and 202159 having used frames 0001-0019.
They are a consecutive pair 2.9 min apart, so besides the convention question they give one
null pair per convention.

Four conventions, the matrix's held-constant stage-1 set (docs/FIELD_PRESETS.md eclipse-day
standard, as in tools/step3_background_ab.py) with only the two axes under test varied:

    background   annular | Gaussian          (background_subtraction_mode)
    estimator    moments | windowed          (centroid_refine_window)

Per convention and field: a free quintic (the 2024 way, corrections on with the Station 1
site and the block's true mid-time from the headers), then

  1. the LEON 18.3 diagnostic -- the radial residual's slope against magnitude per radius
     bin, tangential as the control. The 2024 moment fits gave +3/+12/+22/+31/+10 mas/mag
     and -34 mas bright-minus-faint beyond 2500 px; the question is what the windowed
     estimator leaves;
  2. the fit's own numbers: stars, rms, plate scale, HC3 scale uncertainty;
  3. the null: field 2 constant-only against field 1's quintic, the eclipse Sun imposed at
     px (4309, 2730), G <= 12, R > 2 R_sun, Method 1 (base and vertical-deg-2) and Method 2.

Usage: s1_zenith_raw_ab.py [convention ...]   (default: all four, in series; run two
processes for two-at-a-time -- each stack holds nineteen 61-Mpx frames).
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
RAW = r"I:/Mexico 2024/Station 1 Zenith"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/zenith_raw_ab"
NX, NY, PS = 9576, 6388, 1.84847
SUNPX, SUNPY, R_SUN_AS = 4309.0, 2730.0, 958.2
BLOCKS = [('f1', '2024-04-08_05_32_53Z', '05:33:30'), ('f2', '2024-04-08_05_35_48Z', '05:36:25')]

S1_BASE = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
           '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
           '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
           '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0']
VARIANTS = {
    'windowed_annular':  ['--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular'],
    'windowed_gaussian': ['--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=Gaussian'],
    'moments_annular':   ['--set', 'centroid_refine_window=False', '--set', 'background_subtraction_mode=annular'],
    'moments_gaussian':  ['--set', 'centroid_refine_window=False', '--set', 'background_subtraction_mode=Gaussian'],
}
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=760.0',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stage1(d, frames, s1):
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY, '-m', 'mee2024.cli', 'stack', *frames, *s1, '--no-scan', '--no-display', '--quiet', '-o', d],
            os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    return z[0] if z else None


def stage2(d, cz, tmid, refs=None, fixed=None, tol=0.3):
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    if hit:
        return hit[0]
    cmd = [PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic', '--date-from-header',
           '--set', 'distortion_fit_tol=%s' % tol, '--set', 'max_star_mag_dist=15',
           '--set', 'rough_match_threshhold=36', *SITE, '--set', 'observation_time=' + tmid,
           '--no-display', '--quiet', '-o', d]
    if refs:
        cmd += ['--fix-distortion', *refs, '--set', 'distortion_fixed_coefficients=' + fixed, '--set', 'max_star_mag_dist=13']
    run(cmd, os.path.join(d, 'stage2.log'))
    hit = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return hit[0] if hit else None


def matched(resfile):
    """CATALOGUE_MATCHED_ERRORS beside a distortion_results.txt, residuals in mas on the sky."""
    z = glob.glob(os.path.join(os.path.dirname(os.path.dirname(resfile)), 'distortion_data*.zip')) or \
        glob.glob(os.path.join(os.path.dirname(resfile), '..', '..', 'distortion_data*.zip'))
    if z:
        zf = zipfile.ZipFile(z[0]); nm = [n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')]
        d = pd.read_csv(zf.open(nm[0]))
    else:
        csv = glob.glob(os.path.join(os.path.dirname(resfile), '**', 'CATALOGUE_MATCHED_ERRORS.csv'), recursive=True)
        d = pd.read_csv(csv[0])
    d.columns = [c.strip() for c in d.columns]
    d['dx'] = (d['RA(obs)'] - d['RA(catalog)'])*np.cos(np.radians(d['DEC(catalog)']))*3600
    d['dy'] = (d['DEC(obs)'] - d['DEC(catalog)'])*3600
    d['r'] = np.hypot(d.px-NX/2, d.py-NY/2)
    ux, uy = (d.px-NX/2)/d.r, (d.py-NY/2)/d.r
    d['rad'] = (d.dx*ux + d.dy*uy)*1000; d['tan'] = (-d.dx*uy + d.dy*ux)*1000
    return d


RB = [(0, 1500), (1500, 2500), (2500, 3500), (3500, 4500), (4500, 6000)]


def bias(D):
    slopes = []
    for lo, hi in RB:
        k = (D.r >= lo) & (D.r < hi)
        slopes.append(np.polyfit(D.magV[k], D.rad[k], 1)[0] if k.sum() > 20 else np.nan)
    outer = D[D.r > 2500]
    bf = outer.rad[outer.magV < 11].mean() - outer.rad[(outer.magV >= 12) & (outer.magV < 13)].mean()
    tslope = np.polyfit(outer.magV, outer['tan'], 1)[0]
    return slopes, bf, tslope


def hc3(twod):
    d = pd.read_csv(twod); W = NX/2.0
    x, y = (d.px.values-NX/2)/W, (d.py.values-NY/2)/W
    X = np.column_stack([np.ones_like(x), x, y, x*x, y*x, y*y]); XtXi = np.linalg.inv(X.T@X)
    h = np.einsum('ij,jk,ik->i', X, XtXi, X); f = 1/(1-h)**2
    cov = lambda e: XtXi @ ((X*(e**2*f)[:, None]).T @ X) @ XtXi
    return float(np.hypot(cov(d.dx_px.values)[1, 1]**.5, cov(d.dy_px.values)[2, 2]**.5)/W*1e6)


def null(twod):
    d = pd.read_csv(twod); d = d[d['magV'] <= 12]
    px, py = d['px'].values, d['py'].values
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec']); dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry); keep = R > 2*R_SUN_AS
    px, py, dx, dy, rx, ry, R = (a[keep] for a in (px, py, dx, dy, rx, ry, R))
    W = NX/2.0; xs, ys = (px-NX/2)/W, (py-NY/2)/W; ux, uy = rx/R, ry/R; n = len(px); Z = np.zeros(n)

    def fit(nd, with_scale):
        cx = [np.ones(n), Z, -(py-NY/2)*PS]; cy = [Z, np.ones(n), (px-NX/2)*PS]; lab = ['N1', 'N2', 'Th']
        if with_scale:
            cx.append((px-NX/2)*PS); cy.append((py-NY/2)*PS); lab.append('S')
        cx.append(ux*R_SUN_AS/R); cy.append(uy*R_SUN_AS/R); lab.append('L')
        if nd:
            for i in range(nd+1):
                for j in range(nd+1-i):
                    if i == 0 and j == 0:
                        continue
                    cx.append(Z); cy.append(xs**i*ys**j); lab.append('v')
        A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
        c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
        return c[lab.index('L')], (1e6*c[lab.index('S')] if with_scale else np.nan)
    return dict(n=n, rms=float(np.sqrt(np.mean(dx**2+dy**2)/2)), Lb=fit(0, False)[0], Lv=fit(2, False)[0],
                L2=fit(2, True)[0], S2=fit(2, True)[1])


conv = sys.argv[1:] or list(VARIANTS)
rows = []
for v in conv:
    print('\n===== %s =====' % v, flush=True)
    res, tw = {}, {}
    for tag, block, tmid in BLOCKS:
        frames = sorted(glob.glob(os.path.join(RAW, block, '*_00[0-1][0-9].FIT')))
        frames = [f for f in frames if not f.endswith('_0000.FIT')]         # 0001-0019, as in 2024
        d1 = os.path.join(OUT, v, tag)
        cz = stage1(d1, frames, S1_BASE + VARIANTS[v])
        if not cz:
            print('  %s: stage 1 FAILED' % tag, flush=True); break
        r = json.load(zipfile.ZipFile(cz).open('results.txt'))
        print('  %s: %d centroids (%s / %s)' % (tag, r['n_centroids'], r.get('centroid estimator'), r.get('background stubtraction mode')), flush=True)
        rf = stage2(os.path.join(d1, 'stage2_free'), cz, tmid)
        if not rf:
            print('  %s: free quintic FAILED' % tag, flush=True); break
        j = json.load(open(rf, encoding='utf-8')); res[tag] = rf
        twod = glob.glob(os.path.join(d1, 'stage2_free', '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        D = matched(rf); sl, bf, ts = bias(D)
        h3 = hc3(twod[0]) if twod else np.nan
        print('  %s free quintic: %d stars rms %.4f" ps %.7f (%+.1f ppm vs 2024 mean) HC3 %.1f ppm' % (tag, j['#stars used'], j['final rms error (arcseconds)'], j['platescale (arcseconds/pixel)'], 1e6*(j['platescale (arcseconds/pixel)']-1.8484656)/1.8484656, h3), flush=True)
        print('     18.3 radial slope by radius bin (mas/mag): ' + '  '.join('%+.1f' % s for s in sl) + '   | bright-minus-faint beyond 2500 px %+.0f mas | tangential slope %+.1f' % (bf, ts), flush=True)
        rows.append(dict(conv=v, field=tag, n=j['#stars used'], rms=j['final rms error (arcseconds)'], ps=j['platescale (arcseconds/pixel)'], hc3=h3,
                         s0=sl[0], s1=sl[1], s2=sl[2], s3=sl[3], s4=sl[4], bf=bf, tslope=ts))
    if len(res) == 2:
        dn = os.path.join(OUT, v, 'null_f2_vs_f1')
        cz2 = glob.glob(os.path.join(OUT, v, 'f2', 'centroid_data*.zip'))[0]
        rn = stage2(dn, cz2, BLOCKS[1][2], refs=[res['f1']], fixed='constant', tol=2.0)
        twod = glob.glob(os.path.join(dn, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
        if twod:
            q = null(twod[0])
            print('  NULL f2 vs f1: N=%d rms %.3f/ax  M1 base %+.3f  M1 v-deg2 %+.3f  M2 %+.3f (S %+.1f ppm)   [2024 archives, same pair: M1 v-deg2 -0.088, M2 -0.022, rms 0.067]' % (q['n'], q['rms'], q['Lb'], q['Lv'], q['L2'], q['S2']), flush=True)
            rows[-1].update({'null_' + k: val for k, val in q.items()})
        else:
            print('  NULL failed', flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'summary_%s.csv' % '_'.join(conv)), index=False)
print('->', OUT)
