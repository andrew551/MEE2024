"""Cell 2, Station 1 (Mexico 2024): the zenith floor and the null test, from the 2024-era
zenith archives -- no raw frames needed.

The raw zenith FITS are not on this machine (the Cloud-Drive folder the 2024 reductions
were run from holds only the 396-byte ASICap sidecars), but the seventeen stage-1 archives
and their free-quintic fits are (`D:/MEE2024 output/Station 1/zenith fields` and
`zenith calibrations`), and the current pipeline imports a v0.4.5 result as a frozen
reference without complaint (tested: one constant-only refit, rc 0, scale imported to the
seventh digit). So the two floor measurements every cell needs can be made now:

  1. **the quasi-static floor** -- each field re-fitted with the quintic-and-above FROZEN
     from the average of all seventeen and the quadratic free, the table's construction;
  2. **the null test** -- each field fitted constant-only against the PREVIOUS field in
     capture order, the ECLIPSE field's own Sun position imposed (px 4309, 2730 from the
     ephemeris at 18:12 UTC through the eclipse fit's affine), the science cuts, and BOTH
     estimators: Method 1 with the vertical-deg-2 nuisance (for comparison with the other
     cells) and Method 2 with the scale free (how Station 1 is actually reduced).

Three things about this campaign that make its floor row different from the others:

  * the zenith fields were taken 05:32-06:15 UTC on eclipse day, twelve hours before
    totality, at 3 s and gain 100 against the eclipse's 0.25-0.4 s at gain 0. Capture order
    and times come from the block folder names in each archive's `source_files` (the 2024
    fits carry a placeholder observation time of 05:45 for all seventeen);
  * the Sun stood at alt 70 deg (airmass 1.07), so these near-zenith fields are only ~20 deg
    from the eclipse geometry -- much closer than Leon's zenith set was to its horizon;
  * the centroids are the 2024 footprint moments. Everything measured here is in that
    convention, and the eclipse field must be reduced in the same one until the raw zenith
    frames turn up and the quintic can be re-derived windowed.

Writes station1_record/zenith_floor.csv and zenith_nulls.csv (the corrections-off first pass
is kept beside them as zenith_floor_nocorr.csv / zenith_nulls_nocorr.csv).
"""
import glob, json, os, re, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
Z = r"D:/MEE2024 output/Station 1/zenith calibrations"
F = r"D:/MEE2024 output/Station 1/zenith fields"
OUT = r"D:/MEE2024 output/MEE_output/station1_record"
# The 2024 quintics were fitted against APPARENT places -- refraction and aberration on, the
# site below, 10 C / 760 mb / RH 0.25, and a placeholder 05:45 UTC for all seventeen -- so the
# refits must be too. Run with them OFF (the first pass, kept as *_nocorr.*) every field
# disagrees with its own quintic by -163 ppm of scale: the zenith refraction compression plus
# aberration, which the frozen model carries and the true-place catalogue does not. Method 1
# then reads that as -3.9" of "deflection" in every pair; Method 2 absorbs it into S exactly.
CORR = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_time=05:45', '--set', 'observation_long=105 16 22.1 W',
        '--set', 'observation_lat=23 50 58.3 N', '--set', 'observation_temp=10.0',
        '--set', 'observation_pressure=760.0', '--set', 'observation_humidity=0.25',
        '--set', 'observation_height=2400.0']
QF, NULLS = os.path.join(OUT, 'zenith_quadfree_corr'), os.path.join(OUT, 'zenith_nulls_corr')
NX, NY, PS = 9576, 6388, 1.84847
SUNPX, SUNPY, R_SUN_AS = 4309.0, 2730.0, 958.2
MAGCUT, RCUT = 12.0, 2.0
os.makedirs(QF, exist_ok=True); os.makedirs(NULLS, exist_ok=True)


def fields():
    out = []
    for z in sorted(glob.glob(os.path.join(F, 'centroid_data*.zip'))):
        import zipfile
        r = json.load(zipfile.ZipFile(z).open('results.txt'))
        src = r.get('source_files')
        if isinstance(src, str):
            src = json.loads(src.replace("'", '"'))
        m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})Z', src[0])
        minute = int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/60
        stamp = os.path.basename(z)[13:27]
        ref = glob.glob(os.path.join(Z, 'DISTORTION_OUTPUT*__centroid_data%s*' % stamp, '**', 'distortion_results.txt'), recursive=True)
        if not ref:
            continue
        j = json.load(open(ref[0], encoding='utf-8'))
        out.append(dict(stamp=stamp, zip=z, ref=ref[0], date=m.group(1), minute=minute,
                        block=m.group(0), ps=j['platescale (arcseconds/pixel)'], n=j['#stars used'],
                        rms=j['final rms error (arcseconds)'], ra=j['RA'], dec=j['DEC']))
    return sorted(out, key=lambda x: x['minute'])


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def refit(root, tag, zip_path, refs, fixed, tol):
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    run([PY, '-m', 'mee2024.cli', 'distortion', zip_path, '--order', 'quintic',
         '--fix-distortion', *refs, '--set', 'distortion_fixed_coefficients=' + fixed,
         '--set', 'distortion_fit_tol=%s' % tol, '--set', 'max_star_mag_dist=13',
         '--set', 'rough_match_threshhold=36', *CORR, '--no-display', '--quiet', '-o', d],
        os.path.join(d, 'stage2.log'))
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def design(px, py, rx, ry, R, nuis_deg=None, with_scale=False):
    W = NX/2.0
    xs, ys = (px-NX/2)/W, (py-NY/2)/W
    ux, uy = rx/R, ry/R
    n = len(px); Zc = np.zeros(n)
    cx = [np.ones(n), Zc, -(py-NY/2)*PS]; cy = [Zc, np.ones(n), (px-NX/2)*PS]; lab = ['N1', 'N2', 'Th']
    if with_scale:
        cx.append((px-NX/2)*PS); cy.append((py-NY/2)*PS); lab.append('S')
    cx.append(ux*R_SUN_AS/R); cy.append(uy*R_SUN_AS/R); lab.append('L')
    if nuis_deg:
        for i in range(nuis_deg+1):
            for j in range(nuis_deg+1-i):
                if i == 0 and j == 0:
                    continue
                cx.append(Zc); cy.append(xs**i*ys**j); lab.append(f'v{i}{j}')
    return np.vstack([np.column_stack(cx), np.column_stack(cy)]), lab


def fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=None, with_scale=False):
    A, lab = design(px, py, rx, ry, R, nuis_deg, with_scale)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[lab.index('L')], (1e6*c[lab.index('S')] if with_scale else np.nan)


FL = fields()
print('%d zenith fields in capture order:' % len(FL))
for f in FL:
    print('  %s  %s  %02d:%02d  RA %7.3f DEC %+7.3f  ps %.7f  n=%4d rms %.4f'
          % (f['stamp'], f['block'], int(f['minute']//60), int(f['minute'] % 60), f['ra'], f['dec'], f['ps'], f['n'], f['rms']))
ps = np.array([f['ps'] for f in FL])
print('  plate scale over the session: mean %.7f, rms %.1f ppm, first-to-last %+.1f ppm'
      % (ps.mean(), 1e6*ps.std(ddof=1)/ps.mean(), 1e6*(ps[-1]-ps[0])/ps[0]))

# ---------------------------------------------------------------- A. the quasi-static floor
allrefs = [f['ref'] for f in FL]
rows = []
for f in FL:
    path = refit(QF, f['stamp'], f['zip'], allrefs, 'quadratic', 0.2)
    if path is None:
        print(f['stamp'], 'quadratic-free refit failed', flush=True); continue
    d = pd.read_csv(path); d = d[d['magV'] <= MAGCUT]
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec']); dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    m = np.hypot(dx, dy); lim = max(3*1.4826*np.median(np.abs(m-np.median(m))) + np.median(m), 2.5); k = m < lim
    dx, dy = dx[k], dy[k]
    rows.append(dict(stamp=f['stamp'], block=f['block'], n=int(k.sum()), qs_rms=float(np.sqrt(np.mean(dx**2+dy**2))),
                     qs_x=float(np.sqrt(np.mean(dx**2))), qs_y=float(np.sqrt(np.mean(dy**2)))))
    print('  %s quadratic-free: N=%4d  rms %.3f  (x %.3f, y %.3f)' % (f['stamp'], k.sum(), rows[-1]['qs_rms'], rows[-1]['qs_x'], rows[-1]['qs_y']), flush=True)
QS = pd.DataFrame(rows); QS.to_csv(os.path.join(OUT, 'zenith_floor.csv'), index=False)
print('\nStation 1 quadratic-free floor (quintic frozen from the 17-field average): rms %.3f (%.3f-%.3f), sensor y/x %.2f'
      % (QS.qs_rms.mean(), QS.qs_rms.min(), QS.qs_rms.max(), QS.qs_y.mean()/QS.qs_x.mean()), flush=True)

# ---------------------------------------------------------------- B. the nulls
rng = np.random.default_rng(11)
nrows = []
for prev, cur in zip(FL, FL[1:]):
    path = refit(NULLS, cur['stamp'] + '_vs_' + prev['stamp'], cur['zip'], [prev['ref']], 'constant', 2.0)
    if path is None:
        print(cur['stamp'], 'null refit failed', flush=True); continue
    d = pd.read_csv(path); d = d[d['magV'] <= MAGCUT]
    px, py = d['px'].values, d['py'].values
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec']); dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    err = d['error_arcsec'].values
    rx, ry = (px-SUNPX)*PS, (py-SUNPY)*PS; R = np.hypot(rx, ry)
    keep = R > RCUT*R_SUN_AS
    px, py, dx, dy, rx, ry, R, err = (a[keep] for a in (px, py, dx, dy, rx, ry, R, err))
    Lb, _ = fit_L(dx, dy, px, py, rx, ry, R)
    Lv, _ = fit_L(dx, dy, px, py, rx, ry, R, 2)
    L2, S2 = fit_L(dx, dy, px, py, rx, ry, R, 2, True)
    L2b, S2b = fit_L(dx, dy, px, py, rx, ry, R, None, True)
    floor = float(np.std([fit_L(dx + rng.normal(0, err/np.sqrt(2)), dy + rng.normal(0, err/np.sqrt(2)), px, py, rx, ry, R, 2)[0]
                          for _ in range(40)], ddof=1))
    gap = cur['minute'] - prev['minute']
    step = 1e6*(cur['ps']-prev['ps'])/prev['ps']
    h = 1/np.mean((R_SUN_AS/R)**2)
    nrows.append(dict(field=cur['stamp'], ref=prev['stamp'], gap_min=gap, step_ppm=step, n=int(keep.sum()), h=h,
                      Lb=Lb, Lv=Lv, L_m2=L2, S_m2=S2, L_m2_base=L2b, S_m2_base=S2b, floor=floor,
                      rms=float(np.sqrt(np.mean(dx**2+dy**2)/2))))
    print('  %s vs %s  gap %4.1f min  N=%4d h=%4.1f rms %.3f/ax  M1 base %+.3f  M1 v-deg2 %+.3f  M2 %+.3f (S %+6.1f ppm)  step %+5.1f ppm  floor %.3f'
          % (cur['stamp'][-6:], prev['stamp'][-6:], gap, keep.sum(), h, nrows[-1]['rms'], Lb, Lv, L2, S2, step, floor), flush=True)
N = pd.DataFrame(nrows); N.to_csv(os.path.join(OUT, 'zenith_nulls.csv'), index=False)
print('\n%d Station 1 zenith nulls, consecutive fields %.1f-%.1f min apart, h = %.1f-%.1f' % (len(N), N.gap_min.min(), N.gap_min.max(), N.h.min(), N.h.max()))
for col, nm in (('Lb', 'Method 1, base'), ('Lv', 'Method 1, v-deg2'), ('L_m2', 'Method 2 (scale free), v-deg2'), ('L_m2_base', 'Method 2, base')):
    v = N[col].values
    print('  %-30s rms %.3f  max %.3f  mean %+.3f' % (nm, np.sqrt((v**2).mean()), np.abs(v).max(), v.mean()))
print('  fitted scale steps (M2): rms %.1f ppm; free-fit steps rms %.1f ppm; bootstrap floor %.3f' % (np.sqrt((N.S_m2.values**2).mean()), np.sqrt((N.step_ppm.values**2).mean()), N.floor.median()))
print('  (Leon zenith: M1 v-deg2 0.124, scale-free 0.061; Leakey 0.120 / 0.069; Bruns night one-sided 0.150 / 0.099)')
print('->', OUT)
