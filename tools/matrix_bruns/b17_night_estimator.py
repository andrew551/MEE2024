"""Cell 1: settle Bruns' centroid convention on his NIGHT calibrations, not his day-time ones.

Douglas, 2026-09-03, on being told Leon's eclipse field cannot distinguish the conventions and
that its choice was made on the zenith diagnostic instead: "this should probably also be
decided by the Bruns 2017 night-time calibrations, rather than the day-time ones."

He is right, and the reason is the same one that makes the Mexico eclipse field so different
from its zenith fields. The magnitude-dependent centroid bias that separates a footprint
moment from a windowed centroid is a property of the PSF's asymmetry, and it is measured by
asking whether the mean radial residual depends on a star's brightness. On a **day-time**
field that measurement is confounded: `docs/CALIBRATION_FRAMES.md` records Bruns' L/R
calibration fields sitting in "a daylight-sky-limited regime at 3400 ADU background", and a
bright background is exactly what makes an annular subtraction leave a flux-dependent residual
of its own. The cell-1 A/B that put the estimator at "under 2 ppm of plate scale" was run on
those day-time fields.

His **night** fields have none of that problem, and they are far better material besides:
thirty of them, about 1150 matched stars each at a fit residual near 0.064 ", against 26 stars
on the eclipse field. Every one kept its `STACKED*.fit`, so each can be re-centroided in both
conventions with the 2017 alignment and stack untouched -- the same trick validated on the
Mexico zenith fields to 0.8 ppm of plate scale (`s1_zenith_recentroid.py`).

Radius is reported as a **fraction of the frame's half-diagonal** rather than in pixels, so the
bins mean the same thing on Bruns' 3296 x 2472 sensor as on Leon's 4144 x 2822 and Station 1's
9576 x 6388, and the three campaigns' numbers can be put in one table.

The comparison to beat, both measured on dark-sky calibration fields:

    Leon 2026 zenith, footprint moments   +299 mas bright-minus-faint beyond r/rmax ~ 0.8,
                                          in twelve fields out of twelve (LEON 18.3)
    Station 1 zenith, footprint moments   +22 to +31 mas/mag in the outer bins, 17/17 fields,
                                          falling to +0.3 to +3.2 windowed
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
NIGHTS = r"D:/MEE2024 output/MEE_output/bruns2017_nights"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_night_estimator"
NX, NY = 3296, 2472
RMAX = float(np.hypot(NX/2, NY/2))
MAXFIELDS = int(os.environ.get('B17_NIGHT_FIELDS', '12'))
FRACBINS = [(0.0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.85), (0.85, 1.05)]

S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0']
CONV = {'windowed': ['--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular'],
        'moments':  ['--set', 'centroid_refine_window=False', '--set', 'background_subtraction_mode=annular']}
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def fields():
    out = []
    for d in sorted(glob.glob(os.path.join(NIGHTS, 'EC*'))):
        img = sorted(glob.glob(os.path.join(d, 'CENTROID_OUTPUT*', 'STACKED2*.fit')))
        img = [p for p in img if 'FLOAT' not in os.path.basename(p)]
        res = sorted(glob.glob(os.path.join(d, 'stage2', '**', 'distortion_results.txt'), recursive=True))
        if not img or not res:
            continue
        j = json.load(open(res[0], encoding='utf-8'))
        out.append(dict(name=os.path.basename(d), img=img[0], date=j.get('observation_date'),
                        n24=j.get('#stars used'), rms24=j.get('final rms error (arcseconds)')))
    return out[:MAXFIELDS]


def slopes(resfile):
    z = glob.glob(os.path.join(os.path.dirname(os.path.dirname(resfile)), 'distortion_data*.zip')) or \
        glob.glob(os.path.join(os.path.dirname(resfile), '..', '..', 'distortion_data*.zip'))
    zf = zipfile.ZipFile(z[0])
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    d['dx'] = (d['RA(obs)']-d['RA(catalog)'])*np.cos(np.radians(d['DEC(catalog)']))*3600
    d['dy'] = (d['DEC(obs)']-d['DEC(catalog)'])*3600
    d['r'] = np.hypot(d.px-NX/2, d.py-NY/2)
    d['frac'] = d.r/RMAX
    ux, uy = (d.px-NX/2)/d.r, (d.py-NY/2)/d.r
    d['rad'] = (d.dx*ux + d.dy*uy)*1000
    d['tan'] = (-d.dx*uy + d.dy*ux)*1000
    sl, tl = [], []
    for lo, hi in FRACBINS:
        k = (d.frac >= lo) & (d.frac < hi)
        sl.append(np.polyfit(d.magV[k], d.rad[k], 1)[0] if k.sum() > 30 else np.nan)
        tl.append(np.polyfit(d.magV[k], d['tan'][k], 1)[0] if k.sum() > 30 else np.nan)
    o = d[d.frac > 0.65]
    bf = o.rad[o.magV < o.magV.quantile(0.25)].mean() - o.rad[o.magV > o.magV.quantile(0.75)].mean()
    return sl, tl, float(bf), len(d), d.magV.min(), d.magV.max()


FL = fields()
print('%d Bruns night fields with a stacked image and a 2017-era fit (of 30 available)\n' % len(FL), flush=True)
rows = []
for f in FL:
    for conv in ('windowed', 'moments'):
        d1 = os.path.join(OUT, '%s_%s' % (f['name'], conv))
        os.makedirs(d1, exist_ok=True)
        z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
        if not z:
            run([PY, '-m', 'mee2024.cli', 'stack', f['img'], *S1, *CONV[conv],
                 '--no-scan', '--no-display', '--quiet', '-o', d1], os.path.join(d1, 'stage1.log'))
            z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
        if not z:
            print('  %s/%s: stage 1 FAILED' % (f['name'], conv), flush=True); continue
        d2 = os.path.join(d1, 'stage2_free')
        os.makedirs(d2, exist_ok=True)
        res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if not res:
            run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'cubic',
                 '--set', 'distortion_fit_tol=0.4', '--set', 'max_star_mag_dist=13',
                 '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=False',
                 '--set', 'enable_corrections_ref=False', '--set', 'observation_date=' + str(f['date']),
                 '--set', 'guess_date=False', '--no-display', '--quiet', '-o', d2],
                os.path.join(d2, 'stage2.log'))
            res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if not res:
            print('  %s/%s: stage 2 FAILED' % (f['name'], conv), flush=True); continue
        j = json.load(open(res[0], encoding='utf-8'))
        sl, tl, bf, n, gmin, gmax = slopes(res[0])
        rows.append(dict(field=f['name'], conv=conv, n=j['#stars used'],
                         rms=j['final rms error (arcseconds)'], ps=j['platescale (arcseconds/pixel)'],
                         bf=bf, gmin=gmin, gmax=gmax,
                         **{('s%d' % i): s for i, s in enumerate(sl)},
                         **{('t%d' % i): s for i, s in enumerate(tl)}))
        print('  %-6s %-9s %5d stars rms %.4f" ps %.6f | radial slope by r/rmax: %s | bright-faint %+.0f mas'
              % (f['name'], conv, j['#stars used'], j['final rms error (arcseconds)'],
                 j['platescale (arcseconds/pixel)'],
                 ' '.join('%+6.1f' % s for s in sl), bf), flush=True)

T = pd.DataFrame(rows)
if not len(T):
    raise SystemExit('nothing reduced')
T.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
print('\n=== Bruns 2017 NIGHT fields, %d fields, radial bias in mas per magnitude ===' % T.field.nunique())
print('%-10s' % 'r/rmax' + ''.join('%13s' % ('%.2f-%.2f' % b) for b in FRACBINS) + '%16s' % 'bright-faint')
for conv in ('windowed', 'moments'):
    t = T[T.conv == conv]
    if not len(t):
        continue
    print('%-10s' % conv + ''.join('%13.1f' % t['s%d' % i].mean() for i in range(5)) + '%14.0f mas' % t.bf.mean())
    print('%-10s' % '  (tang)' + ''.join('%13.1f' % t['t%d' % i].mean() for i in range(5)))
for conv in ('windowed', 'moments'):
    t = T[T.conv == conv]
    if len(t):
        same = int((np.sign(t.bf) == np.sign(t.bf.mean())).sum())
        print('  %-9s bright-minus-faint %+.0f mas, same sign in %d of %d fields; median fit rms %.4f "'
              % (conv, t.bf.mean(), same, len(t), t.rms.median()))
w, m = T[T.conv == 'windowed'], T[T.conv == 'moments']
if len(w) and len(m):
    print('\n  outer-bin slope, moments minus windowed: %+.1f mas/mag'
          % (m[['s3', 's4']].mean().mean() - w[['s3', 's4']].mean().mean()))
    print('  fit residual, moments / windowed: %.4f / %.4f " = %.2f' % (m.rms.median(), w.rms.median(), m.rms.median()/w.rms.median()))
    print('  plate scale, moments minus windowed: %+.1f ppm' % (1e6*(m.ps.mean()-w.ps.mean())/w.ps.mean()))
print('\ncompare: Station 1 zenith moments +22..+31 mas/mag outer, windowed +0.3..+3.2 (17/17 fields)')
print('         Leon 2026 zenith moments +299 mas bright-minus-faint outer (12/12 fields)')
print('\n->', OUT)
