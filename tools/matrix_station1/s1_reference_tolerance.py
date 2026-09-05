"""Station 1: does the fit tolerance used to BUILD the zenith reference move L?

Douglas, 2026-09-05: the windowed reference was fitted at a 0.3" tolerance where the 2025
analysis used 0.1" and the Bruns/Leon night calibrations 0.2" -- that admits many more stars;
what did it do to the residual, and why 0.3? The honest answer to "why" is that 0.3 was chosen
when `s1_zenith_recentroid.py` was written and never justified. This tool supplies the
measurement that should have gone with it.

The tolerance on a free quintic is a selection gate on the residual: a tight gate clips the
tails of the star population and reports an rms below the population's true scatter, a loose
one keeps the tails and reports more. The reported rms therefore says little on its own (0.061"
at 0.1 against 0.124" at 0.3 is mostly the gate). What matters for the record is whether the
COEFFICIENTS that are transferred to the eclipse field depend on the gate -- and that is
answered by rebuilding the seventeen-field reference at 0.1" and 0.2", refitting the four
eclipse blocks two-pass against each, and pooling.

Then Douglas' point, measured (2026-09-05): a quintic needs corner coverage more than it needs
a low rms, because the corners are where its terms have leverage and where the residuals are
largest, so a tight gate strips exactly the stars the high orders depend on. Three things are
computed from the finished fits: (a) accepted stars per field by frame radius, with the
residual rms in each annulus; (b) the field-to-field stability of the model -- the rms over
the seventeen fields of the displacement each field's own quintic predicts, on a grid, in
mas; (c) the difference between the seventeen-field average model at 0.1 or 0.2 and the
record's at 0.3, on the same grid.

Writes station1_record/zenith_recentroid_tol/tol0pX/<stamp>/, eclipse_corona/<block>/
stage2_twopass_reftol0pX/, pooled_fit/twopass_reftol0pX/, reference_tolerance.csv and
reference_tolerance_geometry.csv.
"""
import glob, json, os, subprocess
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
REC = r"D:/MEE2024 output/MEE_output/station1_record"
RECEN = os.path.join(REC, 'zenith_recentroid')
TOLDIR = os.path.join(REC, 'zenith_recentroid_tol')
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=760.0',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
       '--set', 'observation_temp=15.0', '--set', 'observation_pressure=760.0',
       '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
BLOCKS = (('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
          ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02'))
TOLS = (0.1, 0.2)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def tag(tol):
    return 'tol%s' % ('%.1f' % tol).replace('.', 'p')


def fields():
    out = []
    for d in sorted(glob.glob(os.path.join(RECEN, '*'))):
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
        res = glob.glob(os.path.join(d, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
        if z and res:
            j = json.load(open(res[0], encoding='utf-8'))
            out.append(dict(stamp=os.path.basename(d), zip=z[0], tmid=j['observation_time (UTC)'],
                            n03=j['#stars used'], rms03=j['final rms error (arcseconds)']))
    return out


FL = fields()
print('%d windowed re-centroided zenith fields (the record, tol 0.3)' % len(FL), flush=True)
rows = []
for tol in TOLS:
    refs = []
    print('\n=== reference rebuilt at tolerance %.1f" ===' % tol, flush=True)
    for f in FL:
        d2 = os.path.join(TOLDIR, tag(tol), f['stamp']); os.makedirs(d2, exist_ok=True)
        res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if not res:
            run([PY, '-m', 'mee2024.cli', 'distortion', f['zip'], '--order', 'quintic',
                 '--set', 'distortion_fit_tol=%.1f' % tol, '--set', 'max_star_mag_dist=15',
                 '--set', 'rough_match_threshhold=36', *SITE, '--set', 'observation_time=' + f['tmid'],
                 '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
            res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if not res:
            print('  %s: refit FAILED' % f['stamp'], flush=True); continue
        j = json.load(open(res[0], encoding='utf-8'))
        refs.append(res[0])
        rows.append(dict(tol=tol, stamp=f['stamp'], stars=j['#stars used'], rms=j['final rms error (arcseconds)'],
                         platescale=j['platescale (arcseconds/pixel)']))
        print('  %s  %4d stars  rms %.4f"  (record at 0.3: %4d stars, rms %.4f")'
              % (f['stamp'], j['#stars used'], j['final rms error (arcseconds)'], f['n03'], f['rms03']), flush=True)
    # the eclipse blocks against this reference
    sub = 'stage2_twopass_ref' + tag(tol)
    for bt, tm in BLOCKS:
        cz = glob.glob(os.path.join(REC, 'eclipse_corona', bt, 'centroid_data*.zip'))[0]
        d = os.path.join(REC, 'eclipse_corona', bt, sub); os.makedirs(d, exist_ok=True)
        if not glob.glob(os.path.join(d, '**', 'distortion_data*.zip'), recursive=True):
            rc = run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'quintic', '--fix-distortion', *refs,
                      '--set', 'distortion_fixed_coefficients=constant', '--set', 'distortion_free_scale=True',
                      '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
                      '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *MET,
                      '--set', 'observation_time=' + tm, '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            print('  eclipse %s against this reference: rc %d' % (bt, rc), flush=True)
    r = subprocess.run([PY, os.path.join(REPO, 'tools', 'matrix_station1', 's1_pooled_fit.py'), '--ref', 'twopass',
                        '--sub', sub, '--tag', 'ref' + tag(tol), '--boot', '400'], cwd=REPO, capture_output=True, text=True)
    keep = [l for l in r.stdout.splitlines() if 'bootstrap' in l or 'observations of' in l or l.strip().startswith('L =')]
    print('  pooled against the %.1f" reference:\n    ' % tol + '\n    '.join(keep), flush=True)

pd.DataFrame(rows).to_csv(os.path.join(REC, 'reference_tolerance.csv'), index=False)
print('\n=== summary: the reference at three tolerances, and L against each ===')
for tol in TOLS:
    a = pd.DataFrame(rows); a = a[a.tol == tol]
    js = json.load(open(os.path.join(REC, 'pooled_fit', 'twopass_ref' + tag(tol), 'pooled_summary.json')))
    print('  tol %.1f"  stars/field %4.0f  rms %.4f"  ->  L = %.3f +- %.3f (%d obs, %d stars)'
          % (tol, a.stars.mean(), a.rms.mean(), js['L'], js['sigma_bootstrap'], js['observations'], js['stars']))
js = json.load(open(os.path.join(REC, 'pooled_fit', 'twopass', 'pooled_summary.json')))
print('  tol 0.3"  stars/field %4.0f  rms %.4f"  ->  L = %.3f +- %.3f (%d obs, %d stars)   [the record]'
      % (np.mean([f['n03'] for f in FL]), np.mean([f['rms03'] for f in FL]), js['L'], js['sigma_bootstrap'], js['observations'], js['stars']))
print('->', os.path.join(REC, 'reference_tolerance.csv'))

# ---------------------------------------------------------------- the geometry: where the stars are, and what the model does
import contextlib, io, zipfile
from mee2024 import distortion_polynomial as dp
from mee2024.config import get_default_options
NX, NY, PS = 9576, 6388, 1.84847
opts = get_default_options(); opts['distortionOrder'] = 'quintic'; opts['distortion_fixed_coefficients'] = 'None'
half_diag = np.hypot(NX/2, NX/2*NY/NX)
edges = [0, 0.3, 0.5, 0.7, 0.85, 1.0]           # frame radius as a fraction of the half-diagonal
builds = {tol: sorted(glob.glob(os.path.join(TOLDIR, tag(tol), '*', '**', 'distortion_results.txt'), recursive=True)) for tol in TOLS}
builds[0.3] = sorted(glob.glob(os.path.join(RECEN, '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))


def model(files, n=41):
    """The averaged quintic of these reference files on an n x n grid, in mas."""
    with contextlib.redirect_stdout(io.StringIO()):           # _open_distortion_files prints the dicts
        cx, cy, _, _ = dp._open_distortion_files(dict(opts, distortion_reference_files=';'.join(files)))
    X, Y, DX, DY = dp.distortion_field(list(cx.values()), list(cy.values()), (NY, NX), opts, n=n)
    return X, Y, DX*PS*1000, DY*PS*1000


geo = []
print('\n=== (a) accepted stars per field by frame radius (last bin = the corners), and the residual rms there ===')
print('  tol   ' + ''.join('%12s' % ('%.2f-%.2f' % (a, b)) for a, b in zip(edges[:-1], edges[1:])) + '   total')
for tol, files in sorted(builds.items()):
    counts = np.zeros(len(edges)-1); rms2 = np.zeros(len(edges)-1); nf = 0
    for f in files:
        z = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(f))), 'distortion_data*.zip'))
        if not z:
            continue
        zf = zipfile.ZipFile(z[0])
        d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
        d.columns = [c.strip() for c in d.columns]; d = d[d['flag_is_outlier'] == False]
        r = np.hypot(d.px.values-NX/2, d.py.values-NY/2)/half_diag
        counts += np.histogram(r, bins=edges)[0]; nf += 1
        rms2 += np.histogram(r, bins=edges, weights=d['error(\")'].values**2)[0]
    counts /= max(nf, 1); rms = np.sqrt(rms2/np.maximum(counts*nf, 1))
    print('  %.1f   ' % tol + ''.join('%12.0f' % c for c in counts) + '   %5.0f' % counts.sum())
    print('        rms ' + ''.join('%12.3f' % v for v in rms))
    for (a_, b_), c, v in zip(zip(edges[:-1], edges[1:]), counts, rms):
        geo.append(dict(tol=tol, quantity='stars_per_field', r_lo=a_, r_hi=b_, value=c))
        geo.append(dict(tol=tol, quantity='rms_arcsec', r_lo=a_, r_hi=b_, value=v))

ref = {tol: model(files) for tol, files in builds.items()}
X, Y = ref[0.3][0], ref[0.3][1]
rg = np.hypot(X, Y)/half_diag; inner, corner = rg < 0.5, rg > 0.85
print('\n=== (b) field-to-field stability: rms over the 17 fields of the displacement each field alone predicts ===')
for tol, files in sorted(builds.items()):
    mods = np.array([np.stack(model([f])[2:]) for f in files])
    sd = np.sqrt(mods.var(axis=0, ddof=1).sum(axis=0))
    print('  tol %.1f   inner half %5.1f mas   corners %5.1f mas   worst point %5.1f mas' % (tol, np.median(sd[inner]), np.median(sd[corner]), sd.max()))
    geo += [dict(tol=tol, quantity='stability_inner_mas', value=np.median(sd[inner])),
            dict(tol=tol, quantity='stability_corner_mas', value=np.median(sd[corner]))]
print('\n=== (c) the averaged model rebuilt at 0.1 and 0.2, minus the record at 0.3 ===')
for tol in TOLS:
    dx, dy = ref[tol][2]-ref[0.3][2], ref[tol][3]-ref[0.3][3]
    dd = np.hypot(dx, dy); rad = (dx*X + dy*Y)/np.maximum(np.hypot(X, Y), 1)
    print('  %.1f minus 0.3:  inner half median %4.1f mas   corners median %5.1f mas   worst %5.1f mas   mean radial in the corners %+5.1f mas'
          % (tol, np.median(dd[inner]), np.median(dd[corner]), dd.max(), rad[corner].mean()))
    geo += [dict(tol=tol, quantity='model_diff_corner_mas', value=np.median(dd[corner])),
            dict(tol=tol, quantity='model_diff_corner_radial_mas', value=rad[corner].mean())]
pd.DataFrame(geo).to_csv(os.path.join(REC, 'reference_tolerance_geometry.csv'), index=False)
print('->', os.path.join(REC, 'reference_tolerance_geometry.csv'))
