"""Cell 2: re-centroid all seventeen 2024 zenith STACKS in the windowed convention.

Douglas, 2026-09-03: "I just realised I have the stacked zenith fields. That is probably
almost as good as having the original zenith files."

It is as good, for this purpose, and that is measured rather than assumed. Every one of the
seventeen `D:\\MEE2024 output\\Station 1\\zenith fields\\CENTROID_OUTPUT*` folders kept its
full-frame `STACKED*.fit`, and stage 1 will take one as a single light frame and re-run the
centroid step on it. Only the *centroiding* is redone -- the alignment, the frame selection
and the stack itself stay exactly as they were in 2024 -- which is precisely the axis under
test.

Validated on field 1 before the other sixteen were attempted. Re-centroiding the 2024 stack
windowed against re-stacking its nineteen raw frames windowed (`s1_zenith_raw_ab.py`):

    stars     2107  vs  2104
    fit rms   0.1269"  vs  0.1258"
    scale     1.8484826  vs  1.8484811  "/px      (0.8 ppm apart)
    18.3 bias -1.0 +3.9 +2.7 +0.9 -10.3  vs  -0.8 +4.4 +2.8 +0.9 -10.2  mas/mag

while the 2024 moment reduction of that same stacked image gives -1.2 +11.0 +24.7 +35.2
+10.8. So the stacking step contributes nothing to the convention difference and the
estimator contributes all of it, and the fourteen zenith blocks whose raw frames are missing
are not missing anything that matters.

That turns a two- or three-field windowed reference into the full seventeen. Each field is
re-centroided windowed + annular under the matrix's held-constant stage-1 set, then given a
free quintic with corrections on at the Station 1 site and the block's true mid-time (taken
from the 2024 archive's `source_files`, since the fits themselves carry a placeholder 05:45).

Writes station1_record/zenith_recentroid/<stamp>/ and a summary CSV. The seventeen
`distortion_results.txt` it leaves behind are the windowed reference of record.
"""
import glob, json, os, re, subprocess, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
F = r"D:/MEE2024 output/Station 1/zenith fields"
OUT = r"D:/MEE2024 output/MEE_output/station1_record/zenith_recentroid"
NX, NY = 9576, 6388
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
      '--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular']
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
        '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
        '--set', 'observation_temp=10.0', '--set', 'observation_pressure=760.0',
        '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']
RB = [(0, 1500), (1500, 2500), (2500, 3500), (3500, 4500), (4500, 6000)]
os.makedirs(OUT, exist_ok=True)


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def fields():
    """The seventeen stacks, with each block's true mid-time from the 2024 archive."""
    out = []
    for z in sorted(glob.glob(os.path.join(F, 'centroid_data*.zip'))):
        stamp = os.path.basename(z)[13:27]
        stacked = glob.glob(os.path.join(F, 'CENTROID_OUTPUT%s' % stamp, 'STACKED*.fit'))
        if not stacked:
            continue
        r = json.load(zipfile.ZipFile(z).open('results.txt'))
        src = r.get('source_files')
        if isinstance(src, str):
            src = json.loads(src.replace("'", '"'))
        m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})Z', src[0])
        # the block's start plus about half its 70 s duration
        secs = int(m.group(2))*3600 + int(m.group(3))*60 + int(m.group(4)) + 35
        out.append(dict(stamp=stamp, stacked=stacked[0], nframes=len(src),
                        tmid='%02d:%02d:%02d' % (secs//3600, (secs % 3600)//60, secs % 60)))
    return out


def matched_table(resfile):
    z = glob.glob(os.path.join(os.path.dirname(os.path.dirname(resfile)), 'distortion_data*.zip')) or \
        glob.glob(os.path.join(os.path.dirname(resfile), '..', '..', 'distortion_data*.zip'))
    zf = zipfile.ZipFile(z[0])
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    d['dx'] = (d['RA(obs)']-d['RA(catalog)'])*np.cos(np.radians(d['DEC(catalog)']))*3600
    d['dy'] = (d['DEC(obs)']-d['DEC(catalog)'])*3600
    d['r'] = np.hypot(d.px-NX/2, d.py-NY/2)
    ux, uy = (d.px-NX/2)/d.r, (d.py-NY/2)/d.r
    d['rad'] = (d.dx*ux+d.dy*uy)*1000
    return d


def slopes(d):
    out = []
    for lo, hi in RB:
        k = (d.r >= lo) & (d.r < hi)
        out.append(np.polyfit(d.magV[k], d.rad[k], 1)[0] if k.sum() > 20 else np.nan)
    return out


FL = fields()
print('%d stacked zenith fields found\n' % len(FL), flush=True)
rows = []
for f in FL:
    d1 = os.path.join(OUT, f['stamp'])
    os.makedirs(d1, exist_ok=True)
    z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
    if not z:
        run([PY, '-m', 'mee2024.cli', 'stack', f['stacked'], *S1,
             '--no-scan', '--no-display', '--quiet', '-o', d1], os.path.join(d1, 'stage1.log'))
        z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
    if not z:
        print('%s: STAGE 1 FAILED' % f['stamp'], flush=True); continue
    d2 = os.path.join(d1, 'stage2_free')
    os.makedirs(d2, exist_ok=True)
    res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not res:
        run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'quintic',
             '--set', 'distortion_fit_tol=0.3', '--set', 'max_star_mag_dist=15',
             '--set', 'rough_match_threshhold=36', *SITE,
             '--set', 'observation_time=' + f['tmid'], '--no-display', '--quiet', '-o', d2],
            os.path.join(d2, 'stage2.log'))
        res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not res:
        print('%s: stage 2 FAILED' % f['stamp'], flush=True); continue
    j = json.load(open(res[0], encoding='utf-8'))
    sl = slopes(matched_table(res[0]))
    old = glob.glob(os.path.join(r"D:/MEE2024 output/Station 1/zenith calibrations",
                                 '*%s*' % f['stamp'], '**', 'distortion_results.txt'), recursive=True)
    j24 = json.load(open(old[0], encoding='utf-8')) if old else {}
    rows.append(dict(stamp=f['stamp'], tmid=f['tmid'], n=j['#stars used'],
                     rms=j['final rms error (arcseconds)'], ps=j['platescale (arcseconds/pixel)'],
                     ps24=j24.get('platescale (arcseconds/pixel)', np.nan),
                     n24=j24.get('#stars used', np.nan), rms24=j24.get('final rms error (arcseconds)', np.nan),
                     **{('s%d' % i): s for i, s in enumerate(sl)}))
    print('%s %s  windowed %4d stars rms %.4f" ps %.7f | 2024 moments %4d stars rms %.4f" ps %.7f | bias %s'
          % (f['stamp'], f['tmid'], j['#stars used'], j['final rms error (arcseconds)'],
             j['platescale (arcseconds/pixel)'], j24.get('#stars used', -1),
             j24.get('final rms error (arcseconds)', np.nan), j24.get('platescale (arcseconds/pixel)', np.nan),
             ' '.join('%+5.1f' % s for s in sl)), flush=True)

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
if len(T):
    print('\n=== seventeen fields, windowed re-centroiding of the 2024 stacks ===')
    print('  stars   %d-%d (median %d) against the 2024 moments\' %d-%d'
          % (T.n.min(), T.n.max(), int(T.n.median()), int(T.n24.min()), int(T.n24.max())))
    print('  fit rms %.4f-%.4f " (median %.4f) against the 2024 moments\' %.4f-%.4f'
          % (T.rms.min(), T.rms.max(), T.rms.median(), T.rms24.min(), T.rms24.max()))
    m = T.ps.mean()
    print('  plate scale mean %.7f "/px, rms %.1f ppm about it (2024: mean %.7f, rms %.1f ppm)'
          % (m, 1e6*T.ps.std(ddof=1)/m, T.ps24.mean(), 1e6*T.ps24.std(ddof=1)/T.ps24.mean()))
    print('  windowed minus 2024 scale, per field: %+.1f to %+.1f ppm'
          % (1e6*((T.ps-T.ps24)/T.ps24).min(), 1e6*((T.ps-T.ps24)/T.ps24).max()))
    print('\n  18.3 radial bias by field radius, mas/mag, mean over the seventeen:')
    print('    %-12s' % 'windowed' + ''.join('%9.1f' % T['s%d' % i].mean() for i in range(5)))
    print('    (the 2024 moment fits of the same stacks: +3 +12 +22 +31 +10)')
    print('\n  the seventeen distortion_results.txt under %s' % OUT)
    print('  are the windowed reference of record for cell 2.')
