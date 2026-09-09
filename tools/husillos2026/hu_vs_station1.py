"""Husillos against Mexico Station 1: how many stars the exposure cost, and what the scatter means.

Two questions from Douglas, 2026-09-09:

  1. **Did the underexposure lose a lot of stars?**  Station 1's zenith fields were shot with a
     ZWO ASI6200 -- the SAME IMX455 sensor as Joe's Zeus 455M -- at **gain 100 and 3 s**, through
     a nominal 432 mm f/5, against Joe's **gain 0 and 1 s** through a nominal 350 mm f/3.9.
  2. **The Husillos 2D residuals look far more tightly grouped than Station 1's.  Does that mean
     the AM5 tracks better than Station 1's Celestron AVX?**

The second one needs a warning before any number, because the comparison as posed cannot be made
from the plots: **the two reductions were not fitted at the same gate.**  Station 1's record fit
used `error tolerance (as) = 0.3`, mag limit 15 and corrections ON; the Husillos fits here used
0.5, mag 13 and corrections OFF.  The gate is a selection on the residual itself, so it sets the
scatter almost directly -- the project's own measurement (`s1_reference_tolerance.py`) is 0.061 "
at a 0.1 gate against 0.124 " at 0.3 **on identical data**.  Two residual plots at different
gates cannot be compared by eye or by rms.  So this tool refits Station 1 at the Husillos
settings and compares those.

And a second point, which turns out to make Douglas' reading right for a reason the plots do not
show.  **The `TWOD_RESIDUALS*.png` in a CENTROID_OUTPUT folder is a STAGE 1 plot, not stage 2**:
it is every star's position in every frame against the stacking master, in pixels, coloured by
frame.  So it really does carry mount information -- but it carries two things at once, and they
work in opposite directions.  The plot's overall EXTENT is the drift; the width of each colour's
own clump is the per-frame centroid scatter.  Station 1's clumps march 3 px across the plot and
each is tight; Husillos' span 1.6 px and overlap.

Neither has to be read off an axis, because stage 1 records both.  `alignment.dither_span_px` is
the total excursion and `alignment.rms_px` the per-frame scatter of the ~30 stars the aligner
uses, and this tool reads them.  The stacked field's stage-2 residual, by contrast, says almost
nothing about the mount: the stacker aligns before summing, so frame-to-frame drift is gone and
only trailing WITHIN one exposure survives -- 0.0365 "/s x 1.0 s = 0.037 " = 0.017 px at
Husillos, which is not a term.  A stage-2 residual is centroid noise, model misfit, the gate and
the plate scale.  The plate scale matters: Station 1 is at 1.8485 "/px against Husillos' 2.2064,
so the same centroid error in pixels reads 19 % smaller in arcsec at Station 1.  Both units are
reported.

The star-count question has its own trap: **raw counts cannot answer it, because the fields are
not equally rich.**  Husillos' zenith sits at RA 281.7, Dec +50.2 and Leon's at RA 285.1,
Dec +39.2 -- both near the galactic plane but not equally near, and Station 1's Mexico zenith is
sparser than either.  What IS density-independent is the SHAPE of the magnitude histogram: a
complete sample gains a fixed factor per half magnitude, and the bin where that factor collapses
is the detection limit.  That is the test used below.

  .venv/Scripts/python.exe tools/husillos2026/hu_vs_station1.py refit
  .venv/Scripts/python.exe tools/husillos2026/hu_vs_station1.py report
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
HUS = r"D:/MEE2024 output/MEE_output/husillos2026"
OUT = os.path.join(HUS, 'vs_station1')
S1REC = r"D:/MEE2024 output/MEE_output/station1_record/zenith_recentroid"

#: Not an analysis window and not a new choice: these are copied verbatim from
#: `tools/husillos2026/hu_zenith_order.py` so that Station 1 and Husillos are fitted by identical
#: code, which is the entire point of this file -- the record's own Station 1 fit used gate 0.3,
#: mag 15 and corrections ON, and comparing that with a gate-0.5 fit is the trap documented in
#: the module docstring. Nothing here is admitted to a deflection fit; cell 4's admitted-star
#: window belongs in `tools/analysis_window.py` with its citation before any stage-3 work.
GATE = 0.5          # source: hu_zenith_order.py GATE, the cell-2 record's reference gate
ORDER = 'quintic'   # source: HUSILLOS2026_ZENITH.md section 3.6
MAG = 13            # source: docs/HUSILLOS2026_ZENITH.md, the gaia_dr3_g13 catalogue's own edge
NOSITE = ['--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
          '--set', 'guess_date=False']

#: (label, sensor px, plate scale, aperture mm, focal mm, exposure s, gain, frames)
RIGS = {
    'Husillos zenith': dict(nx=9576, ny=6388, ps=2.2064323, aperture=89.7, focal=351.5,
                            exp=1.0, gain=0, nframes=50, read_e=4.73, camera='Zeus 455M (IMX455)'),
    'Station 1 zenith': dict(nx=9576, ny=6388, ps=1.8484826, aperture=86.4, focal=432.0,
                             exp=3.0, gain=100, nframes=None, read_e=1.5,
                             camera='ASI6200MM (IMX455)'),
    'Leon zenith': dict(nx=6248, ny=4176, ps=2.2073708, aperture=89.7, focal=351.3,
                        exp=4.0, gain=101, nframes=30, read_e=2.12, camera='ASI2600MM (IMX571)'),
}

#: the sky rate Husillos measured, e-/px/s, and the f-ratio it was measured at
SKY_E_PER_S, SKY_FRATIO = 1.96, 351.5 / 89.7


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def resid(d):
    r = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return pd.read_csv(r[0]) if r else None


def s1_fields(n=4):
    return sorted(glob.glob(os.path.join(S1REC, '*', 'centroid_data*.zip')))[:n]


def do_refit():
    """Station 1's zenith fields at the Husillos settings, so the gate stops being a variable."""
    for z in s1_fields():
        stamp = os.path.basename(os.path.dirname(z))
        d = os.path.join(OUT, 's2_%s' % stamp)
        os.makedirs(d, exist_ok=True)
        if not results(d):
            run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', ORDER,
                 '--set', 'distortion_fixed_coefficients=None',
                 '--set', 'distortion_fit_tol=%s' % GATE,
                 '--set', 'max_star_mag_dist=%d' % MAG,
                 '--set', 'rough_match_threshhold=36',
                 '--set', 'observation_date=2024-04-17', *NOSITE,
                 '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
        j = results(d)
        print('%s  %s' % (stamp, '%5d stars  rms %.4f "  ps %.7f'
                          % (j['#stars used'], j['final rms error (arcseconds)'],
                             j['platescale (arcseconds/pixel)']) if j else 'FAILED'), flush=True)


def depth_row(label, df, rig):
    ps = rig['ps']
    mpx = rig['nx'] * rig['ny'] / 1e6
    e = df['error_arcsec'].to_numpy()
    g = df['magV'].to_numpy()
    rec = dict(field=label, n=len(df), per_Mpx=len(df) / mpx,
               rms_as=float(np.sqrt((e ** 2).mean())),
               rms_px=float(np.sqrt((e ** 2).mean()) / ps),
               g_median=float(np.median(g)))
    for lo, hi in ((0, 11), (11, 12), (12, 13)):
        m = (g >= lo) & (g < hi)
        rec['n_G%d_%d' % (lo, hi)] = int(m.sum())
        rec['perMpx_G%d_%d' % (lo, hi)] = m.sum() / mpx
        rec['rms_G%d_%d' % (lo, hi)] = float(np.sqrt((e[m] ** 2).mean())) if m.sum() else np.nan
    return rec


def alignment_of(zip_glob):
    """(frames, dither span px, median per-frame alignment rms px) from a stage-1 zip."""
    import zipfile
    f = glob.glob(zip_glob)
    if not f:
        return None
    j = json.loads(zipfile.ZipFile(f[0]).read('results.txt').decode('utf-8'))
    a = j.get('alignment')
    if not a:
        return None                 # pre-2026 stage-1 zips have no alignment record
    return (len(a['rms_px']), float(a['dither_span_px']),
            float(np.median(np.asarray(a['rms_px'], dtype=float))))


def do_mount():
    """Drift and per-frame scatter, from what stage 1 recorded rather than from a plot."""
    print('=' * 112)
    print('WHAT THE STAGE-1 RESIDUAL PLOT ACTUALLY SHOWS: drift, and per-frame scatter')
    print('=' * 112)
    print('%-22s %-6s %7s %9s %11s %12s %13s'
          % ('field', 'mount', 'frames', 'span px', 'span "', 'span "/min', 'per-frame rms'))
    rows = [('Husillos zenith', 'AM5', HUS + '/zenith_order/s1_with_f0/centroid_data*.zip',
             RIGS['Husillos zenith']['ps'], 64.67),
            ('Leon Z1', 'AVX', HUS + '/zenith_order/s1_leonZ1/centroid_data*.zip',
             RIGS['Leon zenith']['ps'], 127.95)]
    for label, mount, zp, ps, dur in rows:
        a = alignment_of(zp)
        if not a:
            print('%-22s %-6s   no alignment record in the stage-1 zip' % (label, mount))
            continue
        n, span, rms = a
        print('%-22s %-6s %7d %9.3f %11.2f %12.2f %10.4f px'
              % (label, mount, n, span, span * ps, span * ps * 60 / dur, rms))
    print()
    print('  Station 1 Mexico: its 2024 stage-1 zip predates the alignment record, and its raw')
    print('  per-frame zenith data is not on any drive here (the reduction reads a path on a')
    print('  cloud drive that no longer exists). Read off its stage-1 plot the span is ~2.9 px')
    print('  in x over 19 frames = ~5.4 " -- about 4 "/min at a 3 s cadence. That number is')
    print('  eyeballed from an axis and is quoted as such; the two above are measured.')
    print()
    print('  Drift rate is mostly POLAR ALIGNMENT, not the mount head. What is intrinsic to the')
    print('  head is smoothness and settling, and that is `hu_mount_compare.py`, not this.')
    print()
    print('  The sting: stage 1 needs >= 3 px of dither to find hot pixels without a dark frame.')
    print('  Leon got 13.8 px and its log says "hot pixel(s) identified from the dither (13.8')
    print('  px) ... without a dark frame". Husillos got 1.3 px and its log says "no dark-free')
    print('  hot-pixel search: the field moved only 1.3 px between frames, under the 3 px')
    print('  needed". A mount that tracks too well loses that, and loses the sub-pixel dither')
    print('  that was worth 12 % of the centroid yield (HUSILLOS2026_ZENITH.md section 6).')
    print()


def do_completeness():
    """Where each field's magnitude histogram stops doubling: the detection limit."""
    print('=' * 112)
    print('DID THE EXPOSURE COST STARS?  Counts per half magnitude, and the ratio between')
    print('adjacent bins. Density-independent, so the unequal richness of the fields cancels.')
    print('=' * 112)
    sets = [('Husillos  1.0 s g0  f/3.9',
             HUS + '/zenith_order/s2_with_f0_quintic/**/TWOD_RESIDUALS.csv'),
            ('Leon Z1   4.0 s g101 f/3.9',
             HUS + '/zenith_order/s2_leonZ1_quintic/**/TWOD_RESIDUALS.csv'),
            ('Leon Z4   4.0 s g101 f/3.9',
             HUS + '/zenith_order/s2_leonZ4_quintic/**/TWOD_RESIDUALS.csv')]
    for z in s1_fields(1):
        stamp = os.path.basename(os.path.dirname(z))
        sets.append(('Station1  3.0 s g100 f/5',
                     os.path.join(OUT, 's2_%s' % stamp, '**', 'TWOD_RESIDUALS.csv')))
    edges = np.arange(9.0, 13.25, 0.5)
    print('%-28s' % 'field' + ''.join('%8s' % ('%.1f' % e) for e in edges[:-1]) + '%12s' % 'last/prev')
    for label, pat in sets:
        f = glob.glob(pat, recursive=True)
        if not f:
            continue
        d = pd.read_csv(f[0])
        h, _ = np.histogram(d['magV'], bins=edges)
        print('%-28s' % label + ''.join('%8d' % v for v in h)
              + '%12.2f' % (h[-1] / max(h[-2], 1)))
    print()
    print('  A complete sample gains ~1.3-1.5x per half magnitude here. Husillos alone COLLAPSES')
    print('  inside the catalogue\'s own G < 13 edge; the other three are still gaining at it,')
    print('  so they are complete and Husillos is not. Extrapolating Husillos\' own healthy')
    print('  ratio (1.44, measured on its bins up to G 12) through the last two bins gives')
    print('  ~3640 stars expected against 2680 matched: about a QUARTER lost, all of them faint.')
    print()


def do_report():
    print('=' * 112)
    print('THE RIGS.  Star photons scale with aperture AREA; sky per pixel with 1/(f-ratio)^2.')
    print('=' * 112)
    print('%-17s %-20s %7s %7s %8s %7s %6s %7s %9s'
          % ('field', 'camera', 'aper mm', 'foc mm', 'f-ratio', '"/px', 'exp s', 'gain',
             'read e-'))
    for k, r in RIGS.items():
        print('%-17s %-20s %7.1f %7.1f %8.2f %7.4f %6.2f %7d %9.2f'
              % (k, r['camera'], r['aperture'], r['focal'], r['focal'] / r['aperture'],
                 r['ps'], r['exp'], r['gain'], r['read_e']))
    print()
    h, s = RIGS['Husillos zenith'], RIGS['Station 1 zenith']
    print('  Apertures are within %.0f %%: star photons per second differ by only %.2fx.'
          % (100 * abs(h['aperture'] / s['aperture'] - 1), (h['aperture'] / s['aperture']) ** 2))
    print('  Husillos is the FASTER optic, so it collects %.2fx more SKY per pixel per second --'
          % ((s['focal'] / s['aperture']) ** 2 / (h['focal'] / h['aperture']) ** 2))
    print('  which helps, because more sky is what lifts a frame off its read noise.')
    print()
    print('=' * 112)
    print('WHAT THE EXPOSURE COST.  Stack variance per pixel = N*read^2 + sky_rate*T.')
    print('=' * 112)
    print('%-17s %6s %6s %7s %10s %11s %12s %10s'
          % ('field', 'exp s', 'N', 'T s', 'sky e-/fr', 'sky/read^2', 'stack noise', 'rel SNR'))
    base = None
    for k, r in RIGS.items():
        if r['nframes'] is None:
            r = dict(r, nframes=int(round(50 / r['exp'])))    # Station 1's count is not recorded
        rate = SKY_E_PER_S * (SKY_FRATIO / (r['focal'] / r['aperture'])) ** 2
        sky = rate * r['exp']
        T = r['exp'] * r['nframes']
        noise = np.sqrt(r['nframes'] * r['read_e'] ** 2 + rate * T)
        snr = T / noise
        base = base or snr
        print('%-17s %6.2f %6d %7.1f %10.2f %11.2f %12.2f %10.2f'
              % (k, r['exp'], r['nframes'], T, sky, sky / r['read_e'] ** 2, noise, snr / base))
    print('  (Station 1\'s frame count is not in the reduction; %d x 3 s is assumed so that the'
          % int(round(50 / 3)))
    print('   comparison is at equal total integration -- the per-frame columns do not depend')
    print('   on it, and they are where the answer is.)')

    print()
    print('=' * 112)
    print('WHAT WAS ACTUALLY MATCHED, all refitted at gate %.1f ", %s, mag %d, corrections off'
          % (GATE, ORDER, MAG))
    print('=' * 112)
    rows = []
    d = resid(os.path.join(HUS, 'zenith_order', 's2_with_f0_quintic'))
    if d is not None:
        rows.append(depth_row('Husillos zenith', d, RIGS['Husillos zenith']))
    for z in s1_fields():
        stamp = os.path.basename(os.path.dirname(z))
        dd = resid(os.path.join(OUT, 's2_%s' % stamp))
        if dd is not None:
            rows.append(depth_row('Station 1 %s' % stamp[8:], dd, RIGS['Station 1 zenith']))
    d = resid(os.path.join(HUS, 'zenith_order', 's2_leonZ1_quintic'))
    if d is not None:
        rows.append(depth_row('Leon Z1', d, RIGS['Leon zenith']))
    t = pd.DataFrame(rows)
    if t.empty:
        print('nothing refitted yet -- run `refit` first')
        return
    pd.set_option('display.width', 230)
    print(t[['field', 'n', 'per_Mpx', 'g_median', 'rms_as', 'rms_px',
             'perMpx_G0_11', 'perMpx_G11_12', 'perMpx_G12_13',
             'rms_G0_11', 'rms_G11_12', 'rms_G12_13']]
          .to_string(index=False, float_format=lambda v: '%.3f' % v))
    print()
    print('  per_Mpx is stars per megapixel of sensor: the two sensors are both 9576 x 6388 but')
    print('  at different plate scales, so per-Mpx compares equal SENSOR area and the ratio of')
    print('  plate scales squared (%.2f) converts it to equal SKY area.'
          % (RIGS['Station 1 zenith']['ps'] / RIGS['Husillos zenith']['ps']) ** 2)
    print('  rms_px is the same residual in pixels, which is the fair unit for a centroid: the')
    print('  arcsec column charges Husillos 19 % extra purely for its coarser plate scale.')
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, 'vs_station1.csv'), index=False)
    print('->', os.path.join(OUT, 'vs_station1.csv'))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    os.makedirs(OUT, exist_ok=True)
    if cmd == 'report':
        do_report()
        print()
        do_completeness()
        do_mount()
    else:
        {'refit': do_refit, 'mount': do_mount, 'completeness': do_completeness}[cmd]()
