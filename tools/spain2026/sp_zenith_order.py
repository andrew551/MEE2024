"""Cell 4's zenith fields: the plate scale, and whether a cubic is enough on this full frame.

Two questions, one set of runs, Spain 2026 (Douglas, 2026-09-09):

  1. **Is the optical train the same as Leon 2026's?**  Both are an FRA500 with the 0.7x
     reducer and 3.76 um pixels, so if the focal length is the same the plate scale must be
     Leon's 2.2054043 "/px (docs/CAL_PILEO_STEP2.md, the canonical CAL_piLeo value).  This
     measures it rather than assuming it.
  2. **Cubic or quintic?**  Leon's ASI2600 reaches 8290" of field radius; this IMX455 reaches
     12690", and a cubic term grows as r^3 and a quintic as r^5, so an order that was
     sufficient there is not automatically sufficient here.  Cell 2 (Mexico Station 1) needed
     the quintic on the same 9576x6388 sensor at 1.848 "/px -- but a different telescope.

The test is the one cell 2 used (`tools/matrix_station1/s1_septic_test.py`): a higher order
always fits better IN sample, so the number that decides it is not the rms.  Three things are
read instead, all from `TWOD_RESIDUALS.csv`:

  * **the mean RADIAL residual per annulus.**  Centroid noise has no preferred direction, so
    its radial mean is zero within its standard error; an order that is too low leaves a
    coherent radial ramp, which is exactly the signature that biases a deflection.
  * **the size of the term being added**, per star, as (residual at order N) - (residual at
    order N+2) on the stars both fits kept.  If that difference is far below the residual
    scatter at the corner, the higher order is fitting noise.
  * **out of sample**: the capture is split into two halves, each stacked and fitted on its
    own, and each half's model is frozen into the other with only the constant free.  A term
    that is real transfers; a term that is noise does not.

No site is known for this data yet (`spain2026_prompt.md`), so every run here has
`enable_corrections=False` -- which is also what the Bruns night estimator does
(`tools/matrix_bruns/b17_night_estimator.py:117`).  At the zenith the refraction a correction
would apply is a radial-linear term absorbed by the plate scale plus a tan^3 z term worth
1e-5 " at this field radius, so the order question does not wait on the site.  The absolute
plate scale does, at the ~0.1 % level, and that is stated with it.

  .venv/Scripts/python.exe tools/spain2026/sp_zenith_order.py stage1
  .venv/Scripts/python.exe tools/spain2026/sp_zenith_order.py stage2
  .venv/Scripts/python.exe tools/spain2026/sp_zenith_order.py transfer
  .venv/Scripts/python.exe tools/spain2026/sp_zenith_order.py report
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
OUT = r"D:/MEE2024 output/MEE_output/spain2026/zenith_order"
G = r"G:/Joe Izen Spain 2026"

FULL = os.path.join(G, "2026-08-13", "zenith", "00_00_21.ser")     # 50 x 9576x6388, 1.0 s
ROI = os.path.join(G, "2026-08-12", "zenith", "23_24_56.ser")      # 16 x 1280x1024, 1.0 s

#: Leon 2026's canonical CAL_piLeo plate scale, same telescope and reducer, 3.76 um pixels.
LEON_PS = 2.2054043
PIXEL_UM = 3.76

# The zenith stage-1 preset, spelled out as every other matrix tool spells it
# (mee2024/field_presets.py; tools/matrix_station1/s1_zenith_recentroid.py:42).
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
      '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
      '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=False',
      '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
      '--set', 'centroid_refine_window=True', '--set', 'background_subtraction_mode=annular']

# No site: see the module docstring.
NOSITE = ['--set', 'enable_corrections=False', '--set', 'enable_corrections_ref=False',
          '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False']

GATE = 0.5
ORDERS = ('cubic', 'quintic', 'septic')

# Leon 2026 is the SAME ECLIPSE from a station ~an hour up the road (Joe's site is "south of
# Leon"), and its zenith fields were shot 48 minutes after Joe's, on the same night, through the
# same FRA500 + 0.7x on 3.76 um pixels -- an ASI2600 rather than an IMX455, so a 6248x4176 frame
# against 9576x6388.  Reduced here at identical settings it is the control for both questions:
# the same focal length reads as the same plate scale, and the same optic on a 1.53x smaller
# field radius says how much of the order question is the sensor rather than the telescope.
LEON_Z = r"G:/Leon Aug 2026/2026-08-12/Zenith"

#: frame 0 is dropped everywhere (F23, and this capture's own frame 0 is no exception).
#: A SER entry is (path, first, last); a FITS entry is a glob.
STACKS = {
    'full49':  (FULL, 1, 49),      # the record candidate: 49 frames
    'halfA':   (FULL, 1, 24),
    'halfB':   (FULL, 25, 49),
    'with_f0': (FULL, 0, 49),      # only to price frame 0
    'drop_last': (FULL, 0, 48),   # 49 frames again, but keeping frame 0: the control for
                                  # full49, which is the same count without it
    'roi':     (ROI, 0, 15),
    # A SECOND Spain full-frame night field at a DIFFERENT POINTING, which is what makes
    # an out-of-sample order test mean something. Two halves of one capture share their
    # stars, so any model that fits this field's own per-star quirks -- a catalogue
    # position error, a blend -- transfers between them perfectly and the test passes a
    # term that is not distortion at all. A different pointing has a different star set.
    'night11': (os.path.join(G, '2026-08-12/Capture/00_54_32.ser'), 0, 99),
    # Which frame is the STACKING MASTER, isolated. mee2024/stacker_implementation.py
    # aligns every frame against files[0], so a trim changes the master as well as the
    # depth. full49 (master frame 1) found 2950 initial centroids and failed to
    # plate-solve; drop_last (master frame 0, the same 49 frames' worth of depth) found
    # 3221 and solved in 0.9 s. These two say whether that is frame 0 helping or frame 1
    # being a poor master: master_f2 starts at 2, master_f1 keeps frame 1 as master with
    # drop_last's depth.
    'master_f2': (FULL, 2, 49),
    'master_f1': (FULL, 1, 48),
    'leonZ1':  os.path.join(LEON_Z, 'Z1_base', '*', '*.fits'),
    'leonZ4':  os.path.join(LEON_Z, 'Z4_top_right', '*', '*.fits'),
}


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def czip(name):
    z = glob.glob(os.path.join(OUT, 's1_' + name, 'centroid_data*.zip'))
    return z[0] if z else None


def results(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def resid(d):
    r = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return pd.read_csv(r[0]) if r else None


def stack_args(spec):
    """The `stack` arguments for one entry: a SER container plus a trim, or a FITS glob."""
    if isinstance(spec, tuple):
        path, first, last = spec
        # the container goes in whole and `--frames` trims it: `resolve_input_files` expands a
        # bare .ser into one reference per frame, but a `path.ser#N` typed on the command line
        # is globbed and matches no file, so passing frame references directly cannot work
        return [path, '--frames', '%d-%d' % (first, last)],             'frames %d-%d of %s' % (first, last, os.path.basename(path))
    return [spec], '%d FITS from %s' % (len(glob.glob(spec)), os.path.basename(os.path.dirname(
        os.path.dirname(spec))))


def do_stage1(only=None):
    for name, spec in STACKS.items():
        if only and name != only:
            continue
        d = os.path.join(OUT, 's1_' + name)
        os.makedirs(d, exist_ok=True)
        if czip(name):
            print('%-8s stage 1 already done' % name, flush=True)
            continue
        inputs, what = stack_args(spec)
        print('%-8s stage 1: %s' % (name, what), flush=True)
        rc = run([PY, '-m', 'mee2024.cli', 'stack', *inputs,
                  *S1, '--no-scan', '--no-display', '--quiet', '-o', d],
                 os.path.join(d, 'stage1.log'))
        z = czip(name)
        if not z:
            print('  FAILED (rc %d), see %s' % (rc, os.path.join(d, 'stage1.log')), flush=True)
            continue
        import zipfile
        j = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))
        print('  %d centroids' % j.get('n_centroids', -1), flush=True)


def do_stage2():
    for name in STACKS:
        z = czip(name)
        if not z:
            print('%-8s no stage-1 output' % name, flush=True)
            continue
        for order in ORDERS:
            d = os.path.join(OUT, 's2_%s_%s' % (name, order))
            os.makedirs(d, exist_ok=True)
            if not results(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', order,
                     '--set', 'distortion_fixed_coefficients=None',
                     '--set', 'distortion_fit_tol=%s' % GATE,
                     '--set', 'max_star_mag_dist=13',
                     '--set', 'rough_match_threshhold=36', *NOSITE,
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            j = results(d)
            print('%-8s %-8s %s' % (name, order,
                                    '%5d stars  rms %.4f "  plate scale %.7f "/px'
                                    % (j['#stars used'], j['final rms error (arcseconds)'],
                                       j['platescale (arcseconds/pixel)']) if j else 'FAILED'),
                  flush=True)


#: Which fits are frozen into which. halfA/halfB share their stars -- two halves of one
#: capture -- so a model that fits this field's own per-star quirks transfers between them and
#: the test passes it. leonZ1 -> leonZ4 does NOT: those are two pointings 6.4 deg apart on the
#: same night through the same optic, so only something belonging to the TELESCOPE transfers.
#: That pair is the honest out-of-sample test; the halves are reported beside it as the
#: within-field control, and the gap between them is the point.
PAIRS = (('halfA', 'halfB'), ('halfB', 'halfA'),
         ('leonZ1', 'leonZ4'), ('leonZ4', 'leonZ1'))


def do_transfer():
    """Each half's own model frozen into the other, constant only free: does the term transfer?"""
    for order in ORDERS:
        for src, dst in PAIRS:
            ref = glob.glob(os.path.join(OUT, 's2_%s_%s' % (src, order), '**',
                                         'distortion_results.txt'), recursive=True)
            z = czip(dst)
            if not ref or not z:
                print('%-6s %s->%s: missing input' % (order, src, dst), flush=True)
                continue
            d = os.path.join(OUT, 'xfer_%s_%s_to_%s' % (order, src, dst))
            os.makedirs(d, exist_ok=True)
            if not results(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', order,
                     '--fix-distortion', ref[0],
                     '--set', 'distortion_fixed_coefficients=constant',
                     '--set', 'distortion_fit_tol=%s' % GATE,
                     '--set', 'max_star_mag_dist=13',
                     '--set', 'rough_match_threshhold=36', *NOSITE,
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            j = results(d)
            print('%-8s %s -> %s : %s' % (order, src, dst,
                                          '%5d stars  rms %.4f "  ps %.7f  (fixed order %s)'
                                          % (j['#stars used'], j['final rms error (arcseconds)'],
                                             j['platescale (arcseconds/pixel)'],
                                             j.get('distortion_fixed_coefficients', '?'))
                                          if j else 'FAILED'), flush=True)


def radial_table(df, ps, nbin=6):
    """Signed mean RADIAL residual per annulus, with a TANGENTIAL control beside it.

    Centroid noise has no preferred direction, so the mean radial component is zero within
    its standard error; a missing higher radial order leaves a coherent ramp.

    The tangential column is not decoration.  On a rectangular sensor the outer annuli
    contain only the corners, so ANY anisotropic field -- uncorrected refraction is one, worth
    k*tan^2(z) = 5.6 ppm at this field's ~8 deg zenith distance, or about 0.07 " at the corner
    -- fails to average away there and reads as a radial signal.  A term that is really a
    missing radial order shows in the radial column and not the tangential one; a term that
    shows in both is not the polynomial order.
    """
    r = df['radius_px'].to_numpy()
    ux, uy = df['px'].to_numpy(), df['py'].to_numpy()
    vx, vy = ux - ux.mean(), uy - uy.mean()
    n = np.hypot(vx, vy)
    n[n == 0] = 1.0
    dx, dy = df['dx_arcsec'].to_numpy(), df['dy_arcsec'].to_numpy()
    rad = (dx * vx + dy * vy) / n
    tan = (-dx * vy + dy * vx) / n
    edges = np.linspace(0, r.max() * 1.0001, nbin + 1)
    rows = []
    for i in range(nbin):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() < 3:
            continue
        k = int(m.sum())
        rows.append(dict(r_lo_as=edges[i] * ps, r_hi_as=edges[i + 1] * ps, n=k,
                         rms_as=float(np.sqrt((df['error_arcsec'].to_numpy()[m] ** 2).mean())),
                         radial_mean_as=float(rad[m].mean()),
                         radial_se_as=float(rad[m].std(ddof=1) / np.sqrt(k)),
                         tang_mean_as=float(tan[m].mean()),
                         tang_se_as=float(tan[m].std(ddof=1) / np.sqrt(k))))
    return pd.DataFrame(rows)


def do_report():
    print('=' * 100)
    print('SPAIN 2026 ZENITH FIELDS -- free fits, no site corrections, gate %.1f "' % GATE)
    print('=' * 100)
    rows = []
    for name in STACKS:
        for order in ORDERS:
            j = results(os.path.join(OUT, 's2_%s_%s' % (name, order)))
            if not j:
                continue
            rows.append(dict(stack=name, order=order, n=j['#stars used'],
                             rms_as=j['final rms error (arcseconds)'],
                             ps_as_px=j['platescale (arcseconds/pixel)'],
                             ra=j.get('RA'), dec=j.get('DEC'), roll=j.get('roll')))
    t = pd.DataFrame(rows)
    if t.empty:
        print('no stage-2 results yet')
        return
    t['focal_mm'] = 206.264806 * PIXEL_UM / t['ps_as_px']
    t['vs_leon_ppm'] = (t['ps_as_px'] / LEON_PS - 1) * 1e6
    print(t.to_string(index=False, float_format=lambda v: '%.6f' % v))
    print()
    print('Leon 2026 CAL_piLeo canonical: %.7f "/px -> %.3f mm at %.2f um pixels'
          % (LEON_PS, 206.264806 * PIXEL_UM / LEON_PS, PIXEL_UM))

    for name in STACKS:
        for order in ORDERS:
            d = os.path.join(OUT, 's2_%s_%s' % (name, order))
            df, j = resid(d), results(d)
            if df is None or j is None:
                continue
            ps = j['platescale (arcseconds/pixel)']
            print()
            print('--- %s / %s : residual by annulus (radius from the field centre) ---'
                  % (name, order))
            tb = radial_table(df, ps)
            tb['rad_sig'] = tb['radial_mean_as'] / tb['radial_se_as']
            tb['tan_sig'] = tb['tang_mean_as'] / tb['tang_se_as']
            print(tb.to_string(index=False, float_format=lambda v: '%.4f' % v))

    # what the higher order actually adds, star by star
    for name in STACKS:
        for lo, hi in (('cubic', 'quintic'), ('quintic', 'septic')):
            a, b = resid(os.path.join(OUT, 's2_%s_%s' % (name, lo))), \
                   resid(os.path.join(OUT, 's2_%s_%s' % (name, hi)))
            if a is None or b is None:
                continue
            m = a.merge(b, on='ID', suffixes=('_lo', '_hi'))
            if m.empty:
                continue
            d = np.hypot(m['dx_arcsec_lo'] - m['dx_arcsec_hi'],
                         m['dy_arcsec_lo'] - m['dy_arcsec_hi'])
            r = m['radius_px_lo'] * results(os.path.join(OUT, 's2_%s_%s' % (name, lo)))[
                'platescale (arcseconds/pixel)']
            outer = r > np.percentile(r, 80)
            print()
            print('--- %s : what %s adds over %s, on the %d stars both kept ---'
                  % (name, hi, lo, len(m)))
            print('    model difference: rms %.4f ", median %.4f ", max %.4f "; '
                  'outer 20%% (r > %.0f "): rms %.4f "'
                  % (np.sqrt((d ** 2).mean()), np.median(d), d.max(),
                     np.percentile(r, 80), np.sqrt((d[outer] ** 2).mean())))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    os.makedirs(OUT, exist_ok=True)
    {'stage1': do_stage1, 'stage2': do_stage2,
     'transfer': do_transfer, 'report': do_report}[cmd](*sys.argv[2:])
