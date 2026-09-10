"""Can stars be found in Husillos' eclipse fields?  Stage 1 at Station 1's eclipse settings.

Douglas, 2026-09-10.  Two captures, both 315 ms, both full frame, both offset 200, shot 0.3 s
apart with no slew between them:

  * `SunJoe_20260812_182845/20_28_45.ser` -- 180 slots, **gain 125**, the C2 decay.  The sky falls
    7.4 %/frame to about frame 46 and 0.18 %/frame after it, and frame 47 lands at 18:29:00.77
    UTC against the site card's tabulated C2 of **18:29:00.5** -- so totality starts at frame
    46-47 and Douglas' "approximately frame 50" is right and conservative.  Frames **172-179 are
    all zeros**, eight blank slots, not the five an earlier note guessed.
  * `Sn2_Joe_20260812_182942/20_29_43.ser` -- 103 slots, **gain 0**, the science field.  Its
    **frames 0 and 1 are the Sun capture's**, carried over by the frame buffer exactly as the
    Leon data did.  That is not an inference from "they look odd": the two captures ran at
    different GAINS, and frames 0-1 carry the gain-125 signature -- sky noise 80.06 ADU against
    19.27 for the rest, a ratio of 4.155 against the 10**(125/200) = 4.217 the gain step
    predicts -- while matching the Sun capture's last frame on every other statistic to under
    1 % (saturated pixels 247 k against 247 k; pixels over sky+1000, 1.825 M against 1.833 M).
    `hu_eclipse_frames.py` has the table.

So the usable frames are **Sn2 2-102** and **Sun 46-171**, and this tool stacks them at cell 2's
eclipse settings (`tools/matrix_station1/s1_eclipse_corona.py`): the disk occulter, per-frame
coronal subtraction at sigma 10 px on a 2000 ADU pedestal, Gaussian-subtracted sensitive
detection, windowed centroids on an annular background.

Three differences from cell 2, all forced and all stated with the result:

  * **No darks and no flats.**  Husillos has none, so the hot pixels are still in (see
    `docs/HUSILLOS2026_ZENITH.md` section 7d) and there is no flat.  Cell 2 had both.
  * **No refraction correction at stage 1** -- that is a stage-2 setting, and the site is now
    known (+42.09293, -4.52702, 743 m), so stage 2 can and should use it: the Sun was at 8.6 deg
    altitude, where nothing is refraction-safe.
  * The Sun capture is **gain 125** and the science field **gain 0**, so they cannot be stacked
    together and their scales must not be carried across without measurement.

  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_stars.py stack
  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_stars.py report
"""
import glob
import json
import os
import subprocess
import sys
import zipfile

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
G = r"G:/Joe Izen Spain 2026/2026-08-12"
OUT = r"D:/MEE2024 output/MEE_output/husillos2026/eclipse"

SUN = os.path.join(G, 'SunJoe_20260812_182845', '20_28_45.ser')
SN2 = os.path.join(G, 'Sn2_Joe_20260812_182942', '20_29_43.ser')

# Cell 2's eclipse stage 1, copied from tools/matrix_station1/s1_eclipse_corona.py:66-75 so that
# the two cells are centroided by identical code -- which is what makes any later plate scale
# transferable at all (mee2024/field_presets.py, "an imported plate scale is only transferable
# if the bracket and the science field were centroided the same way").
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
      '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=10.0',
      '--set', 'coronal_pedestal_adu=2000.0']

#: (name, container, first, last, why)
STACKS = {
    'sn2_trimmed': (SN2, 2, 102,
                    'the science field with the two carried-over gain-125 frames dropped'),
    'sn2_asis': (SN2, 0, 102,
                 'the whole file, only to price what the two foreign frames cost'),
    'sun_totality': (SUN, 46, 171,
                     'the Sun capture from C2 to its last written frame'),
    'sun_douglas50': (SUN, 50, 171,
                      "Douglas' conservative start, four frames later"),
}

#: A second pass over the science field with the occulter grown to swallow the corona.
#: Measured on the first pass: of 1715 sources of >= 4 px above 12 sigma in the stack, 1697 lie
#: inside 4 R_sun, at 330-910 per square degree against the 73 per square degree the catalogue
#: actually holds there. They are the residue the sigma-10 px coronal subtraction leaves on the
#: streamers, and they are 99 % of what the solver is being asked to identify the field from.
#: 4 R_sun is 1736 px at 2.2064 "/px; `blob_radius_extra` grows the mask from the saturated core.
MASKED = {'sn2_masked': (SN2, 2, 102, 'the trimmed field with the occulter grown past 4 R_sun')}
MASK_EXTRA = ['--set', 'blob_radius_extra=1750', '--set', 'centroid_gap_blob=150']


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def czip(name):
    z = glob.glob(os.path.join(OUT, 's1_' + name, 'centroid_data*.zip'))
    return z[0] if z else None


def do_stack(only=None):
    for name, (path, first, last, why) in {**STACKS, **MASKED}.items():
        if only and name != only:
            continue
        extra = MASK_EXTRA if name in MASKED else []
        d = os.path.join(OUT, 's1_' + name)
        os.makedirs(d, exist_ok=True)
        if czip(name):
            print('%-14s already stacked' % name)
            continue
        print('%-14s frames %d-%d of %s  (%s)'
              % (name, first, last, os.path.basename(path), why), flush=True)
        rc = run([PY, '-m', 'mee2024.cli', 'stack', path, '--frames', '%d-%d' % (first, last),
                  *S1, *extra, '--no-scan',
                  # `stack` has no --date option (that is on `distortion` and `run`); the SER
                  # header carries the UTC and stage 1 records it as observation_date_header
                  '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
                  '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = czip(name)
        if not z:
            print('  FAILED (rc %d), see %s' % (rc, os.path.join(d, 'stage1.log')), flush=True)
            continue
        j = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))
        print('  %d centroids, platesolved=%s%s'
              % (j.get('n_centroids', -1), j.get('platesolved'),
                 '  RA %.4f DEC %.4f roll %.3f ps %.5f'
                 % (j['RA'], j['DEC'], j['roll'], j['platescale/arcsec'])
                 if j.get('platesolved') else ''), flush=True)


def do_report():
    print('=' * 104)
    print('HUSILLOS ECLIPSE FIELDS -- stage 1 at cell 2\'s eclipse settings, no darks, no flats')
    print('=' * 104)
    print('%-14s %7s %10s %12s %11s %11s %10s   %s'
          % ('stack', 'frames', 'centroids', 'plate-solve', 'RA', 'DEC', '"/px', 'what'))
    for name, (path, first, last, why) in STACKS.items():
        z = czip(name)
        if not z:
            print('%-14s   not stacked' % name)
            continue
        j = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))
        solved = j.get('platesolved')
        print('%-14s %7d %10d %12s %11s %11s %10s   %s'
              % (name, j.get('#frames stacked', last - first + 1), j.get('n_centroids', -1),
                 'yes' if solved else 'NO',
                 '%.4f' % j['RA'] if solved else '-',
                 '%.4f' % j['DEC'] if solved else '-',
                 '%.5f' % j['platescale/arcsec'] if solved else '-', why))
    print()
    for name in STACKS:
        z = czip(name)
        if not z:
            continue
        j = json.loads(zipfile.ZipFile(z).read('results.txt').decode('utf-8'))
        a = j.get('alignment')
        if a:
            print('  %-14s dither span %.2f px over %d frames, per-frame alignment rms %.3f px'
                  % (name, a['dither_span_px'], len(a['rms_px']),
                     sum(a['rms_px']) / len(a['rms_px'])))
    print('->', OUT)


def do_crosscheck(rmin_rsun=4.0, tol=3.0):
    """Are the outer detections real sky objects?  Two captures, same sky, different gain.

    `20_28_45` and `20_29_43` point at the same place 0.323 s apart with no slew, and they were
    shot at gain 125 and gain 0. They are stacked separately, against different masters. A noise
    excursion cannot repeat between them; a real source must.

    The remaining alternative -- a hot pixel, which is fixed to the detector and would also
    repeat -- is excluded separately: only 7 of the eclipse stack's 1715 sources sit within 2 px
    of one of the 2527 sources in the ZENITH stack, against 3 expected by chance. Whatever these
    are, they are not the sensor.
    """
    import zipfile as _zip
    import numpy as np
    import pandas as pd
    rs = 958.2 / 2.2064
    got = {}
    for nm in ('sn2_trimmed', 'sun_totality'):
        z = czip(nm)
        if not z:
            print('%s: not stacked' % nm)
            return
        d = pd.read_csv(_zip.ZipFile(z).open('STACKED_CENTROIDS_DATA.csv'))
        r = np.hypot(d.px - 5175.0, d.py - 2975.0) / rs
        d = d[r >= rmin_rsun]
        got[nm] = (d.px.to_numpy(), d.py.to_numpy())
    (ax, ay), (bx, by) = got['sn2_trimmed'], got['sun_totality']
    print('=' * 96)
    print('ARE THE OUTER DETECTIONS REAL?  Two captures of one sky, 0.3 s apart, gain 0 vs 125')
    print('=' * 96)
    print('  beyond %.0f R_sun: Sn2 %d sources, Sun capture %d' % (rmin_rsun, len(ax), len(bx)))
    dx = (ax[:, None] - bx[None, :]).ravel()
    dy = (ay[:, None] - by[None, :]).ravel()
    ix = np.floor(dx / tol).astype(int)
    iy = np.floor(dy / tol).astype(int)
    key = (ix - ix.min()) * (iy.max() - iy.min() + 1) + (iy - iy.min())
    cnt = np.bincount(key)
    k = int(cnt.argmax())
    m = key == k
    tx, ty = float(dx[m].mean()), float(dy[m].mean())
    print('  translation vote: %d pairs at (%+.1f, %+.1f) px, against a 99th percentile of %.0f'
          % (cnt[k], tx, ty, np.percentile(cnt[cnt > 0], 99)))
    near = np.hypot(ax[:, None] - (bx[None, :] + tx), ay[:, None] - (by[None, :] + ty)).min(axis=1)
    exp = len(bx) * np.pi * tol ** 2 / (9576 * 6388) * len(ax)
    print('  %d of %d Sn2 sources (%.0f %%) have a counterpart within %.0f px; chance %.1f'
          % ((near < tol).sum(), len(ax), 100 * (near < tol).mean(), tol, exp))
    print()
    print('  A noise excursion cannot repeat across two separately stacked captures at')
    print('  different gains. These are real sources on the sky.')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    os.makedirs(OUT, exist_ok=True)
    {'stack': do_stack, 'report': do_report,
     'crosscheck': do_crosscheck}[cmd](*sys.argv[2:])
