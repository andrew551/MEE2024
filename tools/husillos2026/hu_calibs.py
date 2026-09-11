"""CalibS: the calibration field shot straight after the two eclipse blocks.

Douglas, 2026-09-11: start it.  It is the middle rung of the three-step ladder
(`docs/V1_4_0_TESTING.md` s5) and the only route cell 4 has to a SAME-DAY, SAME-ALTITUDE
imported plate scale -- which is Method 1.  The night zenith's scale is 1365 ppm from the
eclipse field's and cannot be imported at all, and at this field's 0.0324 "/ppm leverage that
gap is worth several arcsec of L, so nothing else outstanding is comparable in value.

    G:\\Joe Izen Spain 2026\\2026-08-12\\CalibS_Joe_20260812_183018\\20_30_18.ser
    145 frames, 9576x6388, 315.0 ms, GAIN 0, offset 200, 17.7 GB, sensor -0.2 C
    start 18:30:18.334, mid 18:30:41.191, end 18:31:04.049 UTC (45.715 s at 3.1718 fps)

THE ONE FACT THAT GOVERNS THE REDUCTION: **C3 -- the end of totality -- is at 18:30:44.2**,
which is 25.87 s into a 45.7 s capture, i.e. about frame 82.  Frames after C3 have the
photosphere back and are useless for astrometry.  But 82 is ARITHMETIC, and the frame counts
in sections 1-2 of the eclipse record were settled by measuring every frame rather than by
dividing -- the last frames before C3 are already brightening, and Baily's beads arrive
before the disk does.  So this measures the boundary the same way, and stacks nothing until
it has.

Reading strategy is `hu_eclipse_frames`': a 600-row band through the Sun rather than whole
frames, and its `scan()` is imported rather than copied so the two captures are measured by
one implementation.  If CalibS points somewhere else the band will say so -- a returning
photosphere floods far more than the disk.

    .venv/Scripts/python.exe tools/husillos2026/hu_calibs.py [scan|stack]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hu_eclipse_frames import scan  # noqa: E402

G = r'G:\Joe Izen Spain 2026\2026-08-12'
OUT = r'D:\MEE2024 output\MEE_output\husillos2026\calibs'
CAL = os.path.join(G, 'CalibS_Joe_20260812_183018', '20_30_18.ser')

#: from the site card, `G:\Joe Izen Spain 2026\Husillos Spain.JPG`
C3_UTC = '18:30:44.2'
START_UTC = '18:30:18.334'
FPS = 3.1718


def do_scan():
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, 'calibs_frames.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        print('scanning %s (145 frames, 17.7 GB -- the band is ~1.1 GB of it)'
              % os.path.basename(CAL), flush=True)
        df = scan(CAL, 'calibs_20_30_18')
        df.to_csv(cache, index=False)

    print()
    print('=' * 92)
    print('WHERE IS C3 IN CalibS?   predicted frame %.1f from the clock (C3 %s, start %s, %.4f fps)'
          % ((25.866 * FPS), C3_UTC, START_UTC, FPS))
    print('=' * 92)
    cols = ['frame', 't_s', 'utc', 'sky_adu', 'band_median', 'max_adu', 'sat_px',
            'px_over_1000']
    print(df[cols].iloc[::5].to_string(index=False, float_format=lambda v: '%.1f' % v))

    # the transition itself: the photosphere returning is not subtle
    sat = df['sat_px'].values
    quiet = np.median(sat[:20])
    thresh = max(quiet * 10, 100)
    over = np.where(sat > thresh)[0]
    print()
    if len(over):
        k = int(over[0])
        print('first frame with sat_px > %d (10x the first-20-frame median of %d): FRAME %d, '
              't = %.2f s, %s' % (thresh, quiet, k, df['t_s'][k], df['utc'][k]))
        print('the eight frames around it:')
        print(df[cols].iloc[max(k - 4, 0):k + 4]
              .to_string(index=False, float_format=lambda v: '%.2f' % v))
    else:
        print('no saturation transition found in the band -- CalibS may not be the same '
              'pointing as the eclipse captures, or the band misses the disk.')
        print('band max over the capture: %.0f to %.0f ADU'
              % (df['max_adu'].min(), df['max_adu'].max()))

    # and the gentler diagnostic, which moves before the disk does
    print()
    print('brightness ramp (the last totality frames brighten before C3):')
    print(df[['frame', 't_s', 'sky_adu', 'flux_above_sky', 'px_over_1000']]
          .iloc[max(len(df) // 2 - 15, 0):]
          .head(30).to_string(index=False, float_format=lambda v: '%.1f' % v))


#: Measured by `scan`, not divided from the clock -- though the two agree.
#:
#: CalibS is NOT pointed at the Sun: no pixel saturates in any of the 145 frames (band max
#: 2399 to 10417 ADU against a 65535 full scale), there is no disk, no beads and no
#: photosphere.  It is a genuine OFFSET calibration field, which is what the middle rung of
#: the ladder is supposed to be.
#:
#: So C3 shows itself in the SKY LEVEL rather than in saturation, and it is unmistakable.
#: Through the last of totality the band sky creeps up at ~2.4 ADU/frame (1852 at frame 57 to
#: 1913 at frame 82); from frame 83 the rate accelerates every frame (+7, +11, +13, +17 ...)
#: and by frame 140 the sky is 8002 ADU, 4.4x its starting value.  The knee is at frame 82-83
#: against the clock's prediction of 82.0.  Two independent routes, one answer.
#:
#: Frame 0 is dropped as everywhere else in this cell.
FIRST, LAST = 1, 81

#: THE MOUNT WAS STILL SETTLING FOR THE FIRST ~20 FRAMES, and it is the largest defect in this
#: capture.  Sn2 ends 18:30:15.178 and CalibS starts 18:30:18.334 -- a 3.16 s gap in which the
#: mount slewed the 10.17 deg from the Sun to this field, at about 3.2 deg/s -- and the capture
#: began before it had settled.  Measured from stage 1's own per-frame alignment record:
#:
#:     field          frames   total drift   max step
#:     CalibS             81      23.0 px      2.5 px     <-- peak rate 470x the AM5's tracking
#:     eclipse gain 0    101       1.0 px      0.8 px
#:     eclipse gain 125  126       3.2 px      0.7 px
#:     zenith             50       1.2 px      0.4 px
#:
#: The drift grows and asymptotes -- the signature of settling, not of tracking error -- and
#: what it costs is SMEAR WITHIN each 315 ms exposure against a ~1.6 px PSF:
#:
#:     stack index  0-10   0.79 px per exposure     20-30   0.27 px
#:                 10-20   0.54 px                  30-80   <= 0.21 px
#:
#: So frames 21-81 are the settled ones.  This is exactly Douglas' standing rule from 2026-09-09
#: -- "it is always better to use a good frame in the middle of the series than the first one at
#: the beginning" -- and it bites twice here, because `add_img_to_stack` aligns EVERY frame
#: against files[0], so the whole stack was referenced to the most disturbed frame in it.
SETTLED_FIRST = 21

if os.environ.get('HU_CALIBS_SETTLED'):
    FIRST = SETTLED_FIRST

#: THE ZENITH PRESET WAS THE WRONG CHOICE HERE, and Douglas caught it: "I'm surprised there
#: were fewer stars detected at gain zero and 315ms than the corresponding exposure with the
#: Sun. Were the same sensitivity settings used?"  They were not.  Read back from the runs:
#:
#:                              CalibS (zenith)   eclipse blocks (cell 2's)
#:     sigma threshold                  5.0             4.0
#:     min_area                         4               2
#:     sigma_subtract                   3.0             0.0
#:     Gaussian-subtracted detection    False           True
#:     centroids                        29              115 / 97
#:
#: Every difference runs in the strict direction for CalibS.  The original reasoning was half
#: right -- this field has no Sun in it, so it needs no occulter and no coronal subtraction --
#: but it then took the zenith preset's DETECTION THRESHOLDS as well, which nothing required.
#: Two things were changed at once and only one of them was justified.
#:
#: It is not tidiness.  CalibS' +-25.2 ppm scale is the binding term in the whole cell (it is
#: what makes Method 1 +-0.884 " against Method 2's +-0.430 "), and it comes from 26 stars.
#: And `mee2024/field_presets.py` states that an imported plate scale is only transferable if
#: the calibration and science fields were centroided the same way: the ESTIMATOR does match
#: (windowed 2.0 px, annular), but detection decides WHICH stars are admitted and how bright
#: they are, and this optic's centroid bias is brightness-dependent.
ZENITH_S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
             '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
             '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=False',
             '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0',
             '--set', 'centroid_refine_window=True',
             '--set', 'background_subtraction_mode=annular']

#: cell 2's eclipse stage 1 verbatim (hu_eclipse_stars.S1), which is what the science blocks
#: this field feeds were centroided with.  On CalibS the extras should be near no-ops and are
#: kept anyway so the chain is identical: the field is flat (band median 12 ADU above sky) so
#: the coronal subtraction has almost nothing to remove, and no pixel exceeds 16 % of full
#: scale so there is no saturated blob to occult.
ECLIPSE_S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
              '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
              '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
              '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
              '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
              '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
              '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
              '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=10.0',
              '--set', 'coronal_pedestal_adu=2000.0']

#: the same detection WITHOUT the coronal subtraction and the occulter, so the two changes are
#: priced separately rather than confounded the way the first run confounded them
PLAIN_S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
            '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
            '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
            '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
            '--set', 'delete_saturated_blob=False', '--set', 'remove_edgy_centroids=True',
            '--set', 'coronal_subtract=False']

VARIANTS = {'zenith': ZENITH_S1, 'ecl': ECLIPSE_S1, 'plain': PLAIN_S1}
S1 = ZENITH_S1          # what the first run used; kept so that run stays reproducible

#: corrections ON: the Sun is at 8.5 deg altitude, so nothing here is refraction-safe
SITE = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
        '--set', 'observation_date=2026-08-12', '--set', 'guess_date=False',
        '--set', 'observation_lat=42 05 34.55 N', '--set', 'observation_long=4 31 37.26 W',
        '--set', 'observation_height=743.0',
        '--set', 'observation_temp=25.0', '--set', 'observation_pressure=926.5',
        '--set', 'observation_humidity=0.35']

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
DARK = os.path.join(HUS, 'hotpixels', 'husillos_synthetic_dark_all.fit')
REF = os.path.join(HUS, 'step3', 'ref')


def _midtime():
    """Mid-time of the frames actually stacked, from their own SER timestamps."""
    df = pd.read_csv(os.path.join(OUT, 'calibs_frames.csv'))
    t = pd.to_datetime(df.loc[(df.frame >= FIRST) & (df.frame <= LAST), 'utc'])
    return t.iloc[len(t) // 2].strftime('%H:%M:%S')


def do_stack(variant='zenith'):
    import glob
    import json
    import subprocess

    def run(cmd, log):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, 'w') as fh:
            return subprocess.run(cmd, cwd=REPO, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode

    suffix = '' if variant == 'zenith' else '_' + variant
    if os.environ.get('HU_CALIBS_SETTLED'):
        suffix += '_settled'
    d1 = os.path.join(OUT, 's1_calibs' + suffix)
    z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
    if not z:
        print('stage 1: frames %d-%d of %s' % (FIRST, LAST, os.path.basename(CAL)), flush=True)
        # the container plus --frames: a `path.ser#N` on the command line globs to nothing
        run([PY, '-m', 'mee2024.cli', 'stack', CAL,
             '--frames', '%d-%d' % (FIRST, LAST), '--dark', DARK,
             *VARIANTS[variant], '--no-display', '--quiet', '-o', d1],
            os.path.join(d1, 'stage1.log'))
        z = glob.glob(os.path.join(d1, 'centroid_data*.zip'))
    if not z:
        print('stage 1 FAILED -- see %s' % os.path.join(d1, 'stage1.log'))
        return
    import zipfile
    n_cent = len(pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv')))
    print('stage 1 done: %d centroids' % n_cent, flush=True)

    r = glob.glob(os.path.join(REF, '**', 'distortion_data*.zip'), recursive=True)
    if not r:
        print('no zenith reference; run hu_step3.py ref first')
        return
    tmid = _midtime()
    d2 = os.path.join(OUT, 's2_calibs' + suffix)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        print('stage 2 against the zenith quintic reference, mid-time %s UTC' % tmid, flush=True)
        # the ladder's MIDDLE rung: a daytime calibration field is fitted `quadratic`
        # (docs/V1_4_0_TESTING.md s5), which is also what the eclipse blocks use here
        run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', 'quintic',
             '--set', 'distortion_reference_files=' + r[0],
             '--set', 'distortion_fixed_coefficients=quadratic',
             '--set', 'distortion_free_scale=True',
             '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
             '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
             *SITE, '--set', 'observation_time=' + tmid,
             '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
    rr = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not rr:
        print('stage 2 FAILED -- see %s' % os.path.join(d2, 'stage2.log'))
        return
    j = json.load(open(rr[0], encoding='utf-8'))
    print()
    print('CalibS SOLVED  (stage-1 variant: %s)' % variant)
    print('  stars used      %d' % j['#stars used'])
    print('  rms             %.4f arcsec' % j['final rms error (arcseconds)'])
    print('  plate scale     %.7f arcsec/px  (fitted, free scale)'
          % j['platescale (arcseconds/pixel)'])
    print('  RA / DEC / ROLL %.4f / %.4f / %.4f deg' % (j['RA'], j['DEC'], j['ROLL']))
    print()
    print('  eclipse gain 0    2.2029009    gain 125   2.2027459')
    print('  night zenith      2.2059136')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if cmd == 'scan':
        do_scan()
    else:
        do_stack(sys.argv[2] if len(sys.argv) > 2 else 'zenith')
