"""A hot-pixel mask for Husillos, from the night data, with no dark frame at all.

Douglas, 2026-09-10: *"Joe did not take any darks but he could do that now. In the meantime, is
it possible to create a hot pixel mask using the zenith field that we have?"*

**Not from the zenith capture itself.**  `2026-08-13/zenith/00_00_21.ser` dithers **1.21 px** over
its 50 frames, and `hotpixels.MIN_DITHER_PX` is 3 px -- below that a hot pixel and a star are
indistinguishable by the persistence test and every star would be flagged.  That is exactly why
stage 1 declined on it, in as many words.

**But from its siblings, yes.**  Four captures share the zenith's settings exactly -- gain 0,
offset 220, 1.0 s, same camera, same night, same sensor temperature -- and dither far more,
because the mount was drifting or re-pointing:

  | capture | frames | dither |
  |---|---|---|
  | `2026-08-12/cal 8 deg/22_53_15` | 100 | **42.7 px** |
  | `2026-08-12/cal 8 deg/22_56_41` | 100 | **18.9 px** |
  | `2026-08-12/10 deg/23_31_59` | 100 | (measured here) |
  | `2026-08-13/zenith/00_00_21` | 50 | 1.2 px -- unusable |

A hot pixel is fixed to the DETECTOR and a star to the SKY, so 19-43 px of dither separates them
cleanly.  The mask those captures yield applies to the zenith capture because the sensor, gain,
offset, exposure and temperature are the same; hot pixels are a property of the silicon.

This calls `mee2024.hotpixels.persistence_mask` -- the project's own implementation, validated at
96.3 % recall with no false positives on the bundled example -- rather than carrying a second
copy.  Two things it needs and this supplies:

  * **the shifts, in the pipeline's convention.**  Measured against a stage-1 run whose shifts
    are recorded: `shifts_px = (-dy, -dx)` where dx, dy are the displacements `ser_track.py`
    reports relative to frame 0 (correlation -0.998 and -0.994, slopes -1.02 and -0.98).  A sign
    error here would silently flag the stars instead, which is the one failure mode that looks
    like success.
  * **candidates come from `files[0]` alone** (see the function), so the mask is built per
    capture and the results are UNIONED: a pixel hot in any capture is hot.

`candidate_sigmas` is lowered from its default of 20 to 5, and that CANNOT loosen the answer.
It is a pre-filter on one frame that limits how many pixels are examined; the criterion is
`MIN_DETECTOR_PERSISTENCE = 5.0`, which the flagged pixel must clear in the WEAKEST of all the
frames -- a far harder test than 5 sigma in one of them. At 20 sigma the search finds only the
brightest hot pixels (332 of them), while the dither experiment of `HUSILLOS2026_ZENITH.md`
section 6 implies some 2300 matter in a deep stack: they matter because a stack of 100 frames has
a tenth of one frame's noise, so a pixel far below 20 sigma per frame is a strong detection in
the stack. Lowering the pre-filter to the criterion's own level is what lets the criterion see
them.

The output is a synthetic master dark: zeros everywhere, flagged pixels set high.  The pipeline
then flags exactly those through its ordinary `--dark` path (`hotpixels.dark_mask` cuts at
median + max(10 ADU, 20 sigma), and a zero dark has median 0 and sigma 0), while SUBTRACTING
nothing -- which is what makes it safe to use on frames whose bias it does not share.

  .venv/Scripts/python.exe tools/husillos2026/hu_hotpixels.py build
  .venv/Scripts/python.exe tools/husillos2026/hu_hotpixels.py check
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

G = r"G:/Joe Izen Spain 2026"
OUT = r"F:/MEE_output/husillos2026/hotpixels"
TRACK = r"F:/MEE_output/husillos2026/mount"

NX, NY = 9576, 6388
#: value given to a flagged pixel in the synthetic dark. Anything over dark_mask's 10 ADU floor
#: works; 1000 is far clear of it and still nowhere near saturation.
HOT_ADU = 1000

#: Two mask families, because a mask belongs to a GAIN. Hot pixels are the same silicon defects
#: whatever the gain, but how far each one stands above the noise is not: gain 125 multiplies the
#: ADU per electron by 4.217, and a 0.315 s frame accumulates a third of a 1.0 s frame's dark
#: current, so the same pixel sits 1.3x further above the noise in a gain-125 / 0.315 s frame
#: than in a gain-0 / 1.0 s one. Building one mask per family lets each be found in the
#: conditions it will be used in, and lets the two be compared -- which is a check, since the
#: same silicon should be flagged both times.
FAMILIES = {
    # gain 0, offset 220, 1.0 s -- the zenith capture's settings, and Sn2's gain
    'g0': [
        ('cal8deg_22_53_15', G + '/2026-08-12/cal 8 deg/22_53_15.ser',
         TRACK + '/am5_22_53_15.csv'),
        ('cal8deg_22_56_41', G + '/2026-08-12/cal 8 deg/22_56_41.ser',
         TRACK + '/am5_22_56_41.csv'),
    ],
    # gain 125, 0.315 s -- the Sun capture's settings (its offset is 200 and these are 50, which
    # a mask does not care about: an offset is a constant and the test is on the excess)
    'g125': [
        ('capture_00_57_43', G + '/2026-08-12/Capture/00_57_43.ser', TRACK + '/am5_00_57_43.csv'),
        ('capture_00_59_36', G + '/2026-08-12/Capture/00_59_36.ser', TRACK + '/am5_00_59_36.csv'),
        ('capture_00_54_32', G + '/2026-08-12/Capture/00_54_32.ser', TRACK + '/am5_00_54_32.csv'),
    ],
}
SOURCES = FAMILIES['g0']


def shifts_for(csv):
    """(frame indices, shifts in the pipeline's convention) from a ser_track/sp_am5 table."""
    d = pd.read_csv(csv)
    dx = d['dx'].to_numpy() - d['dx'].to_numpy()[0]
    dy = d['dy'].to_numpy() - d['dy'].to_numpy()[0]
    # verified against a recorded stage-1 alignment: shifts_px = (-dy, -dx), axis order (y, x)
    return d['frame'].to_numpy().astype(int), np.column_stack([-dy, -dx])


def build(family='g0', candidate_sigmas=5.0):
    from mee2024 import hotpixels
    candidate_sigmas = float(candidate_sigmas)
    os.makedirs(OUT, exist_ok=True)
    union = np.zeros((NY, NX), dtype=bool)
    per = {}
    print('=== family %s ===' % family)
    for label, ser, csv in FAMILIES[family]:
        if not os.path.exists(csv):
            print('%-18s no tracker table at %s' % (label, csv))
            continue
        frames, shifts = shifts_for(csv)
        files = ['%s#%d' % (ser, k) for k in frames]
        print('%-18s %d frames, dither %.1f px' % (label, len(files),
                                                   hotpixels.dither_span(shifts)), flush=True)
        mask, info = hotpixels.persistence_mask(files, shifts,
                                                candidate_sigmas=candidate_sigmas)
        if mask is None:
            print('   declined: %s' % info['declined'])
            continue
        print('   %d candidates -> %d flagged (%.4f %% of the frame), noise %.2f ADU'
              % (info['n_candidates'], info['n_flagged'],
                 100 * info['n_flagged'] / mask.size, info.get('noise_adu', float('nan'))))
        per[label] = mask
        union |= mask
        np.save(os.path.join(OUT, 'mask_%s.npy' % label), mask)
    if not per:
        print('nothing built')
        return
    print()
    print('union of %d captures: %d hot pixels (%.4f %% of the frame)'
          % (len(per), union.sum(), 100 * union.sum() / union.size))
    if len(per) > 1:
        ms = list(per.values())
        both = np.logical_and.reduce(ms)
        print('  flagged by every capture: %d (%.0f %% of the union) -- the agreement between'
              % (both.sum(), 100 * both.sum() / max(union.sum(), 1)))
        print('  independent captures is the check that this is the detector and not the sky')
    np.save(os.path.join(OUT, 'mask_union_%s.npy' % family), union)

    from astropy.io import fits
    dark = np.zeros((NY, NX), dtype=np.uint16)
    dark[union] = HOT_ADU
    hdu = fits.PrimaryHDU(dark)
    hdu.header['COMMENT'] = 'Synthetic master dark for Husillos 2026: NOT a dark exposure.'
    hdu.header['COMMENT'] = 'Zero everywhere except pixels the dither-persistence test flagged'
    hdu.header['COMMENT'] = 'as hot, which are set to %d ADU. Subtracting it changes nothing;' % HOT_ADU
    hdu.header['COMMENT'] = 'its only purpose is to carry the mask through --dark.'
    hdu.header['COMMENT'] = 'Built by tools/husillos2026/hu_hotpixels.py from gain 0 /'
    hdu.header['COMMENT'] = 'offset 220 / 1.0 s captures with 19-43 px of dither.'
    path = os.path.join(OUT, 'husillos_synthetic_dark_%s.fit' % family)
    hdu.writeto(path, overwrite=True)
    print('->', path)
    print('   use it as:  --dark "%s"' % path)


def check():
    """Does the mask explain the small detections the stacks are full of?"""
    import zipfile
    m = os.path.join(OUT, 'mask_union_g0.npy')
    if not os.path.exists(m):
        print('no mask yet; run `build`')
        return
    mask = np.load(m)
    print('mask: %d hot pixels (%.4f %% of the frame)' % (mask.sum(), 100 * mask.mean()))
    from scipy.ndimage import binary_dilation
    grown = binary_dilation(mask, iterations=2)
    print()
    print('%-34s %8s %10s %12s %12s' % ('stack', 'cent.', 'area<=2px', 'on a hot px',
                                        'of the <=2px'))
    for root, nm in ((r'F:/MEE_output/husillos2026/eclipse', 'sn2_trimmed'),
                     (r'F:/MEE_output/husillos2026/eclipse', 'sn2_masked'),
                     (r'F:/MEE_output/husillos2026/eclipse', 'sun_totality'),
                     (r'F:/MEE_output/husillos2026/zenith_order', 'with_f0')):
        z = glob.glob(os.path.join(root, 's1_' + nm, 'centroid_data*.zip'))
        if not z:
            continue
        d = pd.read_csv(zipfile.ZipFile(z[0]).open('STACKED_CENTROIDS_DATA.csv'))
        ix = np.clip(d.px.to_numpy().round().astype(int), 0, NX - 1)
        iy = np.clip(d.py.to_numpy().round().astype(int), 0, NY - 1)
        on = grown[iy, ix]
        small = d['area (pixels)'].to_numpy() <= 2
        print('%-34s %8d %10d %12d %12s'
              % (nm, len(d), small.sum(), on.sum(),
                 '%d (%.0f %%)' % ((on & small).sum(),
                                   100 * (on & small).sum() / max(small.sum(), 1))))
    print()
    print('  A centroid sitting on a flagged pixel is one the mask would have removed.')
    print('  The mask is built from NIGHT captures at gain 0 / offset 220 / 1.0 s; the eclipse')
    print('  captures are 0.315 s at offset 200, where a hot pixel accumulates a third of the')
    print('  dark current, so the mask over-covers there rather than under-covers.')


def combine():
    """Union the gain families into the mask to actually use.

    Hot pixels are the same silicon defects at any gain, so the families should nest -- and they
    do: 286 of the gain-0 mask's 389 are also flagged at gain 125, against 0.0 expected by
    chance. They do not nest perfectly because how far a pixel stands above the noise depends on
    the gain: gain 125 has 1.38 e- of read noise against gain 0's 4.73, so it sees six times as
    many (2484 against 389). The union is what each family found in the conditions it was best
    able to look, and it is the mask to use for both fields.
    """
    from astropy.io import fits
    parts = {}
    for fam in FAMILIES:
        f = os.path.join(OUT, 'mask_union_%s.npy' % fam)
        if os.path.exists(f):
            parts[fam] = np.load(f)
    if not parts:
        print('build the families first')
        return
    union = np.logical_or.reduce(list(parts.values()))
    for fam, m in parts.items():
        print('  %-5s %5d pixels' % (fam, m.sum()))
    print('  union %5d pixels (%.4f %% of the frame)' % (union.sum(), 100 * union.mean()))
    np.save(os.path.join(OUT, 'mask_union_all.npy'), union)
    dark = np.zeros((NY, NX), dtype=np.uint16)
    dark[union] = HOT_ADU
    hdu = fits.PrimaryHDU(dark)
    hdu.header['COMMENT'] = 'Synthetic master dark for Husillos 2026: NOT a dark exposure.'
    hdu.header['COMMENT'] = 'Union of the gain-0 and gain-125 dither-persistence masks.'
    hdu.header['COMMENT'] = 'Zero everywhere except flagged pixels, set to %d ADU, so that' % HOT_ADU
    hdu.header['COMMENT'] = 'subtracting it changes nothing and only the mask is carried.'
    path = os.path.join(OUT, 'husillos_synthetic_dark_all.fit')
    hdu.writeto(path, overwrite=True)
    print('->', path)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    {'build': build, 'check': check, 'combine': combine}[cmd](*sys.argv[2:])
