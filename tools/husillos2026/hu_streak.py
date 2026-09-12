"""Are the stars in a 100-frame horizon STACK streaked by refraction evolution?  Measure them.

hu_perframe.py found that reducing 23_34_38 frame by frame and taking per-star medians cuts
the bright-end residual from 1.171 " (the stack) to 0.393 " -- a factor of three from the
construction alone, on the same photons.  Leon's per-frame driver says why it was built that
way (tools/refraction/drive_horizon.py): "a 45-frame stack smears ~0.5 arcsec of refraction
evolution into the frame edges (measured on the 10-frame pilot)".

THE MECHANISM, if that is it.  At 10 deg the field sets ~9.5 "/s, so over the 132 s of this
capture it drops 0.35 deg; the differential refraction across a +-2.9 deg field changes by
roughly d2R/dz2 x 0.35 x 2.9 ~ 7 " between the centre and the top and bottom edges.  Frames
are aligned to frame 0 by ONE GLOBAL SHIFT, so the centre stays put and the edge stars are
laid down as vertical streaks of several pixels.  A windowed centroid on a uniform streak
locks onto whichever part of it noise made brightest -- an error of up to half the streak
length, random in sign from star to star, growing toward the vertical edges, the same for a
bright star and a faint one.  That is exactly a magnitude-independent, non-smooth,
vertically polarised "structure", which is what section 3v measured and called atmosphere.

THE TEST.  Intensity-weighted second moments of every star in the stack and in one single
frame (frame 50), from the two STACKED_FLOAT images already on disk: the width along sensor y
(within 15 deg of the vertical, section 3h) against the width along x, in three bands of
distance from the field's vertical centre.  Prediction if the mechanism is real: in the stack
the y-width grows from the centre band to the edge bands and the x-width does not; in the
single frame neither does.

WHAT IT FOUND (2026-09-12), which was NOT that.  The stack's stars ARE streaked and the
streak DOES grow toward the edges -- but along x, not y: sigma_x 1.98 -> 2.58 -> 3.43 px from
the centre band outward against sigma_y 2.14 -> 2.30 -> 2.40, and the single frame flat at
~2.2 / ~1.75 in every band.  Streaks along RA growing with the offset perpendicular to it are
the signature of DIFFERENTIAL DRIFT, and the 99 single-frame solves then showed the field
centre's RA advancing at 15.09 "/s -- the sidereal rate to 0.3 % -- with the roll constant to
0.0007 deg: an equatorial mount with tracking OFF.  The stack's own alignment record agrees
(767 px of drift over 99 frames, 7.8 px/frame), and so do 23_31_59's and 23_37_17's; every
other capture in the campaign drifted 1-9 px.  So the refraction-evolution smear this tool was
written to look for is not what limits these stacks: they are stacks of an untracked capture,
with each frame's stars already trailed 6 px along RA (the single frame's sigma_x > sigma_y
says so) and the edge stars smeared a further ~10 px by the cos(Dec) spread in drift rate
across the field, which one global shift cannot follow.  Record section 3w.

    .venv/Scripts/python.exe tools/husillos2026/hu_streak.py
"""
import glob
import os
import zipfile

import numpy as np
import pandas as pd
from astropy.io import fits

HOR = r'D:\MEE2024 output\MEE_output\husillos2026\horizon'
SETS = [('100-frame STACK', os.path.join(HOR, 's1d_h10_g125a')),
        ('frame 50 alone', os.path.join(HOR, 'perframe_h10_g125a', 's1', 'f050', 's1'))]
NY = 6388
BOX = 9                     # half-width of the cutout, px: holds a 7 " (3 px) streak with margin
#: peak above background in units of the local noise.  NOT an ADU floor: the stack is a MEAN,
#: so a real 8-sigma star in it sits 8 ADU above a 1766 ADU background, and an ADU floor
#: written for a single frame (first version: 300 ADU) rejected every star in the stack but
#: the brightest few and printed nothing, which is not the same as printing "no streaks".
MIN_SNR = 15.0


def moments(img, x, y):
    x0, y0 = int(round(x)), int(round(y))
    if not (BOX < x0 < img.shape[1] - BOX and BOX < y0 < img.shape[0] - BOX):
        return None
    c = img[y0 - BOX:y0 + BOX + 1, x0 - BOX:x0 + BOX + 1].astype(float)
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg, sig = np.median(ring), 1.4826 * np.median(np.abs(ring - np.median(ring)))
    c = c - bg
    if sig <= 0 or c.max() < MIN_SNR * sig:
        return None
    c[c < 3 * sig] = 0.0
    yy, xx = np.mgrid[-BOX:BOX + 1, -BOX:BOX + 1]
    s = c.sum()
    mx, my = (c * xx).sum() / s, (c * yy).sum() / s
    vx = (c * (xx - mx) ** 2).sum() / s
    vy = (c * (yy - my) ** 2).sum() / s
    return np.sqrt(max(vx, 0)), np.sqrt(max(vy, 0))


def main():
    print('intensity-weighted second-moment widths, px (sigma), median over stars; y is within '
          '15 deg of the vertical')
    print('%-18s %-22s %6s %8s %8s %8s' % ('image', 'band (distance from', 'stars', 'sigma_x',
                                           'sigma_y', 'y/x'))
    print('%-18s %-22s' % ('', ' vertical centre)'))
    for lab, d in SETS:
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))[0]
        t = pd.read_csv(zipfile.ZipFile(z).open('STACKED_CENTROIDS_DATA.csv'))
        img = fits.getdata(glob.glob(os.path.join(d, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit'))[0])
        rows = []
        for x, y in zip(t['px'].values, t['py'].values):
            m = moments(img, x, y)
            if m:
                rows.append((abs(y - NY / 2), *m))
        R = np.array(rows)
        for name, lo, hi in (('centre  <1000 px', 0, 1000), ('middle 1000-2200', 1000, 2200),
                             ('edge    >2200 px', 2200, 1e9)):
            k = (R[:, 0] >= lo) & (R[:, 0] < hi)
            if k.sum() < 5:
                continue
            sx, sy = np.median(R[k, 1]), np.median(R[k, 2])
            print('%-18s %-22s %6d %8.2f %8.2f %8.2f' % (lab, name, k.sum(), sx, sy, sy / sx))
        print()
    print('If the stack\'s sigma_y climbs from the centre band to the edge bands while sigma_x')
    print('and the single frame do not, the stack\'s edge stars are refraction streaks.')


if __name__ == '__main__':
    main()
