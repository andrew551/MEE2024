"""Did the seeing step at the gain boundary?  Per-frame star width across both eclipse blocks.

The half-block test (hu_gain_or_time.py) found a Sun-centred 1/r-shaped difference that
appears across the gain boundary (3.3 sigma in 21 s) and not within a gain (1.5 sigma in
20 s).  The mechanism first proposed for it -- saturation physics -- cannot reach the stars
it is measured on: at gain 125 saturation ends at 1.7 R_sun (27 % of pixels at 1.3-1.7,
0.0 % beyond), at gain 0 it ends at 1.3, the disk occulter is a detection GATE that modifies
no pixel, the masked blur's edge effect dies ~30 px past the core, the stack is float64 so
the pedestal never clips, and the detection footprints are identical in both blocks at every
radius.  The two-witness stars all lie beyond 2.35 R_sun.  So the mechanism is open.

One thing that IS Sun-centred, 1/r-ish, and can change between two sequential captures is
the SEEING.  A star on a steep background gradient has its centroid pulled toward the Sun by
an amount that grows with the PSF width squared -- a wider star integrates more of the
gradient -- and the gradient is steepest near the Sun and falls off outward.  If the seeing
changed between the gain-125 capture (18:29:00-42) and the gain-0 capture (18:29:43-18:30:15),
the boundary between them is also a boundary in atmosphere, and a test that separates gain
from time by ~20 s pairs cannot tell a step at the boundary from the gain itself.

This measures the star width per frame, on the SAME stars in every frame of both captures,
so a seeing change shows as a step in width at 18:29:42.  Stars are the two-witness set
beyond 6 R_sun and brighter than G 8.5 (far from the gradient, so the width is the PSF and
not the corona, and bright enough to measure in one 315 ms frame), located from the gain-0
stack; the two stacks' grids differ by under a pixel and a 16 px box holds the drift.  The
width is the FWHM from each star's half-maximum footprint with background and sigma from its
own border ring -- see the note below on why the first version, built on ser_track.moments()
and a whole-frame sigma, measured nothing.

Reading 227 full frames (28 GB) from the read-only G: drive: run it in the background.

    .venv/Scripts/python.exe tools/husillos2026/hu_seeing.py
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import read_header, read_frame  # noqa: E402

G = r'G:\Joe Izen Spain 2026\2026-08-12'                       # READ ONLY
HUS = r'F:\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'seeing')

#: (label, container, first, last, gain, start UTC, fps) -- the record blocks' own frames.
#: Sn2's timestamp trailer is written and could be used; the Sun capture's is not (see
#: hu_halves.midtime), so both are put on StartCapture + k / ActualFrameRate for symmetry.
BLOCKS = [('gain125', os.path.join(G, 'SunJoe_20260812_182845', '20_28_45.ser'),
           46, 171, 125, '18:28:45.594', 3.1705),
          ('gain0', os.path.join(G, 'Sn2_Joe_20260812_182942', '20_29_43.ser'),
           2, 102, 0, '18:29:42.690', 3.1704)]
SUNX, SUNY, PS, RS = 5043.0, 3386.0, 2.2028, 947.1
BOX = 8            # half-width: 16 px box, room for a sub-3-px drift around a ~2 px PSF
R_MIN = 6.0        # solar radii: beyond the steep gradient, so width means PSF
G_MAX = 8.5        # only stars bright enough to measure in ONE 315 ms frame

# The first version of this measure was blind, and the reason is worth keeping. It used
# ser_track.sky_level for the background and sigma -- a 1-in-49 subsample of the WHOLE frame,
# which the corona dominates -- so 'bg + 6 sigma' sat far above any star and moments() rejected
# all of them ('width nan on 0 stars'). And the few that passed gave second-moment widths of
# 4.1-4.5 px in a 20 px box for a PSF of ~1.6 px FWHM: the moment of a faint star over a box
# that size measures the box (20/sqrt(12) = 5.8 px), not the star. So: background and sigma
# come from each star's own border ring, and the width is the FWHM from the half-maximum
# footprint, 2*sqrt(n_half/pi), which a faint star cannot inflate.


def two_witness_far_stars():
    """px, py of the two-witness stars beyond R_MIN, from the gain-0 stage-2 table."""
    def matched(d):
        z = glob.glob(os.path.join(HUS, 'step3', d, '**', 'distortion_data*.zip'),
                      recursive=True)[0]
        zf = zipfile.ZipFile(z)
        n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
        t = pd.read_csv(zf.open(n), dtype={'ID': str})
        t.columns = [c.strip() for c in t.columns]
        t['ID'] = t['ID'].astype(str).str.strip()
        return t.set_index('ID')
    a, b = matched('eclipse_gain0'), matched('eclipse_gain125')
    both = sorted(a.index.intersection(b.index))
    t = a.loc[both]
    r = np.hypot(t['px'].values - SUNX, t['py'].values - SUNY) * PS / RS
    keep = (r >= R_MIN) & (t['magV'].values <= G_MAX)
    print('%d two-witness stars, %d beyond %.0f R_sun and G <= %.1f used for the width'
          % (len(both), int(keep.sum()), R_MIN, G_MAX), flush=True)
    return t['px'].values[keep], t['py'].values[keep]


def fwhm_local(d, y, x, box):
    """FWHM (px) of the star in a box, from its half-maximum footprint, or None.

    Background and sigma from the box's own border ring. Peak must clear 8 sigma of that
    ring; the width is 2*sqrt(n/pi) over the pixels above half of (peak - bg).
    """
    y0, x0 = int(round(y)) - box, int(round(x)) - box
    if y0 < 0 or x0 < 0 or y0 + 2 * box > d.shape[0] or x0 + 2 * box > d.shape[1]:
        return None
    s = d[y0:y0 + 2 * box, x0:x0 + 2 * box].astype(np.float32)
    ring = np.concatenate([s[0, :], s[-1, :], s[:, 0], s[:, -1]])
    bg = float(np.median(ring))
    sig = float(1.4826 * np.median(np.abs(ring - bg))) or 1.0
    peak = float(s.max())
    if peak < bg + 8 * sig:
        return None
    n_half = int(((s - bg) > 0.5 * (peak - bg)).sum())
    return 2.0 * np.sqrt(n_half / np.pi)


def main():
    os.makedirs(OUT, exist_ok=True)
    px, py = two_witness_far_stars()
    rows = []
    for label, path, first, last, gain, t0, fps in BLOCKS:
        H = read_header(path)
        t0s = sum(float(x) * m for x, m in zip(t0.split(':'), (3600, 60, 1)))
        with open(path, 'rb') as f:
            for k in range(first, last + 1):
                d = read_frame(f, H, k)
                bg, sig = float(np.median(d[::7, ::7])), 0.0
                w, n = [], 0
                for x, y in zip(px, py):
                    fw = fwhm_local(d, y, x, BOX)
                    if fw is None:
                        continue
                    w.append(fw)
                    n += 1
                rows.append(dict(block=label, frame=k, gain=gain,
                                 t_utc_s=t0s + (k - first) / fps + first / fps,
                                 n_stars=n, width_px=float(np.median(w)) if w else np.nan,
                                 width_iqr=float(np.subtract(*np.percentile(w, [75, 25])))
                                 if len(w) > 3 else np.nan,
                                 sky=bg, sky_sigma=sig))
                if k % 20 == 0:
                    print('  %s frame %3d  FWHM %.2f px on %d stars' % (
                        label, k, rows[-1]['width_px'], n), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'per_frame_width.csv'), index=False)

    print()
    print('STAR FWHM (half-maximum footprint, px) BY BLOCK, stars beyond %.0f R_sun' % R_MIN)
    for label in ('gain125', 'gain0'):
        s = df[df.block == label]
        print('   %-8s frames %3d-%3d  median %.3f px  IQR %.3f  first-10 %.3f  last-10 %.3f'
              % (label, s.frame.min(), s.frame.max(), s.width_px.median(),
                 s.width_px.quantile(.75) - s.width_px.quantile(.25),
                 s.width_px.head(10).median(), s.width_px.tail(10).median()))
    a = df[df.block == 'gain125'].width_px.tail(20)
    b = df[df.block == 'gain0'].width_px.head(20)
    print()
    print('ACROSS THE BOUNDARY (last 20 frames of gain 125 vs first 20 of gain 0):')
    print('   %.3f +- %.3f  ->  %.3f +- %.3f px   step %+.3f px = %.1f sigma'
          % (a.median(), a.std() / np.sqrt(len(a)), b.median(), b.std() / np.sqrt(len(b)),
             b.median() - a.median(),
             abs(b.median() - a.median()) / np.hypot(a.std() / np.sqrt(len(a)),
                                                     b.std() / np.sqrt(len(b)))))
    print('   a width STEP at the boundary would let a seeing change masquerade as the gain;')
    print('   no step means the seeing is not the carrier and the mechanism is still open.')


if __name__ == '__main__':
    main()
