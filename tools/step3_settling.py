"""Did the AVX finish settling before the calibration frames started?

The script gave the mount 10 s of rest after the 9.8 deg slew (SharpCap's 5 s hardware
settling plus its own DELAY 5). This measures whether that was enough, from the frames:
track star centroids through each CAL_piLeo block and fit a drift rate. A mount still
settling shows a rate that DECAYS from block to block; steady tracking error and
differential refraction do not.

Two methodological traps, both of which produced a wrong answer first:

  * PHASE CORRELATION IS UNUSABLE HERE. The ASI2600's hot pixels are a fixed sensor
    pattern, far more correlated frame-to-frame than a sparse star field, so the
    correlation locks onto them and reports zero motion no matter what the sky does. It
    passed an injected-shift test and still returned 0.00 px of drift over 215 s, against
    Portland's measured +-4-7" of periodic error. Star centroids, not correlation.
  * A SOURCE MUST SPAN >= 4 CONNECTED PIXELS to be a star. Stars here are ~2.1 px FWHM,
    hot pixels are single. Without that cut the "stars" are hot pixels and the answer is
    zero again.

Usage:  step3_settling.py [cal_tree]
"""
import os
import sys
import glob
from datetime import datetime

import numpy as np
from scipy import ndimage
from astropy.io import fits

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    r"G:\Leon Aug 2026\2026-08-12\Eclipse\CAL_piLeo"
PX = 2.2054043                                   # arcsec/px, CAL_piLeo canonical
BOX = 12                                         # centroid half-box, px
# The mount reported the slew done ~10 s before the first calibration frame opened
# (5 s SharpCap settling + DELAY 5), i.e. at about 18:29:07.5.
T0 = datetime(2026, 8, 12, 18, 29, 7, 500000)


def obs(path):
    return datetime.strptime(fits.getheader(path)["DATE-OBS"][:26],
                             "%Y-%m-%dT%H:%M:%S.%f")


def find_stars(d, n=25):
    """Bright, EXTENDED, unsaturated sources. The >= 4 px rule is what rejects hot pixels."""
    bg = np.median(d)
    sig = 1.4826 * np.median(np.abs(d - bg))
    m = (d > bg + 12 * sig) & (d < 60000)
    lab, k = ndimage.label(m)
    if k == 0:
        return []
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    peaks = ndimage.maximum(d, lab, range(1, k + 1))
    ok = sorted((i for i in range(k) if 4 <= sizes[i] <= 200), key=lambda i: -peaks[i])
    cen = ndimage.center_of_mass(d, lab, [i + 1 for i in ok[:n]])
    return [(y, x) for y, x in cen
            if BOX < y < d.shape[0] - BOX and BOX < x < d.shape[1] - BOX]


def centroid(d, y, x):
    s = d[int(y) - BOX:int(y) + BOX, int(x) - BOX:int(x) + BOX].astype(np.float32)
    s = s - np.median(s)
    s[s < 0] = 0
    if s.sum() <= 0:
        return None
    gy, gx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    return (int(y) - BOX + (s * gy).sum() / s.sum(),
            int(x) - BOX + (s * gx).sum() / s.sum())


def track(files):
    """Median star displacement vs the block's first frame, per frame."""
    files = sorted(files, key=obs)
    ref = fits.getdata(files[0]).astype(np.float32)
    stars = find_stars(ref)
    base = [centroid(ref, y, x) for y, x in stars]
    keep = [i for i, b in enumerate(base) if b]
    out = []
    for f in files:
        d = fits.getdata(f).astype(np.float32)
        dd = [(c[0] - base[i][0], c[1] - base[i][1]) for i in keep
              for c in [centroid(d, *stars[i])] if c]
        if dd:
            a = np.array(dd)
            out.append((obs(f), np.median(a[:, 0]), np.median(a[:, 1]), len(dd)))
    return out


def drift(files):
    tr = track(files)
    if len(tr) < 3:
        return None
    t = np.array([(x[0] - tr[0][0]).total_seconds() for x in tr])
    dy = np.array([x[1] for x in tr])
    dx = np.array([x[2] for x in tr])
    py, px_ = np.polyfit(t, dy, 1), np.polyfit(t, dx, 1)
    res = np.hypot(dy - np.polyval(py, t), dx - np.polyval(px_, t))
    return np.hypot(py[0], px_[0]) * PX, np.std(res) * PX, tr[0][3], t[-1]


BLOCKS = [("cycle 1, 1.0 s", "1.0s/18_29_19"), ("cycle 1, 2.0 s", "2.0s/18_29_27"),
          ("cycle 2, 2.0 s", "2.0s/18_29_57"), ("cycle 3, 2.0 s", "discard/18_30_27"),
          ("cycle 4, 2.0 s", "discard/18_30_56")]


def main():
    print("Star-centroid drift per capture block. A settling mount decays; steady tracking")
    print("error and refraction do not. Late blocks lose their stars to the post-C3 sky.\n")
    print(f"{'block':<32} {'t after slew-complete':>22} {'drift':>11} {'scatter':>9} {'N*':>4}")
    for lbl, d in BLOCKS:
        fs = sorted(glob.glob(os.path.join(ROOT, d, "*.fits")))
        if not fs:
            continue
        t_start = (obs(fs[0]) - T0).total_seconds()
        r = drift(fs)
        if not r:
            print(f"{lbl + '  ' + d:<32} {t_start:19.0f} s   -- too few stars")
            continue
        print(f"{lbl + '  ' + d:<32} {t_start:16.0f}-{t_start + r[3]:.0f} s "
              f"{r[0]:8.3f} \"/s {r[1]:7.3f}\" {r[2]:4d}")
    print("\nFor scale: differential refraction alone, at the 10.5 deg altitude and the")
    print("0.0031 deg/s the field was setting at, is ~0.09 \"/s (3\" per 0.1 deg).")


if __name__ == "__main__":
    main()
