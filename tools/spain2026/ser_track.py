"""Per-frame star tracking through a SER capture: settling, drift, jitter, PSF, transparency.

Written for Joe Izen's Spain 2026 data (matrix cell 4), 2026-09-09, to answer four questions
that all need the same measurement -- where each star sits in every frame:

  * **is the mount settled when the capture starts, and still tracking when it ends?**  A slew
    that has not damped shows as a translation that decays; a new slew at the end shows as one
    that runs away.  Douglas' rule (2026-09-09): prefer a frame from the MIDDLE of a series as
    the stacking master, because the first frame is where the defect lives.  This tool supplies
    the evidence for the choice rather than assuming it (F23).
  * **how big is the mount's tracking error, and does it look periodic?**  Leon 2026 used a
    Celestron AVX (worm, 180 teeth -> a 479 s period); Joe used a ZWO AM5 (strain-wave).  The
    per-frame translation is the raw material; the capture has to be long enough to see a
    cycle, which is stated with the answer rather than assumed.
  * **what is the PSF?**  Second moments per star per frame give sigma and the elongation, and
    elongation that decays with time is settling rather than optics.
  * **was it clear?**  Per-frame aperture flux on the brightest stars, which is what shows
    cloud; sky level does not (Station 2, `docs/STEP3_2026.md`).

Three traps, the first two inherited from `tools/step3_settling.py` and each of which gave a
wrong answer first:

  * **Phase correlation is unusable.**  A fixed hot-pixel pattern is far more correlated
    frame-to-frame than a sparse star field, so the correlation locks onto the sensor and
    reports zero motion.  Star centroids, not correlation.
  * **A source must span >= 4 connected pixels to be a star.**  Hot pixels are single; without
    the cut the "stars" are the sensor and the answer is zero again.
  * **A CENTROID BOX WITH NO STAR IN IT STILL RETURNS A NUMBER**, and the number is quiet: the
    centroid of a noise field sits near the box centre, so the reported displacement is ~0 and
    a mount that has moved 40 px reads as one that has not moved at all.  Measured on the
    16-frame ROI zenith capture of 2026-08-12 on the first version of this tool: frames 0-4 and
    11-15 reported |dx| < 8 px while the field had in fact walked out of the box, and the tell
    was the second moment pinned at 5.77 px = 20/sqrt(12), the sigma of a uniform 20 px box.
    So two defences, both required: an anchor pass with a wide search box re-locates the field
    before any small box is placed, and a box is only believed when its peak stands 6 sigma
    above the background.

One trap of its own: the reference frame for detection is taken from the MIDDLE of the capture,
never frame 0, for the same reason the stacking master is.

`scale_ppm` is meaningful only as a CHANGE through the capture, not as an absolute: the
reference positions come from a labelled centre of mass and the frame positions from a boxed
moment, and the two estimators differ by a fixed offset of a few tens of ppm.

Writes <out>/<capture>_frames.csv (one row per frame) and <out>/<capture>_stars.csv (one row per
star per frame).  Reads each frame exactly once.

  .venv/Scripts/python.exe tools/spain2026/ser_track.py <capture.ser> <out_dir> [--stars N]
      [--first F] [--last L] [--box B] [--search S]
"""
import argparse
import datetime
import os
import struct

import numpy as np
from scipy import ndimage

EPOCH = datetime.datetime(1, 1, 1)
HEADER_BYTES = 178


def read_header(path):
    with open(path, 'rb') as f:
        h = f.read(HEADER_BYTES)
    if h[:14].decode('latin-1', 'replace') != 'LUCAM-RECORDER':
        raise SystemExit('%s: not a SER file' % path)
    _lu, _color, _endian, w, hgt, depth, n = struct.unpack('<7i', h[14:42])
    dt, dt_utc = struct.unpack('<2q', h[162:178])
    bpp = 2 if depth > 8 else 1
    size = os.path.getsize(path)
    frame_bytes = w * hgt * bpp
    trailer = size - (HEADER_BYTES + n * frame_bytes)
    return dict(w=w, h=hgt, n=n, bpp=bpp, frame_bytes=frame_bytes, trailer=trailer,
                utc=EPOCH + datetime.timedelta(microseconds=dt_utc / 10) if dt_utc > 0 else None)


def read_timestamps(path, H):
    """Per-frame UTC from the trailer, or None.  100 ns ticks since 0001-01-01, as the header."""
    if H['trailer'] != 8 * H['n']:
        return None
    with open(path, 'rb') as f:
        f.seek(HEADER_BYTES + H['n'] * H['frame_bytes'])
        ticks = np.frombuffer(f.read(8 * H['n']), dtype='<i8')
    if not np.any(ticks > 0):
        return None                      # the space is there and unwritten (an aborted capture)
    return [EPOCH + datetime.timedelta(microseconds=int(t) / 10) for t in ticks]


def read_frame(f, H, k):
    f.seek(HEADER_BYTES + k * H['frame_bytes'])
    dt = '<u2' if H['bpp'] == 2 else 'u1'
    return np.frombuffer(f.read(H['frame_bytes']), dtype=dt).reshape(H['h'], H['w'])


def sky_level(d):
    """Background and its MAD sigma, from a 1-in-49 subsample: a whole 61 Mpx median is waste."""
    sub = d[::7, ::7].astype(np.float32)
    bg = float(np.median(sub))
    return bg, float(1.4826 * np.median(np.abs(sub - bg))) or 1.0


def find_stars(d, n_want, margin, sat=60000, nsigma=12.0, min_px=4):
    """Bright, EXTENDED, unsaturated sources, brightest first.

    The >= `min_px` connected pixel rule is what rejects hot pixels (step3_settling.py); the
    upper size cut rejects the odd satellite trail and any saturated blob.

    `nsigma` and `min_px` are arguments because a field's depth per frame varies by more than an
    order of magnitude across one night's captures: measured on 2026-09-09, the 12 Aug zenith
    frame yields 391 sources at 12 sigma and >= 4 px while the 11 Aug `Capture` frame yields 3
    -- and a 1 px matched filter finds ~2600 in BOTH, so the difference is per-pixel signal to
    noise, not an empty sky. A caller measuring pointing rather than astrometry may lower them;
    the defaults are what the settling and drift work used and are left alone.
    """
    bg, sig = sky_level(d)
    mask = (d > bg + nsigma * sig) & (d < sat)
    lab, k = ndimage.label(mask)
    if k == 0:
        return [], bg, sig
    idx = np.arange(1, k + 1)
    sizes = ndimage.sum(mask, lab, idx)
    peaks = np.asarray(ndimage.maximum(d, lab, idx), dtype=np.float64)
    keep = [i for i in range(k) if min_px <= sizes[i] <= 400]
    keep.sort(key=lambda i: -peaks[i])
    cen = ndimage.center_of_mass(d, lab, [i + 1 for i in keep[:n_want * 4]])
    out = []
    for y, x in cen:
        if margin < y < d.shape[0] - margin and margin < x < d.shape[1] - margin:
            out.append((float(y), float(x)))
        if len(out) >= n_want:
            break
    return out, bg, sig


def moments(d, y, x, box, bg, sig):
    """First and second moments in a box, or None when the box holds no star.

    The peak test is the defence against the quiet failure described in the module docstring:
    without it a box of pure noise returns a centroid at the box centre and a second moment of
    box/sqrt(3), and the caller reads a stationary mount.
    """
    y0, x0 = int(round(y)) - box, int(round(x)) - box
    if y0 < 0 or x0 < 0 or y0 + 2 * box > d.shape[0] or x0 + 2 * box > d.shape[1]:
        return None
    s = d[y0:y0 + 2 * box, x0:x0 + 2 * box].astype(np.float32)
    if s.max() < bg + 6 * sig:
        return None
    # the border ring is the local background: a box this small has no room for an annulus
    ring = np.concatenate([s[0, :], s[-1, :], s[:, 0], s[:, -1]])
    s = s - np.median(ring)
    s[s < 0] = 0
    tot = s.sum()
    if tot <= 0:
        return None
    gy, gx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    cy = float((s * gy).sum() / tot)
    cx = float((s * gx).sum() / tot)
    vy = float((s * (gy - cy) ** 2).sum() / tot)
    vx = float((s * (gx - cx) ** 2).sum() / tot)
    return y0 + cy, x0 + cx, float(tot), np.sqrt(max(vx, 0.0)), np.sqrt(max(vy, 0.0))


def coarse_shift(d, anchors, box, search, bg, sig):
    """Where has the field gone?  Median of the brightest stars' displacements, wide box.

    Returns (dy, dx, n_used).  A wide search is what lets the small boxes be placed correctly
    afterwards; without it a shift larger than `box` is invisible (see the module docstring).
    """
    dys, dxs = [], []
    for y, x in anchors:
        y0, x0 = int(round(y)) - search, int(round(x)) - search
        y1, x1 = y0 + 2 * search, x0 + 2 * search
        if y0 < 0 or x0 < 0 or y1 > d.shape[0] or x1 > d.shape[1]:
            continue
        s = d[y0:y1, x0:x1]
        if s.max() < bg + 8 * sig:
            continue
        py, px = np.unravel_index(np.argmax(s), s.shape)
        m = moments(d, y0 + py, x0 + px, box, bg, sig)
        if m is None:
            continue
        dys.append(m[0] - y)
        dxs.append(m[1] - x)
    if len(dys) < 3:
        return 0.0, 0.0, len(dys)
    return float(np.median(dys)), float(np.median(dxs)), len(dys)


def similarity_fit(ref_xy, xy, good):
    """Least-squares similarity (dx, dy, rotation, scale) taking ref onto this frame.

    Rotation in arcsec and scale in ppm, both about the stars' centroid; the translation is the
    mean displacement.  Returns None if fewer than four stars survived.
    """
    if good.sum() < 4:
        return None
    a = ref_xy[good] - ref_xy[good].mean(axis=0)
    b = xy[good] - xy[good].mean(axis=0)
    # complex least squares: b = z * a, z = (a.b*) / (a.a*)
    za = a[:, 0] + 1j * a[:, 1]
    zb = b[:, 0] + 1j * b[:, 1]
    z = np.vdot(za, zb) / np.vdot(za, za)
    d = xy[good].mean(axis=0) - ref_xy[good].mean(axis=0)
    resid = np.abs(zb - z * za)
    return dict(dx=float(d[1]), dy=float(d[0]),
                rot_as=float(np.angle(z) * 206264.806),
                scale_ppm=float((abs(z) - 1.0) * 1e6),
                resid_px=float(np.sqrt((resid ** 2).mean())), nstar=int(good.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ser')
    ap.add_argument('out')
    ap.add_argument('--stars', type=int, default=80)
    ap.add_argument('--box', type=int, default=10, help='centroid half-box, px')
    ap.add_argument('--search', type=int, default=64, help='anchor search half-box, px')
    ap.add_argument('--anchors', type=int, default=15)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--last', type=int, default=-1)
    args = ap.parse_args()

    H = read_header(args.ser)
    stamps = read_timestamps(args.ser, H)
    last = H['n'] - 1 if args.last < 0 else min(args.last, H['n'] - 1)
    frames = list(range(args.first, last + 1))
    name = os.path.splitext(os.path.basename(args.ser))[0]
    tag = os.path.basename(os.path.dirname(args.ser)) + '_' + name
    os.makedirs(args.out, exist_ok=True)

    print('%s: %d x %d, %d frames, UTC start %s, trailer %s'
          % (args.ser, H['w'], H['h'], H['n'], H['utc'],
             'per-frame timestamps' if stamps else 'none'))

    with open(args.ser, 'rb') as f:
        mid = frames[len(frames) // 2]
        ref = read_frame(f, H, mid)
        seeds, bg0, sig0 = find_stars(ref, args.stars, args.search + args.box)
        print('reference frame %d (the middle one, never frame 0): sky %.1f ADU, MAD sigma %.1f, '
              '%d stars' % (mid, bg0, sig0, len(seeds)))
        if len(seeds) < 4:
            raise SystemExit('too few stars to track')
        ref_xy = np.array([[y, x] for y, x in seeds])
        anchors = seeds[:args.anchors]

        rows_f, rows_s = [], []
        for k in frames:
            d = read_frame(f, H, k)
            bg, sig = sky_level(d)
            sy, sx, n_anch = coarse_shift(d, anchors, args.box, args.search, bg, sig)
            xy = np.full((len(seeds), 2), np.nan)
            fl = np.full(len(seeds), np.nan)
            sgx = np.full(len(seeds), np.nan)
            sgy = np.full(len(seeds), np.nan)
            for i, (y, x) in enumerate(seeds):
                m = moments(d, y + sy, x + sx, args.box, bg, sig)
                if m is None:
                    continue
                xy[i] = (m[0], m[1])
                fl[i], sgx[i], sgy[i] = m[2], m[3], m[4]
            good = np.isfinite(xy[:, 0])
            fit = similarity_fit(ref_xy, xy, good)
            t = stamps[k] if stamps else None
            rows_f.append(dict(
                frame=k, utc=t.isoformat() if t else '',
                t_s=(t - stamps[frames[0]]).total_seconds() if stamps else np.nan,
                sky_adu=bg, mad_sig=sig, nstar=int(good.sum()), n_anchor=n_anch,
                coarse_dx=sx, coarse_dy=sy,
                dx_px=fit['dx'] if fit else np.nan, dy_px=fit['dy'] if fit else np.nan,
                rot_as=fit['rot_as'] if fit else np.nan,
                scale_ppm=fit['scale_ppm'] if fit else np.nan,
                resid_px=fit['resid_px'] if fit else np.nan,
                flux_med=float(np.nanmedian(fl)) if good.any() else np.nan,
                sigx_med=float(np.nanmedian(sgx)) if good.any() else np.nan,
                sigy_med=float(np.nanmedian(sgy)) if good.any() else np.nan))
            for i in range(len(seeds)):
                if good[i]:
                    rows_s.append(dict(frame=k, star=i, y=xy[i, 0], x=xy[i, 1],
                                       flux=fl[i], sigx=sgx[i], sigy=sgy[i]))
            if k % 10 == 0 or k == frames[-1]:
                print('  frame %4d  sky %7.1f  n %3d/%d  coarse (%+7.1f, %+7.1f)  '
                      'dx %+8.3f  dy %+8.3f px'
                      % (k, bg, good.sum(), len(seeds), sx, sy,
                         rows_f[-1]['dx_px'], rows_f[-1]['dy_px']), flush=True)

    import pandas as pd
    fp = os.path.join(args.out, tag + '_frames.csv')
    sp = os.path.join(args.out, tag + '_stars.csv')
    pd.DataFrame(rows_f).to_csv(fp, index=False)
    pd.DataFrame(rows_s).to_csv(sp, index=False)
    print('wrote %s\nwrote %s' % (fp, sp))


if __name__ == '__main__':
    main()
