"""The AM5's periodic error, by Portland's rate-scatter method, on captures sharing a pointing.

A single 65 s capture cannot see a period of minutes; what it can see is curvature
(`sp_mount_compare.py`).  The other reading Portland used is **rate scatter between captures**
(`docs/PORTLAND_2026-07-29.md` section 4): if the mount's tracking rate is modulated by a
sinusoid of amplitude A and period P, the rate is modulated by A*w, so a set of captures spread
over a fraction of a cycle shows a spread of fitted drift rates, and A = (rate amplitude) / w.

The method has one requirement that Portland's own six fields did not meet and this tool
enforces: **the captures must share a pointing.**  Polar misalignment contributes a drift rate
that depends on where the mount is looking, so rate scatter across a mosaic mixes periodic error
with pointing.  Two groups in Joe Izen's data qualify:

  * `2026-08-12/Capture` -- four captures, 11 Aug 22:54-23:03 UTC, the mount reporting the same
    RA and Dec throughout (a tracking mount holds its reported RA; the 12 Aug "zenith" ROI set,
    by contrast, reports an RA advancing with the wall clock, which is a mount that is not
    tracking at all -- and its frames drift at 9.3 "/s, the sidereal rate at that declination);
  * `2026-08-12/cal 8 deg` -- three captures, 12 Aug 20:53-20:59 UTC.  These settings files carry
    no mount line at all, so the shared pointing is checked from the frames instead.

The pointing check is not optional and is printed with the answer: the brightest stars of each
capture's middle frame are re-located in the group's first capture, and a group whose captures
sit more than a few hundred pixels apart is reported as a mosaic and not pooled.

What this can and cannot say is the same as before: **four to seven rate samples over 6-9
minutes cannot pin a period.**  They give the rate spread, and the amplitude that spread implies
at each assumed period.  Portland got +-4 to +-7 " for the Celestron AVX this way from six
samples over 13 minutes and said plainly that the period was not pinned; this is the same
statement for the AM5.

  .venv/Scripts/python.exe tools/spain2026/sp_am5_periodic.py [--stride N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import (find_stars, moments, coarse_shift, similarity_fit,  # noqa: E402
                       sky_level, read_header, read_timestamps, read_frame)

OUT = r"D:/MEE2024 output/MEE_output/spain2026/mount"
PS = 2.2054043
G = r"G:/Joe Izen Spain 2026"

GROUPS = {
    'Capture 11Aug (mount held one RA/Dec)': [
        (os.path.join(G, '2026-08-12/Capture/00_54_32.ser'), 1),   # 1.0 s, 100 fr, 131.6 s
        (os.path.join(G, '2026-08-12/Capture/00_57_43.ser'), 3),   # 315 ms, 100 fr, 31.6 s
        (os.path.join(G, '2026-08-12/Capture/00_59_36.ser'), 3),
        (os.path.join(G, '2026-08-12/Capture/01_01_09.ser'), 3),
    ],
    'cal 8 deg 12Aug (no mount telemetry)': [
        (os.path.join(G, '2026-08-12/cal 8 deg/22_53_15.ser'), 1),
        (os.path.join(G, '2026-08-12/cal 8 deg/22_56_41.ser'), 1),
        (os.path.join(G, '2026-08-12/cal 8 deg/22_59_14.ser'), 1),
    ],
}

BOX, SEARCH, ANCH, NSTAR = 12, 40, 20, 120
#: These captures are far shallower per frame than the zenith one (3 sources at the default
#: 12 sigma / >= 4 px against the zenith's 391), while a 1 px matched filter finds ~2600 in
#: both -- so the stars are there and the per-pixel signal to noise is not. Pointing is what is
#: being measured here, not astrometry, so the detection is loosened and the number of stars
#: actually used is printed with every rate.
NSIGMA, MINPX = 6.0, 3
#: how far apart two captures' fields may sit and still count as the same pointing
SAME_POINTING_PX = 400


def track_ser(path, stride):
    """Per-frame similarity fit against this capture's own middle frame.

    The DRIFT RATE is a derivative, so it does not matter that each capture uses its own
    reference star list -- only that the estimator is the same one.
    """
    H = read_header(path)
    stamps = read_timestamps(path, H)
    idx = list(range(0, H['n'], stride))
    with open(path, 'rb') as f:
        mid = idx[len(idx) // 2]
        ref = read_frame(f, H, mid)
        seeds, _bg, _sig = find_stars(ref, NSTAR, SEARCH + BOX, nsigma=NSIGMA, min_px=MINPX)
        if len(seeds) < 4:
            return None, []
        ref_xy = np.array([[y, x] for y, x in seeds])
        rows = []
        for k in idx:
            d = read_frame(f, H, k)
            bg, sig = sky_level(d)
            sy, sx, _ = coarse_shift(d, seeds[:ANCH], BOX, SEARCH, bg, sig)
            xy = np.full((len(seeds), 2), np.nan)
            for i, (y, x) in enumerate(seeds):
                m = moments(d, y + sy, x + sx, BOX, bg, sig)
                if m is not None:
                    xy[i] = (m[0], m[1])
            good = np.isfinite(xy[:, 0])
            fit = similarity_fit(ref_xy, xy, good)
            if fit:
                t = (stamps[k] - stamps[idx[0]]).total_seconds() if stamps else float(k)
                rows.append(dict(frame=k, t_s=t, **fit))
    df = pd.DataFrame(rows)
    return (df if len(df) >= 5 else None), seeds


def field_offset(seeds_a, seeds_b, tol=3.0):
    """Offset (dy, dx) between two captures' star lists, by a vote over pairwise differences.

    The check that the group really is one pointing.  Returns (dy, dx, n_matched); a pointing
    change of hundreds of pixels shows up as a large offset, a mosaic as no consistent vote.
    """
    if len(seeds_a) < 5 or len(seeds_b) < 5:
        return np.nan, np.nan, 0
    a = np.array(seeds_a[:60])
    b = np.array(seeds_b[:60])
    diffs = (b[:, None, :] - a[None, :, :]).reshape(-1, 2)
    best, bn = (np.nan, np.nan), 0
    for cand in diffs[::max(1, len(diffs) // 800)]:
        n = int((np.hypot(*(diffs - cand).T) < tol).sum())
        if n > bn:
            bn, best = n, cand
    if bn < 4:
        return np.nan, np.nan, bn
    near = diffs[np.hypot(*(diffs - best).T) < tol]
    return float(near[:, 0].mean()), float(near[:, 1].mean()), bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride-scale', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for gname, caps in GROUPS.items():
        print('=' * 96)
        print(gname)
        print('=' * 96)
        t0, rows, first_seeds = None, [], None
        for path, stride in caps:
            if not os.path.exists(path):
                print('  missing %s' % path)
                continue
            name = os.path.basename(path)[:-4]
            cache = os.path.join(OUT, 'am5_%s.csv' % name)
            seed_cache = os.path.join(OUT, 'am5_%s_seeds.csv' % name)
            if os.path.exists(cache) and os.path.exists(seed_cache):
                df = pd.read_csv(cache)
                sd = pd.read_csv(seed_cache)
                seeds = list(zip(sd.y, sd.x))
            else:
                df, seeds = track_ser(path, stride * args.stride_scale)
                if df is None:
                    print('  %-10s could not be tracked' % name)
                    continue
                df.to_csv(cache, index=False)
                pd.DataFrame(seeds, columns=['y', 'x']).to_csv(seed_cache, index=False)
            if first_seeds is None:
                first_seeds, off = seeds, (0.0, 0.0, len(seeds))
            else:
                off = field_offset(first_seeds, seeds)
            H = read_header(path)
            stamps = read_timestamps(path, H)
            abs_t0 = stamps[0] if stamps else None
            if t0 is None:
                t0 = abs_t0
            t = df.t_s.to_numpy()
            noise = float((df.resid_px * PS / np.sqrt(df.nstar)).mean())
            rx = np.polyfit(t, df.dx.to_numpy() * PS, 1)[0]
            ry = np.polyfit(t, df.dy.to_numpy() * PS, 1)[0]
            rows.append(dict(capture=name, n=len(df),
                             t_mid_s=(abs_t0 - t0).total_seconds() + t.mean() if abs_t0 else np.nan,
                             span_s=t[-1] - t[0], stars=int(df.nstar.median()), noise_as=noise,
                             rate_x=rx * 60, rate_y=ry * 60,
                             rate=60 * np.hypot(rx, ry),
                             off_px=float(np.hypot(off[0], off[1])), off_n=off[2]))
            print('  %-10s %3d fr  span %6.1f s  %3d stars  rate (%+7.3f, %+7.3f) "/min  '
                  '|rate| %6.3f "/min  noise %.3f "  field offset %s px on %d stars'
                  % (name, len(df), t[-1] - t[0], rows[-1]['stars'],
                     rows[-1]['rate_x'], rows[-1]['rate_y'], rows[-1]['rate'], noise,
                     ('%.1f' % rows[-1]['off_px']) if np.isfinite(rows[-1]['off_px']) else 'NO MATCH',
                     off[2]), flush=True)
        if len(rows) < 3:
            print('  fewer than three captures tracked; no rate scatter')
            continue
        t = pd.DataFrame(rows)
        base = t.t_mid_s.max() - t.t_mid_s.min()
        print()
        moved = t[t.off_px.isna() | (t.off_px > SAME_POINTING_PX)]
        if len(moved):
            print('  NOT ONE POINTING: %d capture(s) sit more than %d px away or do not match; '
                  'rate scatter here mixes periodic error with pointing and is NOT pooled.'
                  % (len(moved), SAME_POINTING_PX))
            print()
            continue
        print('  one pointing confirmed: every capture within %.1f px of the first'
              % t.off_px.max())
        print('  baseline %.1f s (%.1f min), %d captures' % (base, base / 60, len(t)))
        for ax in ('rate_x', 'rate_y'):
            v = t[ax].to_numpy()
            print('  %s: %s  -> spread %.3f "/min (half-range %.3f), sd %.3f'
                  % (ax, np.array2string(v, precision=3),
                     v.max() - v.min(), (v.max() - v.min()) / 2, v.std(ddof=1)))
        # the amplitude a sinusoid of period P needs to modulate the rate by the observed
        # half-range: rate amplitude = A * w, so A = half-range / w
        half = max((t.rate_x.max() - t.rate_x.min()) / 2,
                   (t.rate_y.max() - t.rate_y.min()) / 2) / 60.0     # "/s
        print()
        print('  rate half-range %.4f "/s -> sinusoid amplitude by period:' % half)
        print('   ', '  '.join('%ds: %.1f "' % (P, half * P / (2 * np.pi))
                               for P in (180, 300, 450, 537, 600, 900)))
        print('  (a LOWER bound on the amplitude: %d samples over %.1f min sample only part of'
              % (len(t), base / 60))
        print('   a cycle, and the extremes of the modulation need not have been caught.)')
        print()


if __name__ == '__main__':
    main()
