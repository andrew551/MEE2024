"""ZWO AM5 against Celestron AVX, on the same optic, with one estimator.

Douglas, 2026-09-09: can the periodic error of Joe Izen's AM5 (strain-wave) be judged against
Leon 2026's Celestron AVX (worm)?  The two datasets share the telescope -- FRA500 + 0.7x, 3.76
um pixels -- so a comparison of per-frame pointing is fair as long as the same estimator reads
both.

**Mexico Station 1 is in the table too** (added 2026-09-09 when Douglas pointed at
`I:/Mexico 2024/Station 1 Zenith`, which an earlier pass had failed to find and had therefore
reduced to a number eyeballed off a plot axis).  It is a third Celestron AVX, a different optic
(432 mm f/5, ASI6200MM at 1.8485 "/px) and a 3.0 s cadence -- so its PLATE SCALE is not the
others' and the conversion to arcsec is per field, not global.  Its 20 frames span 65.9 s, within
2 % of Husillos' 64.7 s, which makes it the closest thing to a matched window in the project.  This tool imports its estimator from `ser_track.py` rather than carrying a second copy,
for the reason `tools/record_charts.py` exists: four private copies of one measurement diverge
four ways.

What can and cannot be concluded from a short window, stated before the numbers:

  * **A capture shorter than the period cannot measure the period.**  Leon's zenith fields run
    128 s, Husillos' 65 s; a worm period is 480-640 s and a strain-wave period is minutes.  So
    neither dataset can fit a sinusoid, and this tool does not pretend to.
  * **What a short window DOES measure is curvature.**  A sinusoid of amplitude A and period P
    sampled over T << P leaves a definite rms about the best-fit straight line, and that rms is
    a function of A, P and phase.  So an observed rms about a line -- with the measurement noise
    subtracted in quadrature -- converts into a band of admissible A for each P.  This is the
    second of Portland's two readings (`docs/PORTLAND_2026-07-29.md` section 4), made explicit.
  * **The drift RATE is not comparable across pointings.**  Polar misalignment contributes a
    rate that depends on where the mount is looking, so rate scatter between fields at different
    pointings mixes two effects.  Rates are reported per field and not pooled.
  * **Nor is an rms about a line comparable across window lengths.**  Curvature accumulates as
    T^2, so Leon's 128 s fields would read several times worse than Husillos' 65 s one on
    identical mount behaviour and the comparison would mean nothing.  Two repairs, both
    reported: every field is also measured on sliding windows of the SAME duration, and the
    primary statistic is the fitted quadratic curvature in arcsec per second squared, which is a
    property of the motion rather than of the window.  For a sinusoid |accel| <= A w^2, so
    A >= |accel| / w^2 at each period.

Leon's AVX periodic error was inferred, never fitted: **+-4 to +-7 "** from rate scatter over
1.5 cycles (`docs/PORTLAND_2026-07-29.md` section 4, the same FRA500 + 0.7x + AVX rig).  The
London HEQ5 Pro's **+-7.7 " at 537 s** (`docs/ROADMAP.md` section 1.4) is a different mount and
is quoted only as the one fitted period the project owns.  `docs/STEP3_2026.md`, "What an AM5
would have done", predicts a strain-wave mount carries **larger** periodic error than a good
worm, "order +-10-20 " over a period of minutes"; that prediction is what this measures.

  .venv/Scripts/python.exe tools/husillos2026/hu_mount_compare.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import (find_stars, moments, coarse_shift, similarity_fit,  # noqa: E402
                       sky_level, read_header, read_timestamps, read_frame)

OUT = r"F:/MEE_output/husillos2026/mount"
PS = 2.2054043          # "/px, the FRA500 + 0.7x canonical (docs/CAL_PILEO_STEP2.md)
#: per-field plate scale, because Station 1 is a different optic. Anything absent uses PS.
PS_BY_FIELD = {'station1 Mexico zenith': 1.8484826, 'station1 Mexico zenith 2': 1.8484826,
               'station1 Mexico zenith 3': 1.8484826, 'station1 Mexico zenith 4': 1.8484826}
WINDOW = 60             # s: the common window every field is also measured on

HUSILLOS = r"G:/Joe Izen Spain 2026"
S1Z = r"I:/Mexico 2024/Station 1 Zenith"
LEON = r"G:/Leon Aug 2026"

#: (label, kind, path, mount).  Night, 1 s or 4 s, star fields, no Sun anywhere.
FIELDS = [
    ('husillos zenith 08-13', 'ser', os.path.join(HUSILLOS, '2026-08-13/zenith/00_00_21.ser'), 'AM5'),
    ('husillos zenith ROI 08-12', 'ser',
     os.path.join(HUSILLOS, '2026-08-12/zenith/23_24_56.ser'), 'AM5'),
    ('leon Z1_base 08-12', 'fits', os.path.join(LEON, '2026-08-12/Zenith/Z1_base'), 'AVX'),
    ('leon Z2_mid_left 08-12', 'fits',
     os.path.join(LEON, '2026-08-12/Zenith/Z2_mid_left'), 'AVX'),
    ('leon Z3_top_left 08-12', 'fits',
     os.path.join(LEON, '2026-08-12/Zenith/Z3_top_left'), 'AVX'),
    ('leon Z4_top_right 08-12', 'fits',
     os.path.join(LEON, '2026-08-12/Zenith/Z4_top_right'), 'AVX'),
    ('leon Z5_mid_right 08-12', 'fits',
     os.path.join(LEON, '2026-08-12/Zenith/Z5_mid_right'), 'AVX'),
    ('leon Z6_bottom_right 08-12', 'fits',
     os.path.join(LEON, '2026-08-12/Zenith/Z6_bottom_right'), 'AVX'),
    ('station1 Mexico zenith', 'fits', os.path.join(S1Z, '2024-04-08_05_32_53Z'), 'AVX'),
    ('station1 Mexico zenith 2', 'fits', os.path.join(S1Z, '2024-04-08_05_35_48Z'), 'AVX'),
    ('station1 Mexico zenith 3', 'fits', os.path.join(S1Z, '2024-04-08_05_38_32Z'), 'AVX'),
    ('station1 Mexico zenith 4', 'fits', os.path.join(S1Z, '2024-04-08_05_51_25Z'), 'AVX'),
]


def fits_frames(root):
    """(paths, seconds-since-first) for a SharpCap FITS folder, ordered by DATE-OBS."""
    from astropy.io import fits
    from datetime import datetime
    files = sorted(glob.glob(os.path.join(root, '*', '*.fits'))) or \
        sorted(glob.glob(os.path.join(root, '*.fits'))) or \
        sorted(glob.glob(os.path.join(root, '*.FIT')))
    # Station 1's frame 0000 is 122 MB of something that is not FITS -- astropy reports "No
    # SIMPLE card found". Unreadable frames are dropped with a note rather than killing the run.
    good, times = [], []
    for f in files:
        try:
            times.append(datetime.strptime(fits.getheader(f)['DATE-OBS'][:26],
                                           '%Y-%m-%dT%H:%M:%S.%f'))
            good.append(f)
        except Exception as exc:
            print('    skipping %s: %s' % (os.path.basename(f), str(exc)[:60]))
    files = good
    if not files:
        return [], []
    order = np.argsort(times)
    files = [files[i] for i in order]
    times = [times[i] for i in order]
    return files, [(t - times[0]).total_seconds() for t in times]


def track(label, kind, path, stars=150, box=12, search=200, anchors=20):
    """The per-frame similarity fit, whichever container the frames live in."""
    if kind == 'ser':
        H = read_header(path)
        stamps = read_timestamps(path, H)
        n = H['n']
        t = [(stamps[k] - stamps[0]).total_seconds() for k in range(n)] if stamps else \
            list(np.arange(n, dtype=float))
        fh = open(path, 'rb')

        def get(k):
            return read_frame(fh, H, k)
    else:
        from astropy.io import fits
        files, t = fits_frames(path)
        n = len(files)
        fh = None

        def get(k):
            return fits.getdata(files[k]).astype(np.float32)
    if n < 4:
        return None
    try:
        mid = n // 2
        ref = get(mid)
        seeds, _bg0, _s0 = find_stars(ref, stars, search + box)
        if len(seeds) < 4:
            print('%-28s only %d stars; skipped' % (label, len(seeds)))
            return None
        ref_xy = np.array([[y, x] for y, x in seeds])
        rows = []
        for k in range(n):
            d = get(k)
            bg, sig = sky_level(d)
            sy, sx, _ = coarse_shift(d, seeds[:anchors], box, search, bg, sig)
            xy = np.full((len(seeds), 2), np.nan)
            for i, (y, x) in enumerate(seeds):
                m = moments(d, y + sy, x + sx, box, bg, sig)
                if m is not None:
                    xy[i] = (m[0], m[1])
            good = np.isfinite(xy[:, 0])
            f = similarity_fit(ref_xy, xy, good)
            if f:
                rows.append(dict(frame=k, t_s=t[k], **f))
        return pd.DataFrame(rows)
    finally:
        if fh:
            fh.close()


def curvature(t, y, noise_as):
    """Quadratic curvature (d2y/dt2, arcsec/s^2), its standard error, and the rms left over.

    Window-length independent, which an rms about a straight line is not.
    """
    A = np.vstack([np.ones_like(t), t, t ** 2]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    cov = np.linalg.inv(A.T @ A) * noise_as ** 2
    return 2 * c[2], 2 * np.sqrt(cov[2, 2]), float(r.std(ddof=3))


def sliding_rms_line(t, y, window_s):
    """Median rms about a straight line over every contiguous window of `window_s` seconds.

    This is what makes a 128 s capture comparable with a 65 s one.
    """
    out = []
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + window_s))
        if j - i < 5 or j > len(t):
            continue
        tt, yy = t[i:j], y[i:j]
        out.append((yy - np.polyval(np.polyfit(tt, yy, 1), tt)).std(ddof=2))
    return float(np.median(out)) if out else np.nan


def curvature_bound(t, resid_rms_as, noise_as, periods=(180, 300, 450, 600, 900)):
    """For each period, the sinusoid amplitude whose curvature would give this rms about a line.

    Phase is unknown, so the amplitude is reported for the phase that produces the LARGEST
    curvature (an amplitude bound: a real sinusoid at an unlucky phase could be bigger and hide).
    """
    excess = np.sqrt(max(resid_rms_as ** 2 - noise_as ** 2, 0.0))
    out = {}
    A = np.vstack([np.ones_like(t), t]).T
    for P in periods:
        w = 2 * np.pi / P
        best = 0.0
        for ph in np.linspace(0, np.pi, 37):
            y = np.sin(w * t + ph)
            r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            best = max(best, r.std())
        out[P] = excess / best if best > 1e-9 else np.inf
    return excess, out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    global PS
    for label, kind, path, mount in FIELDS:
        if not os.path.exists(path):
            print('%-28s missing: %s' % (label, path))
            continue
        PS = PS_BY_FIELD.get(label, 2.2054043)      # a different optic needs its own scale
        cache = os.path.join(OUT, label.replace(' ', '_') + '.csv')
        if os.path.exists(cache):
            df = pd.read_csv(cache)          # every frame is read once, ever
        else:
            df = track(label, kind, path)
            if df is None or len(df) < 5:
                continue
            df.to_csv(cache, index=False)
        if df is None or len(df) < 5:
            continue
        t = df['t_s'].to_numpy()
        # the mean position's own noise: per-star scatter / sqrt(N)
        noise = float((df['resid_px'] * PS / np.sqrt(df['nstar'])).mean())
        rec = dict(field=label, mount=mount, n=len(df), dur_s=t[-1] - t[0],
                   cad_s=float(np.median(np.diff(t))), stars=int(df['nstar'].median()),
                   noise_as=noise, ps=PS)
        for ax in ('dx', 'dy'):
            y = df[ax].to_numpy() * PS
            p = np.polyfit(t, y, 1)
            rec[ax + '_rate_as_s'] = p[0]
            rec[ax + '_rms_line_as'] = float((y - np.polyval(p, t)).std(ddof=2))
            a, ase, rq = curvature(t, y, noise)
            rec[ax + '_accel'], rec[ax + '_accel_se'], rec[ax + '_rms_quad_as'] = a, ase, rq
            rec[ax + '_win'] = sliding_rms_line(t, y, WINDOW)
        rec['rate_as_min'] = 60 * np.hypot(rec['dx_rate_as_s'], rec['dy_rate_as_s'])
        rec['rms_line_as'] = float(np.hypot(rec['dx_rms_line_as'], rec['dy_rms_line_as']))
        rec['rms_line_win'] = float(np.hypot(rec['dx_win'], rec['dy_win']))
        rec['rms_quad_as'] = float(np.hypot(rec['dx_rms_quad_as'], rec['dy_rms_quad_as']))
        rec['accel_as_s2'] = float(np.hypot(rec['dx_accel'], rec['dy_accel']))
        rec['accel_se'] = float(np.hypot(rec['dx_accel_se'], rec['dy_accel_se']))
        rec['excess_as'], bounds = curvature_bound(t, rec['rms_line_as'], noise * np.sqrt(2))
        for P, A in bounds.items():
            rec['A@%ds' % P] = A
        rows.append(rec)
        print('%-28s %s  %2d fr  %5.1f s  rate %5.2f "/min  rms-about-line %.3f "  noise %.3f "'
              % (label, mount, rec['n'], rec['dur_s'], rec['rate_as_min'],
                 rec['rms_line_as'], noise), flush=True)

    t = pd.DataFrame(rows)
    if t.empty:
        return
    t.to_csv(os.path.join(OUT, 'mount_compare.csv'), index=False)
    print()
    print('=== per field: all rms and noise columns in arcsec ===')
    print('    rms_line_win is the same statistic on a common %d s window, which is what makes'
          % WINDOW)
    print('    a 128 s capture comparable with a 65 s one.')
    print(t[['field', 'mount', 'n', 'dur_s', 'cad_s', 'stars', 'noise_as', 'rate_as_min',
             'rms_line_as', 'rms_line_win']]
          .to_string(index=False, float_format=lambda v: '%.3f' % v))
    print()
    print('=== curvature (window-independent) and the sinusoid it would take ===')
    tt = t.copy()
    for P in (180, 300, 450, 600, 900):
        tt['A%ds' % P] = tt['accel_as_s2'] / (2 * np.pi / P) ** 2
    print(tt[['field', 'mount', 'accel_as_s2', 'accel_se', 'rms_quad_as',
              'A180s', 'A300s', 'A450s', 'A600s', 'A900s']]
          .to_string(index=False, float_format=lambda v: '%.4f' % v))
    print('    accel_as_s2: |d2(position)/dt2| in arcsec/s^2, with its standard error from the')
    print('    per-frame noise. A<P>s: the sinusoid amplitude in arcsec that would produce that')
    print('    curvature at period P (A = accel / w^2) -- a LOWER bound, since the phase that')
    print('    was sampled may not be the one of maximum curvature.')
    print()
    print('=== amplitude of a sinusoid whose curvature would explain the excess, per period ===')
    print('    (an estimate at the most-favourable phase; a real sinusoid at an unlucky phase')
    print('     is larger, so read these as "at least this big if the excess is periodic")')
    cols = [c for c in t.columns if c.startswith('A@')]
    print(t[['field', 'mount', 'excess_as'] + cols]
          .to_string(index=False, float_format=lambda v: '%.2f' % v))
    print()
    for m, g in t.groupby('mount'):
        print('%s: %d field(s), rate %.2f-%.2f "/min, rms about a line on a common %d s window '
              '%.3f-%.3f ", curvature %.4f-%.4f "/s^2'
              % (m, len(g), g.rate_as_min.min(), g.rate_as_min.max(), WINDOW,
                 g.rms_line_win.min(), g.rms_line_win.max(),
                 g.accel_as_s2.min(), g.accel_as_s2.max()))


if __name__ == '__main__':
    main()
