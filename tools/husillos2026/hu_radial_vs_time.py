"""Does the Sun-centred structure STEP at the gain boundary, or RAMP through totality?

Per-frame positions of the bright two-witness stars across both eclipse blocks, so the
inner-minus-outer radial displacement can be followed frame by frame instead of in ~20 s
half-block averages.

WHY THE HALF-BLOCK TEST CANNOT SETTLE IT.  hu_gain_or_time.py reads, inner minus outer:

    g125_A -> g125_B   same gain     +57 +- 37 ppm   (nothing)
    g125_B -> g0_A     cross gain   -119 +- 36 ppm   (the boundary)
    g0_A   -> g0_B     same gain     -50 +- 45 ppm   (1.1 sigma -- but the SAME SIGN)

That was read as "follows the gain".  It is equally consistent with something that is absent
in the first 20 s, starts around 18:29:30-50, and keeps growing through the gain-0 block --
i.e. an ONSET IN TIME that happens to straddle the boundary.  At 27-34 shared stars per pair
the within-gain-0 term is not significant either way, so the two readings are not
distinguishable from those four numbers.  The seeing was measured (hu_seeing.py) and does not
step at the boundary (-0.15 +- 0.11 px, 1.3 sigma, and in the wrong direction for the effect).

A per-frame centroid is scale-invariant -- a windowed, background-subtracted centroid does not
care whether the star is 500 or 2100 ADU -- and the two blocks collected the SAME PHOTONS.  So
if the inner-minus-outer displacement STEPS at frame 171 -> 2 while the sky is continuous, the
step is in the electronics; if it RAMPS across the boundary without noticing it, the carrier
is time (atmosphere, the Moon's motion across the corona, the sky brightening toward C3) and
the gain is a bystander.  That is the discrimination the half-blocks could not make.

Method, per frame: windowed centroid (Gaussian weight, sigma 2 px, three iterations, ring
background) for every two-witness star with G <= 9.0 and r >= 2.3 R_sun; a similarity
(translation + rotation + scale) fitted from the stacked gain-0 positions to the frame's, so
whole-field motion and any scale change are removed; the residual's RADIAL component about the
Sun, averaged over inner (r < 5 R_sun) and outer (r > 7) stars; the difference, in 10-frame
bins with its standard error.

    .venv/Scripts/python.exe tools/husillos2026/hu_radial_vs_time.py
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
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'seeing')

BLOCKS = [('gain125', os.path.join(G, 'SunJoe_20260812_182845', '20_28_45.ser'),
           46, 171, 125, '18:28:45.594', 3.1705),
          ('gain0', os.path.join(G, 'Sn2_Joe_20260812_182942', '20_29_43.ser'),
           2, 102, 0, '18:29:42.690', 3.1704)]
SUNX, SUNY, PS, RS = 5043.0, 3386.0, 2.2028, 947.1
BOX = 8
G_MAX = 9.0
R_MIN, R_INNER, R_OUTER = 2.3, 5.0, 7.0
BIN = 10


def stars():
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
    print('%d two-witness stars; %d with G <= %.1f beyond %.1f R_sun: %d inner (< %.0f), '
          '%d outer (> %.0f)' % (len(both), keep.sum(), G_MAX, R_MIN,
                                 ((r < R_INNER) & keep).sum(), R_INNER,
                                 ((r > R_OUTER) & keep).sum(), R_OUTER), flush=True)
    return t['px'].values[keep], t['py'].values[keep], r[keep]


def windowed_centroid(d, y, x, box, sig_w=2.0, iters=3):
    """Gaussian-weighted centroid with a ring background; None if the box holds no star."""
    y0, x0 = int(round(y)) - box, int(round(x)) - box
    if y0 < 0 or x0 < 0 or y0 + 2 * box > d.shape[0] or x0 + 2 * box > d.shape[1]:
        return None
    s = d[y0:y0 + 2 * box, x0:x0 + 2 * box].astype(np.float64)
    ring = np.concatenate([s[0, :], s[-1, :], s[:, 0], s[:, -1]])
    bg = float(np.median(ring))
    sig = float(1.4826 * np.median(np.abs(ring - bg))) or 1.0
    if s.max() < bg + 8 * sig:
        return None
    s = s - bg
    gy, gx = np.mgrid[0:s.shape[0], 0:s.shape[1]].astype(np.float64)
    cy, cx = float(box), float(box)
    for _ in range(iters):
        w = np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * sig_w ** 2))
        ws = np.clip(s, 0, None) * w
        tot = ws.sum()
        if tot <= 0:
            return None
        cy, cx = float((ws * gy).sum() / tot), float((ws * gx).sum() / tot)
    return y0 + cy, x0 + cx


def similarity_residual(ref_x, ref_y, x, y, ok):
    """Remove translation + rotation + scale between reference and frame; residual in px."""
    rx, ry = ref_x[ok] - SUNX, ref_y[ok] - SUNY
    dx, dy = x[ok] - ref_x[ok], y[ok] - ref_y[ok]
    n = ok.sum()
    Z, O = np.zeros(n), np.ones(n)
    M = np.vstack([np.column_stack([O, Z, -ry, rx]), np.column_stack([Z, O, rx, ry])])
    c, *_ = np.linalg.lstsq(M, np.concatenate([dx, dy]), rcond=None)
    fit = M @ c
    res_x, res_y = dx - fit[:n], dy - fit[n:]
    R = np.hypot(rx, ry)
    radial = (res_x * rx + res_y * ry) / R
    return radial, c[3] * 1e6


def main():
    os.makedirs(OUT, exist_ok=True)
    px, py, r = stars()
    inner, outer = r < R_INNER, r > R_OUTER
    rows = []
    for label, path, first, last, gain, t0, fps in BLOCKS:
        H = read_header(path)
        t0s = sum(float(v) * m for v, m in zip(t0.split(':'), (3600, 60, 1)))
        with open(path, 'rb') as f:
            for k in range(first, last + 1):
                d = read_frame(f, H, k)
                x = np.full(len(px), np.nan)
                y = np.full(len(px), np.nan)
                for i, (xx, yy) in enumerate(zip(px, py)):
                    c = windowed_centroid(d, yy, xx, BOX)
                    if c is not None:
                        y[i], x[i] = c
                ok = ~np.isnan(x)
                if ok.sum() < 8 or (ok & inner).sum() < 2 or (ok & outer).sum() < 3:
                    continue
                radial, scale_ppm = similarity_residual(px, py, x, y, ok)
                ri = radial[inner[ok]].mean()
                ro = radial[outer[ok]].mean()
                rows.append(dict(block=label, frame=k, t_s=t0s + k / fps,
                                 n=int(ok.sum()), n_in=int((ok & inner).sum()),
                                 n_out=int((ok & outer).sum()), inner_px=ri, outer_px=ro,
                                 diff_px=ri - ro, scale_ppm=scale_ppm))
        print('  %s: %d frames measured' % (label, sum(1 for q in rows if q['block'] == label)),
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'radial_vs_time.csv'), index=False)

    print()
    print('INNER (< %.0f R_sun) MINUS OUTER (> %.0f) RADIAL RESIDUAL, px, %d-frame bins'
          % (R_INNER, R_OUTER, BIN))
    print('   (relative to the gain-0 STACK, so the gain-0 block should centre near 0;')
    print('    positive = inner stars sit further from the Sun than in the gain-0 stack)')
    print('   %-8s %-9s %-13s %7s %14s %10s' % ('block', 'frames', 'mid UTC', 'n', 'inner-outer',
                                               'scale ppm'))
    for label in ('gain125', 'gain0'):
        s = df[df.block == label].reset_index(drop=True)
        for a in range(0, len(s), BIN):
            b = s.iloc[a:a + BIN]
            if len(b) < 4:
                continue
            m = b.t_s.mean()
            hh, mm, ss = int(m // 3600), int(m % 3600 // 60), m % 60
            print('   %-8s %3d-%-3d   %02d:%02d:%05.2f %7d %+8.3f +- %-5.3f %+8.0f'
                  % (label, b.frame.min(), b.frame.max(), hh, mm, ss, len(b),
                     b.diff_px.mean(), b.diff_px.std(ddof=1) / np.sqrt(len(b)),
                     b.scale_ppm.mean()))
    a = df[df.block == 'gain125']
    b = df[df.block == 'gain0']
    print()
    print('WHOLE BLOCKS: gain 125 %+.3f +- %.3f px   gain 0 %+.3f +- %.3f px   step %+.3f px'
          % (a.diff_px.mean(), a.diff_px.std(ddof=1) / np.sqrt(len(a)),
             b.diff_px.mean(), b.diff_px.std(ddof=1) / np.sqrt(len(b)),
             a.diff_px.mean() - b.diff_px.mean()))
    print('   a STEP between the last gain-125 bins and the first gain-0 bins, with flat bins')
    print('   either side, is the electronics; a RAMP that ignores the boundary is time.')


if __name__ == '__main__':
    main()
