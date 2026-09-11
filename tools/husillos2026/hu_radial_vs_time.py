"""Does the Sun-centred structure STEP at the gain boundary, or RAMP through totality?

Inner-minus-outer radial displacement of the two-witness stars, in 15-frame sub-stacks across
both eclipse blocks: five times the time resolution of the half-block test, on the same
photons, with an estimator that does not care about the gain.

WHY THE HALF-BLOCK TEST CANNOT SETTLE IT.  hu_gain_or_time.py reads, inner minus outer:

    g125_A -> g125_B   same gain     +57 +- 37 ppm   (nothing)
    g125_B -> g0_A     cross gain   -119 +- 36 ppm   (the boundary)
    g0_A   -> g0_B     same gain     -50 +- 45 ppm   (1.1 sigma -- but the SAME SIGN)

That was read as "follows the gain".  It is equally consistent with something absent in the
first 20 s, starting around 18:29:30-50 and growing through the gain-0 block -- an ONSET IN
TIME that straddles the boundary.  At 27-34 shared stars per pair the within-gain-0 term is
not significant either way.  The seeing was measured (hu_seeing.py) and does not step at the
boundary (-0.15 +- 0.11 px, 1.3 sigma, wrong direction, and ~30x too weak by estimate).

A windowed, background-subtracted centroid is scale-invariant -- it does not care whether a
star is 500 or 2100 ADU -- and the two blocks collected the SAME PHOTONS.  So if the
inner-minus-outer displacement STEPS at frame 171 -> 2 while the sky is continuous, the step is
in the electronics; if it RAMPS through the boundary without noticing it, the carrier is time
(atmosphere, the Moon crossing the corona at ~20"/39 s, the sky brightening toward C3) and the
gain is a bystander.

WHY SUB-STACKS AND NOT SINGLE FRAMES.  The first version measured each frame on its own and
reached 2 frames of 126 and 5 of 101: a single 315 ms frame shows only the brightest handful
of stars at 8 sigma, and the inner 2.3-5 R_sun holds few bright ones.  Summing 15 consecutive
frames raises the SNR 3.9x and reaches G ~ 10.  No alignment is applied inside a sub-stack --
the mount drifts at most 3.2 px over a whole block, so under 0.4 px in 15 frames, a uniform
smear that cannot bias inner against outer -- and the drift BETWEEN sub-stacks is a
translation, which the similarity fit removes.

Method, per sub-stack: windowed centroid (Gaussian weight, sigma 2 px, three iterations, ring
background, peak > 8 sigma of the ring) for every two-witness star with G <= 10 and
r >= 2.3 R_sun; a similarity (translation + rotation + scale) fitted from the stacked gain-0
positions, so whole-field motion and any scale change are removed; the residual's RADIAL
component about the Sun, averaged over inner (r < 5 R_sun) and outer (r > 7); the difference
with a standard error from the star-to-star scatter.

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
G_MAX = 10.0
R_MIN, R_INNER, R_OUTER = 2.3, 5.0, 7.0
BIN = 15
MIN_BIN = 10       # a trailing sub-stack shorter than this is dropped rather than compared


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
    """Remove translation + rotation + scale between reference and frame; radial residual, px."""
    rx, ry = ref_x[ok] - SUNX, ref_y[ok] - SUNY
    dx, dy = x[ok] - ref_x[ok], y[ok] - ref_y[ok]
    n = int(ok.sum())
    Z, O = np.zeros(n), np.ones(n)
    M = np.vstack([np.column_stack([O, Z, -ry, rx]), np.column_stack([Z, O, rx, ry])])
    c, *_ = np.linalg.lstsq(M, np.concatenate([dx, dy]), rcond=None)
    fit = M @ c
    res_x, res_y = dx - fit[:n], dy - fit[n:]
    R = np.hypot(rx, ry)
    return (res_x * rx + res_y * ry) / R, c[3] * 1e6, res_x, res_y


def measure(sub, px, py, inner, outer):
    x = np.full(len(px), np.nan)
    y = np.full(len(px), np.nan)
    for i, (xx, yy) in enumerate(zip(px, py)):
        c = windowed_centroid(sub, yy, xx, BOX)
        if c is not None:
            y[i], x[i] = c
    ok = ~np.isnan(x)
    if ok.sum() < 8 or (ok & inner).sum() < 3 or (ok & outer).sum() < 5:
        return None
    radial, scale_ppm, res_x, res_y = similarity_residual(px, py, x, y, ok)
    ri, ro = radial[inner[ok]], radial[outer[ok]]
    se = float(np.hypot(ri.std(ddof=1) / np.sqrt(len(ri)), ro.std(ddof=1) / np.sqrt(len(ro))))
    # per-star 2-D residuals are kept so the FIELD can be examined afterwards: whether the
    # structure is Sun-centred (a 1/r radial term) or a generic smooth distortion that only
    # looks radial when projected -- the discriminator between "about the Sun" and "the
    # atmosphere", which the inner-minus-outer number alone cannot make
    stars_df = pd.DataFrame(dict(star=np.where(ok)[0], px=px[ok], py=py[ok],
                                 res_x=res_x, res_y=res_y, radial=radial))
    return dict(n=int(ok.sum()), n_in=len(ri), n_out=len(ro), inner_px=float(ri.mean()),
                outer_px=float(ro.mean()), diff_px=float(ri.mean() - ro.mean()), diff_se=se,
                scale_ppm=float(scale_ppm)), stars_df


def main():
    os.makedirs(OUT, exist_ok=True)
    px, py, r = stars()
    inner, outer = r < R_INNER, r > R_OUTER
    rows, star_rows = [], []
    for label, path, first, last, gain, t0, fps in BLOCKS:
        H = read_header(path)
        t0s = sum(float(v) * m for v, m in zip(t0.split(':'), (3600, 60, 1)))
        with open(path, 'rb') as f:
            k = first
            while k <= last:
                k1 = min(k + BIN - 1, last)
                if k1 - k + 1 < MIN_BIN:
                    break
                sub = np.zeros((H['h'], H['w']), dtype=np.float64)
                for j in range(k, k1 + 1):
                    sub += read_frame(f, H, j)
                got = measure(sub, px, py, inner, outer)
                if got is None:
                    print('  %s frames %3d-%3d: too few stars' % (label, k, k1), flush=True)
                else:
                    m, sd = got
                    m.update(block=label, f0=k, f1=k1, t_s=t0s + 0.5 * (k + k1) / fps)
                    rows.append(m)
                    sd['block'], sd['f0'], sd['t_s'] = label, k, m['t_s']
                    star_rows.append(sd)
                    print('  %s frames %3d-%3d  inner-outer %+.3f +- %.3f px on %d stars'
                          % (label, k, k1, m['diff_px'], m['diff_se'], m['n']), flush=True)
                k = k1 + 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'radial_vs_time.csv'), index=False)
    pd.concat(star_rows).to_csv(os.path.join(OUT, 'radial_vs_time_stars.csv'), index=False)

    print()
    print('INNER (< %.0f R_sun) MINUS OUTER (> %.0f) RADIAL RESIDUAL, px, %d-frame sub-stacks'
          % (R_INNER, R_OUTER, BIN))
    print('   relative to the gain-0 STACK; positive = inner stars further from the Sun')
    print('   %-8s %-9s %-12s %4s %4s %4s %16s %9s'
          % ('block', 'frames', 'mid UTC', 'n', 'in', 'out', 'inner-outer', 'scale'))
    for _, q in df.iterrows():
        m = q.t_s
        print('   %-8s %3d-%-3d   %02d:%02d:%05.2f %4d %4d %4d %+8.3f +- %-5.3f %+6.0f ppm'
              % (q.block, q.f0, q.f1, int(m // 3600), int(m % 3600 // 60), m % 60,
                 q.n, q.n_in, q.n_out, q.diff_px, q.diff_se, q.scale_ppm))
    a, b = df[df.block == 'gain125'], df[df.block == 'gain0']
    if len(a) and len(b):
        wa = 1 / a.diff_se ** 2
        wb = 1 / b.diff_se ** 2
        ma, mb = (a.diff_px * wa).sum() / wa.sum(), (b.diff_px * wb).sum() / wb.sum()
        ea, eb = 1 / np.sqrt(wa.sum()), 1 / np.sqrt(wb.sum())
        print()
        print('WHOLE BLOCKS (inverse-variance): gain 125 %+.3f +- %.3f px   gain 0 %+.3f +- %.3f '
              'px   difference %+.3f +- %.3f px' % (ma, ea, mb, eb, ma - mb, np.hypot(ea, eb)))
        # a step needs the last gain-125 bins to differ from the first gain-0 bins while each
        # side is flat; a ramp needs a trend within a block as large as the step
        for lab, s in (('gain 125', a), ('gain 0', b)):
            if len(s) >= 3:
                t = (s.t_s - s.t_s.mean()).values
                sl = float(np.polyfit(t, s.diff_px.values, 1)[0])
                print('   within %-8s trend %+.4f px/s = %+.3f px over the block'
                      % (lab, sl, sl * (s.t_s.max() - s.t_s.min())))
    print('   a STEP between the last gain-125 bins and the first gain-0 bins, with flat bins')
    print('   either side, is the electronics; a RAMP that ignores the boundary is time.')


if __name__ == '__main__':
    main()
