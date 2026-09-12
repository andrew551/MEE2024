"""Is Husillos' larger night-map residual atmosphere, or noise, or the model?  The magnitude test.

Douglas, 2026-09-12: "Although the vertical polarisation of the Leon site may be greater, the
absolute value of the atmospheric disturbance at Husillos looks considerably larger, even at
the zenith, but particularly near the horizon."

It does -- 0.159 " against 0.067 " at the zenith, 1.0-1.2 " against 0.18-0.35 " near the
horizon -- and before agreeing that the difference is the ATMOSPHERE, two things that are not
atmosphere have to be taken out, because both are larger at Husillos:

  * PER-STAR CENTROID NOISE.  Husillos' 1 s gain-0 frames are read-noise limited by 11x
    (docs/HUSILLOS2026_ZENITH.md); Leon's 4 s gain-101 frames are sky-limited.  Noise is
    magnitude-DEPENDENT: faint stars are noisier, bright ones are not.  Structure -- model
    error, refraction error, atmosphere -- displaces a bright star and a faint star by the
    same angle.  So binning the residuals in magnitude separates them, which is exactly what
    tools/floor_vs_sampling.py does for the zenith floors of three instruments and what
    docs/STEP3_CHARTS_AND_SETTINGS.md section 2 quotes ("at G 8-10 the three floors agree to
    1.4x in ARCSEC").  The bright-end asymptote is the structure;
  * MODEL TRANSFER.  Both cells freeze cubic-and-above from a zenith reference onto fields at
    9-12 deg, but Leon froze a CUBIC averaged over SIX zenith fields per night and Husillos
    freezes cubic-through-QUINTIC from ONE.  hu_lowfield.py measured what that costs at 10 deg:
    freeing the quintic takes the residual 1.256 -> 0.928 ".  The free-quintic residuals are
    binned beside the frozen ones, so the transfer's share is visible per magnitude bin.

The binning is floor_vs_sampling.py's, copied rather than imported because that module runs
at import: one rms per field per bin, fields averaged with equal weight, a bin needs 8 stars,
each field clipped at 3 x MAD with a 2.5 arcsec floor after its own median is removed.  The
Leon horizon rows are rebuilt the way its map was drawn (tools/step3_atmosphere_maps.py
clip_medians): the per-star MEDIAN over the ~45 per-frame corrections-ON quadratic-free fits,
a star needing 20 frames.  That is the like-for-like row for Husillos' frozen 10 deg fields --
same rung, same corrections, same gate.

Not like-for-like, and labelled: the two zenith rows.  Leon's is the night's six-field cubic
frozen and the quadratic free; Husillos' is a free quintic on its one field, which has nothing
to freeze onto it.

Writes atmosphere/maps_bymag.csv.

    .venv/Scripts/python.exe tools/husillos2026/hu_maps_bymag.py
"""
import glob
import os

import numpy as np
import pandas as pd

R = r'D:\MEE2024 output\MEE_output'
HUS = os.path.join(R, 'husillos2026')
HOR = os.path.join(HUS, 'horizon')
OUT = os.path.join(HUS, 'atmosphere', 'maps_bymag.csv')
BINS = [(4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]   # floor_vs_sampling.py's

#: (set, kind, construction note, list of residual-file globs or ('leon-horizon', window))
SETS = [
    ('Husillos zenith', 'own free quintic, refraction on, gate 0.5 " -- NOT the Leon construction',
     [os.path.join(HUS, 'step3', 'ref', '**', 'TWOD_RESIDUALS.csv')]),
    ('Leon zenith (12)', "night's six-field cubic frozen, quadratic free, corrections off",
     [os.path.join(R, 'step3_record', 'zenith_quadfree', '*', '**', 'TWOD_RESIDUALS.csv')]),
    ('Husillos 15 deg, frozen', 'zenith quintic cubic+ frozen, quadratic free, corr ON (the map)',
     [os.path.join(HOR, 's2d_h10_g125d', '**', 'TWOD_RESIDUALS.csv')]),
    ('Husillos 15 deg, FREE', 'whole quintic free, corr ON -- model transfer removed',
     [os.path.join(HOR, 's2f_h10_g125d', '**', 'TWOD_RESIDUALS.csv')]),
    ('Husillos 10 deg, frozen (2)', 'zenith quintic cubic+ frozen, quadratic free, corr ON (the map)',
     [os.path.join(HOR, 's2d_h10_g0', '**', 'TWOD_RESIDUALS.csv'),
      os.path.join(HOR, 's2d_h10_g125a', '**', 'TWOD_RESIDUALS.csv')]),
    ('Husillos 10 deg, FREE', 'whole quintic free, corr ON -- model transfer removed (gain 0 only)',
     [os.path.join(HOR, 's2f_h10_g0', '**', 'TWOD_RESIDUALS.csv')]),
    # the two 10 deg captures were UNTRACKED (record section 3w); the rows above are their
    # smeared stacks, kept so the correction can be seen; this row is the valid reduction
    ('Husillos 10 deg, PER-FRAME (2)', 'per-star medians over 99 separately solved frames (the valid '
     'reduction of an untracked capture)',
     [os.path.join(HOR, 'perframe_h10_g0', 's1_TWOD_RESIDUALS.csv'),
      os.path.join(HOR, 'perframe_h10_g125a', 's1_TWOD_RESIDUALS.csv')]),
    ('Leon horizon (9)', 'per-star median of ~45 per-frame quadratic-free corr-ON fits (the map)',
     'leon-horizon'),
]
LEON_PF = os.path.join(R, 'refraction', 'perframe')


def clip(dx, dy, mag):
    """floor_vs_sampling.fields' clip: own median removed, 3 MAD on the vector, 2.5 " floor."""
    dx, dy = dx - np.median(dx), dy - np.median(dy)
    m = np.hypot(dx, dy)
    lim = max(3 * 1.4826 * np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
    k = m < lim
    return pd.DataFrame(dict(magV=mag[k], dx=dx[k], dy=dy[k]))


def field_from_csv(path):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    return clip(d.dx_arcsec.values, d.dy_arcsec.values, d.magV.values)


def leon_horizon_fields():
    """step3_atmosphere_maps.clip_medians, per window: median over frames per star."""
    out = []
    for w in ('N1', 'N2', 'N3'):
        for f in ('H1', 'H2', 'H3'):
            files = sorted(glob.glob(os.path.join(LEON_PF, w, f, 'f*', 'corr_on', '**',
                                                  'TWOD_RESIDUALS.csv'), recursive=True))
            acc = {}
            for fp in files:
                d = pd.read_csv(fp)
                d.columns = [c.strip() for c in d.columns]
                d = d[d['magV'] <= BINS[-1][1]]     # the last bin's edge, nothing else
                for sid, dx, dy, mag in zip(d.ID, d.dx_arcsec, d.dy_arcsec, d.magV):
                    acc.setdefault(sid, []).append((dx, dy, mag))
            ids = [k for k, v in acc.items() if len(v) >= 20]
            if not ids:
                continue
            P = np.array([[np.median([q[0] for q in acc[i]]), np.median([q[1] for q in acc[i]]),
                           acc[i][0][2]] for i in ids])
            out.append(clip(P[:, 0], P[:, 1], P[:, 2]))
    return out


def by_bin(flds, lo, hi, min_stars=8):
    """floor_vs_sampling.by_bin: one rms per field per bin, fields equal-weighted."""
    vals = []
    for d in flds:
        k = (d.magV >= lo) & (d.magV < hi)
        if k.sum() < min_stars:
            continue
        vals.append(float(np.sqrt(np.mean(d.dx[k] ** 2 + d.dy[k] ** 2))))
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def main():
    print('per-field residual rms by Gaia G, averaged over fields (arcsec; fields contributing)')
    print('%-28s %s %10s %10s %7s' % ('set', '  '.join('%9s' % ('%d-%d' % b) for b in BINS),
                                       'bright', 'faint', 'f/b'))
    rows = []
    for name, note, src in SETS:
        if src == 'leon-horizon':
            flds = leon_horizon_fields()
        else:
            flds = [field_from_csv(p) for g in src for p in sorted(glob.glob(g, recursive=True))]
        if not flds:
            print('%-28s no residuals found' % name)
            continue
        row = [by_bin(flds, lo, hi) for lo, hi in BINS]
        bright = float(np.nanmean([v for (v, n), b in zip(row, BINS) if 8 <= b[0] < 10]))
        faint = float(np.nanmean([v for (v, n), b in zip(row, BINS) if b[0] >= 11]))
        print('%-28s %s %10.3f %10.3f %7.2f' % (
            name, '  '.join('%6.3f/%-2d' % (v, n) if v == v else '%9s' % '-' for v, n in row),
            bright, faint, faint / bright))
        rows.append(dict(set=name, construction=note, fields=len(flds),
                         **{'G%d_%d' % b: v for (v, n), b in zip(row, BINS)},
                         bright_G8_10=bright, faint_G11_13=faint, faint_over_bright=faint / bright))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print()
    print('bright = mean of the 8-9 and 9-10 bins (structure: model, refraction, atmosphere);')
    print('faint = mean of 11-12 and 12-13 (structure plus per-star noise).  A faint/bright')
    print('ratio near 1 is structure-dominated; well above 1 is noise-dominated.  FREE rows have')
    print('the frozen zenith model taken out, so their bright end is refraction + atmosphere only.')


if __name__ == '__main__':
    main()
