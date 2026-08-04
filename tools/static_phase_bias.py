"""
How much of the *static* (stage-2) error is pixel-discreteness error?

The static term is what stacking never removes. Within it, the part a better
PSF-aware centroid method could remove has a signature nothing else shares: it
depends on each star's subpixel phase. A distortion polynomial varies over
thousands of pixels and cannot absorb a signal with period one pixel, so any
phase-dependence surviving in stage-2 residuals is genuinely pixel-level —
centroid bias from pixel sampling, or sub-pixel detector structure.

Takes the CATALOGUE_MATCHED_ERRORS.csv of a stage-2 run. The pixel-frame residual
is recovered convention-free: fit the linear map pixels -> tangent plane from the
matched catalogue positions themselves, then pull the sky residual back through
its inverse. Significance comes from a permutation null (phases decoupled from
residuals), not from assumptions.

    python tools/static_phase_bias.py <stage2_dir>/CATALOGUE_MATCHED_ERRORS.csv \
        --label rasalhague --out docs/bench/psf/phase_bias
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

BINS = 8


def pixel_frame_residuals(table):
    """(residual_px[n,2] as (dy,dx), scale_arcsec) via a data-derived linear map."""
    ra_c = np.radians(table['RA(catalog)'].to_numpy())
    dec_c = np.radians(table['DEC(catalog)'].to_numpy())
    ra_o = np.radians(table['RA(obs)'].to_numpy())
    dec_o = np.radians(table['DEC(obs)'].to_numpy())
    ra0 = np.arctan2(np.mean(np.sin(ra_c)), np.mean(np.cos(ra_c)))
    dec0 = float(np.mean(dec_c))

    def tangent(ra, dec):
        denom = (np.sin(dec0) * np.sin(dec)
                 + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
        xi = np.cos(dec) * np.sin(ra - ra0) / denom
        eta = (np.cos(dec0) * np.sin(dec)
               - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / denom
        return np.degrees(np.c_[xi, eta]) * 3600.0          # arcsec

    cat_tan = tangent(ra_c, dec_c)
    obs_tan = tangent(ra_o, dec_o)
    pixels = np.c_[table['py'], table['px']]
    design = np.c_[pixels, np.ones(len(pixels))]
    coeff, *_ = np.linalg.lstsq(design, cat_tan, rcond=None)   # tan = M@(py,px)+t
    M = coeff[:2].T
    residual_px = (obs_tan - cat_tan) @ np.linalg.inv(M).T
    scale = float(np.sqrt(abs(np.linalg.det(M))))              # arcsec/px
    return residual_px, scale


def phase_curve(phase, residual):
    """Mean signed residual in each phase bin, and that curve's rms contribution."""
    which = np.minimum((phase * BINS).astype(int), BINS - 1)
    means = np.full(BINS, np.nan)
    counts = np.zeros(BINS, dtype=int)
    for b in range(BINS):
        sel = which == b
        counts[b] = int(sel.sum())
        if counts[b] >= 5:
            means[b] = float(np.mean(residual[sel]))
    ok = ~np.isnan(means)
    amplitude = float(np.nanmax(means) - np.nanmin(means)) / 2 if ok.sum() >= 4 else np.nan
    # the variance this curve contributes to the per-star residuals: the
    # count-weighted mean square of the binned means
    rms = float(np.sqrt(np.average(means[ok] ** 2, weights=counts[ok]))) if ok.any() else np.nan
    return means, counts, amplitude, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('matched_csv')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--permutations', type=int, default=500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.matched_csv)
    good = ~(table['flag_is_outlier'].astype(bool)
             | table['flag_is_double'].astype(bool)
             | table['flag_missing_pm'].astype(bool))
    table = table[good].reset_index(drop=True)
    residual_px, scale = pixel_frame_residuals(table)
    total_rms = float(np.sqrt(np.mean(np.sum(residual_px ** 2, axis=1))))
    print(f'{args.label}: {len(table)} stars, platescale {scale:.3f} "/px, '
          f'2-D rms {total_rms:.4f} px = {total_rms*scale:.4f}"')

    phases = np.c_[table['py'] % 1.0, table['px'] % 1.0]
    rng = np.random.default_rng(0)
    lines, record = [], {'label': args.label, 'n_stars': len(table),
                         'platescale_arcsec': scale, 'total_rms_px': total_rms}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    centres = (np.arange(BINS) + 0.5) / BINS
    for axis, name in ((0, 'y'), (1, 'x')):
        means, counts, amplitude, rms = phase_curve(phases[:, axis],
                                                    residual_px[:, axis])
        null_rms = []
        for _ in range(args.permutations):
            shuffled = rng.permutation(residual_px[:, axis])
            null_rms.append(phase_curve(phases[:, axis], shuffled)[3])
        null_rms = np.array(null_rms)
        p_value = float(np.mean(null_rms >= rms))
        # what survives above the sampling noise of the binned means themselves
        excess = float(np.sqrt(max(rms ** 2 - np.median(null_rms) ** 2, 0.0)))
        lines.append(f'{name}-axis: bias amplitude {amplitude:.4f} px, curve rms '
                     f'{rms:.4f} px (null {np.median(null_rms):.4f}, p {p_value:.3f}) '
                     f'-> genuine pixel-level bias {excess:.4f} px')
        record[name] = {'amplitude_px': amplitude, 'curve_rms_px': rms,
                        'null_median_px': float(np.median(null_rms)),
                        'p_value': p_value, 'excess_px': excess,
                        'means': [None if np.isnan(m) else m for m in means],
                        'counts': counts.tolist()}
        ax = axes[axis]
        ax.axhline(0, color='#888', lw=1)
        ax.plot(centres, means, 'o-', label='binned mean residual')
        ax.fill_between(centres, -np.median(null_rms), np.median(null_rms),
                        alpha=0.2, color='#999', label='permutation null (rms)')
        ax.set_xlabel(f'subpixel phase, {name}')
        ax.set_title(f'{name}: p = {p_value:.3f}', fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('mean signed stage-2 residual (px)')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{args.label}: pixel-phase dependence of the static error '
                 f'(total {total_rms:.3f} px)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / f'{args.label}_phase_bias.png', dpi=140)

    both = float(np.hypot(record['x']['excess_px'], record['y']['excess_px']))
    share = (both / total_rms) ** 2 if total_rms else float('nan')
    lines.append(f'combined pixel-level bias {both:.4f} px '
                 f'= {100*share:.1f}% of the static variance')
    record['combined_excess_px'] = both
    record['share_of_static_variance'] = share
    text = '\n'.join(lines)
    print(text)
    (args.out / f'{args.label}.txt').write_text(text, encoding='utf-8')
    with open(args.out / f'{args.label}.json', 'w', encoding='utf-8') as fp:
        json.dump(record, fp, indent=2)


if __name__ == '__main__':
    main()
