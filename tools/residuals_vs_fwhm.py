"""
Do stage-2 residuals track the local PSF?

The frame-to-frame budget can only test PSF tracking of the *random* error, and only
on datasets that have per-frame data — which are exactly the ones with a nearly
uniform PSF. This closes the gap from the other side: on a deep stack the stage-2
residual is almost entirely the static term, so correlating each matched star's
residual against its own fitted FWHM (from the PSF exploration's stars.json) tests
whether the static error lives where the PSF is bad.

    python tools/residuals_vs_fwhm.py docs/bench/psf/rasalhague/stars.json \
        <stage2_dir>/CATALOGUE_MATCHED_ERRORS.csv --out docs/bench/psf/rasalhague
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stars_json')
    ap.add_argument('matched_csv')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--match-radius', type=float, default=2.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.stars_json, encoding='utf-8') as fp:
        loaded = json.load(fp)
    stars = loaded['stars'] if isinstance(loaded, dict) else loaded
    stars = [s for s in stars if s.get('fit_rms') is not None]
    psf_xy = np.array([(s['x'], s['y']) for s in stars])
    psf_fwhm = np.array([s['fwhm'] for s in stars])

    table = pd.read_csv(args.matched_csv)
    good = ~(table['flag_is_outlier'].astype(bool)
             | table['flag_is_double'].astype(bool)
             | table['flag_missing_pm'].astype(bool))
    table = table[good]

    tree = cKDTree(psf_xy)
    distance, index = tree.query(np.c_[table['px'], table['py']],
                                 distance_upper_bound=args.match_radius)
    hit = np.isfinite(distance)
    matched = table[hit].copy()
    matched['fwhm'] = psf_fwhm[index[hit]]
    print(f'{len(matched)} matched of {len(table)} stage-2 stars '
          f'({len(stars)} PSF-fitted stars available)')

    residual = matched['error(")'].to_numpy()
    fwhm = matched['fwhm'].to_numpy()
    mag = matched['magV'].to_numpy()

    rho, p = stats.spearmanr(fwhm, residual)

    def partial_spearman(a, b, control):
        ranks = [stats.rankdata(v) for v in (a, b, control)]
        r_ab = np.corrcoef(ranks[0], ranks[1])[0, 1]
        r_ac = np.corrcoef(ranks[0], ranks[2])[0, 1]
        r_bc = np.corrcoef(ranks[1], ranks[2])[0, 1]
        return (r_ab - r_ac * r_bc) / np.sqrt((1 - r_ac ** 2) * (1 - r_bc ** 2))

    # two confounds: faint stars both centroid worse and can sit where the field is
    # bad; and both FWHM and residual can grow with field radius for separate reasons
    # (aberrations vs a polynomial underfitting the corners)
    centre_y = (table['py'].min() + table['py'].max()) / 2
    centre_x = (table['px'].min() + table['px'].max()) / 2
    radius = np.hypot(matched['py'] - centre_y, matched['px'] - centre_x).to_numpy()
    partial = partial_spearman(fwhm, residual, mag)
    partial_r = partial_spearman(fwhm, residual, radius)

    lines = [f'stars matched               : {len(matched)}',
             f'Spearman residual vs FWHM   : rho {rho:+.3f} (p {p:.2g})',
             f'same, magnitude partialled  : rho {partial:+.3f}',
             f'same, field-radius partialled: rho {partial_r:+.3f}',
             f'median residual, best third of FWHM : '
             f'{np.median(residual[fwhm < np.percentile(fwhm, 33)]):.4f}"',
             f'median residual, worst third of FWHM: '
             f'{np.median(residual[fwhm > np.percentile(fwhm, 67)]):.4f}"']
    text = '\n'.join(lines)
    print('\n' + text)
    (args.out / 'residuals_vs_fwhm.txt').write_text(text, encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(fwhm, residual, '.', ms=5, alpha=0.5)
    axes[0].set_xlabel('fitted FWHM (px)')
    axes[0].set_ylabel('stage-2 residual (arcsec)')
    axes[0].set_title(f'per star: Spearman ρ = {rho:+.2f} (p = {p:.1g}), '
                      f'mag-partialled {partial:+.2f}', fontsize=10)
    axes[0].grid(alpha=0.3)
    order = np.argsort(fwhm)
    window = max(len(matched) // 8, 5)
    running = np.array([np.median(residual[order][max(0, k - window):k + window])
                        for k in range(len(order))])
    axes[0].plot(fwhm[order], running, '-', c='#d62728', lw=2,
                 label='running median')
    axes[0].legend(fontsize=8)

    shown = axes[1].scatter(matched['px'], matched['py'], c=residual, s=14,
                            cmap='viridis',
                            vmax=np.percentile(residual, 95))
    fig.colorbar(shown, ax=axes[1], shrink=0.85, label='residual (arcsec)')
    axes[1].invert_yaxis()
    axes[1].set_title('stage-2 residuals across the field', fontsize=10)
    axes[1].set_xlabel('x (px)'); axes[1].set_ylabel('y (px)')
    fig.tight_layout()
    fig.savefig(args.out / 'residuals_vs_fwhm.png', dpi=140)
    print(f'\nresults in {args.out}')


if __name__ == '__main__':
    main()
