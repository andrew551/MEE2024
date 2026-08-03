"""
Explore the point spread function of real frames.

    python tools/psf_explore.py "tests/data/fits/00_23_49/*.fits" --out docs/bench/psf/zenith --platescale 1.55
    python tools/psf_explore.py tests/data/fits/rasalhague/Rasalhaguemean50.fit --out docs/bench/psf/rasalhague --platescale 1.65

Answers, per dataset, the questions PSF_REVIEW.md says decide everything downstream:

* How wide is the PSF in pixels (which centroiding algorithm is honest to use)?
* Is it Gaussian or Moffat (do the wings matter)?
* Is one PSF enough for the whole field, or does it vary — and does the *asymmetry* vary,
  which is the astrometrically dangerous part?

Writes figures and a summary into --out, and a per-star JSON reused by the centroiding
evaluation, so the expensive detection pass runs once.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from mee2024 import psf                                     # noqa: E402
from mee2024.config import get_default_options              # noqa: E402
from mee2024.stacker_implementation import (                # noqa: E402
    get_centroids_blur, open_image)


def detect(image, options):
    blank = np.zeros(image.shape, dtype=bool)
    found = get_centroids_blur((image, blank, blank), options=options)
    return np.array([c[2] for c in found]), np.array([c[0] for c in found], dtype=float)


def polynomial_maps(stars, shape, quantity):
    """rms residual of constant / linear / quadratic models of a per-star quantity.

    The comparison the request asks for, made quantitative: if quadratic explains little
    more than constant, one PSF describes the field.
    """
    ys = np.array([s['y'] for s in stars])
    xs = np.array([s['x'] for s in stars])
    values = np.array([s[quantity] for s in stars])
    # normalised coordinates keep the design matrix well conditioned
    yn = (ys - shape[0] / 2) / (shape[0] / 2)
    xn = (xs - shape[1] / 2) / (shape[1] / 2)
    designs = {
        'constant': np.ones((len(values), 1)),
        'linear': np.c_[np.ones_like(xn), xn, yn],
        'quadratic': np.c_[np.ones_like(xn), xn, yn, xn * yn, xn ** 2, yn ** 2],
    }
    out = {}
    for name, design in designs.items():
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        out[name] = float(np.sqrt(np.mean((values - design @ coeff) ** 2)))
    return out


def stacked_profile(image, stars, cut=psf.CUT):
    """A high-SNR mean PSF from the brightest clean stars, recentred to subpixel accuracy.

    Shifting by the *fitted* centres before averaging is what makes the wings believable:
    averaging integer-aligned cutouts convolves the profile with the centroid scatter.
    """
    from scipy.ndimage import shift as subpixel_shift

    bright = [s for s in stars if s.get('fit_rms') is not None]
    bright.sort(key=lambda s: -s['flux'])
    chosen = bright[:40]
    if len(chosen) < 5:
        return None, 0
    size = 2 * cut + 1
    accumulated = np.zeros((size, size))
    used = 0
    for star in chosen:
        row, col = int(round(star['y'])), int(round(star['x']))
        if not (cut <= row < image.shape[0] - cut and cut <= col < image.shape[1] - cut):
            continue
        cutout = np.asarray(
            image[row - cut:row + cut + 1, col - cut:col + cut + 1], dtype=float)
        ring = np.concatenate([cutout[0, :], cutout[-1, :], cutout[1:-1, 0],
                               cutout[1:-1, -1]])
        cutout = cutout - np.median(ring)
        total = cutout.sum()
        if total <= 0:
            continue
        offset = (cut - (star['y'] - row) - cut, cut - (star['x'] - col) - cut)
        recentred = subpixel_shift(cutout / total,
                                   ((row - star['y']), (col - star['x'])), order=3)
        accumulated += recentred
        used += 1
    return (accumulated / used if used else None), used


def radial(profile):
    centre = (profile.shape[0] - 1) / 2
    ys, xs = np.indices(profile.shape)
    radii = np.hypot(ys - centre, xs - centre).ravel()
    values = profile.ravel()
    order = np.argsort(radii)
    return radii[order], values[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='a frame, or a glob; the first match is analysed')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--platescale', type=float, default=None, help='arcsec per pixel')
    args = ap.parse_args()

    files = sorted(glob.glob(args.frames))
    if not files:
        raise SystemExit(f'nothing matches {args.frames}')
    args.out.mkdir(parents=True, exist_ok=True)

    options = get_default_options()
    options.update(flag_display=False, centroid_gaussian_subtract=True,
                   centroid_gaussian_thresh=5.0, sigma_subtract=3.0, min_area=4)

    image = open_image(files[0])
    print(f'{files[0]}: {image.shape}', flush=True)
    positions, fluxes = detect(image, options)
    print(f'{len(positions)} detections', flush=True)

    stars, summary = psf.measure_field(image, positions,
                                       platescale_arcsec=args.platescale)
    print(f'{summary["n_stars"]} usable stars; excluded {summary["excluded"]}', flush=True)
    if not stars:
        raise SystemExit('no usable stars')

    fwhms = np.array([s['fwhm'] for s in stars])
    ells = np.array([s['ellipticity'] for s in stars])
    mags = -2.5 * np.log10(np.maximum([s['flux'] for s in stars], 1.0))

    lines = [f'frame                : {files[0]}',
             f'shape                : {image.shape}',
             f'detections           : {len(positions)}',
             f'usable stars         : {summary["n_stars"]}   excluded {summary["excluded"]}',
             f'median FWHM          : {summary["fwhm_px"]:.2f} px'
             + (f' = {summary["fwhm_arcsec"]:.2f}"' if args.platescale else ''),
             f'FWHM scatter (robust): {summary["fwhm_px_scatter"]:.2f} px',
             f'median ellipticity   : {summary["ellipticity"]:.3f}',
             f'sampling             : '
             + ('UNDERSAMPLED (FWHM < 2 px): pixel-phase bias territory; ePSF is the '
                'honest centroider' if summary['undersampled'] else
                'adequately sampled (FWHM >= 2 px): windowed centroid ~ fits')]

    # ---- constant vs varying, quantified
    lines.append('')
    lines.append('constant vs varying PSF (rms residual of each model):')
    for quantity, label in (('fwhm', 'FWHM (px)'), ('e1', 'e1'), ('e2', 'e2')):
        models = polynomial_maps(stars, image.shape, quantity)
        gain = (1 - models['quadratic'] / models['constant']) * 100 if \
            models['constant'] > 0 else 0
        lines.append(f'  {label:10s}: constant {models["constant"]:.4f}  '
                     f'linear {models["linear"]:.4f}  quadratic {models["quadratic"]:.4f}'
                     f'   (quadratic explains {gain:.0f}% of the constant-model scatter)')

    # ---- the stacked profile: Gaussian vs Moffat
    profile, n_stacked = stacked_profile(image, stars)
    beta_line = 'stacked profile      : too few clean bright stars'
    if profile is not None:
        radii, values = radial(profile)
        moffat = psf.fit_radial_moffat(radii, values)
        gauss = psf.fit_gaussian(profile * 1e6, noise=1.0)   # scaled: fitter wants counts
        if moffat:
            _, alpha, beta, fwhm_moffat = moffat
            beta_line = (f'stacked profile      : {n_stacked} stars; Moffat beta '
                         f'{beta:.2f} (4.77 = pure seeing; lower = optics wings), '
                         f'FWHM {fwhm_moffat:.2f} px')
            if gauss:
                beta_line += f'; Gaussian-fit FWHM {gauss["fwhm"]:.2f} px'
    lines.append(beta_line)

    # ================================================================== figures
    # 1. FWHM and ellipticity distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(fwhms, bins=40, color='#4878cf')
    axes[0].axvline(2.0, color='#d62728', ls='--', lw=1.2, label='2 px sampling limit')
    axes[0].set_xlabel('FWHM (px)'); axes[0].set_ylabel('stars'); axes[0].legend()
    top = ('FWHM: median %.2f px' % summary['fwhm_px']) + \
        ((' = %.2f"' % summary['fwhm_arcsec']) if args.platescale else '')
    axes[0].set_title(top)
    axes[1].hist(ells, bins=40, color='#6acc65')
    axes[1].set_xlabel('ellipticity 1 - b/a'); axes[1].set_ylabel('stars')
    axes[1].set_title('ellipticity: median %.3f' % summary['ellipticity'])
    fig.tight_layout(); fig.savefig(args.out / 'distributions.png', dpi=140)
    plt.close(fig)

    # 2. FWHM against instrumental magnitude: saturation and fit quality by brightness
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(mags, fwhms, s=8, alpha=0.5, c='#4878cf')
    ax.set_xlabel('instrumental magnitude  (brighter →  left)')
    ax.set_ylabel('FWHM (px)')
    ax.set_title('a bend at the bright end = saturation reaching the fits')
    ax.invert_xaxis(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.out / 'fwhm_vs_magnitude.png', dpi=140)
    plt.close(fig)

    # 3. the profile itself, log scale, against both models
    if profile is not None:
        radii, values = radial(profile)
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.plot(radii, np.maximum(values, 1e-9), '.', ms=3, alpha=0.4,
                label=f'stacked profile ({n_stacked} stars)')
        grid = np.linspace(0.01, radii.max(), 300)
        if moffat:
            i0, alpha, beta, _ = moffat
            ax.plot(grid, i0 * (1 + (grid / alpha) ** 2) ** (-beta), '-',
                    label=f'Moffat  beta={beta:.2f}')
        if gauss:
            sigma = gauss['fwhm'] / 2.3548
            peak = float(values[np.argmin(radii)])
            ax.plot(grid, peak * np.exp(-grid ** 2 / (2 * sigma ** 2)), '--',
                    label='Gaussian (same core)')
        ax.set_yscale('log')
        ax.set_ylim(max(values.max() * 1e-5, 1e-9), values.max() * 2)
        ax.set_xlabel('radius (px)'); ax.set_ylabel('normalised intensity')
        ax.set_title('where the Gaussian dies and the Moffat lives: the wings')
        ax.legend(); ax.grid(alpha=0.25, which='both')
        fig.tight_layout(); fig.savefig(args.out / 'radial_profile.png', dpi=140)
        plt.close(fig)

    # 4. FWHM across the field + the whisker map (the lensing-survey diagnostic)
    ys = np.array([s['y'] for s in stars]); xs = np.array([s['x'] for s in stars])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    scatter = axes[0].scatter(xs, ys, c=fwhms, s=14, cmap='viridis')
    fig.colorbar(scatter, ax=axes[0], label='FWHM (px)')
    axes[0].set_title('FWHM across the field')
    e1 = np.array([s['e1'] for s in stars]); e2 = np.array([s['e2'] for s in stars])
    nbin = 6
    ybin = np.clip((ys / image.shape[0] * nbin).astype(int), 0, nbin - 1)
    xbin = np.clip((xs / image.shape[1] * nbin).astype(int), 0, nbin - 1)
    for j in range(nbin):
        for i in range(nbin):
            sel = (ybin == j) & (xbin == i)
            if sel.sum() < 3:
                continue
            me1, me2 = float(np.median(e1[sel])), float(np.median(e2[sel]))
            magnitude = np.hypot(me1, me2)
            angle = 0.5 * np.arctan2(me2, me1)
            cy = (j + 0.5) * image.shape[0] / nbin
            cx = (i + 0.5) * image.shape[1] / nbin
            length = magnitude * image.shape[1] / nbin * 4      # scaled to be visible
            axes[1].plot([cx - length * np.cos(angle), cx + length * np.cos(angle)],
                         [cy - length * np.sin(angle), cy + length * np.sin(angle)],
                         '-', color='#d62728', lw=2)
    axes[1].set_xlim(0, image.shape[1]); axes[1].set_ylim(0, image.shape[0])
    axes[1].set_title('ellipticity whiskers (direction and strength of elongation)')
    for ax in axes:
        ax.set_aspect('equal')
        ax.invert_yaxis()               # match the image, as everywhere else in the app
    fig.tight_layout(); fig.savefig(args.out / 'field_maps.png', dpi=140)
    plt.close(fig)

    # 5. FWHM against distance from field centre (focal-plane curvature shows here first)
    field_radius = np.hypot(ys - image.shape[0] / 2, xs - image.shape[1] / 2)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(field_radius, fwhms, s=8, alpha=0.5, c='#4878cf')
    ax.set_xlabel('distance from field centre (px)'); ax.set_ylabel('FWHM (px)')
    ax.set_title('field curvature / tilt shows as a trend here')
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.out / 'fwhm_vs_field_radius.png', dpi=140)
    plt.close(fig)

    # ---- persist everything the centroid evaluation will want
    with open(args.out / 'stars.json', 'w', encoding='utf-8') as fp:
        json.dump({'frame': files[0], 'platescale_arcsec': args.platescale,
                   'summary': summary, 'stars': stars}, fp)

    text = '\n'.join(lines)
    print('\n' + text)
    (args.out / 'summary.txt').write_text(text, encoding='utf-8')
    print(f'\nfigures written to {args.out}')


if __name__ == '__main__':
    main()
