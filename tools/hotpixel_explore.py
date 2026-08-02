"""
Can hot pixels be told from stars without a dark frame?

    python tools/hotpixel_explore.py "tests/data/fits/example_with_darks/070424_040415/*.fits" \
        --darks "tests/data/fits/example_with_darks/070424_050036 darks 10s/*.fits" \
        --out docs/bench/hotpix

The idea under test: a star is fixed to the **sky**, a hot pixel is fixed to the
**detector**, and a dithered sequence separates the two. Take a bright candidate site and
ask two questions of the other frames:

* is it still bright at the same *detector* pixel?  -> detector persistence
* is it still bright at the same *sky* position, i.e. offset by that frame's dither?
                                                    -> sky persistence

A star answers no, yes. A hot pixel answers yes, no. Nothing else in a star field does.

This matters because darks are not always taken, and when they are they are not always
usable -- the darks in the bundled example were shot 45 minutes late and run three times
hotter than the lights (see progress.md), which is precisely when you would want a method
that does not need them.

Where darks *are* available they are used here only as ground truth, never as input to the
statistic, so the discrimination can be scored rather than admired.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from mee2024.config import get_default_options            # noqa: E402
from mee2024.stacker_implementation import (               # noqa: E402
    attempt_align, get_centroids_blur, hot_pixel_mask, open_image)


def local_background(image, box=64):
    """A coarse background and noise map, so 'bright' means bright for its neighbourhood."""
    from scipy.ndimage import uniform_filter
    small = image[::4, ::4]
    median = float(np.median(small))
    sigma = 1.4826 * float(np.median(np.abs(small - median)))
    # a smooth background rather than a single number: vignetting and gradients are real
    background = uniform_filter(image, size=box)
    return background, max(sigma, 1e-6)


def frame_shifts(files, options):
    """Dither offsets of each frame relative to the first, from the pipeline's own aligner."""
    centroids = []
    for path in files:
        image = open_image(path)
        blank = np.zeros(image.shape, dtype=bool)
        found = get_centroids_blur((image, blank, blank), options=options)
        centroids.append(np.array([c[2] for c in found]))
        print(f'  {Path(path).name}: {len(centroids[-1])} centroids', flush=True)
    shifts = [(0.0, 0.0)]
    previous = (0, 0)
    for i in range(1, len(files)):
        _, _, _, shift, _ = attempt_align(centroids[0], centroids[i], options,
                                          guess=previous, framenum=i)
        shifts.append((0.0, 0.0) if shift is None else (float(shift[0]), float(shift[1])))
        previous = shifts[-1]
    return shifts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lights', help='glob for the light frames')
    ap.add_argument('--darks', default=None, help='glob for darks, used as ground truth only')
    ap.add_argument('--out', type=Path, default=Path('docs/bench/hotpix'))
    ap.add_argument('--sigma', type=float, default=20.0,
                    help='how far above local background a pixel must be to be a candidate')
    args = ap.parse_args()

    files = sorted(glob.glob(args.lights))
    if len(files) < 3:
        raise SystemExit('need at least three dithered frames for this to mean anything')
    args.out.mkdir(parents=True, exist_ok=True)

    options = get_default_options()
    options.update(flag_display=False, flag_display2=False, centroid_gaussian_subtract=True,
                   centroid_gaussian_thresh=5.0, sigma_subtract=3.0, min_area=4)

    print(f'{len(files)} light frames; measuring dither offsets...', flush=True)
    shifts = frame_shifts(files, options)
    for path, shift in zip(files, shifts):
        print(f'  {Path(path).name}: dy {shift[0]:+8.2f}  dx {shift[1]:+8.2f}')
    spread = max(np.hypot(s[0], s[1]) for s in shifts)
    print(f'largest offset from frame 0: {spread:.1f} px')

    # ---------------------------------------------------------------- ground truth
    truth = None
    if args.darks:
        dark_files = sorted(glob.glob(args.darks))
        print(f'\n{len(dark_files)} darks, for scoring only...', flush=True)
        master = np.zeros(open_image(dark_files[0]).shape, dtype=np.float64)
        for path in dark_files:
            master += open_image(path)
        master /= len(dark_files)
        truth = hot_pixel_mask(master, 20.0)
        print(f'dark-based mask: {int(np.sum(truth))} hot pixels')
        del master

    # ------------------------------------------------------------------ candidates
    print('\nfinding bright candidates on frame 0...', flush=True)
    first = open_image(files[0])
    background, sigma = local_background(first)
    excess = first - background
    candidate = excess > args.sigma * sigma
    rows, cols = np.nonzero(candidate)
    print(f'{len(rows)} candidate pixels at >{args.sigma:.0f} sigma '
          f'(sigma {sigma:.1f} ADU)')
    peak0 = excess[rows, cols]
    del first, excess, candidate

    # --------------------------------------------------- the two persistence measures
    # Sampled at the candidate sites only: the full cube would be 1.3 GB.
    n = len(rows)
    detector = np.empty((len(files), n), dtype=np.float32)
    sky = np.empty((len(files), n), dtype=np.float32)
    sky_nearest = np.empty((len(files), n), dtype=np.float32)
    for i, (path, shift) in enumerate(zip(files, shifts)):
        image = open_image(path)
        base, _ = local_background(image)
        excess = image - base
        detector[i] = excess[rows, cols]
        # frame i sees the sky position of (row, col) shifted by -shift. The dither is
        # sub-pixel, and rounding it to the nearest pixel is not good enough: on the steep
        # flank of a bright star a one-pixel error is a large error, so the weakest sample
        # across frames lands in a trough and the wings of bright stars look like hot
        # pixels. Interpolating removes that, and it is measured below rather than assumed.
        fr = np.clip(rows - shift[0], 0, image.shape[0] - 1.001)
        fc = np.clip(cols - shift[1], 0, image.shape[1] - 1.001)
        r0, c0 = np.floor(fr).astype(int), np.floor(fc).astype(int)
        wr, wc = fr - r0, fc - c0
        sky[i] = ((1 - wr) * (1 - wc) * excess[r0, c0]
                  + (1 - wr) * wc * excess[r0, c0 + 1]
                  + wr * (1 - wc) * excess[r0 + 1, c0]
                  + wr * wc * excess[r0 + 1, c0 + 1])
        sky_nearest[i] = excess[np.rint(fr).astype(int), np.rint(fc).astype(int)]
        print(f'  sampled {Path(path).name}', flush=True)
        del image, base, excess

    # persistence = the weakest appearance across frames, in units of the noise. A star
    # is weak at a fixed detector pixel because it moved away; a hot pixel is weak at a
    # fixed sky position for the same reason, mirrored.
    det_persist = np.min(detector, axis=0) / sigma
    sky_persist = np.min(sky, axis=0) / sigma
    sky_persist_nearest = np.min(sky_nearest, axis=0) / sigma
    score = det_persist - sky_persist       # positive: detector-fixed, i.e. hot

    np.savez_compressed(args.out / 'candidates.npz', rows=rows, cols=cols, peak0=peak0,
                        det_persist=det_persist, sky_persist=sky_persist,
                        sky_persist_nearest=sky_persist_nearest,
                        truth=None if truth is None else truth[rows, cols],
                        shifts=np.array(shifts))

    labels = truth[rows, cols] if truth is not None else np.zeros(n, dtype=bool)
    print(f'\ncandidates: {n}, of which {int(np.sum(labels))} are dark-confirmed hot')
    report(args.out, rows, cols, peak0, det_persist, sky_persist, score, labels, shifts,
           sigma, spread)


def report(out, rows, cols, peak0, det_persist, sky_persist, score, labels, shifts,
           sigma, spread):
    hot, star = labels, ~labels

    # ---- figure 1: the two persistence measures against each other
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(sky_persist[star], det_persist[star], s=6, alpha=0.35, c='#1f77b4',
               label=f'not hot per the darks ({int(np.sum(star))})')
    ax.scatter(sky_persist[hot], det_persist[hot], s=26, alpha=0.9, c='#d62728',
               marker='x', label=f'hot per the darks ({int(np.sum(hot))})')
    lim = [min(sky_persist.min(), det_persist.min()) - 2,
           max(np.percentile(sky_persist, 99.9), np.percentile(det_persist, 99.9)) + 2]
    ax.plot(lim, lim, '--', c='#888', lw=1, label='equal persistence')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('sky persistence  (weakest appearance at the same sky position, sigma)')
    ax.set_ylabel('detector persistence  (weakest appearance at the same pixel, sigma)')
    ax.set_title('A star is fixed to the sky; a hot pixel is fixed to the detector\n'
                 f'largest dither offset {spread:.0f} px, noise {sigma:.1f} ADU')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'persistence_scatter.png', dpi=140)
    plt.close(fig)

    # ---- figure 2: the discriminant, as a distribution
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    bins = np.linspace(np.percentile(score, 0.1), np.percentile(score, 99.9), 90)
    axes[0].hist(score[star], bins=bins, color='#1f77b4', alpha=0.85, label='not hot')
    axes[0].set_ylabel('candidates'); axes[0].set_yscale('log')
    axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].hist(score[hot], bins=bins, color='#d62728', alpha=0.85, label='hot')
    axes[1].set_ylabel('candidates'); axes[1].legend(); axes[1].grid(alpha=0.25)
    axes[1].set_xlabel('detector persistence minus sky persistence (sigma)')
    axes[0].set_title('The discriminant, without using a dark frame')
    fig.tight_layout()
    fig.savefig(out / 'discriminant_hist.png', dpi=140)
    plt.close(fig)

    # ---- figure 3: how well does a threshold on it actually do?
    lines = []
    if hot.any():
        order = np.argsort(-score)
        is_hot = hot[order]
        tp = np.cumsum(is_hot)
        fp = np.cumsum(~is_hot)
        recall = tp / max(int(hot.sum()), 1)
        precision = tp / np.maximum(tp + fp, 1)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.plot(recall, precision, c='#2c7', lw=2)
        ax.set_xlabel('recall: dark-confirmed hot pixels found')
        ax.set_ylabel('precision: of those flagged, the fraction really hot')
        ax.set_title('Ranking candidates by the dark-free discriminant')
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.02)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / 'precision_recall.png', dpi=140)
        plt.close(fig)

        for target in (0.5, 0.8, 0.9, 0.95, 0.99):
            reached = np.nonzero(recall >= target)[0]
            if reached.size:
                k = reached[0]
                lines.append(f'  recall {target:4.0%}: precision {precision[k]:6.1%}, '
                             f'threshold {score[order][k]:8.1f} sigma, '
                             f'{int(fp[k])} false positives')
        # and the operating point a pipeline would actually pick: no false positives
        clean = np.nonzero(np.cumsum(~is_hot) == 0)[0]
        if clean.size:
            k = clean[-1]
            lines.append(f'  at zero false positives: recall {recall[k]:6.1%} '
                         f'({int(tp[k])} of {int(hot.sum())}), threshold '
                         f'{score[order][k]:.1f} sigma')

    summary = [
        f'candidates                : {len(rows)}',
        f'dark-confirmed hot        : {int(hot.sum())}',
        f'largest dither offset     : {spread:.1f} px',
        f'background noise          : {sigma:.1f} ADU',
        '',
        'median detector persistence:',
        f'  hot       : {np.median(det_persist[hot]) if hot.any() else float("nan"):8.1f} sigma',
        f'  not hot   : {np.median(det_persist[star]):8.1f} sigma',
        'median sky persistence:',
        f'  hot       : {np.median(sky_persist[hot]) if hot.any() else float("nan"):8.1f} sigma',
        f'  not hot   : {np.median(sky_persist[star]):8.1f} sigma',
        '',
        'separation by a threshold on (detector - sky) persistence:',
    ] + (lines or ['  no ground truth supplied'])
    text = '\n'.join(summary)
    print('\n' + text)
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    print(f'\nfigures and summary written to {out}')


if __name__ == '__main__':
    main()
