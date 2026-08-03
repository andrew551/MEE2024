"""
Which centroiding algorithm should this pipeline use? Measure, don't argue.

    python tools/centroid_eval.py synthetic --out docs/bench/psf/centroids
    python tools/centroid_eval.py real "tests/data/fits/example_with_darks/070424_040415/*.fits" --out docs/bench/psf/centroids

Two protocols, because each can lie alone:

* **synthetic** — frames of pixel-integrated Moffat stars (β=3, the measured profile) at
  the three FWHM regimes the real datasets span (1.3, 2.4, 3.0 px), with the truth known
  exactly. Reports rms error against SNR per estimator. Synthetic truth can flatter a
  method whose model matches the simulation, which is why there is also:
* **real** — the 7-frame dithered eclipse set. Each estimator measures the same stars on
  every frame; after removing a per-frame translation, the per-star scatter across frames
  is that estimator's repeatability on genuine pixels. Atmospheric differential motion
  hits all estimators equally, so *differences* are estimator noise.

Estimators, cheapest first:

  pipeline   what stage 1 does today (thresholded, variance-normalised COM)
  com        plain centre of mass over the background-subtracted cutout
  windowed   iterative Gaussian-weighted centroid (SExtractor XWIN style)
  gauss      pixel-integrated elliptical Gaussian least squares
  epsf       Anderson & King effective PSF (photutils), built from the frame's own stars
"""

import argparse
import glob
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mee2024 import psf  # noqa: E402

CUT = psf.CUT


# ------------------------------------------------------------------ estimators
# Each takes a background-subtracted cutout and returns (cy, cx) in cutout
# coordinates, or None. The frame-level pipeline estimator is handled separately.

def centroid_com(data):
    data = np.maximum(np.asarray(data, dtype=float), 0)
    total = data.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(data.shape)
    return float((data * ys).sum() / total), float((data * xs).sum() / total)


def centroid_windowed(data):
    moments = psf.windowed_moments(data)
    return None if moments is None else (moments[0], moments[1])


def centroid_gauss(data):
    fit = psf.fit_gaussian(data)
    return None if fit is None else (fit['cy'], fit['cx'])


class EpsfCentroider:
    """Anderson & King, via photutils: build once per frame from its brightest stars,
    then fit every cutout against the oversampled empirical model."""

    def __init__(self, image, positions, oversampling=4, max_build_stars=50):
        from astropy.nddata import NDData
        from astropy.table import Table
        from photutils.psf import EPSFBuilder, extract_stars

        cutouts = psf.extract_cutouts(image, positions)
        clean = [c for c in cutouts if c['isolated'] and not c['saturated']]
        clean.sort(key=lambda c: -c['flux'])
        rows = [(c['origin'][1] + CUT, c['origin'][0] + CUT)
                for c in clean[:max_build_stars]]
        table = Table(rows=rows, names=('x', 'y'))
        # EPSFBuilder expects background-subtracted inputs; built on raw pixels the model
        # carries the sky as a pedestal and every subsequent fit is biased by it —
        # measured at 0.2-0.5 px of error before this subtraction, which is not the
        # method's fault but the caller's
        image = np.asarray(image, dtype=float)
        sky = float(np.median(image[::8, ::8]))
        data = NDData(image - sky)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stars = extract_stars(data, table, size=2 * CUT + 1)
            builder = EPSFBuilder(oversampling=oversampling, maxiters=12,
                                  progress_bar=False)
            self.model, _ = builder(stars)
        self.n_built = len(stars)

    def __call__(self, data):
        from astropy.modeling.fitting import TRFLSQFitter

        model = self.model.copy()
        size = data.shape[0]
        start = centroid_com(data)
        if start is None:
            return None
        model.x_0, model.y_0 = start[1], start[0]
        model.flux = float(np.sum(data))
        ys, xs = np.indices(data.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fitted = TRFLSQFitter()(model, xs, ys, np.asarray(data, dtype=float),
                                        maxiter=200)
            except Exception:
                return None
        cy, cx = float(fitted.y_0.value), float(fitted.x_0.value)
        if not (0 <= cy < size and 0 <= cx < size):
            return None
        return cy, cx


def pipeline_positions(image):
    """What stage 1's detector would report for this frame, as an (n, 2) array."""
    from mee2024.config import get_default_options
    from mee2024.stacker_implementation import get_centroids_blur

    options = get_default_options()
    options.update(flag_display=False, centroid_gaussian_subtract=True,
                   centroid_gaussian_thresh=5.0, sigma_subtract=3.0, min_area=4)
    blank = np.zeros(image.shape, dtype=bool)
    found = get_centroids_blur((image, blank, blank), options=options)
    return np.array([c[2] for c in found]) if found else np.zeros((0, 2))


def match_nearest(estimates, truths, radius=2.5):
    """index of the estimate nearest each truth, or -1. One-to-one not enforced: at these
    densities double-assignment is negligible and the error metric is robust to it."""
    if not len(estimates):
        return np.full(len(truths), -1)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(estimates)
    distance, index = nn.kneighbors(truths)
    matched = np.where(distance[:, 0] <= radius, index[:, 0], -1)
    return matched


# ------------------------------------------------------------------- synthetic

def synthetic_frame(fwhm, fluxes, size=1200, beta=3.0, sky=150.0, noise=6.0, seed=0):
    """A frame of pixel-integrated Moffat stars at random subpixel phases."""
    rng = np.random.default_rng(seed)
    n = len(fluxes)
    margin = 3 * CUT
    truths = np.c_[rng.uniform(margin, size - margin, n),
                   rng.uniform(margin, size - margin, n)]
    # enforce isolation so the metric measures centroiding, not deblending
    from sklearn.neighbors import NearestNeighbors
    keep = np.ones(n, dtype=bool)
    nn = NearestNeighbors(n_neighbors=2).fit(truths)
    distance, _ = nn.kneighbors(truths)
    keep &= distance[:, 1] > 4 * CUT
    truths, fluxes = truths[keep], np.asarray(fluxes)[keep]

    image = rng.normal(sky, noise, (size, size))
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    stamp = 3 * CUT
    fine_step = 0.2
    for (cy, cx), flux in zip(truths, fluxes):
        row, col = int(cy), int(cx)
        grid = np.arange(-stamp - 0.5 + fine_step / 2, stamp + 0.5, fine_step)
        xs, ys = np.meshgrid(grid + (col - cx), grid + (row - cy))
        model = (1 + (np.hypot(xs, ys) / alpha) ** 2) ** (-beta)
        binned = model.reshape(2 * stamp + 1, 5, 2 * stamp + 1, 5).mean(axis=(1, 3))
        binned *= flux / binned.sum()
        image[row - stamp:row + stamp + 1, col - stamp:col + stamp + 1] += \
            rng.poisson(np.maximum(binned, 0))    # photon noise on the star itself
    return image, truths, fluxes


def run_synthetic(out):
    results = {}
    flux_ladder = np.repeat([3e2, 1e3, 3e3, 1e4, 3e4, 1e5], 60).astype(float)
    for fwhm in (1.3, 2.4, 3.0):
        image, truths, fluxes = synthetic_frame(fwhm, flux_ladder, seed=int(fwhm * 10))
        print(f'FWHM {fwhm}: {len(truths)} stars', flush=True)
        cutouts = psf.extract_cutouts(image, truths, isolation_px=0)
        by_index = {c['index']: c for c in cutouts}

        estimators = {'com': centroid_com, 'windowed': centroid_windowed,
                      'gauss': centroid_gauss}
        try:
            estimators['epsf'] = EpsfCentroider(image, truths)
            print(f'  ePSF built from {estimators["epsf"].n_built} stars', flush=True)
        except Exception as exc:
            print(f'  ePSF unavailable: {exc}', flush=True)

        table = {name: [] for name in list(estimators) + ['pipeline']}
        for index, truth in enumerate(truths):
            cutout = by_index.get(index)
            if cutout is None:
                continue
            row0, col0 = cutout['origin']
            for name, estimator in estimators.items():
                got = estimator(cutout['data'])
                if got is not None:
                    err = np.hypot(row0 + got[0] - truth[0], col0 + got[1] - truth[1])
                    table[name].append((float(fluxes[index]), float(err)))

        found = pipeline_positions(image)
        matched = match_nearest(found, truths)
        for index, truth in enumerate(truths):
            if matched[index] >= 0:
                err = float(np.hypot(*(found[matched[index]] - truth)))
                table['pipeline'].append((float(fluxes[index]), err))
        results[f'fwhm_{fwhm}'] = table
    with open(out / 'synthetic.json', 'w', encoding='utf-8') as fp:
        json.dump(results, fp)
    return results


# ------------------------------------------------------------------------ real

def run_real(pattern, out):
    from mee2024.config import get_default_options
    from mee2024.stacker_implementation import attempt_align, open_image

    files = sorted(glob.glob(pattern))
    if len(files) < 3:
        raise SystemExit('need at least three dithered frames')
    options = get_default_options()
    options.update(flag_display=False)
    frames = [open_image(f) for f in files]
    print(f'{len(frames)} frames', flush=True)

    detections = [pipeline_positions(image) for image in frames]
    print('detections per frame:', [len(d) for d in detections], flush=True)
    reference = detections[0]
    shifts = [(0.0, 0.0)]
    for i in range(1, len(frames)):
        _, _, _, shift, _ = attempt_align(reference, detections[i], options, framenum=i)
        shifts.append((float(shift[0]), float(shift[1])))

    # keep only clean reference stars, then measure them on every frame
    cutouts = psf.extract_cutouts(frames[0], reference)
    clean = [c for c in cutouts if c['isolated'] and not c['saturated']]
    print(f'{len(clean)} clean reference stars', flush=True)

    estimators = {'com': centroid_com, 'windowed': centroid_windowed,
                  'gauss': centroid_gauss}
    try:
        estimators['epsf'] = EpsfCentroider(frames[0], reference)
        print(f'ePSF built from {estimators["epsf"].n_built} stars', flush=True)
    except Exception as exc:
        print(f'ePSF unavailable: {exc}', flush=True)

    # positions[name][star][frame] = (y, x) in frame coordinates, minus the dither shift
    collected = {name: [] for name in estimators}
    fluxes = []
    for cutout in clean:
        ref_y = cutout['origin'][0] + CUT
        ref_x = cutout['origin'][1] + CUT
        fluxes.append(cutout['flux'])
        per_estimator = {name: [] for name in estimators}
        for image, (sy, sx) in zip(frames, shifts):
            row = int(round(ref_y - sy))
            col = int(round(ref_x - sx))
            if not (CUT <= row < image.shape[0] - CUT and
                    CUT <= col < image.shape[1] - CUT):
                continue
            data = np.asarray(image[row - CUT:row + CUT + 1,
                                    col - CUT:col + CUT + 1], dtype=float)
            ring = np.concatenate([data[0, :], data[-1, :], data[1:-1, 0], data[1:-1, -1]])
            data = data - np.median(ring)
            for name, estimator in estimators.items():
                got = estimator(data)
                per_estimator[name].append(
                    None if got is None else (row + got[0] + sy, col + got[1] + sx))
        for name in estimators:
            collected[name].append(per_estimator[name])

    # the pipeline estimator: its own detections per frame, matched to the reference
    pipeline_tracks = []
    for cutout in clean:
        pipeline_tracks.append([])
    for image_index, (dets, (sy, sx)) in enumerate(zip(detections, shifts)):
        shifted = dets + np.array([[sy, sx]])
        matched = match_nearest(shifted,
                                [(c['origin'][0] + CUT, c['origin'][1] + CUT)
                                 for c in clean], radius=3.0)
        for star_index, hit in enumerate(matched):
            pipeline_tracks[star_index].append(
                None if hit < 0 else (float(shifted[hit][0]), float(shifted[hit][1])))
    collected['pipeline'] = pipeline_tracks

    # per-frame translation removed per estimator (its own systematic), then per-star rms
    summary = {}
    per_star = {}
    for name, tracks in collected.items():
        rows = []
        for track in tracks:
            points = [p for p in track if p is not None]
            if len(points) < max(3, len(frames) - 2):
                rows.append(None)
                continue
            rows.append(np.array(points))
        # translation per frame: median offset of all stars measured on that frame
        offsets = []
        for f in range(len(frames)):
            deltas = []
            for track in tracks:
                if f < len(track) and track[f] is not None:
                    points = [p for p in track if p is not None]
                    mean = np.mean(np.array(points), axis=0)
                    deltas.append(np.array(track[f]) - mean)
            offsets.append(np.median(np.array(deltas), axis=0) if deltas else np.zeros(2))
        scatters = []
        for track in tracks:
            adjusted = [np.array(p) - offsets[f]
                        for f, p in enumerate(track) if p is not None]
            if len(adjusted) < max(3, len(frames) - 2):
                scatters.append(None)
                continue
            adjusted = np.array(adjusted)
            centre = adjusted.mean(axis=0)
            scatters.append(float(np.sqrt(np.mean(np.sum((adjusted - centre) ** 2,
                                                          axis=1)))))
        per_star[name] = scatters
        valid = [s for s in scatters if s is not None]
        summary[name] = {'n': len(valid),
                         'median_px': float(np.median(valid)) if valid else None,
                         'p90_px': float(np.percentile(valid, 90)) if valid else None}
    with open(out / 'real.json', 'w', encoding='utf-8') as fp:
        json.dump({'files': files, 'summary': summary,
                   'fluxes': fluxes, 'per_star': per_star}, fp)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['synthetic', 'real'])
    ap.add_argument('pattern', nargs='?', help='frame glob (real mode)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mode == 'synthetic':
        results = run_synthetic(args.out)
        print(f'\n== synthetic truth: rms centroid error (px) by SNR regime ==')
        for regime, table in results.items():
            print(f'\n{regime}')
            print(f'  {"estimator":10s} {"faint rms":>10} {"mid rms":>10} {"bright rms":>11}  n')
            for name, pairs in table.items():
                if not pairs:
                    continue
                arr = np.array(pairs)
                faint = arr[arr[:, 0] <= 1e3][:, 1]
                mid = arr[(arr[:, 0] > 1e3) & (arr[:, 0] <= 1e4)][:, 1]
                bright = arr[arr[:, 0] > 1e4][:, 1]
                fmt = lambda v: f'{np.sqrt(np.mean(v**2)):.4f}' if len(v) else '     -'
                print(f'  {name:10s} {fmt(faint):>10} {fmt(mid):>10} {fmt(bright):>11}  '
                      f'{len(arr)}')
    else:
        summary = run_real(args.pattern, args.out)
        print(f'\n== real frames: per-star scatter across the dither (px rms) ==')
        for name, row in sorted(summary.items(), key=lambda kv: kv[1]['median_px'] or 9):
            print(f'  {name:10s} median {row["median_px"]:.4f}  p90 {row["p90_px"]:.4f}  '
                  f'(n={row["n"]})')
    print(f'\n{time.time() - t0:.0f}s; results in {args.out}')


if __name__ == '__main__':
    main()
