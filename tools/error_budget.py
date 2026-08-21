"""
Where does the centroid error actually come from? An error budget, measured.

    python tools/error_budget.py "I:/MEE test frames/fits/example_with_darks/070424_040415/*.fits" \
        --darks "I:/MEE test frames/fits/example_with_darks/070424_050036 darks 10s/*.fits" \
        --out docs/bench/psf/budget_eclipse
    python tools/error_budget.py "I:/MEE test frames/fits/00_23_49/Zenith_*.fits" --out docs/bench/psf/budget_zenith

Each candidate limit has its own signature, so each gets its own measurement:

* **photon + pixel noise** — the Cramér–Rao bound per star, computed from its own fitted
  profile, the measured background noise, and a gain measured from the frames themselves
  (photon transfer on frame differences). Faint stars must ride this curve; a bright-star
  plateau *above* it is, by construction, everything that is not pixel noise.
* **atmosphere** — differential tip-tilt is *spatially correlated* between stars; pixel
  noise is white. The two-point correlation of the per-frame residual fields splits the
  plateau into a correlated (atmospheric) part and a white part.
* **mount / field rotation** — remove a translation per frame, then a full affine; the
  scatter the affine removes on top of translation is rotation/scale wobble.
* **pixel size (undersampling)** — bin residuals against subpixel phase; a systematic
  dependence is pixel-phase bias, the undersampling signature.
* **algorithm** — bounded separately in docs/bench/CENTROIDS.md: three unrelated
  estimators tie on these frames, so the estimator choice contributes ~nothing.
* **optics (static)** — absent from frame-to-frame scatter entirely; it lives in the
  stage-2 fit residual, compared against the stacked centroid noise elsewhere.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from mee2024 import psf  # noqa: E402

CUT = psf.CUT


# ------------------------------------------------------------- gain measurement

def measure_gain(frames, dark_frames=None):
    """(gain e-/ADU, read-ish noise ADU), from photon transfer on frame differences.

    Two frames of the same scene differ only by noise; var(A-B)/2 at mean level m obeys
    sigma^2 = m/g + r^2. Star pixels are sigma-clipped away so only the sky participates.
    With darks available there is a second, different mean level, which separates g from
    r; without them the fit assumes r is small at sky level (true of modern CMOS) and the
    result carries that caveat.
    """
    def level_and_variance(a, b):
        difference = (a - b).ravel()
        mean = ((a + b) / 2).ravel()
        # sigma-clip in the difference: stars, hot pixels and cosmic hits all leave
        centre = np.median(difference)
        spread = 1.4826 * np.median(np.abs(difference - centre))
        keep = np.abs(difference - centre) < 5 * spread
        return float(np.median(mean[keep])), float(np.var(difference[keep]) / 2)

    points = []
    for a, b in zip(frames[:-1], frames[1:]):
        points.append(level_and_variance(a, b))
    if dark_frames and len(dark_frames) >= 2:
        for a, b in zip(dark_frames[:-1], dark_frames[1:]):
            points.append(level_and_variance(a, b))
    points = np.array(points)
    caveat = ''
    if len(np.unique(np.round(points[:, 0]))) >= 2:
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        gain = 1.0 / max(slope, 1e-9)
        read = float(np.sqrt(max(intercept, 0.0)))
        if not 0.05 <= gain <= 20:
            # A negative or absurd slope means the level axis is corrupted -- typically a
            # bias pedestal, or darks whose level exceeds the sky (ours run hot). A wrong
            # gain silently miscalibrates the bright-star bound, so refuse instead.
            caveat = (f'photon-transfer fit implausible (gain {gain:.3g}); levels are '
                      f'pedestal-corrupted. Shot-noise bound uses an assumed 1 e-/ADU.')
            gain = 1.0
            read = float(np.sqrt(points[:, 1].mean()))
    else:
        level, variance = points.mean(axis=0)
        gain = level / max(variance, 1e-9)
        read = 0.0
        caveat = ('single sky level only: gain assumes no read noise and an unknown bias '
                  'pedestal inflates it; treat the shot term as indicative')
        if not 0.05 <= gain <= 20:
            gain, caveat = 1.0, caveat + ' (clamped to 1 e-/ADU)'
    return float(gain), read, points, caveat


# --------------------------------------------------------- Cramér–Rao per star

def cramer_rao_px(amplitude_adu, sigma_px, sky_noise_adu, gain=None):
    """The per-axis position bound for a Gaussian star, evaluated numerically.

    Fisher information per axis: sum over pixels of (d mu/dx)^2 / var, with mu in
    electrons and var = mu + (g*sigma_sky)^2. No closed form needed; the pixel grid is
    tiny. This is the floor no estimator may beat, and three estimators tying just above
    it is what 'the algorithm is not the limit' looks like.
    """
    half = 3 * max(sigma_px, 0.8)
    grid = np.arange(-int(np.ceil(half)), int(np.ceil(half)) + 1)
    xs, ys = np.meshgrid(grid, grid)
    # gain=None gives the background-only bound, in which the gain cancels exactly: it is
    # the most optimistic pixel-noise floor possible, fully measured, no calibration
    # required. The gain-aware version adds the star shot noise on top.
    profile = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma_px ** 2))
    if gain is None:
        dmu_dx = amplitude_adu * profile * (-xs / sigma_px ** 2)
        fisher = np.sum(dmu_dx ** 2) / max(sky_noise_adu, 1e-9) ** 2
    else:
        mu = gain * amplitude_adu * profile
        dmu_dx = mu * (-xs / sigma_px ** 2)
        variance = mu + (gain * sky_noise_adu) ** 2
        fisher = np.sum(dmu_dx ** 2 / np.maximum(variance, 1e-9))
    return float(1.0 / np.sqrt(max(fisher, 1e-12)))


# ------------------------------------------------------------ track collection

def collect_tracks(frames, shifts, reference_cutouts):
    """Windowed-centroid positions of every clean star on every frame.

    The windowed centroid is used because deliverable (d) measured it indistinguishable
    from the pipeline's own estimator, and it runs standalone on a cutout.
    """
    tracks = []
    for cutout in reference_cutouts:
        ref_y = cutout['origin'][0] + CUT
        ref_x = cutout['origin'][1] + CUT
        track = []
        for image, (sy, sx) in zip(frames, shifts):
            row = int(round(ref_y - sy))
            col = int(round(ref_x - sx))
            if not (CUT <= row < image.shape[0] - CUT and
                    CUT <= col < image.shape[1] - CUT):
                track.append(None)
                continue
            data = np.asarray(image[row - CUT:row + CUT + 1,
                                    col - CUT:col + CUT + 1], dtype=float)
            ring = np.concatenate([data[0, :], data[-1, :],
                                   data[1:-1, 0], data[1:-1, -1]])
            data = data - np.median(ring)
            got = psf.windowed_moments(data)
            track.append(None if got is None else
                         (row + got[0] + sy, col + got[1] + sx))
        tracks.append(track)
    return tracks


def residual_fields(tracks, model='translation'):
    """Per-frame residuals after removing a per-frame alignment model.

    Returns (residuals[frame] -> list of (star_index, dy, dx), per_star_scatter).
    'translation' removes the median offset; 'affine' also removes rotation, scale and
    shear, fitted to the same stars — the difference between the two *is* the mount term.
    """
    n_frames = max(len(t) for t in tracks)
    means = []
    for track in tracks:
        points = [p for p in track if p is not None]
        means.append(np.mean(np.array(points), axis=0) if len(points) >= 3 else None)

    fields = []
    for f in range(n_frames):
        deltas, positions, indices = [], [], []
        for i, track in enumerate(tracks):
            if means[i] is None or f >= len(track) or track[f] is None:
                continue
            deltas.append(np.array(track[f]) - means[i])
            positions.append(means[i])
            indices.append(i)
        deltas = np.array(deltas)
        positions = np.array(positions)
        if len(deltas) < 8:
            fields.append([])
            continue
        if model == 'translation':
            fitted = np.tile(np.median(deltas, axis=0), (len(deltas), 1))
        else:
            design = np.c_[positions, np.ones(len(positions))]
            fitted = np.empty_like(deltas)
            for axis in range(2):
                coeff, *_ = np.linalg.lstsq(design, deltas[:, axis], rcond=None)
                fitted[:, axis] = design @ coeff
        residual = deltas - fitted
        fields.append([(i, r[0], r[1], p[0], p[1])
                       for i, r, p in zip(indices, residual, positions)])

    scatter = {}
    per_star = {}
    for field in fields:
        for i, dy, dx, *_ in field:
            per_star.setdefault(i, []).append((dy, dx))
    for i, rs in per_star.items():
        rs = np.array(rs)
        if len(rs) >= 3:
            scatter[i] = float(np.sqrt(np.mean(np.sum(rs ** 2, axis=1))))
    return fields, scatter


def correlation_by_separation(fields, positions, bins):
    """<r_i . r_j>/2 for star pairs, binned by their separation, averaged over frames.

    Atmospheric tip-tilt gives positive short-range correlation decaying with distance;
    pixel noise gives zero everywhere off zero. The units are px^2 per axis.
    """
    sums = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1, dtype=int)
    for field in fields:
        if len(field) < 2:
            continue
        arr = np.array([(dy, dx, py, px_) for _, dy, dx, py, px_ in field])
        residual = arr[:, :2]
        position = arr[:, 2:]
        n = len(arr)
        if n > 400:                        # pair count control
            chosen = np.random.default_rng(0).choice(n, 400, replace=False)
            residual, position = residual[chosen], position[chosen]
            n = 400
        diff = position[:, None, :] - position[None, :, :]
        separation = np.hypot(diff[..., 0], diff[..., 1])
        dot = (residual[:, None, :] * residual[None, :, :]).sum(axis=2) / 2
        upper = np.triu_indices(n, k=1)
        which = np.digitize(separation[upper], bins) - 1
        for b in range(len(bins) - 1):
            sel = which == b
            sums[b] += dot[upper][sel].sum()
            counts[b] += int(sel.sum())
    with np.errstate(invalid='ignore'):
        return sums / np.maximum(counts, 1), counts


# --------------------------------------------------- does scatter track FWHM?

def fwhm_tracking(rows, image_shape, grid=5, min_cell=4):
    """Do stars with a locally worse PSF also centroid worse, frame to frame?

    Uses the affine-removed scatter: a rotation residual grows with field radius and
    optical FWHM usually does too, so translation-only scatter would correlate with
    FWHM through the mount term alone. Returns per-star rank correlations and a pair of
    grid maps (median FWHM, median scatter) with their cell-level correlation.
    """
    from scipy import stats

    usable = [r for r in rows if r['fwhm'] is not None and r['scatter_a'] is not None]
    result = {'n': len(usable)}
    if len(usable) < 20:
        return result, None
    fwhm = np.array([r['fwhm'] for r in usable])
    scatter = np.array([r['scatter_a'] for r in usable])
    excess = scatter / np.array([r['cr_2d'] for r in usable])
    flux = np.array([r['flux'] for r in usable])
    bright = flux > np.percentile(flux, 50)

    rho, p = stats.spearmanr(fwhm[bright], scatter[bright])
    rho_x, p_x = stats.spearmanr(fwhm[bright], excess[bright])
    result.update(n_bright=int(bright.sum()),
                  spearman_scatter=(float(rho), float(p)),
                  spearman_excess=(float(rho_x), float(p_x)))

    ys = np.array([r['y'] for r in usable])
    xs = np.array([r['x'] for r in usable])
    cell_y = np.clip((ys / image_shape[0] * grid).astype(int), 0, grid - 1)
    cell_x = np.clip((xs / image_shape[1] * grid).astype(int), 0, grid - 1)
    map_fwhm = np.full((grid, grid), np.nan)
    map_scatter = np.full((grid, grid), np.nan)
    for cy in range(grid):
        for cx in range(grid):
            sel = (cell_y == cy) & (cell_x == cx)
            if sel.sum() >= min_cell:
                map_fwhm[cy, cx] = np.median(fwhm[sel])
                map_scatter[cy, cx] = np.median(scatter[sel])
    both = ~np.isnan(map_fwhm) & ~np.isnan(map_scatter)
    if both.sum() >= 6:
        r_cell, p_cell = stats.pearsonr(map_fwhm[both], map_scatter[both])
        result.update(cell_r=(float(r_cell), float(p_cell)), n_cells=int(both.sum()))
    return result, (map_fwhm, map_scatter)


# ------------------------------------------- how does the rms stack down with N?

def stack_scaling(fields, n_frames, bright_indices):
    """2-D rms of N-frame-averaged positions, for every N with at least two groups.

    Frames are split into disjoint groups of N; each star's group-mean positions are
    scattered against each other (ddof=1, so the estimate is unbiased however few
    groups there are) and the median over bright stars is the curve point. 'consecutive'
    keeps frames in time order, so drift that a per-frame model missed shows up as a
    departure from 1/sqrt(N); 'shuffled' breaks time order, so the comparison of the
    two isolates temporal correlation from everything else.
    """
    series = {}
    for f, field in enumerate(fields):
        for i, dy, dx, *_ in field:
            series.setdefault(i, {})[f] = (dy, dx)

    orders = {'consecutive': np.arange(n_frames),
              'shuffled': np.random.default_rng(1).permutation(n_frames)}
    curves = {}
    for name, order in orders.items():
        curve = []
        for n_in_group in range(1, n_frames // 2 + 1):
            n_groups = n_frames // n_in_group
            groups = [order[g * n_in_group:(g + 1) * n_in_group]
                      for g in range(n_groups)]
            per_star = []
            for i in bright_indices:
                s = series.get(i, {})
                means = []
                for group in groups:
                    points = [s[f] for f in group if f in s]
                    if len(points) == len(group):     # complete groups only
                        means.append(np.mean(points, axis=0))
                if len(means) >= 2:
                    means = np.array(means)
                    deviation = means - means.mean(axis=0)
                    per_star.append(np.sqrt(np.sum(deviation ** 2)
                                            / (len(means) - 1)))
            if len(per_star) >= 10:
                curve.append({'n': n_in_group, 'rms': float(np.median(per_star)),
                              'stars': len(per_star)})
        curves[name] = curve
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pattern')
    ap.add_argument('--darks', default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--platescale', type=float, default=None)
    ap.add_argument('--gain', type=float, default=None,
                    help='e-/ADU from the camera header, when photon transfer is '
                         'degenerate (e.g. 0.2 s frames whose sky adds nothing '
                         'above the dark level)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from mee2024.config import get_default_options
    from mee2024.stacker_implementation import attempt_align, open_image
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from centroid_eval import pipeline_positions

    files = sorted(glob.glob(args.pattern))
    frames = [open_image(f) for f in files]
    print(f'{len(frames)} frames of {frames[0].shape}', flush=True)
    dark_frames = ([open_image(f) for f in sorted(glob.glob(args.darks))[:4]]
                   if args.darks else None)

    # ---- gain, from the data itself (or the header, when the data cannot say)
    gain, read_adu, points, gain_caveat = measure_gain(frames[:4], dark_frames)
    if args.gain is not None:
        gain, gain_caveat = args.gain, f'gain {args.gain} e-/ADU supplied (camera header)'
    print(f'photon transfer: gain {gain:.3f} e-/ADU, base noise {read_adu:.2f} ADU '
          f'({len(points)} frame pairs)'
          + (f'  [{gain_caveat}]' if gain_caveat else ''), flush=True)

    # ---- tracks
    options = get_default_options()
    options.update(flag_display=False)
    detections = [pipeline_positions(image) for image in frames]
    reference = detections[0]
    shifts = [(0.0, 0.0)]
    for i in range(1, len(frames)):
        _, _, _, shift, _ = attempt_align(reference, detections[i], options, framenum=i)
        shifts.append((float(shift[0]), float(shift[1])))
    cutouts = psf.extract_cutouts(frames[0], reference)
    clean = [c for c in cutouts if c['isolated'] and not c['saturated']]
    print(f'{len(clean)} clean stars', flush=True)
    tracks = collect_tracks(frames, shifts, clean)

    # ---- per-star Cramér–Rao from each star's own fit
    stars, summary = psf.measure_field(frames[0], reference, fit_top_n=None)
    by_position = {}
    for s in stars:
        by_position[(int(round(s['y'])), int(round(s['x'])))] = s
    sigma_med = summary['fwhm_px'] / 2.3548

    fields_t, scatter_t = residual_fields(tracks, 'translation')
    fields_a, scatter_a = residual_fields(tracks, 'affine')

    rows = []
    for i, cutout in enumerate(clean):
        if i not in scatter_t:
            continue
        key = (cutout['origin'][0] + CUT, cutout['origin'][1] + CUT)
        star = by_position.get(key)
        sigma = (star['fwhm'] / 2.3548) if star else sigma_med
        amplitude = cutout['peak']
        bound = cramer_rao_px(amplitude, sigma, cutout['noise'], gain)
        bound_bg = cramer_rao_px(amplitude, sigma, cutout['noise'], gain=None)
        rows.append({'i': i, 'y': key[0], 'x': key[1],
                     'fwhm': star['fwhm'] if star and star['fit_rms'] is not None else None,
                     'flux': cutout['flux'], 'scatter_t': scatter_t[i],
                     'scatter_a': scatter_a.get(i),
                     'cr_2d': bound * np.sqrt(2),      # bound is per axis; scatter is 2-D
                     'cr_bg_2d': bound_bg * np.sqrt(2)})

    flux = np.array([r['flux'] for r in rows])
    measured = np.array([r['scatter_t'] for r in rows])
    measured_a = np.array([r['scatter_a'] or np.nan for r in rows])
    bound2d = np.array([r['cr_2d'] for r in rows])

    bound_bg2d = np.array([r['cr_bg_2d'] for r in rows])
    bright = flux > np.percentile(flux, 75)
    faint = flux < np.percentile(flux, 25)
    floor = float(np.median(measured[bright]))
    floor_affine = float(np.nanmedian(measured_a[bright]))
    cr_bright = float(np.median(bound2d[bright]))
    cr_bg_bright = float(np.median(bound_bg2d[bright]))
    faint_ratio = float(np.median(measured[faint] / bound2d[faint]))

    # ---- spatial correlation
    bins = np.array([0, 200, 500, 1000, 2000, 4000, 8000], dtype=float)
    corr, counts = correlation_by_separation(fields_t, None, bins)
    short_range = corr[0]
    atmosphere_2d = float(np.sqrt(max(short_range, 0.0) * 2))  # per-axis^2 -> 2-D rms
    # the same correlation after affine removal: an affine field is itself spatially
    # correlated, so this is what separates mount/refraction terms from anisoplanatism
    corr_a, counts_a = correlation_by_separation(fields_a, None, bins)
    atmosphere_after_affine_2d = float(np.sqrt(max(corr_a[0], 0.0) * 2))

    # ---- pixel phase
    phase_bias = {}
    for axis, name in ((0, 'y'), (1, 'x')):
        bins_p = np.linspace(0, 1, 9)
        accumulator = [[] for _ in range(8)]
        for f, field in enumerate(fields_t):
            sy, sx = shifts[f]
            for i, dy, dx, py, px_ in field:
                position = (py, px_)[axis] - (sy, sx)[axis]
                b = min(int((position % 1.0) * 8), 7)
                accumulator[b].append((dy, dx)[axis])
        means = [np.mean(a) if len(a) > 20 else np.nan for a in accumulator]
        amplitude = float(np.nanmax(means) - np.nanmin(means)) / 2
        phase_bias[name] = amplitude

    # ---- does the scatter track the FWHM across the field?
    tracking, maps = fwhm_tracking(rows, frames[0].shape)

    # ---- how does the rms average down with N stacked frames?
    bright_i = [r['i'] for r, b in zip(rows, bright) if b]
    scaling = {'translation': stack_scaling(fields_t, len(frames), bright_i),
               'affine': stack_scaling(fields_a, len(frames), bright_i)}

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.loglog(flux, measured, '.', ms=4, alpha=0.45, label='measured per-star scatter (2-D rms)')
    order = np.argsort(flux)
    ax.loglog(flux[order], bound2d[order], '-', lw=2, c='#d62728',
              label='Cramér–Rao bound (photon + pixel noise)')
    ax.axhline(floor, color='#6acc65', ls='--',
               label=f'bright-star floor {floor:.3f} px')
    ax.set_xlabel('star flux (ADU)'); ax.set_ylabel('per-frame scatter (px)')
    ax.set_title('faint stars ride the noise bound; the bright plateau is everything else')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(args.out / 'scatter_vs_bound.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    centres = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(centres[counts > 0], corr[counts > 0], 'o-')
    ax.axhline(0, color='#888', lw=1)
    ax.set_xlabel('separation between stars (px)')
    ax.set_ylabel('residual correlation (px² per axis)')
    ax.set_title('correlated at short range = atmosphere; white = pixel noise')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / 'residual_correlation.png', dpi=140)
    plt.close(fig)

    if maps is not None:
        map_fwhm, map_scatter = maps
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        for ax, grid_map, label in ((axes[0], map_fwhm, 'median FWHM (px)'),
                                    (axes[1], map_scatter,
                                     'median per-frame scatter, affine removed (px)')):
            shown = ax.imshow(grid_map, origin='upper', cmap='viridis')
            fig.colorbar(shown, ax=ax, shrink=0.85)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        usable = [r for r in rows if r['fwhm'] is not None and r['scatter_a'] is not None]
        f_arr = np.array([r['fwhm'] for r in usable])
        s_arr = np.array([r['scatter_a'] for r in usable])
        fx = np.array([r['flux'] for r in usable])
        b_arr = fx > np.percentile(fx, 50)
        axes[2].plot(f_arr[~b_arr], s_arr[~b_arr], '.', ms=4, alpha=0.35,
                     c='#aaa', label='faint half')
        axes[2].plot(f_arr[b_arr], s_arr[b_arr], '.', ms=5, alpha=0.6,
                     c='#1f77b4', label='bright half')
        if 'spearman_scatter' in tracking:
            rho, p = tracking['spearman_scatter']
            axes[2].set_title(f'bright-star Spearman ρ = {rho:+.2f} (p = {p:.1g})',
                              fontsize=10)
        axes[2].set_xlabel('per-star FWHM (px)')
        axes[2].set_ylabel('per-frame scatter (px)')
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out / 'fwhm_vs_scatter.png', dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    styles = {('translation', 'consecutive'): ('o-', '#1f77b4'),
              ('translation', 'shuffled'): ('o--', '#8ab8e0'),
              ('affine', 'consecutive'): ('s-', '#d62728'),
              ('affine', 'shuffled'): ('s--', '#e8a2a2')}
    for (model, order), (style, colour) in styles.items():
        curve = scaling[model][order]
        if curve:
            ns = [c['n'] for c in curve]
            ax.loglog(ns, [c['rms'] for c in curve], style, c=colour,
                      label=f'{model}, {order} groups')
    reference = scaling['affine']['consecutive'] or scaling['translation']['consecutive']
    if reference:
        base = reference[0]['rms']
        ns = np.array([c['n'] for c in reference])
        ax.loglog(ns, base / np.sqrt(ns), 'k:', lw=1.5, label='1/√N from N=1')
    ax.set_xlabel('frames averaged per group, N')
    ax.set_ylabel('2-D rms of the N-frame mean position (px)')
    ax.set_title('does stacking average the error down as 1/√N?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(args.out / 'stack_scaling.png', dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------- budget
    lines = [
        f'frames               : {len(frames)} × {frames[0].shape}, '
        f'{len(rows)} stars in the budget',
        f'gain (measured)      : {gain:.3f} e-/ADU; base noise {read_adu:.2f} ADU',
        '',
        'per-frame 2-D rms centroid scatter, translation removed:',
        f'  bright-star floor          : {floor:.4f} px',
        f'  Cramér–Rao at those fluxes : {cr_bright:.4f} px  '
        f'(photon + pixel noise: {100*cr_bright/floor:.0f}% of the floor)',
        f'  background-only bound      : {cr_bg_bright:.4f} px  '
        f'(gain-free, most optimistic possible)',
        f'  gain caveat                : {gain_caveat or "none"}',
        f'  faint stars vs their bound : ×{faint_ratio:.2f}  '
        f'(1.0 = at the noise limit)',
        '',
        'decomposition of the bright-star floor:',
        f'  spatially correlated (atmosphere-like)   : {atmosphere_2d:.4f} px '
        f'({100*(atmosphere_2d/floor)**2:.0f}% of floor variance)'
        if short_range > 0 else
        '  spatially correlated (atmosphere-like)   : not detected',
        f'  removed by affine over translation       : '
        f'{np.sqrt(max(floor**2 - floor_affine**2, 0)):.4f} px '
        f'(mount rotation/scale/refraction; floor {floor:.4f} -> {floor_affine:.4f})',
        f'  still correlated after affine            : '
        f'{atmosphere_after_affine_2d:.4f} px (anisoplanatic atmosphere)',
        f'  pixel-phase bias amplitude (x, y)        : '
        f'{phase_bias["x"]:.4f}, {phase_bias["y"]:.4f} px',
        '',
        'algorithm term: bounded separately (docs/bench/CENTROIDS.md) -- three unrelated',
        'estimators tie on these frames, so the estimator choice is not the limit.',
    ]

    lines.append('')
    lines.append('does the per-star scatter track the per-star FWHM? (affine removed)')
    if 'spearman_scatter' in tracking:
        rho, p = tracking['spearman_scatter']
        rho_x, p_x = tracking['spearman_excess']
        lines.append(f'  Spearman, bright half ({tracking["n_bright"]} stars)  : '
                     f'rho {rho:+.2f} (p {p:.2g})')
        lines.append(f'  same, scatter normalised by each CR bound: '
                     f'rho {rho_x:+.2f} (p {p_x:.2g})')
        if 'cell_r' in tracking:
            r_cell, p_cell = tracking['cell_r']
            lines.append(f'  cell-level map correlation ({tracking["n_cells"]} cells) '
                         f': r {r_cell:+.2f} (p {p_cell:.2g})')
    else:
        lines.append(f'  too few stars with both a PSF fit and a track '
                     f'({tracking["n"]})')

    lines.append('')
    lines.append('rms of the N-frame mean position (bright stars, disjoint groups):')
    for model in ('translation', 'affine'):
        for c in scaling[model]['consecutive']:
            expected = scaling[model]['consecutive'][0]['rms'] / np.sqrt(c['n'])
            shuffled = next((s['rms'] for s in scaling[model]['shuffled']
                             if s['n'] == c['n']), None)
            lines.append(f'  {model:<12} N={c["n"]}: {c["rms"]:.4f} px  '
                         f'(1/sqrt(N) predicts {expected:.4f}'
                         + (f'; shuffled groups {shuffled:.4f}' if shuffled else '')
                         + f'; {c["stars"]} stars)')
    text = '\n'.join(lines)
    print('\n' + text)
    (args.out / 'summary.txt').write_text(text, encoding='utf-8')
    with open(args.out / 'budget.json', 'w', encoding='utf-8') as fp:
        json.dump({'gain': gain, 'gain_caveat': gain_caveat, 'read_adu': read_adu,
                   'cr_bg_bright_px': cr_bg_bright,
                   'atmosphere_after_affine_2d_px': atmosphere_after_affine_2d,
                   'floor_px': floor,
                   'floor_affine_px': floor_affine, 'cr_bright_px': cr_bright,
                   'faint_ratio': faint_ratio, 'atmosphere_2d_px': atmosphere_2d,
                   'phase_bias': phase_bias,
                   'fwhm_tracking': tracking, 'stack_scaling': scaling,
                   'correlation': {'bins': bins.tolist(), 'value': corr.tolist(),
                                   'counts': counts.tolist()}}, fp)
    print(f'\n{time.time() - t0:.0f}s; results in {args.out}')


if __name__ == '__main__':
    main()
