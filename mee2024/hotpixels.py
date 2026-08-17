"""
Finding hot pixels, with a dark frame or without one.

A hot pixel reads high whatever the sky is doing, and the worst of them saturate.
Subtracting a dark does not remove those -- saturation clips, and clipping is not linear --
and because they are fixed to the detector while the field is dithered, they smear across
the stack as a small constellation of fake stars. They can also enter the centroid list
that drives frame alignment and the plate solve.

Two ways to find them:

* **From a master dark** (:func:`dark_mask`). Cheap and direct, when darks were taken and
  match the lights.
* **From the dither** (:func:`persistence_mask`). A star is fixed to the *sky*, a hot pixel
  to the *detector*, so asking whether a bright site persists at a fixed detector pixel or
  at a fixed sky position separates them without any dark at all. Measured on the bundled
  example: 96.3% of the dark-confirmed hot pixels found with no false positives.
  ``docs/bench/HOTPIX.md`` has the numbers, the rules that did worse, and the figures.

The dark-free path exists because darks are not always taken, and when they are they are
not always usable -- the darks in that example were shot 45 minutes late and run three
times hotter than the lights.
"""

import numpy as np

#: How far above the master dark's own noise a pixel must sit to be called hot. The bulk of
#: a dark is tight -- on the measured example a median of 306 ADU with a robust sigma of 5
#: and a 99.999th percentile of 396 -- so 20 sigma is far outside the honest pixels.
DARK_SIGMAS = 20.0

#: The floor under that cut, in ADU above the dark's own bias. A multiple of sigma alone is
#: not a fixed threshold: the master is a *mean* of N darks, so its robust sigma falls as
#: 1/sqrt(N), and 20 sigma measured 5.3 ADU with 50 darks against 8.7 ADU with 10. Taking
#: more darks therefore masked more pixels, and changed which stars survived to the fit --
#: a calibration decision quietly driven by how long the observer spent on darks.
#:
#: 10 ADU is what actually perturbs a centroid: a 10 ADU defect 2 px from a 1000 ADU star
#: pulls the measured position by about 37 mas, which is the scale of the residuals the fit
#: is trying to measure. Below that a defect is not worth losing a star over.
#:
#: Used as ``max(MIN_DARK_ADU, sigmas * sigma)``, so N now affects only the *confidence*:
#: on a well-averaged master the absolute floor governs, while a single noisy dark still
#: gets the wide statistical cut it needs. The threshold can only ever rise against the
#: old behaviour, so this masks fewer pixels, never more.
MIN_DARK_ADU = 10.0

#: Threshold on log(detector persistence) - log(sky persistence). Chosen from the middle of
#: a wide plateau: anywhere between 1.0 and 3.5 gives 98-100% precision and 95-99% recall on
#: the measured example, so this is not tuned to an edge.
LOG_RATIO = 2.0

#: A flagged pixel must also be *present*: this rejects noise in a frame with little signal,
#: where a ratio of two small numbers can be large by accident.
MIN_DETECTOR_PERSISTENCE = 5.0

#: How far the field must move between frames for the idea to mean anything. With no dither
#: the two persistence measures are identical by construction and every star would be
#: called hot. The measured example separates cleanly at 5.6 px.
MIN_DITHER_PX = 3.0

#: Ceiling on how many bright sites are tested. A lunar or solar disc in frame can put a
#: million pixels above threshold; they classify correctly (a blob is sky-fixed) but the
#: work is wasted, and an uncapped list is an unbounded allocation on unseen data.
MAX_CANDIDATES = 200_000

#: Sites within this distance of a flagged pixel are treated as spoiled too, when filtering
#: centroids: a centroid beside a hot pixel is pulled off the star it belongs to.
CENTROID_RADIUS_PX = 2


def dark_mask(dark, sigmas=DARK_SIGMAS, min_adu=MIN_DARK_ADU):
    """Detector pixels whose dark level puts them outside the population of real pixels.

    The cut is ``median + max(min_adu, sigmas * sigma)``: an absolute floor in ADU above
    the dark's own bias, widened to a statistical one when the dark is noisy enough to
    need it. See :data:`MIN_DARK_ADU` for why a pure multiple of sigma was wrong -- it
    moved with the number of darks combined, so a longer calibration run masked more
    pixels.

    ``min_adu=0`` restores the old pure-sigma behaviour, and a very large ``sigmas``
    still disables the exclusion entirely.
    """
    dark = np.asarray(dark)
    if dark.ndim != 2:
        return None
    # strided for the statistics: a full-frame median of 47 M pixels is not worth the
    # second it costs when every 4th pixel gives the same answer
    sample = dark[::4, ::4]
    median = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - median)))
    threshold = max(float(min_adu), sigmas * sigma)
    if not threshold > 0:                  # a synthetic flat dark with the floor disabled
        return np.zeros(dark.shape, dtype=bool)
    return dark > median + threshold


def dither_span(shifts):
    """How far the field moved, in pixels: the largest offset between any two frames."""
    points = np.array([(0.0, 0.0) if s is None else (float(s[0]), float(s[1]))
                       for s in shifts])
    if len(points) < 2:
        return 0.0
    spread = points[:, None, :] - points[None, :, :]
    return float(np.max(np.hypot(spread[..., 0], spread[..., 1])))


def _background_and_noise(image, box=64):
    """A smooth background and a robust noise level, so 'bright' is local."""
    from scipy.ndimage import uniform_filter
    sample = image[::4, ::4]
    median = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - median)))
    return uniform_filter(image, size=box), max(sigma, 1e-6)


def _sample_bilinear(excess, rows, cols, shift):
    """``excess`` at the sky position of (rows, cols), undoing this frame's dither.

    Interpolated, not rounded: the dither is sub-pixel, and one pixel of error on the steep
    flank of a bright star is a large error, which made stellar wings look detector-fixed.
    """
    fr = np.clip(rows - shift[0], 0, excess.shape[0] - 1.001)
    fc = np.clip(cols - shift[1], 0, excess.shape[1] - 1.001)
    r0, c0 = np.floor(fr).astype(int), np.floor(fc).astype(int)
    wr, wc = fr - r0, fc - c0
    return ((1 - wr) * (1 - wc) * excess[r0, c0]
            + (1 - wr) * wc * excess[r0, c0 + 1]
            + wr * (1 - wc) * excess[r0 + 1, c0]
            + wr * wc * excess[r0 + 1, c0 + 1])


def persistence_mask(files, shifts, blob_mask=None, candidate_sigmas=20.0,
                     log_ratio=LOG_RATIO, min_detector=MIN_DETECTOR_PERSISTENCE,
                     min_dither=MIN_DITHER_PX, max_candidates=MAX_CANDIDATES):
    """Hot pixels from the dither alone. Returns ``(mask, info)``; mask is None if declined.

    ``info`` always explains itself, so a caller can say why nothing happened rather than
    leaving the user to guess.

    Costs one pass over the frames -- measured at 4.8 s for seven 47-megapixel frames
    against 138 s for the whole of stage 1 -- of which the background filter is nearly all.
    """
    info = {'n_candidates': 0, 'n_flagged': 0,
            'dither_px': dither_span(shifts), 'declined': None}
    if len(files) < 3:
        info['declined'] = (f'only {len(files)} frame(s): telling a fixed pixel from a '
                            f'fixed star needs at least three')
        return None, info
    if info['dither_px'] < min_dither:
        info['declined'] = (
            f'the field moved only {info["dither_px"]:.1f} px between frames, under the '
            f'{min_dither:.0f} px needed: with no dither a hot pixel and a star are '
            f'indistinguishable by this test and every star would be flagged')
        return None, info

    from mee2024.stacker_implementation import open_image

    first = open_image(files[0])
    shape = first.shape
    background, sigma = _background_and_noise(first)
    excess = first - background
    candidate = excess > candidate_sigmas * sigma
    if blob_mask is not None:
        # a saturated blob is sky-fixed and would classify correctly, but it can be a
        # million pixels of pointless work, and its centroids are already dropped upstream
        candidate &= ~np.asarray(blob_mask, dtype=bool)
    rows, cols = np.nonzero(candidate)
    info['n_candidates'] = int(len(rows))
    if not len(rows):
        info['declined'] = 'no pixel stands out from its background'
        return None, info
    if len(rows) > max_candidates:
        keep = np.argpartition(-excess[rows, cols], max_candidates)[:max_candidates]
        rows, cols = rows[keep], cols[keep]
        info['capped_to'] = int(max_candidates)
    del first, background, excess, candidate

    detector = np.empty((len(files), len(rows)), dtype=np.float32)
    sky = np.empty((len(files), len(rows)), dtype=np.float32)
    for i, (path, shift) in enumerate(zip(files, shifts)):
        offset = (0.0, 0.0) if shift is None else (float(shift[0]), float(shift[1]))
        image = open_image(path)
        base, _ = _background_and_noise(image)
        frame_excess = image - base
        detector[i] = frame_excess[rows, cols]
        sky[i] = _sample_bilinear(frame_excess, rows, cols, offset)
        del image, base, frame_excess

    # the *weakest* appearance across frames, so one good frame cannot carry a bad pixel
    det_persist = np.min(detector, axis=0) / sigma
    sky_persist = np.min(sky, axis=0) / sigma
    # a ratio, not a difference: a difference conflates a faint hot pixel with a bright
    # star, and measured 0.962 average precision against 0.996 for this
    score = (np.log(np.maximum(det_persist, 0) + 1.0)
             - np.log(np.maximum(sky_persist, 0) + 1.0))
    flagged = (score > log_ratio) & (det_persist > min_detector)

    mask = np.zeros(shape, dtype=bool)
    mask[rows[flagged], cols[flagged]] = True
    info['n_flagged'] = int(np.sum(flagged))
    info['noise_adu'] = float(sigma)
    return mask, info


def spoiled_by(mask, radius=CENTROID_RADIUS_PX):
    """The mask grown by ``radius``, for deciding which centroids are unusable."""
    if mask is None or not mask.any():
        return mask
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=int(radius))


def drop_masked_centroids(centroid_data, spoiled):
    """Remove centroids that sit on or beside a flagged pixel.

    Filtering the list the detector already produced, rather than detecting again: a bad
    centroid only has to be dropped, and a second detection pass over full frames would
    cost more than everything else here put together.
    """
    if spoiled is None or not spoiled.any():
        return centroid_data, 0
    kept = []
    for entry in centroid_data:
        row = int(round(entry[2][0]))
        col = int(round(entry[2][1]))
        if 0 <= row < spoiled.shape[0] and 0 <= col < spoiled.shape[1] and spoiled[row, col]:
            continue
        kept.append(entry)
    return kept, len(centroid_data) - len(kept)
