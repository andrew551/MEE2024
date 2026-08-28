"""Centroid finding and the stacking helpers.

These tests double as the baseline for the milestone-E centroid benchmark: they measure
how accurately each finder recovers a known sub-pixel position, which is the quantity any
replacement method has to beat.
"""

import numpy as np
import pytest

from mee2024 import stacker_implementation as si
from tests.conftest import gaussian_star_field


def match_to_truth(found, truth, tol=2.0):
    """Pair each true position with the nearest found one; return the offsets."""
    offsets = []
    for ty, tx in truth:
        if len(found) == 0:
            offsets.append(None)
            continue
        d = np.linalg.norm(np.asarray(found) - np.array([ty, tx]), axis=1)
        i = int(np.argmin(d))
        offsets.append(np.asarray(found)[i] - np.array([ty, tx]) if d[i] < tol else None)
    return offsets


# ----------------------------------------------------------------- array helpers

def test_roll_fillzero_shifts_and_zero_fills():
    a = np.arange(25).reshape(5, 5)
    rolled = si.roll_fillzero(a, (1, 0))
    assert np.all(rolled[0, :] == 0)          # vacated row is zeroed, not wrapped
    assert np.array_equal(rolled[1:, :], a[:-1, :])


def test_roll_fillzero_negative_shift():
    a = np.arange(25).reshape(5, 5)
    rolled = si.roll_fillzero(a, (0, -2))
    assert np.all(rolled[:, -2:] == 0)
    assert np.array_equal(rolled[:, :-2], a[:, 2:])


def test_roll_fillzero_zero_shift_is_identity():
    a = np.arange(25).reshape(5, 5)
    assert np.array_equal(si.roll_fillzero(a, (0, 0)), a)


def test_expand_mask_grows_the_masked_region():
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 30:34] = True
    grown = si.expand_mask(mask, 5)
    assert grown.sum() > mask.sum()
    assert grown[30:34, 30:34].all()  # the original region stays masked


# ------------------------------------------------------------ blob removal

def test_remove_saturated_blob_is_a_no_op_when_disabled():
    img = np.zeros((128, 128), dtype=np.float32)
    out, mask, mask2 = si.remove_saturated_blob(img, perform=False)
    assert out is img
    assert not mask.any() and not mask2.any()


def test_remove_saturated_blob_ignores_a_small_bright_spot():
    """A star is not the Moon: below min_size nothing should be masked."""
    img = np.zeros((256, 256), dtype=np.float32)
    img[100:104, 100:104] = 65535
    out, mask, mask2 = si.remove_saturated_blob(img, sat_val=None, min_size=20000)
    assert not mask.any()


def test_remove_saturated_blob_masks_a_large_saturated_disc():
    img = np.zeros((512, 512), dtype=np.float32)
    yy, xx = np.mgrid[0:512, 0:512]
    img[(yy - 256) ** 2 + (xx - 256) ** 2 < 100 ** 2] = 65535
    out, mask, mask2 = si.remove_saturated_blob(img, sat_val=None, radius=40, radius2=80,
                                                min_size=20000)
    assert mask.any(), 'a saturated disc of ~31000 pixels should be masked'
    assert mask[256, 256]
    assert mask2.sum() >= mask.sum()  # the outer exclusion zone is at least as big
    assert out[256, 256] < 65535, 'the blob should have been darkened'


# --------------------------------------------------------- centroid accuracy

def test_simple_get_centroids_finds_every_star(star_positions):
    img = gaussian_star_field(positions=star_positions, noise=2.0)
    found = si.simple_get_centroids(img)
    offsets = match_to_truth(found, star_positions)
    assert all(o is not None for o in offsets), f'missed stars: {offsets}'


# The half-pixel convention offset that simple_get_centroids applies. See
# test_the_two_centroid_finders_disagree_by_half_a_pixel below.
SIMPLE_CENTROID_OFFSET = np.array([0.5, 0.5])


def test_simple_get_centroids_is_accurate_to_a_fraction_of_a_pixel(star_positions):
    """Scatter about its own (offset) convention -- the milestone-E baseline."""
    img = gaussian_star_field(positions=star_positions, noise=2.0)
    found = si.simple_get_centroids(img)
    offsets = np.array([o for o in match_to_truth(found, star_positions) if o is not None])
    scatter = offsets - SIMPLE_CENTROID_OFFSET
    rms = np.sqrt(np.mean(np.sum(scatter ** 2, axis=1)))
    assert rms < 0.25, f'centroid rms {rms:.4f} pixels about the +0.5 convention'


def test_simple_get_centroids_returns_brightest_first():
    positions = [(50.0, 50.0), (100.0, 100.0), (150.0, 150.0)]
    img = gaussian_star_field(positions=positions, fluxes=[500, 3000, 1500], noise=1.0)
    found = si.simple_get_centroids(img)
    assert len(found) >= 3
    # the 3000-flux star must come first (allowing for the +0.5 convention)
    assert np.allclose(found[0] - SIMPLE_CENTROID_OFFSET, (100.0, 100.0), atol=0.3)


def test_the_two_centroid_finders_disagree_by_half_a_pixel(options):
    """PINS A KNOWN DEFECT -- the two finders use different pixel conventions.

    simple_get_centroids (the non-sensitive path) adds 0.5 to both axes, treating an
    integer index as a pixel corner. get_centroids_blur (the sensitive path) uses
    skimage's centroid_weighted, where an integer index is the pixel centre. The two
    therefore disagree by (0.5, 0.5) pixels -- about 1.3 arcsec diagonally at the
    ~1.85 arcsec/pixel of the example data, i.e. larger than the signal being measured.

    A constant offset is absorbed by the plate solution, so it does not bias the
    deflection constant, but it does mean centroids from the two modes are not
    interchangeable, and stage-1 RA/DEC is off by ~0.9 arcsec per axis in the
    non-sensitive mode. do_stack's plotting code already compensates for this at display
    time only ('shift = 0 if centroid_gaussian_subtract else 0.5').

    If the convention is unified, this test should fail -- update it deliberately.
    """
    truth = [(50.0, 50.0), (100.0, 150.0), (200.5, 200.5)]
    img = gaussian_star_field(shape=(256, 256), positions=truth,
                              fluxes=[5000] * len(truth), noise=0.0)

    simple = si.simple_get_centroids(img)

    opts = dict(options, centroid_gaussian_subtract=True, centroid_gaussian_thresh=4.0,
                sigma_subtract=1.0, min_area=3)
    blank = np.zeros(img.shape, dtype=bool)
    sensitive = [d[2] for d in si.get_centroids_blur((img, blank, blank), options=opts)]

    simple_off = np.array([o for o in match_to_truth(simple, truth) if o is not None])
    sensitive_off = np.array([o for o in match_to_truth(sensitive, truth) if o is not None])

    assert np.allclose(simple_off.mean(axis=0), [0.5, 0.5], atol=0.02)
    assert np.allclose(sensitive_off.mean(axis=0), [0.0, 0.0], atol=0.02)


def test_simple_get_centroids_on_an_empty_field():
    img = gaussian_star_field(positions=[], noise=1.0)
    found = si.simple_get_centroids(img)
    assert len(found) == 0


def test_get_centroids_blur_sensitive_mode_finds_stars(options, star_positions):
    options['centroid_gaussian_subtract'] = True
    options['centroid_gaussian_thresh'] = 4.0
    options['sigma_subtract'] = 1.0
    options['min_area'] = 3
    img = gaussian_star_field(shape=(256, 256), positions=star_positions,
                              fluxes=[4000] * len(star_positions), noise=2.0)
    mask = np.zeros(img.shape, dtype=bool)

    data = si.get_centroids_blur((img, mask, mask), options=options)

    found = [d[2] for d in data]
    offsets = match_to_truth(found, star_positions)
    assert all(o is not None for o in offsets), f'missed stars: {offsets}'
    arr = np.array([o for o in offsets if o is not None])
    rms = np.sqrt(np.mean(np.sum(arr ** 2, axis=1)))
    assert rms < 0.25, f'sensitive-mode centroid rms {rms:.4f} pixels'


def test_get_centroids_blur_returns_flux_area_position(options, star_positions):
    options['centroid_gaussian_subtract'] = True
    options['centroid_gaussian_thresh'] = 4.0
    options['sigma_subtract'] = 1.0
    options['min_area'] = 3
    img = gaussian_star_field(positions=star_positions,
                              fluxes=[4000] * len(star_positions), noise=2.0)
    mask = np.zeros(img.shape, dtype=bool)
    data = si.get_centroids_blur((img, mask, mask), options=options)
    assert data, 'expected some centroids'
    flux, area, position, peak = data[0]
    assert flux > 0 and area > 0 and len(position) == 2
    # the peak is the raw image maximum near the star, so it must be at least the star's
    # amplitude above the background rather than a repeat of the noise-normed flux
    assert peak > 0 and peak <= img.max()
    # sorted brightest first
    assert [d[0] for d in data] == sorted([d[0] for d in data], reverse=True)


# ------------------------------------------------------------------ filtering

def test_filter_very_edgy_centroids_drops_points_near_the_border():
    img = np.zeros((100, 100))
    data = [(1, 1, (2.0, 50.0)),     # too close to the top
            (1, 1, (50.0, 50.0)),    # fine
            (1, 1, (97.0, 50.0))]    # too close to the bottom
    kept = si.filter_very_edgy_centroids(data, img, f=5)
    assert [d[2] for d in kept] == [(50.0, 50.0)]


def test_filter_bad_centroids_drops_masked_points():
    shape = (100, 100)
    mask2 = np.zeros(shape, dtype=bool)
    mask2[40:60, 40:60] = True
    data = [(1, 1, (50.0, 50.0)),    # inside the mask
            (1, 1, (10.0, 10.0))]    # outside
    kept = si.filter_bad_centroids(data, mask2, shape)
    assert [d[2] for d in kept] == [(10.0, 10.0)]


# ------------------------------------------------------------------ alignment

def test_attempt_align_recovers_a_known_translation(options):
    rng = np.random.default_rng(4)
    c1 = rng.uniform(100, 900, size=(40, 2))
    shift = np.array([7.0, -13.0])
    c2 = c1 - shift

    _, matches1, _, shift2, rms = si.attempt_align(c1, c2, options)

    assert np.allclose(shift2, shift, atol=1e-3), f'recovered {shift2}, expected {shift}'
    assert rms < 1e-3
    assert len(matches1) >= 30


def test_attempt_align_raises_when_there_are_no_centroids(options):
    with pytest.raises(Exception, match='No centroids found'):
        si.attempt_align(np.zeros((0, 2)), np.ones((5, 2)), options)


def test_attempt_align_raises_when_nothing_matches(options):
    """Two unrelated fields must fail loudly rather than return a bogus shift."""
    rng = np.random.default_rng(9)
    c1 = rng.uniform(0, 1000, size=(30, 2))
    c2 = rng.uniform(0, 1000, size=(30, 2))
    opts = dict(options, pxl_tol=0.01)
    with pytest.raises(Exception, match='failed to match stars'):
        si.attempt_align(c1, c2, opts)


# -------------------------------------------------------------------- stacking

def test_add_img_to_stack_accumulates_with_counts():
    img = np.ones((10, 10))
    out = np.zeros((10, 10))
    count = np.zeros((10, 10), dtype=int)
    si.add_img_to_stack((img, (0, 0)), out, count)
    si.add_img_to_stack((img, (2, 0)), out, count)
    # rows 2..9 got both frames, rows 0..1 only the first
    assert count[5, 5] == 2
    assert count[0, 5] == 1
    assert out[5, 5] == 2.0


def test_add_img_to_stack_rounds_fractional_shifts():
    """Documents the current integer-shift behaviour (a milestone-E target)."""
    img = np.zeros((10, 10))
    img[5, 5] = 1.0
    out = np.zeros((10, 10))
    count = np.zeros((10, 10), dtype=int)
    si.add_img_to_stack((img, (1.4, 0.0)), out, count)
    assert out[6, 5] == 1.0, 'a 1.4 pixel shift is currently rounded to 1'
    assert out[7, 5] == 0.0


# ------------------------------------------------------- saturation (F16)

def test_a_clipped_star_passes_the_radial_sanity_check():
    """The reason F16 has to exist. `sanity_check_centroids` is the only thing in stage 1
    that looks at a star's shape, and it asks whether the profile decreases outward -- which
    a flat-topped, clipped star does. So nothing at any stage was testing a peak value, and
    a clipped star was removed only if its position error happened to exceed
    `distortion_fit_tol`. On the eclipse field that tolerance is 999 by design."""
    full_scale = 65535.0
    yy, xx = np.mgrid[0:60, 0:60]
    # amplitude well past full scale, so the top genuinely flattens rather than merely
    # being bright -- an earlier draft of this test used 60000 on a 100 ADU background and
    # never clipped at all, which is exactly the failure it is meant to catch
    star = 100.0 + 90000.0 * np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / (2 * 3.0 ** 2))
    clipped = np.minimum(star, full_scale)

    assert (clipped == full_scale).sum() > 8, 'the star should have a flat top'

    # sanity_check's own test: mean(3x3) > mean(5x5) > mean(7x7) > mean(9x9)
    means = [clipped[30 - r:30 + r + 1, 30 - r:30 + r + 1].mean() for r in range(1, 5)]
    assert means == sorted(means, reverse=True), 'the sanity check would pass this star'
    assert si.peak_value(clipped, 30, 30) == full_scale, 'but the peak says it clipped'


def test_peak_value_is_nan_at_the_frame_edge():
    """nan means "not known", and `_drop_saturated` must read that as grounds to keep a
    star rather than to reject one."""
    img = np.ones((20, 20))
    assert np.isnan(si.peak_value(img, 1, 10))
    assert not np.isnan(si.peak_value(img, 10, 10))
