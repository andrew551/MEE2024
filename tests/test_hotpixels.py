"""
Finding hot pixels from the dither, with no dark frame.

A star is fixed to the sky, a hot pixel to the detector. The measured behaviour on real
data is in docs/bench/HOTPIX.md; these tests pin the contract, the guards that stop the
method being applied where it cannot work, and the arithmetic that separates the two.
"""

import numpy as np
import pytest
from astropy.io import fits

from mee2024 import hotpixels


# ---------------------------------------------------------------- dither measurement

@pytest.mark.parametrize('shifts,expected', [
    ([(0, 0)], 0.0),
    ([(0, 0), (3, 4)], 5.0),                     # the largest offset, not the last one
    ([(0, 0), (3, 4), (0, 0)], 5.0),
    ([(0, 0), (-3, 0), (4, 0)], 7.0),            # between two frames, neither of them first
    ([(0, 0), None, (0, 6)], 6.0),               # a failed alignment counts as no move
])
def test_dither_span_is_the_largest_separation(shifts, expected):
    assert hotpixels.dither_span(shifts) == pytest.approx(expected)


# ------------------------------------------------------------------------- dark mask

def test_the_dark_mask_finds_the_tail_and_leaves_the_bulk():
    rng = np.random.default_rng(3)
    dark = rng.normal(300.0, 5.0, (64, 64))
    dark[8, 8] = 16380.0
    dark[20, 40] = 900.0
    mask = hotpixels.dark_mask(dark)
    assert mask[8, 8] and mask[20, 40]
    assert int(mask.sum()) == 2


def test_a_flat_dark_yields_nothing_rather_than_everything():
    """Zero spread would make any multiple of it zero, flagging the whole frame."""
    assert not hotpixels.dark_mask(np.full((8, 8), 100.0)).any()


# ------------------------------------------------------- the guards on the dither path

def _synthetic_field(tmp_path, n_frames=5, shift_per_frame=(3.0, 2.0), hot=((30, 30),),
                     stars=((10, 12), (50, 44), (20, 55)), size=80, seed=0):
    """Frames with stars that move with the sky and hot pixels that do not."""
    rng = np.random.default_rng(seed)
    files, shifts = [], []
    for i in range(n_frames):
        dy, dx = shift_per_frame[0] * i, shift_per_frame[1] * i
        frame = rng.normal(100.0, 3.0, (size, size))
        for (sy, sx) in stars:
            # a star sits at its sky position, which in this frame is offset by the dither
            y, x = int(round(sy + dy)), int(round(sx + dx))
            frame[y - 1:y + 2, x - 1:x + 2] += 400.0
            frame[y, x] += 2000.0
        for (hy, hx) in hot:
            frame[hy, hx] += 6000.0            # never moves
        path = tmp_path / f'f{i}.fits'
        fits.writeto(path, frame.astype(np.float32))
        files.append(str(path))
        shifts.append((-dy, -dx))              # the shift that brings frame i onto frame 0
    return files, shifts


def test_two_frames_are_refused(tmp_path):
    files, shifts = _synthetic_field(tmp_path, n_frames=2)
    mask, info = hotpixels.persistence_mask(files, shifts)
    assert mask is None
    assert 'at least three' in info['declined']


def test_an_undithered_sequence_is_refused_rather_than_guessed_at(tmp_path):
    """With no dither the two measures are identical and every star would be flagged."""
    files, shifts = _synthetic_field(tmp_path, shift_per_frame=(0.0, 0.0))
    mask, info = hotpixels.persistence_mask(files, shifts)
    assert mask is None
    assert 'dither' in info['declined'] and '0.0 px' in info['declined']


def test_a_small_dither_is_refused_with_the_number_in_the_message(tmp_path):
    files, shifts = _synthetic_field(tmp_path, shift_per_frame=(0.4, 0.3))
    mask, info = hotpixels.persistence_mask(files, shifts, min_dither=3.0)
    assert mask is None and 'px' in info['declined']


def test_the_declining_path_still_reports_the_dither(tmp_path):
    files, shifts = _synthetic_field(tmp_path, shift_per_frame=(0.0, 0.0))
    _, info = hotpixels.persistence_mask(files, shifts)
    assert info['dither_px'] == pytest.approx(0.0)


# ----------------------------------------------------------- the discrimination itself

def test_the_hot_pixel_is_found_and_the_stars_are_not(tmp_path):
    files, shifts = _synthetic_field(tmp_path)
    mask, info = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0)
    assert mask is not None, info
    assert mask[30, 30], 'the fixed pixel was not flagged'
    # the stars, at their frame-0 positions, must survive
    for (sy, sx) in ((10, 12), (50, 44), (20, 55)):
        assert not mask[sy - 1:sy + 2, sx - 1:sx + 2].any(), f'star at {sy},{sx} flagged'


def test_several_hot_pixels_are_all_found(tmp_path):
    sites = ((30, 30), (12, 60), (65, 20))
    files, shifts = _synthetic_field(tmp_path, hot=sites)
    mask, _ = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0)
    assert all(mask[y, x] for y, x in sites)


def test_a_field_with_no_hot_pixels_yields_an_empty_mask(tmp_path):
    files, shifts = _synthetic_field(tmp_path, hot=())
    mask, info = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0)
    assert mask is None or not mask.any(), 'invented a hot pixel where there was none'


def test_the_blob_region_is_left_alone(tmp_path):
    """A saturated blob is sky-fixed and classifies correctly, but it is not worth testing
    a million pixels, and its centroids are dropped upstream anyway."""
    files, shifts = _synthetic_field(tmp_path)
    blob = np.zeros((80, 80), dtype=bool)
    blob[28:33, 28:33] = True                  # covers the hot pixel
    mask, info = hotpixels.persistence_mask(files, shifts, blob_mask=blob,
                                            candidate_sigmas=10.0)
    assert mask is None or not mask[30, 30], 'tested a pixel inside the blob mask'


def test_the_candidate_list_is_capped(tmp_path):
    files, shifts = _synthetic_field(tmp_path)
    mask, info = hotpixels.persistence_mask(files, shifts, candidate_sigmas=1.0,
                                            max_candidates=50)
    assert info.get('capped_to') == 50


def test_a_higher_ratio_threshold_flags_no_more_than_a_lower_one(tmp_path):
    files, shifts = _synthetic_field(tmp_path, hot=((30, 30), (12, 60)))
    loose, _ = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0,
                                          log_ratio=0.5)
    strict, _ = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0,
                                           log_ratio=6.0)
    assert int(strict.sum()) <= int(loose.sum())


def test_presence_is_required_as_well_as_the_ratio(tmp_path):
    """A ratio of two small numbers can be large by accident; demanding the pixel actually
    be there rejects that."""
    files, shifts = _synthetic_field(tmp_path)
    mask, _ = hotpixels.persistence_mask(files, shifts, candidate_sigmas=10.0,
                                         min_detector=1e9)
    assert not mask.any()


# ------------------------------------------------------------- filtering the centroids

def _entry(row, col, flux=100.0):
    return (flux, 9.0, (row, col))


def test_a_centroid_on_a_flagged_pixel_is_dropped():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True
    kept, dropped = hotpixels.drop_masked_centroids(
        [_entry(20.2, 19.9), _entry(5, 5)], hotpixels.spoiled_by(mask))
    assert dropped == 1
    assert [e[2] for e in kept] == [(5, 5)]


def test_a_centroid_beside_a_flagged_pixel_is_dropped_too():
    """The star's measured centre is pulled toward the hot pixel, so it is not usable."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True
    kept, dropped = hotpixels.drop_masked_centroids([_entry(22, 20)],
                                                    hotpixels.spoiled_by(mask, radius=2))
    assert dropped == 1 and kept == []


def test_a_centroid_well_clear_of_one_survives():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True
    kept, dropped = hotpixels.drop_masked_centroids([_entry(30, 30)],
                                                    hotpixels.spoiled_by(mask))
    assert dropped == 0 and len(kept) == 1


def test_no_mask_means_no_filtering():
    entries = [_entry(1, 1), _entry(2, 2)]
    assert hotpixels.drop_masked_centroids(entries, None) == (entries, 0)


def test_an_empty_mask_means_no_filtering():
    entries = [_entry(1, 1)]
    empty = np.zeros((10, 10), dtype=bool)
    assert hotpixels.drop_masked_centroids(entries, empty) == (entries, 0)


def test_a_centroid_outside_the_frame_does_not_raise():
    """Positions come from a fit and can land just off the edge."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    kept, dropped = hotpixels.drop_masked_centroids(
        [_entry(-2, 4), _entry(99, 3)], hotpixels.spoiled_by(mask))
    assert dropped == 0 and len(kept) == 2
