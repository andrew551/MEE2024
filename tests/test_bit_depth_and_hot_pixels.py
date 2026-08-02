"""
Bit depth, hot pixels, and what the stacked FITS actually contains.

Three defects found on the `example_with_darks` dataset, each of which completed a run
without complaining:

* the stack was min-max stretched into the full 16-bit range, so 12-bit input came out
  16-bit and the numbers no longer meant ADU;
* hot pixels survived dark subtraction -- they clip, and clipping is not linear -- and
  smeared across the stack as fake stars, because they sit still while the field dithers;
* nothing checked that darks and lights were counting in the same units.
"""

import numpy as np
import pytest
from astropy.io import fits

from mee2024.MEE2024util import _version
from mee2024.stacker_implementation import (add_img_to_stack, assert_matching_bit_depth,
                                            hot_pixel_mask, read_bit_depth,
                                            write_stacked_fits)


def _frame(tmp_path, name, data, **cards):
    path = tmp_path / name
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    fits.writeto(path, np.asarray(data, dtype=np.int16), header=header)
    return str(path)


# ------------------------------------------------------------------- bit depth

def test_bitdepth_beats_bitpix(tmp_path):
    """BITPIX is the container; BITDEPTH is the sensor. A 12-bit camera writes both."""
    path = _frame(tmp_path, 'a.fits', np.zeros((4, 4)), BITDEPTH=12)
    assert read_bit_depth(path) == 12


def test_bitpix_is_the_fallback(tmp_path):
    path = _frame(tmp_path, 'b.fits', np.zeros((4, 4)))
    assert read_bit_depth(path) == 16


def test_a_string_bitdepth_is_still_a_number(tmp_path):
    """The example dataset writes it as '12', quoted."""
    path = _frame(tmp_path, 'c.fits', np.zeros((4, 4)), BITDEPTH='12')
    assert read_bit_depth(path) == 12


def test_an_unreadable_file_reports_nothing_rather_than_guessing(tmp_path):
    bad = tmp_path / 'not-a-fits.txt'
    bad.write_text('hello')
    assert read_bit_depth(str(bad)) is None


def test_matching_depths_are_accepted_and_reported(tmp_path):
    lights = [_frame(tmp_path, 'l.fits', np.zeros((4, 4)), BITDEPTH=12)]
    darks = [_frame(tmp_path, 'd.fits', np.zeros((4, 4)), BITDEPTH=12)]
    assert assert_matching_bit_depth(lights, darks) == 12


def test_a_mismatched_dark_is_refused(tmp_path):
    """12-bit lights with 16-bit darks subtract numbers a fixed factor too large."""
    lights = [_frame(tmp_path, 'l.fits', np.zeros((4, 4)), BITDEPTH=12)]
    darks = [_frame(tmp_path, 'd.fits', np.zeros((4, 4)), BITDEPTH=16)]
    with pytest.raises(ValueError, match='do not share a bit depth'):
        assert_matching_bit_depth(lights, darks)


def test_the_refusal_names_which_group_is_odd(tmp_path):
    lights = [_frame(tmp_path, 'l.fits', np.zeros((4, 4)), BITDEPTH=12)]
    flats = [_frame(tmp_path, 'f.fits', np.zeros((4, 4)), BITDEPTH=14)]
    with pytest.raises(ValueError) as exc:
        assert_matching_bit_depth(lights, (), flats)
    assert '12-bit: light' in str(exc.value) and '14-bit: flat' in str(exc.value)


def test_frames_that_declare_nothing_are_skipped_not_assumed(tmp_path):
    """A file with no depth must not be read as conflicting with one that has it."""
    lights = [_frame(tmp_path, 'l.fits', np.zeros((4, 4)), BITDEPTH=12)]
    bad = tmp_path / 'unknown.txt'
    bad.write_text('hello')
    assert assert_matching_bit_depth(lights, [str(bad)]) == 12


def test_no_frames_at_all_is_not_an_error():
    assert assert_matching_bit_depth([], [], []) is None


# ------------------------------------------------------------------ hot pixels

def _dark_with_hot_pixels(shape=(64, 64), level=300.0, noise=5.0, seed=1):
    rng = np.random.default_rng(seed)
    dark = rng.normal(level, noise, shape)
    dark[10, 10] = 16380.0        # saturated, as in the measured example
    dark[20, 30] = 4000.0
    dark[40, 50] = 900.0
    return dark


def test_hot_pixels_are_found_and_ordinary_ones_are_not():
    dark = _dark_with_hot_pixels()
    hot = hot_pixel_mask(dark, sigmas=20.0)
    assert hot[10, 10] and hot[20, 30] and hot[40, 50]
    # 20 sigma on a sigma-5 dark is 100 ADU out; nothing normal should reach it
    assert int(np.sum(hot)) == 3


def test_a_higher_threshold_keeps_more_pixels():
    dark = _dark_with_hot_pixels()
    assert int(np.sum(hot_pixel_mask(dark, sigmas=200.0))) < \
           int(np.sum(hot_pixel_mask(dark, sigmas=20.0)))


def test_a_perfectly_flat_dark_has_no_hot_pixels():
    """A synthetic dark has zero spread; a zero threshold would flag the whole frame."""
    assert not hot_pixel_mask(np.full((8, 8), 100.0)).any()


# ------------------------------------------------- excluding them from the stack

def test_an_excluded_pixel_is_not_counted_rather_than_averaged_in():
    """Dropping it from the sum but not the count would dilute it, not remove it."""
    out = np.zeros((3, 3))
    count = np.zeros((3, 3), dtype=int)
    img = np.full((3, 3), 10.0)
    img[1, 1] = 9999.0
    valid = np.ones((3, 3), dtype=bool)
    valid[1, 1] = False

    add_img_to_stack((img, (0, 0)), out, count, valid=valid)
    assert count[1, 1] == 0 and out[1, 1] == 0
    assert count[0, 0] == 1 and out[0, 0] == 10.0


def test_without_a_mask_every_pixel_counts():
    out = np.zeros((3, 3))
    count = np.zeros((3, 3), dtype=int)
    add_img_to_stack((np.full((3, 3), 4.0), (0, 0)), out, count)
    assert np.all(count == 1) and np.all(out == 4.0)


def test_a_dithered_hot_pixel_costs_one_frame_not_the_sky_position():
    """The point of masking: the detector site moves across the sky between frames, so
    each sky position loses only the frames whose bad pixel landed on it."""
    out = np.zeros((5, 5))
    count = np.zeros((5, 5), dtype=int)
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False                       # the same detector pixel every time
    for shift in ((0, 0), (1, 0), (0, 1)):
        add_img_to_stack((np.full((5, 5), 8.0), shift), out, count, valid=valid)
    stacked = np.divide(out, count, out=np.zeros_like(out), where=count > 0)
    # no sky position is corrupted -- the bad pixel contributed nowhere
    assert np.all(stacked[count > 0] == 8.0)
    # the detector site landed on three different sky positions, and each of those lost
    # exactly one of the three frames; the position it never touched kept all three
    assert count[2, 2] == 2 and count[3, 2] == 2 and count[2, 3] == 2
    assert count[3, 3] == 3


# --------------------------------------------------------- the written stack

def test_the_stack_keeps_the_input_scale(tmp_path):
    """The old code stretched min..max onto 0..65535, turning 12-bit data into 16-bit."""
    path = tmp_path / 's.fit'
    stacked = np.array([[0.0, 100.0], [2000.0, 16380.0]])
    write_stacked_fits(path, stacked, bit_depth=12)
    assert np.array_equal(fits.getdata(path), stacked.astype(np.uint16))


def test_the_depth_and_provenance_are_recorded(tmp_path):
    path = tmp_path / 's.fit'
    write_stacked_fits(path, np.ones((4, 4)) * 500, bit_depth=12, n_frames=7)
    header = fits.getheader(path)
    assert header['BITDEPTH'] == 12
    assert header['NCOMBINE'] == 7
    assert header['MEE2024'] == _version()


def test_a_negative_background_is_offset_not_clipped_away(tmp_path):
    """A mismatched dark drives the sky below zero; clipping would flatten the frame."""
    path = tmp_path / 's.fit'
    stacked = np.array([[-200.0, 0.0], [300.0, 1000.0]])
    pedestal, clipped = write_stacked_fits(path, stacked)
    assert pedestal == 200 and clipped == 0
    data = fits.getdata(path)
    assert fits.getheader(path)['PEDESTAL'] == 200
    # subtracting the recorded pedestal recovers the calibrated ADU exactly
    assert np.array_equal(np.asarray(data, dtype=float) - pedestal, stacked)


def test_no_pedestal_keyword_when_none_was_needed(tmp_path):
    path = tmp_path / 's.fit'
    pedestal, _ = write_stacked_fits(path, np.array([[0.0, 5.0]]))
    assert pedestal == 0
    assert 'PEDESTAL' not in fits.getheader(path)


def test_values_too_wide_for_16_bits_keep_their_values_not_their_dtype(tmp_path):
    path = tmp_path / 's.fit'
    write_stacked_fits(path, np.array([[0.0, 200000.0]]))
    data = fits.getdata(path)
    assert data.dtype.kind == 'f'
    assert data.max() == pytest.approx(200000.0)


def test_nans_do_not_reach_the_file(tmp_path):
    """Uncovered pixels divide 0 by 0; a nan cast to uint16 is undefined."""
    path = tmp_path / 's.fit'
    write_stacked_fits(path, np.array([[np.nan, 10.0], [np.inf, -np.inf]]))
    assert np.all(np.isfinite(np.asarray(fits.getdata(path), dtype=float)))
