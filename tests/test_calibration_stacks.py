"""
Keeping the combined dark and flat.

A master calibration frame is worth more than the frames it came from -- it can calibrate
a later session without hauling every original around -- so the combined result is written
beside the results. The rule that matters is *when*: combining one frame produces a copy of
its input, and writing that back out under a new name is clutter that reads like a product.
"""

import numpy as np
import pytest
from astropy.io import fits

from mee2024 import events
from mee2024.MEE2024util import _version
from mee2024.stacker_implementation import save_calibration_stacks

DARK = np.full((4, 5), 3.0)
FLAT = np.full((4, 5), 1.5)


def _save(tmp_path, n_darks, n_flats):
    return save_calibration_stacks(tmp_path, '_T1', ['d'] * n_darks, DARK,
                                   ['f'] * n_flats, FLAT)


def test_two_or_more_frames_are_worth_keeping(tmp_path):
    written = _save(tmp_path, 3, 2)
    assert [p.name for p in written] == ['DARK_STACK_T1.fit', 'FLAT_STACK_T1.fit']
    assert all(p.exists() for p in written)


def test_a_single_frame_is_not_written_back_out(tmp_path):
    """It would be a byte-for-byte copy of the input under a name implying more."""
    assert _save(tmp_path, 1, 1) == []
    assert list(tmp_path.iterdir()) == []


def test_no_calibration_frames_at_all_writes_nothing(tmp_path):
    assert save_calibration_stacks(tmp_path, '_T1', [], DARK, None, FLAT) == []
    assert list(tmp_path.iterdir()) == []


def test_darks_and_flats_are_decided_separately(tmp_path):
    """Two darks and one flat keeps the dark only."""
    written = _save(tmp_path, 2, 1)
    assert [p.name for p in written] == ['DARK_STACK_T1.fit']


def test_the_pixels_survive_the_round_trip(tmp_path):
    written = _save(tmp_path, 2, 2)
    assert np.allclose(fits.getdata(written[0]), DARK)
    assert np.allclose(fits.getdata(written[1]), FLAT)


def test_the_header_records_what_went_into_it(tmp_path):
    """Reusing a master frame later means knowing how it was made."""
    written = _save(tmp_path, 7, 2)
    header = fits.getheader(written[0])
    assert header['NCOMBINE'] == 7
    assert header['COMBTYPE'] == 'mean'
    assert header['MEE2024'] == _version()


def test_it_is_stored_as_float_not_the_input_dtype(tmp_path):
    """A mean of integer frames is not an integer; rounding it would waste the averaging."""
    written = save_calibration_stacks(tmp_path, '_T1', ['a', 'b'],
                                      np.full((3, 3), 2.5), [], FLAT)
    stored = fits.getdata(written[0]).dtype
    # FITS is big-endian on disk, so compare kind and width rather than the dtype object
    assert (stored.kind, stored.itemsize) == ('f', 4)
    assert np.allclose(fits.getdata(written[0]), 2.5)


def test_rerunning_into_the_same_folder_does_not_fail(tmp_path):
    """The name carries a timestamp, but a repeat within the same second must not crash."""
    _save(tmp_path, 2, 2)
    assert len(_save(tmp_path, 2, 2)) == 2


def test_what_was_saved_is_reported(tmp_path):
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        _save(tmp_path, 4, 2)
    said = [e['text'] for e in sink.events if e['type'] == events.LOG]
    assert any('dark' in t and '4 frames' in t for t in said)
    assert any('flat' in t and '2 frames' in t for t in said)


@pytest.mark.parametrize('n', [0, 1])
def test_the_boundary_is_two(tmp_path, n):
    assert _save(tmp_path, n, n) == []
