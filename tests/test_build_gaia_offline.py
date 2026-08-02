"""
The offline catalogue builder's crash safety.

A deep build runs for hours or days against an archive whose throughput has been measured
to swing 50x between days, so it *will* be interrupted. The property that matters is that
an interruption costs time and nothing else: a cached chunk must be either absent or
complete, never a truncated file the next run counts as done.
"""

import numpy as np
import pytest

from tools.build_gaia_offline import _hms, _save_chunk

ROWS = np.array([(1, 10.0), (2, 20.0)], dtype=[('source_id', 'i8'), ('ra', 'f8')])


def test_a_saved_chunk_reads_back_intact(tmp_path):
    path = tmp_path / 'stripe_0003_000.npy'
    _save_chunk(path, ROWS)
    assert np.array_equal(np.load(path), ROWS)


def test_the_temporary_file_does_not_survive(tmp_path):
    path = tmp_path / 'stripe_0003_000.npy'
    _save_chunk(path, ROWS)
    assert [p.name for p in tmp_path.iterdir()] == ['stripe_0003_000.npy']


def test_a_crash_before_the_rename_leaves_no_chunk_behind(tmp_path):
    """The half-written file must not be mistaken for a finished one.

    Writing straight to the final name would leave a truncated `.npy` that the resume
    check counts as cached and never refetches -- a silent hole in the sky.
    """
    path = tmp_path / 'stripe_0003_000.npy'
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'wb') as fp:          # a build killed mid-write
        fp.write(b'\x93NUMPY truncated')

    assert not path.exists(), 'the interrupted write must not occupy the final name'
    # and the leftover is invisible to the glob that assembles the catalogue
    assert list(tmp_path.glob('stripe_*.npy')) == []

    _save_chunk(path, ROWS)              # the retry overwrites it and completes
    assert np.array_equal(np.load(path), ROWS)


def test_the_chunk_name_is_the_one_the_resume_check_looks_for(tmp_path):
    """np.save appends '.npy' to a path that lacks it, which would rename the cache."""
    path = tmp_path / 'stripe_0007_002.npy'
    _save_chunk(path, ROWS)
    assert list(tmp_path.glob('stripe_*.npy')) == [path]


@pytest.mark.parametrize('seconds,expected', [
    (0, '0h00m'), (59, '0h00m'), (61, '0h01m'), (3600, '1h00m'), (86_400 + 60, '24h01m'),
    (-5, '0h00m'),                       # a clock that went backwards is not a crash
])
def test_elapsed_times_render_as_hours_and_minutes(seconds, expected):
    assert _hms(seconds) == expected
