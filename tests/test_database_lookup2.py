"""Bounding-box lookup against the bundled Tycho catalogue format."""

import numpy as np
import pytest

from mee2024 import database_lookup2


@pytest.fixture
def tiny_catalogue(tmp_path):
    """An .npz in the compressed format: columns are (ra_rad, dec_rad, magnitude)."""
    entries = [
        (10.0, 20.0, 5.0),
        (11.0, 21.0, 8.0),
        (12.0, 22.0, 13.0),   # too faint for a mag-12 cut
        (350.0, 5.0, 6.0),    # just below the RA=0 wrap
        (5.0, 5.0, 7.0),      # just above it
        (100.0, -40.0, 9.0),  # far away
    ]
    data = np.array([[np.radians(ra), np.radians(dec), mag] for ra, dec, mag in entries],
                    dtype=np.float32)
    path = tmp_path / 'tiny.npz'
    np.savez_compressed(path, mydata=data)
    return path


def test_npz_load_builds_unit_vectors(tiny_catalogue):
    dbs = database_lookup2.database_searcher(tiny_catalogue)
    assert dbs.star_table.shape == (6, 6)
    norms = np.linalg.norm(dbs.star_table[:, 2:5], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_lookup_objects_restricts_to_the_box(tiny_catalogue):
    dbs = database_lookup2.database_searcher(tiny_catalogue)
    table, _ = dbs.lookup_objects((9.0, 11.5), (19.0, 21.5), star_max_magnitude=12)
    assert table.shape[0] == 2
    assert np.allclose(np.degrees(sorted(table[:, 0])), [10.0, 11.0], atol=1e-4)


def test_lookup_objects_applies_the_magnitude_cut(tiny_catalogue):
    dbs = database_lookup2.database_searcher(tiny_catalogue)
    bright, _ = dbs.lookup_objects((9.0, 13.0), (19.0, 23.0), star_max_magnitude=12)
    deep, _ = dbs.lookup_objects((9.0, 13.0), (19.0, 23.0), star_max_magnitude=14)
    assert bright.shape[0] == 2
    assert deep.shape[0] == 3


def test_lookup_objects_handles_the_ra_zero_wrap(tiny_catalogue):
    """A (high, low) RA range means 'wrap through 0', as get_bbox produces."""
    dbs = database_lookup2.database_searcher(tiny_catalogue)
    table, _ = dbs.lookup_objects((340.0, 10.0), (0.0, 10.0), star_max_magnitude=12)
    ras = sorted(np.degrees(table[:, 0]))
    assert len(ras) == 2
    assert ras[0] == pytest.approx(5.0, abs=1e-3)
    assert ras[1] == pytest.approx(350.0, abs=1e-3)


def test_lookup_objects_can_return_nothing(tiny_catalogue):
    dbs = database_lookup2.database_searcher(tiny_catalogue)
    table, _ = dbs.lookup_objects((200.0, 210.0), (60.0, 70.0), star_max_magnitude=12)
    assert table.shape[0] == 0


def test_bundled_tycho_catalogue_loads():
    """The catalogue shipped in resources/ must be readable and sane."""
    from mee2024.MEE2024util import resource_path
    dbs = database_lookup2.database_searcher(resource_path('resources/compressed_tycho2024epoch.npz'))
    assert dbs.star_table.shape[0] > 1000000
    # A handful of stars sit a couple of arcseconds outside [0, 2pi): proper motion
    # propagated them across the RA=0 boundary. Tolerate that, but not more.
    margin = np.radians(0.01)
    assert np.all(dbs.star_table[:, 0] >= -margin)
    assert np.all(dbs.star_table[:, 0] <= 2 * np.pi + margin)
    outside = np.sum((dbs.star_table[:, 0] < 0) | (dbs.star_table[:, 0] > 2 * np.pi))
    assert outside < 10, f'{outside} stars outside [0, 2pi) -- expected a mere handful'
    assert np.all(np.abs(dbs.star_table[:, 1]) <= np.pi / 2 + 1e-6)
    # sorted brightest first
    assert dbs.star_table[0, 5] <= dbs.star_table[-1, 5]
