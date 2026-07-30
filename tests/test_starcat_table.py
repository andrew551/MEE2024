"""StarTable: the unified star-data type."""

import numpy as np
import pytest

from mee2024.starcat import ORIGIN_GAIA, ORIGIN_TYCHO, StarTable, concat


def make_table(n=5, epoch=2016.0, **kwargs):
    rng = np.random.default_rng(0)
    defaults = dict(
        ra=np.radians(rng.uniform(0, 360, n)),
        dec=np.radians(rng.uniform(-60, 60, n)),
        mag=rng.uniform(6, 12, n),
        ids=np.arange(1, n + 1, dtype=np.int64),
        epoch=epoch,
    )
    defaults.update(kwargs)
    return StarTable(**defaults)


# ---------------------------------------------------------------- construction

def test_length_and_accessors():
    t = make_table(7)
    assert len(t) == 7 and t.nstars() == 7
    assert t.get_ra().shape == (7,)
    assert t.get_ra_dec().shape == (7, 2)
    assert t.get_pmotion().shape == (7, 2)


def test_positions_are_float64():
    """float32 resolves only ~0.1 arcsec, larger than the signal we measure."""
    t = make_table()
    assert t.ra.dtype == np.float64
    assert t.dec.dtype == np.float64
    assert t.get_vectors().dtype == np.float64


def test_optional_columns_default_to_nan():
    t = make_table(4)
    for column in (t.pmra, t.pmdec, t.parallax, t.radial_velocity, t.nn_sep, t.nn_mag):
        assert np.all(np.isnan(column))


def test_ids_survive_a_19_digit_gaia_source_id():
    """A Gaia source_id has ~19 digits; a float64 mantissa holds only 53 bits."""
    big = np.array([1926193296392975104, 2577063532462714368], dtype=np.int64)
    t = make_table(2, ids=big)
    assert np.array_equal(t.get_ids(), big)
    assert t.ids.dtype == np.int64


def test_mismatched_column_length_is_rejected():
    with pytest.raises(ValueError, match='length'):
        make_table(5, pmra=np.zeros(3))


def test_unknown_column_is_rejected():
    with pytest.raises(TypeError, match='unexpected columns'):
        make_table(5, proper_motion=np.zeros(5))


def test_bad_band_is_rejected():
    with pytest.raises(ValueError, match='band must be'):
        make_table(3, band='R')


def test_vectors_are_unit_length_and_match_ra_dec():
    t = make_table(20)
    v = t.get_vectors()
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(np.arcsin(v[:, 2]), t.dec)
    assert np.allclose(np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi), t.ra % (2 * np.pi))


def test_vectors_are_cached_but_invalidated_by_select():
    t = make_table(10)
    first = t.get_vectors()
    assert t.get_vectors() is first          # cached
    assert t.select([0, 1]).get_vectors().shape == (2, 3)


# ----------------------------------------------------------- per-star metadata

def test_has_proper_motion_is_per_star():
    """0.8% of Gaia G<12 sources have no PM, so this cannot be a table-level flag."""
    pmra = np.array([1.0, np.nan, 3.0, 4.0])
    pmdec = np.array([1.0, 2.0, np.nan, 4.0])
    t = make_table(4, pmra=pmra, pmdec=pmdec)
    assert list(t.has_proper_motion()) == [True, False, False, True]


def test_origin_may_be_a_scalar_or_an_array():
    assert np.all(make_table(3).origin == ORIGIN_GAIA)
    mixed = make_table(3, origin=[ORIGIN_GAIA, ORIGIN_TYCHO, ORIGIN_GAIA])
    assert list(mixed.is_gaia()) == [True, False, True]


def test_is_double_uses_the_neighbour_column():
    t = make_table(3, nn_sep=[0.5, 30.0, np.nan])
    assert list(t.is_double(cutoff_arcsec=10.0)) == [True, False, False]


def test_is_double_is_false_when_neighbours_were_never_computed():
    """An absent neighbour column must mean 'nothing flagged', not 'all flagged'."""
    assert not make_table(5).is_double(10.0).any()


# ------------------------------------------------------------------ operations

def test_select_returns_a_new_table_and_leaves_the_original_alone():
    t = make_table(6)
    subset = t.select([0, 2, 4])
    assert len(subset) == 3
    assert len(t) == 6
    subset.ra[0] = 1.234
    assert t.ra[0] != 1.234, 'select() must not alias the parent arrays'


def test_select_accepts_a_boolean_mask():
    t = make_table(5)
    assert len(t.select(t.mag < np.median(t.mag))) == 2


def test_brightest_returns_brightest_first():
    t = make_table(10)
    top = t.brightest(4)
    assert len(top) == 4
    assert np.all(np.diff(top.mag) >= 0)
    assert top.mag[0] == t.mag.min()


def test_within_box_filters_and_handles_ra_wrap():
    ra = np.radians([1.0, 10.0, 350.0, 359.0])
    dec = np.radians([10.0, 10.0, 10.0, 10.0])
    t = StarTable(ra=ra, dec=dec, mag=[8, 8, 8, 8], ids=[1, 2, 3, 4], epoch=2016.0)

    plain = t.within_box((0.0, 20.0), (0.0, 20.0))
    assert sorted(plain.get_ids()) == [1, 2]

    wrapped = t.within_box((340.0, 5.0), (0.0, 20.0))
    assert sorted(wrapped.get_ids()) == [1, 3, 4]


def test_within_box_applies_the_magnitude_cut():
    t = make_table(20)
    limited = t.within_box((0.0, 360.0), (-90.0, 90.0), max_magnitude=9.0)
    assert np.all(limited.mag < 9.0)


def test_concat_joins_tables():
    a, b = make_table(3), make_table(4)
    joined = concat([a, b])
    assert len(joined) == 7
    assert np.array_equal(joined.ra[:3], a.ra)


def test_concat_refuses_mixed_epochs():
    with pytest.raises(ValueError, match='mixed epochs'):
        concat([make_table(2, epoch=2016.0), make_table(2, epoch=2024.0)])


def test_concat_refuses_mixed_bands():
    with pytest.raises(ValueError, match='mixed bands'):
        concat([make_table(2, band='G'), make_table(2, band='V')])


def test_concat_of_nothing_raises():
    with pytest.raises(ValueError, match='nothing to concatenate'):
        concat([])


# ------------------------------------------------------------ epoch propagation

def test_at_epoch_moves_a_star_by_its_proper_motion():
    """A 100 mas/yr proper motion over 10 years must move the star by 1 arcsec."""
    t = StarTable(ra=np.radians([180.0]), dec=np.radians([0.0]), mag=[8.0], ids=[1],
                  epoch=2016.0, pmra=[100.0], pmdec=[0.0], parallax=[10.0],
                  radial_velocity=[0.0])
    moved = t.at_epoch(2026.0)
    shift = (moved.ra[0] - t.ra[0]) * np.cos(t.dec[0])
    assert np.degrees(shift) * 3600 == pytest.approx(1.0, rel=1e-3)
    assert moved.epoch == 2026.0


def test_at_epoch_leaves_the_original_untouched():
    t = StarTable(ra=np.radians([10.0]), dec=np.radians([20.0]), mag=[8.0], ids=[1],
                  epoch=2016.0, pmra=[500.0], pmdec=[500.0], parallax=[20.0])
    before = t.ra.copy()
    t.at_epoch(2030.0)
    assert np.array_equal(t.ra, before)
    assert t.epoch == 2016.0


def test_at_epoch_is_a_no_op_for_the_same_epoch():
    t = make_table(5, epoch=2016.0)
    same = t.at_epoch(2016.0)
    assert np.array_equal(same.ra, t.ra)


def test_at_epoch_keeps_stars_that_have_no_proper_motion_put():
    t = StarTable(ra=np.radians([10.0, 20.0]), dec=np.radians([0.0, 0.0]),
                  mag=[8.0, 9.0], ids=[1, 2], epoch=2016.0,
                  pmra=[1000.0, np.nan], pmdec=[0.0, np.nan], parallax=[10.0, np.nan])
    moved = t.at_epoch(2026.0)
    assert moved.ra[0] != t.ra[0], 'the star with a PM should move'
    assert moved.ra[1] == t.ra[1], 'the star without a PM must not move'
    assert moved.dec[1] == t.dec[1]


def test_at_epoch_round_trip_returns_to_the_start():
    t = StarTable(ra=np.radians([123.0]), dec=np.radians([45.0]), mag=[7.0], ids=[1],
                  epoch=2016.0, pmra=[250.0], pmdec=[-130.0], parallax=[15.0],
                  radial_velocity=[12.0])
    back = t.at_epoch(2024.0).at_epoch(2016.0)
    # measured 0.023 mas -- four orders of magnitude below the ~100 mas we achieve
    assert np.degrees(abs(back.ra[0] - t.ra[0])) * 3.6e6 < 0.1
    assert np.degrees(abs(back.dec[0] - t.dec[0])) * 3.6e6 < 0.1


def test_at_epoch_on_an_empty_table():
    empty = make_table(0)
    assert len(empty.at_epoch(2024.0)) == 0


# ------------------------------------------------- deprecated in-place wrappers

def test_select_indices_mutates_in_place():
    t = make_table(6)
    t.select_indices([1, 3])
    assert len(t) == 2


def test_update_epoch_mutates_in_place():
    t = StarTable(ra=np.radians([10.0]), dec=np.radians([0.0]), mag=[8.0], ids=[1],
                  epoch=2016.0, pmra=[100.0], pmdec=[0.0], parallax=[10.0])
    before = t.ra[0]
    t.update_epoch(2026.0)
    assert t.epoch == 2026.0
    assert t.ra[0] != before


def test_copy_does_not_alias():
    from copy import copy
    t = make_table(4)
    duplicate = copy(t)
    duplicate.ra[0] = 9.9
    assert t.ra[0] != 9.9
    assert duplicate.epoch == t.epoch, 'copy must preserve the epoch'


def test_repr_reports_size_epoch_and_origins():
    t = make_table(3, origin=[ORIGIN_GAIA, ORIGIN_TYCHO, ORIGIN_TYCHO])
    text = repr(t)
    assert '3 stars' in text and 'gaia=1' in text and 'tycho=2' in text
