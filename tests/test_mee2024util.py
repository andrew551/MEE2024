"""Config I/O, path helpers and the date/bbox utilities."""

import json

import numpy as np
import pytest

from mee2024 import MEE2024util
from mee2024.config import DEFAULT_OPTIONS, get_default_options


@pytest.mark.parametrize('iso', ['2020-01-01', '2017-08-21', '2024-04-08', '1999-12-31'])
def test_date_float_round_trip(iso):
    as_float = MEE2024util.date_string_to_float(iso)
    assert MEE2024util.date_from_float(as_float) == iso


def test_date_string_to_float_is_monotonic():
    a = MEE2024util.date_string_to_float('2020-01-01')
    b = MEE2024util.date_string_to_float('2021-01-01')
    assert b > a
    assert b - a == pytest.approx(1.0, abs=0.01)  # one year apart


def test_get_bbox_simple():
    corners = np.array([[10.0, 100.0], [12.0, 104.0], [10.0, 104.0], [12.0, 100.0]])
    ra_range, dec_range = MEE2024util.get_bbox(corners)
    assert ra_range == (100.0, 104.0)
    assert dec_range == (10.0, 12.0)


def test_get_bbox_wraps_across_ra_zero():
    """A field straddling RA=0 must come back as (high, low) so callers can OR the range."""
    corners = np.array([[10.0, 359.0], [12.0, 1.0], [10.0, 1.0], [12.0, 359.0]])
    ra_range, _ = MEE2024util.get_bbox(corners)
    assert ra_range == (359.0, 1.0)


def test_get_bbox_wrap_keeps_corners_away_from_zero():
    """Regression: the zwo3 Zenith-Center2 field, centred at RA 1.7.

    The old min/max swap returned (359.82, 0.166) -- the sliver between the two
    corners nearest the wrap -- silently dropping the corners at RA 3.1 and 3.6.
    Verification then fetched ~51 catalogue stars instead of ~1000 and the plate
    solve rejected its own correct solution (6 matched against a threshold of 11).
    """
    corners = np.array([[45.6, 3.595], [44.2, 359.820], [46.0, 0.166], [43.8, 3.146]])
    ra_range, dec_range = MEE2024util.get_bbox(corners)
    assert ra_range == (359.820, 3.595)
    assert dec_range == (43.8, 46.0)


def test_output_path_uses_output_dir_when_set(tmp_path):
    options = {'output_dir': str(tmp_path)}
    result = MEE2024util.output_path('/somewhere/else/thing.fit', options)
    assert result == str(tmp_path / 'thing.fit')


def test_output_path_passes_through_when_output_dir_blank():
    options = {'output_dir': '   '}
    assert MEE2024util.output_path('/a/b/c.fit', options) == '/a/b/c.fit'


def test_read_ini_round_trip(tmp_path):
    path = tmp_path / 'cfg.txt'
    options = get_default_options()
    options['max_star_mag_dist'] = 11.5
    options['distortionOrder'] = 'quintic'
    MEE2024util.write_ini(options, path=path)

    loaded = get_default_options()
    MEE2024util.read_ini(loaded, path=path)
    assert loaded['max_star_mag_dist'] == 11.5
    assert loaded['distortionOrder'] == 'quintic'


def test_read_ini_keeps_defaults_for_missing_keys(tmp_path):
    """A config written by an older version must not drop newly added options."""
    path = tmp_path / 'cfg.txt'
    path.write_text(json.dumps({'max_star_mag_dist': 9.0}), encoding='utf-8')
    options = get_default_options()
    MEE2024util.read_ini(options, path=path)
    assert options['max_star_mag_dist'] == 9.0
    assert options['distortionOrder'] == DEFAULT_OPTIONS['distortionOrder']


def test_read_ini_survives_a_corrupt_file(tmp_path):
    path = tmp_path / 'cfg.txt'
    path.write_text('this is not json{{{', encoding='utf-8')
    options = get_default_options()
    MEE2024util.read_ini(options, path=path)  # must not raise
    assert options == DEFAULT_OPTIONS


def test_get_default_options_returns_a_fresh_copy():
    a = get_default_options()
    a['max_star_mag_dist'] = 99
    assert get_default_options()['max_star_mag_dist'] != 99
    assert DEFAULT_OPTIONS['max_star_mag_dist'] != 99


def test_migration_promotes_the_v2_solver_once(tmp_path):
    """A pre-v1.1.0 config carrying the old default is moved to v2, with a note;
    a v1.1.0 config that says 'triangle' said it deliberately and is kept."""
    old = {'__version__': 'v1.0.1', 'platesolver': 'triangle'}
    notes = MEE2024util.migrate_config(old)
    assert old['platesolver'] == 'v2'
    assert any('platesolver' in n for n in notes)

    deliberate = {'__version__': MEE2024util._version(), 'platesolver': 'triangle'}
    assert MEE2024util.migrate_config(deliberate) == []
    assert deliberate['platesolver'] == 'triangle'
