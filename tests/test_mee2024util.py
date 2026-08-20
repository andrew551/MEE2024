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


def test_the_two_version_numbers_agree():
    """The version lives in two files, and one of them was missed at v1.3.9.

    `MEE2024util._version()` names the executable and stamps every FITS header; the
    `version` in setup.cfg is the pip package metadata. Every release from v1.3.2 to
    v1.3.8 bumped both, by hand, and v1.3.9 shipped with setup.cfg still saying 1.3.8 --
    so `pip show mee2024` reported a version that had never been built.

    Compared as a numeric triple rather than as text, so a pre-release suffix on a
    development branch (`v1.4.0-dev`) does not have to be mirrored into packaging
    metadata that has its own spelling rules.
    """
    import configparser
    from pathlib import Path

    from mee2024.MEE2024util import _version, _version_tuple

    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).parent.parent / 'setup.cfg', encoding='utf-8')
    packaged = cfg['metadata']['version']

    assert _version_tuple(packaged) == _version_tuple(_version()), (
        f'setup.cfg says {packaged}, MEE2024util._version() says {_version()} -- '
        'bump both when releasing')


def _declared_floor():
    """`python_requires` from setup.cfg, as an (major, minor) tuple."""
    import configparser
    import re
    from pathlib import Path

    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).parent.parent / 'setup.cfg', encoding='utf-8')
    declared = cfg['options']['python_requires']
    match = re.fullmatch(r'>=\s*(\d+)\.(\d+)', declared.strip())
    assert match, f'python_requires is {declared!r}, which this test cannot read'
    return int(match.group(1)), int(match.group(2))


def test_ci_tests_the_python_version_the_package_claims_to_require():
    """The floor is a promise; CI is the only evidence for it. They are separate edits.

    `python_requires` in setup.cfg is what pip enforces on anyone installing from source.
    `python-version` in .github/workflows/tests.yml is what the suite actually runs on.
    Nothing but this test connects them, and that gap is not hypothetical: the floor said
    `>=3.9` until 2026-08-20 while CI had only ever run 3.12, so the 3.9 claim was never
    exercised by anything. It was false -- numpy 2.5 and scipy 1.18 both require 3.12 -- and
    a collaborator on 3.10.7 found out by getting a clean install and a failing suite.

    Asserts the floor is *among* the tested versions rather than equal to the only one, so a
    matrix that adds a newer interpreter still passes. What must not happen is CI moving off
    the floor and leaving the promise untested again.
    """
    import re
    from pathlib import Path

    workflow = (Path(__file__).parent.parent
                / '.github' / 'workflows' / 'tests.yml').read_text(encoding='utf-8')
    tested = {tuple(int(part) for part in m.group(1).split('.'))
              for m in re.finditer(r"""python-version:\s*['"]?(\d+\.\d+)""", workflow)}
    assert tested, 'no python-version found in tests.yml -- has the workflow changed shape?'

    floor = _declared_floor()
    assert floor in tested, (
        f'setup.cfg requires >={floor[0]}.{floor[1]} but CI runs '
        f'{", ".join(f"{v[0]}.{v[1]}" for v in sorted(tested))} -- the floor is a promise to '
        'users and CI is the only thing that checks it, so one of the two is wrong')


def test_this_interpreter_satisfies_the_declared_floor():
    """Run on something older than the floor and this says so, rather than letting an
    unrelated test fail for a reason nobody connects to the interpreter.

    That is precisely what happened on 2026-08-20: a 3.10.7 machine installed cleanly,
    because `python_requires` still said `>=3.9`, and the only symptom was a SER timestamp
    test failing on a stdlib difference (`datetime.fromisoformat` gained full ISO 8601 in
    3.11). One clear failure beats a puzzling one.
    """
    import sys

    floor = _declared_floor()
    assert sys.version_info[:2] >= floor, (
        f'this is Python {sys.version_info.major}.{sys.version_info.minor}, but setup.cfg '
        f'requires >={floor[0]}.{floor[1]} -- rebuild the venv on a supported interpreter')


def test_environment_line_names_the_interpreter_and_the_packages_that_move_numbers():
    """Written beside every result, because none of these is pinned.

    setup.cfg names numpy, scipy, astropy and photutils without versions, so two installs a
    month apart resolve differently and nothing else records which one produced a number. A
    collaborator's environment held astropy 6.1.7 against this machine's 8.0.1 -- two major
    versions, in the library the refraction correction transforms through -- and the only
    reason anyone found out was that he happened to compare pip list output.
    """
    import sys

    from mee2024.MEE2024util import environment_line

    line = environment_line()
    assert f'python {sys.version_info.major}.{sys.version_info.minor}' in line
    for package in ('numpy', 'scipy', 'astropy', 'photutils'):
        assert package in line, f'{package} is in the measurement path and must be recorded'
    assert '?' not in line, f'a version could not be read: {line}'
