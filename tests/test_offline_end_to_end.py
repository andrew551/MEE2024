"""Stage 2 through a real offline catalogue directory, with no network.

Builds a small catalogue on disk from the captured Gaia fixture, registers it the way a
locally built or downloaded catalogue is registered, and runs the whole of stage 2 through
it. This is the test that proves the offline path is a drop-in replacement: it asserts the
same numbers as the online run measured in docs/STARCAT_DESIGN.md §1.1.
"""

import json
import zipfile

import numpy as np
import pytest

from mee2024.starcat import providers, store
from tests.fixture_catalogue import build_centroid_zip, load_gaia_fixture
from tests.test_starcat_store import table_from_fixture

# field, order, expected rms (mas), expected stars, expected guessed date.
# Re-pinned at v1.1.0 for v2-seeded fits: the solver seed moved by ~0.3 arcsec,
# which lands the partially-degenerate date+distortion fit in a neighbouring
# optimum -- rms/star-count/nn_corr are equivalent and the date shift is well
# inside the honest sigma_t (~16 days for zwo3; see progress.md 2026-07-30).
CASES = [
    ('zwo3_zenith', 'quintic', 108.9, 433, '2023-10-16'),
    ('zwo1_zenith', 'quintic', 115.1, 1565, '2023-09-07'),
]


def install_offline_catalogue(monkeypatch, tmp_path, field):
    """Write a catalogue directory and make open_catalogue() resolve to it by name."""
    from mee2024 import database_cache

    directory = tmp_path / 'catalogue'
    store.write_catalogue(directory, table_from_fixture(field), name=field,
                          magnitude_limit=12.0)
    provider = providers.GaiaOfflineProvider(directory)
    key = f'offline:{field}'
    monkeypatch.setitem(database_cache._cache.catalogue_cache, key, provider)
    return key, provider


@pytest.mark.slow
@pytest.mark.parametrize('field,order,rms_mas,nstars,expected_date', CASES)
def test_stage2_through_an_offline_catalogue(monkeypatch, tmp_path, options,
                                            field, order, rms_mas, nstars,
                                            expected_date):
    from mee2024 import distortion_fitter

    key, _ = install_offline_catalogue(monkeypatch, tmp_path, field)
    options.update(catalogue=key, distortionOrder=order, guess_date=True,
                   observation_date='2023-10-29', DEFAULT_DATE='2020-01-01',
                   output_dir=str(tmp_path), no_plot=True)

    zip_in = build_centroid_zip(field, tmp_path / 'centroid_data.zip')
    out_zip = distortion_fitter.match_and_fit_distortion(str(zip_in), options, None)
    with zipfile.ZipFile(out_zip) as z:
        result = json.load(z.open('distortion_results.txt'))

    assert result['final rms error (arcseconds)'] * 1000 == pytest.approx(rms_mas, abs=1.0)
    assert result['#stars used'] == pytest.approx(nstars, abs=4)
    assert result['observation_date'] == expected_date


def test_offline_catalogue_is_resolved_by_name(tmp_path, monkeypatch):
    """A locally built catalogue directory must be usable by name alone."""
    from mee2024 import database_cache

    root = tmp_path / 'catalogues'
    root.mkdir()
    store.write_catalogue(root / 'my_catalogue', table_from_fixture('zwo3_zenith'),
                          name='my_catalogue', magnitude_limit=12.0)
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root', lambda: root)

    resolved = database_cache._installed_catalogue_dir('my_catalogue')
    assert resolved is not None and resolved.name == 'my_catalogue'
    assert database_cache._installed_catalogue_dir('does_not_exist') is None


def test_starcat_registry_names_are_recognised():
    from mee2024 import database_cache
    assert database_cache._is_starcat_name('tycho')
    assert database_cache._is_starcat_name('merged')
    assert not database_cache._is_starcat_name('some/path/to.npz')
    assert not database_cache._is_starcat_name(None)


def test_offline_neighbour_flags_come_from_the_catalogue(tmp_path):
    """Offline double-star flagging uses the stored columns, not a mag-17 query."""
    field = 'zwo3_zenith'
    table = table_from_fixture(field)
    # give a handful of stars a close companion
    table.nn_sep[:] = 999.0
    table.nn_mag[:] = 20.0
    table.nn_sep[:7] = 3.0
    table.nn_mag[:7] = 11.0
    store.write_catalogue(tmp_path, table, name=field, magnitude_limit=12.0)

    provider = providers.GaiaOfflineProvider(tmp_path)
    stars = provider.lookup((354.2567, 358.0621), (44.0301, 46.2267), 12.0, 2023.84)
    neighbours = provider.lookup_neighbours(stars, 10.0, 17.0)
    assert len(neighbours) == 7
    assert np.all(neighbours.nn_sep < 10.0)


def test_offline_catalogue_covers_the_full_field(tmp_path):
    """A gap in coverage would quietly reduce the star count and inflate the RMS."""
    for field, box in [('zwo3_zenith', ((354.2567, 358.0621), (44.0301, 46.2267))),
                       ('zwo1_zenith', ((26.0048, 31.1285), (42.7480, 47.8709)))]:
        directory = tmp_path / field
        table = table_from_fixture(field)
        store.write_catalogue(directory, table, name=field, magnitude_limit=12.0)
        rows, epoch = load_gaia_fixture(field)
        found = providers.GaiaOfflineProvider(directory).lookup(*box, 12.0, epoch)
        assert len(found) == len(rows), f'{field}: {len(found)} of {len(rows)} returned'
