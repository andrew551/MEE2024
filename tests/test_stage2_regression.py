"""Stage 2 end to end, offline and deterministic.

Locks in the numbers measured on the two real ZWO fields before the star-catalogue
refactor begins (docs/STARCAT_DESIGN.md §1.1). The Gaia archive is replayed from a saved
response, so these tests need no network -- only the triangle database for the plate
solve, hence the `slow` marker.

The guess-date assertions are the sharpest check in the suite: the pipeline is told the
field was taken on 2020-01-01 and recovers the true 2023-10-29 from proper motions alone.
Break the projection, the matching, the epoch handling or the polynomial fit, and the
recovered date moves by far more than the tolerances below.
"""

import json
import zipfile

import numpy as np
import pytest

from tests.fixture_catalogue import build_centroid_zip, install, load_field

TRUE_DATE = '2023-10-29'          # from the FITS DATE-OBS of both example fields
BLIND_START = '2020-01-01'        # what guess_date is seeded with

# field, order, expected rms (mas), expected stars, expected guessed date, max day error
CASES = [
    ('zwo3_zenith', 'cubic',   113.0, 433,  '2023-11-02',  10),
    ('zwo3_zenith', 'quintic', 109.6, 434,  '2023-10-28',  10),
    ('zwo3_zenith', 'septic',  107.2, 434,  '2023-10-27',  10),
    ('zwo1_zenith', 'quintic', 111.9, 1564, '2023-09-06',  90),
    ('zwo1_zenith', 'septic',  100.9, 1564, '2023-09-20',  90),
]


def run_stage2(monkeypatch, tmp_path, options, field_name, order, guess_date):
    from mee2024 import distortion_fitter

    tmp_path.mkdir(parents=True, exist_ok=True)
    install(monkeypatch, field_name, options)
    options['distortionOrder'] = order
    options['guess_date'] = guess_date
    options['observation_date'] = TRUE_DATE
    options['DEFAULT_DATE'] = BLIND_START
    options['output_dir'] = str(tmp_path)
    options['no_plot'] = True

    zip_in = build_centroid_zip(field_name, tmp_path / 'centroid_data.zip')
    out_zip = distortion_fitter.match_and_fit_distortion(str(zip_in), options, None)
    with zipfile.ZipFile(out_zip) as z:
        return json.load(z.open('distortion_results.txt'))


def day_difference(a, b):
    import datetime
    return abs((datetime.date.fromisoformat(a) - datetime.date.fromisoformat(b)).days)


@pytest.mark.slow
@pytest.mark.parametrize('field,order,rms_mas,nstars,expected_date,day_tol', CASES)
def test_stage2_reproduces_measured_results(monkeypatch, tmp_path, options,
                                            field, order, rms_mas, nstars,
                                            expected_date, day_tol):
    result = run_stage2(monkeypatch, tmp_path, options, field, order, guess_date=True)

    assert result['final rms error (arcseconds)'] * 1000 == pytest.approx(rms_mas, abs=3.0)
    assert result['#stars used'] == pytest.approx(nstars, abs=8)
    assert result['date_guessed?'] is True

    guessed = result['observation_date']
    assert day_difference(guessed, expected_date) <= day_tol, (
        f'guessed {guessed}, previously {expected_date}')


@pytest.mark.slow
@pytest.mark.parametrize('field,order,expected_days', [
    ('zwo3_zenith', 'quintic', 5),     # measured: 1 day out
    ('zwo3_zenith', 'septic', 5),      # measured: 2 days out
])
def test_guess_date_recovers_the_true_date_blind(monkeypatch, tmp_path, options,
                                                 field, order, expected_days):
    """Seeded with 2020-01-01, recover 2023-10-29 to within a few days.

    Nothing about the telescope, the pointing or the date is supplied -- the only
    information is the pattern of proper motions across the field.
    """
    result = run_stage2(monkeypatch, tmp_path, options, field, order, guess_date=True)
    error_days = day_difference(result['observation_date'], TRUE_DATE)
    assert error_days <= expected_days, (
        f"guessed {result['observation_date']}, true {TRUE_DATE}, "
        f'off by {error_days} days')


@pytest.mark.slow
def test_nn_correlation_flags_an_underfitted_distortion(monkeypatch, tmp_path, options):
    """The wide field needs a high order; nn_corr is what reveals it, not rms.

    Cubic on zwo1 leaves 525 mas and a nearest-neighbour error correlation of 0.92.
    Septic brings it to 101 mas and 0.19. This is the signal the auto-calibration mode
    should key on.
    """
    cubic = run_stage2(monkeypatch, tmp_path / 'a', options.copy(),
                       'zwo1_zenith', 'cubic', guess_date=False)
    septic = run_stage2(monkeypatch, tmp_path / 'b', options.copy(),
                        'zwo1_zenith', 'septic', guess_date=False)

    assert cubic['nearest-neighbour error correlation'] > 0.8
    assert septic['nearest-neighbour error correlation'] < 0.35
    assert cubic['final rms error (arcseconds)'] > 3 * septic['final rms error (arcseconds)']


@pytest.mark.slow
def test_platescale_is_stable_across_distortion_orders(monkeypatch, tmp_path, options):
    """Plate scale is the quantity the deflection constant is degenerate with.

    It must not depend on the distortion order at more than the 1e-3 relative level.
    """
    scales = []
    for i, order in enumerate(['cubic', 'quintic', 'septic']):
        result = run_stage2(monkeypatch, tmp_path / str(i), options.copy(),
                            'zwo3_zenith', order, guess_date=False)
        scales.append(result['platescale (arcseconds/pixel)'])
    assert np.ptp(scales) / np.mean(scales) < 1e-3, f'platescales {scales}'


@pytest.mark.slow
def test_fixture_catalogue_covers_the_whole_field(monkeypatch, tmp_path, options):
    """Guard: if the saved Gaia response did not cover the field, everything else lies."""
    catalogue = install(monkeypatch, 'zwo3_zenith', options)
    field = load_field('zwo3_zenith')
    assert len(field['centroids']) == field['n_centroids_kept']
    assert len(catalogue.rows) > 400
    mags = catalogue.rows['phot_g_mean_mag']
    assert mags.min() < 6.0, 'expected at least one bright star in the field'
    assert mags.max() > 11.5, 'fixture should reach the magnitude limit'
    # every star must carry a usable identifier
    assert np.all(catalogue.rows['source_id'] > 0)
    assert len(np.unique(catalogue.rows['source_id'])) == len(catalogue.rows)
