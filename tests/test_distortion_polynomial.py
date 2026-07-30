"""The distortion basis and the OLS fit that removes optical distortion."""

import json

import numpy as np
import pytest

from mee2024 import distortion_polynomial as dp
from mee2024 import transforms


ORDERS = ['linear', 'quadratic', 'cubic', 'quartic', 'quintic', 'sextic', 'septic']


@pytest.mark.parametrize('order', ORDERS)
def test_basis_width_matches_coefficient_names(options, order):
    """get_coeff_names includes the constant term; get_basis does not."""
    options['distortionOrder'] = order
    y = np.linspace(-100, 100, 20)
    x = np.linspace(-80, 80, 20)
    basis = dp.get_basis(y, x, w=100.0, m=1, options=options)
    names = dp.get_coeff_names(options)
    assert basis.shape == (20, len(names) - 1)


@pytest.mark.parametrize('order', ORDERS)
def test_basis_term_count_is_the_triangular_number(options, order):
    options['distortionOrder'] = order
    n = dp.mapping[order]
    expected = (n + 2) * (n + 1) // 2 - 1
    basis = dp.get_basis(np.zeros(5), np.zeros(5), w=1.0, m=1, options=options)
    assert basis.shape[1] == expected


def test_coeff_names_are_readable():
    options = {'distortionOrder': 'quadratic'}
    assert dp.get_coeff_names(options) == ['1', 'x', 'y', 'x^2', 'x * y', 'y^2']


class _FakeStarData:
    """Just enough of StarData for do_cubic_fit: it only calls get_vectors()."""

    def __init__(self, vectors):
        self._vectors = vectors

    def get_vectors(self):
        return self._vectors


def _cubic_fit_setup(options, seed=11):
    options['distortionOrder'] = 'cubic'
    options['distortion_fixed_coefficients'] = 'None'
    options['no_plot'] = True
    img_shape = (2000, 3000)
    w = max(img_shape) / 2
    truth = (np.radians(1.5 / 3600), 1.0, 0.3, 0.5)  # platescale, ra, dec, roll
    rng = np.random.default_rng(seed)
    plate = np.c_[rng.uniform(-img_shape[0] / 2, img_shape[0] / 2, 400),
                  rng.uniform(-img_shape[1] / 2, img_shape[1] / 2, 400)]
    return img_shape, w, truth, plate


def test_do_cubic_fit_recovers_its_own_parameterisation(options):
    """Distort exactly the way apply_corrections models it, and demand exact recovery.

    Only cubic terms are used: _get_corrected_q absorbs the constant and linear terms
    into (platescale, ra, dec, roll), so those would not come back as coefficients.
    """
    img_shape, w, truth, plate_observed = _cubic_fit_setup(options)

    names = dp.get_coeff_names(options)[1:]  # drop the constant
    cx_true = np.zeros(len(names))
    cy_true = np.zeros(len(names))
    cx_true[names.index('x^3')] = 3.0
    cx_true[names.index('x * y^2')] = 1.5
    cy_true[names.index('y^3')] = 2.5
    cy_true[names.index('x^2 * y')] = -1.0

    basis = dp.get_basis(plate_observed[:, 0], plate_observed[:, 1], w, 1, options)
    plate_true = plate_observed + np.c_[basis @ cy_true, basis @ cx_true]
    stardata = _FakeStarData(transforms.linear_transform(truth, plate_true))

    q, plate_corrected, coeff_x, coeff_y, _ = dp.do_cubic_fit(
        plate_observed, stardata, truth, img_shape, options)

    assert np.allclose(plate_corrected, plate_true, atol=1e-6)
    assert np.allclose(coeff_x[1:], cx_true, atol=1e-6)
    assert np.allclose(coeff_y[1:], cy_true, atol=1e-6)
    # the linear solution should be untouched by a purely cubic distortion
    assert np.allclose(q, truth, rtol=1e-8, atol=1e-10)


def test_do_cubic_fit_inverts_a_forward_distortion_to_below_a_milliarcsecond(options):
    """Real optics distort the *true* position; the model corrects the *observed* one.

    Those are inverse maps, so they cannot agree exactly -- the mismatch is second
    order, of size D*grad(D). This test pins how large that residual actually is, so a
    regression that made it worse would be caught.
    """
    img_shape, w, truth, plate_true = _cubic_fit_setup(options)

    yn, xn = plate_true[:, 0] / w, plate_true[:, 1] / w
    dx = 3.0 * xn ** 3 + 1.5 * xn * yn ** 2
    dy = 2.5 * yn ** 3 - 1.0 * yn * xn ** 2
    plate_observed = plate_true - np.c_[dy, dx]
    stardata = _FakeStarData(transforms.linear_transform(truth, plate_true))

    q, plate_corrected, _, _, _ = dp.do_cubic_fit(
        plate_observed, stardata, truth, img_shape, options)

    residual_px = np.linalg.norm(plate_corrected - plate_true, axis=1)
    assert np.max(residual_px) < 0.01, f'max residual {np.max(residual_px)} pixels'

    sky = transforms.linear_transform(q, plate_corrected, img_shape)
    err_arcsec = np.degrees(np.linalg.norm(sky - stardata.get_vectors(), axis=1)) * 3600
    # 1.5 arcsec/pixel * 0.01 pixel -- comfortably below the 0.35 arcsec signal
    assert np.max(err_arcsec) < 0.015


def test_do_cubic_fit_is_a_no_op_on_undistorted_data(options):
    options['distortionOrder'] = 'cubic'
    options['distortion_fixed_coefficients'] = 'None'
    options['no_plot'] = True

    img_shape = (1000, 1000)
    truth = (np.radians(2.0 / 3600), 2.0, -0.2, 1.1)
    rng = np.random.default_rng(5)
    plate = rng.uniform(-500, 500, size=(200, 2))
    stardata = _FakeStarData(transforms.linear_transform(truth, plate))

    q, plate_corrected, coeff_x, coeff_y, _ = dp.do_cubic_fit(
        plate, stardata, truth, img_shape, options)

    assert np.allclose(plate_corrected, plate, atol=1e-6)
    # every non-constant coefficient should be ~0
    assert np.max(np.abs(coeff_x[1:])) < 1e-6
    assert np.max(np.abs(coeff_y[1:])) < 1e-6


def test_open_distortion_files_rejects_mismatched_order(tmp_path, options):
    ref = tmp_path / 'ref.txt'
    ref.write_text(json.dumps({
        'platescale (arcseconds/pixel)': 1.0,
        'distortion order': 'quintic',
        'distortion coeffs x': {'1': 0.0, 'x': 0.0},
        'distortion coeffs y': {'1': 0.0, 'x': 0.0},
    }), encoding='utf-8')
    options['distortion_reference_files'] = str(ref)
    options['distortionOrder'] = 'cubic'
    with pytest.raises(Exception, match='not consistent'):
        dp._open_distortion_files(options)


def test_open_distortion_files_averages_coefficients(tmp_path, options):
    def write(path, scale, cx):
        path.write_text(json.dumps({
            'platescale (arcseconds/pixel)': scale,
            'distortion order': 'cubic',
            'distortion coeffs x': {'1': 0.0, 'x': cx},
            'distortion coeffs y': {'1': 0.0, 'x': 0.0},
        }), encoding='utf-8')

    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    write(a, 1.0, 2.0)
    write(b, 3.0, 4.0)
    options['distortion_reference_files'] = f'{a};{b}'
    options['distortionOrder'] = 'cubic'

    coeff_x, coeff_y, platescale, _ = dp._open_distortion_files(options)
    assert coeff_x['x'] == pytest.approx(3.0)
    assert platescale == pytest.approx(2.0)


# --------------------------------------------------- advanced analysis payload

def _payload(options, n_stars=40, seed=5, bins=8):
    """A payload built from a known cubic distortion, with a known residual added."""
    options['distortionOrder'] = 'cubic'
    options['residual_bins'] = bins
    img_shape = (400, 600)
    rng = np.random.default_rng(seed)
    plate = np.c_[rng.uniform(-200, 200, n_stars), rng.uniform(-300, 300, n_stars)]
    corrections = np.c_[plate[:, 0] * 1e-3, plate[:, 1] * 2e-3]
    residuals = np.c_[rng.normal(0, 0.05, n_stars), rng.normal(0, 0.05, n_stars)]
    n_coeff = len(dp.get_coeff_names(options))
    coeff_x = np.zeros(n_coeff)
    coeff_y = np.zeros(n_coeff)
    coeff_x[1] = 0.5     # a linear term in the basis, so the surface is not flat
    coeff_y[2] = -0.3
    payload = dp.analysis_payload(plate, corrections, residuals, coeff_x, coeff_y,
                                  img_shape, options, platescale_arcsec=1.5)
    return payload, plate, corrections, residuals


def test_analysis_payload_is_json_serialisable(options):
    payload, _, _, _ = _payload(options)
    json.dumps(payload)          # must not raise: it travels to the frontend as JSON


def test_analysis_payload_reports_measured_displacement_not_the_fit(options):
    """A star's plotted height must be fit + residual, so it sits off the surface by
    exactly its residual -- that offset is the whole point of the view."""
    payload, plate, corrections, residuals = _payload(options)
    measured_x = np.array(payload['stars']['dx'])
    expected = corrections[:, 1] - residuals[:, 1]
    assert measured_x == pytest.approx(expected, abs=1e-3)


def test_analysis_payload_keeps_residuals_separately(options):
    payload, _, _, residuals = _payload(options)
    assert np.array(payload['stars']['rx']) == pytest.approx(residuals[:, 1], abs=1e-3)
    assert np.array(payload['stars']['ry']) == pytest.approx(residuals[:, 0], abs=1e-3)


def test_analysis_payload_surface_grid_is_rectangular(options):
    payload, _, _, _ = _payload(options)
    surface = payload['surface']
    ny, nx = len(surface['y']), len(surface['x'])
    assert len(surface['dx']) == ny and len(surface['dy']) == ny
    assert all(len(row) == nx for row in surface['dx'])
    assert all(len(row) == nx for row in surface['dy'])


def test_analysis_payload_surface_spans_the_image(options):
    payload, _, _, _ = _payload(options)
    assert payload['image_size'] == [400, 600]
    # pixels from the centre, matching distortion_field and the star coordinates
    assert min(payload['surface']['x']) == pytest.approx(-300)
    assert max(payload['surface']['x']) == pytest.approx(300)
    assert min(payload['surface']['y']) == pytest.approx(-200)


def test_analysis_payload_carries_the_bin_count_and_order(options):
    payload, _, _, _ = _payload(options)
    assert payload['bins'] == 8
    assert payload['order'] == 'cubic'
    assert payload['platescale'] == pytest.approx(1.5)


def test_analysis_payload_star_positions_are_x_then_y(options):
    """plate is stored (y, x); the payload must expose them the right way round."""
    payload, plate, _, _ = _payload(options)
    assert np.array(payload['stars']['x']) == pytest.approx(plate[:, 1], abs=1e-3)
    assert np.array(payload['stars']['y']) == pytest.approx(plate[:, 0], abs=1e-3)


# ------------------------------------------------- residual-map bin selection

@pytest.mark.parametrize('n_stars,expected', [
    (430, 7),      # a typical field: ~8.8 stars per cell
    (2000, 16),
    (4608, 24),    # the ceiling, reached at 24*24*8 stars
    (100000, 24),  # and held there
])
def test_bins_are_chosen_from_the_star_count(n_stars, expected):
    assert dp.suggest_residual_bins(n_stars) == expected


def test_bins_never_go_below_the_floor():
    """Even a handful of stars must produce a grid rather than a single cell."""
    for n in (0, 1, 5, 20, 60):
        assert dp.suggest_residual_bins(n) == 4


def test_bins_never_exceed_the_ceiling():
    assert dp.suggest_residual_bins(10**7) == 24


def test_bin_count_keeps_cells_populated():
    """The whole point: enough stars per cell that a cell mean is not noise."""
    for n in (50, 200, 430, 1000, 5000):
        bins = dp.suggest_residual_bins(n)
        per_cell = n / bins**2
        assert per_cell >= 3, f'{n} stars over {bins}x{bins} gives {per_cell:.1f} per cell'


def test_an_explicit_bin_count_overrides_the_choice(options):
    assert dp.suggest_residual_bins(430, configured=20) == 20
    # ...but still cannot ask for something absurd
    assert dp.suggest_residual_bins(430, configured=500) == 32
    assert dp.suggest_residual_bins(430, configured=1) == 4


def test_zero_means_automatic(options):
    assert dp.suggest_residual_bins(430, configured=0) == 7


def test_payload_chooses_bins_from_its_own_star_count(options):
    payload, _, _, _ = _payload(options, n_stars=430, bins=0)
    assert payload['bins'] == dp.suggest_residual_bins(430) == 7
    sparse, _, _, _ = _payload(options, n_stars=40, bins=0)
    assert sparse['bins'] == 4
