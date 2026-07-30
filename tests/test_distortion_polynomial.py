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
