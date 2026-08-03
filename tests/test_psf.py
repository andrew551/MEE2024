"""
Measuring the point spread function.

The measurements drive decisions — which centroiding algorithm is honest (sampling), is the
focuser tilted (variation), is the star round (diagnostics) — so what these tests pin is
that known synthetic profiles come back with their known parameters, and that the guards
(saturation, crowding, edges) actually guard.
"""

import numpy as np
import pytest

from mee2024 import psf


def moffat_star(shape, cy, cx, fwhm=2.8, beta=3.0, flux=50000.0, ellipticity=0.0,
                angle=0.0, oversample=5):
    """A pixel-integrated elliptical Moffat star, the profile real frames showed."""
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    step = 1.0 / oversample
    fine = np.arange(-0.5 + step / 2, shape[0] - 0.5, step)
    xs, ys = np.meshgrid(fine, fine)
    minor = 1.0 - ellipticity
    ct, st = np.cos(angle), np.sin(angle)
    xr = (xs - cx) * ct + (ys - cy) * st
    yr = (-(xs - cx) * st + (ys - cy) * ct) / minor
    model = (1 + (np.hypot(xr, yr) / alpha) ** 2) ** (-beta)
    binned = model.reshape(shape[0], oversample, shape[0], oversample).mean(axis=(1, 3))
    return flux * binned / binned.sum()


def field_with_stars(positions, size=256, fwhm=2.8, flux=50000.0, noise=3.0, seed=1,
                     background=100.0, **star_kwargs):
    rng = np.random.default_rng(seed)
    image = rng.normal(background, noise, (size, size))
    for (cy, cx) in positions:
        row, col = int(cy), int(cx)
        stamp = 25
        star = moffat_star((2 * stamp + 1,) * 2, stamp + (cy - row), stamp + (cx - col),
                           fwhm=fwhm, flux=flux, **star_kwargs)
        image[row - stamp:row + stamp + 1, col - stamp:col + stamp + 1] += star
    return image


GRID = [(40.3, 50.7), (60.2, 180.4), (120.8, 90.1), (180.5, 200.6), (200.1, 60.3),
        (90.4, 220.2), (150.7, 150.2), (220.6, 130.8)]


# ------------------------------------------------------------------- measurement

def test_fwhm_of_a_gaussian_field_is_recovered_exactly():
    """beta=30 is Gaussian to within a percent, so the fit must return the truth."""
    image = field_with_stars(GRID, fwhm=2.8, beta=30.0)
    stars, summary = psf.measure_field(image, GRID)
    assert summary['n_stars'] >= 6
    assert summary['fwhm_px'] == pytest.approx(2.8, rel=0.04)


def test_moffat_wings_pull_the_gaussian_fit_wide_as_the_physics_says():
    """On a beta=3 profile the Gaussian-equivalent FWHM runs ~10-15% above the Moffat's
    true FWHM -- the wings drag it wide. Measured at +15% on the real eclipse field, so
    the summary number is a Gaussian-equivalent width and this pins that convention."""
    image = field_with_stars(GRID, fwhm=2.8, beta=3.0)
    _, summary = psf.measure_field(image, GRID)
    assert 2.85 < summary['fwhm_px'] < 3.35


def test_a_sharp_field_is_flagged_undersampled():
    image = field_with_stars(GRID, fwhm=1.4)
    _, summary = psf.measure_field(image, GRID)
    assert summary['undersampled'] is True


def test_a_soft_field_is_not_flagged():
    image = field_with_stars(GRID, fwhm=3.2)
    _, summary = psf.measure_field(image, GRID)
    assert summary['undersampled'] is False


def test_ellipticity_and_angle_are_recovered():
    image = field_with_stars(GRID, fwhm=3.0, ellipticity=0.25, angle=0.5)
    stars, summary = psf.measure_field(image, GRID)
    assert summary['ellipticity'] == pytest.approx(0.25, abs=0.06)
    fitted = [s['angle'] for s in stars if s['fit_rms'] is not None]
    # angles are mod pi; compare on the doubled circle
    doubled = np.exp(2j * np.array(fitted))
    assert np.angle(doubled.mean()) / 2 == pytest.approx(0.5, abs=0.15)


def test_round_stars_report_near_zero_ellipticity():
    image = field_with_stars(GRID, fwhm=2.8)
    _, summary = psf.measure_field(image, GRID)
    assert summary['ellipticity'] < 0.08


def test_platescale_converts_to_arcsec():
    image = field_with_stars(GRID, fwhm=2.8)
    _, summary = psf.measure_field(image, GRID, platescale_arcsec=1.5)
    assert summary['fwhm_arcsec'] == pytest.approx(summary['fwhm_px'] * 1.5)


def test_fit_cap_fits_only_the_brightest():
    image = field_with_stars(GRID, fwhm=2.8)
    stars, _ = psf.measure_field(image, GRID, fit_top_n=3)
    fitted = sum(1 for s in stars if s['fit_rms'] is not None)
    assert fitted <= 3
    assert len(stars) > 3, 'the rest must still be measured by moments'


# ------------------------------------------------------------------------ guards

def test_a_saturated_star_is_excluded():
    image = field_with_stars(GRID, fwhm=2.8)
    image[int(GRID[0][0]) - 1:int(GRID[0][0]) + 2,
          int(GRID[0][1]) - 1:int(GRID[0][1]) + 2] = image.max() * 3
    # a clipping plateau: several pixels at the exact same top value
    image = np.minimum(image, np.sort(image.ravel())[-6])
    stars, summary = psf.measure_field(image, GRID)
    assert summary['excluded']['saturated'] >= 1


def test_a_close_pair_is_excluded_as_crowded():
    pair = GRID + [(GRID[0][0] + 6, GRID[0][1] + 6)]
    image = field_with_stars(pair, fwhm=2.8)
    _, summary = psf.measure_field(image, pair)
    assert summary['excluded']['crowded'] >= 2


def test_an_edge_star_is_dropped_not_mismeasured():
    positions = GRID + [(3.0, 128.0)]
    image = field_with_stars(GRID, fwhm=2.8)      # nothing actually drawn at the edge
    _, summary = psf.measure_field(image, positions)
    assert summary['excluded']['edge'] >= 1


# ------------------------------------------------------------- profiles and fits

def test_moffat_beta_is_recovered_from_a_radial_profile():
    alpha, beta = 2.0, 3.0
    radii = np.linspace(0.1, 12, 200)
    values = 1000.0 * (1 + (radii / alpha) ** 2) ** (-beta)
    fit = psf.fit_radial_moffat(radii, values)
    assert fit is not None
    assert fit[2] == pytest.approx(beta, rel=0.05)
    assert fit[1] == pytest.approx(alpha, rel=0.05)


def test_gaussian_fit_centroid_is_subpixel_accurate():
    truth = (25.0 + 0.37, 25.0 - 0.21)
    star = moffat_star((51, 51), truth[0], truth[1], fwhm=2.8, beta=30.0)  # ~Gaussian
    rng = np.random.default_rng(2)
    fit = psf.fit_gaussian(star + rng.normal(0, 0.5, star.shape), noise=0.5)
    assert fit is not None
    assert fit['cy'] == pytest.approx(truth[0], abs=0.05)
    assert fit['cx'] == pytest.approx(truth[1], abs=0.05)


def test_windowed_moments_track_a_shifted_star():
    star = moffat_star((21, 21), 10.4, 9.6, fwhm=2.6)
    moments = psf.windowed_moments(star)
    assert moments is not None
    assert moments[0] == pytest.approx(10.4, abs=0.08)
    assert moments[1] == pytest.approx(9.6, abs=0.08)


def test_stacked_profile_needs_enough_stars():
    image = field_with_stars(GRID[:3], fwhm=2.8)
    stars, _ = psf.measure_field(image, GRID[:3])
    profile, used = psf.stacked_profile(image, stars)
    assert profile is None and used == 0


# -------------------------------------------------------------------- the payload

def test_event_payload_is_json_serialisable_and_complete():
    import json

    image = field_with_stars(GRID, fwhm=2.8)
    payload = psf.event_payload(image, GRID, platescale_arcsec=1.5)
    assert payload is not None
    json.dumps(payload)
    assert payload['summary']['fwhm_px'] > 0
    assert set(payload['stars']) == {'x', 'y', 'fwhm', 'e1', 'e2'}
    assert payload['image_size'] == [256, 256]
    assert len(payload['stars']['x']) == payload['summary']['n_stars']


def test_event_payload_declines_a_hopeless_frame():
    rng = np.random.default_rng(0)
    image = rng.normal(100, 3, (128, 128))
    assert psf.event_payload(image, [(64.0, 64.0)]) is None
