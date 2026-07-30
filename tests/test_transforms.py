"""Projection round trips. These underpin every arcsecond the pipeline reports."""

import numpy as np
import pytest

from mee2024 import transforms


def random_solutions(n=12, seed=1):
    """Plausible (platescale, ra, dec, roll) tuples in radians."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        platescale = np.radians(rng.uniform(0.2, 5.0) / 3600)  # 0.2-5 arcsec/pixel
        ra = rng.uniform(0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-0.95, 0.95))  # avoid the poles
        roll = rng.uniform(0, 2 * np.pi)
        yield (platescale, ra, dec, roll)


@pytest.mark.parametrize('x', list(random_solutions()))
def test_linear_transform_detransform_round_trip(x):
    """plate -> sky -> plate must return the original pixel coordinates."""
    rng = np.random.default_rng(7)
    plate = rng.uniform(-1000, 1000, size=(50, 2))

    vectors = transforms.linear_transform(x, plate)
    back = transforms.detransform_vectors(x, vectors)

    # sub-milli-pixel: any larger error would swamp the sub-arcsecond target
    assert np.allclose(back, plate, atol=1e-6)


def test_linear_transform_produces_unit_vectors():
    x = (np.radians(1.0 / 3600), 1.2, 0.4, 0.9)
    plate = np.random.default_rng(3).uniform(-500, 500, size=(30, 2))
    vectors = transforms.linear_transform(x, plate)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-12)


def test_to_polar_returns_dec_then_ra_in_degrees():
    """to_polar's column order is (dec, ra) -- easy to get backwards, so pin it."""
    ra, dec = np.radians(123.0), np.radians(-45.0)
    v = np.array([[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]])
    polar = transforms.to_polar(v)
    assert polar.shape == (1, 2)
    assert polar[0, 0] == pytest.approx(-45.0)
    assert polar[0, 1] == pytest.approx(123.0)


def test_to_polar_wraps_ra_into_0_360():
    ra, dec = np.radians(350.0), np.radians(10.0)
    v = np.array([[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]])
    assert transforms.to_polar(v)[0, 1] == pytest.approx(350.0)


def test_icoord_to_vector_does_not_mutate_its_input():
    """Regression: reshape can return a view, and the function writes to column 1."""
    icoords = np.array([[0.01, 0.02], [-0.03, 0.04]])
    original = icoords.copy()
    transforms.icoord_to_vector(icoords)
    assert np.array_equal(icoords, original), 'icoord_to_vector modified its argument'


def test_icoord_to_vector_preserves_leading_shape():
    icoords = np.zeros((4, 5, 2))
    assert transforms.icoord_to_vector(icoords).shape == (4, 5, 3)


def test_icoord_to_vector_origin_maps_to_x_axis():
    v = transforms.icoord_to_vector(np.array([[0.0, 0.0]]))
    assert np.allclose(v, [[1.0, 0.0, 0.0]])
