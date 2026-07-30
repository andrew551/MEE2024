"""The plate solver: its statistical acceptance test, and real fields end to end."""

import json
from pathlib import Path

import numpy as np
import pytest

from mee2024 import platesolve_triangle as pst

FIELDS_DIR = Path(__file__).parent / 'data' / 'fields'


def load_field(name):
    return json.loads((FIELDS_DIR / f'{name}.json').read_text(encoding='utf-8'))


def all_field_names():
    return sorted(p.stem for p in FIELDS_DIR.glob('*.json'))


# ------------------------------------------------------- acceptance threshold

def test_acceptance_threshold_is_a_plausible_star_count():
    """A few thousand catalogue stars and an arcsecond match radius: a handful of stars."""
    thresh = pst.estimate_acceptance_threshold(
        n_obs=100, N_stars_catalog=300000, threshold_match=np.radians(10 / 3600),
        g=18, addon=3)
    assert 4 <= thresh <= 30


def test_acceptance_threshold_rises_with_a_looser_match_radius():
    """A sloppier match makes chance coincidences likelier, so demand more stars."""
    tight = pst.estimate_acceptance_threshold(
        100, 300000, np.radians(1 / 3600), 18)
    loose = pst.estimate_acceptance_threshold(
        100, 300000, np.radians(60 / 3600), 18)
    assert loose > tight


def test_acceptance_threshold_rises_with_a_denser_catalogue():
    sparse = pst.estimate_acceptance_threshold(100, 10000, np.radians(10 / 3600), 18)
    dense = pst.estimate_acceptance_threshold(100, 1000000, np.radians(10 / 3600), 18)
    assert dense > sparse


def test_acceptance_threshold_addon_is_additive():
    base = pst.estimate_acceptance_threshold(100, 300000, np.radians(10 / 3600), 18, addon=0)
    plus = pst.estimate_acceptance_threshold(100, 300000, np.radians(10 / 3600), 18, addon=5)
    assert plus == base + 5


# --------------------------------------------------------------- input checks

def test_platesolve_rejects_wrongly_shaped_input():
    with pytest.raises(Exception, match='n by 2'):
        pst.platesolve(np.zeros((10, 3)), (1000, 1000))


def test_find_rotation_matrix_recovers_a_known_rotation():
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', [0.3, -0.2, 1.1]).as_matrix()
    rng = np.random.default_rng(2)
    v = rng.normal(size=(40, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    recovered = pst._find_rotation_matrix(v, (rot @ v.T).T)
    assert np.allclose((recovered.T @ v.T).T, (rot @ v.T).T, atol=1e-10)


# ------------------------------------------------------- real fields (slow)

@pytest.mark.slow
@pytest.mark.parametrize('field_name', all_field_names())
def test_platesolve_recovers_known_real_fields(field_name, options):
    """Regression corpus: centroids captured from real runs must keep solving.

    Needs the 127 MB triangle database, so it only runs under --runslow.
    """
    field = load_field(field_name)
    centroids = np.array(field['centroids'])
    expected = field['expected']

    result = pst.platesolve(centroids, tuple(field['img_shape']), options=options)

    assert result['success'], f'{field_name} failed to platesolve'
    assert result['ra'] == pytest.approx(expected['ra'], abs=0.01)
    assert result['dec'] == pytest.approx(expected['dec'], abs=0.01)
    assert result['roll'] == pytest.approx(expected['roll'], abs=0.05)
    assert result['platescale/arcsec'] == pytest.approx(
        expected['platescale_arcsec'], rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('field_name', all_field_names())
def test_platesolve_platescale_agrees_with_the_optics(field_name):
    """Both example fields are a 420 mm lens with ~3.8 micron pixels: ~1.85 arcsec/px.

    An independent physical cross-check that the solution is not merely self-consistent.
    """
    expected = load_field(field_name)['expected']
    assert 1.80 < expected['platescale_arcsec'] < 1.92
