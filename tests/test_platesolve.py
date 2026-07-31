"""The plate solver: its statistical acceptance test, and real fields end to end."""

import json
from pathlib import Path

import numpy as np
import pytest

from mee2024 import platesolve_triangle as pst
from tests.fixture_catalogue import skip_unless_triangle_db

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


def _exact_threshold(n_obs, N_cat, theta, p_fail=1e-3, density=None):
    """Smallest k with P(max of N Poisson draws >= k) <= p_fail -- the ground truth."""
    import math
    from scipy import stats
    p = (density * math.pi * theta**2 if density is not None
         else N_cat * theta**2 / 4)
    lam = p * (n_obs - 3)
    N = math.comb(N_cat, 3) * math.comb(18, 3) * pst.TOLERANCE**2
    for k in range(3, 400):
        log_cdf = stats.poisson.logcdf(k - 1, lam)
        if -np.expm1(N * log_cdf) <= p_fail:
            return k + 3
    raise AssertionError('no threshold found')


@pytest.mark.parametrize('n_obs,theta_arcsec', [(30, 36.0), (100, 5.0), (100, 36.0),
                                                (100, 120.0), (300, 36.0)])
def test_acceptance_threshold_beats_the_exact_quantile(n_obs, theta_arcsec):
    """The Lambert-W approximation plus the +3 addon must cover the exact 1e-3 quantile."""
    theta = np.radians(theta_arcsec / 3600)
    estimated = pst.estimate_acceptance_threshold(n_obs, 1034887, theta, 18, addon=3)
    exact = _exact_threshold(n_obs, 1034887, theta)
    assert estimated >= exact, f'estimator {estimated} below exact quantile {exact}'
    assert estimated <= exact + 8, f'estimator {estimated} wastefully above exact {exact}'


@pytest.mark.parametrize('density_factor', [3.0, 5.0, 10.0])
def test_acceptance_threshold_tracks_local_density(density_factor):
    """The galactic plane runs 3-10x the mean star density.

    Without the local_density parameter the threshold is unsafe by 10-25 matches there;
    with it, the estimator must again cover the exact quantile.
    """
    theta = np.radians(36 / 3600)
    N_cat = 1034887
    rho = density_factor * N_cat / (4 * np.pi)
    estimated = pst.estimate_acceptance_threshold(100, N_cat, theta, 18, addon=3,
                                                  local_density=rho)
    exact = _exact_threshold(100, N_cat, theta, density=rho)
    assert estimated >= exact, (
        f'at {density_factor}x density: estimator {estimated} below exact {exact}')


def test_acceptance_threshold_without_density_matches_the_mean_density():
    """local_density = N_cat/(4 pi) must reproduce the isotropic default exactly."""
    theta = np.radians(36 / 3600)
    N_cat = 1034887
    default = pst.estimate_acceptance_threshold(100, N_cat, theta, 18)
    explicit = pst.estimate_acceptance_threshold(100, N_cat, theta, 18,
                                                 local_density=N_cat / (4 * np.pi))
    assert default == explicit


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
    skip_unless_triangle_db()
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


# ------------------------------------------------- synthetic fields (slow)

def _offline_catalogue_or_skip():
    from mee2024.starcat import providers
    try:
        return providers.GaiaOfflineProvider.from_installed(['gaia_dr3_g12'])
    except Exception:
        pytest.skip('gaia_dr3_g12 not installed; build it with tools/build_gaia_offline.py')


def _solver_ready_or_skip(solver, options):
    """Point options at the requested solver, skipping if its database is absent."""
    if solver == 'v2':
        from mee2024.platesolve2 import pattern_db
        try:
            pattern_db.resolve({})
        except RuntimeError:
            pytest.skip('no v2 pattern database; build one with '
                        '`mee2024 build-pattern-db`')
        options['platesolver'] = 'v2'
    else:
        skip_unless_triangle_db()


@pytest.mark.slow
@pytest.mark.parametrize('solver', ['triangle', 'v2'])
@pytest.mark.parametrize('fov,ra,dec', [(2.0, 210.0, 35.0), (6.0, 103.0, -5.0)])
def test_platesolve_solves_synthetic_fields(options, solver, fov, ra, dec):
    """Ground-truth fields synthesized from the offline Gaia catalogue must solve."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.synthetic_field import synthesize_field, solution_matches_truth

    _solver_ready_or_skip(solver, options)
    catalogue = _offline_catalogue_or_skip()
    centroids, truth = synthesize_field(catalogue, ra, dec, roll_deg=57.0,
                                        fov_width_deg=fov, seed=int(fov * 10))
    result = pst.platesolve(centroids, (2000, 3000), options=options)
    assert solution_matches_truth(result, truth), (
        f'failed to recover fov={fov} at ({ra}, {dec}): {result.get("ra")}')


@pytest.mark.slow
@pytest.mark.parametrize('solver', ['triangle', 'v2'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_platesolve_rejects_junk_fields(options, solver, seed):
    """Uniform random centroids contain no sky; accepting one is a false positive."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.synthetic_field import junk_field

    _solver_ready_or_skip(solver, options)
    result = pst.platesolve(junk_field((2000, 3000), n=120, seed=seed), (2000, 3000),
                            options=options)
    assert not result['success'], 'accepted a field of pure noise'
