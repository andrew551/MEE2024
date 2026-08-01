"""The v2 solver, end to end on a miniature pattern database built in-test.

v1 never had a fast end-to-end test because its Tycho database takes minutes to build.
The v2 builder is parametrised, so a few hundred synthetic stars give a regional
database in well under a second -- and with it: build -> solve -> verify -> contract,
all in CI, no network, no --runslow.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mee2024 import events
from mee2024.platesolve2 import build, geometry, pattern_db
from mee2024.platesolve2 import platesolve as platesolve_v2
from mee2024.starcat.table import StarTable

RA0, DEC0 = 180.0, 10.0        # patch centre, degrees
PATCH_HALF = 3.0               # patch half-width, degrees

MINI_PARAMS = dict(a=50, b=100, d=400, e=18, theta_sep_deg=0.4,
                   theta_pat_deg=1.7, double_star_arcsec=36.0)


def synthetic_star_table(n=400, seed=7):
    rng = np.random.default_rng(seed)
    ra = np.radians(RA0 + rng.uniform(-PATCH_HALF, PATCH_HALF, n)
                    / np.cos(np.radians(DEC0)))
    dec = np.radians(DEC0 + rng.uniform(-PATCH_HALF, PATCH_HALF, n))
    mag = rng.uniform(5.0, 11.0, n).astype(np.float32)
    return StarTable(ra=ra, dec=dec, mag=mag, ids=np.arange(n, dtype=np.int64),
                     epoch=2024.0)


class PatchCatalogue:
    """A tiny provider over one StarTable: the seam both synthesis and verification use."""

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def lookup(self, ra_range, dec_range, max_magnitude=None, epoch=None):
        ra = np.degrees(self.table.ra)
        dec = np.degrees(self.table.dec)
        lo, hi = ra_range
        keep = ((ra >= lo) & (ra <= hi)) if lo < hi else ((ra >= lo) | (ra <= hi))
        keep &= (dec >= min(dec_range)) & (dec <= max(dec_range))
        if max_magnitude is not None:
            keep &= self.table.mag < max_magnitude
        return self.table.select(np.nonzero(keep)[0])


@pytest.fixture(scope='module')
def mini_db(tmp_path_factory):
    stars = synthetic_star_table()
    out_dir, manifest = build.build_pattern_db(
        stars, 'test_mini', out_root=tmp_path_factory.mktemp('patterndb'),
        params=MINI_PARAMS,
        verify_spec={'provider': 'test', 'mag_limit': 12.0, 'epoch': 2024.0})
    return pattern_db.PatternDB(out_dir), PatchCatalogue(stars), manifest


# ------------------------------------------------------------------- format

def test_mini_db_arrays_are_consistent(mini_db):
    db, _, manifest = mini_db
    offsets = np.asarray(db.anchor_tri_offset)
    assert offsets[0] == 0
    assert np.all(np.diff(offsets) >= 0)
    assert offsets[-1] == manifest['n_triangles'] == db.tri_inv.shape[0]
    assert db.anchors.shape == (manifest['n_anchors'], 3)
    # unit vectors, canonical invariant ranges
    assert np.allclose(np.linalg.norm(np.asarray(db.anchors), axis=1), 1, atol=1e-5)
    tri = np.asarray(db.tri_inv)
    assert np.all(tri[:, 0] > 0) and np.all(tri[:, 0] <= 1.0 + 1e-6)
    assert np.all(tri[:, 1] >= 0) and np.all(tri[:, 1] < 2 * np.pi + 1e-6)


def test_sparse_sky_builds_with_partial_patterns(tmp_path):
    """Anchors with fewer than e legs own fewer triangle rows -- the case v1's
    builder answered with 'edge case handling unimplemented!'."""
    stars = synthetic_star_table(n=30, seed=11)
    out_dir, manifest = build.build_pattern_db(
        stars, 'test_sparse', out_root=tmp_path,
        params=dict(MINI_PARAMS, a=10, b=10, d=30))
    db = pattern_db.PatternDB(out_dir)
    per_anchor = np.diff(np.asarray(db.anchor_tri_offset))
    full = 18 * 17 // 2
    assert per_anchor.max() < full          # nobody has 18 legs in a 30-star patch
    assert manifest['n_triangles'] == per_anchor.sum()
    # decode still works on the ragged layout
    if manifest['n_triangles']:
        anchor, j, k = db.triangle_anchor_and_legs(
            np.arange(manifest['n_triangles']))
        assert np.all(j < k)
        n_legs_of_anchor = np.asarray(db.pattern_ind)[anchor] >= 0
        assert np.all(k < n_legs_of_anchor.sum(axis=1))


def test_manifest_pins_dtypes_on_disk(mini_db):
    db, _, manifest = mini_db
    for key, entry in manifest['columns'].items():
        loaded = np.load(db.directory / entry['file'], mmap_mode='r')
        assert loaded.dtype == np.dtype(entry['dtype']), key


def test_verify_passes_and_detects_corruption(mini_db, tmp_path):
    db, _, _ = mini_db
    assert pattern_db.verify(db.directory) == []


def test_resolve_names_the_remedy_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pattern_db, 'get_patterndb_root', lambda: tmp_path)
    with pytest.raises(RuntimeError, match='build-pattern-db'):
        pattern_db.resolve({'pattern_db': 'nonexistent'})
    with pytest.raises(RuntimeError, match='build-pattern-db'):
        pattern_db.resolve({})


# ------------------------------------------------------------------ solving

def _field(catalogue, fov=2.0, seed=3, **kw):
    from tools.synthetic_field import synthesize_field
    return synthesize_field(catalogue, RA0, DEC0, roll_deg=57.0, fov_width_deg=fov,
                            shape=(1000, 1500), mag_limit=12.0, epoch=2024.0,
                            n_detect=60, seed=seed, **kw)


def test_v2_solves_a_synthetic_field(mini_db):
    from tools.synthetic_field import solution_matches_truth
    db, catalogue, _ = mini_db
    centroids, truth = _field(catalogue)
    result = platesolve_v2(centroids, (1000, 1500),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert result['success']
    assert solution_matches_truth(result, truth)
    assert result['mirror'] is False


def test_v2_solves_the_mirrored_field(mini_db):
    from tools.synthetic_field import solution_matches_truth
    db, catalogue, _ = mini_db
    centroids, truth = _field(catalogue)
    mirrored = centroids[:, [1, 0]]
    result = platesolve_v2(mirrored, (1500, 1000),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert result['success']
    assert result['mirror'] is True
    assert solution_matches_truth(result, truth)


def test_v2_rejects_junk(mini_db):
    from tools.synthetic_field import junk_field
    db, catalogue, _ = mini_db
    result = platesolve_v2(junk_field((1000, 1500), n=60, seed=1), (1000, 1500),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert not result['success']


def test_v2_result_contract_and_events(mini_db):
    """The keys, shapes and events the pipeline and UI rely on, v1-identical."""
    db, catalogue, _ = mini_db
    centroids, truth = _field(catalogue)
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        result = platesolve_v2(centroids, (1000, 1500),
                               options={'rough_match_threshhold': 36},
                               catalogue=catalogue, db=db)

    for key in ('success', 'x', 'ra', 'dec', 'roll', 'platescale/arcsec',
                'matched_centroids', 'matched_stars', 'mirror', 'diagnostics'):
        assert key in result, key
    assert len(result['x']) == 4                      # (scale, ra, dec, roll) radians
    assert result['matched_stars'].shape[1] == 6      # ra, dec, 3-vector, mag
    assert result['matched_centroids'].shape[0] == result['matched_stars'].shape[0]
    # catalogue positions in radians, magnitudes in the last column
    assert np.all(np.abs(result['matched_stars'][:, 1]) <= np.pi / 2)
    assert np.all(result['matched_stars'][:, 5] < 12.5)
    assert result['diagnostics']['n_candidates'] > 0

    solve_results = [e for e in sink.events if e['type'] == events.SOLVE_RESULT]
    assert len(solve_results) == 1
    for key in ('success', 'ra', 'dec', 'roll', 'platescale', 'mirror', 'n_matched'):
        assert key in solve_results[0], key
    assert any(e['type'] == events.SOLVE_CANDIDATE for e in sink.events)


# ------------------------------------------------------------ kendall (S2)

@pytest.fixture(scope='module')
def mini_db_kendall(tmp_path_factory):
    stars = synthetic_star_table()
    out_dir, manifest = build.build_pattern_db(
        stars, 'test_mini_kendall', out_root=tmp_path_factory.mktemp('patterndb'),
        params=dict(MINI_PARAMS, invariant='kendall', tolerance=0.004),
        verify_spec={'provider': 'test', 'mag_limit': 12.0, 'epoch': 2024.0})
    return pattern_db.PatternDB(out_dir), PatchCatalogue(stars), manifest


def test_kendall_db_reps_are_unit_norm_with_valid_perms(mini_db_kendall):
    db, _, manifest = mini_db_kendall
    assert manifest['invariant'] == 'kendall'
    xy = np.asarray(db.tri_inv, dtype=np.float64)
    z_sq = 1.0 - xy[:, 0] ** 2 - xy[:, 1] ** 2
    assert np.all(z_sq > -1e-5)                      # on or inside the unit circle
    assert np.all(np.asarray(db.tri_perm) < 6)       # codes 6/7 are impossible


def test_kendall_solves_and_mirror_needs_no_requery(mini_db_kendall):
    from tools.synthetic_field import solution_matches_truth
    db, catalogue, _ = mini_db_kendall
    centroids, truth = _field(catalogue)

    result = platesolve_v2(centroids, (1000, 1500),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert result['success'] and solution_matches_truth(result, truth)
    assert result['mirror'] is False

    mirrored = centroids[:, [1, 0]]
    result_m = platesolve_v2(mirrored, (1500, 1000),
                             options={'rough_match_threshhold': 36},
                             catalogue=catalogue, db=db)
    assert result_m['success'] and result_m['mirror'] is True
    assert solution_matches_truth(result_m, truth)
    # the pointing must agree between the two solves
    assert result_m['ra'] == pytest.approx(result['ra'], abs=1e-3)
    assert result_m['dec'] == pytest.approx(result['dec'], abs=1e-3)


def test_kendall_rejects_junk(mini_db_kendall):
    from tools.synthetic_field import junk_field
    db, catalogue, _ = mini_db_kendall
    result = platesolve_v2(junk_field((1000, 1500), n=60, seed=1), (1000, 1500),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert not result['success']


def test_kendall_agrees_with_ratio_dphi_solution(mini_db, mini_db_kendall):
    """Same field, same anchors -- the two invariants must find the same sky."""
    db1, catalogue, _ = mini_db
    db2, _, _ = mini_db_kendall
    centroids, truth = _field(catalogue)
    r1 = platesolve_v2(centroids, (1000, 1500),
                       options={'rough_match_threshhold': 36},
                       catalogue=catalogue, db=db1)
    r2 = platesolve_v2(centroids, (1000, 1500),
                       options={'rough_match_threshhold': 36},
                       catalogue=catalogue, db=db2)
    assert r1['success'] and r2['success']
    assert r2['ra'] == pytest.approx(r1['ra'], abs=1e-2)
    assert r2['dec'] == pytest.approx(r1['dec'], abs=1e-2)
    assert r2['platescale/arcsec'] == pytest.approx(r1['platescale/arcsec'],
                                                    rel=1e-3)


# ----------------------------------------------------- progressive anchors (S5)

def test_anchor_rounds_rescue_a_field_with_bright_artifacts(mini_db_kendall):
    """Twelve saturated artifacts occupy every brightest-9 rank: round one has no
    real anchor at all, so a single-round solver cannot succeed, and round two
    (ranks 9..17, holding six real stars) must recover the field."""
    from tools.synthetic_field import solution_matches_truth
    db, catalogue, _ = mini_db_kendall
    centroids, truth = _field(catalogue)
    rng = np.random.default_rng(12)
    fakes = np.c_[rng.uniform(2, 998, 12), rng.uniform(2, 1498, 12)]
    poisoned = np.r_[fakes, centroids]

    single = platesolve_v2(poisoned, (1000, 1500),
                           options={'rough_match_threshhold': 36,
                                    'v2_anchor_rounds': 1},
                           catalogue=catalogue, db=db)
    assert not single['success']

    result = platesolve_v2(poisoned, (1000, 1500),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert result['success'] and solution_matches_truth(result, truth)


# ----------------------------------------------------------- bucket index (S5)

def test_bucket_query_is_identical_to_a_kd_tree(mini_db_kendall):
    """The bucket gather must return exactly the KD-tree's candidate sets."""
    from scipy.spatial import KDTree
    db, _, manifest = mini_db_kendall
    assert db.is_bucketed
    xy = np.asarray(db.tri_inv, dtype=np.float64)
    pts3 = np.c_[xy, np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 0.0))]
    tree = KDTree(pts3)

    rng = np.random.default_rng(8)
    queries = pts3[rng.integers(0, len(pts3), 40)] \
        + rng.normal(scale=0.002, size=(40, 3))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    radii = rng.uniform(0.001, 0.02, 40)

    cand, rows, dist = db.query_ball(queries, radii)
    reference = tree.query_ball_point(queries, radii)
    for q in range(40):
        mine = sorted(cand[rows == q].tolist())
        assert mine == sorted(reference[q]), f'query {q} differs'
    # and the returned distances are the true 3-D distances
    xyq = np.asarray(db.tri_inv, dtype=np.float64)[cand]
    z = np.sqrt(np.maximum(1 - xyq[:, 0]**2 - xyq[:, 1]**2, 0.0))
    assert np.allclose(dist, np.linalg.norm(np.c_[xyq, z] - queries[rows], axis=1))


def test_bucketed_db_never_builds_a_tree(mini_db_kendall):
    """Opening and querying must not touch the KD-tree path at all."""
    db, _, _ = mini_db_kendall
    fresh = pattern_db.PatternDB(db.directory)
    fresh.query_ball(np.array([[0.6, 0.1, np.sqrt(1 - 0.37)]]), [0.005])
    assert fresh._kd_tree is None


# ---------------------------------------------------- quaternion consensus (S4)

def test_quaternion_canonicalisation_pairs_boundary_rotations():
    """q and -q are one rotation; near the sum(q)=0 boundary the measured sign is
    arbitrary, and the twin mechanism must keep such a cluster together."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(4)
    # a rotation whose quaternion sits almost exactly on the boundary
    q0 = rng.normal(size=4)
    q0 -= q0.sum() / 4                       # sum(q0) = 0
    q0 /= np.linalg.norm(q0)
    jitter = rng.normal(scale=1e-5, size=(40, 4))
    cluster = q0 + jitter
    cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
    signs = np.where(rng.random(40) < 0.5, 1.0, -1.0)   # measured on either side
    quat = cluster * signs[:, None]

    canonical, twin_idx = geometry.canonicalise_quaternions(quat, band=1e-3)
    assert len(twin_idx) == 40               # all sit inside the boundary band
    # every member must be within a tiny distance of q0 or -q0 after adding twins
    data = np.r_[canonical, -canonical[twin_idx]]
    close_to_plus = np.linalg.norm(data - q0, axis=1) < 1e-3
    close_to_minus = np.linalg.norm(data + q0, axis=1) < 1e-3
    assert np.all(close_to_plus | close_to_minus)
    # and each original rotation is represented on the +q0 side by itself or twin
    represented = close_to_plus[:40] | close_to_plus[40:]
    assert np.all(represented)

    # far from the boundary no twins are made and canonicalisation is stable
    r = Rotation.from_euler('xyz', [0.4, -1.0, 2.2]).as_quat()
    stack = np.r_[[r], [-r]]
    canonical2, twin2 = geometry.canonicalise_quaternions(stack, band=1e-3)
    assert len(twin2) == 0
    assert np.allclose(canonical2[0], canonical2[1])


def test_rotations_to_quaternions_handles_degenerate_rows():
    from scipy.spatial.transform import Rotation
    good = Rotation.from_euler('zyx', [[0.1, 0.2, 0.3], [1.0, -0.5, 2.0]]).as_matrix()
    bad = np.full((1, 3, 3), 1e30)
    quat = geometry.rotations_to_quaternions(np.r_[good, bad])
    assert np.all(np.isfinite(quat))
    # good rows round-trip; the bad row is a far-away placeholder
    assert np.allclose(np.abs(Rotation.from_matrix(good).as_quat()),
                       np.abs(quat[:2]), atol=1e-6)
    assert np.linalg.norm(quat[2]) > 1e5


@pytest.fixture(scope='module')
def mini_db_polar(tmp_path_factory):
    """A star patch covering the celestial pole: the legacy consensus chart's
    singular point."""
    rng = np.random.default_rng(19)
    n = 900
    # a dense 3-degree polar cap: the synthetic generator's RA-window query loses
    # in-frame stars on the far side of the pole, so density compensates to keep
    # the detected count (and hence triangle sizes) representative of the bench's
    # real polar fields
    r = np.degrees(np.arccos(1 - rng.random(n) * (1 - np.cos(np.radians(3.0)))))
    theta = rng.uniform(0, 2 * np.pi, n)
    dec = np.radians(90.0 - r)
    ra = theta
    mag = rng.uniform(5.0, 11.0, n).astype(np.float32)
    stars = StarTable(ra=ra, dec=dec, mag=mag, ids=np.arange(n, dtype=np.int64),
                      epoch=2024.0)
    out_dir, _ = build.build_pattern_db(
        stars, 'test_mini_polar', out_root=tmp_path_factory.mktemp('patterndb'),
        params=dict(MINI_PARAMS, invariant='kendall'),
        verify_spec={'provider': 'test', 'mag_limit': 12.0, 'epoch': 2024.0})
    return pattern_db.PatternDB(out_dir), PatchCatalogue(stars)


def test_quaternion_consensus_solves_at_the_pole(mini_db_polar):
    """dec = +89: the legacy (centre, roll) chart is singular here (the S0 bench
    measured 0/4); the quaternion key has no chart to be singular in.

    Frame and detection count mirror the bench's polar cases: per-candidate
    orientation noise scales as (centroid noise) / (triangle size in px), and the
    consensus tolerance is a fixed 4.4e-4, so an unrealistically small frame makes
    even a perfect metric fragment. (A size-aware consensus radius is S5 work.)
    """
    from tools.synthetic_field import synthesize_field, solution_matches_truth
    db, catalogue = mini_db_polar
    centroids, truth = synthesize_field(catalogue, 180.0, 89.0, roll_deg=57.0,
                                        fov_width_deg=2.4, shape=(2000, 3000),
                                        mag_limit=12.0, epoch=2024.0, n_detect=120,
                                        seed=6)
    result = platesolve_v2(centroids, (2000, 3000),
                           options={'rough_match_threshhold': 36},
                           catalogue=catalogue, db=db)
    assert result['success'] and solution_matches_truth(result, truth)


# -------------------------------------------------------- tolerance model (S3)

def test_query_radius_model_shape(mini_db_kendall):
    """More assumed noise -> wider; larger triangles -> tighter; always capped."""
    from mee2024.platesolve2 import solve
    db, _, _ = mini_db_kendall
    sizes = np.array([200.0, 800.0, 2500.0])
    r_small_eps = solve.query_radii(db, sizes, 0.3)
    r_big_eps = solve.query_radii(db, sizes, 3.0)
    assert np.all(np.diff(r_small_eps) < 0)          # bigger triangle, tighter
    assert np.all(r_big_eps >= r_small_eps)          # noisier, wider
    assert np.all(r_big_eps <= solve.RADIUS_MAX + 1e-12)


def test_fixed_tolerance_override_still_solves(mini_db_kendall):
    """The S2 rollback: a positive v2_fixed_tolerance bypasses the adaptive model."""
    from tools.synthetic_field import solution_matches_truth
    db, catalogue, _ = mini_db_kendall
    centroids, truth = _field(catalogue)
    result = platesolve_v2(centroids, (1000, 1500),
                           options={'rough_match_threshhold': 36,
                                    'v2_fixed_tolerance': 0.005},
                           catalogue=catalogue, db=db)
    assert result['success'] and solution_matches_truth(result, truth)


# ------------------------------------------------------- kendall geometry unit

def _random_triangles_2d(n, rng):
    return rng.normal(size=(n, 3, 2)) * 100


def test_kendall_rep_is_invariant_to_labelling_rotation_and_scale():
    rng = np.random.default_rng(0)
    pts = _random_triangles_2d(50, rng)
    xyz, _ = geometry.kendall_rep_2d(pts[:, 0], pts[:, 1], pts[:, 2])
    # any relabelling of the same vertices gives the same canonical point
    for perm in ((1, 2, 0), (2, 1, 0), (0, 2, 1)):
        xyz_p, _ = geometry.kendall_rep_2d(pts[:, perm[0]], pts[:, perm[1]],
                                           pts[:, perm[2]])
        assert np.allclose(xyz_p, xyz, atol=1e-9), perm
    # rigid rotation plus uniform scale changes nothing
    theta = 0.83
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scaled = 3.7 * pts @ rot.T + np.array([12.0, -4.0])
    xyz_r, _ = geometry.kendall_rep_2d(scaled[:, 0], scaled[:, 1], scaled[:, 2])
    assert np.allclose(xyz_r, xyz, atol=1e-9)


def test_kendall_rep_is_unit_norm():
    rng = np.random.default_rng(1)
    pts = _random_triangles_2d(200, rng)
    xyz, code = geometry.kendall_rep_2d(pts[:, 0], pts[:, 1], pts[:, 2])
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-9)
    assert np.all(xyz[:, 2] >= 0)
    assert np.all(code < 6)


def test_mirrored_triangle_lands_on_the_mirror_point():
    rng = np.random.default_rng(2)
    pts = _random_triangles_2d(100, rng)
    xyz, _ = geometry.kendall_rep_2d(pts[:, 0], pts[:, 1], pts[:, 2])
    flipped = pts[:, :, [1, 0]]     # transpose the pixel plane = mirror the field
    xyz_m, _ = geometry.kendall_rep_2d(flipped[:, 0], flipped[:, 1], flipped[:, 2])
    assert np.allclose(xyz_m, geometry.mirror_rep(xyz), atol=1e-9)


@pytest.mark.parametrize('seed', range(6))
def test_perm_code_recovers_vertex_correspondence(seed):
    """Canonical position m of the image must be canonical position m of the DB.

    The image is produced exactly the way the pipeline produces one: a gnomonic
    projection via ``transforms.detransform_vectors`` followed by the solver's
    (y, x) -> (x, y) column swap. The opposite chirality conventions of
    ``kendall_rep_2d`` and ``kendall_rep_3d`` absorb the handedness flip between
    the two frames, so reps and permutation orders must agree exactly.
    """
    from mee2024 import transforms

    rng = np.random.default_rng(seed)
    centre = rng.normal(size=3)
    centre /= np.linalg.norm(centre)
    verts3d = centre + rng.normal(scale=0.01, size=(3, 3))
    verts3d /= np.linalg.norm(verts3d, axis=1, keepdims=True)

    ra = np.arctan2(centre[1], centre[0]) % (2 * np.pi)
    dec = np.arcsin(centre[2])
    x_params = (1e-5, ra, dec, rng.uniform(0, 2 * np.pi))   # scale, ra, dec, roll
    plate_yx = transforms.detransform_vectors(x_params, verts3d)
    pixels = plate_yx[:, ::-1]                              # the solver's (x, y)

    xyz_db, code_db = geometry.kendall_rep_3d(verts3d[None, 0], verts3d[None, 1],
                                              verts3d[None, 2])
    xyz_im, code_im = geometry.kendall_rep_2d(pixels[None, 0], pixels[None, 1],
                                              pixels[None, 2])
    # the reps agree up to projection curvature, O((field radius)^2) ~ 1e-4 here --
    # the irreducible term in the S3 tolerance model, not an implementation error
    assert np.allclose(xyz_db, xyz_im, atol=3e-4)

    order_db = geometry.PERM_TABLE[code_db[0]]
    order_im = geometry.PERM_TABLE[code_im[0]]
    # pairing canonical positions must pair the *same physical vertices*
    assert list(order_db) == list(order_im)


# ----------------------------------------------------------------- geometry

def test_find_rotation_matrix_recovers_a_known_rotation():
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('xyz', [0.3, -0.2, 1.1]).as_matrix()
    rng = np.random.default_rng(2)
    v = rng.normal(size=(40, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    recovered = geometry.find_rotation_matrix(v, (rot @ v.T).T)
    assert np.allclose((recovered.T @ v.T).T, (rot @ v.T).T, atol=1e-10)


def test_find_rotation_matrix_never_returns_a_reflection():
    """Mirrored vector sets defeat v1's fit; the det correction keeps it a rotation."""
    rng = np.random.default_rng(5)
    v = rng.normal(size=(10, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    reflected = v * np.array([1.0, 1.0, -1.0])
    fitted = geometry.find_rotation_matrix(v, reflected)
    assert np.linalg.det(fitted) == pytest.approx(1.0, abs=1e-9)


def test_acceptance_threshold_tolerance_parameter():
    """A tighter shape tolerance means fewer hypotheses, so a lower threshold."""
    from mee2024.platesolve_triangle import estimate_acceptance_threshold
    loose = estimate_acceptance_threshold(100, 3_000_000, np.radians(36 / 3600), 18,
                                          tolerance=0.01)
    tight = estimate_acceptance_threshold(100, 3_000_000, np.radians(36 / 3600), 18,
                                          tolerance=0.001)
    assert tight < loose


def test_preflight_reports_unavailable_when_nothing_is_installed(monkeypatch,
                                                                 tmp_path):
    """The v1.1.0 default-with-fallback hinges on preflight never raising."""
    from mee2024.platesolve2 import preflight
    monkeypatch.setattr(pattern_db, 'get_patterndb_root', lambda: tmp_path)
    ok, reason = preflight({})
    assert not ok and 'build-pattern-db' in reason
