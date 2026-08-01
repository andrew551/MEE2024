"""
The v2 solve: triangle query, consensus clustering, verification, acceptance.

Two query paths share one downstream (consensus -> verify -> accept):

* ``ratio_dphi`` -- the S1 near-verbatim port of ``platesolve_triangle``, kept
  bit-for-bit so the A/B against v1 isolates the catalogue and format. Mirrored
  fields are handled by the caller's transpose retry, exactly as v1.
* ``kendall`` (S2) -- triangles matched on the Kendall shape sphere with permutation
  codes recovering vertex correspondence. Every image triangle is queried at its
  shape point *and* at the point its mirrored copy would occupy, in one batched
  KD-tree call, so a mirrored field costs one extra consensus pass instead of a full
  re-extraction and re-query.

Convention notes: the (x, y) vs (y, x) mixtures and the +180 roll shifts are v1's,
kept bit-for-bit; the bench asserts v1/v2 parity on the real fields.
"""

import itertools
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from mee2024 import events, transforms
from mee2024.platesolve2 import geometry, verify
from mee2024.platesolve2.pattern_db import INVARIANT_KENDALL

# v1's constants (parametrised so later stages can move them). The invariant-space
# tolerance itself comes from the pattern database's manifest (db.tolerance).
F_ANCHORS = 9              # how many of the brightest centroids to try as anchors
TOL_CENT = np.radians(0.025)
TOL_ROLL = np.radians(0.025)
LOG_TOL_SCALE = 0.01
MAX_MATCH = 100            # verification centroids

# The S3 tolerance model, calibrated on 4677 identity-verified true triangle pairs
# over 24 synthetic fields (docs/bench/BENCH.md, S3):
#
#     r = C0 + C1 * (2*sqrt(2) * eps_px / S_img) + C2 * (theta_db / 2)^2
#
# C0 = the floor (catalogue + residual small-field curvature); C1 = the q99.5 noise
# envelope (folds the corpus's optical-distortion term); C2 ~ 1 confirms the
# projective-curvature prediction of the design doc. eps_px comes from
# options['platesolve_noise_px']; at query time theta_db is unknown so the curvature
# term uses the pattern-disc envelope, and candidates are re-cut exactly afterwards.
# 0.81% of true pairs canonicalise onto the mirror point (noisy chirality of
# near-degenerate triangles) and are unreachable at any radius -- the >=4 consensus
# absorbs that loss band.
MODEL_C0 = 0.0006
MODEL_C1 = 4.8
MODEL_C2 = 0.93
RADIUS_MAX = 0.02          # global cap: keeps worst-case candidate counts bounded
#: escalation ladder: failed solves retry with the noise assumption scaled up, so a
#: noisier-than-assumed image costs retries instead of a permanently inflated radius
ESCALATION = (1.0, 3.0)
#: per-pool candidate ceiling: an escalated radius on a hopeless field (junk, poles)
#: can return millions of hits whose consensus clustering is quadratic-ish. Keeping
#: the most shape-consistent candidates preserves the recovery power -- true pairs
#: sit well inside their bound -- while bounding the failure path's cost.
CANDIDATE_BUDGET = 1_500_000


# ------------------------------------------------------------ ratio_dphi query

def match_triangles(db, centroids, image_shape, tolerance):
    """v1's query: every (anchor, pair) triangle of the brightest centroids.

    Returns (match_cand, match_data, match_vect, match_info) exactly as v1 shapes
    them: candidate triangle rows, [longer side r1, phi1], the three image vertices
    (x, y) with the anchor first, and the centroid index triplet.
    """
    g = db.pattern_width
    # zero-centred pixel vectors in (x, y) convention, as v1
    vectors = (np.c_[centroids[:, 1], centroids[:, 0]]
               - np.array([image_shape[1], image_shape[0]]) / 2)
    kd_tree = db.kd_tree

    match_cand, match_data, match_vect, match_info = [], [], [], []
    for i in range(F_ANCHORS):
        for j, k in itertools.combinations(range(g), 2):
            if j == i or k == i or max(i, k) >= vectors.shape[0]:
                continue
            v0 = vectors[i, :]
            v1 = vectors[j, :] - v0
            v2 = vectors[k, :] - v0
            r1 = np.linalg.norm(v1)
            r2 = np.linalg.norm(v2)
            ratio = r2 / r1
            phi1 = np.arctan2(v1[1], v1[0])
            phi2 = np.arctan2(v2[1], v2[0])
            dphi = phi2 - phi1
            triplet = (i, j, k)
            if ratio > 1:
                ratio = 1 / ratio
                dphi = -dphi
                phi1 = phi2
                r1 = r2
                v1, v2 = v2, v1
                triplet = (i, k, j)
            dphi = dphi % (2 * np.pi)
            cand = kd_tree.query_ball_point([ratio, dphi], tolerance)
            if not cand:
                continue
            array_vect = np.c_[v0, v1 + v0, v2 + v0].T
            for cand_ in cand:
                match_cand.append(cand_)
                match_data.append([r1, phi1])
                match_vect.append(array_vect)
                match_info.append(triplet)
    return (np.array(match_cand, dtype=np.int64), np.array(match_data),
            np.array(match_vect), match_info)


def compute_platescale(db, match_cand, match_data, match_vect):
    """Per candidate: platescale, roll, boresight, catalogue target vertex rows.

    v1's vectorised solve, with the fixed ``anchor*153 + pair`` decode replaced by
    the prefix-sum offsets.
    """
    anchors = np.asarray(db.anchors)
    pattern_data = np.asarray(db.pattern_data)
    anchor_idx, leg_j, leg_k = db.triangle_anchor_and_legs(match_cand)

    s1 = pattern_data[anchor_idx, leg_j]
    s2 = pattern_data[anchor_idx, leg_k]
    sdat = np.stack([s1, s2])
    swap = s1[:, 0] < s2[:, 0]          # make sdat[0] the longer catalogue side
    sdat[:, swap, :] = sdat[:, swap, :][(1, 0), :, :]
    scale = sdat[0, :, 0] / match_data[:, 0]
    target_cols = np.stack([anchors[anchor_idx], sdat[0, :, 2:5], sdat[1, :, 2:5]],
                           axis=2)
    scale, roll, center_vect, _ = _orientation_from_pairs(match_vect, target_cols,
                                                          scale)
    return scale, roll, center_vect, target_cols.swapaxes(1, 2)


# --------------------------------------------------------------- kendall query

def _image_triplets(n_centroids, f, g):
    """The (i, j, k) query triangles, with v1's skip rules."""
    triplets = []
    for i in range(f):
        for j, k in itertools.combinations(range(g), 2):
            if j == i or k == i or max(i, k) >= n_centroids:
                continue
            triplets.append((i, j, k))
    return np.array(triplets, dtype=np.int64).reshape(-1, 3)


def query_radii(db, s_img_px, eps_px):
    """The S3 model's per-triangle query radius, with the curvature envelope.

    theta_db is unknown before the query, so the curvature term uses the pattern
    disc's worst case (a leg-to-leg side spans up to 2 * theta_pat); candidates are
    re-cut with their actual size afterwards.
    """
    theta_pat = np.radians(float(db.manifest['params'].get('theta_pat_deg', 1.7)))
    envelope = MODEL_C2 * theta_pat ** 2      # ((2 * theta_pat) / 2) ** 2
    r = MODEL_C0 + MODEL_C1 * (2 * np.sqrt(2) * eps_px / s_img_px) + envelope
    return np.minimum(r, RADIUS_MAX)


def match_triangles_kendall(db, centroids, image_shape, eps_px, fixed_tolerance=0.0):
    """Batched shape-sphere query of one image orientation.

    Returns (match_cand, canonical image points (n, 3, 2), canonical centroid-index
    triplets (n, 3), candidate shape distances, candidate image sizes, mean radius).
    The permutation code computed per image triangle reorders both the points and
    the index triplet into the canonical frame, so candidates pair with database
    vertices positionally.
    """
    g = db.pattern_width
    vectors = (np.c_[centroids[:, 1], centroids[:, 0]]
               - np.array([image_shape[1], image_shape[0]]) / 2)
    triplets = _image_triplets(vectors.shape[0], F_ANCHORS, g)
    empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 3, 2)),
             np.zeros((0, 3), dtype=np.int64), np.zeros(0), np.zeros(0), 0.0)
    if len(triplets) == 0:
        return empty

    points = vectors[triplets]                      # (n_tri, 3, 2)
    xyz, code = geometry.kendall_rep_2d(points[:, 0], points[:, 1], points[:, 2])
    order = geometry.PERM_TABLE[code]               # (n_tri, 3)
    rows = np.arange(len(triplets))[:, None]
    canon_points = points[rows, order]              # canonical vertex order
    canon_triplets = triplets[rows, order]
    # the canonical first side is the longest: the noise term's size scale
    s_img = np.linalg.norm(canon_points[:, 0] - canon_points[:, 1], axis=1)

    if fixed_tolerance > 0:
        radii = np.full(len(triplets), fixed_tolerance)
    else:
        radii = query_radii(db, s_img, eps_px)
    hit_lists = db.kd_tree.query_ball_point(xyz, radii)
    match_cand, tri_rows = [], []
    for t, hits in enumerate(hit_lists):
        match_cand.extend(hits)
        tri_rows.extend([t] * len(hits))
    if not match_cand:
        return empty
    tri_rows = np.array(tri_rows, dtype=np.int64)
    match_cand = np.array(match_cand, dtype=np.int64)

    xy = np.asarray(db.tri_inv, dtype=np.float64)[match_cand]
    z = np.sqrt(np.maximum(1.0 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 0.0))
    dist = np.linalg.norm(np.c_[xy, z] - xyz[tri_rows], axis=1)

    if len(match_cand) > CANDIDATE_BUDGET:
        # keep the most shape-consistent fraction of each query's ball
        ratio = dist / radii[tri_rows]
        keep = np.argpartition(ratio, CANDIDATE_BUDGET)[:CANDIDATE_BUDGET]
        keep.sort()
        match_cand, tri_rows, dist = match_cand[keep], tri_rows[keep], dist[keep]
    return (match_cand, canon_points[tri_rows].astype(np.float64),
            canon_triplets[tri_rows], dist, s_img[tri_rows], float(np.mean(radii)))


def compute_platescale_kendall(db, match_cand, canon_points, exact_cut=None):
    """Orientation per candidate, pairing canonical image and database vertices.

    ``exact_cut = (dist, s_img, eps_px)`` re-applies the S3 model with each
    candidate's *actual* catalogue-triangle size in the curvature term, before the
    expensive orientation solve. Returns (..., keep) so the caller can slice its
    parallel arrays; keep is None when no cut was applied.
    """
    anchors = np.asarray(db.anchors)
    pattern_data = np.asarray(db.pattern_data)
    tri_perm = np.asarray(db.tri_perm)
    anchor_idx, leg_j, leg_k = db.triangle_anchor_and_legs(match_cand)

    verts = np.stack([anchors[anchor_idx],
                      pattern_data[anchor_idx, leg_j, 2:5],
                      pattern_data[anchor_idx, leg_k, 2:5]], axis=1)  # rows = V1..V3
    order = geometry.PERM_TABLE[tri_perm[match_cand]]
    target_rows = verts[np.arange(len(match_cand))[:, None], order]

    # canonical first side: catalogue chord -> angle
    chord = np.linalg.norm(target_rows[:, 0] - target_rows[:, 1], axis=1)
    angle = 2 * np.arcsin(0.5 * np.minimum(chord, 2.0))

    keep = None
    if exact_cut is not None:
        dist, s_img, eps_px = exact_cut
        bound = (MODEL_C0 + MODEL_C1 * (2 * np.sqrt(2) * eps_px / s_img)
                 + MODEL_C2 * (angle / 2) ** 2)
        keep = dist <= np.minimum(bound, RADIUS_MAX)
        canon_points = canon_points[keep]
        target_rows = target_rows[keep]
        angle = angle[keep]

    pixels = np.linalg.norm(canon_points[:, 0] - canon_points[:, 1], axis=1)
    scale = angle / pixels

    target_cols = target_rows.swapaxes(1, 2)
    scale, roll, center_vect, rmatrix = _orientation_from_pairs(canon_points,
                                                                target_cols, scale)
    # The candidate map is IMPROPER (det = -1): the pixel plane and the sky have
    # opposite handedness, which is also why v1's roll decode needs its empirical
    # +90/+180 shifts. Composing with a fixed reflection of the image frame makes
    # it a genuine rotation, so the quaternion conversion is exact; being the same
    # fixed reflection for every candidate, same-solution candidates still share
    # one quaternion.
    proper = rmatrix.copy()
    proper[:, :, 1] = -proper[:, :, 1]
    quat = geometry.rotations_to_quaternions(proper)
    return scale, roll, center_vect, target_rows, keep, quat


# ------------------------------------------------------------ shared downstream

def _orientation_from_pairs(image_points, target_cols, scale):
    """v1's batched 3-point orientation: rmatrix = target @ inv(image 3-vectors)."""
    scaled = np.einsum('ijk,i -> ijk', image_points, scale)
    as_3vect = transforms.icoord_to_vector(scaled).swapaxes(1, 2)
    inv_matrix = np.zeros(as_3vect.shape, as_3vect.dtype)
    for i in range(3):
        for j in range(3):
            ia = [x for x in range(3) if x != i]
            ib = [x for x in range(3) if x != j]
            inv_matrix[:, j, i] = (as_3vect[:, ia[0], ib[0]] * as_3vect[:, ia[1], ib[1]]
                                   - as_3vect[:, ia[1], ib[0]]
                                   * as_3vect[:, ia[0], ib[1]]) * (-1) ** (i + j)
    det = np.linalg.det(as_3vect).reshape((-1, 1, 1))
    # degenerate (near-collinear) image triangles produce singular systems; keep the
    # rows but poison their solutions so they can never join a consensus
    bad = np.abs(det[:, 0, 0]) < 1e-30
    det[bad] = 1.0
    inv_matrix /= det
    rmatrix = np.einsum('...ij,...jk -> ...ik', target_cols, inv_matrix)
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2 * np.pi)
    if np.any(bad):
        scale = scale.copy()
        scale[bad] = 1e9
    return scale, roll, center_vect, rmatrix


def _consensus_and_verify(db, catalogue, scale, roll, center_vect, match_info,
                          target_rows, centroids, image_size, options, diagnostics,
                          output_dir=None, est_tolerance=None, adapt_depth=False,
                          quat=None):
    """Cluster candidate orientations, verify each strong cluster, keep the best.

    With ``quat`` (S4), candidates cluster on [log s, q] -- a singularity-free,
    uniform metric on scale x SO(3). The legacy [log s, roll, centre] key (used by
    the frozen ratio path, and as the kendall rollback) has two chart defects:
    roll enters unwrapped, so a consensus straddling roll = 0/2pi splits, and roll
    itself is ill-conditioned near the celestial poles, where candidate rolls
    scatter far beyond TOL_ROLL for one physical pointing.
    """
    verify_spec = db.manifest.get('verify') or {}
    mag_limit = float(verify_spec.get('mag_limit', 12.0))
    epoch = float(verify_spec.get('epoch', 2024.0))
    n_stars_catalogue = verify.catalogue_size(catalogue, mag_limit)

    best_result = _failure_result(diagnostics)
    if scale.shape[0] == 0:
        return best_result

    n_obs = centroids.shape[0]
    all_star_plate = centroids - np.array([image_size[0] / 2, image_size[1] / 2])

    N = scale.shape[0]
    with np.errstate(divide='ignore'):
        log_scale = np.log(scale) / LOG_TOL_SCALE
        if quat is None:
            data = np.c_[log_scale, roll / TOL_ROLL, center_vect / TOL_CENT]
            index_map = np.arange(N)
        else:
            # double cover: q and -q are one rotation. Canonicalise the sign and
            # add negated twins for the sliver of candidates near the boundary,
            # so one physical cluster cannot be split in two.
            canonical, twin_idx = geometry.canonicalise_quaternions(
                quat, band=2 * TOL_ROLL)
            base = np.c_[log_scale, canonical / TOL_ROLL]
            twins = np.c_[log_scale[twin_idx], -canonical[twin_idx] / TOL_ROLL]
            data = np.r_[base, twins]
            index_map = np.r_[np.arange(N), twin_idx]
    tree_matches = KDTree(data)
    candidate_pairs = tree_matches.query_pairs(1)
    M = data.shape[0]
    graph = csr_matrix(([1 for _ in candidate_pairs],
                        ([x[0] for x in candidate_pairs],
                         [x[1] for x in candidate_pairs])), shape=(M, M))
    n_components, labels = connected_components(csgraph=graph, directed=False,
                                                return_labels=True)

    # Iterate clusters without touching Python for the hopeless ones. The kendall
    # invariant is vertex-symmetric, so the same physical star triple stored under
    # several DB anchors matches one image triangle several times with identical
    # solutions -- tens of thousands of raw clusters pass a naive >=4-count gate.
    # Gate on *distinct image triples* instead, computed wholly in numpy: a cluster
    # needs 4 non-redundant triangles to be worth the Python-level work.
    triples = np.sort(np.asarray(match_info, dtype=np.int64), axis=1)
    triple_ids = (triples[:, 0] << 40) | (triples[:, 1] << 20) | triples[:, 2]
    order = np.lexsort((index_map, labels))
    sorted_labels = labels[order]
    boundaries = np.r_[0, np.nonzero(np.diff(sorted_labels))[0] + 1, M]
    starts, stops = boundaries[:-1], boundaries[1:]
    big = np.nonzero(stops - starts >= 4)[0]

    best = -1
    best_non_redundant = None
    n_matches = 0
    for b in big:
        # ascending candidate order; a twin resolves to its original candidate
        indices = np.unique(index_map[order[starts[b]:stops[b]]])
        if len(indices) < 4 or len(np.unique(triple_ids[indices])) < 4:
            continue
        seen = set()
        non_redundant = []
        for ind in indices:
            key = tuple(match_info[ind])
            if key in seen:
                continue
            seen.update(itertools.permutations(key))
            non_redundant.append(ind)
        if len(non_redundant) < 4:
            continue
        diagnostics['n_clusters_checked'] += 1

        matchset = dict()
        for ind in non_redundant:
            matchset.update(zip(tuple(match_info[ind]), target_rows[ind]))
        el = non_redundant[0]

        # note the (y, x) plate vectors here against the (x, y) used for the query
        # triangles: v1's mixed convention, absorbed by the fitted rotation and the
        # roll shifts below. Kept bit-for-bit.
        ivects = transforms.icoord_to_vector(
            np.array([all_star_plate[_] for _ in matchset]) * scale[el])
        catvects = np.array([_ for _ in matchset.values()])
        rotation_matrix = geometry.find_rotation_matrix(ivects, catvects)
        acc_ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                       rotation_matrix[0, 0])) % 360
        acc_dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                        np.linalg.norm(rotation_matrix[1:3, 2])))
        acc_roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                         rotation_matrix[2, 2])) % 360
        acc_roll = (acc_roll + 180) % 360   # v1's convention shift, kept as-is

        platescale = (np.degrees(scale[el]), acc_ra, acc_dec, acc_roll + 180)
        stardata, plate2, max_error, local_density = verify.match_centroids(
            centroids[:MAX_MATCH, :], np.radians(platescale), image_size, options,
            catalogue, mag_limit, epoch, adapt_depth=adapt_depth)
        thresh = verify.estimate_acceptance_threshold(
            min(n_obs, MAX_MATCH), n_stars_catalogue, max_error, db.pattern_width,
            addon=3, local_density=local_density,
            tolerance=est_tolerance if est_tolerance else db.tolerance)
        diagnostics['threshold'] = int(thresh)

        events.emit(events.SOLVE_CANDIDATE, n_triangles=len(non_redundant),
                    n_matched=int(stardata.shape[0]), threshold=int(thresh),
                    accepted=bool(stardata.shape[0] >= thresh),
                    ra=float(acc_ra), dec=float(acc_dec),
                    platescale=float(3600 * np.degrees(scale[el])))
        if stardata.shape[0] >= thresh:
            n_matches += 1
            print(f'MATCH ACCEPTED (nstars matched = {stardata.shape[0]}, '
                  f'thresh = {thresh})')
            if stardata.shape[0] > best:
                best = stardata.shape[0]
                best_non_redundant = non_redundant
                best_result = {
                    'success': True,
                    'x': np.radians(platescale),
                    'platescale/arcsec': 3600 * np.degrees(scale[el]),
                    'ra': acc_ra, 'dec': acc_dec, 'roll': acc_roll,
                    'matched_centroids': plate2 + np.array([image_size[0] / 2,
                                                            image_size[1] / 2]),
                    'matched_stars': stardata,
                    'diagnostics': diagnostics,
                }

    if n_matches > 1:
        print(f'WARNING: multiple ({n_matches}) platesolves were successful, '
              'returning best one')
    if best_result['success'] and output_dir is not None:
        _save_triangle_plot(centroids, image_size, best_non_redundant, match_info,
                            best_result, output_dir)
    return best_result


def _failure_result(diagnostics):
    return {'success': False, 'x': None, 'platescale': None,
            'matched_centroids': None, 'matched_stars': None,
            'platescale/arcsec': None, 'ra': None, 'dec': None, 'roll': None,
            'diagnostics': diagnostics}


def solve_helper(db, catalogue, centroids, image_size, options, output_dir=None):
    """One solve attempt on one image orientation (the ratio_dphi path)."""
    t0 = time.perf_counter()
    match_cand, match_data, match_vect, match_info = match_triangles(
        db, centroids, image_size, db.tolerance)
    diagnostics = {'n_candidates': int(len(match_cand)), 'n_clusters_checked': 0,
                   'threshold': None}
    if len(match_cand) == 0:
        return _failure_result(diagnostics)

    scale, roll, center_vect, target_rows = compute_platescale(
        db, match_cand, match_data, match_vect)
    result = _consensus_and_verify(db, catalogue, scale, roll, center_vect,
                                   match_info, target_rows, centroids, image_size,
                                   options, diagnostics, output_dir)
    diagnostics['time_s'] = round(time.perf_counter() - t0, 3)
    if not result['success']:
        print('Platesolve FAILED')
    return result


def solve_kendall(db, catalogue, centroids, image_size, options, output_dir=None,
                  try_mirror_also=True):
    """The S2/S3 solve: one query pass covers the field and its mirror image, at a
    radius set by the calibrated tolerance model.

    Pools are evaluated lazily -- a successful normal solve never pays for the
    mirror query -- and a fully failed attempt escalates the noise assumption
    (ESCALATION) before giving up, so a noisier-than-assumed image costs a retry
    rather than a permanently inflated radius for everyone.
    """
    t0 = time.perf_counter()
    mirrored_centroids = centroids[:, [1, 0]]
    mirrored_size = (image_size[1], image_size[0])

    pools = [(centroids, image_size, False)]
    if try_mirror_also:
        pools.append((mirrored_centroids, mirrored_size, True))

    eps0 = float(options.get('platesolve_noise_px', 0.3) or 0.3)
    fixed = float(options.get('v2_fixed_tolerance', 0) or 0)
    attempts = (1.0,) if fixed > 0 else ESCALATION

    diagnostics = {'n_candidates': 0, 'n_clusters_checked': 0, 'threshold': None,
                   'noise_px_used': None}
    result = _failure_result(diagnostics)
    for scale_up in attempts:
        eps = eps0 * scale_up
        diagnostics['noise_px_used'] = eps if fixed <= 0 else None
        for pool_centroids, pool_size, is_mirror in pools:
            cand, points, triplets, dist, s_img, mean_radius = \
                match_triangles_kendall(db, pool_centroids, pool_size, eps,
                                        fixed_tolerance=fixed)
            diagnostics['n_candidates'] += int(len(cand))
            if len(cand) == 0:
                continue
            exact = None if fixed > 0 else (dist, s_img, eps)
            scale, roll, center_vect, target_rows, keep, quat = \
                compute_platescale_kendall(db, cand, points, exact_cut=exact)
            if keep is not None:
                triplets = triplets[keep]
            if scale.shape[0] == 0:
                continue
            legacy = options.get('v2_consensus') == 'legacy'
            result = _consensus_and_verify(
                db, catalogue, scale, roll, center_vect,
                [tuple(t) for t in triplets], target_rows,
                pool_centroids, pool_size, options, diagnostics, output_dir,
                est_tolerance=(fixed if fixed > 0 else mean_radius),
                adapt_depth=True, quat=None if legacy else quat)
            if result['success']:
                result['mirror'] = is_mirror
                if is_mirror:
                    result['matched_centroids'][:, [0, 1]] = \
                        result['matched_centroids'][:, [1, 0]]
                break
        if result['success']:
            break
    result['mirror'] = result.get('mirror', False)
    if not result['success']:
        print('Platesolve FAILED')
    diagnostics['time_s'] = round(time.perf_counter() - t0, 3)
    return result


def _save_triangle_plot(centroids, image_size, non_redundant, match_info,
                        result, output_dir):
    """The stage-1 output figure: matched triangles over the detected centroids."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(centroids[:, 1], centroids[:, 0])
    plt.xlim(0, image_size[1])
    plt.ylim(0, image_size[0])
    for t in non_redundant:
        tri = match_info[t]
        v = np.array([centroids[_] for _ in tri] + [centroids[tri[0]]])
        plt.plot(v[:, 1], v[:, 0], color='red')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.xlabel('pixel X', fontsize=16)
    plt.ylabel('pixel Y', fontsize=16)
    plt.title(f"{len(non_redundant)} triangles matched\n"
              f"platescale={result['platescale/arcsec']:.4f} arcsec/pixel\n"
              f"ra={result['ra']:.4f}, dec={result['dec']:.4f}, "
              f"roll={result['roll']:.4f}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'triangle_matches.png', dpi=600)
    plt.close()
