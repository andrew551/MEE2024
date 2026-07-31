"""
The v2 solve: triangle query, consensus clustering, verification, acceptance.

Stage S1 is a deliberate near-verbatim port of ``platesolve_triangle`` reading the new
pattern database, so the A/B against v1 isolates the catalogue and format. Everything
convention-shaped -- the (x, y) vs (y, x) mixtures, the +180 roll shifts -- is kept
bit-for-bit; the bench asserts v1/v2 parity on the real fields. Two deliberate
differences, both argued in docs/PLATESOLVER_V2_DESIGN.md: verification runs against
the catalogue in the DB manifest (see verify.py), and the orientation fit carries the
Kabsch determinant correction (geometry.py).

Later stages change this file: S2 swaps the invariant, S3 the tolerance model, S4 the
consensus metric, S5 the query strategy and index.
"""

import itertools
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from mee2024 import events, transforms
from mee2024.platesolve2 import geometry, verify

# v1's constants, kept identical for S1 (parametrised so later stages can move them)
F_ANCHORS = 9              # how many of the brightest centroids to try as anchors
TOLERANCE = 0.01           # invariant-space match radius
TOL_CENT = np.radians(0.025)
TOL_ROLL = np.radians(0.025)
LOG_TOL_SCALE = 0.01
MAX_MATCH = 100            # verification centroids


def match_triangles(db, centroids, image_shape, tolerance):
    """Query every (anchor, pair) triangle of the brightest centroids against the DB.

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
    """Per candidate: platescale, roll, boresight and the catalogue target vectors.

    v1's vectorised solve, with the fixed ``anchor*153 + pair`` decode replaced by the
    prefix-sum offsets (anchors may own fewer than the full complement of triangles).
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
    scaled = np.einsum('ijk,i -> ijk', match_vect, scale)
    as_3vect = transforms.icoord_to_vector(scaled).swapaxes(1, 2)
    target = np.stack([anchors[anchor_idx], sdat[0, :, 2:5], sdat[1, :, 2:5]], axis=2)
    # batched 3x3 cofactor inverse (avoids a per-candidate np.linalg.inv call)
    inv_matrix = np.zeros(as_3vect.shape, as_3vect.dtype)
    for i in range(3):
        for j in range(3):
            ia = [x for x in range(3) if x != i]
            ib = [x for x in range(3) if x != j]
            inv_matrix[:, j, i] = (as_3vect[:, ia[0], ib[0]] * as_3vect[:, ia[1], ib[1]]
                                   - as_3vect[:, ia[1], ib[0]]
                                   * as_3vect[:, ia[0], ib[1]]) * (-1) ** (i + j)
    inv_matrix /= np.linalg.det(as_3vect).reshape((-1, 1, 1))
    rmatrix = np.einsum('...ij,...jk -> ...ik', target, inv_matrix)
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2 * np.pi)
    return scale, roll, center_vect, target


def solve_helper(db, catalogue, centroids, image_size, options, output_dir=None):
    """One solve attempt (no mirror retry). Returns v1's result dict + diagnostics."""
    verify_spec = db.manifest.get('verify') or {}
    mag_limit = float(verify_spec.get('mag_limit', 12.0))
    epoch = float(verify_spec.get('epoch', 2024.0))
    n_stars_catalogue = verify.catalogue_size(catalogue, mag_limit)

    t0 = time.perf_counter()
    match_cand, match_data, match_vect, match_info = match_triangles(
        db, centroids, image_size, TOLERANCE)
    diagnostics = {'n_candidates': int(len(match_cand)), 'n_clusters_checked': 0,
                   'threshold': None}
    best_result = {'success': False, 'x': None, 'platescale': None,
                   'matched_centroids': None, 'matched_stars': None,
                   'platescale/arcsec': None, 'ra': None, 'dec': None, 'roll': None,
                   'diagnostics': diagnostics}
    if len(match_cand) == 0:
        return best_result

    scale, roll, center_vect, target_vectors = compute_platescale(
        db, match_cand, match_data, match_vect)

    n_obs = centroids.shape[0]
    all_star_plate = centroids - np.array([image_size[0] / 2, image_size[1] / 2])

    vector_plates = np.c_[np.log(scale) / LOG_TOL_SCALE, roll / TOL_ROLL,
                          center_vect / TOL_CENT]
    tree_matches = KDTree(vector_plates)
    candidate_pairs = tree_matches.query_pairs(1)
    N = vector_plates.shape[0]
    graph = csr_matrix(([1 for _ in candidate_pairs],
                        ([x[0] for x in candidate_pairs],
                         [x[1] for x in candidate_pairs])), shape=(N, N))
    n_components, labels = connected_components(csgraph=graph, directed=False,
                                                return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))

    best = -1
    best_non_redundant = None
    n_matches = 0
    for i in range(n_components):
        if counts[i] < 4:
            continue
        indices = np.nonzero(labels == i)[0]
        # remove redundant triangles (a, b, c), (b, a, c) etc.
        seen = set()
        non_redundant = []
        for ind in indices:
            if match_info[ind] in seen:
                continue
            seen.update(itertools.permutations(match_info[ind]))
            non_redundant.append(ind)
        if len(non_redundant) < 4:
            continue
        diagnostics['n_clusters_checked'] += 1

        matchset = dict()
        for ind in non_redundant:
            matchset.update(zip(match_info[ind], target_vectors[ind].T))
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
            catalogue, mag_limit, epoch)
        thresh = verify.estimate_acceptance_threshold(
            min(n_obs, MAX_MATCH), n_stars_catalogue, max_error, db.pattern_width,
            addon=3, local_density=local_density, tolerance=TOLERANCE)
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

    diagnostics['time_s'] = round(time.perf_counter() - t0, 3)
    if n_matches > 1:
        print(f'WARNING: multiple ({n_matches}) platesolves were successful, '
              'returning best one')
    elif n_matches == 0:
        print('Platesolve FAILED')

    if best_result['success'] and output_dir is not None:
        _save_triangle_plot(centroids, image_size, best_non_redundant, match_info,
                            best_result, output_dir)
    return best_result


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
