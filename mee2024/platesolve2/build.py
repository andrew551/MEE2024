"""
Build a v2 pattern database from a star catalogue.

The selection rules are a faithful port of v1's ``platesolve_new.generate()`` -- same
anchor/leg semantics, same (dtheta, phi) pattern encoding, same (ratio, dphi) triangle
invariant -- so that stage S1's A/B isolates exactly two variables: the catalogue
(Gaia instead of Tycho) and the on-disk format. What changed is engineering:

* the 1.4 M scalar ``query_ball_point`` calls become batched, multi-threaded queries
  followed by a pure-flag sweep (the sweep itself is inherently sequential: whether a
  star is an anchor depends on which brighter stars already are);
* anchors with fewer than ``e`` legs keep what they have (prefix-sum offsets) instead
  of raising v1's "edge case handling unimplemented";
* dtypes are pinned, progress is reported, and the result carries a manifest.

Positions are used at the catalogue's own reference epoch (recorded in the manifest).
Proper-motion drift over the ~decade that matters is well under an arcsecond for all
but a handful of extreme stars -- invisible at the 36 arcsec verification radius and
the S1 shape tolerance. Stage S3's tolerance model accounts for it explicitly.
"""

import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mee2024.MEE2024util import get_patterndb_root
from mee2024.platesolve2 import pattern_db
from mee2024.progress import NullProgress

#: v1's structural parameters, kept identical for the S1 A/B
DEFAULT_PARAMS = dict(
    a=80000,                  # brightest stars kept as anchors unconditionally
    b=160000,                 # next-brightest, kept as anchors only if isolated
    d=700000,                 # star-list depth (legs)
    e=18,                     # legs per pattern (the solver's g)
    theta_sep_deg=0.4,        # anchor isolation radius
    theta_pat_deg=1.7,        # pattern disc radius
    double_star_arcsec=36.0,  # blend rejection radius
)

ANCHOR_CHUNK = 8192
#: kendall bucket grid: 400 x 400 cells of 0.005 over the (x, y) unit square. Cell
#: size only affects gather efficiency, never correctness -- a query spans however
#: many cells its radius needs.
BUCKET_GRID_N = 400


def select_anchors_and_legs(vectors, a, b, theta_sep, theta_double, progress):
    """v1's magnitude-ordered sweep, with the neighbour queries batched up front.

    vectors must be brightest-first unit vectors. Returns (anchor mask, leg mask).
    A star is rejected outright if a kept *anchor* sits within the blend radius;
    it is an anchor if in the top-a (and not blended), or in the next-b and no
    anchor sits within theta_sep. Every kept star is a leg.
    """
    tree = cKDTree(vectors)
    # the radii are chord lengths, matching v1's convention of passing radians
    # directly as Euclidean radii (the difference is O(theta^3), irrelevant here)
    progress.start(3, 'anchor selection: neighbour queries')
    nbrs_sep = tree.query_ball_point(vectors, theta_sep, workers=-1)
    progress.update(1)
    nbrs_dbl = tree.query_ball_point(vectors, theta_double, workers=-1)
    progress.update(2)

    n = len(vectors)
    kept = np.zeros(n, dtype=bool)      # anchors
    kept2 = np.zeros(n, dtype=bool)     # legs
    for i in range(n):
        if not np.any(kept[nbrs_sep[i]]):
            if i < a + b:
                kept[i] = True
            kept2[i] = True
        elif not np.any(kept[nbrs_dbl[i]]):
            if i < a:
                kept[i] = True
            kept2[i] = True
    progress.finish()
    return kept, kept2


def extract_patterns(vectors, kept, kept2, e, theta_pat, progress,
                     dimmer_legs_only=False):
    """Per anchor: the e brightest legs within theta_pat, as (dtheta, phi, vector).

    Returns (pattern_ind, pattern_data, n_legs), padded where an anchor has fewer
    than e legs in range. pattern_ind holds *star-list* row indices.

    ``dimmer_legs_only`` stores each star triple once, under its brightest member:
    the dev-platesolve dedupe rule (the S2b variant). With the vertex-symmetric
    kendall invariant this imposes no ordering assumption on the *query* side -- a
    query triple matches whichever anchor the database stored it under.
    """
    anchor_rows = np.nonzero(kept)[0]
    leg_rows = np.nonzero(kept2)[0]
    vectors2 = vectors[kept2]
    tree2 = cKDTree(vectors2)
    # star-list row -> leg row (v1's cumsum trick); valid wherever kept2 is set
    to_leg = np.cumsum(kept2) - 1

    n_anchor = len(anchor_rows)
    pattern_ind = np.full((n_anchor, e), -1, dtype=np.int32)
    pattern_data = np.zeros((n_anchor, e, 5), dtype=np.float32)
    n_legs = np.zeros(n_anchor, dtype=np.int64)
    z = np.array([0.0, 0.0, 1.0])

    progress.start(n_anchor, 'patterns')
    for start in range(0, n_anchor, ANCHOR_CHUNK):
        stop = min(start + ANCHOR_CHUNK, n_anchor)
        chunk_rows = anchor_rows[start:stop]
        neighbour_lists = tree2.query_ball_point(vectors[chunk_rows], theta_pat,
                                                 workers=-1)
        for i, neighbours in zip(range(start, stop), neighbour_lists):
            self_leg = int(to_leg[anchor_rows[i]])
            if dimmer_legs_only:
                # legs strictly dimmer than the anchor (leg rows are
                # brightness-ordered, and the anchor is itself a leg)
                neighbours = [idx for idx in neighbours if idx > self_leg]
            else:
                neighbours.remove(self_leg)          # don't match self
            # leg rows are brightness-ordered: smallest indices = brightest
            chosen = sorted(neighbours)[:e]
            k = len(chosen)
            n_legs[i] = k
            if k < 2:
                continue    # no triangle can come from this anchor
            anchor_vec = vectors[anchor_rows[i]]
            delta = vectors2[chosen] - anchor_vec
            dtheta = 2 * np.arcsin(0.5 * np.linalg.norm(delta, axis=1))
            tangent_phi = np.cross(z, anchor_vec)
            tangent_phi /= np.linalg.norm(tangent_phi)
            tangent_theta = np.cross(tangent_phi, anchor_vec)
            tangent_theta /= np.linalg.norm(tangent_theta)
            phi = np.arctan2(delta @ tangent_phi, delta @ tangent_theta)

            pattern_ind[i, :k] = leg_rows[chosen]
            pattern_data[i, :k, 0] = dtheta
            pattern_data[i, :k, 1] = phi
            pattern_data[i, :k, 2:5] = vectors2[chosen]
        progress.update(stop)
    progress.finish()
    return pattern_ind, pattern_data, n_legs


def _triangle_layout(n_legs, e):
    pairs = np.array(list(itertools.combinations(range(e), 2)), dtype=np.int64)
    counts = np.where(n_legs >= e, pairs.shape[0], n_legs * (n_legs - 1) // 2)
    offsets = np.zeros(len(n_legs) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return pairs, offsets


def compute_triangles(pattern_data, n_legs, e, progress):
    """v1's (ratio, dphi) invariant for every leg pair of every anchor.

    Returns (anchor_tri_offset, tri_legs, tri_inv).
    """
    pairs, offsets = _triangle_layout(n_legs, e)
    tri_legs = np.zeros((int(offsets[-1]), 2), dtype=np.uint8)
    tri_inv = np.zeros((int(offsets[-1]), 2), dtype=np.float32)

    progress.start(len(n_legs), 'triangles')
    for i in range(len(n_legs)):
        k = int(n_legs[i])
        if k < 2:
            continue
        valid = pairs[pairs[:, 1] < k]
        dt = pattern_data[i, :, 0]
        ph = pattern_data[i, :, 1]
        ratio = dt[valid[:, 1]] / dt[valid[:, 0]]
        dphi = ph[valid[:, 1]] - ph[valid[:, 0]]
        flip = ratio > 1
        ratio[flip] = 1.0 / ratio[flip]
        dphi[flip] = -dphi[flip]
        dphi %= 2 * np.pi
        rows = slice(int(offsets[i]), int(offsets[i + 1]))
        tri_legs[rows] = valid
        tri_inv[rows, 0] = ratio
        tri_inv[rows, 1] = dphi
        if i % ANCHOR_CHUNK == 0:
            progress.update(i)
    progress.finish()
    return offsets, tri_legs, tri_inv


def compute_triangles_kendall(anchors, pattern_data, n_legs, e, progress):
    """The Kendall shape invariant for every leg pair of every anchor, chunked.

    Returns (anchor_tri_offset, tri_legs, tri_inv (x, y), tri_perm).
    """
    from mee2024.platesolve2 import geometry

    pairs, offsets = _triangle_layout(n_legs, e)
    tri_legs = np.zeros((int(offsets[-1]), 2), dtype=np.uint8)
    tri_inv = np.zeros((int(offsets[-1]), 2), dtype=np.float32)
    tri_perm = np.zeros(int(offsets[-1]), dtype=np.uint8)

    progress.start(len(n_legs), 'triangles')
    for start in range(0, len(n_legs), ANCHOR_CHUNK):
        stop = min(start + ANCHOR_CHUNK, len(n_legs))
        # gather (v1, v2, v3) for every triangle of every anchor in the chunk
        chunk_rows, chunk_j, chunk_k = [], [], []
        for i in range(start, stop):
            k = int(n_legs[i])
            if k < 2:
                continue
            valid = pairs[pairs[:, 1] < k]
            chunk_rows.append(np.full(len(valid), i, dtype=np.int64))
            chunk_j.append(valid[:, 0])
            chunk_k.append(valid[:, 1])
        if not chunk_rows:
            progress.update(stop)
            continue
        rows = np.concatenate(chunk_rows)
        jj = np.concatenate(chunk_j)
        kk = np.concatenate(chunk_k)
        v1 = anchors[rows].astype(np.float64)
        v2 = pattern_data[rows, jj, 2:5].astype(np.float64)
        v3 = pattern_data[rows, kk, 2:5].astype(np.float64)
        xyz, code = geometry.kendall_rep_3d(v1, v2, v3)

        out = slice(int(offsets[start]), int(offsets[stop]))
        tri_legs[out, 0] = jj
        tri_legs[out, 1] = kk
        tri_inv[out] = xyz[:, :2]
        tri_perm[out] = code
        progress.update(stop)
    progress.finish()
    return offsets, tri_legs, tri_inv, tri_perm


def build_pattern_db(stars, name, out_root=None, params=None, verify_spec=None,
                     source=None, progress=None, provenance=''):
    """Build a database from a StarTable(-like) object and write it under out_root.

    ``stars`` needs .mag and .get_vectors(); it is sorted brightest-first here.
    Structural parameters a/b/d are clipped to the table size, so a small regional
    table (a test fixture) builds a small regional database with the same rules.
    """
    progress = progress or NullProgress()
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})

    order = np.argsort(np.asarray(stars.mag), kind='stable')
    d = min(int(p['d']), len(order))
    a = min(int(p['a']), d)
    b = min(int(p['b']), max(0, d - a))
    vectors = np.ascontiguousarray(stars.get_vectors()[order][:d], dtype=np.float32)

    theta_sep = np.radians(p['theta_sep_deg'])
    theta_double = np.radians(p['double_star_arcsec'] / 3600.0)
    theta_pat = np.radians(p['theta_pat_deg'])
    e = int(p['e'])

    dedupe = p.get('dedupe_rule', 'none')
    kept, kept2 = select_anchors_and_legs(vectors, a, b, theta_sep, theta_double,
                                          progress)
    pattern_ind, pattern_data, n_legs = extract_patterns(
        vectors, kept, kept2, e, theta_pat, progress,
        dimmer_legs_only=(dedupe == 'dimmer_legs'))
    # pattern_ind holds rows of the truncated, brightness-sorted list; map back to
    # the caller's table rows so the indices stay meaningful provenance
    valid = pattern_ind >= 0
    pattern_ind[valid] = order[:d][pattern_ind[valid]].astype(np.int32)

    invariant = p.get('invariant', pattern_db.INVARIANT_RATIO_DPHI)
    arrays = {
        'anchors': vectors[kept],
        'pattern_ind': pattern_ind,
        'pattern_data': pattern_data,
    }
    if invariant == pattern_db.INVARIANT_KENDALL:
        offsets, tri_legs, tri_inv, tri_perm = compute_triangles_kendall(
            arrays['anchors'], pattern_data, n_legs, e, progress)
        # Bucket-sort the triangle columns over an (x, y) grid: the solver then
        # queries by gathering grid cells from the memory-mapped files -- no search
        # index is ever built, so opening the database costs pages, not seconds.
        # tri_anchor replaces the prefix-sum decode, which cannot describe
        # grid-ordered rows.
        grid_n = int(p.get('bucket_grid_n', BUCKET_GRID_N))
        tri_anchor = np.repeat(np.arange(len(offsets) - 1, dtype=np.int32),
                               np.diff(offsets))
        cell = 2.0 / grid_n
        ix = np.clip(((tri_inv[:, 0] + 1) / cell).astype(np.int64), 0, grid_n - 1)
        iy = np.clip(((tri_inv[:, 1] + 1) / cell).astype(np.int64), 0, grid_n - 1)
        bucket = ix * grid_n + iy
        order_tri = np.argsort(bucket, kind='stable')
        arrays.update(
            tri_inv=tri_inv[order_tri], tri_legs=tri_legs[order_tri],
            tri_perm=tri_perm[order_tri], tri_anchor=tri_anchor[order_tri],
            tri_bucket_offset=np.searchsorted(
                bucket[order_tri], np.arange(grid_n * grid_n + 1)).astype(np.int64))
        p = dict(p, bucket_grid_n=grid_n)
    else:
        offsets, tri_legs, tri_inv = compute_triangles(pattern_data, n_legs, e,
                                                       progress)
        arrays.update(anchor_tri_offset=offsets, tri_legs=tri_legs, tri_inv=tri_inv)

    stored = dict(p, invariant=invariant, a=a, b=b, d=d, dedupe_rule=dedupe)
    out_dir = (Path(out_root) if out_root else get_patterndb_root()) / name
    manifest = pattern_db.write_pattern_db(
        out_dir, arrays, stored,
        source=source or {},
        verify_spec=verify_spec or {},
        provenance=provenance)
    return out_dir, manifest


def build_from_catalogue(name=pattern_db.DEFAULT_NAME,
                         catalogue_names=('gaia_dr3_g12',),
                         params=None, progress=None):
    """Build from installed offline Gaia archives: the CLI entry point's core."""
    from mee2024.starcat import providers
    from mee2024.starcat.table import concat

    provider = providers.GaiaOfflineProvider.from_installed(list(catalogue_names))
    # one all-sky read at the archives' native epoch (epoch=None skips propagation,
    # which the provider-level lookup does not offer)
    parts = [c.lookup((0.0, 360.0), (-90.0, 90.0), max_magnitude=None, epoch=None)
             for c in provider.catalogues]
    base_epoch = parts[0].epoch
    parts = [p if p.epoch == base_epoch else p.at_epoch(base_epoch) for p in parts]
    stars = parts[0] if len(parts) == 1 else concat(parts)

    source = {
        'catalogues': [c.manifest['name'] for c in provider.catalogues],
        'n_stars': int(len(stars)),
        'epoch': float(stars.epoch),
        'band': stars.band,
    }
    verify_spec = {
        'provider': 'gaia_offline',
        'releases': list(catalogue_names),
        'mag_limit': 12.0,     # v1 verified against Tycho at this same cut
        'epoch': 2024.0,       # v1's Tycho npz is propagated to 2024; keep parity
    }
    return build_pattern_db(stars, name, params=params, progress=progress,
                            verify_spec=verify_spec, source=source,
                            provenance='built by mee2024 build-pattern-db')
