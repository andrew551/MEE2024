"""
Geometry for the v2 solver: the Kendall shape invariant (S2) and the orientation fit.

Quaternion consensus utilities arrive with stage S4 (docs/PLATESOLVER_V2_DESIGN.md).

The shape coordinates and their canonicalisation follow the verified math from the
dev-platesolve spike. For a triangle with side lengths (r1, r2, r3) = (|V1V2|, |V2V3|,
|V3V1|) in canonical order and D = r1^2 + r2^2 + r3^2:

    x = sqrt(3) (r1^2 - r2^2) / D
    y = (r1^2 + r2^2 - 2 r3^2) / D
    z = 4 sqrt(3) Area / D

is an exact unit vector (Heron's identity), i.e. a point on the Kendall shape sphere,
with z >= 0 always. Canonical order = a chirality-fixing swap of r1/r2 followed by a
cyclic shift making r1 the largest side; the 3-bit permutation code (values 0-5)
records which of the six vertex orderings that was, so matched vertices can be paired
positionally with no search.

A mirrored triangle keeps its side multiset but reverses traversal, which in the
canonical frame exchanges r2 and r3. On (x, y) that is the reflection

    x' = (x + sqrt(3) y) / 2,   y' = (sqrt(3) x - y) / 2,   z' = z

so a single extra KD query at the reflected point covers the mirrored sky -- the basis
of S2's single-pass mirror handling.
"""

import numpy as np

#: code -> the (V1, V2, V3) index triple in canonical order. V-order is (anchor,
#: leg j, leg k) on the database side and the query triplet on the image side.
#: bit 0 = chirality swap (reverses to V3,V2,V1), bit 1 = r2 was largest (cyclic
#: V2,V3,V1), bit 2 = r3 was largest (cyclic V3,V1,V2); cyclic applies after the swap.
PERM_TABLE = np.array([
    (0, 1, 2),   # 0: identity
    (2, 1, 0),   # 1: swap
    (1, 2, 0),   # 2: r2 largest
    (1, 0, 2),   # 3: swap, then r2 largest
    (2, 0, 1),   # 4: r3 largest
    (0, 2, 1),   # 5: swap, then r3 largest
    (0, 1, 2),   # 6, 7: impossible (r2 and r3 cannot both be the strict maximum);
    (0, 1, 2),   #        present so a corrupt code cannot index out of bounds
], dtype=np.int64)

#: the (x, y) reflection a mirrored triangle induces in the canonical frame
MIRROR_XY = np.array([[0.5, np.sqrt(3) / 2],
                      [np.sqrt(3) / 2, -0.5]])


def _shape_xyz(r1, r2, r3):
    """Shape coordinates from canonically ordered side lengths (arrays or scalars)."""
    a, b, c = r1 ** 2, r2 ** 2, r3 ** 2
    denom = a + b + c
    s = 0.5 * (r1 + r2 + r3)
    area_sq = s * (s - r1) * (s - r2) * (s - r3)
    area = np.sqrt(np.maximum(area_sq, 0.0))    # float noise near degeneracy
    x = np.sqrt(3.0) * (a - b) / denom
    y = (a + b - 2 * c) / denom
    z = 4 * np.sqrt(3.0) * area / denom
    return x, y, z


def canonicalise_sides(r1, r2, r3, flip_chirality):
    """(canonical r1, r2, r3, permutation code) for arrays of side lengths.

    ``flip_chirality`` is the boolean chirality test result per triangle -- the
    caller supplies it because the test differs between the 3-D catalogue frame
    (signed volume) and the 2-D pixel frame (cross product), and the two use
    opposite sign conventions to absorb the pixel/sky handedness flip.
    """
    r1 = np.array(r1, dtype=np.float64, copy=True)
    r2 = np.array(r2, dtype=np.float64, copy=True)
    r3 = np.array(r3, dtype=np.float64, copy=True)
    swap = np.asarray(flip_chirality, dtype=bool)
    r1[swap], r2[swap] = r2[swap], r1[swap]

    max_r = np.maximum(np.maximum(r1, r2), r3)
    is2 = r2 == max_r
    is3 = (r3 == max_r) & ~is2          # break exact ties toward the r2 rotation
    code = (swap.astype(np.uint8) | (is2.astype(np.uint8) << 1)
            | (is3.astype(np.uint8) << 2))

    c1 = np.where(is2, r2, np.where(is3, r3, r1))
    c2 = np.where(is2, r3, np.where(is3, r1, r2))
    c3 = np.where(is2, r1, np.where(is3, r2, r3))
    return c1, c2, c3, code


def kendall_rep_3d(v1, v2, v3):
    """Batched shape reps for catalogue-frame triangles ((n, 3) unit vectors each).

    Returns (xyz (n, 3) float64, perm code (n,) uint8). The chirality test is the
    signed volume of the triangle's edges against v2, negated -- the dev-platesolve
    convention, verified against its shipped database.
    """
    e1 = v1 - v2
    e2 = v2 - v3
    e3 = v3 - v1
    orientation = -np.einsum('ij,ij->i', np.cross(e1, e2), v2)
    r1 = np.linalg.norm(e1, axis=-1)
    r2 = np.linalg.norm(e2, axis=-1)
    r3 = np.linalg.norm(e3, axis=-1)
    c1, c2, c3, code = canonicalise_sides(r1, r2, r3, orientation < 0)
    x, y, z = _shape_xyz(c1, c2, c3)
    return np.c_[x, y, z], code


def kendall_rep_2d(v1, v2, v3):
    """Batched shape reps for image-frame triangles ((n, 2) pixel points each).

    The chirality test is the 2-D cross product being *positive* -- opposite to the
    3-D convention, absorbing the handedness flip between pixel and sky frames, so a
    correctly solved (unmirrored) field matches at the same canonical point.
    """
    e1 = v1 - v2
    e2 = v2 - v3
    e3 = v3 - v1
    cross = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    r1 = np.linalg.norm(e1, axis=-1)
    r2 = np.linalg.norm(e2, axis=-1)
    r3 = np.linalg.norm(e3, axis=-1)
    c1, c2, c3, code = canonicalise_sides(r1, r2, r3, cross > 0)
    x, y, z = _shape_xyz(c1, c2, c3)
    return np.c_[x, y, z], code


def mirror_rep(xyz):
    """The shape point a mirrored copy of the same triangle canonicalises to."""
    xyz = np.asarray(xyz, dtype=np.float64)
    out = xyz.copy()
    out[..., :2] = xyz[..., :2] @ MIRROR_XY.T
    return out


def batch_polar_orthogonalise(matrices):
    """One Newton-Schulz step projecting near-rotations back onto SO(3).

    Per-candidate rotations come from exact 3-point solves of noisy data, so they
    are only approximately orthogonal; one step is enough at that error scale (the
    dev-platesolve spike validated a second step changes nothing).
    """
    eye = np.eye(3)
    mtm = np.einsum('nij,nik->njk', matrices, matrices)
    return np.einsum('nij,njk->nik', matrices, (3 * eye - mtm) / 2)


def rotations_to_quaternions(matrices):
    """Unit quaternions (x, y, z, w) for a batch of near-rotation matrices.

    Degenerate candidates (poisoned by the determinant floor upstream) produce
    non-finite rows; those are replaced by distinct far-away placeholders so a
    KD-tree accepts the array and the rows can never join any cluster.
    """
    from scipy.spatial.transform import Rotation

    with np.errstate(over='ignore', invalid='ignore'):
        polished = batch_polar_orthogonalise(matrices)
        # a genuine near-rotation polishes to det ~ +1; degenerate solves stay
        # wildly non-orthogonal (or overflow), however finite their entries
        det = np.linalg.det(polished)
    bad = ~np.isfinite(polished).all(axis=(1, 2)) | ~np.isfinite(det) \
        | (np.abs(det - 1.0) > 0.5)
    if np.any(bad):
        polished[bad] = np.eye(3)
    quat = Rotation.from_matrix(polished).as_quat()
    if np.any(bad):
        quat[bad] = 0.0
        quat[bad, 0] = 1e6 + np.arange(int(bad.sum()))
    return quat


def canonicalise_quaternions(quat, band):
    """Fix the q / -q double cover: flip each quaternion so sum(q) >= 0.

    A rotation whose quaternion sits within ``band`` of the sum(q) = 0 hyperplane
    can be measured on either side of it, splitting one physical cluster in two --
    the failure mode the dev spike tried to patch (its version selected ~50% of
    candidates by a missing abs() and never negated the twins). Returns the
    canonicalised array and the indices needing a *negated* twin inserted.
    """
    s = quat.sum(axis=1)
    canonical = quat * np.where(s >= 0, 1.0, -1.0)[:, None]
    twin_idx = np.nonzero(np.abs(s) < band)[0]
    return canonical, twin_idx


def find_rotation_matrix(image_vectors, catalog_vectors):
    """Least-squares rotation between two ordered sets of unit vectors (Wahba/Kabsch).

    Unlike the v1 `_find_rotation_matrix`, the determinant sign is corrected: without
    it, near-degenerate or mirrored configurations can come back as a *reflection*,
    which decodes into a plausible-looking but wrong (ra, dec, roll).
    """
    H = image_vectors.T @ catalog_vectors
    U, S, V = np.linalg.svd(H)
    if np.linalg.det(U @ V) < 0:
        U = U.copy()
        U[:, -1] = -U[:, -1]
    return U @ V
