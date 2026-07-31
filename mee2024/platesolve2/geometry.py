"""
Geometry for the v2 solver.

S1 carries only the orientation fit; the Kendall shape coordinates and quaternion
utilities arrive with stages S2 and S4 (docs/PLATESOLVER_V2_DESIGN.md).
"""

import numpy as np


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
