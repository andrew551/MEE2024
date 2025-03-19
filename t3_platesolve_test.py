import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import zipfile
import itertools
from scipy.spatial import KDTree
from collections import Counter
import cProfile
import pickle

# MEE2024 imports
import transforms

'''
PARAMETERS (TODO: make controllable by options)
'''
f = 7 # how many anchor stars to check
g = 18 # how many neighbour to check
TOLERANCE_TRIANGLE = 0.001

dbase = np.load("TripleTrianglePlatesolveDatabase2/TripleTriangle_pattern_data2.npz")

triangles = dbase['triangles']
pattern_data = dbase['pattern_data']
anchors = dbase['anchors']
permutation_data = dbase['permutation_data']

#kd_tree = KDTree(triangles.reshape(-1, 3))
with open("kdtree.pkl", "rb") as kdfile:
    kd_tree = pickle.load(kdfile)

print("built kd_tree")
with open("kdtree.pkl", "wb") as kdfile:
    pickle.dump(kd_tree, kdfile)
print(permutation_data.shape, permutation_data.dtype, permutation_data[:10])
print(triangles.shape)

#print(np.linalg.norm(triangles[0, :, :], axis=1))

if 0:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(triangles[0, :, 0], triangles[0, :, 1], triangles[0, :, 2])
    ax.scatter(triangles[1, :, 0], triangles[1, :, 1], triangles[1, :, 2])
    ax.scatter(triangles[3, :, 0], triangles[3, :, 1], triangles[3, :, 2])
    ax.scatter(triangles[4, :, 0], triangles[4, :, 1], triangles[4, :, 2])
    ax.scatter(triangles[5, :, 0], triangles[5, :, 1], triangles[5, :, 2])

    ax.set_aspect('equal')
    plt.show()


'''
vectorised function which computes
the platescale, roll, and ra/dec (as some redundant information in a rotation matrix)
given the database data (triangles, pattern_data, anchors) and match information (match_cand, match_data, match_vect)
'''


def rotation_matrix_error(R):
    """
    Computes a metric for how close a 3x3 matrix is to being a rotation matrix.
    
    Metric:
        d(R) = || R^T R - I ||_F + |det(R) - 1|
        
    Returns:
        A scalar value; smaller values indicate a closer match to a rotation matrix.
    """
    I = np.eye(3)
    ortho_error = np.linalg.norm(R.T @ R - I, ord='fro')  # Frobenius norm
    det_error = abs(np.linalg.det(R) - 1)

    U, S, Vt = np.linalg.svd(R)  # Compute SVD
    R_proctustes = U @ Vt  # Compute the closest rotation matrix

    R_peturbed = R @ ((3 * np.eye(3) - R.T @ R) / 2)
    R_peturbed = R_peturbed @ ((3 * np.eye(3) - R_peturbed.T @ R_peturbed) / 2)
    
    return ortho_error + det_error, np.linalg.norm(R - R_proctustes), np.linalg.norm(R_peturbed - R_proctustes)


def batch_polar_approximation(M):
    """Vectorized one-step polar decomposition for a batch of rotation-like matrices.
    
    M: ndarray of shape (N, 3, 3), where N is the batch size.
    Returns: ndarray of shape (N, 3, 3) with refined rotation matrices.
    """
    N, _, _ = M.shape  # Get batch size
    I = np.eye(3)  # Identity matrix of shape (3, 3)
    
    # Compute M^T @ M for each matrix in the batch
    MtM = np.einsum("nij,njk->nik", M.transpose(0, 2, 1), M)  # (N, 3, 3)

    # Compute the correction factor (3I - M^T M) / 2
    correction = (3 * I - MtM) / 2  # Broadcasts correctly to (N, 3, 3)

    # Compute the refined rotation matrix: M @ correction
    R = np.einsum("nij,njk->nik", M, correction)  # Batch matrix multiplication

    return R

def test_case(target, match, sdat):
    print("in test case")
    print(sdat)
    print(target)
    print(match)
    distances = np.linalg.norm(target[:, 0] - target[:, 1]), np.linalg.norm(target[:, 1] - target[:, 2]), np.linalg.norm(target[:, 2] - target[:, 0])
    dm = np.linalg.norm(match[0] - match[1]), np.linalg.norm(match[1] - match[2]), np.linalg.norm(match[2] - match[0])
    print(distances, dm)
    print(distances / max(distances), dm / max(dm))

def compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect):
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    n = match_cand // triangles.shape[1]
    rem = match_cand % triangles.shape[1]
    t1 = triangles.reshape((-1, 2))[match_cand]
    s1 = pattern_data[n, pairs[rem][:, 0]]
    s2 = pattern_data[n, pairs[rem][:, 1]]
    sdat = np.stack([s1, s2])
    #swap = s1[:, 0] < s2[:, 0]
    #sdat[:, swap, :] = sdat[:, swap, :][(1, 0), :, :]
    


    vec0 = anchors[n]
    vec1 = sdat[0, :, 2:5]
    vec2 = sdat[1, :, 2:5]
    
    perm = permutation_data[n, rem]
    print(Counter(perm))
    mask_12 = (perm & 1).astype(bool)
    mask_2f = (perm & 2).astype(bool)[:, np.newaxis]
    mask_3f = (perm & 4).astype(bool)[:, np.newaxis]

    vec0[mask_12], vec2[mask_12] = vec2[mask_12], vec0[mask_12]

    vec0_final = np.where(mask_2f, vec1, np.where(mask_3f, vec2, vec0))
    vec1_final = np.where(mask_2f, vec2, np.where(mask_3f, vec0, vec1))
    vec2_final = np.where(mask_2f, vec0, np.where(mask_3f, vec1, vec2))

    target = np.stack([vec0_final, vec1_final, vec2_final], axis=2)

    #print(target.shape)
    #for i in range(0, 10000, 2000):
    #    test_case(target[i], match_vect[i], sdat[0, i, 0])

    scale = np.linalg.norm(vec0_final - vec1_final, axis = 1) / np.linalg.norm(match_vect[:, 0] - match_vect[:, 1], axis = 1)
    scaled = match_vect * scale[:, np.newaxis, np.newaxis]
    ####
    #print(match_data[:, plookup[perm]].shape, sdat[0, :, 0].shape)
    #scale = sdat[0, :, 0] / match_data[np.arange(perm.size), plookup[perm]] # note approximate roll: (match_data[:, 1] - s1[:, 1]) % (np.pi*2)
    #scaled = np.einsum('ijk,i -> ijk', match_vect, scale) # scale each triangle
    as_3vect = transforms.icoord_to_vector(scaled).swapaxes(1, 2)
    


    
    # TODO: apply permutation required
    # also: fix scale (also permutation based? or just lookup stars?)
    inv_matrix = np.zeros(as_3vect.shape, as_3vect.dtype)
    for i in range(3):
        for j in range(3):
            ia = [x for x in range(3) if x != i]
            ib = [x for x in range(3) if x != j]
            inv_matrix[:, j, i] = (as_3vect[:, ia[0], ib[0]] * as_3vect[:, ia[1], ib[1]] - as_3vect[:, ia[1], ib[0]] * as_3vect[:, ia[0], ib[1]]) * (-1)**(i+j)
    inv_matrix /= np.linalg.det(as_3vect).reshape((-1, 1, 1))
    rmatrix = np.einsum('...ij,...jk -> ...ik', target, inv_matrix)
    rmatrix = batch_polar_approximation(rmatrix)
    r_errors = np.array([rotation_matrix_error(x) for x in rmatrix])
    print("errors", r_errors[::3000])
    
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2*np.pi)
    return scale, roll, center_vect, rmatrix, target

def get_2Dtriang_rep(v1, v2, v3):
    '''
    v1, v2, v3: x-y coordinates of triangle as arrays
    '''
    r1, r2, r3 = v1-v2, v2-v3, v3-v1
    swap_flag = np.cross(r1, r2) < 0
    r1 = (r1[0]**2+r1[1]**2)**0.5
    r2 = (r2[0]**2+r2[1]**2)**0.5
    r3 = (r3[0]**2+r3[1]**2)**0.5

    if swap_flag:
        r1, r2 = r2, r1
        perm = (2, 1, 0)
    else:
        perm = (0, 1, 2)
    # cylic permute to make r1 the largest of (r1, r2, r3)
    if r2 > r1 and r2 > r3:
        r1, r2, r3 = r2, r3, r1
        perm = (perm[1], perm[2], perm[0]) # TODO check this 50/50 guess
    elif r3 > r1 and r3 > r2:
        r1, r2, r3 = r3, r1, r2
        perm = (perm[2], perm[0], perm[1])
    if (r1 + r2 < r3) or (r1 + r3 < r2) or (r2 + r3 < r1):
        raise ValueError("The given side lengths do not form a valid triangle.")
            
    s = 0.5 * (r1 + r2 + r3)
    area = (s * (s - r1) * (s - r2) * (s - r3))**0.5
    
    # Normalizing denominator (scale invariant quantity)
    denom = r1**2 + r2**2 + r3**2
    
    x = 3**0.5 * (r1**2 - r2**2) / denom
    y = (r1**2 + r2**2 - 2 * r3**2) / denom
    z = (4 * 3**0.5 * area) / denom
    return (x, y, z), perm

def match_image_triangles(centroids, image_shape):
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    vectors = np.c_[centroids[:, 1], centroids[:, 0]] - np.array([image_shape[1], image_shape[0]]) / 2
    reps = []
    match_cand = [] # index of triangle matches
    match_data = [] # [r, phi] the longer side and polar angle of the matched triangles
    match_vect = [] # [v1, v2, v3]
                    # v0: coordinate of the center star in 2D-pixel space,
                    # v1, v2: vectors from center star to the two neighbouring stars
    match_info = []
    triangle_info = []
    for i in range(f):
        for n, (j, k) in enumerate(itertools.combinations(range(g), 2)):
            if j == i or k == i or max(i, k) >= vectors.shape[0]:
                continue
            triplet = (i, j, k)
            rep, perm = get_2Dtriang_rep(vectors[i], vectors[j], vectors[k])
            cand = kd_tree.query_ball_point(rep, TOLERANCE_TRIANGLE)
            ind = np.array(cand) // triangles.shape[1]
            
            array_vect = np.c_[vectors[triplet[perm[0]]], vectors[triplet[perm[1]]], vectors[triplet[perm[2]]]]
            r1 = max(np.linalg.norm(vectors[i]-vectors[j]), np.linalg.norm(vectors[i]-vectors[k]), np.linalg.norm(vectors[j]-vectors[k]))
            sidelengths = (np.linalg.norm(array_vect[:, 0] - array_vect[:, 1]), np.linalg.norm(array_vect[:, 2] - array_vect[:, 1]), np.linalg.norm(array_vect[:, 0] - array_vect[:, 2]))
            #print(triplet, perm)
            #print(r1)
            #print(np.linalg.norm(array_vect[:, 0] - array_vect[:, 1]), np.linalg.norm(array_vect[:, 2] - array_vect[:, 1]), np.linalg.norm(array_vect[:, 0] - array_vect[:, 2]))
            #print(np.linalg.norm(vectors[i]-vectors[j]), np.linalg.norm(vectors[j]-vectors[k]), np.linalg.norm(vectors[k]-vectors[i]))
            if not r1 == np.linalg.norm(array_vect[:, 0] - array_vect[:, 1]):
                raise Exception("max-permute failed")
            phi1 = 0 # dummy var
            for ind_, cand_ in zip(ind, cand):              
                #matches[i][ind_].append((cand_, r1, phi1, array_vect))
                match_cand.append(cand_)
                #match_data.append([r1, phi1])
                match_data.append(sidelengths)
                match_vect.append(array_vect.T)
                match_info.append(triplet)
                rem = cand_ % triangles.shape[1]
                triangle_info.append(pairs[rem])
            reps.append(rep)
    match_cand = np.array(match_cand)
    match_data = np.array(match_data)
    match_vect = np.array(match_vect)
    print(f'{len(reps)=}')
    reps = np.array(reps)
    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2])
        ax.set_aspect('equal')
        plt.show()
    # compute platescales
    scale, roll, center_vect, matrix, target = compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)

    print(scale.shape)
    print(roll.shape)
    print(center_vect.shape)
    print(matrix.shape)
    print(target.shape)
    
def platesolve(centroids, image_shape, options={'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':False, 'flag_debug':False}, output_dir=None, try_mirror_also=True):
    '''
    input:
        centroids: n by 2 array of centroids positions (in pixel space)
        image_shape: shape of image in pixels
        options: dictionary of other parameters
        try_mirror_also: tolerate a mirrored input by also trying to platesolve the mirrored image
    output: dictionary
            "success": True or False
            "platescale", "ra", "dec", "roll": (scale, ra, dec, roll) in arcsec/degrees
            "x": tuple of the above but in RADIANS, and with a 180 degree (pi) flip in roll for some convention consistency (TODO: fix?)
            "matched_centroids": n by 2 array
            "matched_stars": n by 6 array (ra, dec, 3-vect, mag) (but with ra/dec in RADIANS)
    '''
    centroids = np.array(centroids)
    if not len(centroids.shape)==2 or not centroids.shape[1] == 2:
        raise Exception("ERROR: expected an n by 2 array for centroids")
    #plt.scatter(centroids[:, 0], centroids[:, 1])
    #plt.show()
    
    matched_triangs = match_image_triangles(centroids, image_shape)
    

if __name__ == '__main__':
    #database_cache.prepare_triangles()
    options = {'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':1, 'flag_debug':0}
    path_data = r'D:\feb7test\Don2017_clean2\eclipse_field\centroid_data20250214172224.zip' # eclipse (Don)
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('results.txt'))
    df = pd.read_csv(archive.open('STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    centroids = np.c_[df['py'], df['px']] # important: (y, x) representation expected
    cProfile.run("platesolve(centroids, meta_data['img_shape'], options)")
    #result = platesolve(centroids, meta_data['img_shape'], options)

