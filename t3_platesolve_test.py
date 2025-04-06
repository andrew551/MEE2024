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
from scipy.stats import binom
from scipy.spatial.distance import pdist, cdist
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation as scipy_R
from collections import deque
from scipy.sparse import csr_matrix
import time
import math
import scipy
# MEE2024 imports
import transforms
from MEE2024util import resource_path, get_bbox
import database_cache
from sklearn.neighbors import NearestNeighbors
import tqdm
import line_profiler

'''
PARAMETERS (TODO: make controllable by options)
'''
f = 7 # how many anchor stars to check
g = 18 # how many neighbour to check
TOLERANCE_TRIANGLE = 0.001 # recommended value: 4 / img_size (?)
# The following are quite loose bounds. The aim of this is too be tolerant of significant image distortion
TOL_CENT = np.radians(0.025) # 0.025 degrees in center of frame tolerance == 90 arcsec
TOL_ROLL = np.radians(0.025) # 0.025 degrees for roll tolerances == 90 arcsec
log_TOL_SCALE = 0.01      # 1 part in 100 for platescale
MAX_MATCH = 32 # maximum number of verification stars

dbase = np.load("TripleTrianglePlatesolveDatabase2/TripleTriangle_pattern_data2.npz")

triangles = dbase['triangles']
pattern_data = dbase['pattern_data']
anchors = dbase['anchors']
permutation_data = dbase['permutation_data']

kd_tree = KDTree(triangles.reshape(-1, 3))

READ_OLD = True

if READ_OLD:
    with open("kdtree.pkl", "rb") as kdfile:
        kd_tree = pickle.load(kdfile)
    print("built kd_tree")
else:
    with open("kdtree.pkl", "wb") as kdfile:
    	pickle.dump(kd_tree, kdfile)
print(permutation_data.shape, permutation_data.dtype, permutation_data[:10])
print(triangles.shape)

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

@line_profiler.profile # profile the code
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
    #print(Counter(perm))
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
    rmatrix = batch_polar_approximation(rmatrix) # first order iterative correction to rotation matrix
    #r_errors = np.array([rotation_matrix_error(x) for x in rmatrix])
    #print("errors", r_errors[::3000])
    # TODO: get quaternion rep of rmatrix
    rot = scipy_R.from_matrix(rmatrix)
    quat = rot.as_quat()
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2*np.pi)
    return scale, roll, center_vect, rmatrix, target, quat

def get_2Dtriang_rep(v1, v2, v3):
    '''
    v1, v2, v3: x-y coordinates of triangle as arrays
    '''
    r1, r2, r3 = v1-v2, v2-v3, v3-v1
    swap_flag = r1[0]*r2[1] - r1[1]*r2[0] > 0
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

@line_profiler.profile # profile the code
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
    dmat = np.array([[((va[0]-vb[0])**2+(va[1]-vb[1])**2)**0.5 for vb in vectors] for va in vectors])
    array_vect = np.zeros((2, 3), dtype=np.float64)
    for i in range(f):
        for n, (j, k) in enumerate(itertools.combinations(range(g), 2)):
            if j == i or k == i or max(i, k) >= vectors.shape[0]:
                continue
            triplet = (i, j, k)
            rep, perm = get_2Dtriang_rep(vectors[i], vectors[j], vectors[k])
            triplet = (triplet[perm[0]], triplet[perm[1]], triplet[perm[2]]) # apply permutation to triplet to get "standard order"
            cand = kd_tree.query_ball_point(rep, TOLERANCE_TRIANGLE)
            ind = np.array(cand) // triangles.shape[1]
            for ii in range(3):
                array_vect[:, ii] = vectors[triplet[ii]]
                
            sidelengths = (dmat[triplet[0], triplet[1]], dmat[triplet[1], triplet[2]], dmat[triplet[2], triplet[0]])

            array_vect_copy = array_vect.copy().T
            for ind_, cand_ in zip(ind, cand):  
                match_cand.append(cand_)
                match_data.append(sidelengths)
                match_vect.append(array_vect_copy)
                match_info.append(triplet)
                rem = cand_ % triangles.shape[1]
                triangle_info.append(pairs[rem])
            reps.append(rep)
    match_cand = np.array(match_cand)
    match_data = np.array(match_data)
    match_vect = np.array(match_vect)
    reps = np.array(reps)
    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2])
        ax.set_aspect('equal')
        plt.show()
    # compute platescales
    scale, roll, center_vect, matrix, target, quat = compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)

    return scale, roll, center_vect, match_info, triangle_info, vectors, target, quat

def match_platescales(centroids, image_size, options={'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':False, 'flag_debug':False}, output_dir=None, try_mirror_also=True, print_flag=True):
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
    result = match_platescales_helper(centroids, image_size, options, output_dir=output_dir, print_flag=print_flag)
    # if we are friendly, could mirror (x, y) and try again if failed
    result['mirror'] = False
    if result['success'] or not try_mirror_also:
        return result
    if print_flag:
        print('platesolve failed ... trying mirror image of field')
    centroids = np.copy(centroids)
    centroids[:, [0, 1]] = centroids[:, [1, 0]]
    image_size = (image_size[1], image_size[0])
    result = match_platescales_helper(centroids, image_size, options, output_dir=output_dir, print_flag=print_flag)
    if result['success']:
        result['mirror'] = True
        result['matched_centroids'][:, [0, 1]] = result['matched_centroids'][:, [1, 0]]
    return result

@line_profiler.profile # profile the code
def match_platescales_helper(centroids, image_size, options, output_dir=None, print_flag=True):
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
    dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    N_stars_catalog = dbs.star_table.shape[0]

    t00 = time.perf_counter(), time.process_time()
    
    scale, roll, center_vect, match_info, triangle_info, vectors, target_vectors, quat = match_image_triangles(centroids, image_size)
    if print_flag:
        print(f'initial triangle matches: {scale.shape[0]}')
        print(f'{quat.shape=}')
    n_obs = centroids.shape[0]
    all_star_plate = centroids - np.array([image_size[0]/2, image_size[1]/2])
    t2 = time.perf_counter(), time.process_time()

    vector_plates = np.c_[np.log(scale) / log_TOL_SCALE, roll / TOL_ROLL, center_vect / TOL_CENT] 
    tree_matches = KDTree(vector_plates)
    t3 = time.perf_counter(), time.process_time()
    


    

    if 0:
        candidate_pairs = tree_matches.query_pairs(1) # efficiently find all pairs of agreeing triangles
        N = vector_plates.shape[0]
        graph = csr_matrix(([1 for _ in candidate_pairs], ([x[0] for x in candidate_pairs], [x[1] for x in candidate_pairs])), shape=(N, N))
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        unique, counts = np.unique(labels, return_counts=True)

    # TODO: include twice vectors slightly below equator
    tree_quat = KDTree(np.c_[np.log(scale) / log_TOL_SCALE, quat / TOL_ROLL])
    candidate_quat_pairs = tree_quat.query_pairs(1)
    N = vector_plates.shape[0]
    graph_quat = csr_matrix(([1 for _ in candidate_quat_pairs], ([x[0] for x in candidate_quat_pairs], [x[1] for x in candidate_quat_pairs])), shape=(N, N))
    n_components_quat, labels_quat = connected_components(csgraph=graph_quat, directed=False, return_labels=True)
    unique_quat, counts_quat = np.unique(labels_quat, return_counts=True)
    
    
    if print_flag:
        #print("counts:", Counter(counts))
        print("counts_quat:", Counter(counts_quat))
    nviews_unique = sum(Counter(counts_quat).values())
    counts = dict(zip(unique_quat, counts_quat))
    #print(f'{nviews_unique=}')
    best=-1
    best_result = {'success':False, 'x':None, 'platescale':None, 'matched_centroids':None, 'matched_stars':None, 'platescale/arcsec':None, 'ra':None, 'dec':None, 'roll':None}
    n_matches = 0
    t33 = time.perf_counter(), time.process_time()
    for i in range(n_components_quat):#np.argsort(counts)[::-1]: # try most promising matches first
        if counts_quat[i] >= 3:
            indices = np.nonzero(labels_quat==i)[0]
            # remove redundant triangles (a, b, c), (b, a, c) etc.
            seen = set()
            non_redundant = []
            for ind in indices:
                if match_info[ind] in seen:
                    continue
                seen.update(list(itertools.permutations(match_info[ind])))
                non_redundant.append(ind)
            if len(non_redundant) >= 3:
                matchset = dict()
                for ind in non_redundant:
                    matchset.update(zip(match_info[ind], target_vectors[ind].T))

                el = non_redundant[0]
                radec = transforms.to_polar(center_vect[el])
                #print('triangle match:', len(non_redundant), [match_info[_] for _ in non_redundant])
                #print(counts[i], radec, scale[el], roll[el], match_info[el])
                #print(matchset)
                if options['flag_debug']:
                    # show platesolve
                    plt.scatter(centroids[:, 0], centroids[:, 1])
                    for t in non_redundant:
                        tri = match_info[t]
                        v = np.array([centroids[_] for _ in tri]+[centroids[tri[0]]])
                        plt.plot(v[:, 0], v[:, 1], color='red')
                    plt.gca().invert_yaxis()
                    plt.title(f"{len(non_redundant)} triangles matched\nplate scale={np.degrees(scale[el])*3600:.4f} arcsec/pixel\nra={radec[0][1]:.4f}, dec={radec[0][0]:.4f}")
                    plt.show()
                plate = (np.degrees(scale[el]), radec[0][1], radec[0][0], np.degrees(roll[el])+90) # this plus 90 is very weird and probably is need because of a coordinate bug
                #print('scale/degrees, ra, dec, roll', plate)
                
                ivects = transforms.icoord_to_vector(np.array([all_star_plate[_] for _ in matchset])*scale[el])
                catvects = np.array([_ for _ in matchset.values()])
                #print(ivects)
                #print(catvects)
                rotation_matrix = _find_rotation_matrix(ivects, catvects)
                acc_ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                                   rotation_matrix[0, 0])) % 360
                acc_dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                                    np.linalg.norm(rotation_matrix[1:3, 2])))
                acc_roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                                     rotation_matrix[2, 2])) % 360
                acc_roll = (acc_roll + 180) % 360 # ???
                
                #print((rotation_matrix.T @ ivects.T).T)
                platescale = (np.degrees(scale[el]), acc_ra, acc_dec, acc_roll+180) # do weird +180 roll thing as usual
                stardata, plate2, max_error, errors = match_centroids2(centroids[:MAX_MATCH, :], np.radians(platescale), image_size, options)
                #print('max_error', max_error)
                rms_error_match = np.mean(errors**2)**0.5
                thresh = estimate_acceptance_threshold(min(n_obs, MAX_MATCH), N_stars_catalog, rms_error_match, g, addon=3)
                p_value = calculate_pvalue(min(n_obs, MAX_MATCH), N_stars_catalog, errors, nviews_unique, stardata.shape[0])
                p_value_thresh = calculate_pvalue(min(n_obs, MAX_MATCH), N_stars_catalog, errors, nviews_unique, thresh-3)
                print(f"p_value={p_value} {n_obs=} nmatch={stardata.shape[0]} {max_error=}; {thresh=} {p_value_thresh=}")
                if stardata.shape[0] >= thresh:
                    n_matches += 1
                    rms = 3600*np.degrees(np.linalg.norm(catvects - (rotation_matrix.T @ ivects.T).T) / catvects.shape[0])
                    
                    if print_flag:
                        print(f"MATCH ACCEPTED (nstars matched = {stardata.shape[0]}, thresh = {thresh})")
                        print('accurate ra dec roll', acc_ra, acc_dec, acc_roll, 'rough rms=', rms, 'arcsec')
                    if stardata.shape[0] > best:
                        best = stardata.shape[0]
                        best_non_redundant = non_redundant
                        best_result = {'success':True, 'x': np.radians(platescale), 'platescale/arcsec':3600*np.degrees(scale[el]), 'ra':acc_ra, 'dec':acc_dec, 'roll':acc_roll, 'matched_centroids':plate2+np.array([image_size[0]/2, image_size[1]/2]), 'matched_stars':stardata}
                else:
                    if print_flag:
                        print(f"note: candidate match rejected (nstars matched = {stardata.shape[0]}, thresh = {thresh})")         
    
    t4 = time.perf_counter(), time.process_time()
    tff = time.perf_counter(), time.process_time()
    if print_flag:
        print(f'npairs = {len(candidate_quat_pairs)}')
        print(f" Real star matching: {t4[0] - t33[0]:.2f} {t33[0] - t3[0]:.2f} {t3[0] - t2[0]:.2f} seconds")
        print(f" TOTAL TIME: {tff[0] - t00[0]:.2f} seconds")
        if n_matches > 1:
            print(f"WARNING: multiple ({n_matches}) platesolves were successful, returning best one")
        elif n_matches == 0:
            print("Platesolve FAILED")
        elif n_matches == 1:
            print("Platescale SUCCESS")
    if (options['flag_display2'] or not output_dir is None) and n_matches >= 1:
        # show platesolve
        plt.scatter(centroids[:, 1], centroids[:, 0])
        plt.xlim(0, image_size[1])
        plt.ylim(0, image_size[0])
        
        for t in best_non_redundant:
            tri = match_info[t]
            v = np.array([centroids[_] for _ in tri]+[centroids[tri[0]]])
            plt.plot(v[:, 1], v[:, 0], color='red')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("pixel X", fontsize=16)
        plt.ylabel("pixel Y", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f"{len(best_non_redundant)} triangles matched\nplatescale={best_result['platescale/arcsec']:.4f} arcsec/pixel\nra={best_result['ra']:.4f}, dec={best_result['dec']:.4f}, roll={best_result['roll']:.4f}", fontsize=16)
        plt.tight_layout()
        if not output_dir is None:
            plt.savefig(output_dir / 'triangle_matches.png', dpi=600)
        if options['flag_display2']:
            plt.show()
        plt.close()
    return best_result
    
'''
statistically estimate how many stars need to be matched to a given accuracy in order to accept a platesolve
n_obs: how many stars were observed
N_star_catalog: how many stars in the catalog
threshold_match: radians: all matched stars are within this limit of each other
g: how many oberserved stars are used to platesolve
addon: empirical integer to add to threshold to get a "significant" value. For the limit as N_stars -> infinity, addon=2 already
will provide an assurance approaching certainty that the match is correct. Default: 3

note: not taken into account that stars dimmer than the dimmest star in the catalog should be excluded from the observed stars
note2: we assume stars are isotropically distributed in the sky
'''
def estimate_acceptance_threshold(n_obs, N_stars_catalog, threshold_match, g, addon=3):
    p = N_stars_catalog * threshold_match**2 / 4 # propability that a randomly chosen point will be with threshold of a star.
    # the factor of 4 comes from the ratio of the surface area of a sphere to a circle of a given radius

    poisson_lambda = p*(n_obs-3) # for a single random match, the number of matches can be approximated by a Poisson distribution
    # the minus three is because three of the observed stars are used to platesolve a match
    
    N = math.comb(N_stars_catalog, 3) * math.comb(g, 3) * TOLERANCE_TRIANGLE**2 / 4
    # number of "attempts" at sampling the Poisson distribution we have by matching a triangle of
    # observed stars to a triangle of catalogue stars
    # note that this is quite a vast overestimate - since almost all triangles will not
    # have matching shapes. However, as we only deal with log(N) I think this will end up being
    # an O(1) correction to the computed threshold. Also note that overestimating N will cause an
    # overestimate in the threshold for platesolve acceptance, which is better than an underestimate
    # **UPDATE** add a "TOLERANCE**2" correction to the number of possible triangles. This is still probably a good upper bound for N,
    # but within a smaller numeric factor (probably some function of pi)
                                                        
    #Now we make use of result in "A note on the distribution of the maximum of a set of Poisson random variables
    #Keith Briggs∗, Linlin Song (BT Research, Martlesham) & Thomas Prellberg (Mathematics, QMUL), 2009-03-12

    x0 = math.log(N) / scipy.special.lambertw(math.log(N) / (math.exp(1) * poisson_lambda)).real
    x1 = x0 + (math.log(poisson_lambda) - poisson_lambda - math.log(2*math.pi)/2 - 3 * math.log(x0)/2) / (math.log(x0) - math.log(poisson_lambda))
    # Threshold ~ 3 + int(x1) (the '3' comes from the triangle of 3 stars that is already matched)
    threshold = round(x1) + 3
    ans = threshold + addon
    if ans > n_obs:
        ans -= addon
    return ans
    
def calculate_pvalue(n_obs, N_stars_catalog, errors, nviews, nmatched):
    p_arr = N_stars_catalog * errors**2 / 4 # propability that a randomly chosen point will be with threshold of a star.
    p = 1 - np.exp(np.log(1-p_arr).mean()) # bound a sum of Bernoulli Distributions with a binomial (** TODO: insert proof)
    binom_p = scipy.stats.binom.sf(nmatched - 3 - 1, n_obs - 3, p) # minus three because triangle is already matched; minus one for off-by-one from survival function
    print("binom_p: ", binom_p)
    pvalue = 1 - math.exp(-binom_p*nviews) # more numerically stable approximation of 1 - (1-binom_p) ** nviews
    return pvalue

@line_profiler.profile # profile the code
def match_centroids2(centroids, platescale_fit, image_size, options):
    confusion_ratio = 2 # closest match must be 2x closer than second place
    dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    corners = transforms.to_polar(transforms.linear_transform(platescale_fit, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    stardata = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=12)[0]
    all_star_plate = centroids - np.array([image_size[0]/2, image_size[1]/2])
    all_vectors = transforms.linear_transform(platescale_fit, all_star_plate)
    transformed_all = transforms.to_polar(all_vectors)
    # match nearest neighbours
    candidate_stars = np.zeros((stardata.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(stardata[:, 1])
    candidate_stars[:, 1] = np.degrees(stardata[:, 0])
    candidate_star_vectors = stardata[:, 2:5]
    if options['flag_debug']:
        plt.scatter(transformed_all[:, 1], transformed_all[:, 0])
        plt.scatter(candidate_stars[:, 1], candidate_stars[:, 0])
        for i in range(min(1000,stardata.shape[0])):
            plt.gca().annotate(f'mag={stardata[i, 5]:.2f}', (np.degrees(stardata[i, 0]), np.degrees(stardata[i, 1])), color='black', fontsize=5)
        plt.show()
    if candidate_star_vectors.shape[0] < 3:
        # failure case - shouldn't happen unless input is incorrectly specified
        return np.empty((0, 2)), None, np.radians(options['rough_match_threshhold']/3600)

    cand_tree = KDTree(candidate_star_vectors, balanced_tree=False, compact_nodes=False)
    matched_ind_cata = []
    matched_ind_obs = []
    matched_errors = []
    for i in range(all_vectors.shape[0]):
        close = cand_tree.query_ball_point(all_vectors[i], np.radians(options['rough_match_threshhold']/3600))
        if len(close):
            errors_i = np.linalg.norm(stardata[close, 2:5]-all_vectors[i], axis=1)
            argsorted = np.argsort(errors_i)
            if len(close) == 1 or errors_i[argsorted[0]] < errors_i[argsorted[1]] / confusion_ratio:
                matched_ind_cata.append(close[argsorted[0]])
                matched_ind_obs.append(i)
                matched_errors.append(errors_i[argsorted[0]])

    stardata = stardata[matched_ind_cata, :]            
    plate2 = all_star_plate[matched_ind_obs, :]
    max_error = max(matched_errors)
    return stardata, plate2, max_error, np.array(matched_errors)



# TODO: fix performance of this function. It should be much faster

def match_centroids(centroids, platescale_fit, image_size, options):
    dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    corners = transforms.to_polar(transforms.linear_transform(platescale_fit, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    stardata = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=12)[0]
    all_star_plate = centroids - np.array([image_size[0]/2, image_size[1]/2])
    all_vectors = transforms.linear_transform(platescale_fit, all_star_plate)
    transformed_all = transforms.to_polar(all_vectors)
    # match nearest neighbours
    candidate_stars = np.zeros((stardata.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(stardata[:, 1])
    candidate_stars[:, 1] = np.degrees(stardata[:, 0])
    candidate_star_vectors = stardata[:, 2:5]
    if options['flag_debug']:
        plt.scatter(transformed_all[:, 1], transformed_all[:, 0])
        plt.scatter(candidate_stars[:, 1], candidate_stars[:, 0])
        for i in range(min(1000,stardata.shape[0])):
            plt.gca().annotate(f'mag={stardata[i, 5]:.2f}', (np.degrees(stardata[i, 0]), np.degrees(stardata[i, 1])), color='black', fontsize=5)
        plt.show()
    if candidate_star_vectors.shape[0] < 3:
        # failure case - shouldn't happen unless input is incorrectly specified
        return np.empty((0, 2)), None, np.radians(options['rough_match_threshhold']/3600)
     
    # find nearest two catalogue stars to each observed star
    # use 3-vector distance (and small angle approximation)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(candidate_star_vectors)
    distances, indices = neigh.kneighbors(all_vectors)

    # find nearest observed star to each catalogue star
    neigh_bar = NearestNeighbors(n_neighbors=1)

    neigh_bar.fit(all_vectors)
    distances_bar, indices_bar = neigh_bar.kneighbors(candidate_star_vectors)

    match_threshhold = np.radians(options['rough_match_threshhold']/3600) # threshold in arcsec -> radians
    confusion_ratio = 2 # closest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio)
    keep = np.logical_and(keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0])) # is the nearest-neighbour relation reflexive? [this eliminates 1-to-many matching]
    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]
    if options['flag_debug']:
        plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
        plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations')
        for i in range(stardata.shape[0]):
            if i in indices[keep_i, 0]:
                plt.gca().annotate(f'mag={stardata[i, 5]:.2f}', (np.degrees(stardata[i, 0]), np.degrees(stardata[i, 1])), color='black', fontsize=5)
        plt.xlabel('RA')
        plt.ylabel('DEC')
        plt.title('initial rough fit')
        plt.legend()
        plt.show()
        plt.close()

    stardata= stardata[indices[keep_i, 0].flatten(), :]
    plate2 = all_star_plate[keep_i, :][0]

    all_vectors = all_vectors[keep_i, :][0]
    errors = np.linalg.norm(stardata[:, 2:5]-all_vectors, axis=1)
    max_error = np.max(errors) if errors.size else match_threshhold
    return stardata, plate2, max_error, errors

# note: lifted from tetra
def _find_rotation_matrix(image_vectors, catalog_vectors):
    """Calculate the least squares best rotation matrix between the two sets of vectors.
    image_vectors and catalog_vectors both Nx3. Must be ordered as matching pairs.
    """
    # find the covariance matrix H between the image and catalog vectors
    H = np.dot(image_vectors.T, catalog_vectors)
    # use singular value decomposition to find the rotation matrix
    (U, S, V) = np.linalg.svd(H)
    return np.dot(U, V)

if __name__ == '__main__':
    #database_cache.prepare_triangles()
    print("in main")
    options = {'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':0, 'flag_debug':0}
    #path_data = r'D:\feb7test\Don2017_clean2\eclipse_field\centroid_data20250214172224.zip' # eclipse (Don)
    path_data = r'D:\feb7test\station1\centroid_data20250320001655.zip' # zenith (Don)
    #path_data = r'D:\Station 1 data\centroid_data20240416232626.zip' # Station 1 2024
    #path_data = r'D:\feb7test\station1\centroid_data20250320225454.zip' # moon (hard)
    path_data = r'D:\feb7test\station1\centroid_data20250320231043.zip' # moon (hard)
    #path_data = r'/home/maxim/Downloads/centroid_data20250320230811.zip'
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('results.txt'))
    print(f'{meta_data["img_shape"]=}')
    df = pd.read_csv(archive.open('STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    centroids = np.c_[df['py'], df['px']] # important: (y, x) representation expected
    #centroids = np.c_[df['px'], df['py']] # important: (y, x) representation expected
    #cProfile.run("platesolve(centroids, meta_data['img_shape'], options)")
    result = match_platescales(centroids, meta_data['img_shape'], options, print_flag=True)
    #print(result)
    #end
    def test():
        np.random.seed(123)
        for sim in tqdm.tqdm(range(30)):
            simarr = np.random.random((30, 2))
            result = match_platescales(simarr, [1,1], options, print_flag=False)
            #print(result['success'])
    test()
    #cProfile.run("test()", sort='time')
        #if result['success']:
        #    print(sim)
    

