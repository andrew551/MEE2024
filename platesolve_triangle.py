"""
@author: Andrew Smith
Version 23 March 2024
"""

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats
import scipy
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist
from sklearn.preprocessing import normalize
import itertools
import zipfile
import pandas as pd
from collections import defaultdict
import transforms
import itertools
import json
import matplotlib.pyplot as plt
import cProfile
import time
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import database_cache
from MEE2024util import resource_path, get_bbox
from sklearn.neighbors import NearestNeighbors
import math

'''
PARAMETERS (TODO: make controllable by options)
'''
f = 5 # how many anchor stars to check
g = 8 # how many neighbour to check
TOLERANCE = 0.01 # tolerance for triangle matching
TOL_CENT = np.radians(0.025) # 0.025 degrees in center of frame tolerance
TOL_ROLL = np.radians(0.025) # 0.025 degrees for roll tolerances
log_TOL_SCALE = 0.01      # 1 part in 100 for platescale
MAX_MATCH = 100 # maximum number of verification stars

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
    
    N = math.comb(N_stars_catalog, 3) * math.comb(g, 3) * TOLERANCE**2
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
    #print(n_obs, N_stars_catalog, poisson_lambda, N, x0, x1, threshold)
    return threshold + addon
    
    

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
    '''
    plt.scatter(transformed_all[:, 1], transformed_all[:, 0])
    plt.scatter(candidate_stars[:, 1], candidate_stars[:, 0])
    for i in range(stardata.shape[0]):
        plt.gca().annotate(f'mag={stardata[i, 5]:.2f}', (np.degrees(stardata[i, 0]), np.degrees(stardata[i, 1])), color='black', fontsize=5)
    plt.show()
    ''' 
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
    return stardata, plate2, max_error

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


'''
vectorised function which computes
the platescale, roll, and ra/dec (as some redundant information in a rotation matrix)
given the database data (triangles, pattern_data, anchors) and match information (match_cand, match_data, match_vect)
'''
def compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect):
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    n = match_cand // triangles.shape[1]
    rem = match_cand % triangles.shape[1]
    t1 = triangles.reshape((-1, 2))[match_cand]
    s1 = pattern_data[n, pairs[rem][:, 0]]
    s2 = pattern_data[n, pairs[rem][:, 1]]
    sdat = np.stack([s1, s2])
    swap = s1[:, 0] < s2[:, 0]
    sdat[:, swap, :] = sdat[:, swap, :][(1, 0), :, :]
    scale = sdat[0, :, 0] / match_data[:, 0] # note approximate roll: (match_data[:, 1] - s1[:, 1]) % (np.pi*2)
    scaled = np.einsum('ijk,i -> ijk', match_vect, scale)
    as_3vect = transforms.icoord_to_vector(scaled).swapaxes(1, 2)
    target = np.stack([anchors[n], sdat[0, :, 2:5], sdat[1, :, 2:5]], axis=2)
    inv_matrix = np.zeros(as_3vect.shape, as_3vect.dtype)
    for i in range(3):
        for j in range(3):
            ia = [x for x in range(3) if x != i]
            ib = [x for x in range(3) if x != j]
            inv_matrix[:, j, i] = (as_3vect[:, ia[0], ib[0]] * as_3vect[:, ia[1], ib[1]] - as_3vect[:, ia[1], ib[0]] * as_3vect[:, ia[0], ib[1]]) * (-1)**(i+j)
    inv_matrix /= np.linalg.det(as_3vect).reshape((-1, 1, 1))
    rmatrix = np.einsum('...ij,...jk -> ...ik', target, inv_matrix)
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2*np.pi)
    return scale, roll, center_vect, rmatrix, target

def load():
    data = database_cache.open_catalogue("TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz")
    return data.kd_tree, data.anchors, data.pattern_ind, data.pattern_data, data.triangles
    
def match_triangles(centroids, image_shape, options):
    t0 = time.perf_counter(), time.process_time()
    kd_tree, anchors, pattern_ind, pattern_data, triangles = load()
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    print('loaded database')
    vectors = np.c_[centroids[:, 1], centroids[:, 0]] - np.array([image_shape[1], image_shape[0]]) / 2 # zero-centre pixel vectors, also (for some reason) use (x, y) convention
    #vectors = np.c_[df['px'], df['py']] - np.array([meta_data['img_shape'][1], meta_data['img_shape'][0]]) / 2
    #plt.scatter(vectors[:, 0], vectors[:, 1])
    #plt.show()
    print('mean:', np.mean(vectors, axis=0))
    #matches = [defaultdict(list) for _ in range(f)]
    match_cand = [] # index of triangle matches
    match_data = [] # [r, phi] the longer side and polar angle of the matched triangles
    match_vect = [] # [v1, v2, v3]
                    # v0: coordinate of the center star in 2D-pixel space,
                    # v1, v2: vectors from center star to the two neighbouring stars
    match_info = []
    triangle_info = []
    t1 = time.perf_counter(), time.process_time()
    print(f" Real time loading: {t1[0] - t0[0]:.2f} seconds")
    for i in range(f):
        for n, (j, k) in enumerate(itertools.combinations(range(g), 2)):
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
                ratio = 1/ratio
                dphi = -dphi
                phi1, phi2 = phi2, phi1
                r1, r2 = r2, r1
                v1, v2 = v2, v1
                triplet = (i, k, j)
            dphi = dphi % (2 * np.pi)
            cand = kd_tree.query_ball_point([ratio, dphi], TOLERANCE)
            ind = np.array(cand) // triangles.shape[1]
            array_vect = np.c_[v0, v1 + v0, v2 + v0]
            for ind_, cand_ in zip(ind, cand):              
                #matches[i][ind_].append((cand_, r1, phi1, array_vect))
                match_cand.append(cand_)
                match_data.append([r1, phi1])
                match_vect.append(array_vect.T)
                match_info.append(triplet)
                rem = cand_ % triangles.shape[1]
                triangle_info.append(pairs[rem])
    t2 = time.perf_counter(), time.process_time()
    #print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
    match_cand = np.array(match_cand)
    match_data = np.array(match_data)
    match_vect = np.array(match_vect)
    print(f" Real time prepare: {t2[0] - t1[0]:.2f} seconds")
    #find_matching_triangles(matches, triangles, pattern_data, anchors, given_scale)
    t3 = time.perf_counter(), time.process_time()
    print(f" Real time match: {t3[0] - t2[0]:.2f} seconds")
    #cProfile.runctx('compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)', globals(), locals())
    scale, roll, center_vect, matrix, target = compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)
    t4 = time.perf_counter(), time.process_time()
    print(f" Real time platescale compute: {t4[0] - t3[0]:.2f} seconds")
    return scale, roll, center_vect, match_info, triangle_info, vectors, target

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
def platesolve(centroids, image_shape, options={'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':False, 'flag_debug':False}, output_dir=None, try_mirror_also=True):
    centroids = np.array(centroids)
    if not len(centroids.shape)==2 or not centroids.shape[1] == 2:
        raise Exception("ERROR: expected an n by 2 array for centroids")
    result = _platesolve_helper(centroids, image_shape, options, output_dir=output_dir)
    # if we are friendly, could mirror (x, y) and try again if failed
    result['mirror'] = False
    if result['success'] or not try_mirror_also:
        return result
    print('platesolve failed ... trying mirror image of field')
    centroids = np.copy(centroids)
    centroids[:, [0, 1]] = centroids[:, [1, 0]]
    image_shape = (image_shape[1], image_shape[0])
    result = _platesolve_helper(centroids, image_shape, options, output_dir=output_dir)
    if result['success']:
        result['mirror'] = True
        result['matched_centroids'][:, [0, 1]] = result['matched_centroids'][:, [1, 0]]
    return result

def _platesolve_helper(centroids, image_size, options, output_dir=None):
    dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    N_stars_catalog = dbs.star_table.shape[0]
    t00 = time.perf_counter(), time.process_time()
    scale, roll, center_vect, match_info, triangle_info, vectors, target_vectors = match_triangles(centroids, image_size, options)
    print(f'initial triangle matches: {scale.shape[0]}')
    n_obs = centroids.shape[0]
    all_star_plate = centroids - np.array([image_size[0]/2, image_size[1]/2])
    t2 = time.perf_counter(), time.process_time()

    vector_plates = np.c_[np.log(scale) / log_TOL_SCALE, roll / TOL_ROLL, center_vect / TOL_CENT] 
    tree_matches = KDTree(vector_plates)
    t3 = time.perf_counter(), time.process_time()
    candidate_pairs = tree_matches.query_pairs(1) # efficiently find all pairs of agreeing triangles
    N = vector_plates.shape[0]
    graph = csr_matrix(([1 for _ in candidate_pairs], ([x[0] for x in candidate_pairs], [x[1] for x in candidate_pairs])), shape=(N, N))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    counts = dict(zip(unique, counts))
    best=-1
    best_result = {'success':False, 'x':None, 'platescale':None, 'matched_centroids':None, 'matched_stars':None, 'platescale/arcsec':None, 'ra':None, 'dec':None, 'roll':None}
    n_matches = 0
    for i in range(n_components):
        if counts[i] >= 3:
            indices = np.nonzero(labels==i)[0]
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
                print('triangle match:', len(non_redundant), [match_info[_] for _ in non_redundant])
                #print(counts[i], radec, scale[el], roll[el], match_info[el])
                #print(matchset)
                if options['flag_debug']:
                    # show platesolve
                    plt.scatter(vectors[:, 0], vectors[:, 1])
                    for t in non_redundant:
                        tri = match_info[t]
                        v = np.array([vectors[_] for _ in tri]+[vectors[tri[0]]])
                        plt.plot(v[:, 0], v[:, 1], color='red')
                    plt.gca().invert_yaxis()
                    plt.title(f"{len(non_redundant)} triangles matched\nplatescale={np.degrees(scale[el])*3600:.4f} arcsec/pixel\nra={radec[0][1]:.4f}, dec={radec[0][0]:.4f}")
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
                stardata, plate2, max_error = match_centroids(centroids[:MAX_MATCH, :], np.radians(platescale), image_size, options)
                #print('max_error', max_error)
                thresh = estimate_acceptance_threshold(min(n_obs, MAX_MATCH), N_stars_catalog, max_error, g, addon=3)
                
                if stardata.shape[0] >= thresh:
                    n_matches += 1
                    print(f"MATCH ACCEPTED (nstars matched = {stardata.shape[0]}, thresh = {thresh})")
                    rms = 3600*np.degrees(np.linalg.norm(catvects - (rotation_matrix.T @ ivects.T).T) / catvects.shape[0])
                    print('accurate ra dec roll', acc_ra, acc_dec, acc_roll, 'rough rms=', rms, 'arcsec')
                    if stardata.shape[0] > best:
                        best = stardata.shape[0]
                        best_non_redundant = non_redundant
                        best_result = {'success':True, 'x': np.radians(platescale), 'platescale/arcsec':3600*np.degrees(scale[el]), 'ra':acc_ra, 'dec':acc_dec, 'roll':acc_roll, 'matched_centroids':plate2+np.array([image_size[0]/2, image_size[1]/2]), 'matched_stars':stardata}
                else:
                    print(f"note: candidate match rejected (nstars matched = {stardata.shape[0]}, thresh = {thresh})")         
    print(f'npairs = {len(candidate_pairs)}')
    t4 = time.perf_counter(), time.process_time()
    print(f" Real star matching: {t4[0] - t2[0]:.2f} seconds")
    tff = time.perf_counter(), time.process_time()
    print(f" TOTAL TIME: {tff[0] - t00[0]:.2f} seconds")
    if n_matches > 1:
        print(f"WARNING: multiple ({n_matches}) platesolves were successful, returning best one")
    elif n_matches == 0:
        print("Platesolve FAILED")
    elif n_matches == 1:
        print("Platescale SUCCESS")
    if (options['flag_display'] or not output_dir is None) and n_matches >= 1:
        # show platesolve
        plt.scatter(vectors[:, 0]+image_size[1], vectors[:, 1]+image_size[0])
        for t in best_non_redundant:
            tri = match_info[t]
            v = np.array([vectors[_] for _ in tri]+[vectors[tri[0]]])
            plt.plot(v[:, 0]+image_size[1], v[:, 1]+image_size[0], color='red')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title(f"{len(best_non_redundant)} triangles matched\nplatescale={best_result['platescale/arcsec']:.4f} arcsec/pixel\nra={best_result['ra']:.4f}, dec={best_result['dec']:.4f}, roll={best_result['roll']:.4f}")
        plt.tight_layout()
        if not output_dir is None:
            plt.savefig(output_dir / 'triangle_matches.png', dpi=600)
        if options['flag_display']:
            plt.show()
        plt.close()
    return best_result

if __name__ == '__main__':
    database_cache.prepare_triangles()
    #cProfile.run('main()')
    options = {'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':1, 'flag_debug':0}
    #path_data = 'D:/output4/CENTROID_OUTPUT20240229002931/data.zip' # zwo 3 zd 30
    path_data = 'D:/output4/CENTROID_OUTPUT20240303034855/data.zip' # eclipse (Don)
    #path_data = 'D:/output4/CENTROID_OUTPUT20240303040025/data.zip' # eclipse (Don) right
    #path_data = 'D:/output4/CENTROID_OUTPUT20240310015116/data.zip' # eclipse (Berry)
    #path_data = 'D:/output4/CENTROID_OUTPUT20240310020236/data.zip' # ZWO 1
    #path_data = 'D:/output4/CENTROID_OUTPUT20240310194735/data.zip' # moontest 1
    #path_data = 'D:/output4/CENTROID_OUTPUT20240310195034/data.zip' # moontest 3
    #path_data = 'E:/extra data\data.zip' # another moon test
    #path_data = 'D:/output4/CENTROID_OUTPUT20240310200107/data.zip' # zwo 3 zd 75
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('data/results.txt'))
    df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    centroids = np.c_[df['py'], df['px']] # important: (y, x) representation expected

    result = platesolve(centroids, meta_data['img_shape'], options)

    print(result)

    path_data = 'D:/output4/CENTROID_OUTPUT20240310195034/data.zip' # moontest 3
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('data/results.txt'))
    df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    centroids = np.c_[df['py'], df['px']] # important: (y, x) representation expected

    result = platesolve(centroids, meta_data['img_shape'], options)

    print(result)
    
    
