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
from distortion_fitter import get_bbox
import database_cache
from MEE2024util import resource_path
from sklearn.neighbors import NearestNeighbors

'''
PARAMETERS
'''
f = 5 # how many anchor stars to check
g = 8 # how many neighbour to check
TOLERANCE = 0.01 # tolerance for triangle matching
TOL_CENT = np.radians(0.025) # 0.025 degrees in center of frame tolerance
TOL_ROLL = np.radians(0.025) # 0.025 degrees for roll tolerances
log_TOL_SCALE = 0.01      # 1 part in 100 for platescale
NSTARTHRESHOLD = 7 # how many stars need to be matched for platesolve to be accepted

def match_centroids(df, platescale_fit, image_size, options):
    dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    corners = transforms.to_polar(transforms.linear_transform(platescale_fit, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    stardata = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=12)[0]
    all_star_plate = np.array([df['py'], df['px']]).T - np.array([image_size[0]/2, image_size[1]/2])
    transformed_all = transforms.to_polar(transforms.linear_transform(platescale_fit, all_star_plate))
    
    # match nearest neighbours
    candidate_stars = np.zeros((stardata.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(stardata[:, 1])
    candidate_stars[:, 1] = np.degrees(stardata[:, 0])
    
    plt.scatter(transformed_all[:, 1], transformed_all[:, 0])
    plt.scatter(candidate_stars[:, 1], candidate_stars[:, 0])
    for i in range(stardata.shape[0]):
        plt.gca().annotate(f'mag={stardata[i, 5]:.2f}', (np.degrees(stardata[i, 0]), np.degrees(stardata[i, 1])), color='black', fontsize=5)
    plt.show()
    
    # find nearest two catalogue stars to each observed star
    neigh = NearestNeighbors(n_neighbors=2)

    neigh.fit(candidate_stars)
    distances, indices = neigh.kneighbors(transformed_all)
    #print(indices)
    #print(distances)

    # find nearest observed star to each catalogue star
    neigh_bar = NearestNeighbors(n_neighbors=1)

    neigh_bar.fit(transformed_all)
    distances_bar, indices_bar = neigh_bar.kneighbors(candidate_stars)

    match_threshhold = options['rough_match_threshhold'] # in degrees
    confusion_ratio = 2 # closest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio) # note: this distance metric is not perfect (doesn't take into account meridian etc.)
    keep = np.logical_and(keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0])) # is the nearest-neighbour relation reflexive? [this eliminates 1-to-many matching]
    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]
    if options['flag_display2']:
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

    return stardata, plate2

# from tetra
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
    cata_data = database_cache.open_catalogue('TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz')
    path_data = 'D:/output4/CENTROID_OUTPUT20240229002931/data.zip' # zwo 3 zd 30
    #path_data = 'D:\output4\CENTROID_OUTPUT20240303034855/data.zip' # eclipse (Don)
    #path_data = 'D:\output4\CENTROID_OUTPUT20240303040025/data.zip' # eclipse (Don) right
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310015116/data.zip' # eclipse (Berry)
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310020236/data.zip' # ZWO 1
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310194735/data.zip' # moontest 1
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310195034/data.zip' # moontest 3
    #path_data = 'E:\extra data\data.zip' # another moon test
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310200107\data.zip' # zwo 3 zd 75
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('data/results.txt'))
    df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    triangles = cata_data['triangles'] # (n x T x 2 array) - radius ratio and angular seperation for each triangle (note: T = N(N-1)/2)
    anchors = cata_data['anchors'] # vector rep of each "anchor" star
    pattern_data = cata_data['pattern_data'] # (n x N x 5 array) of (dtheta, phi, star_vector) for each neighbour star
    pattern_ind = cata_data['pattern_ind'] # n x N array of integer : the indices of neighbouring stars
    kd_tree = KDTree(triangles.reshape((-1, 2)), boxsize=[9999999, np.pi*2]) # use a 2-pi periodic condition for polar angle (and basically infinity for ratio)
    return kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data
    
def main():
    t0 = time.perf_counter(), time.process_time()
    kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data = load()
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    print('loaded') 
    vectors = np.c_[df['px'], df['py']] - np.array([meta_data['img_shape'][1], meta_data['img_shape'][0]]) / 2
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
            if ratio > 1:
                ratio = 1/ratio
                dphi = -dphi
                phi1, phi2 = phi2, phi1
                r1, r2 = r2, r1
                v1, v2 = v2, v1
            dphi = dphi % (2 * np.pi)
            cand = kd_tree.query_ball_point([ratio, dphi], TOLERANCE)
            ind = np.array(cand) // triangles.shape[1]
            array_vect = np.c_[v0, v1 + v0, v2 + v0]
            for ind_, cand_ in zip(ind, cand):              
                #matches[i][ind_].append((cand_, r1, phi1, array_vect))
                match_cand.append(cand_)
                match_data.append([r1, phi1])
                match_vect.append(array_vect.T)
                match_info.append((i,j,k))
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
    return scale, roll, center_vect, match_info, triangle_info, vectors, df, meta_data, target
            
if __name__ == '__main__':
    #cProfile.run('main()')
    options = {'flag_display':False, 'rough_match_threshhold':0.01, 'flag_display2':1}
    t00 = time.perf_counter(), time.process_time()
    scale, roll, center_vect, match_info, triangle_info, vectors, df, meta_data, target_vectors = main()
    image_size = meta_data['img_shape']
    all_star_plate = np.array([df['py'], df['px']]).T - np.array([image_size[0]/2, image_size[1]/2])
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
                print(len(non_redundant), [match_info[_] for _ in non_redundant])
                print(counts[i], radec, scale[el], roll[el], match_info[el])
                print(matchset)
                if options['flag_display']:
                    # show platesolve
                    plt.scatter(vectors[:, 0], vectors[:, 1])
                    for t in non_redundant:
                        tri = match_info[t]
                        v = np.array([vectors[_] for _ in tri]+[vectors[tri[0]]])
                        plt.plot(v[:, 0], v[:, 1], color='red')
                    plt.title(f"{len(non_redundant)} triangles matched\nplatescale={np.degrees(scale[el])*3600:.4f} arcsec/pixel\nra={radec[0][1]:.4f}, dec={radec[0][0]:.4f}")
                    plt.show()
                plate = (np.degrees(scale[el]), radec[0][1], radec[0][0], np.degrees(roll[el])+90) # this plus 90 is very weird and probably is need because of a coordinate bug
                print('scale/degrees, ra, dec, roll', plate)
                
                ivects = transforms.icoord_to_vector(np.array([all_star_plate[_] for _ in matchset])*scale[el])
                catvects = np.array([_ for _ in matchset.values()])
                rotation_matrix = _find_rotation_matrix(ivects, catvects)
                acc_ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                                   rotation_matrix[0, 0])) % 360
                acc_dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                                    np.linalg.norm(rotation_matrix[1:3, 2])))
                acc_roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                                     rotation_matrix[2, 2])) % 360
                acc_roll = (acc_roll + 180) % 360 # ???
                print('accurate ra dec roll', acc_ra, acc_dec, acc_roll)
                platescale = (np.degrees(scale[el]), acc_ra, acc_dec, acc_roll+180) # ???
                stardata, plate2 = match_centroids(df, np.radians(platescale), meta_data['img_shape'], options)
                
                if stardata.shape[0] >= NSTARTHRESHOLD:
                    print(f"MATCH ACCEPTED (nstars matched = {stardata.shape[0]})")
                else:
                    print(f"match rejected (nstars matched = {stardata.shape[0]})")
                

    print(f'npairs = {len(candidate_pairs)}')
    t4 = time.perf_counter(), time.process_time()
    print(f" Real time tree operations: {t4[0] - t2[0]:.2f} seconds")
    tff = time.perf_counter(), time.process_time()
    print(f" TOTAL TIME: {tff[0] - t00[0]:.2f} seconds")
    
