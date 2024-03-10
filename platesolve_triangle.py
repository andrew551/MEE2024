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
'''
PARAMETERS
'''
f = 5 # how many anchor stars to check
g = 8 # how many neighbour to check
TOLERANCE = 0.01 # tolerance for triangle matching
#SCALE_TOL = 0.005 # 1 % scale tolerance
#ROLL_TOL = 0.03 # radians
#given_scale = 1.0122130319377387e-05
#TOLERANCE_RADEC = 0.005 # degrees (== 7.2 arcsec)


# what roll and platescale does the triangle imply?
#a is a tuple containing four elements
# 0th element: matched triangle index
# 1st, 2nd: observed r (pixels) and phi
# 3d element (triangle: (v0, v1-v0, v2-v0))
def get_scale_and_roll(triangles, pattern_data, anchors, a):
    n = a[0] // triangles.shape[1]
    rem = a[0] % triangles.shape[1]
    t1 = triangles.reshape((-1, 2))[a[0]]
    pairs = list(itertools.combinations(range(pattern_data.shape[1]), r=2))
    s1 = pattern_data[n, pairs[rem][0]]
    s2 = pattern_data[n, pairs[rem][1]]
    if s1[0] < s2[0]:
        s1, s2 = s2, s1 # larger length first convention
    scale = s1[0] / a[1]   # in radians per pixel 
    roll = (a[2] - s1[1]) % (np.pi*2) # - correction?
    #print(a)
    x = a[3].T
    #print(x.shape, x)
    as_3vect = transforms.icoord_to_vector(x*scale).T
    #print(as_3vect)
    target = np.c_[anchors[n], s1[2:5], s2[2:5]]
    rmatrix = target @ np.linalg.inv(as_3vect)
    center_vect = rmatrix @ np.array([1, 0, 0]).T
    pol = transforms.to_polar(center_vect)
    dec = pol[0][0]
    ra = pol[0][1]
    #print(target)
    #print(ra, dec)
    #end
    return scale, roll, ra, dec

'''
vectorised version of get_scale_and_roll
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
    scale = sdat[0, :, 0] / match_data[:, 0]
    scaled = np.einsum('ijk,i -> ijk', match_vect, scale)
    as_3vect = transforms.icoord_to_vector(scaled).swapaxes(1, 2)
    target = np.stack([anchors[n], sdat[0, :, 2:5], sdat[1, :, 2:5]], axis=2)
    rmatrix = np.einsum('...ij,...jk -> ...ik', target, np.linalg.inv(as_3vect))
    center_vect = rmatrix[:, :, 0]
    roll = np.arctan2(rmatrix[:, 1, 2], rmatrix[:, 2, 2]) % (2*np.pi)
    return scale, roll, center_vect, rmatrix

def load():
    cata_data = np.load('pattern_data.npz')
    #path_data = 'D:/output4/CENTROID_OUTPUT20240229002931/data.zip' # zwo 3 zd 30
    #path_data = 'D:\output4\CENTROID_OUTPUT20240303034855/data.zip' # eclipse (Don)
    path_data = 'D:\output4\CENTROID_OUTPUT20240303040025/data.zip' # eclipse (Don) right
    #path_data = 'D:\output4\CENTROID_OUTPUT20240310015116/data.zip' # eclipse (Berry)
    path_data = 'D:\output4\CENTROID_OUTPUT20240310020236/data.zip' # ZWO 1 
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('data/results.txt'))
    df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    triangles = cata_data['triangles'] # (n x T x 2 array) - radius ratio and angular seperation for each triangle (note: T = N(N-1)/2)
    anchors = cata_data['anchors'] # vector rep of each "anchor" star
    pattern_data = cata_data['pattern_data'] # (n x N x 5 array) of (dtheta, phi, star_vector) for each neighbour star
    pattern_ind = cata_data['pattern_ind'] # n x N array of integer : the indices of neighbouring stars
    kd_tree = KDTree(triangles.reshape((-1, 2)))
    return kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data

def find_matching_triangles(matches, triangles, pattern_data, anchors, given_scale):
    match_cand = [] # index of triangle matches
    match_data = [] # [r, phi] the longer side and polar angle of the matched triangles
    match_vect = [] # [v1, v2, v3]
                    # v0: coordinate of the center star in 2D-pixel space,
                    # v1, v2: vectors from center star to the two neighbouring stars
    seg_lens = []
    for i, matches_i in enumerate(matches):
        print(len(matches_i))
        for k, v in matches_i.items():
            # plan: collect all implied (scale, roll), and then find consistent ones
            #entry = anchors[k, :]
            #print(entry)
            #coord = transforms.to_polar(entry)
            #print(coord)
            #ra = coord[0,1]
            #dec = coord[0,0]
            if 0 and given_scale:
                for s1 in scale_roll:
                    if np.abs(s1[0] / given_scale - 1) < SCALE_TOL:
                        print('MATCH (with scale)', i, s1, ra, dec)
            elif len(v) >= 2:
                #print(i, k, v, ra, dec)
                seg_lens.append(len(v))
                for a in v:
                    match_cand.append(a[0])
                    match_data.append([a[1], a[2]])
                    match_vect.append(a[3].T)
                '''
                scale_roll = [get_scale_and_roll(triangles, pattern_data, anchors, a) for a in v]
                for s1, s2 in itertools.combinations(scale_roll, r=2):
                    # check if the scale / roll for the two triangles is consistent
                    if (given_scale is None or np.abs(s1[0] / given_scale - 1) < SCALE_TOL) and np.abs(s1[0]/s2[0]-1) < SCALE_TOL and np.abs(s1[1] - s2[1]) < ROLL_TOL \
                    and np.abs(s1[2]-s2[2]) < TOLERANCE_RADEC and np.abs(s1[3]-s2[3]) < TOLERANCE_RADEC:
                        print('MATCH', i, s1, s2, ra, dec)
                '''
    match_cand = np.array(match_cand)
    match_data = np.array(match_data)
    match_vect = np.array(match_vect)
    scale, roll, center_vect, rmatrix = compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)
    plate = np.c_[scale, roll, center_vect]
    seg_inds = np.cumsum([0]+seg_lens)
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    for i in range(len(seg_inds) - 1):
        ind = np.arange(seg_inds[i], seg_inds[i+1])
        plate_i = plate[ind, :]

        plate_tup = [tuple(x) for x in plate_i]

        for s1, s2 in itertools.combinations(plate_tup, r=2):
        
            if (given_scale is None or np.abs(s1[0] / given_scale - 1) < SCALE_TOL) and np.abs(s1[0]/s2[0]-1) < SCALE_TOL and np.abs(s1[1] - s2[1]) < ROLL_TOL \
                        and np.linalg.norm(np.array(s1[2:5]) - np.array(s2[2:5])) < TOLERANCE_RADEC:
                            radec = transforms.to_polar(np.array(s1[2:5]))
                            print('MATCH', s1, s2, radec, match_vect[seg_inds[i]: seg_inds[i+1]])
                            print(match_cand[seg_inds[i]: seg_inds[i+1]])
                            print(match_data[seg_inds[i]: seg_inds[i+1]])
                            #print(pattern_data[match_cand[seg_inds[i]] // triangles.shape[1], match_cand[seg_inds[i]] % triangles.shape[1]])
                            #print(pattern_data[match_cand[seg_inds[i]+1] // triangles.shape[1], match_cand[seg_inds[i]+1] % triangles.shape[1]])
                            n = match_cand[seg_inds[i]] // triangles.shape[1]
                            print(n)
                            rem = match_cand[seg_inds[i]] % triangles.shape[1]
                            print(rem)
                            print(pattern_data.shape, pairs.shape)
                            print(pairs[rem])
                            s1 = pattern_data[n, pairs[rem][0]]
                            s2 = pattern_data[n, pairs[rem][1]]
                            n = match_cand[seg_inds[i]+1] // triangles.shape[1]
                            rem2 = match_cand[seg_inds[i]+1] % triangles.shape[1]
                            print(pairs[rem2])
                            s3 = pattern_data[n, pairs[rem2][0]]
                            s4 = pattern_data[n, pairs[rem2][1]]
                            print(rem, rem2)
                            print(s1)
                            print(s2)
                            print(s3)
                            print(s4)
                            # TODO: fix double star problem
                            # TODO: for each match, try to fit other centroids using the platescale
    
#and np.abs(s1[2]-s2[2]) < TOLERANCE_RADEC and np.abs(s1[3]-s2[3]) < TOLERANCE_RADEC: 

    
def main():
    t0 = time.perf_counter(), time.process_time()
    kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data = load()
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    #print(triangles[31633, :, :])
    #print(pattern_data[31633, :])
    #print(pattern_ind[31633, :])
    print('loaded') 
    vectors = np.c_[df['px'], df['py']] - np.array([meta_data['img_shape'][1], meta_data['img_shape'][0]]) / 2
    #plt.scatter(vectors[:, 0], vectors[:, 1])
    #plt.show()
    print('mean:', np.mean(vectors, axis=0))
    matches = [defaultdict(list) for _ in range(f)]
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
            if j == i or k == i:
                continue
            #if i > j or i > k:
            #    continue
            
            v0 = vectors[i, :]
            v1 = vectors[j, :] - v0
            v2 = vectors[k, :] - v0
            #print(v0, vectors[j, :], vectors[k, :])
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
            # convention: the 
            dphi = dphi % (2 * np.pi)
            if dphi > np.pi:
                dphi -= np.pi * 2
            #print(ratio, dphi)
            cand = kd_tree.query_ball_point([ratio, dphi], TOLERANCE)
            #print(len(cand))
            ind = np.array(cand) // triangles.shape[1]
            array_vect = np.c_[v0, v1 + v0, v2 + v0]
            for ind_, cand_ in zip(ind, cand):
                
                matches[i][ind_].append((cand_, r1, phi1, array_vect))
                match_cand.append(cand_)
                match_data.append([r1, phi1])
                match_vect.append(array_vect.T)
                match_info.append((i,j,k))
                rem = cand_ % triangles.shape[1]
                triangle_info.append(pairs[rem])
    t2 = time.perf_counter(), time.process_time()
    print(f" Real time prepare: {t2[0] - t1[0]:.2f} seconds")
    #print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
    match_cand = np.array(match_cand)
    match_data = np.array(match_data)
    match_vect = np.array(match_vect)
    
    #find_matching_triangles(matches, triangles, pattern_data, anchors, given_scale)
    t3 = time.perf_counter(), time.process_time()
    print(f" Real time match: {t3[0] - t2[0]:.2f} seconds")
    scale, roll, center_vect, matrix = compute_platescale(triangles, pattern_data, anchors, match_cand, match_data, match_vect)
    t4 = time.perf_counter(), time.process_time()
    print(f" Real time platescale compute: {t4[0] - t3[0]:.2f} seconds")
    return scale, roll, center_vect, match_info, triangle_info
    
    #
            #if (np.abs(ra - 142.9898) < 0.01 and np.abs(dec - 9.715665) < 0.01):
            #    print(i, k , v)
    '''
            if (np.abs(ra - 34.5154) < 0.01 and np.abs(dec - 14.4667) < 0.01) \
        or (np.abs(ra - 35.4465) < 0.01 and np.abs(dec - 15.51918) < 0.01) \
        or (np.abs(ra - 35.50254) < 0.01 and np.abs(dec - 15.9911) < 0.01): 
                #print('       ', entry, transforms.to_polar(entry))
                print(i, k, v)
            '''
            
if __name__ == '__main__':
    #cProfile.run('main()')
    t00 = time.perf_counter(), time.process_time()
    scale, roll, center_vect, match_info, triangle_info = main()
    t2 = time.perf_counter(), time.process_time()
    ra_des = np.radians(129)
    dec_des = np.radians(6.84)
    #ra_des = np.radians(143)
    #dec_des = np.radians(9.25)
    '''
    des = np.array([np.cos(ra_des)*np.cos(dec_des), np.sin(ra_des)*np.cos(dec_des), np.sin(dec_des)])
    for i, (sc, rll, cvect, minfo, tinfo) in enumerate(zip(scale, roll, center_vect, match_info, triangle_info)):
        if np.abs((sc - given_scale) / given_scale) < 0.05 and np.linalg.norm(des-cvect) < 0.4:
            print (i, ':', sc, rll, cvect, transforms.to_polar(cvect), (sc - given_scale) / given_scale, minfo, tinfo)
    '''
    log_scale = np.log(scale)
    TOL_CENT = np.radians(0.02) # 0.02 degrees
    TOL_ROLL = np.radians(0.02) # 0.02 degrees
    log_TOL_SCALE = 0.005       # 1 part in 200

    vector_plates = np.c_[log_scale / log_TOL_SCALE, roll / TOL_ROLL, center_vect / TOL_CENT]

    
    tree_matches = KDTree(vector_plates)


    
    t3 = time.perf_counter(), time.process_time()
    candidate_pairs = tree_matches.query_pairs(1) # efficiently find all pairs of agreeing triangles
    N = vector_plates.shape[0]
    graph = csr_matrix(([1 for _ in candidate_pairs], ([x[0] for x in candidate_pairs], [x[1] for x in candidate_pairs])), shape=(N, N))
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)

    counts = dict(zip(unique, counts))
    #counts = dict(zip(np.unique(labels, return_counts=True)))
    for i in range(n_components):
        if counts[i] >= 3:
            indices = np.nonzero(labels==i)[0]
            el = indices[0]
            radec = transforms.to_polar(center_vect[el])
            #print(indices)
            #print([match_info[_] for _ in indices])
            # remove redundant triangles
            seen = set()
            non_redundant = []
            for ind in indices:
                if match_info[ind] in seen:
                    continue
                seen.update(list(itertools.permutations(match_info[ind])))
                non_redundant.append(ind)
            if len(non_redundant) >= 3:
                print(len(non_redundant), [match_info[_] for _ in non_redundant])
                print(counts[i], radec, scale[el], roll[el], match_info[el])

    print(f'npairs = {len(candidate_pairs)}')
    t4 = time.perf_counter(), time.process_time()
    print(f" Real time tree operations: {t4[0] - t2[0]:.2f} seconds")
    tff = time.perf_counter(), time.process_time()
    print(f" TOTAL TIME: {tff[0] - t00[0]:.2f} seconds")
    '''
    seen = set()

    for i in range(vector_plates.shape[0]):
        if i in seen:
            continue
        neighbours = tree_matches.query_ball_point(vector_plates[i], 1)
        tovisit = deque(neighbours)
        visited = set()
        # do a bfs search for neighbours of neighbours and so on
        while tovisit:
            x = tovisit.pop()
            visited.add(x)
            neighbours_next = tree_matches.query_ball_point(vector_plates[x], 1)
            for y in neighbours_next:
                if not y in visited:
                    tovisit.append(y)
        seen.update(visited)
        if len(visited) >= 4:
            el = list(visited)[0]
            radec = transforms.to_polar(center_vect[el])
            print(radec, scale[el], roll[el], match_info[el])
            print(visited)
            print([match_info[_] for _ in visited])

    '''    
                
        
    '''
    candidate_pairs = tree_matches.query_pairs(1)

    for pair in candidate_pairs:
        if 1 or (np.abs(np.log(given_scale) - log_scale[pair[0]]) < 0.01):
            neigh = tree_matches.query_ball_point(vector_plates[pair[0]], 1)
            if len(neigh) >= 4:
                delta = vector_plates[pair[0]] - vector_plates[pair[1]]
                #if np.max(np.abs(delta)) < 1e-5:
                #    continue # delta == 0 -> trivial matching of the same triangle
                v = center_vect[pair[0]]
                radec = transforms.to_polar(v)
                print(len(neigh), radec, roll[pair[0]], delta, np.linalg.norm(delta), scale[pair[0]], match_info[pair[0]])
    '''
    #t3 = time.perf_counter(), time.process_time()
    #print(f" Real time match matches: {t3[0] - t2[0]:.2f} seconds")
