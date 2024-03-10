import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats
import scipy
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist
from sklearn.preprocessing import normalize
import itertools
import os
from database_lookup2 import database_searcher
from MEE2024util import resource_path
from pathlib import Path

if __name__ == '__main__':
    dbs = database_searcher(resource_path("resources/compressed_tycho2024epoch.npz"))
    print(dbs.star_table.shape)

    # parameters for step 1
    a = 80000
    b = 120000
    theta_sep = np.radians(0.65) # 0.65 degrees
    theta_double_star = np.radians(0.01) # 36 arcsec
    # parameters for step 2
    c = 0
    d = 700000
    if a+b >= d:
        raise Exception("weird choice for paramters a,b,d")
    e = 18
    theta_pat = np.radians(1.7)

    vectors = dbs.star_table[:d, 2:5].astype(np.float32)
    print(f'keeping down to mag {dbs.star_table[a, 5]}')
    kd_tree1 = KDTree(vectors)

    kept = np.zeros(d, dtype=bool)
    kept2 = np.zeros(d, dtype=bool)
    '''
    step 1: find set of "anchor" stars
        (1.1) #a brightest stars (exclude double stars)
        (1.2) any of the #b next-brightest stars which are further than theta_sep
        away from any brighter star than themselves
    '''
    
    for i in range(d):
        neighbours = kd_tree1.query_ball_point(vectors[i], theta_sep)    
        neighbours2 = kd_tree1.query_ball_point(vectors[i], theta_double_star)
            
        if not np.any(kept[neighbours]):
            if i < a+b:
                kept[i] = 1
            kept2[i] = 1
        elif not np.any(kept[neighbours2]):
            if i < a:
                kept[i] = 1
            kept2[i] = 1
    print(f'note kept {np.sum(kept[:a])} of first {a} stars')
    print(f'note kept {np.sum(kept[:a+b])} of first {a+b} stars')
    print(f'note kept {np.sum(kept2)} of first {d} stars')
    
    vectors_kept = vectors[kept, :]
    kept_vectors_ind = np.nonzero(kept)[0] # np.nonzero returns tuples
    kept_vectors_ind2 = np.nonzero(kept)[0] # np.nonzero returns tuples

    
    
    nkept = vectors_kept.shape[0]

    for i in range(a+b):
        if (np.abs(np.degrees(dbs.star_table[i, 0]) - 34.5154) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 14.4667) < 0.01) \
        or (np.abs(np.degrees(dbs.star_table[i, 0]) - 35.4465) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 15.51918) < 0.01) \
        or (np.abs(np.degrees(dbs.star_table[i, 0]) - 35.50254) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 15.9911) < 0.01) \
        or (np.abs(np.degrees(dbs.star_table[i, 0]) - 142.989) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 9.715665) < 0.01) \
        or (np.abs(np.degrees(dbs.star_table[i, 0]) - 143.3175) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 9.1684) < 0.01) \
        or (np.abs(np.degrees(dbs.star_table[i, 0]) - 143.7188) < 0.01 and np.abs(np.degrees(dbs.star_table[i, 1]) - 9.6767) < 0.01) \
        :           
            print(i, dbs.star_table[i, 5], kept[i])
            
    '''
    step 2: (2.1) find the #c closest stars (within theta_pat) (among the #d brightest)
            to each anchor star
            (2.2) find the #e brightest stars (within theta_pat) of each anchor
            anchor star which have not yet been accounted for in (2.1)
            (2.3) if insufficient stars found, make a note of it
                  statistically, this should be improbable for a good
                  choice of parameters
    '''
    vectors2 = vectors[kept2, :]
    kd_tree2 = KDTree(vectors2)
    temp_dict1 = dict(enumerate(kept_vectors_ind))
    temp_dict2 = dict(map(reversed, enumerate(kept_vectors_ind2)))
    cumsum = np.cumsum(np.logical_not(kept2).astype(int))
    pattern_ind = np.ones((nkept, c+e), dtype=int)*-1
    pattern_data = np.zeros((nkept, c+e, 5), dtype=np.float32) # store r and phi of patterns
    z = np.array([0, 0, 1])
    for i in range(nkept):
        neighbours = kd_tree2.query_ball_point(vectors_kept[i], theta_pat)
        #neighbours.remove(i) # don't match self
        ind = kept_vectors_ind[i]
        neighbours.remove(ind - cumsum[ind]) # don't match self
        if i == 1661:#31633:
            print(neighbours)
        if len(neighbours) < c+e:
            print(f'note: insufficient neighbours found on index {i}')
            raise Exception('edge case handling unimplemented!')
        # delta vectors
        delta = vectors2[neighbours] - vectors_kept[i]
        
        #dtheta
        dtheta = 2 * np.arcsin(.5 * np.linalg.norm(delta, axis=1))
        #dphi calculation
        tangent_vector_phi = np.cross(z, vectors_kept[i]) # spherical polar "phi_hat"
        tangent_vector_phi /= np.linalg.norm(tangent_vector_phi)
        tangent_vector_theta = np.cross(tangent_vector_phi, vectors_kept[i]) # spherical polar "theta_hat"
        tangent_vector_theta /= np.linalg.norm(tangent_vector_theta)
        x = np.dot(tangent_vector_theta, delta.T)
        y = np.dot(tangent_vector_phi, delta.T)
        phi = np.arctan2(y, x)

        # get the c nearest neighbours
        chosen_c = list(np.argsort(dtheta)[:c])
        chosen_c_ind = list(np.array(neighbours)[chosen_c])
        # get the e brightest unselected stars
        # note that this following line makes use of python set's order-preserving behaviour
        chosen_e = [x for x in np.argsort(neighbours) if x not in chosen_c_ind][:e]
        chosen_e_ind = list(np.array(neighbours)[chosen_e])              

        pattern_ind[i, :] = chosen_c_ind + chosen_e_ind
        pattern_data[i, :, 0] = dtheta[chosen_c+chosen_e]
        pattern_data[i, :, 1] = phi[chosen_c+chosen_e]
        pattern_data[i, :, 2:5] = vectors2[chosen_c_ind + chosen_e_ind] # technically this information is redundant, but it's convenient to have
    '''
    step 3: compute all (c+e)*(c+e-1)/2 desired triangles for each anchor star. One of the vertices
            of each pattern is the anchor star

    '''
    print('starting to find patterns...')
    triangles = np.zeros((nkept, (c+e)*(c+e-1)//2, 2), dtype=np.float32)

    for i in range(nkept):
        for n, (j, k) in enumerate(itertools.combinations(range(c+e), 2)):
            ratio = pattern_data[i, k, 0] / pattern_data[i, j, 0]
            dphi  = pattern_data[i, k, 1] - pattern_data[i, j, 1]
            if ratio > 1:
                ratio = 1/ratio
                dphi = -dphi
            dphi = dphi % (2 * np.pi)
            triangles[i, n, 0] = ratio
            triangles[i, n, 1] = dphi
    Path("TripleTrianglePlatesolveDatabase").mkdir(exist_ok=True)
    np.savez_compressed("TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz", anchors = vectors_kept, pattern_ind=pattern_ind, pattern_data=pattern_data, triangles=triangles)
    print(f"completed generating triangle database -- {triangles.size//2} triangles saved")        
            
