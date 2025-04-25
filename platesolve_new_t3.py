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
import database_cache
import tqdm

def generate():
    #dbs = database_cache.open_catalogue(resource_path("resources/compressed_tycho2024epoch.npz"))
    dbs = database_cache.open_catalogue("gaia_offline")
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
    e = 18 # was 18
    theta_pat = np.radians(1.7)

    vectors = dbs.star_table[:d, 2:5].astype(np.float32)
    print(f'keeping down to mag {dbs.star_table[a, 5]}')
    kd_tree1 = KDTree(vectors)

    kept = np.zeros(d, dtype=bool) # "anchor stars"
    kept2 = np.zeros(d, dtype=bool) # "leg stars"
    '''
    step 1: find set of "anchor" stars
        (1.1) #a brightest stars (exclude double stars)
        (1.2) any of the #b next-brightest stars which are further than theta_sep
        away from any brighter star than themselves
    '''
    
    for i in tqdm.tqdm(range(d), desc='collecting stars'):
        neighbours = kd_tree1.query_ball_point(vectors[i], theta_sep)    
        neighbours2 = kd_tree1.query_ball_point(vectors[i], theta_double_star)

        if np.any(kept2[neighbours2]):
            continue # double star skip
        kept2[i] = 1
        in_gap = not np.any(kept[neighbours])
        if (in_gap and i < a + b) or i < a:
            kept[i] = 1
        '''    
        if not np.any(kept[neighbours]): # dim stars, in "holes"
            if i < a+b:
                kept[i] = 1
            kept2[i] = 1
        elif not np.any(kept[neighbours2]): # bright stars, not double
            if i < a:
                kept[i] = 1
            kept2[i] = 1
        '''
    print(f'note kept {np.sum(kept[:a])} of first {a} stars as anchors')
    print(f'note kept {np.sum(kept[:a+b])} of first {a+b} stars as anchors')
    print(f'note kept {np.sum(kept2)} of first {d} stars as legs')
    
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
    kd_tree2 = KDTree(vectors2) # KDTree of all "leg stars"
    temp_dict1 = dict(enumerate(kept_vectors_ind))
    temp_dict2 = dict(map(reversed, enumerate(kept_vectors_ind2)))
    cumsum = np.cumsum(np.logical_not(kept2).astype(int)) # use this array to correct for mismatch between anchor indices and leg indices
    pattern_ind = np.ones((nkept, c+e), dtype=int)*-1
    pattern_data = np.zeros((nkept, c+e, 5), dtype=np.float32) # store r and phi of patterns
    z = np.array([0, 0, 1])
    for i in tqdm.tqdm(range(nkept), desc='finding pattern points'):
        neighbours = kd_tree2.query_ball_point(vectors_kept[i], theta_pat)
        #neighbours.remove(i) # don't match self
        ind = kept_vectors_ind[i]
        neighbours.remove(ind - cumsum[ind]) # don't match self
        #if i == 1661:#31633:
        #    print(neighbours)
        neighbours = [idx for idx in neighbours if idx > ind - cumsum[ind]]
        neighbours.sort()
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
    
    if 0:
        print('starting to find patterns...')
        triangles = np.zeros((nkept, (c+e)*(c+e-1)//2, 2), dtype=np.float32)

        for i in tqdm.tqdm(range(nkept)):
            for n, (j, k) in enumerate(itertools.combinations(range(c+e), 2)):
                ratio = pattern_data[i, k, 0] / pattern_data[i, j, 0]
                dphi  = pattern_data[i, k, 1] - pattern_data[i, j, 1]
                if ratio > 1:
                    ratio = 1/ratio
                    dphi = -dphi
                dphi = dphi % (2 * np.pi)
                triangles[i, n, 0] = ratio
                triangles[i, n, 1] = dphi

    '''
    step 3: new version

    compute triangle side lengths a, b, c
    compute triangle vector in rep space

    '''

    
    print('starting to find patterns...')


    '''
    triangles = np.zeros((nkept, (c+e)*(c+e-1)//2, 3), dtype=np.float32)
    for i in tqdm.tqdm(range(nkept)):
        for n, (j, k) in enumerate(itertools.combinations(range(c+e), 2)):

            vi = vectors_kept[i]
            vj = pattern_data[i, j, 2:5]
            vk = pattern_data[i, k, 2:5]
            r1 = vi-vj
            r2 = vj-vk
            r3 = vk-vi
            orientation = -np.dot(np.cross(r1, r2), vj)
            
            r1 = (r1[0]**2+r1[1]**2+r1[2]**2)**0.5
            r2 = (r2[0]**2+r2[1]**2+r2[2]**2)**0.5
            r3 = (r3[0]**2+r3[1]**2+r3[2]**2)**0.5
            # handle orientation
            if orientation < 0:
                r1, r2 = r2, r1
            # cylic permute to make r1 the largest of (r1, r2, r3)
            if r2 > r1 and r2 > r3:
                r1, r2, r3 = r2, r3, r1
            elif r3 > r1 and r3 > r2:
                r1, r2, r3 = r3, r1, r2
            
            if (r1 + r2 < r3) or (r1 + r3 < r2) or (r2 + r3 < r1):
                raise ValueError("The given side lengths do not form a valid triangle.")
            
            s = 0.5 * (r1 + r2 + r3)
            area = (s * (s - r1) * (s - r2) * (s - r3))**0.5
    
            # Normalizing denominator (scale invariant quantity)
            denom = r1**2 + r2**2 + r3**2
    
            triangles[i, n, 0] = 2*(r1**2 - r2**2) / denom
            triangles[i, n, 1] = 2*(r1**2 + r2**2 - 2 * r3**2) / (3**0.5 * denom)
            triangles[i, n, 2] = (4 * 3**0.5 * area) / denom
    '''
    def compute_triangles(vectors_kept, pattern_data):
        nkept = vectors_kept.shape[0]
        num_combinations = (c + e) * (c + e - 1) // 2
        triangles = np.zeros((nkept, num_combinations, 3), dtype=np.float32)
        
        indices = np.array(list(itertools.combinations(range(c+e), 2)))  # (num_combinations, 2)
        j_indices, k_indices = indices[:, 0], indices[:, 1]
        
        vi = vectors_kept[:, np.newaxis, :]  # (nkept, 1, 3)
        vj = pattern_data[:, j_indices, 2:5]  # (nkept, num_combinations, 3)
        vk = pattern_data[:, k_indices, 2:5]  # (nkept, num_combinations, 3)
        
        r1 = vi - vj
        r2 = vj - vk
        r3 = vk - vi
        
        cross_r1_r2 = np.cross(r1, r2)  # (nkept, num_combinations, 3)
        orientation = -np.einsum('ijk,ijk->ij', cross_r1_r2, vj)  # Dot product along last axis
        
        r1_norm = np.linalg.norm(r1, axis=-1)
        r2_norm = np.linalg.norm(r2, axis=-1)
        r3_norm = np.linalg.norm(r3, axis=-1)
        
        # Handle orientation
        swap_mask = orientation < 0
        r1_norm[swap_mask], r2_norm[swap_mask] = r2_norm[swap_mask], r1_norm[swap_mask]
        
        # Cyclic permutation to make r1 the largest
        max_r = np.maximum(np.maximum(r1_norm, r2_norm), r3_norm)
        
        swap_r2_mask = r2_norm == max_r
        swap_r3_mask = r3_norm == max_r

        # TODO insert permutation tracking = p(swap_mask, swap_r2, swap_r3)
        # 000, 010, 001, 100, 110, 101 -> 0 to 5
        p0 = swap_mask.astype(np.uint8)  # Convert boolean to 0/1
        p1 = swap_r2_mask.astype(np.uint8) << 1  # Shift left by 1
        p2 = swap_r3_mask.astype(np.uint8) << 2  # Shift left by 2

        permutation_data = (p0 | p1 | p2)
        
        r1_final = np.where(swap_r2_mask, r2_norm, np.where(swap_r3_mask, r3_norm, r1_norm))
        r2_final = np.where(swap_r2_mask, r3_norm, np.where(swap_r3_mask, r1_norm, r2_norm))
        r3_final = np.where(swap_r2_mask, r1_norm, np.where(swap_r3_mask, r2_norm, r3_norm))
        
        # Triangle inequality check
        invalid_mask = (r1_final + r2_final < r3_final) | (r1_final + r3_final < r2_final) | (r2_final + r3_final < r1_final)
        if np.any(invalid_mask):
            raise ValueError("The given side lengths do not form a valid triangle.")
        
        # Compute area using Heron's formula
        s = 0.5 * (r1_final + r2_final + r3_final)
        area = np.sqrt(s * (s - r1_final) * (s - r2_final) * (s - r3_final))
        print(np.nonzero(np.isnan(area)))
        print(r1_final[np.nonzero(np.isnan(area))], r2_final[np.nonzero(np.isnan(area))],r3_final[np.nonzero(np.isnan(area))]) 
        # Normalizing denominator (scale-invariant quantity)
        denom = r1_final**2 + r2_final**2 + r3_final**2
        
        triangles[:, :, 0] = 3**0.5 * (r1_final**2 - r2_final**2) / denom
        triangles[:, :, 1] = (r1_final**2 + r2_final**2 - 2 * r3_final**2) / denom
        triangles[:, :, 2] = (4 * 3**0.5 * area) / denom
        
        return triangles, permutation_data

    triangles, permutation_data = compute_triangles(vectors_kept, pattern_data)
            
    Path("TripleTrianglePlatesolveDatabase3").mkdir(exist_ok=True)
    np.savez_compressed("TripleTrianglePlatesolveDatabase3/TripleTriangle_pattern_data3.npz", anchors = vectors_kept, pattern_ind=pattern_ind, pattern_data=pattern_data, triangles=triangles, permutation_data=permutation_data)
    print(f"completed generating triangle database -- {triangles.size//3} triangles saved")        
            
if __name__ == '__main__':
    generate()
