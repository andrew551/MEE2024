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

'''
PARAMETERS
'''
f = 8 # how many anchor stars to check
g = 8 # how many neighbour to check
TOLERANCE = 0.01 # tolerance for triangle matching
SCALE_TOL = 0.005 # 1 % scale tolerance
ROLL_TOL = 0.03 # radians
given_scale = 1.0122130319377387e-05
TOLERANCE_RADEC = 0.02 # degrees


# what roll and platescale does the triangle imply?
#a, b are tuples containing three values
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
    roll = (a[2] - s1[1]) % (np.pi*2)
    #print(a)
    x = np.c_[a[3][:, 0], a[3][:, 0] + a[3][:, 1], a[3][:, 0] + a[3][:, 2]].T
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

def load():
    cata_data = np.load('pattern_data.npz')
    #path_data = 'D:/output4/CENTROID_OUTPUT20240229002931/data.zip' # zwo 3 zd 30
    path_data = 'D:\output4\CENTROID_OUTPUT20240303034855/data.zip' # eclipse
    #path_data = 'D:\output4\CENTROID_OUTPUT20240303040025/data.zip' # eclipse right
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('data/results.txt'))
    df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    triangles = cata_data['triangles']
    anchors = cata_data['anchors']
    pattern_data = cata_data['pattern_data']
    pattern_ind = cata_data['pattern_ind']
    kd_tree = KDTree(triangles.reshape((-1, 2)))
    return kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data

def find_matching_triangles(matches, triangles, pattern_data, anchors, given_scale):
    for i, matches_i in enumerate(matches):
        for k, v in matches_i.items():
            # plan: collect all implied (scale, roll), and then find consistent ones
            entry = anchors[k, :]
            #print(entry)
            coord = transforms.to_polar(entry)
            #print(coord)
            ra = coord[0,1]
            dec = coord[0,0]
            if 0 and given_scale:
                for s1 in scale_roll:
                    if np.abs(s1[0] / given_scale - 1) < SCALE_TOL:
                        print('MATCH (with scale)', i, s1, ra, dec)
            elif len(v) >= 2:
                #print(i, k, v, ra, dec)
                scale_roll = [get_scale_and_roll(triangles, pattern_data, anchors, a) for a in v]

                for s1, s2 in itertools.combinations(scale_roll, r=2):
                    # check if the scale / roll for the two triangles is consistent
                    if (given_scale is None or np.abs(s1[0] / given_scale - 1) < SCALE_TOL) and np.abs(s1[0]/s2[0]-1) < SCALE_TOL and np.abs(s1[1] - s2[1]) < ROLL_TOL \
                    and np.abs(s1[2]-s2[2]) < TOLERANCE_RADEC and np.abs(s1[3]-s2[3]) < TOLERANCE_RADEC:
                        print('MATCH', i, s1, s2, ra, dec)

def main():
    
    kd_tree, anchors, pattern_ind, pattern_data, triangles, df, meta_data = load()
    
    #print(triangles[31633, :, :])
    #print(pattern_data[31633, :])
    #print(pattern_ind[31633, :])
    print('loaded') 

    vectors = np.c_[df['px'], df['py']] - np.array([meta_data['img_shape'][1], meta_data['img_shape'][0]]) / 2
    #plt.scatter(vectors[:, 0], vectors[:, 1])
    #plt.show()
    print('mean:', np.mean(vectors, axis=0))
    matches = [defaultdict(list) for _ in range(f)]
    for i in range(f):
        for n, (j, k) in enumerate(itertools.combinations(range(g), 2)):
            if j == i or k == i:
                continue
            
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

            for ind_, cand_ in zip(ind, cand):
                matches[i][ind_].append((cand_, r1, phi1, np.c_[v0, v1, v2]))
    find_matching_triangles(matches, triangles, pattern_data, anchors, given_scale)
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
    cProfile.run('main()')
