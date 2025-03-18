import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import zipfile
import itertools
from scipy.spatial import KDTree

'''
PARAMETERS (TODO: make controllable by options)
'''
f = 7 # how many anchor stars to check
g = 18 # how many neighbour to check
TOLERANCE_TRIANGLE = 0.001

dbase = np.load("TripleTrianglePlatesolveDatabase2/TripleTriangle_pattern_data2.npz")

triangles = dbase['triangles']
pattern_data = dbase['pattern_data']

kd_tree = KDTree(triangles.reshape(-1, 3))
print("built kd_tree")

print(triangles.shape)

print(np.linalg.norm(triangles[0, :, :], axis=1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(triangles[0, :, 0], triangles[0, :, 1], triangles[0, :, 2])
ax.scatter(triangles[1, :, 0], triangles[1, :, 1], triangles[1, :, 2])
ax.scatter(triangles[3, :, 0], triangles[3, :, 1], triangles[3, :, 2])
ax.scatter(triangles[4, :, 0], triangles[4, :, 1], triangles[4, :, 2])
ax.scatter(triangles[5, :, 0], triangles[5, :, 1], triangles[5, :, 2])

ax.set_aspect('equal')
plt.show()


def get_2Dtriang_rep(v1, v2, v3):
    '''
    v1, v2, v3: x-y coordinates of triangle as arrays
    '''
    r1, r2, r3 = v1-v2, v2-v3, v3-v1
    swap_flag = np.cross(r1, r2) > 0
    r1 = (r1[0]**2+r1[1]**2)**0.5
    r2 = (r2[0]**2+r2[1]**2)**0.5
    r3 = (r3[0]**2+r3[1]**2)**0.5

    if swap_flag:
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
    
    x = 3**0.5 * (r1**2 - r2**2) / denom
    y = (r1**2 + r2**2 - 2 * r3**2) / denom
    z = (4 * 3**0.5 * area) / denom
    return x, y, z

def match_image_triangles(centroids):
    pairs = np.array(list(itertools.combinations(range(pattern_data.shape[1]), r=2))) # helper array to convert index i -> pairs (j, k)
    vectors = np.c_[centroids[:, 1], centroids[:, 0]]
    reps = []
    for i in range(f):
        for n, (j, k) in enumerate(itertools.combinations(range(g), 2)):
            if j == i or k == i or max(i, k) >= vectors.shape[0]:
                continue
            rep = get_2Dtriang_rep(vectors[i], vectors[j], vectors[k])
            cand = kd_tree.query_ball_point(rep, TOLERANCE_TRIANGLE)
            print(len(cand))
            reps.append(rep)
    reps = np.array(reps)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2])
    ax.set_aspect('equal')
    plt.show()

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
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.show()
    
    matched_triangs = match_image_triangles(centroids)
    

if __name__ == '__main__':
    #database_cache.prepare_triangles()
    options = {'flag_display':False, 'rough_match_threshhold':36, 'flag_display2':1, 'flag_debug':0}
    path_data = r'D:\feb7test\Don2017_clean2\eclipse_field\centroid_data20250214172224.zip' # eclipse (Don)
    archive = zipfile.ZipFile(path_data, 'r')
    meta_data = json.load(archive.open('results.txt'))
    df = pd.read_csv(archive.open('STACKED_CENTROIDS_DATA.csv'))
    df = df.astype({'px':float, 'py':float}) # fix datatypes
    centroids = np.c_[df['py'], df['px']] # important: (y, x) representation expected
    result = platesolve(centroids, meta_data['img_shape'], options)

    print(result)
