import tetra3
import database_lookup2
import gaia_search
import numpy as np
from scipy.spatial import KDTree
import time
import platesolve_new

class _cache:

    database_cache = {}

    catalogue_cache = {}

class TriangleData:

    def __init__(self, cata_data):
        self.triangles = cata_data['triangles'] # (n x T x 2 array) - radius ratio and angular seperation for each triangle (note: T = N(N-1)/2)
        self.anchors = cata_data['anchors'] # vector rep of each "anchor" star
        self.pattern_data = cata_data['pattern_data'] # (n x N x 5 array) of (dtheta, phi, star_vector) for each neighbour star
        self.pattern_ind = cata_data['pattern_ind'] # n x N array of integer : the indices of neighbouring stars
        self.kd_tree = KDTree(self.triangles.reshape((-1, 2)), boxsize=[9999999, np.pi*2]) # use a 2-pi periodic condition for polar angle (and basically infinity for ratio)


triangles_path = "TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz"
def prepare_triangles():
    try:
        _cache.catalogue_cache[triangles_path] = TriangleData(np.load(triangles_path))
        print("preloaded triangles")
    except Exception:
        print("no triangles file found: will now generate one (this will take a few minutes)")
        platesolve_new.generate()
        _cache.catalogue_cache[triangles_path] = TriangleData(np.load(triangles_path))

def open_database(path):
    if not path in _cache.database_cache:
        _cache.database_cache[path] = tetra3.Tetra3(load_database=path)

    return _cache.database_cache[path]

def open_catalogue(path, debug_folder=None):
    if not path in _cache.catalogue_cache:
        if path == 'gaia':
            _cache.catalogue_cache[path] = gaia_search.dbs_gaia()
        elif path == "TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz":
            raise Exception("expected triangles to be pre-loaded")
        else:
            _cache.catalogue_cache[path] = database_lookup2.database_searcher(path, debug_folder=debug_folder, star_max_magnitude=12)

    return _cache.catalogue_cache[path]


