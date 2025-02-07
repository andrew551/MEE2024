import tetra3
import database_lookup2
import gaia_search
import numpy as np
from scipy.spatial import KDTree
import time
import platesolve_new
from multiprocessing import Process, Queue
from multiprocessing import Manager
class _cache:

    database_cache = {}

    catalogue_cache = {}

    q = None

    prepare_process = None

class TriangleData:

    def __init__(self, cata_data):
        self.triangles = cata_data['triangles'] # (n x T x 2 array) - radius ratio and angular seperation for each triangle (note: T = N(N-1)/2)
        self.anchors = cata_data['anchors'] # vector rep of each "anchor" star
        self.pattern_data = cata_data['pattern_data'] # (n x N x 5 array) of (dtheta, phi, star_vector) for each neighbour star
        self.pattern_ind = cata_data['pattern_ind'] # n x N array of integer : the indices of neighbouring stars
        self.kd_tree = KDTree(self.triangles.reshape((-1, 2)), boxsize=[9999999, np.pi*2]) # use a 2-pi periodic condition for polar angle (and basically infinity for ratio)

triangles_path = "TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz"
def work(q):
    print("working on loading triangles")
    try:
        q.put(TriangleData(np.load(triangles_path)))
        print("preloaded triangles")
    except Exception:
        print("no triangles platesolving database found: will now generate one (this will take a few minutes)")
        platesolve_new.generate()
        q.put(TriangleData(np.load(triangles_path)))
    print("finished preparation work")

def prepare_triangles():
    print('preparing')
    manager = Manager()
    result_queue = manager.Queue()
    _cache.q=result_queue
    _cache.prepare_process = Process(target=work, args = (_cache.q,))
    _cache.prepare_process.start()
    
def open_database(path):
    if not path in _cache.database_cache:
        _cache.database_cache[path] = tetra3.Tetra3(load_database=path)

    return _cache.database_cache[path]

def open_catalogue(path, debug_folder=None, **kwaargs):
    if not path in _cache.catalogue_cache:
        if path == 'gaia':
            _cache.catalogue_cache[path] = gaia_search.dbs_gaia(**kwaargs)
        elif path == triangles_path:
            print(_cache.prepare_process, _cache.prepare_process.is_alive())
            i = 1          
            while _cache.q.empty() and not path in _cache.catalogue_cache:
                print(f"triangles not ready yet ... waiting for them to be ready ({i})")
                time.sleep(1)
                i+=1
            if not path in _cache.catalogue_cache:
                _cache.catalogue_cache[path] = _cache.q.get()
                _cache.prepare_process.join()
                print("joined")
            
        else:
            _cache.catalogue_cache[path] = database_lookup2.database_searcher(path, debug_folder=debug_folder, star_max_magnitude=12)

    return _cache.catalogue_cache[path]


