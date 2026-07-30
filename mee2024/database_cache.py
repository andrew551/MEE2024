"""
Process-level caches for the star catalogues and the triangle pattern database.

The triangle database is ~127 MB and takes several seconds to load (and several minutes
to generate the first time). ``prepare_triangles()`` starts that work in a background
process so it can overlap with the user filling in the GUI; ``open_catalogue()`` then
blocks until it is ready. Callers that never called ``prepare_triangles()`` -- the CLI
and the tests -- fall back to loading it synchronously.
"""

import time

import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Process, Manager

from mee2024 import database_lookup2
from mee2024 import gaia_search
from mee2024 import platesolve_new
from mee2024.MEE2024util import get_triangle_db_path


class _cache:

    catalogue_cache = {}

    q = None

    manager = None

    prepare_process = None


class TriangleData:

    def __init__(self, cata_data):
        self.triangles = cata_data['triangles'] # (n x T x 2 array) - radius ratio and angular seperation for each triangle (note: T = N(N-1)/2)
        self.anchors = cata_data['anchors'] # vector rep of each "anchor" star
        self.pattern_data = cata_data['pattern_data'] # (n x N x 5 array) of (dtheta, phi, star_vector) for each neighbour star
        self.pattern_ind = cata_data['pattern_ind'] # n x N array of integer : the indices of neighbouring stars
        self.kd_tree = KDTree(self.triangles.reshape((-1, 2)), boxsize=[9999999, np.pi*2]) # use a 2-pi periodic condition for polar angle (and basically infinity for ratio)


def _load_triangles():
    """Load the triangle database, generating it first if it is not there yet."""
    try:
        return TriangleData(np.load(get_triangle_db_path()))
    except Exception:
        print("no triangles platesolving database found: will now generate one (this will take a few minutes)")
        platesolve_new.generate()
        return TriangleData(np.load(get_triangle_db_path()))


def work(q):
    print("working on loading triangles")
    q.put(_load_triangles())
    print("finished preparation work")


def prepare_triangles():
    """Start loading the triangle database in the background. Idempotent."""
    if _cache.prepare_process is not None:
        return
    print('preparing')
    _cache.manager = Manager()
    _cache.q = _cache.manager.Queue()
    _cache.prepare_process = Process(target=work, args=(_cache.q,))
    _cache.prepare_process.start()


def shutdown_triangles():
    """Tear down the background preparation process and its manager, if any."""
    proc, manager = _cache.prepare_process, _cache.manager
    _cache.prepare_process, _cache.q, _cache.manager = None, None, None
    try:
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    except Exception:
        pass
    try:
        if manager is not None:
            manager.shutdown()
    except Exception:
        pass


def _get_triangles():
    """The triangle database, from the background process if one was started."""
    if _cache.q is None:
        return _load_triangles()  # nobody called prepare_triangles(): just load it here
    i = 1
    while _cache.q.empty():
        print(f"triangles not ready yet ... waiting for them to be ready ({i})")
        time.sleep(1)
        i += 1
    data = _cache.q.get()
    if _cache.prepare_process is not None:
        _cache.prepare_process.join()
    return data


def open_catalogue(path, debug_folder=None, **kwargs):
    if path not in _cache.catalogue_cache:
        if path == 'gaia':
            _cache.catalogue_cache[path] = gaia_search.dbs_gaia(**kwargs)
        elif path == get_triangle_db_path():
            _cache.catalogue_cache[path] = _get_triangles()
        else:
            _cache.catalogue_cache[path] = database_lookup2.database_searcher(
                path, debug_folder=debug_folder, star_max_magnitude=12)

    return _cache.catalogue_cache[path]
