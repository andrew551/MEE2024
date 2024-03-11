import tetra3
import database_lookup2
import gaia_search
import numpy as np

class _cache:

    database_cache = {}

    catalogue_cache = {}


def open_database(path):
    if not path in _cache.database_cache:
        _cache.database_cache[path] = tetra3.Tetra3(load_database=path)

    return _cache.database_cache[path]

def open_catalogue(path, debug_folder=None):
    if not path in _cache.catalogue_cache:
        if path == 'gaia':
            _cache.catalogue_cache[path] = gaia_search.dbs_gaia()
        elif path == "TripleTrianglePlatesolveDatabase/TripleTriangle_pattern_data.npz":
            _cache.catalogue_cache[path] = np.load(path)
        else:
            _cache.catalogue_cache[path] = database_lookup2.database_searcher(path, debug_folder=debug_folder, star_max_magnitude=12)

    return _cache.catalogue_cache[path]
