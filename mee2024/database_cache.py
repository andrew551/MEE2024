"""
Process-level caches for the star catalogues and the triangle pattern database.

The triangle database is ~127 MB and takes several seconds to load (and several minutes
to generate the first time). ``prepare_triangles()`` starts that work in a background
process so it can overlap with the user filling in the GUI; ``open_catalogue()`` then
blocks until it is ready. Callers that never called ``prepare_triangles()`` -- the CLI
and the tests -- fall back to loading it synchronously.
"""

import gc
import time
import traceback

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
        if path == get_triangle_db_path():
            _cache.catalogue_cache[path] = _get_triangles()
        elif _is_starcat_name(path):
            from mee2024.starcat import providers
            _cache.catalogue_cache[path] = providers.build(str(path), **kwargs)
        elif (directory := _installed_catalogue_dir(path)) is not None:
            from mee2024.starcat import providers
            _cache.catalogue_cache[path] = providers.GaiaOfflineProvider(directory)
        else:
            _cache.catalogue_cache[path] = database_lookup2.database_searcher(
                path, debug_folder=debug_folder, star_max_magnitude=12)

    return _cache.catalogue_cache[path]


def release_catalogues():
    """Drop every cached star catalogue, closing the files they hold mapped.

    The cache exists so a second run does not re-open the archive, which is the right
    default and the wrong one when the user is trying to delete that archive: a mapped
    file cannot be removed on Windows. Called before a deletion, so the app can free
    the disk without being restarted. The triangle database is deliberately left alone
    -- it is not a catalogue and reloading it is expensive.

    The v2 **pattern** databases are released too. They are a different directory and a
    different format, but the same mapping problem and the same caller intent: whoever
    calls this is trying to free disk, and a pattern database is the larger of the two
    (230 MB against 138 MB). Deliberately *not* done at the end of every run -- the
    KD-tree over the invariants is built on first use, and a watch-mode session solving
    field after field would pay for it again each time.
    """
    released = []
    try:
        from mee2024.platesolve2 import pattern_db
        released += pattern_db.release_databases()
    except Exception:
        traceback.print_exc()
    for path, cached in list(_cache.catalogue_cache.items()):
        if isinstance(cached, TriangleData):
            continue
        closer = getattr(cached, 'close', None)
        if callable(closer):
            try:
                closer()
            except Exception:
                traceback.print_exc()
        del _cache.catalogue_cache[path]
        released.append(str(path))
    # a memmap is only really gone once nothing refers to it
    gc.collect()
    return released


def _is_starcat_name(path):
    """Is this the name of a provider in the starcat registry?"""
    if not isinstance(path, str):
        return False
    from mee2024.starcat import providers
    return path in providers.known_catalogues()


def _installed_catalogue_dir(path):
    """The directory of an offline catalogue present on disk, or None.

    Accepts both a registered release name and the name of a locally built catalogue,
    so a catalogue produced by tools/build_gaia_offline.py is usable immediately.

    **Bundled copies count.** The executable ships the compact archive inside itself, and
    `CatalogueRelease.is_installed` says so, which is what puts it in the app's catalogue
    list. This only looked in the user's data directory, so a catalogue the app offered
    could not then be opened: selecting the bundled `gaia_dr3_g10` fell through to the
    legacy Tycho reader and died on `open('gaia_dr3_g10')`. Availability and location have
    to agree about where a catalogue may live.
    """
    if not isinstance(path, str):
        return None
    from mee2024.MEE2024util import get_catalogue_root
    from mee2024.starcat import download, store
    candidates = []
    release = download.RELEASES.get(path)
    if release is not None:
        candidates += [release.directory(), release.bundled_directory()]
    candidates.append(get_catalogue_root() / path)   # locally built, not a release
    for directory in candidates:
        try:
            store.read_manifest(directory)
            return directory
        except (FileNotFoundError, ValueError, OSError):
            continue
    return None
