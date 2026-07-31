"""
The v2 pattern database: a directory of uncompressed .npy files plus a manifest.

Mirrors the star-catalogue store (``mee2024/starcat/store.py``): per-file SHA-256,
a format version, pinned dtypes, and named variants under
``<user data>/patterndb/<name>/`` so several databases (scales, invariants, dedupe
rules) can coexist and a run can select one by name.

Layout::

    <patterndb root>/<name>/
        manifest.json           format, params, source catalogue, verification spec
        anchors.npy             float32 (n_anchors, 3)   anchor unit vectors
        anchor_tri_offset.npy   int64   (n_anchors + 1,) first triangle row per anchor
        pattern_ind.npy         int32   (n_anchors, e)   catalogue row of each leg, -1 pad
        pattern_data.npy        float32 (n_anchors, e, 5) [dtheta, phi, vx, vy, vz]
        tri_legs.npy            uint8   (n_tri, 2)       leg pair (j, k), j < k
        tri_inv.npy             float32 (n_tri, 2)       invariant (S1: ratio, dphi)

The prefix-sum ``anchor_tri_offset`` replaces v1's fixed ``anchor * 153 + pair``
arithmetic: an anchor with fewer than ``e`` legs simply owns fewer rows, so the sparse
sky that made v1's builder raise "edge case handling unimplemented" is just data here.

A missing database is an error naming the exact remedy -- never a silent inline
rebuild (v1's ``database_cache._load_triangles`` regenerates for minutes on any load
failure, which is the behaviour this module exists to retire).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mee2024.MEE2024util import get_patterndb_root
from mee2024.starcat.store import sha256

FORMAT = 'mee2024-patterndb'
FORMAT_VERSION = 1

INVARIANT_RATIO_DPHI = 'ratio_dphi'

#: (manifest key, filename, dtype) -- dtypes are pinned so a DB built on one
#: platform loads identically on another (np.int_ is int32 on Windows, int64 on Linux)
COLUMNS = (
    ('anchors', 'anchors.npy', np.float32),
    ('anchor_tri_offset', 'anchor_tri_offset.npy', np.int64),
    ('pattern_ind', 'pattern_ind.npy', np.int32),
    ('pattern_data', 'pattern_data.npy', np.float32),
    ('tri_legs', 'tri_legs.npy', np.uint8),
    ('tri_inv', 'tri_inv.npy', np.float32),
)

MANIFEST_FILE = 'manifest.json'

#: the variant a run uses when options['pattern_db'] is empty and it is installed
DEFAULT_NAME = 'patdb_g12_t17'


def write_pattern_db(directory, arrays, params, source, verify_spec, provenance=''):
    """Write the arrays and manifest. ``arrays`` maps the COLUMNS keys to ndarrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    columns = {}
    for key, filename, dtype in COLUMNS:
        path = directory / filename
        np.save(path, np.ascontiguousarray(arrays[key], dtype=dtype))
        columns[key] = {'file': filename, 'dtype': np.dtype(dtype).name,
                        'sha256': sha256(path)}

    manifest = {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'name': directory.name,
        'invariant': params['invariant'],
        'params': params,
        'source': source,          # the star catalogue the DB was built from
        'verify': verify_spec,     # what the solver verifies candidate solutions against
        'n_anchors': int(arrays['anchors'].shape[0]),
        'n_triangles': int(arrays['tri_inv'].shape[0]),
        'built': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'provenance': provenance,
        'columns': columns,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2),
                                           encoding='utf-8')
    return manifest


def read_manifest(directory):
    directory = Path(directory)
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f'no {MANIFEST_FILE} in {directory}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT:
        raise ValueError(f'{directory} is not a {FORMAT} database')
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{directory} has format version '
                         f'{manifest.get("format_version")}, expected {FORMAT_VERSION}')
    return manifest


def verify(directory, quick=False):
    """Check every file against the manifest. Returns the list of problems found."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    problems = []
    for key, entry in manifest['columns'].items():
        path = directory / entry['file']
        if not path.exists():
            problems.append(f'missing: {entry["file"]}')
        elif not quick and sha256(path) != entry['sha256']:
            problems.append(f'checksum mismatch: {entry["file"]}')
    return problems


def installed_databases():
    """Names of every valid pattern database on disk."""
    root = get_patterndb_root()
    names = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            read_manifest(path)
        except Exception:
            continue
        names.append(path.name)
    return names


#: one PatternDB per directory for the process lifetime: the KD-tree it builds on
#: first use is expensive, and a watch-mode session solves many fields in a row
_DB_CACHE = {}


def open_db(directory):
    key = str(Path(directory).resolve())
    if key not in _DB_CACHE:
        _DB_CACHE[key] = PatternDB(directory)
    return _DB_CACHE[key]


def resolve(options=None):
    """The PatternDB a run should use, honouring options['pattern_db'].

    An empty selection prefers DEFAULT_NAME, then the only installed database. A
    missing database raises with the exact command that creates one.
    """
    requested = (options or {}).get('pattern_db', '')
    if requested:
        directory = get_patterndb_root() / requested
        if not directory.is_dir():
            raise RuntimeError(
                f'pattern database {requested!r} is not installed at {directory}. '
                f'Build it with `mee2024 build-pattern-db --name {requested}`.')
        return open_db(directory)
    installed = installed_databases()
    if DEFAULT_NAME in installed:
        return open_db(get_patterndb_root() / DEFAULT_NAME)
    if installed:
        return open_db(get_patterndb_root() / installed[0])
    raise RuntimeError(
        'no v2 pattern database is installed. Build the default with '
        '`mee2024 build-pattern-db` (needs the gaia_dr3_g12 offline catalogue), '
        "or set platesolver='triangle' to use the production solver.")


class PatternDB:
    """Read access to one pattern database. Arrays are memory-mapped on open.

    The KD-tree over the invariants is built on first use (S1 keeps v1's
    build-at-load behaviour so the A/B isolates the catalogue; S5b replaces it
    with a precomputed bucket index).
    """

    def __init__(self, directory, verify_checksums=False):
        self.directory = Path(directory)
        self.manifest = read_manifest(self.directory)
        if verify_checksums:
            problems = verify(self.directory)
            if problems:
                raise ValueError(f'{self.directory} failed verification: {problems}')
        self._arrays = {}
        self._kd_tree = None

    def _open(self, key):
        if key not in self._arrays:
            entry = self.manifest['columns'][key]
            self._arrays[key] = np.load(self.directory / entry['file'], mmap_mode='r')
        return self._arrays[key]

    anchors = property(lambda self: self._open('anchors'))
    anchor_tri_offset = property(lambda self: self._open('anchor_tri_offset'))
    pattern_ind = property(lambda self: self._open('pattern_ind'))
    pattern_data = property(lambda self: self._open('pattern_data'))
    tri_legs = property(lambda self: self._open('tri_legs'))
    tri_inv = property(lambda self: self._open('tri_inv'))

    @property
    def name(self):
        return self.manifest['name']

    @property
    def invariant(self):
        return self.manifest['invariant']

    @property
    def pattern_width(self):
        """e: how many legs each anchor's pattern may hold (the solver's g)."""
        return int(self.manifest['params']['e'])

    @property
    def kd_tree(self):
        if self._kd_tree is None:
            from scipy.spatial import KDTree
            if self.invariant != INVARIANT_RATIO_DPHI:
                raise ValueError(f'unknown invariant {self.invariant!r}')
            # dphi is periodic; ratio is not (the huge first boxsize disables wrap
            # on that axis) -- v1's exact construction
            self._kd_tree = KDTree(np.asarray(self.tri_inv),
                                   boxsize=[9999999, np.pi * 2])
        return self._kd_tree

    def triangle_anchor_and_legs(self, triangle_indices):
        """Decode flat triangle rows -> (anchor index, leg j, leg k)."""
        offsets = np.asarray(self.anchor_tri_offset)
        anchor = np.searchsorted(offsets, triangle_indices, side='right') - 1
        legs = np.asarray(self.tri_legs)[triangle_indices]
        return anchor, legs[:, 0], legs[:, 1]

    def __repr__(self):
        return (f'<PatternDB {self.name}: {self.manifest["n_anchors"]} anchors, '
                f'{self.manifest["n_triangles"]} triangles, {self.invariant}>')
