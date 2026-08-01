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

INVARIANT_RATIO_DPHI = 'ratio_dphi'    # v1's pair: tri_inv = (ratio, dphi)
INVARIANT_KENDALL = 'kendall'          # tri_inv = shape-sphere (x, y); z derived;
                                       # tri_perm = 3-bit vertex permutation code

#: (manifest key, filename, dtype) -- dtypes are pinned so a DB built on one
#: platform loads identically on another (np.int_ is int32 on Windows, int64 on Linux).
#: Keys absent from the arrays dict are simply not written (tri_perm exists only for
#: the kendall invariant; tri_anchor and tri_bucket_offset only for bucket-sorted
#: databases, whose triangle rows are grid-ordered rather than anchor-ordered).
COLUMNS = (
    ('anchors', 'anchors.npy', np.float32),
    ('anchor_tri_offset', 'anchor_tri_offset.npy', np.int64),
    ('pattern_ind', 'pattern_ind.npy', np.int32),
    ('pattern_data', 'pattern_data.npy', np.float32),
    ('tri_legs', 'tri_legs.npy', np.uint8),
    ('tri_inv', 'tri_inv.npy', np.float32),
    ('tri_perm', 'tri_perm.npy', np.uint8),
    ('tri_anchor', 'tri_anchor.npy', np.int32),
    ('tri_bucket_offset', 'tri_bucket_offset.npy', np.int64),
)

MANIFEST_FILE = 'manifest.json'

#: the variant a run uses when options['pattern_db'] is empty and it is installed.
#: The kendall variant won the S2 bench (docs/bench/BENCH.md); patdb_g12_t17 remains
#: the named rollback.
DEFAULT_NAME = 'patdb_g12_t17k'


def write_pattern_db(directory, arrays, params, source, verify_spec, provenance=''):
    """Write the arrays and manifest. ``arrays`` maps the COLUMNS keys to ndarrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    columns = {}
    for key, filename, dtype in COLUMNS:
        if key not in arrays:
            continue
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


#: The blind multi-scale layer set (S6), one group per FOV scale, primary group first.
#: Within a group the first installed name wins, so a layer rebuilt against a newer
#: catalogue supersedes its predecessor without any configuration; groups whose members
#: are all absent are simply skipped.
LAYER_SET = (
    ('patdb_g13_t17k', 'patdb_g12_t17k'),      # ~1.4-10 deg: the primary
    ('patdb_g13_t06k', 'patdb_g12_t06k'),      # ~1-2 deg
    ('patdb_g13_t40k', 'patdb_g12_t40k'),      # ~8-18 deg
)


def resolve(options=None):
    """The primary PatternDB a run should use, honouring options['pattern_db'].

    ``pattern_db`` may be a comma-separated layer list; the first entry is the
    primary. An empty selection prefers DEFAULT_NAME, then the only installed
    database. A missing database raises with the exact command that creates one.
    """
    return resolve_layers(options)[0]


def resolve_layers(options=None):
    """Every pattern-database layer a solve should consult, primary first.

    An explicit ``pattern_db`` (single name or comma list) is honoured exactly --
    which is also how a bench pins a single layer. The empty default uses whichever
    members of LAYER_SET are installed, so installing the narrow or wide layer
    deepens blind coverage with no configuration.
    """
    requested = (options or {}).get('pattern_db', '')
    if requested:
        layers = []
        for name in [n.strip() for n in requested.split(',') if n.strip()]:
            directory = get_patterndb_root() / name
            if not directory.is_dir():
                raise RuntimeError(
                    f'pattern database {name!r} is not installed at {directory}. '
                    f'Build it with `mee2024 build-pattern-db --name {name}`.')
            layers.append(open_db(directory))
        return layers
    installed = installed_databases()
    layers = []
    for group in LAYER_SET:
        for name in group:
            if name in installed:
                layers.append(open_db(get_patterndb_root() / name))
                break
    if layers:
        return layers
    if installed:
        return [open_db(get_patterndb_root() / installed[0])]
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
    tri_perm = property(lambda self: self._open('tri_perm'))
    tri_anchor = property(lambda self: self._open('tri_anchor'))
    tri_bucket_offset = property(lambda self: self._open('tri_bucket_offset'))

    @property
    def is_bucketed(self):
        return 'tri_bucket_offset' in self.manifest['columns']

    @property
    def grid_n(self):
        return int(self.manifest['params']['bucket_grid_n'])

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
    def tolerance(self):
        """The invariant-space match radius this database was calibrated for."""
        default = 0.01 if self.invariant == INVARIANT_RATIO_DPHI else 0.004
        return float(self.manifest['params'].get('tolerance') or default)

    @property
    def kd_tree(self):
        if self._kd_tree is None:
            from scipy.spatial import KDTree
            if self.invariant == INVARIANT_RATIO_DPHI:
                # dphi is periodic; ratio is not (the huge first boxsize disables
                # wrap on that axis) -- v1's exact construction
                self._kd_tree = KDTree(np.asarray(self.tri_inv),
                                       boxsize=[9999999, np.pi * 2])
            elif self.invariant == INVARIANT_KENDALL:
                xy = np.asarray(self.tri_inv, dtype=np.float64)
                z = np.sqrt(np.maximum(1.0 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 0.0))
                self._kd_tree = KDTree(np.c_[xy, z])
            else:
                raise ValueError(f'unknown invariant {self.invariant!r}')
        return self._kd_tree

    def triangle_anchor_and_legs(self, triangle_indices):
        """Decode flat triangle rows -> (anchor index, leg j, leg k)."""
        if 'tri_anchor' in self.manifest['columns']:
            anchor = np.asarray(self.tri_anchor)[triangle_indices]
        else:
            offsets = np.asarray(self.anchor_tri_offset)
            anchor = np.searchsorted(offsets, triangle_indices, side='right') - 1
        legs = np.asarray(self.tri_legs)[triangle_indices]
        return anchor, legs[:, 0], legs[:, 1]

    def query_ball(self, points, radii):
        """Triangle rows within each radius of each shape point, with distances.

        Returns (row indices, query-point rows, distances) as flat arrays. On a
        bucket-sorted database this is a pure gather over memory-mapped columns --
        no index is ever built, so a cold process pays pages, not seconds. Databases
        without buckets fall back to the KD-tree (built on first use).
        """
        points = np.asarray(points, dtype=np.float64)
        radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(points),))
        if not self.is_bucketed:
            hit_lists = self.kd_tree.query_ball_point(points, radii)
            cand = np.array([i for hits in hit_lists for i in hits], dtype=np.int64)
            rows = np.repeat(np.arange(len(points)), [len(h) for h in hit_lists])
            xy = np.asarray(self.tri_inv, dtype=np.float64)[cand]
            z = np.sqrt(np.maximum(1.0 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 0.0))
            dist = np.linalg.norm(np.c_[xy, z] - points[rows], axis=1)
            return cand, rows, dist

        grid_n = self.grid_n
        cell = 2.0 / grid_n
        offsets = np.asarray(self.tri_bucket_offset)
        inv = self.tri_inv     # memory-mapped; sliced per gather

        out_cand, out_rows, out_dist = [], [], []
        for q, (point, radius) in enumerate(zip(points, radii)):
            ix0 = int(np.clip((point[0] - radius + 1) / cell, 0, grid_n - 1))
            ix1 = int(np.clip((point[0] + radius + 1) / cell, 0, grid_n - 1))
            iy0 = int(np.clip((point[1] - radius + 1) / cell, 0, grid_n - 1))
            iy1 = int(np.clip((point[1] + radius + 1) / cell, 0, grid_n - 1))
            # y is the minor grid axis, so each x-stripe of cells is one contiguous
            # row range in the bucket-sorted arrays
            pieces = []
            for ix in range(ix0, ix1 + 1):
                lo = int(offsets[ix * grid_n + iy0])
                hi = int(offsets[ix * grid_n + iy1 + 1])
                if hi > lo:
                    pieces.append(np.arange(lo, hi, dtype=np.int64))
            if not pieces:
                continue
            rows = np.concatenate(pieces)
            xy = np.asarray(inv[rows], dtype=np.float64)
            dz = np.sqrt(np.maximum(1.0 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 0.0)) \
                - point[2]
            dist_sq = ((xy[:, 0] - point[0]) ** 2 + (xy[:, 1] - point[1]) ** 2
                       + dz ** 2)
            keep = dist_sq <= radius ** 2
            if not np.any(keep):
                continue
            out_cand.append(rows[keep])
            out_rows.append(np.full(int(keep.sum()), q, dtype=np.int64))
            out_dist.append(np.sqrt(dist_sq[keep]))
        if not out_cand:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0)
        return (np.concatenate(out_cand), np.concatenate(out_rows),
                np.concatenate(out_dist))

    def __repr__(self):
        return (f'<PatternDB {self.name}: {self.manifest["n_anchors"]} anchors, '
                f'{self.manifest["n_triangles"]} triangles, {self.invariant}>')
