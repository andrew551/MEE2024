"""
On-disk format for an offline star catalogue.

A **directory of uncompressed .npy files, memory-mapped, sorted by declination**, with a
one-degree declination band index.

Why not a single compressed .npz: a compressed archive has to be fully decompressed to
read one star. At G<12 that is ~139 MB decompressed on every run. Memory-mapping reads
only the pages a query actually touches, so a bounding-box lookup costs a few hundred
kilobytes regardless of how large the catalogue is.

Why declination-sorted with a band index: every query the pipeline makes is a bounding
box (``MEE2024util.get_bbox`` produces one). Declination bounds therefore reduce to a
single contiguous row range, and the RA filter -- including wrap-around through zero --
is applied in memory over that small slice.

    <directory>/
        manifest.json     format version, provenance, epoch, limits, SHA-256 per file
        ra.npy dec.npy    float64, radians, ascending in dec
        mag.npy           float32
        source_id.npy     int64
        pmra.npy pmdec.npy parallax.npy radial_velocity.npy   float32
        nn_sep.npy nn_mag.npy                                 float32
        dec_index.npy     int64, 181 entries: first row of each 1-degree dec band
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from mee2024.starcat.table import ORIGIN_GAIA, StarTable

FORMAT = 'mee2024-starcat'
FORMAT_VERSION = 1

#: (attribute on StarTable, filename, dtype)
COLUMNS = (
    ('ra', 'ra.npy', np.float64),
    ('dec', 'dec.npy', np.float64),
    ('mag', 'mag.npy', np.float32),
    ('ids', 'source_id.npy', np.int64),
    ('pmra', 'pmra.npy', np.float32),
    ('pmdec', 'pmdec.npy', np.float32),
    ('parallax', 'parallax.npy', np.float32),
    ('radial_velocity', 'radial_velocity.npy', np.float32),
    ('nn_sep', 'nn_sep.npy', np.float32),
    ('nn_mag', 'nn_mag.npy', np.float32),
)

DEC_INDEX_FILE = 'dec_index.npy'
MANIFEST_FILE = 'manifest.json'
N_DEC_BANDS = 180  # one degree each, from -90 to +90


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_dec_index(dec_radians):
    """First row of each 1-degree declination band, for dec-sorted input.

    Returns N_DEC_BANDS + 1 entries; band k covers [-90 + k, -89 + k) degrees, and its
    rows are [index[k], index[k + 1]).
    """
    dec_deg = np.degrees(dec_radians)
    if np.any(np.diff(dec_deg) < 0):
        raise ValueError('declinations must be sorted ascending to build the index')
    band = np.clip(np.floor(dec_deg + 90).astype(np.int64), 0, N_DEC_BANDS - 1)
    # searchsorted over the band number gives each band's first row
    return np.searchsorted(band, np.arange(N_DEC_BANDS + 1), side='left').astype(np.int64)


def write_catalogue(directory, table, name, catalogue='Gaia DR3', provenance='',
                    magnitude_limit=None, built=None):
    """Write a StarTable out in the offline format. The table is sorted by dec first."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table = table.sorted_by_dec()

    columns = {}
    for attribute, filename, dtype in COLUMNS:
        path = directory / filename
        np.save(path, np.ascontiguousarray(getattr(table, attribute), dtype=dtype))
        columns[attribute] = {'file': filename, 'dtype': np.dtype(dtype).name,
                              'sha256': sha256(path)}

    index_path = directory / DEC_INDEX_FILE
    np.save(index_path, build_dec_index(table.dec))
    columns['dec_index'] = {'file': DEC_INDEX_FILE, 'dtype': 'int64',
                            'sha256': sha256(index_path)}

    manifest = {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'name': name,
        'catalogue': catalogue,
        'band': table.band,
        'epoch': table.epoch,
        'magnitude_limit': (float(magnitude_limit) if magnitude_limit is not None
                            else float(np.max(table.mag)) if len(table) else None),
        'n_stars': int(len(table)),
        'dec_band_degrees': 1.0,
        'provenance': provenance,
        'built': built or '',
        'columns': columns,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def rebuild_manifest(directory, name, band='G', epoch=None, magnitude_limit=None,
                     catalogue='Gaia DR3', provenance=''):
    """Reconstruct a lost manifest from the data files, or raise if they are not sound.

    An interrupted install leaves every column written and no manifest -- after which
    the archive reports as absent, silently. The columns carry enough to rebuild the
    manifest, *provided* they are internally consistent, so this validates before it
    writes: equal lengths, declination genuinely sorted (the format's whole premise),
    a band index that matches the data, and finite positions. What cannot be derived
    from the data -- band, epoch, intended depth -- must be supplied by the caller.
    """
    directory = Path(directory)
    if epoch is None:
        raise ValueError('epoch cannot be recovered from the data; supply it')

    arrays, columns = {}, {}
    for attribute, filename, dtype in COLUMNS:
        path = directory / filename
        if not path.exists():
            raise ValueError(f'{filename} is missing: this archive cannot be repaired, '
                             f'reinstall it')
        arrays[attribute] = np.load(path, mmap_mode='r')
        if arrays[attribute].dtype != np.dtype(dtype):
            raise ValueError(f'{filename} holds {arrays[attribute].dtype}, '
                             f'expected {np.dtype(dtype)}')
        columns[attribute] = {'file': filename, 'dtype': np.dtype(dtype).name,
                              'sha256': sha256(path)}

    lengths = {len(a) for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f'columns have differing lengths ({sorted(lengths)}): the '
                         f'archive is truncated, reinstall it')
    n_stars = lengths.pop()
    if not n_stars:
        raise ValueError('the archive is empty')

    dec = np.asarray(arrays['dec'])
    if np.any(np.diff(dec) < 0):
        raise ValueError('declination is not sorted: lookups would silently miss '
                         'stars, so this archive cannot be repaired')
    ra = np.asarray(arrays['ra'])
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValueError('positions contain NaN or infinity')
    if ra.min() < -1e-9 or ra.max() > 2 * np.pi + 1e-9:
        raise ValueError('right ascension is outside [0, 2pi]: wrong units?')

    index_path = directory / DEC_INDEX_FILE
    expected_index = build_dec_index(dec)
    if index_path.exists():
        stored = np.load(index_path)
        if stored.shape != expected_index.shape or np.any(stored != expected_index):
            raise ValueError('the declination band index does not match the data')
    else:
        np.save(index_path, expected_index)
    columns['dec_index'] = {'file': DEC_INDEX_FILE, 'dtype': 'int64',
                            'sha256': sha256(index_path)}

    mag = np.asarray(arrays['mag'])
    faintest = float(np.max(mag))
    if magnitude_limit is not None and faintest > float(magnitude_limit) + 1e-3:
        raise ValueError(f'stars reach G={faintest:.2f}, past the stated limit '
                         f'G<{magnitude_limit}: this is not the archive it claims to be')

    manifest = {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'name': name,
        'catalogue': catalogue,
        'band': band,
        'epoch': float(epoch),
        'magnitude_limit': (float(magnitude_limit) if magnitude_limit is not None
                            else faintest),
        'n_stars': int(n_stars),
        'dec_band_degrees': 1.0,
        'provenance': provenance,
        'built': '',
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
        raise ValueError(f'{directory} is not a {FORMAT} catalogue')
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{directory} has format version '
                         f'{manifest.get("format_version")}, expected {FORMAT_VERSION}')
    return manifest


def verify(directory, quick=False):
    """Check every file against the manifest. Returns the list of problems found."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    problems = []
    for attribute, entry in manifest['columns'].items():
        path = directory / entry['file']
        if not path.exists():
            problems.append(f'missing: {entry["file"]}')
            continue
        if not quick and sha256(path) != entry['sha256']:
            problems.append(f'checksum mismatch: {entry["file"]}')
    return problems


class OfflineCatalogue:
    """Memory-mapped read access to a catalogue directory."""

    def __init__(self, directory, verify_checksums=False):
        self.directory = Path(directory)
        self.manifest = read_manifest(self.directory)
        if verify_checksums:
            problems = verify(self.directory)
            if problems:
                raise ValueError(f'{self.directory} failed verification: {problems}')
        self._mmap = {}
        self.dec_index = self._open('dec_index')

    def _open(self, attribute):
        if attribute not in self._mmap:
            entry = self.manifest['columns'][attribute]
            self._mmap[attribute] = np.load(self.directory / entry['file'], mmap_mode='r')
        return self._mmap[attribute]

    # ------------------------------------------------------------- properties

    @property
    def n_stars(self):
        return self.manifest['n_stars']

    @property
    def epoch(self):
        return float(self.manifest['epoch'])

    @property
    def band(self):
        return self.manifest['band']

    @property
    def magnitude_limit(self):
        return self.manifest['magnitude_limit']

    def __len__(self):
        return self.n_stars

    def __repr__(self):
        return (f'<OfflineCatalogue {self.manifest["name"]}: {self.n_stars} stars, '
                f'epoch {self.epoch}, {self.band}<{self.magnitude_limit}>')

    # ----------------------------------------------------------------- lookup

    def _dec_rows(self, dec_range):
        """The contiguous row range covering a declination interval."""
        lo, hi = min(dec_range), max(dec_range)
        first = int(np.clip(np.floor(lo + 90), 0, N_DEC_BANDS - 1))
        last = int(np.clip(np.floor(hi + 90), 0, N_DEC_BANDS - 1))
        return int(self.dec_index[first]), int(self.dec_index[last + 1])

    def lookup(self, ra_range, dec_range, max_magnitude=None, epoch=None):
        """Stars in a bounding box, as a StarTable.

        Only the declination bands the box touches are read from disk. ra_range may be
        (hi, lo) to wrap through RA = 0.
        """
        start, stop = self._dec_rows(dec_range)
        if stop <= start:
            from mee2024.starcat.providers import empty_table
            return empty_table(epoch=epoch or self.epoch, band=self.band)

        dec = np.asarray(self._open('dec')[start:stop])
        ra = np.asarray(self._open('ra')[start:stop])
        mag = np.asarray(self._open('mag')[start:stop])

        dec_lo, dec_hi = np.radians(min(dec_range)), np.radians(max(dec_range))
        keep = (dec >= dec_lo) & (dec <= dec_hi)
        ra_lo, ra_hi = np.radians(ra_range[0]), np.radians(ra_range[1])
        if ra_lo < ra_hi:
            keep &= (ra >= ra_lo) & (ra <= ra_hi)
        else:  # the box straddles RA = 0
            keep &= (ra >= ra_lo) | (ra <= ra_hi)
        if max_magnitude is not None:
            keep &= mag < max_magnitude

        rows = np.nonzero(keep)[0]
        columns = {}
        for attribute, _, _ in COLUMNS:
            if attribute in ('ra', 'dec', 'mag'):
                continue
            columns[attribute] = np.asarray(self._open(attribute)[start:stop])[rows]
        ids = columns.pop('ids')

        table = StarTable(ra=ra[rows], dec=dec[rows], mag=mag[rows], ids=ids,
                          epoch=self.epoch, origin=ORIGIN_GAIA, band=self.band,
                          **columns)
        return table if epoch is None else table.at_epoch(epoch)


def pack(directory, archive_path):
    """Zip a catalogue directory for distribution. Returns the archive path."""
    directory = Path(directory)
    read_manifest(directory)  # refuse to package something malformed
    archive_path = Path(archive_path)
    base = archive_path.with_suffix('') if archive_path.suffix == '.zip' else archive_path
    created = shutil.make_archive(str(base), 'zip', root_dir=str(directory))
    return Path(created)


def unpack(archive_path, directory, verify_checksums=True):
    """Extract a distributed archive and verify it against its own manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(directory))
    problems = verify(directory)
    if verify_checksums and problems:
        raise ValueError(f'extracted catalogue failed verification: {problems}')
    return directory
