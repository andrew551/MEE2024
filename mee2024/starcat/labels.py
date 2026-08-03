"""
Readable star labels.

A 19-digit Gaia source_id is useless on a plot. This resolves an identifier to a proper
name where one exists, otherwise a HIP number, otherwise the raw id.

Implemented as **sorted key arrays plus np.searchsorted**, not a hash table: smaller (a
hash table needs 30-50% empty slack), memory-mappable, no load step, and one call resolves
a whole field's worth of identifiers at once.

    labels = LabelIndex.bundled()
    labels.label_for(table.ids, table.origin)
    -> ['Vega', 'HIP 91263', 'gaia:2097892344993257344', ...]
"""

import json
from pathlib import Path

import numpy as np

from mee2024.starcat.table import ORIGIN_GAIA, ORIGIN_HIPPARCOS, ORIGIN_TYCHO

FORMAT = 'mee2024-star-labels'
FORMAT_VERSION = 1


class LabelIndex:
    """Maps Gaia source_ids and HIP numbers to HIP numbers and proper names."""

    def __init__(self, directory):
        self.directory = Path(directory)
        manifest_path = self.directory / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'no label index at {self.directory}')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('format') != FORMAT:
            raise ValueError(f'{self.directory} is not a {FORMAT} index')

        self.gaia_id = np.load(self.directory / 'gaia_id.npy', mmap_mode='r')
        self.gaia_hip = np.load(self.directory / 'gaia_hip.npy', mmap_mode='r')
        self.hip = np.load(self.directory / 'hip.npy', mmap_mode='r')
        self.hip_name_offset = np.load(self.directory / 'hip_name_offset.npy', mmap_mode='r')
        self._names_blob = (self.directory / 'names.txt').read_bytes()
        self._named = None                 # named_stars(), built on first use
        self._named_pos_cache = None       # (epoch, positions) for names_by_position()

    @classmethod
    def bundled(cls):
        """The index shipped with the package."""
        from mee2024.MEE2024util import resource_path
        return cls(resource_path('resources/star_labels'))

    @classmethod
    def try_bundled(cls):
        """The bundled index, or None if it was not shipped. Never raises."""
        try:
            return cls.bundled()
        except (FileNotFoundError, ValueError, OSError):
            return None

    def __repr__(self):
        return (f'<LabelIndex {self.manifest["n_hip"]} HIP, '
                f'{self.manifest["n_gaia_crossmatches"]} Gaia crossmatches, '
                f'{self.manifest["n_named"]} names>')

    # ----------------------------------------------------------------- lookups

    def _search(self, keys, sorted_keys, values, missing=0):
        """Vectorised sorted-array lookup. Returns `missing` where a key is absent."""
        keys = np.atleast_1d(np.asarray(keys, dtype=np.int64))
        position = np.searchsorted(sorted_keys, keys)
        position_clipped = np.clip(position, 0, len(sorted_keys) - 1)
        found = (position < len(sorted_keys)) & (
            np.asarray(sorted_keys)[position_clipped] == keys)
        out = np.full(len(keys), missing, dtype=np.int64)
        out[found] = np.asarray(values)[position_clipped[found]]
        return out

    def hip_for(self, ids, origin=ORIGIN_GAIA):
        """HIP numbers for the given identifiers. 0 where unknown.

        A Gaia source_id is resolved through the crossmatch; a HIP id is already a HIP
        number; a Tycho id has no mapping here and yields 0.
        """
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        origin = (np.full(len(ids), origin, dtype=np.uint8)
                  if np.ndim(origin) == 0 else np.asarray(origin, dtype=np.uint8))
        out = np.zeros(len(ids), dtype=np.int64)

        is_gaia = origin == ORIGIN_GAIA
        if is_gaia.any():
            out[is_gaia] = self._search(ids[is_gaia], self.gaia_id, self.gaia_hip)
        is_hip = origin == ORIGIN_HIPPARCOS
        if is_hip.any():
            # confirm the number really is in Hipparcos before trusting it
            out[is_hip] = self._search(ids[is_hip], self.hip, self.hip)
        return out

    def name_for(self, ids, origin=ORIGIN_GAIA):
        """Proper names, or None where the star has none."""
        hip = self.hip_for(ids, origin)
        offsets = self._search(hip, self.hip, self.hip_name_offset, missing=-1)
        names = []
        for hip_number, offset in zip(hip, offsets):
            if hip_number == 0 or offset < 0:
                names.append(None)
                continue
            end = self._names_blob.find(b'\n', offset)
            names.append(self._names_blob[offset:end].decode('utf-8'))
        return names

    def named_stars(self):
        """Every star with a proper name, as (hip, name) pairs. Cached.

        Only about fifty, which is what makes resolving them by position affordable.
        """
        if getattr(self, '_named', None) is None:
            hip = np.asarray(self.hip)
            offsets = np.asarray(self.hip_name_offset)
            pairs = []
            for hip_number, offset in zip(hip, offsets):
                if offset < 0:
                    continue
                end = self._names_blob.find(b'\n', offset)
                name = self._names_blob[offset:end].decode('utf-8')
                if name:
                    pairs.append((int(hip_number), name))
            self._named = pairs
        return self._named

    def names_by_position(self, ra, dec, epoch=2024.0, radius_arcsec=10.0):
        """Proper names for stars at the given sky positions (radians), by position.

        A Gaia source_id resolves to a name through Gaia's own crossmatch to Hipparcos --
        and that crossmatch omits almost exactly the stars that *have* names. Measured on
        the bundled index: **46 of the 49 named stars cannot be reached from a Gaia id**,
        Vega, Sirius, Betelgeuse, Polaris and Arcturus among them, because Gaia struggles
        with the brightest stars and the named ones are the brightest there are.

        So the last resort is the sky itself. Named stars are few and far apart, and their
        Hipparcos positions come from the catalogue already bundled for the bright fill, so
        this is a brute-force match against about fifty candidates propagated to the
        observation epoch. Returns a list of names or None, one per input position.
        """
        out = [None] * len(np.atleast_1d(ra))
        named = self.named_stars()
        if not named:
            return out
        positions = self._named_positions(epoch)
        if positions is None:
            return out
        hip_ids, cat_ra, cat_dec = positions
        by_hip = dict(named)
        ra = np.atleast_1d(np.asarray(ra, dtype=float))
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        limit = np.radians(radius_arcsec / 3600.0)
        for i, (star_ra, star_dec) in enumerate(zip(ra, dec)):
            if not (np.isfinite(star_ra) and np.isfinite(star_dec)):
                continue
            # small-angle separation is ample at ten arcseconds
            dra = (cat_ra - star_ra) * np.cos(star_dec)
            sep = np.hypot((dra + np.pi) % (2 * np.pi) - np.pi, cat_dec - star_dec)
            nearest = int(np.argmin(sep))
            if sep[nearest] <= limit:
                out[i] = by_hip.get(int(hip_ids[nearest]))
        return out

    def _named_positions(self, epoch):
        """(hip, ra, dec) for the named stars, propagated to ``epoch``. Cached per epoch."""
        cache = getattr(self, '_named_pos_cache', None)
        if cache is not None and abs(cache[0] - epoch) < 1e-6:
            return cache[1]
        try:
            from mee2024.starcat.providers import HipparcosProvider
            provider = HipparcosProvider.try_bundled()
            if provider is None:
                return None
            table = provider.lookup((0.0, 360.0), (-90.0, 90.0),
                                    max_magnitude=99.0, epoch=epoch)
        except Exception:
            return None
        wanted = np.array([hip for hip, _ in self.named_stars()], dtype=np.int64)
        keep = np.isin(np.asarray(table.ids, dtype=np.int64), wanted)
        if not keep.any():
            return None
        chosen = table.select(keep)
        result = (np.asarray(chosen.ids, dtype=np.int64),
                  np.asarray(chosen.get_ra(), dtype=float),
                  np.asarray(chosen.get_dec(), dtype=float))
        self._named_pos_cache = (float(epoch), result)
        return result

    def label_for(self, ids, origin=ORIGIN_GAIA, prefer_hip=True):
        """The best available human-readable label for each star.

        Falls back down: proper name -> 'HIP nnnnn' -> the raw identifier. Always safe to
        call, so callers need no conditional logic.
        """
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        origin_array = (np.full(len(ids), origin, dtype=np.uint8)
                        if np.ndim(origin) == 0 else np.asarray(origin, dtype=np.uint8))
        hip = self.hip_for(ids, origin_array)
        names = self.name_for(ids, origin_array)
        out = []
        for identifier, star_origin, hip_number, name in zip(ids, origin_array, hip, names):
            if name:
                out.append(name)
            elif prefer_hip and hip_number:
                out.append(f'HIP {int(hip_number)}')
            else:
                out.append(raw_label(identifier, star_origin))
        return out


def raw_label(identifier, origin):
    """The fallback label when nothing better is known."""
    origin = int(origin)
    if origin == int(ORIGIN_HIPPARCOS):
        return f'HIP {int(identifier)}'
    if origin == int(ORIGIN_TYCHO):
        from mee2024.starcat.providers import decode_tycho_id
        tyc1, tyc2, tyc3 = decode_tycho_id(identifier)
        return f'TYC {int(tyc1)}-{int(tyc2)}-{int(tyc3)}'
    return f'gaia:{int(identifier)}'


def label_or_raw(ids, origin):
    """Convenience: labels if an index is bundled, raw identifiers otherwise."""
    index = LabelIndex.try_bundled()
    if index is None:
        origin_array = (np.full(len(np.atleast_1d(ids)), origin, dtype=np.uint8)
                        if np.ndim(origin) == 0 else np.asarray(origin, dtype=np.uint8))
        return [raw_label(i, o) for i, o in zip(np.atleast_1d(ids), origin_array)]
    return index.label_for(ids, origin)
