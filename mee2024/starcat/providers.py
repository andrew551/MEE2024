"""
Catalogue providers.

One interface behind which the online Gaia archive, a downloaded offline Gaia catalogue,
the bundled Tycho catalogue and any merge of them are interchangeable:

    lookup(ra_range, dec_range, max_magnitude, epoch) -> StarTable
    lookup_neighbours(table, radius_arcsec, max_magnitude) -> StarTable

``ra_range`` is in degrees and may be given as (hi, lo) to mean "wrap through RA = 0",
matching what ``MEE2024util.get_bbox`` produces. ``epoch`` is a Julian year.
"""

import numpy as np

from mee2024.starcat.table import ORIGIN_GAIA, ORIGIN_TYCHO, StarTable, concat

# Tycho V -> approximate Gaia G. Measured median offset over the zwo3 field; the proper
# transformation needs a B-V colour, which the bundled catalogue does not carry.
# See docs/STARCAT_DESIGN.md §5.
TYCHO_V_TO_G_OFFSET = -0.15

# How faint a Tycho-only star may be and still be worth adding for plate solving.
# Tycho positions degrade to ~2.5 arcsec by V=11, so this is a hard ceiling.
BRIGHT_FILL_LIMIT = 9.0


class CatalogueProvider:
    """Interface. Subclasses must implement lookup()."""

    name = 'abstract'
    is_offline = False
    #: the faintest magnitude this provider can serve, or None for no limit
    magnitude_limit = None

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        raise NotImplementedError

    def lookup_neighbours(self, table, radius_arcsec, max_magnitude):
        """Catalogue sources close to the given stars, for double-star flagging.

        The default returns nothing, which flags no star as a double -- the correct
        conservative behaviour for a provider that cannot answer the question.
        """
        return empty_table(epoch=table.epoch, band=table.band)

    def describe(self):
        limit = 'no limit' if self.magnitude_limit is None else f'G<{self.magnitude_limit}'
        return f'{self.name} ({"offline" if self.is_offline else "online"}, {limit})'

    # ---- compatibility with the pre-refactor call site -----------------------
    # database_cache callers still say dbs.lookup_objects(...); keep that working.

    def lookup_objects(self, range_ra, range_dec, star_max_magnitude=12, time=2024):
        return self.lookup(range_ra, range_dec, star_max_magnitude, time)


def empty_table(epoch=2016.0, band='G'):
    return StarTable(ra=np.zeros(0), dec=np.zeros(0), mag=np.zeros(0),
                     ids=np.zeros(0, dtype=np.int64), epoch=epoch, band=band)


# --------------------------------------------------------------------- Tycho

def encode_tycho_id(tyc1, tyc2, tyc3):
    """Pack a TYC1-TYC2-TYC3 designation into one integer.

    TYC1 <= 9537, TYC2 <= 12121, TYC3 <= 4, so TYC1*10^6 + TYC2*10 + TYC3 is unique and
    fits comfortably in an int64.
    """
    return (np.asarray(tyc1, dtype=np.int64) * 1_000_000
            + np.asarray(tyc2, dtype=np.int64) * 10
            + np.asarray(tyc3, dtype=np.int64))


def decode_tycho_id(packed):
    packed = np.asarray(packed, dtype=np.int64)
    return packed // 1_000_000, (packed % 1_000_000) // 10, packed % 10


class TychoProvider(CatalogueProvider):
    """The bundled Tycho catalogue.

    Positions are already propagated to the epoch the npz was built at (2024) and carry
    no proper motion, so ``epoch`` is accepted and ignored -- the table's own epoch is
    reported truthfully so callers can tell.

    Position accuracy degrades from ~0.14 arcsec at V<8 to ~2.5 arcsec at V=11
    (docs/STARCAT_DESIGN.md §1.2). Suitable for plate solving, not for the precision fit.
    """

    name = 'tycho'
    is_offline = True
    magnitude_limit = 12.0
    #: the epoch the bundled npz was built for
    CATALOGUE_EPOCH = 2024.0

    def __init__(self, path=None, magnitude_to_g=True):
        from mee2024.MEE2024util import resource_path
        self.path = path or resource_path('resources/compressed_tycho2024epoch.npz')
        self.magnitude_to_g = magnitude_to_g
        self._searcher = None

    def _load(self):
        if self._searcher is None:
            from mee2024 import database_lookup2
            self._searcher = database_lookup2.database_searcher(self.path)
        return self._searcher

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        star_table, _ = self._load().lookup_objects(
            ra_range, dec_range, star_max_magnitude=max_magnitude)
        mag = star_table[:, 5].astype(np.float32)
        band = 'V'
        if self.magnitude_to_g:
            mag = mag + TYCHO_V_TO_G_OFFSET
            band = 'G_est_from_V'
        return StarTable(
            ra=star_table[:, 0], dec=star_table[:, 1], mag=mag,
            # the bundled npz drops the TYC designation, so ids are unavailable
            ids=np.zeros(star_table.shape[0], dtype=np.int64),
            epoch=self.CATALOGUE_EPOCH, origin=ORIGIN_TYCHO, band=band)


# ---------------------------------------------------------------- Gaia online

class GaiaOnlineProvider(CatalogueProvider):
    """Gaia DR3 over the network, via astroquery.

    Unlike the previous implementation this does not impose a G>3 floor, which silently
    dropped the 150 sources brighter than G=3 as well as everything with NULL photometry
    (docs/STARCAT_DESIGN.md §1.3).
    """

    name = 'gaia'
    is_offline = False

    def __init__(self, gaia_limit=13.0, include_faint_without_photometry=False):
        self.magnitude_limit = gaia_limit
        self.include_faint_without_photometry = include_faint_without_photometry

    def _query(self, ra_range, dec_range, max_magnitude):
        from astroquery.gaia import Gaia
        # ra_range may wrap through zero, in which case it must become an OR
        if ra_range[0] < ra_range[1]:
            ra_clause = f'ra BETWEEN {ra_range[0]} AND {ra_range[1]}'
        else:
            ra_clause = (f'(ra >= {ra_range[0]} OR ra <= {ra_range[1]})')
        dec_lo, dec_hi = min(dec_range), max(dec_range)
        mag_clause = f'phot_g_mean_mag < {max_magnitude}'
        if self.include_faint_without_photometry:
            mag_clause = f'({mag_clause} OR phot_g_mean_mag IS NULL)'
        query = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax, radial_velocity,
phot_g_mean_mag, ref_epoch
FROM gaiadr3.gaia_source
WHERE {ra_clause} AND dec BETWEEN {dec_lo} AND {dec_hi} AND {mag_clause}"""
        return Gaia.launch_job_async(query).get_results()

    def _to_table(self, results, epoch):
        n = len(results)
        if n == 0:
            return empty_table(epoch=epoch)

        def column(name):
            return np.array(results[name], dtype=float)

        ref_epoch = column('ref_epoch')
        table = StarTable(
            ra=np.radians(column('ra')), dec=np.radians(column('dec')),
            mag=column('phot_g_mean_mag'),
            ids=np.array(results['SOURCE_ID'], dtype=np.int64),  # never via float
            epoch=float(np.median(ref_epoch)) if n else 2016.0,
            origin=ORIGIN_GAIA, band='G',
            pmra=column('pmra'), pmdec=column('pmdec'),
            parallax=column('parallax'), radial_velocity=column('radial_velocity'))
        return table.at_epoch(epoch)

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        if self.magnitude_limit is not None and max_magnitude > self.magnitude_limit:
            print(f'note: max_magnitude reduced to {self.magnitude_limit} for safety')
            max_magnitude = self.magnitude_limit
        return self._to_table(self._query(ra_range, dec_range, max_magnitude), epoch)

    def lookup_neighbours(self, table, radius_arcsec, max_magnitude):
        """Every Gaia source within radius_arcsec of any star in the table."""
        from astroquery.gaia import Gaia
        if len(table) == 0:
            return empty_table(epoch=table.epoch)
        ra_deg = np.degrees(table.get_ra())
        dec_deg = np.degrees(table.get_dec())
        boxes = []
        for ra, dec in zip(ra_deg, dec_deg):
            # widen in RA by 1/cos(dec) so the box spans radius_arcsec on the sky
            cos_dec = max(np.cos(np.radians(dec)), 1e-6)
            dra = radius_arcsec / 3600 / cos_dec
            ddec = radius_arcsec / 3600
            boxes.append(f'(ra BETWEEN {ra - dra:.6f} AND {ra + dra:.6f} '
                         f'AND dec BETWEEN {dec - ddec:.6f} AND {dec + ddec:.6f})')
        query = ("SELECT source_id, ra, dec, pmra, pmdec, parallax, radial_velocity, "
                 "phot_g_mean_mag, ref_epoch FROM gaiadr3.gaia_source WHERE ("
                 + ' OR '.join(boxes) + f') AND phot_g_mean_mag < {max_magnitude}')
        return self._to_table(Gaia.launch_job_async(query).get_results(), table.epoch)


# --------------------------------------------------------------- Gaia offline

class GaiaOfflineProvider(CatalogueProvider):
    """A downloaded or locally built Gaia catalogue, read by memory map.

    Positions are propagated locally with astropy, which reproduces Gaia's server-side
    ESDC_EPOCH_PROP_POS to 0.000 mas (docs/STARCAT_DESIGN.md §6), so results are identical
    to the online provider's within the magnitude range the archive covers.

    Several archives may be stacked -- a G<12 base plus an optional 12<G<13 extension --
    in which case the deepest magnitude limit is reported.
    """

    name = 'gaia_offline'
    is_offline = True

    def __init__(self, directories, verify_checksums=False):
        from mee2024.starcat.store import OfflineCatalogue
        if isinstance(directories, (str, bytes)) or hasattr(directories, '__fspath__'):
            directories = [directories]
        self.catalogues = [OfflineCatalogue(d, verify_checksums=verify_checksums)
                           for d in directories]
        if not self.catalogues:
            raise ValueError('at least one catalogue directory is required')

    @classmethod
    def from_installed(cls, names=None, verify_checksums=False):
        """Build from whatever catalogue archives are present on disk."""
        from mee2024.starcat import download
        releases = ([download.get_release(n) for n in names] if names
                    else download.installed_catalogues())
        directories = [r.directory() for r in releases if r.is_installed()]
        if not directories:
            raise RuntimeError(
                'no offline catalogue is installed. '
                f'{download.status()}')
        return cls(directories, verify_checksums=verify_checksums)

    @property
    def magnitude_limit(self):
        limits = [c.magnitude_limit for c in self.catalogues if c.magnitude_limit]
        return max(limits) if limits else None

    def describe(self):
        parts = ', '.join(c.manifest['name'] for c in self.catalogues)
        return f'{self.name} (offline, G<{self.magnitude_limit}) [{parts}]'

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        limit = self.magnitude_limit
        if limit is not None and max_magnitude > limit:
            print(f'note: offline catalogue reaches only G<{limit}; '
                  f'max_magnitude reduced from {max_magnitude}')
            max_magnitude = limit
        parts = [c.lookup(ra_range, dec_range, max_magnitude, epoch=None)
                 for c in self.catalogues]
        parts = [p for p in parts if len(p)]
        if not parts:
            return empty_table(epoch=epoch)
        # each archive stores its own reference epoch; align before joining
        base_epoch = parts[0].epoch
        parts = [p if p.epoch == base_epoch else p.at_epoch(base_epoch) for p in parts]
        return concat(parts).at_epoch(epoch)

    def lookup_neighbours(self, table, radius_arcsec, max_magnitude):
        """Uses the precomputed nn_sep/nn_mag columns rather than a fresh query.

        Returns the neighbours implied by those columns, so downstream double-star
        flagging behaves as it does online without needing a mag-17 catalogue on disk.
        """
        if len(table) == 0:
            return empty_table(epoch=table.epoch, band=table.band)
        close = table.is_double(radius_arcsec)
        close &= np.where(np.isnan(table.nn_mag), False, table.nn_mag < max_magnitude)
        return table.select(close)


# --------------------------------------------------------------------- merged

class MergedProvider(CatalogueProvider):
    """Gaia, with Tycho filling in only the bright stars Gaia lacks.

    Tycho entries are kept only where no Gaia source lies within ``match_radius_arcsec``
    and the star is brighter than ``bright_fill_limit``. They carry ORIGIN_TYCHO so the
    distortion fit can exclude them -- which it must, because Tycho positions are an
    order of magnitude worse than the precision we are chasing.
    """

    name = 'merged'

    def __init__(self, primary=None, secondary=None, bright_fill_limit=BRIGHT_FILL_LIMIT,
                 match_radius_arcsec=2.0):
        self.primary = primary or GaiaOnlineProvider()
        self.secondary = secondary or TychoProvider()
        self.bright_fill_limit = bright_fill_limit
        self.match_radius_arcsec = match_radius_arcsec

    @property
    def is_offline(self):
        return self.primary.is_offline and self.secondary.is_offline

    @property
    def magnitude_limit(self):
        return self.primary.magnitude_limit

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        main = self.primary.lookup(ra_range, dec_range, max_magnitude, epoch)
        fill_limit = min(self.bright_fill_limit, max_magnitude)
        extra = self.secondary.lookup(ra_range, dec_range, fill_limit, epoch)
        extra = extra.select(extra.mag < fill_limit)
        if len(extra) == 0:
            return main
        if len(main) == 0:
            return extra
        keep = _unmatched(extra, main, self.match_radius_arcsec)
        extra = extra.select(keep)
        if len(extra) == 0:
            return main
        # concat requires a common epoch and band; Tycho has no PM so its epoch is fixed
        extra.epoch = main.epoch
        extra.band = main.band
        return concat([main, extra])

    def lookup_neighbours(self, table, radius_arcsec, max_magnitude):
        return self.primary.lookup_neighbours(table, radius_arcsec, max_magnitude)


def _unmatched(candidates, reference, radius_arcsec):
    """Boolean mask: candidates with no reference star within radius_arcsec."""
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1).fit(reference.get_vectors())
    chord, _ = neigh.kneighbors(candidates.get_vectors())
    # chord length -> angular separation
    sep_arcsec = np.degrees(2 * np.arcsin(np.clip(chord[:, 0] / 2, 0, 1))) * 3600
    return sep_arcsec > radius_arcsec


# ------------------------------------------------------------------- registry

_BUILDERS = {}


def register(name, builder):
    """Register a provider factory under a catalogue name."""
    _BUILDERS[name] = builder


def build(name, **kwargs):
    """Construct a provider by catalogue name."""
    if name not in _BUILDERS:
        raise KeyError(f'unknown catalogue {name!r}; known: {sorted(_BUILDERS)}')
    return _BUILDERS[name](**kwargs)


def known_catalogues():
    return sorted(_BUILDERS)


register('gaia', lambda gaia_limit=13.0, **_: GaiaOnlineProvider(gaia_limit=gaia_limit))
register('tycho', lambda **_: TychoProvider())
register('gaia_offline', lambda **_: GaiaOfflineProvider.from_installed())
register('merged', lambda gaia_limit=13.0, **_: MergedProvider(
    primary=GaiaOnlineProvider(gaia_limit=gaia_limit)))
register('merged_offline', lambda **_: MergedProvider(
    primary=GaiaOfflineProvider.from_installed(), secondary=TychoProvider()))
