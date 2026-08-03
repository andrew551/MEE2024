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

from mee2024 import events
from mee2024.starcat.table import (ORIGIN_GAIA, ORIGIN_HIPPARCOS, ORIGIN_TYCHO,
                                   StarTable, concat)


def _warn_truncated(catalogue, requested, limit):
    """Say -- on the console and on the event bus -- that a request was cut short.

    The bus carries it to the app window; the print keeps the CLI and the classic GUI
    informed. Emitted here, at the lookup itself, so no caller can bypass it.
    """
    from mee2024.starcat.download import magnitude_warning
    text = magnitude_warning(catalogue, requested, limit)
    if text:
        events.log(text, level='warning')
        print('note: ' + text)

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


#: Gaia DR3's reference epoch: every position in it is where the star was in early 2016,
#: and getting from there to the observation needs a proper motion.
GAIA_DR3_EPOCH = 2016.0


def empty_table(epoch=GAIA_DR3_EPOCH, band='G'):
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
            epoch=float(np.median(ref_epoch)) if n else GAIA_DR3_EPOCH,
            origin=ORIGIN_GAIA, band='G',
            pmra=column('pmra'), pmdec=column('pmdec'),
            parallax=column('parallax'), radial_velocity=column('radial_velocity'))
        # epoch=None leaves the positions at Gaia's own reference epoch; see
        # GaiaOfflineProvider.lookup for why a caller would want that
        return table if epoch is None else table.at_epoch(epoch)

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        if self.magnitude_limit is not None and max_magnitude > self.magnitude_limit:
            _warn_truncated(self.name, max_magnitude, self.magnitude_limit)
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


# ----------------------------------------------------------------- Hipparcos

class HipparcosProvider(CatalogueProvider):
    """Hipparcos-2, the bright-star fill.

    Gaia DR3 has no entry at all for the brightest stars -- they saturate the instrument.
    18,430 Hipparcos-2 stars have no Gaia counterpart, 2,629 of them brighter than Hp=7,
    and the bundled Tycho catalogue does not cover them either (Tycho-2 moves ~120 very
    bright stars into a separate Supplement 1 file). Hipparcos has all of them, with
    astrometry good enough for the precision fit.

    Identifiers are HIP numbers, not Gaia source_ids. Magnitudes are Gaia G estimated from
    Hp and B-V, with a robust scatter of 0.038 mag, so the band is reported as
    'G_est_from_Hp'.
    """

    name = 'hipparcos'
    is_offline = True
    #: Hipparcos is essentially complete only to about here
    COMPLETENESS_LIMIT = 9.0

    def __init__(self, directory=None):
        from mee2024.MEE2024util import resource_path
        from mee2024.starcat.store import OfflineCatalogue
        self.catalogue = OfflineCatalogue(directory or resource_path('resources/hipparcos2'))

    @classmethod
    def try_bundled(cls):
        """The bundled catalogue, or None if it was not shipped. Never raises."""
        try:
            return cls()
        except (FileNotFoundError, ValueError, OSError):
            return None

    @property
    def magnitude_limit(self):
        return self.catalogue.magnitude_limit

    def describe(self):
        return (f'{self.name} (offline, {self.catalogue.n_stars} stars, '
                f'complete to about G={self.COMPLETENESS_LIMIT})')

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        table = self.catalogue.lookup(ra_range, dec_range, max_magnitude, epoch=epoch)
        # the stored origin column is written by the builder, but be explicit
        table.origin = np.full(len(table), ORIGIN_HIPPARCOS, dtype=np.uint8)
        return table


# --------------------------------------------------------------- Gaia offline

def choose_non_overlapping(catalogues):
    """The subset of archives to read together, covering the widest magnitude range.

    Archives used to be guaranteed disjoint magnitude slices, so reading every
    installed one was right. That is no longer true: the standard archive is a
    superset of the original base + extension pair, and the compact archive is a
    subset of it. Reading overlapping archives lists every star two or three times,
    and duplicate catalogue entries do not merely waste memory -- they defeat the
    "nearest match must be twice as close as the runner-up" test that both the plate
    solver's verification and the distortion fit rely on, so *every* star is rejected
    as ambiguous and a good field silently matches nothing.

    Selection uses both ends of each archive's magnitude range: take the one reaching
    brightest, then keep adding any archive that extends the faint end, skipping those
    already covered.
    """
    ordered = sorted(catalogues, key=lambda c: (c.magnitude_min,
                                                -(c.magnitude_limit or 99)))
    chosen, covered_to = [], None
    for catalogue in ordered:
        faint = catalogue.magnitude_limit or catalogue.magnitude_min
        if covered_to is None:
            chosen.append(catalogue)
            covered_to = faint
            continue
        if faint <= covered_to + 1e-9:
            continue        # entirely inside what is already covered
        if catalogue.magnitude_min > covered_to + 1e-9:
            # a gap: the archives are not contiguous, so reading both would leave a
            # hole rather than a duplicate. Keep it -- a hole is honest, and the
            # depth warning covers the rest.
            chosen.append(catalogue)
            covered_to = faint
            continue
        chosen.append(catalogue)
        covered_to = faint
    if len(chosen) < len(catalogues):
        skipped = [c.manifest['name'] for c in catalogues if c not in chosen]
        print(f'note: ignoring {", ".join(skipped)}: already covered by '
              f'{", ".join(c.manifest["name"] for c in chosen)}')
    return chosen


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

    def __init__(self, directories, verify_checksums=False, allow_overlap=False):
        from mee2024.starcat.store import OfflineCatalogue
        if isinstance(directories, (str, bytes)) or hasattr(directories, '__fspath__'):
            directories = [directories]
        catalogues = [OfflineCatalogue(d, verify_checksums=verify_checksums)
                      for d in directories]
        if not catalogues:
            raise ValueError('at least one catalogue directory is required')
        self.catalogues = (catalogues if allow_overlap
                           else choose_non_overlapping(catalogues))

    @classmethod
    def from_installed(cls, names=None, verify_checksums=False):
        """Build from whatever catalogue archives are present on disk."""
        from mee2024.starcat import download
        releases = ([download.get_release(n) for n in names] if names
                    else download.installed_catalogues())
        # read_directory(), not directory(): the compact archive may be bundled inside
        # the program rather than installed in the user's data directory
        directories = [r.read_directory() for r in releases if r.is_installed()]
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
            _warn_truncated(self.describe(), max_magnitude, limit)
            max_magnitude = limit
        parts = [c.lookup(ra_range, dec_range, max_magnitude, epoch=None)
                 for c in self.catalogues]
        parts = [p for p in parts if len(p)]
        if not parts:
            return empty_table(epoch=GAIA_DR3_EPOCH if epoch is None else epoch)
        # each archive stores its own reference epoch; align before joining
        base_epoch = parts[0].epoch
        parts = [p if p.epoch == base_epoch else p.at_epoch(base_epoch) for p in parts]
        joined = concat(parts)
        # epoch=None means "leave the positions at the catalogue's own epoch", which is
        # what lets a caller fill in a missing proper motion *before* propagation
        return joined if epoch is None else joined.at_epoch(epoch)

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
    """Gaia, with progressively lower-priority catalogues filling the bright end.

    Gaia DR3 simply has no entry for the brightest stars, so a fill is not optional if a
    field might contain one. Fill sources are tried in order, and a star is added only
    where no already-accepted star lies within ``match_radius_arcsec``:

        1. Hipparcos-2 -- has every naked-eye star, astrometry good enough for the
           precision fit, so it is preferred.
        2. Tycho-2 -- fills V~7-9 where Hipparcos thins out. Its positions reach ~2.5
           arcsec by V=11, so it is admissible for plate solving only, and carries
           ORIGIN_TYCHO for the fit to exclude.

    Each fill source has its own magnitude ceiling: beyond it the source does more harm
    than good.
    """

    name = 'merged'

    def __init__(self, primary=None, fills=None, secondary=None,
                 bright_fill_limit=BRIGHT_FILL_LIMIT, match_radius_arcsec=2.0,
                 fill_proper_motions=True):
        """primary: the precision catalogue. fills: [(provider, magnitude ceiling), ...]
        tried in order.

        Passing ``secondary`` selects that single provider as the only fill; passing
        neither builds the default chain (Hipparcos, then Tycho).
        """
        self.primary = primary or GaiaOnlineProvider()
        if fills is None:
            if secondary is not None:
                fills = [(secondary, bright_fill_limit)]
            else:
                fills = []
                hipparcos = HipparcosProvider.try_bundled()
                if hipparcos is not None:
                    fills.append((hipparcos, HipparcosProvider.COMPLETENESS_LIMIT))
                fills.append((TychoProvider(), bright_fill_limit))
        self.fills = list(fills)
        self.match_radius_arcsec = match_radius_arcsec
        #: also lend a proper motion to primary stars that have none -- see
        #: fill_proper_motion for why the brightest stars need it
        self.fill_proper_motions = fill_proper_motions

    @property
    def is_offline(self):
        return self.primary.is_offline and all(f.is_offline for f, _ in self.fills)

    @property
    def magnitude_limit(self):
        return self.primary.magnitude_limit

    def describe(self):
        fills = ', '.join(f'{f.name}<{limit}' for f, limit in self.fills)
        return (f'{self.name} ({"offline" if self.is_offline else "online"}): '
                f'{self.primary.name} + [{fills}]')

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        result = self._primary_with_proper_motion(ra_range, dec_range, max_magnitude,
                                                  epoch)
        for provider, ceiling in self.fills:
            limit = min(ceiling, max_magnitude)
            extra = provider.lookup(ra_range, dec_range, limit, epoch)
            extra = extra.select(extra.mag < limit)
            if len(extra) == 0:
                continue
            if len(result):
                extra = extra.select(_unmatched(extra, result, self.match_radius_arcsec))
                if len(extra) == 0:
                    continue
            # magnitudes from different sources are all approximate G; label the mixture
            extra.epoch = result.epoch if len(result) else extra.epoch
            extra.band = result.band if len(result) else extra.band
            result = concat([result, extra]) if len(result) else extra
            result.band = 'G_mixed'
        return result

    def _primary_with_proper_motion(self, ra_range, dec_range, max_magnitude, epoch):
        """The primary catalogue, with missing proper motions filled before propagating.

        Order matters: the fill has to happen while the positions are still at the
        catalogue's own epoch, because propagation is what turns a missing proper motion
        into a stale position. Providers that do not offer the unpropagated form fall back
        to the plain lookup, so this can only improve on the previous behaviour.
        """
        if not self.fill_proper_motions or not self.fills:
            return self.primary.lookup(ra_range, dec_range, max_magnitude, epoch)
        try:
            native = self.primary.lookup(ra_range, dec_range, max_magnitude, epoch=None)
            if native.epoch is None:
                # the provider took epoch=None literally instead of meaning "your own
                # epoch"; without a reference epoch nothing can be propagated afterwards
                raise ValueError('provider returned a table with no epoch')
        except Exception:
            return self.primary.lookup(ra_range, dec_range, max_magnitude, epoch)
        filled = fill_proper_motion(native, self.fills, ra_range, dec_range)
        if filled:
            from mee2024 import events
            message = (f'{filled} bright star(s) had no Gaia proper motion; took one from '
                       f'the fill catalogue so they can be propagated to the observation '
                       f'epoch')
            print(message)
            events.log(message)
        return native.at_epoch(epoch)

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


#: How far apart two catalogues' positions for the same star may be, at a common epoch,
#: for the pair to be believed. Both sides are propagated to the same epoch first, so for
#: a real match this is milliarcseconds; the allowance is for a poor Hipparcos solution.
PM_FILL_RADIUS_ARCSEC = 2.0

#: Hipparcos reports Hp/V and Gaia reports G, which differ by a few tenths for ordinary
#: stars and more for very red ones. Loose enough not to reject a real pair, tight enough
#: that a chance neighbour of quite different brightness is not adopted.
PM_FILL_MAG_TOLERANCE = 2.0


def fill_proper_motion(table, fills, ra_range, dec_range,
                       radius_arcsec=PM_FILL_RADIUS_ARCSEC,
                       mag_tolerance=PM_FILL_MAG_TOLERANCE):
    """Give stars with no proper motion one from a fill catalogue, in place.

    Gaia's brightest stars often get two-parameter solutions -- a position and nothing
    else -- because they saturate its detectors: **21% of the catalogue brighter than G=4
    has no proper motion**, against 0.8% at G=10-13. Those stars then cannot be propagated
    to the observation epoch, so their positions go stale by the proper motion times the
    epoch gap, and the distortion fit throws them out as outliers. Rasalhague, the
    brightest star in its frame, missed by 1.845 arcsec after 6.6 years of unpropagated
    motion, against 0.01-0.25 arcsec for every other star of the brightest fifteen.

    Hipparcos has excellent proper motions for exactly these stars, and is already merged
    in to fill the bright end. Taking its proper motion while keeping Gaia's position is
    better than either alone: Gaia's position is good to about a milliarcsecond at its own
    epoch, and Hipparcos' proper motion adds only a few mas over a decade, where a pure
    Hipparcos position would carry thirty years of its own propagation error.

    Must be called on a table still at its **catalogue epoch**, before propagation --
    filling the motion in afterwards would leave the position where it already was.
    Returns the number of stars filled.
    """
    if len(table) == 0 or not fills:
        return 0
    missing = ~table.has_proper_motion()
    if not missing.any():
        return 0

    from sklearn.neighbors import NearestNeighbors
    targets = table.select(missing)
    target_index = np.nonzero(missing)[0]
    target_mags = targets.get_mags()
    filled = 0
    for provider, ceiling in fills:
        # Only where this source is trustworthy. Its ceiling is already the magnitude past
        # which it does more harm than good -- Tycho's positions reach ~2.5 arcsec by V=11,
        # so beyond that a 2-arcsec match radius would be picking neighbours at random.
        still = (~table.has_proper_motion()[target_index]) & (target_mags < ceiling)
        if not still.any():
            continue
        # brought to the same epoch as the table, so a real pair is milliarcseconds apart
        try:
            donor = provider.lookup(ra_range, dec_range, float(ceiling), epoch=table.epoch)
        except Exception:
            continue
        donor = donor.select(donor.has_proper_motion())
        if len(donor) == 0:
            continue
        chord, index = NearestNeighbors(n_neighbors=1).fit(
            donor.get_vectors()).kneighbors(targets.get_vectors())
        sep = np.degrees(2 * np.arcsin(np.clip(chord[:, 0] / 2, 0, 1))) * 3600
        donor_mags = donor.get_mags()[index[:, 0]]
        accept = (still & (sep <= radius_arcsec)
                  & (np.abs(donor_mags - target_mags) <= mag_tolerance))
        if not accept.any():
            continue
        rows = target_index[accept]
        donor_rows = index[accept, 0]
        table.pmra[rows] = donor.pmra[donor_rows]
        table.pmdec[rows] = donor.pmdec[donor_rows]
        # a parallax is only used to curve the propagation; take it when Gaia has none
        no_parallax = np.isnan(table.parallax[rows])
        if no_parallax.any():
            table.parallax[rows[no_parallax]] = donor.parallax[donor_rows[no_parallax]]
        table._skycoord = None            # the cached SkyCoord no longer matches
        filled += int(accept.sum())
    return filled


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


def _gaia(gaia_limit=13.0, **_):
    """The default: the offline archive plus the bright fill, or the online archive.

    'Gaia' rather than 'merged' because that is what it is to two decimal places --
    Gaia DR3, plus the ~100 stars so bright that Gaia saturates on them and records
    nothing at all (Sirius, Vega, Arcturus, Canopus...), filled from Hipparcos. The
    online archive is the fallback until an offline archive is installed, since a
    query per field costs minutes where the archive costs milliseconds.
    """
    from mee2024.starcat import download
    if download.installed_catalogues():
        return MergedProvider(primary=GaiaOfflineProvider.from_installed())
    return GaiaOnlineProvider(gaia_limit=gaia_limit)


register('gaia', _gaia)
register('gaia_online', lambda gaia_limit=13.0, **_: GaiaOnlineProvider(
    gaia_limit=gaia_limit))
# Building blocks and diagnostics. Registered so tests and advanced users can name
# them, deliberately absent from USER_CATALOGUES: Tycho and Hipparcos alone are not
# catalogues to reduce a plate against (Tycho is barred from the precision fit
# outright, see StarTable.is_precision_grade), and 'merged'/'merged_offline' are what
# 'gaia' now means.
register('tycho', lambda **_: TychoProvider())
register('hipparcos', lambda **_: HipparcosProvider())
register('gaia_offline', lambda **_: GaiaOfflineProvider.from_installed())
register('merged', lambda gaia_limit=13.0, **_: MergedProvider(
    primary=GaiaOnlineProvider(gaia_limit=gaia_limit)))
register('merged_offline', lambda **_: MergedProvider(
    primary=GaiaOfflineProvider.from_installed()))

#: (name, label, footnote) for the catalogues a user should choose between.
USER_CATALOGUES = (
    ('gaia', 'Gaia',
     'Gaia DR3 from the installed archive, plus the ~100 stars too bright for Gaia '
     'to record, filled from Hipparcos. Falls back to the online archive until an '
     'archive is installed.'),
    ('gaia_online', 'Gaia archive (online)',
     'Queries the ESA archive for every field: minutes per run, and needs a '
     'connection. Use the offline archive unless you have a reason not to.'),
)
