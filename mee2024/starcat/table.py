"""
StarTable -- the one star-data representation.

Replaces two incompatible predecessors: the raw ``(N, 6)`` float array that
``database_lookup2`` returned, and ``StarData``, which wrapped a Gaia result table and
carried a single table-level ``has_pm`` flag.

Design notes (docs/STARCAT_DESIGN.md §2), and the reasons they matter:

* **Columnar numpy is the source of truth**, not an astropy ``Table`` or ``SkyCoord``.
  ``SkyCoord`` construction is expensive, cannot be memory-mapped, and cannot be sliced
  cheaply -- yet slicing happens repeatedly and the offline catalogue needs mmap. A
  ``SkyCoord`` is built lazily, only when epoch propagation or AltAz conversion needs one.
* **float64 for ra/dec.** float32 resolves ~0.1 arcsec at these coordinates, which is
  larger than the signal being measured.
* **NaN means "unknown", per star.** 0.8% of Gaia G<12 sources have no proper motion, and
  a merged Gaia+Tycho table has both kinds at once, so a table-level flag cannot work.
* **Per-star ``origin``.** Bundled Tycho positions degrade to ~2.5 arcsec by V=11, so
  Tycho stars are admissible for plate solving but must be excluded from the precision
  fit. That filter needs per-star provenance.
* **Operations return new tables.** The old in-place ``update_epoch`` combined with a
  ``__copy__`` that shared array references was an aliasing trap.
"""

import numpy as np

ORIGIN_GAIA = np.uint8(0)
ORIGIN_TYCHO = np.uint8(1)

ORIGIN_NAMES = {int(ORIGIN_GAIA): 'gaia', int(ORIGIN_TYCHO): 'tycho'}

# Bands a magnitude column may be expressed in. 'G_est_from_V' marks a Tycho V magnitude
# transformed to an approximate Gaia G, so nothing mistakes it for a measured G.
BANDS = ('G', 'V', 'G_est_from_V')

_COLUMNS = {
    'ra': np.float64, 'dec': np.float64,
    'mag': np.float32,
    'ids': np.int64,
    'origin': np.uint8,
    'pmra': np.float32, 'pmdec': np.float32,
    'parallax': np.float32, 'radial_velocity': np.float32,
    'nn_sep': np.float32, 'nn_mag': np.float32,
}

# columns that are optional; absent ones are filled with NaN
_OPTIONAL = ('pmra', 'pmdec', 'parallax', 'radial_velocity', 'nn_sep', 'nn_mag')


class StarTable:
    """A set of catalogue stars with positions valid at ``epoch``.

    ra, dec        radians, ICRS, at self.epoch
    mag            magnitude in self.band
    ids            int64 identifier (Gaia source_id, or an encoded Tycho designation)
    origin         ORIGIN_GAIA or ORIGIN_TYCHO, per star
    pmra, pmdec    mas/yr, pmra already includes cos(dec); NaN if unknown
    parallax       mas; NaN if unknown
    radial_velocity km/s; NaN if unknown
    nn_sep, nn_mag separation in arcsec to the nearest catalogue neighbour and its
                   magnitude; NaN if never computed
    """

    __slots__ = ('ra', 'dec', 'mag', 'ids', 'origin', 'pmra', 'pmdec', 'parallax',
                 'radial_velocity', 'nn_sep', 'nn_mag', 'epoch', 'band', '_vectors', '_skycoord')

    def __init__(self, ra, dec, mag, ids, epoch, origin=ORIGIN_GAIA, band='G', **optional):
        self.ra = np.ascontiguousarray(ra, dtype=np.float64)
        self.dec = np.ascontiguousarray(dec, dtype=np.float64)
        n = self.ra.shape[0]
        if self.dec.shape[0] != n:
            raise ValueError('ra and dec must be the same length')
        if band not in BANDS:
            raise ValueError(f'band must be one of {BANDS}, got {band!r}')

        self.mag = np.ascontiguousarray(mag, dtype=np.float32)
        self.ids = np.ascontiguousarray(ids, dtype=np.int64)
        self.origin = (np.full(n, origin, dtype=np.uint8)
                       if np.isscalar(origin) or np.ndim(origin) == 0
                       else np.ascontiguousarray(origin, dtype=np.uint8))
        for name in _OPTIONAL:
            value = optional.pop(name, None)
            arr = (np.full(n, np.nan, dtype=_COLUMNS[name]) if value is None
                   else np.ascontiguousarray(value, dtype=_COLUMNS[name]))
            if arr.shape[0] != n:
                raise ValueError(f'{name} has length {arr.shape[0]}, expected {n}')
            setattr(self, name, arr)
        if optional:
            raise TypeError(f'unexpected columns: {sorted(optional)}')

        self.epoch = float(epoch)
        self.band = band
        self._vectors = None
        self._skycoord = None

    # ------------------------------------------------------------------ basics

    def __len__(self):
        return self.ra.shape[0]

    def nstars(self):
        return self.ra.shape[0]

    def __repr__(self):
        origins = ', '.join(f'{ORIGIN_NAMES[int(o)]}={int(np.sum(self.origin == o))}'
                            for o in np.unique(self.origin))
        return (f'<StarTable {len(self)} stars, epoch={self.epoch:.4f}, '
                f'band={self.band}, {origins}>')

    # -------------------------------------------------------------- accessors
    # Names kept identical to the old StarData so callers need not change.

    def get_ra(self):
        return self.ra

    def get_dec(self):
        return self.dec

    def get_ra_dec(self):
        return np.c_[self.ra, self.dec]

    def get_mags(self):
        return self.mag

    def get_ids(self):
        return self.ids

    def get_parallax(self):
        return self.parallax

    def get_pmotion(self):
        """(N, 2) array of (pmra, pmdec) in mas/yr. NaN where unknown."""
        return np.c_[self.pmra, self.pmdec]

    def get_epoch_float(self):
        return self.epoch

    def get_vectors(self):
        """(N, 3) ICRS unit vectors, computed once and cached."""
        if self._vectors is None:
            cos_dec = np.cos(self.dec)
            self._vectors = np.c_[cos_dec * np.cos(self.ra),
                                  cos_dec * np.sin(self.ra),
                                  np.sin(self.dec)]
        return self._vectors

    # ------------------------------------------------------------------ flags

    def has_proper_motion(self):
        """Per star: is a proper motion known?"""
        return ~(np.isnan(self.pmra) | np.isnan(self.pmdec))

    def is_gaia(self):
        return self.origin == ORIGIN_GAIA

    def is_double(self, cutoff_arcsec):
        """Per star: is there a catalogue neighbour within cutoff_arcsec?

        False where nn_sep was never computed, so an absent neighbour column means
        "nothing flagged" rather than "everything flagged".
        """
        return np.where(np.isnan(self.nn_sep), False, self.nn_sep < cutoff_arcsec)

    # ------------------------------------------------------------- operations

    def select(self, index):
        """A new table containing only the selected stars.

        index may be a boolean mask, integer indices, or a slice.
        """
        if not isinstance(index, slice):
            index = np.asarray(index)
        out = StarTable.__new__(StarTable)
        for name in ('ra', 'dec', 'mag', 'ids', 'origin', 'pmra', 'pmdec',
                     'parallax', 'radial_velocity', 'nn_sep', 'nn_mag'):
            source = getattr(self, name)
            value = source[index]
            # a slice yields a view; fancy/boolean indexing already copies. Never let the
            # result alias the parent, or mutating one table would silently change another.
            if np.may_share_memory(value, source):
                value = value.copy()
            setattr(out, name, np.ascontiguousarray(value))
        out.epoch = self.epoch
        out.band = self.band
        out._vectors = None
        out._skycoord = None
        return out

    def brightest(self, n):
        """A new table with the n brightest stars, brightest first."""
        return self.select(np.argsort(self.mag, kind='stable')[:n])

    def sorted_by_dec(self):
        return self.select(np.argsort(self.dec, kind='stable'))

    def within_box(self, ra_range, dec_range, max_magnitude=None):
        """Stars inside a bounding box. ra_range=(hi, lo) means wrap through RA=0."""
        ra_lo, ra_hi = np.radians(ra_range[0]), np.radians(ra_range[1])
        if ra_lo < ra_hi:
            keep = (self.ra >= ra_lo) & (self.ra <= ra_hi)
        else:  # the box straddles RA = 0
            keep = (self.ra >= ra_lo) | (self.ra <= ra_hi)
        dec_lo, dec_hi = np.radians(min(dec_range)), np.radians(max(dec_range))
        keep &= (self.dec >= dec_lo) & (self.dec <= dec_hi)
        if max_magnitude is not None:
            keep &= self.mag < max_magnitude
        return self.select(keep)

    # ------------------------------------------------------ epoch propagation

    def skycoord(self, parallax_floor_mas=1.0):
        """An astropy SkyCoord with proper motion attached, built lazily.

        parallax_floor_mas clamps parallax from below, matching what StarData did. The
        propagation gate showed this makes no measurable difference (0.000 mas), but it
        keeps distances finite for stars with zero or negative measured parallax.
        """
        if self._skycoord is not None:
            return self._skycoord
        import astropy.units as u
        from astropy.coordinates import SkyCoord, Distance
        from astropy.time import Time

        parallax = np.array(self.parallax, dtype=float)
        parallax[np.isnan(parallax)] = 0.0
        parallax[parallax < parallax_floor_mas] = parallax_floor_mas
        pmra = np.nan_to_num(np.array(self.pmra, dtype=float))
        pmdec = np.nan_to_num(np.array(self.pmdec, dtype=float))
        rv = np.nan_to_num(np.array(self.radial_velocity, dtype=float))

        self._skycoord = SkyCoord(
            ra=self.ra * u.rad, dec=self.dec * u.rad,
            distance=Distance(parallax=parallax * u.mas),
            pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
            radial_velocity=rv * u.km / u.s,
            obstime=Time(self.epoch, format='jyear', scale='tcb'))
        return self._skycoord

    def at_epoch(self, epoch):
        """A new table with positions propagated to ``epoch`` (a Julian year).

        Uses astropy apply_space_motion, which the propagation gate showed reproduces
        Gaia's server-side ESDC_EPOCH_PROP_POS exactly (docs/STARCAT_DESIGN.md §6).
        Stars with no proper motion keep their catalogue position.
        """
        epoch = float(epoch)
        if abs(epoch - self.epoch) < 1e-9 or len(self) == 0:
            out = self.select(slice(None))
            out.epoch = epoch
            return out
        from astropy.time import Time

        moved = self.skycoord().apply_space_motion(
            Time(epoch, format='jyear', scale='tcb'))
        out = self.select(slice(None))
        known = self.has_proper_motion()
        out.ra = np.where(known, moved.ra.rad, self.ra)
        out.dec = np.where(known, moved.dec.rad, self.dec)
        out.epoch = epoch
        out._vectors = None
        out._skycoord = None
        return out

    # ----------------------------------------------- deprecated in-place forms
    # Kept so distortion_fitter / gravity_sweep keep working during the migration.

    def select_indices(self, indices):
        """In-place equivalent of select(); prefer select()."""
        other = self.select(indices)
        for name in StarTable.__slots__:
            if name not in ('epoch', 'band'):
                setattr(self, name, getattr(other, name))
        return self

    def update_epoch(self, epoch):
        """In-place equivalent of at_epoch(); prefer at_epoch()."""
        other = self.at_epoch(epoch)
        for name in StarTable.__slots__:
            setattr(self, name, getattr(other, name))
        return self

    def __copy__(self):
        return self.select(slice(None))


def concat(tables):
    """Join tables that share an epoch and band."""
    tables = [t for t in tables if len(t)]
    if not tables:
        raise ValueError('nothing to concatenate')
    if len({t.band for t in tables}) > 1:
        raise ValueError(f'cannot concatenate mixed bands: {[t.band for t in tables]}')
    epochs = {round(t.epoch, 9) for t in tables}
    if len(epochs) > 1:
        raise ValueError(f'cannot concatenate mixed epochs: {sorted(epochs)} '
                         '-- call at_epoch() on each first')
    out = StarTable.__new__(StarTable)
    for name in ('ra', 'dec', 'mag', 'ids', 'origin', 'pmra', 'pmdec',
                 'parallax', 'radial_velocity', 'nn_sep', 'nn_mag'):
        setattr(out, name, np.concatenate([getattr(t, name) for t in tables]))
    out.epoch = tables[0].epoch
    out.band = tables[0].band
    out._vectors = None
    out._skycoord = None
    return out
