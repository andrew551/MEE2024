"""A catalogue provider that replays a saved Gaia response.

This is the seam the starcat provider protocol will formalise (see
docs/STARCAT_DESIGN.md §3). Because the fixture holds raw Gaia columns at their
reference epoch and propagates locally, it behaves exactly like the future GaiaOffline
provider -- and the propagation gate showed local propagation reproduces Gaia's
server-side ESDC_EPOCH_PROP_POS to 0.000 mas, so replaying a fixture is equivalent to
querying the archive.
"""

import json
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / 'data'


def load_field(name):
    return json.loads((DATA / 'fields' / f'{name}.json').read_text(encoding='utf-8'))


def load_gaia_rows(name):
    """The raw Gaia columns at their reference epoch."""
    return load_gaia_fixture(name)[0]


def load_gaia_fixture(name):
    """(rows, prop_epoch). rows carries ra_prop/dec_prop: Gaia's own propagated answer."""
    with np.load(DATA / 'gaia' / f'{name}.npz') as archive:
        return archive['rows'], float(archive['prop_epoch'])


def build_centroid_zip(field_name, path):
    """Recreate the stage-1 output zip that stage 2 consumes, from a field fixture."""
    field = load_field(field_name)
    centroids = field['centroids']
    results = {
        'MEE2024 version': 'fixture',
        'platesolved': True,
        'n_centroids': len(centroids),
        'img_shape': field['img_shape'],
        'RA': field['expected']['ra'],
        'DEC': field['expected']['dec'],
        'roll': field['expected']['roll'],
        'platescale/arcsec': field['expected']['platescale_arcsec'],
        '#frames stacked': 5,
        'source_files': f"['{field_name} fixture']",
        'starttime': field.get('source_starttime', '00000000000000'),
    }
    header = ',px,py,area (pixels),flux (noise-normed)\n'
    rows = ''.join(
        f'{i},{x},{y},{a},{f}\n'
        for i, ((y, x), a, f) in enumerate(zip(centroids, field['area'], field['flux'])))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('results.txt', json.dumps(results, indent=4))
        z.writestr('STACKED_CENTROIDS_DATA.csv', header + rows)
    return path


class FixtureCatalogue:
    """Serves a saved Gaia response, propagating positions to the requested epoch.

    Mirrors gaia_search.dbs_gaia's interface: lookup_objects(ra_range, dec_range,
    star_max_magnitude, time) -> StarData.
    """

    def __init__(self, field_name, gaia_limit=13):
        self.rows = load_gaia_rows(field_name)
        self.gaia_limit = gaia_limit
        self.calls = []

    def _star_data(self, rows, epoch):
        import astropy.units as u
        from mee2024.StarData import StarData

        table = {
            'SOURCE_ID': np.array(rows['source_id'], dtype=np.int64),
            'phot_g_mean_mag': np.array(rows['phot_g_mean_mag'], dtype=float),
            'ra': np.array(rows['ra'], dtype=float) * u.deg,
            'dec': np.array(rows['dec'], dtype=float) * u.deg,
            'pmra': np.array(rows['pmra'], dtype=float),
            'pmdec': np.array(rows['pmdec'], dtype=float),
            'parallax': np.array(rows['parallax'], dtype=float),
        }
        # build at the Gaia reference epoch, then propagate -- the offline behaviour
        ref_epoch = float(np.median(rows['ref_epoch'])) if len(rows) else 2016.0
        stardata = StarData(table, ref_epoch, True)
        if len(rows) and abs(epoch - ref_epoch) > 1e-9:
            stardata.update_epoch(epoch)
        return stardata

    def lookup_objects(self, range_ra, range_dec, star_max_magnitude=12, time=2024):
        self.calls.append((range_ra, range_dec, star_max_magnitude, time))
        if star_max_magnitude > self.gaia_limit:
            star_max_magnitude = self.gaia_limit
        rows = self.rows
        ra, dec = rows['ra'], rows['dec']
        if range_ra[0] < range_ra[1]:
            keep = (ra >= range_ra[0]) & (ra <= range_ra[1])
        else:  # wraps through RA = 0
            keep = (ra >= range_ra[0]) | (ra <= range_ra[1])
        keep &= (dec >= min(range_dec)) & (dec <= max(range_dec))
        keep &= rows['phot_g_mean_mag'] < star_max_magnitude
        return self._star_data(rows[keep], time)

    def lookup_neighbours_stub(self, startable, distance, max_mag_neighbours):
        """Stands in for gaia_search.lookup_nearby.

        The fixture only reaches G<12, so it cannot answer a mag-17 neighbour search.
        Returning an empty table means no star is flagged as a double, which is the
        correct conservative behaviour for a regression test that is not testing
        double-star rejection.
        """
        return self._star_data(self.rows[:0], 2016.0)


def install(monkeypatch, field_name, options, gaia_limit=13):
    """Point the pipeline at a fixture catalogue instead of the Gaia archive."""
    from mee2024 import database_cache, gaia_search

    catalogue = FixtureCatalogue(field_name, gaia_limit=gaia_limit)
    key = f'fixture:{field_name}'
    monkeypatch.setitem(database_cache._cache.catalogue_cache, key, catalogue)
    monkeypatch.setattr(gaia_search, 'lookup_nearby', catalogue.lookup_neighbours_stub)
    options['catalogue'] = key
    return catalogue
