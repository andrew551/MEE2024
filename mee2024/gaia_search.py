"""
Gaia DR3 catalogue access over the network, via astroquery.

Positions are propagated to the requested epoch server-side by ESDC_EPOCH_PROP_POS, so
what comes back is already at the observation epoch.
"""

from astroquery.gaia import Gaia
import astropy.units as u
import numpy as np
from mee2024 import StarData


def get_prop_pos(T1):
    query = f"SELECT COORD1(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})),\
COORD2(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, {T1}, ref_epoch)), pmra, pmdec \
FROM gaiadr3.gaia_source \
WHERE source_id = 5853498713190525696"#  4472832130942575872"
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    print(results)
    return results[0][0], results[0][1]

def select_in_box(T1, ra_range, dec_range, max_mag):
    query = f"SELECT source_id, phot_g_mean_mag, COORD1(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})),\
COORD2(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})), parallax, pmra, pmdec, ref_epoch \
FROM gaiadr3.gaia_source \
WHERE ra BETWEEN {ra_range[0]} AND {ra_range[1]} AND \
dec BETWEEN {dec_range[0]} AND {dec_range[1]} AND \
phot_g_mean_mag BETWEEN 3 AND {max_mag}"
    print(query)
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    results.pprint(max_width=400, max_lines=30)
    return results

'''
Find every Gaia source within `distance` arcseconds of any star in `startable`,
down to magnitude `max_mag_neighbours`. Used to flag double stars.
'''
def lookup_nearby(startable, distance, max_mag_neighbours):
    query = "SELECT source_id, phot_g_mean_mag, ra, dec, ref_epoch \
FROM gaiadr3.gaia_source \
WHERE "

    def helper(ra, dec):
        # ra/dec arrive in degrees; np.cos needs radians. The 1/cos(dec) term widens the
        # RA box so that it spans `distance` arcsec on the sky at this declination.
        cos_dec = max(np.cos(np.radians(dec)), 1e-6)  # guard against the poles
        return f'(ra BETWEEN {(ra - distance/3600/cos_dec):.5f} AND {(ra + distance / 3600 / cos_dec):.5f} AND \
dec BETWEEN  {(dec - distance/3600):.5f} AND {(dec + distance / 3600):.5f})'

    p = [helper(ra, dec) for (ra, dec) in list(zip(np.degrees(startable.get_ra()), np.degrees(startable.get_dec())))]
    query += '(' + ' OR '.join(p) + ')'
    query += f' AND phot_g_mean_mag BETWEEN 3 AND {max_mag_neighbours}'
    print(query)
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    return StarData.StarData(results, 2016, False)

class dbs_gaia:

    def __init__(self, gaia_limit=13):
        self.gaia_limit=gaia_limit

    def lookup_objects(self, range_ra, range_dec, star_max_magnitude=12, time=2024):
        if star_max_magnitude>self.gaia_limit:
            star_max_magnitude = self.gaia_limit # safety
            print(f'note: star_max_magnitude reduced to {self.gaia_limit} for safety')
        results = select_in_box(time, range_ra, range_dec, star_max_magnitude)
        # COORD1/COORD2 are the epoch-propagated positions; StarData reads 'ra'/'dec'
        results['ra'] = results['COORD1'] * u.deg
        results['dec'] = results['COORD2'] * u.deg
        return StarData.StarData(results, time, True)

def select_bright(T1, max_mag):
    query = f"SELECT SOURCE_ID, phot_g_mean_mag, COORD1(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})),\
COORD2(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})) \
FROM gaiadr3.gaia_source \
WHERE phot_g_mean_mag BETWEEN -2 AND {max_mag}"
    print(query)
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    results.pprint(max_width=400, max_lines=200)
    return results
