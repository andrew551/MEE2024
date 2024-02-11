from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
import numpy as np

'''
coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
width = u.Quantity(0.1, u.deg)
height = u.Quantity(0.1, u.deg)
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)
r.pprint(max_lines=12, max_width=400)
print(r)
'''
dist_lim    = 10.0 * u.lightyear                                # Spherical radius in Light Years
dist_lim_pc = dist_lim.to(u.parsec, equivalencies=u.parallax()) # Spherical radius in Parsec

query = f"SELECT source_id, ra, dec, parallax, distance_gspphot, teff_gspphot, azero_gspphot, phot_g_mean_mag, radial_velocity \
FROM gaiadr3.gaia_source \
WHERE distance_gspphot <= {dist_lim_pc.value}\
AND ruwe <1.4"

'''
job     = Gaia.launch_job_async(query)
results = job.get_results()
print(f'Table size (rows): {len(results)}')

results['distance_lightyear'] = results['distance_gspphot'].to(u.lightyear)
results['radial_velocity_ms'] = results['radial_velocity'].to(u.meter/u.second)
print(results)
'''

def get_prop_pos(T1):
    query = f"SELECT COORD1(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})),\
COORD2(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, {T1}, ref_epoch)) \
FROM gaiadr3.gaia_source \
WHERE source_id = 4472832130942575872"#5853498713190525696"
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    print(results)
    return results[0][0], results[0][1]

def select_in_box(T1, ra_range, dec_range, max_mag):
    query = f"SELECT source_id, phot_g_mean_mag, COORD1(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})),\
COORD2(ESDC_EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, radial_velocity, ref_epoch, {T1})), parallax \
FROM gaiadr3.gaia_source \
WHERE ra BETWEEN {ra_range[0]} AND {ra_range[1]} AND \
dec BETWEEN {dec_range[0]} AND {dec_range[1]} AND \
phot_g_mean_mag BETWEEN 3 AND {max_mag}"
    print(query)
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f'Table size (rows): {len(results)}')

    print(results)
    return results

'''
ra, dec = [], []
for t in [2022.0, 2022.25, 2022.5, 2022.75, 2023, 2023.25, 2023.5]:
    rai, deci = get_prop_pos(t)
    ra.append(rai)
    dec.append(deci)
plt.scatter(ra, dec)
plt.show()
'''

gaia_limit=13
class dbs_gaia:
    def lookup_objects(self, range_ra, range_dec, star_max_magnitude=12, time=2024):
        if star_max_magnitude>gaia_limit:
            star_max_magnitude = gaia_limit # safety
            print(f'note: star_max_magnitude reduced to {gaia_limit} for safety')
        results = select_in_box(time, range_ra, range_dec, star_max_magnitude) # TODO: dynamic current epoch
        l = len(results)

        star_table = np.zeros((l, 7), dtype=float)

        star_table[:, 0] = np.radians(results['COORD1'])#[results[i][2] for i in range(l)]
        star_table[:, 1] = np.radians(results['COORD2'])#[results[i][3] for i in range(l)]
        star_table[:, 5] = results['phot_g_mean_mag']#[results[i][1] for i in range(l)]
        star_table[:, 6] = results['parallax']
        star_table[:, 2] = np.cos(star_table[:, 0]) * np.cos(star_table[:, 1])
        star_table[:, 3] = np.sin(star_table[:, 0]) * np.cos(star_table[:, 1])
        star_table[:, 4] = np.sin(star_table[:, 1])
        star_catID = results['source_id']#np.array([results[i][0] for i in range(l)])
        return star_table, star_catID
        
if __name__ == '__main__':
    ra, dec = [], []
    for t in [2022.0, 2022.25, 2022.5, 2022.75, 2023, 2023.25, 2023.5]:
        rai, deci = get_prop_pos(t)
        ra.append(rai)
        dec.append(deci)
    plt.scatter(ra, dec)
    plt.show()
    results = select_in_box(2023, (263, 265), (11.5, 13.5), 8.5)
