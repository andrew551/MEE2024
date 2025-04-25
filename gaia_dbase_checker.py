import numpy as np

data = np.load('gaia_top_stars_HIP_id.npz')

ids = data['ids']
ra = data['ra']
dec = data['dec']
mag = data['magG']

for ind in (1, 2, 5, 10, 100, 1000, 100000, 200000, 150000, 175000, 160000, 155000, 151000):
    print(ids[ind], np.degrees(ra[ind]), np.degrees(dec[ind]), mag[ind])
