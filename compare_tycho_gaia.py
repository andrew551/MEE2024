import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pandas as pd

class stardata:

    def __init__(self, mydata, flag_degrees=False):
        if flag_degrees:
            mydata[:, :2] = np.radians(mydata[:, :2])
        self.num_entries = mydata.shape[0]
        print(f'{self.num_entries=}')
        self.star_table = np.zeros((self.num_entries, 6), dtype=np.float32)
        self.star_table[:, :2] = mydata[:, :2]
        self.star_table[:, 5] = mydata[:, 2]
        self.star_table[:, 2] = np.cos(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
        self.star_table[:, 3] = np.sin(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
        self.star_table[:, 4] = np.sin(self.star_table[:, 1])

# load tycho data


data = np.load("resources/compressed_tycho2024epoch.npz")
print(data['mydata'].shape, data['mydata'].dtype)
tycho = stardata(data['mydata'])

# load gaia data

data_g = np.load('gaia_top_stars_2.npz')
gaia = stardata(data_g['radecmag'], flag_degrees=True)

# load HIP data

data_hip = pd.read_csv('D:\hipparcos-voidmain.csv/hipparcos-voidmain.csv')
print(data_hip['RAdeg'])
#print(np.nonzero(np.isnan(data_hip['RAdeg'])))
mask = ~np.isnan(data_hip['RAdeg'])
print("LEN DHIP", len(data_hip))
data_hip = data_hip[mask]
print("LEN DHIP", len(data_hip))

hip = stardata(np.c_[data_hip['RAdeg'], data_hip['DEdeg'], data_hip['VTmag']], flag_degrees=True)

# make kdtree of gaia data; check tycho
print(hip.star_table)
kdtree = KDTree(gaia.star_table[:, 2:5])
kdtree_hip = KDTree(hip.star_table[:, 2:5])

eps = np.radians(5 / 3600) # 10 arcsec

n = tycho.num_entries
n=50000
vect = np.zeros(n)
vect_hip = np.zeros(n)
mag_tycho = []
mag_gaia = []

mag_tycho_hip = []
mag_hip = []

for i in range(n):
    p = kdtree.query_ball_point(tycho.star_table[i, 2:5], eps)
    vect[i] = bool(p)
    if len(p) == 1:
        mag_tycho.append(tycho.star_table[i, 5])
        mag_gaia.append(gaia.star_table[p[0], 5])
    p = kdtree_hip.query_ball_point(tycho.star_table[i, 2:5], eps)
    vect_hip[i] = bool(p)
    if len(p) == 1:
        mag_tycho_hip.append(tycho.star_table[i, 5])
        mag_hip.append(hip.star_table[p[0], 5])
    #else:
    #    print(i, len(p))
print(np.sum(vect), '/', n)
print("hip", np.sum(vect_hip), '/', n)

plt.plot(np.cumsum(vect.astype(int)), linewidth=0.5)
plt.plot(np.cumsum(vect_hip.astype(int)), linewidth=0.5)
plt.title("Boolean Sequence (Cumsum)")
plt.show()

plt.scatter(mag_tycho, mag_gaia, s=0.2)
plt.scatter(mag_tycho_hip, mag_hip, s=0.2)

plt.title("magnitude comparison")
plt.plot([2,12], [2, 12])
plt.show()

### check kdtree gaia for hip stars; construct gaia database with (ra, dec, magG, HIP_ID)

ids = np.empty(gaia.num_entries, dtype=np.int32)
print(gaia.star_table.dtype)
count = 0
print(data_hip['HIP'])
for i in range(gaia.num_entries):
    p = kdtree_hip.query_ball_point(gaia.star_table[i, 2:5], eps)
    if len(p) == 1:
        ids[i] = data_hip['HIP'].to_numpy()[p[0]]
        count += 1
    else:
        ids[i] = -1
print(count, f'{hip.num_entries}')
print(ids.size)
np.savez_compressed('gaia_top_stars_HIP_id.npz', ids=ids, ra=gaia.star_table[:, 0], dec=gaia.star_table[:, 1], magG=gaia.star_table[:, 5])

