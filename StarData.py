import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
from MEE2024util import date_string_to_float
 
'''
remove NaNs
set anything smaller than 1e-3 arcseconds (1 mas) to a constant
including negative parallax from gaia measurement error
'''
def regularize_parallax(parallax, minimum=1):
    x = np.copy(parallax)
    x[np.isnan(x)] = 0
    x[x < minimum] = minimum
    return x

def regularize_pm(pm):
    x = np.copy(pm)
    x[np.isnan(x)] = 0
    return x

# wrapper for gaia star data
class StarData:

    # r: a gaia result object
    def __init__(self, r=None, date=None, has_pm=None):
        if r is None:
            return # make empty stardata
        self.epoch = Time(date, format='jyear', scale='tcb')
        print('epoch', self.epoch)
        self.has_pm = has_pm
        self.mags = r['phot_g_mean_mag']
        self.ids = r['source_id']
        self.pm = np.zeros((self.nstars(), 2))
        self.parallax = np.ones(self.nstars())*1e-4
        if has_pm:
            self.pm[:, 0] = r['pmra']
            self.pm[:, 1] = r['pmdec']
            self.parallax = regularize_parallax(r['parallax'])
            self.c = c = SkyCoord(ra=r['ra'],
                 dec=r['dec'],
                 distance=Distance(parallax= self.parallax * u.mas),
                 pm_ra_cosdec=regularize_pm(self.pm[:, 0]) * u.mas / u.yr,
                 pm_dec=regularize_pm(self.pm[:, 1]) * u.mas / u.yr,
                 obstime=self.epoch)
        else:
            self.c = c = SkyCoord(ra=r['ra'],
                 dec=r['dec'],
                 obstime=self.epoch)
        
        self._update_vectors()

    def nstars(self):
        return self.ids.shape[0]

    def get_ra(self):
        return self.c.ra.rad

    def get_dec(self):
        return self.c.dec.rad

    def get_ra_dec(self):
        return np.c_[self.get_ra(), self.get_dec()]

    def _update_vectors(self):
        self.vectors = np.zeros((self.ids.shape[0], 3))
        star_table = self.get_ra_dec()
        self.vectors[:, 0] = np.cos(star_table[:, 0]) * np.cos(star_table[:, 1])
        self.vectors[:, 1] = np.sin(star_table[:, 0]) * np.cos(star_table[:, 1])
        self.vectors[:, 2] = np.sin(star_table[:, 1])
    
    # return unit vectors for each star as np array
    def get_vectors(self):
        return self.vectors

    def get_mags(self):
        return self.mags

    def get_parallax(self):
        return self.parallax

    def get_pmotion(self):
        return self.pm

    # return star ids array
    def get_ids(self):
        return self.ids


    def update_epoch(self, date):
        if not self.has_pm:
            raise Exception("cannot update epoch without pm")
        self.epoch = Time(date, format='jyear', scale='tcb')
        #print('updating epoch to', self.epoch)
        self.c = self.c.apply_space_motion(self.epoch)
        self._update_vectors()

    def select_indices(self, indices):
        self.mags = self.mags[indices]
        self.vectors = self.vectors[indices, :]
        self.ids = self.ids[indices]
        self.c = self.c[indices]
        self.pm = self.pm[indices, :]
        self.parallax = self.parallax[indices]

    def get_epoch_float(self):
        # TODO: make less dodgy
        return float(str(self.epoch))#date_string_to_float(self.epoch.TimeISO())

    '''
    # TODO: fix me or delete me?
    def update_data(self, newdata):
        my_ids = self.get_ids()
        other_ids = dict(zip(newdata.get_ids(), np.arange(newdata.data.shape[0])))
        # replace data with newdata for each corresponding id
        # assume each id is present in newdata
        self.epoch = newdata.epoch
        for i in range(my_ids.shape[0]):
            j = other_ids[my_ids[i]]
            self.data[i, :] = newdata.data[j, :]
        self._update_vectors()
    '''

    def __copy__(self):
      newone = type(self)()
      newone.epch = self.epoch
      newone.mags = self.mags
      newone.vectors = self.vectors
      newone.ids = self.ids
      newone.c = self.c
      newone.pm = self.pm
      newone.parallax = self.parallax
      newone.has_pm = self.has_pm
      return newone

