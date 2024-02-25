import pandas as pd
import numpy as np

# wrapper for gaia star data
class StarData:

    # source: gaia or tycho
    def __init__(self, ids, data, epoch):
        self.data = data
        self.ids = ids
        self.epoch = epoch

    def get_ra(self):
        return self.data[:, 0]

    def get_dec(self):
        return self.data[:, 1]
    
    # return unit vectors for each star as np array
    def get_vectors(self):
        return self.data[:, 2:5]

    def get_mags(self):
        return self.data[:, 5]

    def get_parallax(self):
        return self.data[:, 6]

    def get_pmotion(self):
        return self.data[:, 7:9]

    # return star ids array
    def get_ids(self):
        return self.ids

    #def update_epoch(self, new_epoch):
    #    pass

    def select_indices(self, indices):
        self.data = self.data[indices, :]
        self.ids = self.ids[indices]

    def update_data(self, newdata):
        my_ids = self.get_ids()
        other_ids = dict(zip(newdata.get_ids(), np.arange(newdata.data.shape[0])))
        # replace data with newdata for each corresponding id
        # assume each id is present in newdata
        self.epoch = newdata.epoch
        for i in range(my_ids.shape[0]):
            j = other_ids[my_ids[i]]
            self.data[i, :] = newdata.data[j, :]

    def __copy__(self):
      newone = type(self)(self.ids, self.data, self.epoch)
      return newone

