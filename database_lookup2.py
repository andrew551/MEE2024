# Standard imports:
from pathlib import Path
import csv
import logging
import itertools
from time import perf_counter as precision_timestamp
from datetime import datetime
from numbers import Number
import numpy as np
# external imports

import numpy as np

class database_searcher:

    def __init__(self, catalogue_path, star_max_magnitude=12, epoch_proper_motion='now', debug_folder=None):
        self._logger = logging.getLogger('database_searcher.databasesearcher')
        if str(catalogue_path).endswith('.npz'):
            data = np.load(catalogue_path)
            mydata  = data['mydata']
            self.num_entries = mydata.shape[0]
            self.star_table = np.zeros((self.num_entries, 6), dtype=np.float32)
            self.star_table[:, :2] = mydata[:, :2]
            self.star_table[:, 5] = mydata[:, 2]
            self.star_table[:, 2] = np.cos(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
            self.star_table[:, 3] = np.sin(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
            self.star_table[:, 4] = np.sin(self.star_table[:, 1])
            #self.star_catID = data['star_catID'] # leave out catID for now because its format is annoying
            self.star_catID = np.zeros((mydata.shape[0], 1)) # just zeros for catid
            return
        if not self._logger.hasHandlers():
            # Add new handlers to the logger if there are none
            self._logger.setLevel(logging.DEBUG)
            # Console handler at INFO level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # Format and add
            formatter = logging.Formatter('%(asctime)s:%(name)s-%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
            if debug_folder is not None:
                self.debug_folder = Path(debug_folder)
                # File handler at DEBUG level
                fh = logging.FileHandler(self.debug_folder / 'database_lookupDEBUG.txt')
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)
        ####
        star_max_magnitude = float(star_max_magnitude)
        if epoch_proper_motion is None or str(epoch_proper_motion).lower() == 'none':
            epoch_proper_motion = None
            self._logger.debug('Proper motions will not be considered')
        elif isinstance(epoch_proper_motion, Number):
            self._logger.debug('Use proper motion epoch as given')
        elif str(epoch_proper_motion).lower() == 'now':
            epoch_proper_motion = datetime.utcnow().year
            self._logger.debug('Proper motion epoch set to now: ' + str(epoch_proper_motion))
        else:
            raise ValueError('epoch_proper_motion value %s is forbidden' % epoch_proper_motion)
        
        self.num_entries = sum(1 for _ in open(catalogue_path))
        self.epoch_equinox = 2000
        self.pm_origin = 1991.25
        self.star_table = np.zeros((self.num_entries, 6), dtype=np.float32)
        self.star_catID = np.zeros((self.num_entries, 3), dtype=np.uint16)


        with open(catalogue_path, 'r') as star_catalog_file:
                reader = csv.reader(star_catalog_file, delimiter='|')
                incomplete_entries = 0
                for (i, entry) in enumerate(reader):
                    # Skip this entry if mag, ra, or dec are empty.
                    if entry[5].isspace() or entry[8].isspace() or entry[9].isspace():
                        incomplete_entries += 1
                        continue
                    # If propagating, skip if proper motions are empty.
                    if epoch_proper_motion != self.pm_origin \
                            and (entry[12].isspace() or entry[13].isspace()):
                        incomplete_entries += 1
                        continue
                    mag = float(entry[5])
                    if mag > star_max_magnitude:
                        continue
                    # RA/Dec in degrees at 1991.25 proper motion start.
                    alpha = float(entry[8])
                    delta = float(entry[9])
                    cos_delta = np.cos(np.deg2rad(delta))

                    mu_alpha = 0
                    mu_delta = 0
                    if epoch_proper_motion != self.pm_origin:
                        # Pick up proper motion terms. Note that the pmRA field is
                        # "proper motion in right ascension"; see
                        # https://en.wikipedia.org/wiki/Proper_motion; see also section
                        # 1.2.5 in the cdsarc.u-strasbg document cited above.

                        # The 1000/60/60 term converts milliarcseconds per year to
                        # degrees per year.
                        mu_alpha_cos_delta = float(entry[12])/1000/60/60
                        mu_delta = float(entry[13])/1000/60/60

                        # Divide the pmRA field by cos_delta to recover the RA proper
                        # motion rate. Note however that near the poles (delta near plus
                        # or minus 90 degrees) the cos_delta term goes to zero so dividing
                        # by cos_delta is problematic there.
                        # Section 1.2.9 of the cdsarc.u-strasbg document cited above
                        # outlines a change of coordinate system that can overcome
                        # this problem; we simply punt on proper motion near the poles.
                        if cos_delta > 0.1:
                            mu_alpha = mu_alpha_cos_delta / cos_delta
                        else:
                            # abs(dec) > ~84 degrees. Ignore proper motion.
                            mu_alpha = 0
                            mu_delta = 0

                    ra  = np.deg2rad(alpha + mu_alpha * (epoch_proper_motion - self.pm_origin))
                    dec = np.deg2rad(delta + mu_delta * (epoch_proper_motion - self.pm_origin))
                    self.star_table[i,:] = ([ra, dec, 0, 0, 0, mag])
                    # Find ID, depends on the database
                    # is tyc_main
                    self.star_catID[i, :] = [np.uint16(x) for x in entry[1].split()]

                if incomplete_entries:
                    self._logger.info('Skipped %i incomplete entries.' % incomplete_entries)

        # Remove entries in which RA and Dec are both zero
        # (i.e. keep entries in which either RA or Dec is non-zero)
        kept = np.logical_or(self.star_table[:, 0]!=0, self.star_table[:, 1]!=0)
        self.star_table = self.star_table[kept, :]
        self.brightness_ii = np.argsort(self.star_table[:, -1])
        self.star_table = self.star_table[self.brightness_ii, :]  # Sort by brightness
        self.num_entries = self.star_table.shape[0]
        print(self.star_table.shape, self.star_catID.shape)
        self.star_catID = self.star_catID[kept, :][self.brightness_ii, :]

        # calculate direction vectors

        self.star_table[:, 2] = np.cos(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
        self.star_table[:, 3] = np.sin(self.star_table[:, 0]) * np.cos(self.star_table[:, 1])
        self.star_table[:, 4] = np.sin(self.star_table[:, 1])


    def lookup_objects(self, range_ra, range_dec, star_max_magnitude=12):
        if range_ra is not None:
            range_ra = np.deg2rad(range_ra)
            if range_ra[0] < range_ra[1]: # Range does not cross 360deg discontinuity
                kept = np.logical_and(self.star_table[:, 0] > range_ra[0], self.star_table[:, 0] < range_ra[1])
            else:
                kept = np.logical_or(self.star_table[:, 0] > range_ra[0], self.star_table[:, 0] < range_ra[1])
            star_table = self.star_table[kept, :]
            num_entries = star_table.shape[0]
            # Trim down catalogue ID to match
            
            star_catID = self.star_catID[kept, :]
            self._logger.info('Limited to RA range ' + str(np.rad2deg(range_ra)) + ', keeping ' \
                + str(num_entries) + ' stars.')
        if range_dec is not None:
            range_dec = np.deg2rad(range_dec)
            if range_dec[0] < range_dec[1]: # Range does not cross +/-90deg discontinuity
                kept = np.logical_and(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
            else:
                kept = np.logical_or(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
            star_table = star_table[kept, :]
            num_entries = star_table.shape[0]
            # Trim down catalogue ID to match

            star_catID = star_catID[kept, :]
            self._logger.info('Limited to DEC range ' + str(np.rad2deg(range_dec)) + ', keeping ' \
                + str(num_entries) + ' stars.')
        # max magnitude
        kept = star_table[:, 5] < star_max_magnitude
        star_table = star_table[kept, :]
        num_entries = star_table.shape[0]
        # Trim down catalogue ID to match

        star_catID = star_catID[kept, :]
        self._logger.info('Limited to magnitude to ' + str(star_max_magnitude) + ', keeping ' \
            + str(num_entries) + ' stars.')
        
        return star_table, star_catID

    def save_npz(self, file):
        mydata = np.zeros((self.num_entries, 3), dtype=np.float32)
        mydata[:, :2] = self.star_table[:, :2]
        mydata[:, 2] = self.star_table[:, 5]
        np.savez_compressed(file, mydata=mydata)


if __name__ == '__main__':
    dbs = database_searcher("D:/tyc_dbase4/tyc_main.dat", debug_folder="D:/debugging", epoch_proper_motion=2024)
    print(dbs.star_table.shape)
    startable, starid = dbs.lookup_objects((331, 332), (44, 45))
    print(startable)
    print(starid)
    dbs.save_npz("D:/tyc_dbase4/compressed_tycho2024epoch.npz")
