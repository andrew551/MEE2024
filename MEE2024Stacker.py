# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
Version 6 May 2024

--------------------------------------------------------------
Front end of MEE2024 Stacker
--------------------------------------------------------------

"""

import math
import numpy as np

import os
import sys
import stacker_implementation
import UI_handler
#import CLI_handler
from astropy.io import fits
import cProfile
import FreeSimpleGUI as sg
import traceback
import cv2
import json
import time
from multiprocessing import freeze_support
import glob
import MEE2024util
import datetime
import database_cache
from multiprocessing import Process, Manager
import matplotlib
matplotlib.use("TkAgg") # fix exe bug

# default values for all options
options = {
    'flag_display':True,
    'flag_display2':True,
    'flag_debug':False,
    'save_dark_flat':False,
    'sensitive_mode_stack':True,
    'workDir': '',
    'workDir2':'',
    '-DARK-':'',
    '-FLAT-':'',
    'output_dir': '',
    'database':'',
    'catalogue':'gaia',
    'k':12,
    'm':30,
    'n':30,
    'd':100, # how many stacked found stars to display
    'img_edge_distance':5, # how many pixels away from edge
    'pxl_tol':10, # for stacking centroid matching
    'cutoff':100, # for stacking centroid matching, penalty saturation distance
    'delete_saturated_blob':True,
    'blob_saturation_level':100,
    'blob_radius_extra':100, # delete pixels near saturated moon/sun region
    'centroid_gap_blob':30,  # ignore centroids within this distance of saturated region + radius_extra
    'centroid_gaussian_subtract':False, # use the "sensitive mode" of custom centroid detection
    'centroid_gaussian_thresh':5, # threshhold for detecting centroids (sensitive mode)
    'min_area':4, # minimum area for found centroids (sensitive mode)
    'experimental_background_subtract':False, # use experimental "ring" kernel
    'sanity_check_centroids':True,
    'float_fits':False, # output fits files with float type
    'max_star_mag_dist':12,
    'observation_date':'2023-12-01',
    'distortion_fit_tol':1, # arcseconds tolerance
    'remove_edgy_centroids':True,
    'sigma_subtract':3,
    'distortionOrder':'cubic',
    'guess_date': False,
    'DEFAULT_DATE': '2020-01-01', # the default date for date guessing
    'double_star_cutoff': 10, # within how many arcseconds to consider near_neighbour
    'double_star_mag': 17, # max mag of double stars
    'rough_match_threshhold':36, # (in arcsec) (0.01 degrees)
    'enable_corrections':False,
    'observation_time':'',
    'observation_lat':'',
    'observation_long':'',
    'enable_corrections_ref':False,
    'enable_gravitational_def':False,
    'observation_temp':10,
    'observation_pressure':1010,
    'observation_humidity':0,
    'observation_height':0,
    'observation_wavelength':0.65,
    #'triple_triang_platesolve_patterns':(80000, 120000, 0, 700000, 0.01, 0.65, 1.7),
            # (advanced): parameters for generating triangle platesolver patterns
    'do_tetra_platesolve':False,
    'basis_type':'polynomial', # legendre (under development) or polynomial
    'distortion_reference_files':'',
    'distortion_fixed_coefficients':'None',
    'flag_display3':True,
    'background_subtraction_mode':'annular',
    'eclipse_limiting_mag':11,
    'remove_double_stars_eclipse':False,
    'safety_limit_mag':13,
    'object_centre_moon':False,
    'gravity_sweep':False,
    'limit_radial_sun_radii':False,
    'limit_radial_sun_radii_value':9,
    'crop_circle':False,
    'crop_circle_thresh':1.0,
    'remove_double_tab2':False,
    'eclipse_method':'Method 1 & 2',
}

def precheck_files(files, options, flag_write_ini=False):
    good_tasks = []
    for file in files:
        if file=='':
            print("ERROR filename empty")
            continue
        base = os.path.basename(file)
        if base == '':
            print('filename ERROR : ', file)
            continue

        # try to open the file to see if it is possible
        try:
            f=open(file, "rb")
            f.close()
        except:
            traceback.print_exc()
            print('ERROR opening file : ', file)
            continue
        
        if not good_tasks and flag_write_ini:
            # save parameters to config file if this is the first good task
            options['workDir'] = os.path.dirname(file)+"/"
            MEE2024util.write_ini(options)
        good_tasks.append(file)
    if not good_tasks and flag_write_ini:
        MEE2024util.write_ini(options) # save to config file if it never happened
    return good_tasks

def handle_files(files, options, *, flag_command_line = False):
    good_files = precheck_files(files[0], options, flag_write_ini=True)
    good_darks = precheck_files(files[1], options)
    good_flats = precheck_files(files[2], options)

    try : 
       stacker_implementation.do_stack(good_files, good_darks, good_flats, options)
    except:
        print('ERROR ENCOUNTERED')
        traceback.print_exc()
        cv2.destroyAllWindows() # ? TODO needed?
        if not flag_command_line:
            sg.popup_ok('ERROR message: ' + traceback.format_exc()) # show pop_up of error message

"""
-------------------------------------------------------------------------------------------
start of program
--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    freeze_support() # enables multiprocessing for py-2-exe
    
    database_cache.prepare_triangles()
    files = []
    # check for CLI input (unimplemented)
    if len(sys.argv)>1: 
        print('ERROR: CLI is unimplemented')
        
    if 0: #test code for performance test
        MEE2024util.read_ini(options)
        files = UI_handler.inputUI(options)
        cProfile.run('handle_files(files, options)', sort='cumtime')
    else:
        # if no command line arguments, open GUI interface
        if len(files)==0:
            # read initial parameters from config.txt file
            MEE2024util.read_ini(options)
            while True:
                newfiles = UI_handler.inputUI(options) # get files
                if newfiles is None:
                    break # end loop
                files = newfiles
                handle_files(files, options) # handle files
                files.clear()
            MEE2024util.write_ini(options)       
        else:
            handle_files(files, options, flag_command_line = True) # use inputs from CLI
    print('closing')
    # join triangles
    if database_cache._cache.prepare_process.is_alive():
        database_cache._cache.prepare_process.terminate() # terminate the prepare thread
        database_cache._cache.q._close()
