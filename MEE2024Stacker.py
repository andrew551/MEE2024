# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
Version 6 January 2024

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
import PySimpleGUI as sg
import traceback
import cv2
import json
import time
from multiprocessing import freeze_support
import glob
import MEE2024util

options = {
    'flag_display':True,
    'save_dark_flat':False,
    'workDir': '',                  
    'output_dir': '',
    'database':'',
    'k':8,
    'm':30,
    'n':30,
    'd':100,
    'pxl_tol':10,
    'cutoff':100,
    'delete_saturated_blob':True,
    'blob_radius_extra':150,
    'centroid_gap_blob':30,
}

files = []

'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini():
    # check for config.txt file for working directory
    print('loading config file...')

    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'MEE_config.txt')
        with open(mydir_ini, 'r', encoding="utf-8") as fp:
            global options
            options.update(json.load(fp)) # if config has missing entries keep default   
    except Exception:
        print('note: error reading config file - using default parameters')


def write_ini():
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'MEE_config.txt')
        with open(mydir_ini, 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)

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
            write_ini()
        good_tasks.append(file)
    if not good_tasks and flag_write_ini:
        write_ini() # save to config file if it never happened
    return good_tasks

def handle_files(files, flag_command_line = False):
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
    
    # check for CLI input (unimplemented)
    if len(sys.argv)>1: 
        print('ERROR: CLI is unimplemented')
        
    if 0: #test code for performance test
        read_ini()
        files = UI_handler.inputUI(options)
        cProfile.run('handle_files(files, options)', sort='cumtime')
    else:
        # if no command line arguments, open GUI interface
        if len(files)==0:
            # read initial parameters from config.txt file
            read_ini()
            while True:
                newfiles = UI_handler.inputUI(options) # get files
                if newfiles is None:
                    break # end loop
                files = newfiles
                handle_files(files, options) # handle files
                files.clear()
            write_ini()       
        else:
            handle_files(files, options, flag_command_line = True) # use inputs from CLI
