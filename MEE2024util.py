"""
@author: Andrew Smith
Version 18 February 2024
"""

import datetime
import os
import traceback
import sys
import json

def _version():
    return 'v0.2.1'

def clearlog(path, options):
    try:
        with open(output_path(path, options), 'w') as f:
            f.write('start time: ' + str(datetime.datetime.now()) + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)

def write_complete(path, options):
    try:
        with open(output_path(path, options), 'a') as f:
            f.write('end time: ' + str(datetime.datetime.now()) + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)
        

def logme(path, options, s):
    if '_nolog' in options:
        return
    try:
        with open(output_path(path, options), 'a') as f:
            f.write(s + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)

'''
if options['output_dir'] is empty, then output there
else output same file name, but into directory in options
'''
def output_path(path, options):
    if options['output_dir'].strip() == '':
        return path
    return os.path.join(options['output_dir'], os.path.basename(path))

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini(options):
    # check for config.txt file for working directory
    print('loading config file...')
    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'MEE_config.txt')
        with open(mydir_ini, 'r', encoding="utf-8") as fp:
            options.update(json.load(fp)) # if config has missing entries keep default   
    except Exception:
        traceback.print_exc()
        print('note: error reading config file - using default parameters')


def write_ini(options):
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'MEE_config.txt')
        with open(mydir_ini, 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)

'''
convert a iso-format datestring e.g 01/02/2023 to a float (e.g. 2023.08)
'''
def date_string_to_float(x):
    return datetime.datetime.fromisoformat(x).toordinal()/365.24+1
