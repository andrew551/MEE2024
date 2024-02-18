"""
@author: Andrew Smith
Version 18 February 2024
"""

import datetime
import os
import traceback

def _version():
    return 'v0.2.0'

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
