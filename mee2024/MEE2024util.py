"""
@author: Andrew Smith
Version 6 May 2024
"""

import datetime
import os
import traceback
import sys
import json
import logging
import numpy as np
from pathlib import Path
from platformdirs import user_data_dir, user_config_dir

def _version():
    return 'v0.6.0'

'''
if options['output_dir'] is empty, then output there
else output same file name, but into directory in options
'''
def output_path(path, options):
    if options['output_dir'].strip() == '':
        return path
    return os.path.join(options['output_dir'], os.path.basename(path))

def resource_path(relative_path):
    """
    Get absolute path to a resource.

    - Works for PyInstaller (_MEIPASS)
    - Works for pip-installed package (relative to package)
    """
    try:
        # PyInstaller
        base_path = sys._MEIPASS
        return os.path.join(base_path, relative_path)
    except AttributeError:
        # pip-installed package
        package_dir = Path(__file__).parent
        return str(package_dir / relative_path)

APP_NAME = "MEE2024"
APP_AUTHOR = "MEE2024"

def get_triangle_db_path():
    base = Path(
        user_data_dir(
            appname=APP_NAME,
            appauthor=APP_AUTHOR,  # optional but recommended
        )
    )
    db_dir = base / "TripleTrianglePlatesolveDatabase"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "TripleTriangle_pattern_data.npz"

def get_config_path():
    cfg_dir = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "MEE_config.txt"

'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini(options):
    # check for config.txt file for working directory
    print('loading config file...')
    try:
        with open(get_config_path(), 'r', encoding="utf-8") as fp:
            loaded = json.load(fp)
            if not '__version__' in loaded or not loaded['__version__'] == _version(): # update ini
                loaded['__version__'] = _version()
                loaded['rough_match_threshhold'] = 36 # reset threshhold (since it was changed from degrees to arcsec)
            options.update(loaded) # if config has missing entries keep default   
    except FileNotFoundError:
        print('note: no config file found - using default parameters')
    except Exception:
        traceback.print_exc()
        print('note: error reading config file - using default parameters')


def write_ini(options):
    try:
        print('saving config file ...')
        with open(get_config_path(), 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + get_config_path())

'''
convert a iso-format datestring e.g 01/02/2023 to a float (e.g. 2023.08)
'''
def date_string_to_float(x):
    return datetime.datetime.fromisoformat(x).toordinal()/365.24+1

def date_from_float(x):
    return datetime.datetime.fromordinal(int((x - 1) * 365.24)).date().isoformat()

def get_bbox(corners):
    def one_dim(q):
        t = (np.min(q), np.max(q))
        if t[1] - t[0] > 180:
            t = (t[1], t[0])
        return t
    return one_dim(corners[:, 1]), one_dim(corners[:, 0])

'''
logging setup
'''
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
