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
    return 'v1.2.6'


AUTHORS = 'Andrew Smith and Douglas Smith'


def _version_tuple(text):
    """('v1.0.0') -> (1, 0, 0). Unparseable input sorts as the oldest possible version."""
    parts = []
    for piece in str(text).lstrip('vV').split('.'):
        digits = ''.join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

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

def get_data_root():
    return Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))


def get_triangle_db_path():
    db_dir = get_data_root() / "TripleTrianglePlatesolveDatabase"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "TripleTriangle_pattern_data.npz"


def get_catalogue_root():
    """Where downloaded or locally built star catalogues live."""
    catalogue_dir = get_data_root() / "catalogues"
    catalogue_dir.mkdir(parents=True, exist_ok=True)
    return catalogue_dir


def get_patterndb_root():
    """Where the v2 solver's pattern databases live, one directory per variant."""
    patterndb_dir = get_data_root() / "patterndb"
    patterndb_dir.mkdir(parents=True, exist_ok=True)
    return patterndb_dir

def get_config_path():
    cfg_dir = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "MEE_config.txt"

'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def migrate_config(loaded):
    """Apply one-off fixes to a config written by an older version.

    Keyed on the version that wrote the file, so each fix runs once and a future release
    does not silently reset settings the user meant to keep. Returns human-readable notes
    describing anything that was changed.
    """
    written_by = _version_tuple(loaded.get('__version__', 'v0.0.0'))
    notes = []

    if written_by < (1, 0, 0):
        # Until v1.0.0 the rough-match tolerance was divided by 33600 instead of 3600, so
        # the effective tolerance was 9.33x tighter than the value shown. Any setting
        # tuned against that bug is far too large now that it is fixed.
        previous = loaded.get('rough_match_threshhold')
        if previous is not None and abs(float(previous) - 36) > 1e-9:
            notes.append(f'rough_match_threshhold reset from {previous} to 36 arcsec: '
                         'it was tuned against a units bug fixed in v1.0.0')
        loaded['rough_match_threshhold'] = 36

    if written_by < (1, 1, 0):
        # v1.1.0 promotes the rebuilt plate solver to the default. Configs written
        # earlier carry platesolver='triangle' only because that was the old
        # default, not as a choice; runs without the v2 pattern database still
        # fall back to the classic solver automatically.
        if loaded.get('platesolver') == 'triangle':
            notes.append("platesolver 'triangle' -> 'v2': the rebuilt solver is the "
                         'v1.1.0 default (docs/bench/BENCH.md); set it back to '
                         "'triangle' to keep the classic solver deliberately")
            loaded['platesolver'] = 'v2'

    if written_by < (1, 2, 0):
        # v1.2.0 made 'gaia' mean the installed offline archive plus the bright fill,
        # falling back to the online archive only until one is installed. Saying so
        # once is worth it: the same setting now behaves very differently (and much
        # faster), and anyone who really wants the archive queried per field needs
        # to know the new name for it.
        if loaded.get('catalogue') == 'gaia':
            notes.append("catalogue 'gaia' now reads the offline archive when one is "
                         "installed (milliseconds per field instead of minutes); "
                         "choose 'gaia_online' to query the ESA archive every run")

    loaded['__version__'] = _version()
    return notes


def read_ini(options, path=None):
    # check for config.txt file for working directory
    print('loading config file...')
    try:
        with open(path or get_config_path(), 'r', encoding="utf-8") as fp:
            loaded = json.load(fp)
            for note in migrate_config(loaded):
                print('config migration: ' + note)
            options.update(loaded) # if config has missing entries keep default
    except FileNotFoundError:
        print('note: no config file found - using default parameters')
    except Exception:
        traceback.print_exc()
        print('note: error reading config file - using default parameters')


def write_ini(options, path=None):
    try:
        print('saving config file ...')
        with open(path or get_config_path(), 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + str(path or get_config_path()))

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
