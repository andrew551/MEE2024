# -*- coding: utf-8 -*-
"""
@author: Andrew Smith

--------------------------------------------------------------
Front end of MEE2024 Stacker
--------------------------------------------------------------

Entry point. With no arguments this opens the GUI, preserving the historical
behaviour of double-clicking the executable. With arguments it dispatches to the
command-line interface in mee2024.cli.

Configuration defaults live in mee2024.config, and progress reporting in
mee2024.progress, so that neither the CLI nor the tests have to import this module
(which pulls in the GUI toolkit and starts the triangle-database subprocess).
"""

import os
import sys
import traceback
from multiprocessing import freeze_support

from mee2024 import MEE2024util
from mee2024 import database_cache
from mee2024 import stacker_implementation
from mee2024.config import get_default_options
from mee2024.progress import GuiProgress

# Kept as a module-level name for backwards compatibility; the authoritative
# defaults are in mee2024.config.
options = get_default_options()


def precheck_files(files, options, flag_write_ini=False):
    good_tasks = []
    for file in files:
        if file == '':
            print("ERROR filename empty")
            continue
        base = os.path.basename(file)
        if base == '':
            print('filename ERROR : ', file)
            continue

        # try to open the file to see if it is possible
        try:
            f = open(file, "rb")
            f.close()
        except Exception:
            traceback.print_exc()
            print('ERROR opening file : ', file)
            continue

        if not good_tasks and flag_write_ini:
            # save parameters to config file if this is the first good task
            options['workDir'] = os.path.dirname(file) + "/"
            MEE2024util.write_ini(options)
        good_tasks.append(file)
    if not good_tasks and flag_write_ini:
        MEE2024util.write_ini(options)  # save to config file if it never happened
    return good_tasks


def handle_files(files, options, *, flag_command_line=False, progress=None):
    good_files = precheck_files(files[0], options, flag_write_ini=True)
    good_darks = precheck_files(files[1], options)
    good_flats = precheck_files(files[2], options)

    try:
        stacker_implementation.do_stack(good_files, good_darks, good_flats, options,
                                        progress=progress)
    except Exception:
        print('ERROR ENCOUNTERED')
        traceback.print_exc()
        if not flag_command_line:
            import FreeSimpleGUI as sg
            sg.popup_ok('ERROR message: ' + traceback.format_exc())


def run_gui():
    """The interactive GUI loop: show the window, run stage 1, repeat until cancelled."""
    import matplotlib
    matplotlib.use("TkAgg")  # fix exe bug
    from mee2024 import UI_handler

    database_cache.prepare_triangles()
    try:
        MEE2024util.read_ini(options)
        while True:
            newfiles = UI_handler.inputUI(options)
            if newfiles is None:
                break  # user cancelled: end loop
            handle_files(newfiles, options, progress=GuiProgress())
            newfiles.clear()
        MEE2024util.write_ini(options)
        print('closing')
    finally:
        database_cache.shutdown_triangles()


def main(argv=None):
    freeze_support()  # enables multiprocessing for py-2-exe
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        run_gui()
        return 0
    from mee2024 import cli
    return cli.main(argv)


"""
-------------------------------------------------------------------------------------------
start of program
--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    sys.exit(main())
