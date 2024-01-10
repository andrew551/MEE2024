"""
@author: Andrew Smith
Version 5 January 2024
"""

import math
import PySimpleGUI as sg
import sys
import json
import os
import traceback
from PIL import Image, ImageTk
import io

from MEE2024util import resource_path

def check_files(files):
    try:
        for file in files:
            f=open(file, "rb")
            f.close()
    except:
        traceback.print_exc()
        raise Exception('ERROR opening file :'+file+'!')

def interpret_UI_values(options, ui_values, no_file = False):
    options['flag_display'] = ui_values['Show graphics']
    options['delete_saturated_blob'] = ui_values['delete_saturated_blob']
    options['centroid_gaussian_subtract'] = ui_values['centroid_gaussian_subtract']
    try : 
        options['m'] = int(ui_values['-m-']) if ui_values['-m-'] else 10
    except ValueError : 
        raise Exception('invalid m value')
    try : 
        options['n'] = int(ui_values['-n-']) if ui_values['-n-'] else 30
    except ValueError : 
        raise Exception('invalid n value!')
    try : 
        options['k'] = int(ui_values['-k-']) if ui_values['-k-'] else 10
    except ValueError : 
        raise Exception('invalid k value!')
    try : 
        options['pxl_tol'] = float(ui_values['-pxl_tol-']) if ui_values['-pxl_tol-'] else 5
    except ValueError : 
        raise Exception('invalid pxl_tol value!')
    try : 
        options['d'] = int(ui_values['-d-']) if ui_values['-d-'] else 10
    except ValueError : 
        raise Exception('invalid d value!')
    try : 
        options['centroid_gaussian_thresh'] = float(ui_values['-sigma_thresh-']) if ui_values['-sigma_thresh-'] else 7
    except ValueError : 
        raise Exception('invalid sigma_thresh value!')
    try : 
        options['min_area'] = int(ui_values['-min_area-']) if ui_values['-min_area-'] else 2
    except ValueError : 
        raise Exception('invalid min_area value!')
    try : 
        options['centroid_gap_blob'] = int(ui_values['-centroid_gap_blob-']) if ui_values['-centroid_gap_blob-'] else 20
    except ValueError : 
        raise Exception('invalid centroid_gap_blob value!')
    try : 
        options['blob_radius_extra'] = int(ui_values['-blob_radius_extra-']) if ui_values['-blob_radius_extra-'] else 100
    except ValueError : 
        raise Exception('invalid blob_radius_extra value!')
    
    stack_files=ui_values['-FILE-'].split(';')
    dark_files=ui_values['-DARK-'].split(';') if ui_values['-DARK-'] else []
    flat_files=ui_values['-FLAT-'].split(';') if ui_values['-FLAT-'] else []
    options['database'] = ui_values['-DB-']
    options['output_dir'] = ui_values['output_dir']
    if options['output_dir'] and not os.path.isdir(options['output_dir']):
        raise Exception('ERROR opening output folder :'+options['output_dir'])
    if not no_file:  
        check_files(stack_files)
        check_files(dark_files)
        check_files(flat_files)
        return [stack_files, dark_files, flat_files]
        
        

# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------


def get_img_data(f, maxsize=(30, 18), first=False):
    """Generate image data using PIL
    """
    try:
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first:                     # tkinter is inactive the first time
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()
        return ImageTk.PhotoImage(img)
    except Exception:
        traceback.print_exc()
        print(f'note: error reading flag thumbnail file {f}')
        return None
    

def inputUI(options):
    popup_messages = {"no_file_error": "Error: file not entered! Please enter file(s)", "no_folder_error": "Error: Output folder not entered! Please enter folder"}
        
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))

    layout_title = [
        [sg.Text('MEE 2024 Stacker UI', font='Any 14', key='MEE 2024 Stacker UI')],
    ]

    layout_file_input = [
        [sg.Text('File(s)', size=(7, 1), key = 'File(s)'), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE-'),
         sg.FilesBrowse('Choose images to stack', key = 'Choose images to stack', file_types=(("Image Files (FIT, TIF, PNG)", "*.fit *.fts *.fits *.tif *tiff"),),initial_folder=options['workDir'])],
        [sg.Text('Dark(s)', size=(7, 1), key = 'Dark(s)'), sg.InputText(default_text='',size=(75,1),key='-DARK-'),
         sg.FilesBrowse('Choose Dark image(s)', key = 'Choose Dark image(s)', file_types=(("Image Files (FIT, TIF, PNG)", "*.fit *.fts *.fits *.tif *tiff"),),initial_folder=options['workDir'])],
        [sg.Text('Flat(s)', size=(7, 1), key = 'Flat(s)'), sg.InputText(default_text='',size=(75,1),key='-FLAT-'),
         sg.FilesBrowse('Choose Flat image(s)', key = 'Choose Flat image(s)', file_types=(("Image Files (FIT, TIF, PNG)", "*.fit *.fts *.fits *.tif *tiff"),),initial_folder=options['workDir'])],
        [sg.Text('Database', size=(7, 1), key = 'Database'), sg.InputText(default_text=options['database'],size=(75,1),key='-DB-'),
         sg.FilesBrowse('Choose Database', key = 'Choose Database', file_types=((".npz", "*.npz"),),initial_folder=options['workDir'])],
    ]

    layout_folder_output = [
        [sg.Text('Output folder (blank for same as input):', size=(50, 1), key = 'Output Folder (blank for same as input):')],
        [sg.InputText(default_text=options['output_dir'],size=(75,1),key='output_dir'),
            sg.FolderBrowse('Choose output folder', key = 'Choose output folder',initial_folder=options['output_dir'])],
    ]

    layout_base = [
    
    [sg.Checkbox('Show graphics', default=options['flag_display'], key='Show graphics')],
    [sg.Checkbox('Remove big bright object', default=options['delete_saturated_blob'], key='delete_saturated_blob')],
    [sg.Checkbox('Sensitive centroid finder mode (use if moon or sun are close)', default=options['centroid_gaussian_subtract'], key='centroid_gaussian_subtract')],
    [sg.Text('sigma_thresh [sensitive-mode]', key='sigma_thresh', size=(32,1)), sg.Input(default_text=str(options['centroid_gaussian_thresh']), key = '-sigma_thresh-', size=(8,1))],
    [sg.Text('min_area (pixels) [sensitive-mode]', key='min_area (pixels)', size=(32,1)), sg.Input(default_text=str(options['min_area']), key = '-min_area-', size=(8,1))],
    [sg.Text('Show the brightest stars in stack',size=(32,1), key='Show the brightest stars in stack'), sg.Input(default_text=str(options['d']),size=(8,1),key='-d-',enable_events=True)],
    [sg.Text('Advanced Parameters:', font=('Helvetica', 12))],
    [sg.Text('blob_radius_extra',size=(32,1), key='blob_radius_extra'), sg.Input(default_text=str(options['blob_radius_extra']),size=(8,1),key='-blob_radius_extra-',enable_events=True)],
    [sg.Text('centroid_gap_blob',size=(32,1), key='centroid_gap_blob'), sg.Input(default_text=str(options['centroid_gap_blob']),size=(8,1),key='-centroid_gap_blob-',enable_events=True)],
    [sg.Text('m_stars_fit_stack', key='m_stars_fit_stack', size=(32,1)), sg.Input(default_text=str(options['m']), key = '-m-', size=(8,1))],
    [sg.Text('n_stars_verify_stack',size=(32,1), key='n_stars_verify_stack'), sg.Input(default_text=str(options['n']),size=(8,1),key='-n-',enable_events=True)],
    [sg.Text('pixel_tolerance',size=(32,1), key='pixel_tolerance'), sg.Input(default_text=str(options['pxl_tol']),size=(8,1),key='-pxl_tol-',enable_events=True)],
    [sg.Text('k_stars_plate_solve',size=(32,1), key='k_stars_plate_solve'), sg.Input(default_text=str(options['k']),size=(8,1),key='-k-',enable_events=True)],
    [sg.Button('OK'), sg.Cancel(), sg.Push(), sg.Button("Open output folder", key='Open output folder', enable_events=True)]
    ] 

    layout = [
        layout_title + layout_file_input + layout_folder_output + layout_base    
    ]  
    
    window = sg.Window('MEE2024 V0.1.5', layout, finalize=True)
    window.BringToFront()

    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED or event=='Cancel':
            window.close()
            return None

        if event=='Open output folder':
            x = values['output_dir'].strip()
            if not x:
                x = options['workDir']
            if x and os.path.isdir(x):
                path = os.startfile(os.path.realpath(x))
            else:
                sg.Popup(popup_messages['no_folder_error'], keep_on_top=True)
                
        if event=='OK':
            def check_file(s):
                return s and not s == options['workDir']
            if check_file(values['-FILE-']):
                input_okay_flag = True
            else:
                # display pop-up file not entered
                input_okay_flag = False
                sg.Popup(popup_messages['no_file_error'], keep_on_top=True)
            if not values['output_dir'].strip():
                input_okay_flag = False
                sg.Popup(popup_messages['no_folder_error'], keep_on_top=True)
            
            if input_okay_flag:
                try:
                    files = interpret_UI_values(options, values)
                    window.close()
                    return files
                except Exception as inst:
                    traceback.print_exc()
                    sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                    
