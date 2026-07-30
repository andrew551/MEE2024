"""
@author: Andrew Smith
Version 23 March 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mee2024 import transforms
from sklearn.neighbors import NearestNeighbors
# noqa: F401 -- these two force PyInstaller to bundle sklearn's private extension modules
import sklearn.metrics._pairwise_distances_reduction._datasets_pair  # noqa: F401
import sklearn.metrics._pairwise_distances_reduction._middle_term_computer  # noqa: F401
import os
from mee2024.MEE2024util import output_path, date_string_to_float, _version
import json
from pathlib import Path
from mee2024 import database_cache
from mee2024 import events
import datetime
from mee2024 import distortion_polynomial
from mee2024 import gaia_search
from copy import copy
import zipfile
from mee2024 import refraction_correction
from mee2024 import platesolve_triangle
from mee2024.MEE2024util import get_bbox
import shutil
from mee2024 import gravity_sweep

def get_fitfunc(plate, target, transform_function=transforms.linear_transform, img_shape=None):
    def fitfunc(x):
        rotated = transform_function(x, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc



'''
get the error correlation of each point with it's nearest neighbour:
E(cos(theta_ij))
'''
def get_nn_correlation_error(positions, errors, options):
    nn_rs = []
    nn_corrs = []
    
    for i in range(positions.shape[0]):
        min_r = 99999
        min_corr = -13
        for j in range(positions.shape[0]):
            if i == j:
                continue
            r = np.linalg.norm(positions[i, :] - positions[j, :])
            corr_ij = np.dot(errors[i, :], errors[j, :]) / np.linalg.norm(errors[i, :]) / np.linalg.norm(errors[j, :])
            if r < min_r:
                min_corr = corr_ij
                min_r = r
        nn_rs.append(min_r)
        nn_corrs.append(min_corr)

    print(f'nearest neighbour corr={np.mean(nn_corrs)}, mean distance:{np.mean(nn_rs)}')
    return np.mean(nn_corrs), np.mean(nn_rs)
'''
todo: update using the better version in platesolve_triangle
'''
def match_centroids(other_stars_df, rough_platesolve_x, dbs, corners, image_size, lookupdate, options):
    #TODO: this will be broken if we wrap around 360 degrees
    alt, az = None, None
    stardata0 = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(lookupdate)) # convert to decimal year (approximate)
    if options['enable_corrections']:
        astrocorrect = refraction_correction.AstroCorrect()
        stardata, alt, az = astrocorrect.correct_ra_dec(stardata0, options)
    else:
        stardata = stardata0
    all_star_plate = np.array([other_stars_df['py'], other_stars_df['px']]).T - np.array([image_size[0]/2, image_size[1]/2])
    transformed_all = transforms.to_polar(transforms.linear_transform(rough_platesolve_x, all_star_plate))

    # match nearest neighbours
    candidate_stars = np.zeros((stardata.nstars(), 2))
    candidate_stars[:, 0] = np.degrees(stardata.get_dec())
    candidate_stars[:, 1] = np.degrees(stardata.get_ra())

    # find nearest two catalogue stars to each observed star
    neigh = NearestNeighbors(n_neighbors=2)

    neigh.fit(candidate_stars)
    distances, indices = neigh.kneighbors(transformed_all)
    #print(indices)
    #print(distances)

    # find nearest observed star to each catalogue star
    neigh_bar = NearestNeighbors(n_neighbors=1)

    neigh_bar.fit(transformed_all)
    distances_bar, indices_bar = neigh_bar.kneighbors(candidate_stars)
    #print(indices_bar)
    #print(distances_bar)

    # find matches, but exclude ambiguity
    # TODO fix 1-many matching bug

    # rough_match_threshhold is in arcseconds; candidate_stars/transformed_all are in degrees
    match_threshhold = options['rough_match_threshhold'] / 3600
    confusion_ratio = 2 # closest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio) # note: this distance metric is not perfect (doesn't take into account meridian etc.)
    keep = np.logical_and(keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0])) # is the nearest-neighbour relation reflexive? [this eliminates 1-to-many matching]

    if options['crop_circle']:
        radial_dist = 2 * np.linalg.norm(all_star_plate, axis=1) / np.linalg.norm(list(image_size))
        within_circle = radial_dist < options['crop_circle_thresh']
        circle_removed = np.logical_and(keep, ~within_circle)
        keep = np.logical_and(keep, within_circle)

    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]


    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations (used)')
    if options['crop_circle']:
        obs_circle_removed = transformed_all[np.nonzero(circle_removed), :][0]
        plt.scatter(obs_circle_removed[:, 1], obs_circle_removed[:, 0], marker='x', label='observations (excluded)', color='red')
    for i in range(stardata.nstars()):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(stardata.ids[i]) + '\n' + f'mag={stardata.get_mags()[i]:.2f}', (np.degrees(stardata.get_ra()[i])+0.015, np.degrees(stardata.get_dec()[i])), color='black', fontsize=5)
    plt.xlabel('RA/degrees')
    plt.ylabel('DEC/degrees')
    plt.title(f'initial rough fit (nstars={obs_matched.shape[0]})')
    plt.legend()
    if options['flag_display2']:
        plt.show()
    plt.close()
    mask_select = indices[keep_i, 0].flatten()
    stardata.select_indices(mask_select)
    plate2 = all_star_plate[keep_i, :][0]

    return stardata0, stardata, plate2, alt, az, mask_select

def _date_guess_error(guessed, from_header):
    """Signed days between a blind date guess and the FITS header date, or None.

    This is the pipeline's cheapest self-check: the guess uses only proper motions and
    knows nothing about the header, so a large disagreement means something upstream is
    wrong -- a bad plate solve, a wrong catalogue epoch, mismatched stars, or a distortion
    model absorbing the proper-motion signal.
    """
    if not guessed or not from_header:
        return None
    try:
        a = datetime.date.fromisoformat(str(guessed)[:10])
        b = datetime.date.fromisoformat(str(from_header)[:10])
    except ValueError:
        return None
    return (a - b).days


def _lookup_neighbours(dbs, stardata, cutoff_arcsec, max_mag):
    """Nearby catalogue sources, for double-star flagging.

    A starcat provider answers this itself -- the offline one from precomputed columns,
    the online one with a query. Anything else falls back to the Gaia archive.
    """
    if hasattr(dbs, 'lookup_neighbours'):
        return dbs.lookup_neighbours(stardata, cutoff_arcsec, max_mag)
    return gaia_search.lookup_nearby(stardata, cutoff_arcsec, max_mag)


def match_and_fit_distortion(path_data, options, debug_folder=None):
    starttime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    path_catalogue = options['catalogue']
    
    archive = zipfile.ZipFile(path_data, 'r')
    try:
        data = json.load(archive.open('results.txt'))
        other_stars_df = pd.read_csv(archive.open('STACKED_CENTROIDS_DATA.csv'))
    except Exception: # backwards compatibility with old format
        data = json.load(archive.open('data/results.txt'))
        other_stars_df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    other_stars_df = other_stars_df.astype({'px':float, 'py':float}) # fix datatypes
    image_size = data['img_shape']
    basename = Path(path_data).stem + data['starttime']

    output_name = f'DISTORTION_OUTPUT{starttime}__'+basename
    output_dir = Path(output_path(output_name, options))
    data_dir = output_dir / 'distortion'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    plate_solve_result = platesolve_triangle.platesolve(np.c_[other_stars_df['py'], other_stars_df['px']], image_size, dict(options, **{'flag_display':False}))
    if not plate_solve_result['success']: # failed platesolve
        raise Exception("BAD DATA - platesolve failed!")
    if plate_solve_result['mirror']:
        other_stars_df['py'], other_stars_df['px'] = other_stars_df['px'], other_stars_df['py']
        image_size = np.array([image_size[1], image_size[0]])

    initial_guess = plate_solve_result['x']
    ### now try to match other stars

    corners = transforms.to_polar(transforms.linear_transform(plate_solve_result['x'], np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    dbs = database_cache.open_catalogue(path_catalogue, gaia_limit=options['safety_limit_mag'])
    alt, az = None, None
    lookupdate = options['DEFAULT_DATE'] if options['guess_date'] else options['observation_date']
    stardata0, stardata, plate2, alt, az, mask_select = match_centroids(other_stars_df, initial_guess, dbs, corners, image_size, lookupdate, options)
    
    ### fit again 


    if options['guess_date']:
        dateguess = options['DEFAULT_DATE'] # initial guess
        dateguess = distortion_polynomial._date_guess(dateguess, initial_guess, plate2, stardata, image_size, dict(options, **{'flag_display2':False}))
        # re-get gaia database
        _, stardata, plate2, alt, az, _ = match_centroids(other_stars_df, initial_guess, dbs, corners, image_size, dateguess, dict(options, **{'flag_display2':False}))


    # now recompute matches
    
    result, plate2_corrected, _, _, _  = distortion_polynomial.do_cubic_fit(plate2, stardata, initial_guess, image_size, dict(options, **{'flag_display2':False}))

    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    print('pre-outlier removed rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)

    # compute flag:
    neigh_all = _lookup_neighbours(dbs, stardata, options['double_star_cutoff'],
                                   options['double_star_mag'])
    neigh = NearestNeighbors(n_neighbors=2)
    neigh_all_data_extra2 = np.r_[neigh_all.get_ra_dec(), np.array([[-99999,-99999], [-99999, -99999]])] # ensure at least 2 "pseudo-neighbours"
    
    neigh.fit(neigh_all_data_extra2)
    distances, indices = neigh.kneighbors(stardata.get_ra_dec())

    flag_is_double = distances[:, 1] < np.radians(options['double_star_cutoff']/3600)
    flag_missing_pm = np.isnan(stardata.get_pmotion()[:, 0])
    flag_is_outlier = errors_arcseconds >= options['distortion_fit_tol']
    flag_unexplained_outlier = np.logical_and(np.logical_and(flag_is_outlier, np.logical_not(flag_missing_pm)), np.logical_not(flag_is_double))
    print(np.sum(flag_unexplained_outlier), ' unexplained outliers')
    if options['remove_double_tab2']:
        keep_j = np.logical_and(np.logical_and(errors_arcseconds < options['distortion_fit_tol'], ~flag_is_double), ~flag_missing_pm)
    else:
        keep_j = errors_arcseconds < options['distortion_fit_tol']

    # Never let a low-precision catalogue into the fit. A merged catalogue may include
    # Tycho stars to fill the bright end for plate solving, but Tycho positions reach
    # ~2.5 arcsec by V=11 -- an order of magnitude worse than the deflection we measure.
    if hasattr(stardata, 'is_precision_grade'):
        precision_grade = stardata.is_precision_grade()
        n_excluded = int(np.sum(~precision_grade))
        if n_excluded:
            print(f'{n_excluded} star(s) excluded from the fit: catalogue not precision-grade')
        keep_j = np.logical_and(keep_j, precision_grade)

    plate2_unfiltered = plate2
    stardata_unfiltered = copy(stardata)
    plate2 = plate2[keep_j, :]
    stardata.select_indices(keep_j)
    #flag_is_double = flag_is_double[keep_j]
    #flag_missing_pm = flag_missing_pm[keep_j]
    
    print(f'{np.sum(1-keep_j)} outliers more than {options["distortion_fit_tol"]} arcseconds removed')
    # do 2nd fit with outliers removed

    if options['guess_date']:
        dateguess = distortion_polynomial._date_guess(dateguess, initial_guess, plate2, stardata, image_size, options)
        #stardata_new = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(dateguess))
        #stardata.update_data(stardata_new)
        stardata.update_epoch(date_string_to_float(dateguess))

    if options['gravity_sweep']:
        gravity_sweep_L, (result, plate2_corrected, coeff_x, coeff_y, platescale_stderror) = gravity_sweep.gravity_sweep(stardata0, plate2, initial_guess, image_size, mask_select, keep_j, starttime, basename, options)
    else:
        result, plate2_corrected, coeff_x, coeff_y, platescale_stderror = distortion_polynomial.do_cubic_fit(plate2, stardata, initial_guess, image_size, options)
    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    
    print('final rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)
    detransformed = transforms.detransform_vectors(result, stardata.get_vectors())
    px_errors = plate2_corrected-detransformed
    nn_corr, nn_r = get_nn_correlation_error(plate2, px_errors, options)
    coeff_names = distortion_polynomial.get_coeff_names(options)

    # recover errors for filtered points
    
    output_results = { 'MEE2024 version': _version(),
                       'final rms error (arcseconds)': np.degrees(np.mean(mag_errors**2)**0.5)*3600,
                       '#stars used':plate2.shape[0],
                       'observation_date':options['observation_date'] if not options['guess_date'] else dateguess,
                       'date_guessed?': options['guess_date'],
                       'observation_date_header': data.get('observation_date_header'),
                       'date_guess_error_days': _date_guess_error(
                           dateguess if options['guess_date'] else None,
                           data.get('observation_date_header')),
                       'star max magnitude':options['max_star_mag_dist'],
                       'error tolerance (as)':options['distortion_fit_tol'],
                       'platescale (arcseconds/pixel)': np.degrees(result[0])*3600,
                       'platescale_relative_uncertainty': platescale_stderror,
                       'mirror?':plate_solve_result['mirror'],
                       'RA':np.degrees(result[1]),
                       'DEC':np.degrees(result[2]),
                       'ROLL':np.degrees(result[3])-180, # TODO: clarify this dodgy +/- 180 thing
                       'rough fit threshold (arcsec)':options['rough_match_threshhold'],
                       'distortion order': options['distortionOrder'],
                       'distortion coeffs x': dict(zip(coeff_names, coeff_x)),
                       'distortion coeffs y': dict(zip(coeff_names, coeff_y)),
                       'nearest-neighbour error correlation': nn_corr,
                       'aberration/parallax correction enabled?': options['enable_corrections'],
                       'gravitational correction enabled?': options['enable_gravitational_def'],
                       'gravity sweep mode?': options['gravity_sweep'],
                       'refraction correction enabled?': options['enable_corrections_ref'],
                       'source_files':str(data['source_files']) if 'source_files' in data else 'unknown',
                       'fixed distortion order':options['distortion_fixed_coefficients'],
                       'fixed distortion reference files':str(options['distortion_reference_files']),
                       'simultaneous_deflection_and_platescale':str(options['gravity_sweep']),
                       'crop_circle':str(options['crop_circle']),
                       }
    if options['crop_circle']:
        output_results['crop_circle_thresh'] = options['crop_circle_thresh']
    additional_info = { 'observation_time (UTC)':options['observation_time'],
                        'observation_long (degrees)':options['observation_long'],
                        'observation_lat (degrees)':options['observation_lat'],
                        'observation_temp (°C)':options['observation_temp'],
                        'observation_pressure (millibars)':options['observation_pressure'],
                        'observation_humidity (0.0 to 1.0)':options['observation_humidity'],
                        'observation_height (m)':options['observation_height'],
                        'observation_wavelength (μm)':options['observation_wavelength'],
                        'observation alt (degrees)': alt,
                        'observation az (degrees)': az}
    if options['enable_corrections'] or options['enable_corrections_ref']:
        output_results.update(additional_info)

    if options['gravity_sweep']:
        output_results.update({'gravity_sweep_L(arcsec)':gravity_sweep_L})
    
    with open(data_dir / 'distortion_results.txt', 'w', encoding="utf-8") as fp:
        json.dump(output_results, fp, sort_keys=False, indent=4)

    events.emit(events.METRICS, stage='distortion',
                rms_mas=float(np.degrees(np.mean(mag_errors**2)**0.5)*3600*1000),
                n_stars=int(plate2.shape[0]), nn_corr=float(nn_corr),
                platescale=float(np.degrees(result[0])*3600),
                platescale_rel_uncertainty=float(platescale_stderror),
                distortion_order=options['distortionOrder'],
                date_guessed=bool(options['guess_date']),
                observation_date=output_results['observation_date'],
                observation_date_header=output_results['observation_date_header'],
                date_guess_error_days=output_results['date_guess_error_days'],
                ra=float(np.degrees(result[1])), dec=float(np.degrees(result[2])))

    marker_colors = ['red' if is_missing_pm else 'orange' if is_double else '#1f77b4' for (is_missing_pm, is_double)
                     in zip(flag_missing_pm[keep_j], flag_is_double[keep_j])] 

    fig, axs = plt.subplots(2, 2)
    magnitudes = stardata.get_mags()
    axs[0, 0].scatter(magnitudes, np.degrees(mag_errors)*3600, marker='+', color = marker_colors)
    axs[0, 0].set_ylabel('error (arcsec)')
    axs[0, 0].set_xlabel('magnitude\nred: missing proper motion, orange: double-star')
    axs[0, 0].grid()

    axs[0, 1].scatter(stardata.get_parallax(), np.degrees(mag_errors)*3600, marker='+', color = marker_colors)
    axs[0, 1].set_ylabel('residual error (arcsec)')
    axs[0, 1].set_xlabel('parallax (milli-arcsec)')
    axs[0, 1].grid()

    axs[1, 0].scatter(px_errors[:, 1], px_errors[:, 0], marker='+', color = marker_colors)
    axs[1, 0].set_ylabel('y-error(pixels)')
    axs[1, 0].set_xlabel('x-error(pixels)')
    axs[1, 0].grid()
    axs[1, 0].set_aspect('equal')
    radii = np.linalg.norm(plate2, axis=1)
    axs[1, 1].scatter(radii, np.degrees(mag_errors)*3600, marker='+', color = marker_colors)
    axs[1, 1].set_ylabel('error (arcsec)')
    axs[1, 1].set_xlabel('radial coordinate (pixels)')
    axs[1, 1].grid()
    fig.tight_layout()
    plt.savefig(output_dir / 'Error_graphs.png', bbox_inches="tight", dpi=600)
    if options['flag_display2']:
        plt.show()
    plt.close()

    if options.get('distortion_field_plot', True):
        field_fig = distortion_polynomial.render_distortion_field(
            coeff_x, coeff_y, image_size, options,
            platescale_arcsec=np.degrees(result[0]) * 3600,
            save_to=output_dir / 'Distortion_field.png')
        events.png_event('distortion_field', figure=field_fig)
        if options['flag_display2']:
            plt.show()
        plt.close(field_fig)

    # the surfaces and the residual-correlation map the app draws on demand. Emitted
    # regardless of the field-plot setting: it costs one basis evaluation and no figure,
    # and a frontend that ignores the event pays nothing.
    events.emit(events.ANALYSIS, **distortion_polynomial.analysis_payload(
        plate2, plate2_corrected - plate2, px_errors, coeff_x, coeff_y, image_size,
        options, platescale_arcsec=np.degrees(result[0]) * 3600))

    plate2_unfiltered_corrected = distortion_polynomial.apply_corrections(result, plate2_unfiltered, coeff_x, coeff_y, image_size, options)
    transformed_final = transforms.linear_transform(result, plate2_unfiltered_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata_unfiltered.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600

    df_identification = pd.DataFrame({'px': plate2_unfiltered[:, 1]+image_size[1]/2,
                               'py': plate2_unfiltered[:, 0]+image_size[0]/2,
                               'px_dist': plate2_unfiltered_corrected[:, 1]+image_size[1]/2,
                               'py_dist': plate2_unfiltered_corrected[:, 0]+image_size[0]/2,
                               'ID': ['gaia:'+str(_) for _ in stardata_unfiltered.ids],
                               'RA(catalog)': np.degrees(stardata_unfiltered.get_ra()),
                               'DEC(catalog)': np.degrees(stardata_unfiltered.get_dec()),
                               'RA(obs)': transforms.to_polar(transformed_final)[:, 1],
                               'DEC(obs)': transforms.to_polar(transformed_final)[:, 0],
                               'magV': stardata_unfiltered.get_mags(),
                               'error(")':errors_arcseconds,
                               'flag_is_double':flag_is_double,
                               'flag_missing_pm':flag_missing_pm,
                               'flag_is_outlier':flag_is_outlier,})
            
    df_identification.to_csv(data_dir / 'CATALOGUE_MATCHED_ERRORS.csv')
    shutil.make_archive(data_dir, 'zip', Path(data_dir))
    zipfilepath = Path(data_dir).parent / 'distortion.zip'
    final_zip = Path(output_dir).parent / f'distortion_data{starttime}__{basename}.zip'
    shutil.move(zipfilepath, final_zip)
    return final_zip

# Command-line entry points for this module live in mee2024/cli.py
