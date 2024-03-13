import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.spatial.transform import Rotation
import transforms
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics._pairwise_distances_reduction._datasets_pair
import sklearn.metrics._pairwise_distances_reduction._middle_term_computer
import database_lookup2
import os
from MEE2024util import output_path, date_string_to_float
import json
from pathlib import Path
import database_cache
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import distortion_cubic
import gaia_search
from copy import copy
import zipfile
import refraction_correction
import platesolve_triangle
from MEE2024util import get_bbox

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
    stardata = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(lookupdate)) # convert to decimal year (approximate)
    if options['enable_corrections']:
        astrocorrect = refraction_correction.AstroCorrect()
        stardata, alt, az = astrocorrect.correct_ra_dec(stardata, options)

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

    match_threshhold = options['rough_match_threshhold'] # in degrees
    confusion_ratio = 2 # cloest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio) # note: this distance metric is not perfect (doesn't take into account meridian etc.)
    keep = np.logical_and(keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0])) # is the nearest-neighbour relation reflexive? [this eliminates 1-to-many matching]
    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]
    
    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations')
    for i in range(stardata.nstars()):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(stardata.ids[i]), (np.degrees(stardata.get_ra()[i]), np.degrees(stardata.get_dec()[i])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('initial rough fit')
    plt.legend()
    if options['flag_display2']:
        plt.show()
    plt.close()

    stardata.select_indices(indices[keep_i, 0].flatten())
    plate2 = all_star_plate[keep_i, :][0]

    return stardata, plate2, alt, az

def match_and_fit_distortion(path_data, options, debug_folder=None):    
    path_catalogue = options['catalogue']
    
    #data = np.load(path_data,allow_pickle=True) # TODO: can we avoid using pickle? How about using shutil.make_archive?


    archive = zipfile.ZipFile(path_data, 'r')

    data = json.load(archive.open('data/results.txt'))
    image_size = data['img_shape']
    other_stars_df = pd.read_csv(archive.open('data/STACKED_CENTROIDS_DATA.csv'))
    other_stars_df = other_stars_df.astype({'px':float, 'py':float}) # fix datatypes

    basename = Path(path_data).stem + data['starttime']
    
    '''
    if not data['platesolved']: # need initial platesolve
        raise Exception("BAD DATA - Data did not have platesolve included!")
    df_id = pd.read_csv(archive.open('data/STACKED_CENTROIDS_MATCHED_ID.csv'))
    
    #other_stars_df = pd.DataFrame(data=data['detection_arr'],    # values
    #                     columns=data['detection_arr_cols'])
    
    
    #df_id = pd.DataFrame(data=data['identification_arr'],    # values
    #                     columns=data['identification_arr_cols'])
    df_id = df_id.astype({'RA':float, 'DEC':float, 'px':float, 'py':float, 'magV':float}) # fix datatypes

    df_id['vx'] = np.cos(np.radians(df_id['DEC'])) * np.cos(np.radians(df_id['RA']))
    df_id['vy'] = np.cos(np.radians(df_id['DEC'])) * np.sin(np.radians(df_id['RA']))
    df_id['vz'] = np.sin(np.radians(df_id['DEC']))
    
    print(df_id.to_string())
    target = np.array([df_id['vx'], df_id['vy'], df_id['vz']]).T
    plate = np.array([df_id['py'], df_id['px']]).T - np.array([image_size[0]/2, image_size[1]/2])
    f = get_fitfunc(plate, target)

    # note: negative scale == rotation by 180. Do we always want this +180 hack?
    print('ra/dec guess:', data['RA'], data['DEC'])
    initial_guess = tuple(np.radians([data['platescale'], data['RA'], data['DEC'], data['roll']+180]))
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate, target), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)

    resi = transforms.to_polar(transforms.linear_transform(initial_guess, plate))
    resv = transforms.to_polar(transforms.linear_transform(result.x, plate))
    orig = transforms.to_polar(target)
    '''
    plate_solve_result = platesolve_triangle.platesolve(np.c_[other_stars_df['py'], other_stars_df['px']], image_size, options)

    if not plate_solve_result['success']: # failed platesolve
        raise Exception("BAD DATA - platesolve failed!")

    initial_guess = plate_solve_result['x']
    ### now try to match other stars

    corners = transforms.to_polar(transforms.linear_transform(plate_solve_result['x'], np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    dbs = database_cache.open_catalogue(path_catalogue)
    alt, az = None, None
    lookupdate = options['DEFAULT_DATE'] if options['guess_date'] else options['observation_date']
    stardata, plate2, alt, az = match_centroids(other_stars_df, initial_guess, dbs, corners, image_size, lookupdate, options)
    
    ### fit again 


    if options['guess_date']:
        dateguess = options['DEFAULT_DATE'] # initial guess
        dateguess = distortion_cubic._date_guess(dateguess, initial_guess, plate2, stardata, image_size, dict(options, **{'flag_display2':False}))
        # re-get gaia database
        stardata, plate2, alt, az = match_centroids(other_stars_df, initial_guess, dbs, corners, image_size, dateguess, dict(options, **{'flag_display2':False}))


    # now recompute matches
    
    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, stardata, initial_guess, image_size, dict(options, **{'flag_display2':False}))

    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    print('pre-outlier removed rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)

    # compute flag:
    flag_is_double = np.zeros(stardata.ids.shape[0], int)
    neigh_all = gaia_search.lookup_nearby(stardata, options['double_star_cutoff'], options['double_star_mag'])
    neigh = NearestNeighbors(n_neighbors=2)
    neigh_all_data_extra2 = np.r_[neigh_all.get_ra_dec(), np.array([[-99999,-99999], [-99999, -99999]])] # ensure at least 2 "pseudo-neighbours"
    
    neigh.fit(neigh_all_data_extra2)
    distances, indices = neigh.kneighbors(stardata.get_ra_dec())

    flag_is_double = distances[:, 1] < np.radians(options['double_star_cutoff']/3600)
    flag_missing_pm = np.isnan(stardata.get_pmotion()[:, 0])
    flag_is_outlier = errors_arcseconds >= options['distortion_fit_tol']
    flag_unexplained_outlier = np.logical_and(np.logical_and(flag_is_outlier, np.logical_not(flag_missing_pm)), np.logical_not(flag_is_double))
    print(np.sum(flag_unexplained_outlier), ' unexplained outliers')
    keep_j = errors_arcseconds < options['distortion_fit_tol']

    plate2_unfiltered = plate2
    stardata_unfiltered = copy(stardata)
    plate2 = plate2[keep_j, :]
    stardata.select_indices(keep_j)
    #flag_is_double = flag_is_double[keep_j]
    #flag_missing_pm = flag_missing_pm[keep_j]
    
    print(f'{np.sum(1-keep_j)} outliers more than {options["distortion_fit_tol"]} arcseconds removed')
    # do 2nd fit with outliers removed

    if options['guess_date']:
        dateguess = distortion_cubic._date_guess(dateguess, initial_guess, plate2, stardata, image_size, options)
        #stardata_new = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(dateguess))
        #stardata.update_data(stardata_new)
        stardata.update_epoch(date_string_to_float(dateguess))
    
    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, stardata, initial_guess, image_size, options)
    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    
    print('final rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)
    detransformed = transforms.detransform_vectors(result, stardata.get_vectors())
    px_errors = plate2_corrected-detransformed
    nn_corr, nn_r = get_nn_correlation_error(plate2, px_errors, options)
    coeff_names = distortion_cubic.get_coeff_names(options)

    # recover errors for filtered points
    


    output_results = { 'final rms error (arcseconds)': np.degrees(np.mean(mag_errors**2)**0.5)*3600,
                       '#stars used':plate2.shape[0],
                       'observation_date':options['observation_date'] if not options['guess_date'] else dateguess,
                       'date_guessed?': options['guess_date'],
                       'star max magnitude':options['max_star_mag_dist'],
                       'error tolerance (as)':options['distortion_fit_tol'],
                       'platescale (arcseconds/pixel)': np.degrees(result[0])*3600,
                       'RA':np.degrees(result[1]),
                       'DEC':np.degrees(result[2]),
                       'ROLL':np.degrees(result[3])-180, # TODO: clarify this dodgy +/- 180 thing
                       'distortion coeffs x': dict(zip(coeff_names, [reg_x.intercept_]+list( reg_x.coef_))),
                       'distortion coeffs y': dict(zip(coeff_names, [reg_y.intercept_]+list( reg_y.coef_))),
                       'nearest-neighbour error correlation': nn_corr,
                       'aberration/parallax correction enabled?': options['enable_corrections'],
                       'refraction correction enabled?': options['enable_corrections_ref'],
                       'source_files':str(data['source_files']) if 'source_files' in data else 'unknown',
                       }
    additional_info = { 'observation_temp (°C)':options['observation_temp'],
                        'observation_pressure (millibars)':options['observation_pressure'],
                        'observation_humidity (0.0 to 1.0)':options['observation_humidity'],
                        'observation_height (m)':options['observation_height'],
                        'observation_wavelength (μm)':options['observation_wavelength'],
                        'observation alt (degrees)': alt,
                        'observation az (degrees)': az}
    if options['enable_corrections'] or options['enable_corrections_ref']:
        output_results.update(additional_info)

    
    with open(output_path(basename+'distortion_results.txt', options), 'w', encoding="utf-8") as fp:
        json.dump(output_results, fp, sort_keys=False, indent=4)

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
    plt.savefig(output_path('Error_graphs'+basename+'.png', options), bbox_inches="tight", dpi=600)
    if options['flag_display2']:
        plt.show()
    plt.close()


    plate2_unfiltered_corrected = distortion_cubic.apply_corrections(result, plate2_unfiltered, reg_x, reg_y, image_size, options)
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
            
    df_identification.to_csv(output_path('CATALOGUE_MATCHED_ERRORS'+basename+'.csv', options))

# #unused
def show_error_coherence(positions, errors, options):
    if not options['flag_display']:
        return
    dist = []
    corr = []

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
            dist.append(r)
            corr.append(corr_ij)
            if r < min_r:
                min_corr = corr_ij
                min_r = r
        nn_rs.append(min_r)
        nn_corrs.append(min_corr)

    print(f'nearest neighbour corr={np.mean(nn_corrs)}, mean distance:{np.mean(nn_rs)}')
                

    statistic, bin_edges, binnumber = scipy.stats.binned_statistic(dist, corr, bins = 8, range=(0, 1000))
    
    plt.plot(bin_edges[1:], statistic)
    plt.ylabel('error correlation')
    plt.xlabel('r / pixels')
    if options['flag_display2']:
        plt.show()
    plt.close()

if __name__ == '__main__':
    #new_data_path = "D:\output\FULL_DATA1707099836.1575732.npz" # zwo 4 zenith2
    #new_data_path = "D:\output\FULL_DATA1707106711.38932.npz" # E:/020323 moon test 294MM/020323_211852/211850_H-alpha_0000-20.fits
    #new_data_path = "D:\output\FULL_DATA1707152353.575058.npz" # zwo 3 zd0 0-8

    #new_data_path = "D:\output\FULL_DATA1707152555.3757603.npz" # zwo 3 zd 30 centre
    #new_data_path = "D:\output\FULL_DATA1707152800.7054024.npz" # zwo 3 zd 45 centre
    #new_data_path = "D:/output/FULL_DATA1707153601.071847.npz" # Don right calibration
    #new_data_path = "D:\output\FULL_DATA1707167920.0870245.npz" # E:/ZWO#3 2023-10-28/Zenith-01-3s/MEE2024.00003273.Zenith-Center2.fit
    new_data_path = 'D:\output4\CENTROID_OUTPUT20240303034855/data.zip' # eclipse (Don)
    options = {"catalogue":"gaia", "output_dir":"D:/output", 'max_star_mag_dist':12, 'flag_display':False, 'flag_display2':True, 'rough_match_threshhold':0.01, 'guess_date':False}
    match_and_fit_distortion(new_data_path, options , "D:/debugging")
