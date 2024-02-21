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



def get_fitfunc(plate, target, transform_function=transforms.linear_transform, img_shape=None):
    def fitfunc(x):
        rotated = transform_function(x, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc

def get_bbox(corners):
    def one_dim(q):
        t = (np.min(q), np.max(q))
        if t[1] - t[0] > 180:
            t = (t[1], t[0])
        return t
    return one_dim(corners[:, 1]), one_dim(corners[:, 0])

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


def match_centroids(data, result, dbs, corners, image_size, lookupdate, options):
    #TODO: this will be broken if we wrap around 360 degrees
    stardata = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(lookupdate)) # convert to decimal year (approximate)
    other_stars_df = pd.DataFrame(data=data['detection_arr'],    # values
                         columns=data['detection_arr_cols'])
    other_stars_df = other_stars_df.astype({'px':float, 'py':float}) # fix datatypes

    all_star_plate = np.array([other_stars_df['py'], other_stars_df['px']]).T - np.array([image_size[0]/2, image_size[1]/2])
    transformed_all = transforms.to_polar(transforms.linear_transform(result.x, all_star_plate))

    # match nearest neighbours
    candidate_stars = np.zeros((stardata.data.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(stardata.data[:, 1])
    candidate_stars[:, 1] = np.degrees(stardata.data[:, 0])

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

    match_threshhold = 1e-2 # in degrees
    confusion_ratio = 2 # cloest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio) # note: this distance metric is not perfect (doesn't take into account meridian etc.)
    keep = np.logical_and(keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0])) # is the nearest-neighbour relation reflexive? [this eliminates 1-to-many matching]
    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]
    
    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations')
    for i in range(stardata.data.shape[0]):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(stardata.ids[i]) + f'\nMag={stardata.data[i, 5]:.1f}', (np.degrees(stardata.data[i, 0]), np.degrees(stardata.data[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('initial rough fit')
    plt.legend()
    if options['flag_display2']:
        plt.show()
    plt.close()

    stardata.select_indices(indices[keep_i, 0].flatten())
    plate2 = all_star_plate[keep_i, :][0]

    return stardata, plate2

def match_and_fit_distortion(path_data, options, debug_folder=None):    
    path_catalogue = options['catalogue']
    basename = Path(path_data).stem
    
    data = np.load(path_data,allow_pickle=True) # TODO: can we avoid using pickle? How about using shutil.make_archive?
    image_size = data['img_shape']
    if not data['platesolved']: # need initial platesolve
        raise Exception("BAD DATA - Data did not have platesolve included!")
    df_id = pd.DataFrame(data=data['identification_arr'],    # values
                         columns=data['identification_arr_cols'])
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



    ### now try to match other stars

    corners = transforms.to_polar(transforms.linear_transform(result.x, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    dbs = database_cache.open_catalogue(path_catalogue)

    lookupdate = options['DEFAULT_DATE'] if options['guess_date'] else options['observation_date']
    stardata, plate2 = match_centroids(data, result, dbs, corners, image_size, lookupdate, options)
    
    ### fit again

    initial_guess = result.x

    if options['guess_date']:
        dateguess = options['DEFAULT_DATE'] # initial guess
        dateguess, _ = distortion_cubic._date_guess(dateguess, initial_guess, plate2, stardata, image_size, options)
        # re-get gaia database
        stardata, plate2 = match_centroids(data, result, dbs, corners, image_size, dateguess, dict(options, **{'flag_display2':False}))


    # now recompute matches
    
    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, stardata, initial_guess, image_size, dict(options, **{'flag_display2':False}))

    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    print('pre-outlier removed rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)

    keep_j = errors_arcseconds < options['distortion_fit_tol']

    plate2 = plate2[keep_j, :]
    stardata.select_indices(keep_j)
    
    print(f'{np.sum(1-keep_j)} outliers more than {options["distortion_fit_tol"]} arcseconds removed')
    # do 2nd fit with outliers removed

    if options['guess_date']:
        dateguess, _ = distortion_cubic._date_guess(dateguess, initial_guess, plate2, stardata, image_size, options)
        # re-get gaia database # TODO: it would be nice to epoch propagate offline, since we have the pmra, and pmdec
        stardata_new = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=date_string_to_float(dateguess))
        stardata.update_data(stardata_new)
    
    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, stardata, initial_guess, image_size, options)
    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    
    print('final rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)
    detransformed = transforms.detransform_vectors(result, stardata.get_vectors())
    px_errors = plate2_corrected-detransformed
    nn_corr, nn_r = get_nn_correlation_error(plate2, px_errors, options)
    coeff_names = distortion_cubic.get_coeff_names(options)

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
                       'source_files':str(data['source_files']) if 'source_files' in data else 'unknown',
                       }
    with open(output_path(basename+'distortion_results.txt', options), 'w', encoding="utf-8") as fp:
        json.dump(output_results, fp, sort_keys=False, indent=4)

    fig, axs = plt.subplots(2, 2)
    magnitudes = stardata.get_mags()
    axs[0, 0].scatter(magnitudes, np.degrees(mag_errors)*3600, marker='+')
    axs[0, 0].set_ylabel('error (arcsec)')
    axs[0, 0].set_xlabel('magnitude')
    axs[0, 0].grid()
    

    axs[0, 1].scatter(stardata.get_parallax(), np.degrees(mag_errors)*3600, marker='+')
    axs[0, 1].set_ylabel('residual error (arcsec)')
    axs[0, 1].set_xlabel('parallax (milli-arcsec)')
    axs[0, 1].grid()

    axs[1, 0].scatter(px_errors[:, 1], px_errors[:, 0], marker='+')
    axs[1, 0].set_ylabel('y-error(pixels)')
    axs[1, 0].set_xlabel('x-error(pixels)')
    axs[1, 0].grid()
    axs[1, 0].set_aspect('equal')
    radii = np.linalg.norm(plate2, axis=1)
    axs[1, 1].scatter(radii, np.degrees(mag_errors)*3600, marker='+')
    axs[1, 1].set_ylabel('error (arcsec)')
    axs[1, 1].set_xlabel('radial coordinate (pixels)')
    axs[1, 1].grid()
    fig.tight_layout()
    plt.savefig(output_path('Error_graphs'+basename+'.png', options), bbox_inches="tight", dpi=600)
    if options['flag_display2']:
        plt.show()
    plt.close()

    df_identification = pd.DataFrame({'px': plate2[:, 1]+image_size[1]/2,
                               'py': plate2[:, 0]+image_size[0]/2,
                               'px_dist': plate2_corrected[:, 1]+image_size[1]/2,
                               'py_dist': plate2_corrected[:, 0]+image_size[0]/2,
                               'ID': ['gaia:'+str(_) for _ in stardata.ids],
                               'RA(catalog)': np.degrees(stardata.data[:, 0]),
                               'DEC(catalog)': np.degrees(stardata.data[:, 1]),
                               'RA(obs)': transforms.to_polar(transformed_final)[:, 1],
                               'DEC(obs)': transforms.to_polar(transformed_final)[:, 0],
                               'magV': stardata.get_mags(),
                                'error(")':errors_arcseconds})
            
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
    new_data_path = "D:/output/FULL_DATA1707153601.071847.npz" # Don right calibration
    #new_data_path = "D:\output\FULL_DATA1707167920.0870245.npz" # E:/ZWO#3 2023-10-28/Zenith-01-3s/MEE2024.00003273.Zenith-Center2.fit
    
    options = {"catalogue":"D:/tyc_dbase4/tyc_main.dat", "output_dir":"D:/output", 'max_star_mag_dist':10.5, 'flag_display':True}
    match_and_fit_distortion(new_data_path, options , "D:/debugging")
