import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.spatial.transform import Rotation
import transforms
from sklearn.neighbors import NearestNeighbors
import database_lookup2
import os
from MEE2024util import output_path
import json
from pathlib import Path
import database_cache
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import distortion_cubic

def to_polar(v):
    theta = np.arcsin(v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])
    phi[phi < 0] += np.pi * 2
    ret = np.degrees(np.array([theta, phi]))
    
    return ret.T

def get_fitfunc(plate, target, transform_function=transforms.linear_transform, img_shape=None):

    def fitfunc(x):
        rotated = transform_function(x, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc

def get_d_fitfunc(plate, target, x_lin, transform_function, img_shape):

    def fitfunc(x):
        xx = list(x) + list(x_lin)
        rotated = transform_function(xx, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc

def get_e_fitfunc(plate, target, x_dist, transform_function, img_shape):

    def fitfunc(x):
        xx = list(x_dist) + list(x)
        rotated = transform_function(xx, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc

def get_bbox(corners):
    def one_dim(q):
        t = (np.min(q), np.max(q))
        if t[1] - t[0] > 180:
            t = (t[1], t[0])
        return t
    return one_dim(corners[:, 1]), one_dim(corners[:, 0])



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
    if options['flag_display']:
        plt.show()
    plt.close()
    

def match_and_fit_distortion(path_data, options, debug_folder=None):    
    path_catalogue = options['catalogue']
    basename = Path(path_data).stem
    
    data = np.load(path_data,allow_pickle=True)
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
    #print(result.fun**0.5 / result.x[0])

    resi = to_polar(transforms.linear_transform(initial_guess, plate))

    resv = to_polar(transforms.linear_transform(result.x, plate))
    orig = to_polar(target)

    '''
    plt.scatter(orig[:, 1], orig[:, 0], label='catalogue')
    plt.scatter(resv[:, 1], resv[:, 0], label = 'fitted')
    plt.scatter(resi[:, 1], resi[:, 0], label = 'initial guess')
    plt.scatter(df_id['RA'], df_id['DEC'], label='cata2')
    plt.legend()
    plt.show()
    errors = resv - orig
    '''
    ### now try to match other stars

    corners = to_polar(transforms.linear_transform(result.x, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]) - np.array([image_size[0]/2, image_size[1]/2])))
    #dbs = database_lookup2.database_searcher(path_catalogue, debug_folder=debug_folder, star_max_magnitude=12)
    dbs = database_cache.open_catalogue(path_catalogue)
    #print(corners)
    #TODO: this will be broken if we wrap around 360 degrees
    startable, starid = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'], time=datetime.datetime.fromisoformat(options['observation_date']).toordinal()/365.24+1) # convert to decimal year (approximate)
    other_stars_df = pd.DataFrame(data=data['detection_arr'],    # values
                         columns=data['detection_arr_cols'])
    other_stars_df = other_stars_df.astype({'px':float, 'py':float}) # fix datatypes

    all_star_plate = np.array([other_stars_df['py'], other_stars_df['px']]).T - np.array([image_size[0]/2, image_size[1]/2])

    transformed_all = to_polar(transforms.linear_transform(result.x, all_star_plate))
    '''
    plt.scatter(np.degrees(startable[:, 0]), np.degrees(startable[:, 1]), label='catalogue')
    plt.scatter(transformed_all[:, 1], transformed_all[:, 0], marker='+', label='observations')
    for i in range(startable.shape[0]):
        plt.gca().annotate(str(starid[i, :]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.legend()
    plt.show()
    '''
    # match nearest neighbours

    candidate_stars = np.zeros((startable.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(startable[:, 1])
    candidate_stars[:, 1] = np.degrees(startable[:, 0])

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
    for i in range(startable.shape[0]):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(starid[i]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.title('initial rough fit')
    plt.legend()
    if options['flag_display2']:
        plt.show()
    plt.close()
    

    target2_table = startable[indices[keep_i, 0], :][0]
    target2 = target2_table[:, 2:5]
    plate2 = all_star_plate[keep_i, :][0]
    starid = starid[indices[keep_i, 0]][0]
    print(starid.shape)

    ### fit again

    initial_guess = result.x

    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, target2, initial_guess, image_size, dict(options, **{'flag_display2':False}))

    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - target2, axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    magnitudes = startable[:, 5][indices[keep_i, 0]][0]
    print('pre-outlier removed rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)

    keep_j = errors_arcseconds < options['distortion_fit_tol']

    plate2 = plate2[keep_j]
    target2_table = target2_table[keep_j]
    target2 = target2[keep_j]
    magnitudes = magnitudes[keep_j]
    starid = starid[keep_j]
    print(f'{np.sum(1-keep_j)} outliers more than {options["distortion_fit_tol"]} arcseconds removed')
    # do 2nd fit with outliers removed
    
    result, plate2_corrected, reg_x, reg_y = distortion_cubic.do_cubic_fit(plate2, target2, initial_guess, image_size, options)
    transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
    mag_errors = np.linalg.norm(transformed_final - target2, axis=1)
    errors_arcseconds = np.degrees(mag_errors)*3600
    
    print('final rms error (arcseconds):', np.degrees(np.mean(mag_errors**2)**0.5)*3600)
    detransformed = transforms.detransform_vectors(result, target2)
    px_errors = plate2_corrected-detransformed
    nn_corr, nn_r = get_nn_correlation_error(plate2, px_errors, options)
    coeff_names = distortion_cubic.get_coeff_names(options)

    output_results = { 'final rms error (arcseconds)': np.degrees(np.mean(mag_errors**2)**0.5)*3600,
                       '#stars used':plate2.shape[0],
                       'observation_date':options['observation_date'],
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
    
    axs[0, 0].scatter(magnitudes, np.degrees(mag_errors)*3600, marker='+')
    axs[0, 0].set_ylabel('error (arcseconds)')
    axs[0, 0].set_xlabel('magnitude')
    axs[0, 0].grid()
    

    axs[0, 1].scatter(target2_table[:, 6], np.degrees(mag_errors)*3600, marker='+')
    axs[0, 1].set_ylabel('residual error (arcseconds)')
    axs[0, 1].set_xlabel('parallax (milli-arcsec)')
    axs[0, 1].grid()

    axs[1, 0].scatter(px_errors[:, 1], px_errors[:, 0], marker='+')
    axs[1, 0].set_ylabel('y-error(pixels)')
    axs[1, 0].set_xlabel('x-error(pixels)')
    axs[1, 0].grid()
    axs[1, 0].set_aspect('equal')
    radii = np.linalg.norm(plate2, axis=1)
    axs[1, 1].scatter(radii, np.degrees(mag_errors)*3600, marker='+')
    axs[1, 1].set_ylabel('error (pixels)')
    axs[1, 1].set_xlabel('radial coordinate (pixels)')
    axs[1, 1].grid()
    plt.savefig(output_path('Error_graphs'+basename+'.png', options), bbox_inches="tight", dpi=600)
    if options['flag_display2']:
        plt.show()
    plt.close()

    df_identification = pd.DataFrame({'px': plate2[:, 1]+image_size[1]/2,
                               'py': plate2[:, 0]+image_size[0]/2,
                               'px_dist': plate2_corrected[:, 1]+image_size[1]/2,
                               'py_dist': plate2_corrected[:, 0]+image_size[0]/2,
                               'ID': ['gaia:'+str(_) for _ in starid],
                               'RA(catalog)': np.degrees(target2_table[:, 0]),
                               'DEC(catalog)': np.degrees(target2_table[:, 1]),
                               'RA(obs)': to_polar(transformed_final)[:, 1],
                               'DEC(obs)': to_polar(transformed_final)[:, 0],
                               'magV': target2_table[:, 5],
                                'error(")':errors_arcseconds})
            
    df_identification.to_csv(output_path('CATALOGUE_MATCHED_ERRORS'+basename+'.csv', options))

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
