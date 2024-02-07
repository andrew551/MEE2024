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


###### the distortion fitting and matching function #####

def match_and_fit_distortion(path_data, options, debug_folder=None):

    #distortion_function = transforms.cubic_distortion
    #distortion_n_parameters = 8

    #distortion_function = transforms.brown_distortion
    #distortion_n_parameters = 5

    distortion_function = transforms.skew_distortion
    distortion_n_parameters = 4
    
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
    '''
    plt.scatter(df_id['px'], df_id['RA'])
    plt.show()
    plt.scatter(df_id['py'], df_id['DEC'])
    plt.show()
    plt.scatter(df_id['px'], df_id['DEC'])
    plt.show()
    plt.scatter(df_id['py'], df_id['RA'])
    plt.show()
    '''
    
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
    startable, starid = dbs.lookup_objects(*get_bbox(corners), star_max_magnitude=options['max_star_mag_dist'])
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
    '''
    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations')
    for i in range(startable.shape[0]):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(starid[i, :]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.legend()
    if options['flag_display']:
        plt.show()
    plt.close()
    '''

    target2 = startable[indices[keep_i, 0], :][0]
    target2 = target2[:, 2:5]
    plate2 = all_star_plate[keep_i, :][0]
    #print(plate2)
    #print(target2)
    f2 = get_fitfunc(plate2, target2)

    ### fit again

    initial_guess = result.x
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error linear solve: ', result.fun**0.5)

    #resi = to_polar(transforms.linear_transform(initial_guess, plate2, image_size))

    resv = to_polar(transforms.linear_transform(result.x, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(result.x, target2)

    plt.scatter(detransformed[:, 1], detransformed[:, 0], marker='+')
    plt.scatter(plate2[:, 1], plate2[:, 0], marker='+')
    if options['flag_display']:
        plt.show()
    plt.close()
    
    errors = detransformed - plate2

    basis = []
    for i in range(1, 4): # up to cubic order
        for j in range(i+1):
            basis.append(plate2[:, 0] ** j * plate2[:, 1] ** (i-j))
    basis = np.array(basis).T
    reg_x = LinearRegression().fit(basis, errors[:, 1]*result.x[0])
    reg_y = LinearRegression().fit(basis, errors[:, 0]*result.x[0])
    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print(reg_x.predict(basis)/ result.x[0] - errors[:, 1])
    print(reg_y.predict(basis)/ result.x[0] - errors[:, 0])

    plate2_corrected = plate2 + np.array([reg_y.predict(basis), reg_x.predict(basis)]).T / result.x[0]

    initial_guess = result.x
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2_corrected, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error corrected solve (arcseconds): ', np.degrees(result.fun**0.5) * 3600)

    transformed_final = transforms.linear_transform(result.x, plate2_corrected, image_size)

    print('final rms error (arcseconds):', np.degrees(result.fun**0.5)*3600)

    output_results = { 'final rms error (arcseconds)': np.degrees(result.fun**0.5)*3600,
                       '#stars used':plate2.shape[0],
                       'star max magnitude':options['max_star_mag_dist'],
                       'platescale (arcseconds/pixel)': np.degrees(result.x[0])*3600,
                       'RA':np.degrees(result.x[1]),
                       'DEC':np.degrees(result.x[2]),
                       'ROLL':np.degrees(result.x[3])-180, # TODO: clarify this dodgy +/- 180 thing
                       'distortion coeffs x': [reg_x.intercept_]+list( reg_x.coef_),
                       'distortion coeffs y': [reg_y.intercept_]+list( reg_y.coef_),
                       #'BROWN_DISTORTION_COEFFICIENTS (pixels) K1, K2, K3, P1, P2':coefficients[:distortion_n_parameters],
                       }
    with open(output_path(basename+'distortion_results.txt', options), 'w', encoding="utf-8") as fp:
        json.dump(output_results, fp, sort_keys=False, indent=4)
    mag_errors = np.linalg.norm(transformed_final - target2, axis=1)
    magnitudes = startable[:, 5][indices[keep_i, 0]][0]
    plt.scatter(magnitudes, np.degrees(mag_errors)*3600, marker='+')
    plt.ylabel('error (arcseconds)')
    plt.xlabel('magnitude')
    plt.grid()
    if options['flag_display']:
        plt.show()
    plt.close()

    detransformed = transforms.detransform_vectors(result.x, target2)
    plt.scatter(detransformed[:, 1], detransformed[:, 0], marker='+')
    plt.scatter(plate2_corrected[:, 1], plate2_corrected[:, 0], marker='+')
    if options['flag_display']:
        plt.show()
    plt.close()

    ### remove outliers?

    

    
    return
    ####old code ##########
    '''
    plt.scatter(orig[:, 1], orig[:, 0])
    plt.scatter(resv[:, 1], resv[:, 0])
    #plt.scatter(resi[:, 1], resi[:, 0])
    #plt.scatter(df_id['RA'], df_id['DEC'])
    plt.show()
    '''
    errors = (resv - orig) * 3600 # arcseconds

    plate2center = plate2
    
    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(plate2center[:, 1], errors[:, 0], label = 'px-ey')
    axs[0,0].legend()
    axs[1,0].scatter(plate2center[:, 1], errors[:, 1], label = 'px-ex')
    axs[1,0].legend()
    axs[1,1].scatter(plate2center[:, 0], errors[:, 0], label = 'py-ey')
    axs[1,1].legend()
    axs[0,1].scatter(plate2center[:, 0], errors[:, 1], label = 'py-ex')
    axs[0,1].legend()
    if options['flag_display']:
        plt.show()
    '''
    plt.scatter(plate2center[:, 1]*plate2center[:, 0], errors[:, 1], label = 'pxpy-ex')
    plt.show()
    
    radii = np.linalg.norm(plate2center, axis=1)
    unit_vectors = plate2center/np.reshape(radii, (radii.shape[0], 1))

    radial_errors = np.einsum('ji,ji->j', errors, unit_vectors)

    plt.scatter(radii, radial_errors)
    plt.show()
    '''
    '''
    radii = np.linalg.norm(plate2center, axis=1)
    unit_vectors = plate2center/np.reshape(radii, (radii.shape[0], 1))

    radial_errors = np.einsum('ji,ji->j', errors, unit_vectors)

    plt.scatter(radii, radial_errors)
    plt.show()

    n = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    ax.scatter(plate2center[:,1], plate2center[:, 0], radial_errors, marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    ax.scatter(plate2center[:,1], plate2center[:, 0], errors[:, 1], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('x-error')
    if options['flag_display']:
        plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    ax.scatter(plate2center[:,1], plate2center[:, 0], errors[:, 0], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('y-error')

    if options['flag_display']:
        plt.show()
    plt.close()
    # optimize distortion function
    f3 = get_d_fitfunc(plate2, target2, result.x, transform_function=distortion_function, img_shape=image_size)
    initial_guess = [0 for _ in range(distortion_n_parameters)]
    result_d = scipy.optimize.minimize(f3, initial_guess, method = 'Nelder-Mead', options={'maxfev':distortion_n_parameters*800})  # BFGS doesn't like something here
    print(result_d)
    print('rms error distortion solve: ', result_d.fun**0.5)
    # optimize linear function again
    f4 = get_e_fitfunc(plate2, target2, result_d.x, transform_function=distortion_function, img_shape=image_size)
    initial_guess = result.x
    result_e = scipy.optimize.minimize(f4, initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result_e)

    
    coefficients = list(result_d.x) + list(result_e.x)
    print('final coefficients:', coefficients, f'from fitting {plate2.shape[0]} stars')
    transformed_final = distortion_function(coefficients, plate2, image_size)

    print('final rms error (arcseconds):', np.degrees(result_e.fun**0.5)*3600)

    output_results = { 'final rms error (arcseconds)': np.degrees(result_e.fun**0.5)*3600,
                       '#stars used':plate2.shape[0],
                       'star max magnitude':options['max_star_mag_dist'],
                       'platescale (arcseconds/pixel)': np.degrees(coefficients[distortion_n_parameters])*3600,
                       'RA':np.degrees(coefficients[distortion_n_parameters+1]),
                       'DEC':np.degrees(coefficients[distortion_n_parameters+2]),
                       'ROLL':np.degrees(coefficients[distortion_n_parameters+3])-180, # TODO: clarify this dodgy +/- 180 thing
                       'BROWN_DISTORTION_COEFFICIENTS (pixels) K1, K2, K3, P1, P2':coefficients[:distortion_n_parameters],
                       }
    with open(output_path(basename+'distortion_results.txt', options), 'w', encoding="utf-8") as fp:
        json.dump(output_results, fp, sort_keys=False, indent=4)
    
    
    resv = to_polar(transformed_final)
    orig = to_polar(target2)


    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(resv[:, 1], resv[:, 0], marker='+', label='observations')
    for i in range(startable.shape[0]):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(starid[i, :]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.legend()
    if options['flag_display']:
        plt.show()
    plt.close()
    
    errors = (resv - orig) * 3600 # arcseconds

    plate2center = plate2

    
    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(plate2center[:, 1], errors[:, 0], label = 'px-ey')
    axs[0,0].legend()
    axs[1,0].scatter(plate2center[:, 1], errors[:, 1], label = 'px-ex')
    axs[1,0].legend()
    axs[1,1].scatter(plate2center[:, 0], errors[:, 0], label = 'py-ey')
    axs[1,1].legend()
    axs[0,1].scatter(plate2center[:, 0], errors[:, 1], label = 'py-ex')
    axs[0,1].legend()
    if options['flag_display']:
        plt.show()
    plt.close()
    mag_errors = np.linalg.norm(transformed_final - target2, axis=1)
    magnitudes = startable[:, 5][indices[keep_i, 0]][0]
    plt.scatter(magnitudes, np.degrees(mag_errors)*3600, marker='+')
    plt.ylabel('error (arcseconds)')
    plt.xlabel('magnitude')
    plt.grid()
    if options['flag_display']:
        plt.show()
    plt.close()

    #####

    
    

if __name__ == '__main__':
    #new_data_path = "D:\output\FULL_DATA1707099836.1575732.npz" # zwo 4 zenith2
    #new_data_path = "D:\output\FULL_DATA1707106711.38932.npz" # E:/020323 moon test 294MM/020323_211852/211850_H-alpha_0000-20.fits
    #new_data_path = "D:\output\FULL_DATA1707152353.575058.npz" # zwo 3 zd0 0-8

    #new_data_path = "D:\output\FULL_DATA1707152555.3757603.npz" # zwo 3 zd 30 centre
    #new_data_path = "D:\output\FULL_DATA1707152800.7054024.npz" # zwo 3 zd 45 centre
    new_data_path = "D:\output\FULL_DATA1707153601.071847.npz" # Don right calibration
    #new_data_path = "D:\output\FULL_DATA1707167920.0870245.npz" # E:/ZWO#3 2023-10-28/Zenith-01-3s/MEE2024.00003273.Zenith-Center2.fit
    
    options = {"catalogue":"D:/tyc_dbase4/tyc_main.dat", "output_dir":"D:/output", 'max_star_mag_dist':10.5, 'flag_display':True}
    match_and_fit_distortion(new_data_path, options , "D:/debugging")
