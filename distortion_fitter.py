import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.spatial.transform import Rotation
import transforms
from sklearn.neighbors import NearestNeighbors
import database_lookup2

def to_polar(v):
    theta = np.arcsin(v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])
    ret = np.degrees(np.array([theta, phi]))
    ret[ret < 0] += 360
    return ret.T

def get_fitfunc(plate, target, img_shape):

    def fitfunc(x):
        rotated = transforms.linear_transform(x, plate, img_shape)
        return np.linalg.norm(target-rotated)**2
    return fitfunc

def get_bbox(corners):
    def one_dim(q):
        t = (np.min(q), np.max(q))
        if t[1] - t[0] > 180:
            t = (t[1], t[0])
        return t
    return one_dim(corners[:, 1]), one_dim(corners[:, 0])


###### the distortion fitting and matching function #####

def match_and_fit_distortion(path_data):
    data = np.load(path_data,allow_pickle=True)
    image_size = data['img_shape']
    assert(data['platesolved']) # need initial platesolve
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
    plate = np.array([df_id['py'], df_id['px']]).T
    f = get_fitfunc(plate, target, image_size)

    # note: negative scale == rotation by 180. Do we always want this +180 hack?
    initial_guess = tuple(np.radians([data['platescale'], data['RA'], data['DEC'], data['roll']+180]))
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate, target, image_size), initial_guess, method = 'BFGS')  # BFGS doesn't like something here
    print(result)
    print(result.fun**0.5 / result.x[0])

    resi = to_polar(transforms.linear_transform(initial_guess, plate, image_size))

    resv = to_polar(transforms.linear_transform(result.x, plate, image_size))
    orig = to_polar(target)

    plt.scatter(orig[:, 1], orig[:, 0])
    plt.scatter(resv[:, 1], resv[:, 0])
    plt.scatter(resi[:, 1], resi[:, 0])
    plt.scatter(df_id['RA'], df_id['DEC'])
    plt.show()
    errors = resv - orig

    ### now try to match other stars

    corners = to_polar(transforms.linear_transform(result.x, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]), image_size))
    dbs = database_lookup2.database_searcher("D:/tyc_dbase4/tyc_main.dat", debug_folder="D:/debugging")
    print(corners)
    #TODO: this will be broken if we wrap around 360 degrees
    startable, starid = dbs.lookup_objects(*get_bbox(corners))
    other_stars_df = pd.DataFrame(data=data['detection_arr'],    # values
                         columns=data['detection_arr_cols'])
    other_stars_df = other_stars_df.astype({'px':float, 'py':float}) # fix datatypes

    all_star_plate = np.array([other_stars_df['py'], other_stars_df['px']]).T

    transformed_all = to_polar(transforms.linear_transform(result.x, all_star_plate, image_size))

    plt.scatter(np.degrees(startable[:, 0]), np.degrees(startable[:, 1]), label='catalogue')
    plt.scatter(transformed_all[:, 1], transformed_all[:, 0], marker='+', label='observations')
    for i in range(startable.shape[0]):
        plt.gca().annotate(str(starid[i, :]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.legend()
    plt.show()

    # match nearest neighbours

    candidate_stars = np.zeros((startable.shape[0], 2))
    candidate_stars[:, 0] = np.degrees(startable[:, 1])
    candidate_stars[:, 1] = np.degrees(startable[:, 0])


    neigh = NearestNeighbors(n_neighbors=2)

    neigh.fit(candidate_stars)
    distances, indices = neigh.kneighbors(transformed_all)
    print(indices)
    print(distances)

    # find matches, but exclude ambiguity
    # TODO fix 1-many matching bug

    match_threshhold = 1e-2 # in degrees
    confusion_ratio = 2 # cloest match must be 2x closer than second place

    keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio) # note: this distance metric is not perfect (doesn't take into account meridian etc.)
    keep_i = np.nonzero(keep)

    obs_matched = transformed_all[keep_i, :][0]
    cata_matched = candidate_stars[indices[keep_i, 0], :][0]

    plt.scatter(cata_matched[:, 1], cata_matched[:, 0], label='catalogue')
    plt.scatter(obs_matched[:, 1], obs_matched[:, 0], marker='+', label='observations')
    for i in range(startable.shape[0]):
        if i in indices[keep_i, 0]:
            plt.gca().annotate(str(starid[i, :]) + f'\nMag={startable[i, 5]:.1f}', (np.degrees(startable[i, 0]), np.degrees(startable[i, 1])), color='black', fontsize=5)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.legend()
    plt.show()

    target2 = startable[indices[keep_i, 0], :][0]
    target2 = target2[:, 2:5]
    #target = np.array([df_id['vx'], df_id['vy'], df_id['vz']]).T
    plate2 = all_star_plate[keep_i, :][0]
    print(plate2)
    print(target2)
    f2 = get_fitfunc(plate2, target2, image_size)

    ### fit again

    initial_guess = result.x
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2, target2, image_size), initial_guess, method = 'BFGS')  # BFGS doesn't like something here
    print(result)
    print(result.fun**0.5 / result.x[0])

    #resi = to_polar(transforms.linear_transform(initial_guess, plate2, image_size))

    resv = to_polar(transforms.linear_transform(result.x, plate2, image_size))
    orig = to_polar(target2)

    plt.scatter(orig[:, 1], orig[:, 0])
    plt.scatter(resv[:, 1], resv[:, 0])
    #plt.scatter(resi[:, 1], resi[:, 0])
    #plt.scatter(df_id['RA'], df_id['DEC'])
    plt.show()
    errors = resv - orig

if __name__ == '__main__':
    new_data_path = "D:\output\FULL_DATA1707099836.1575732.npz" # zwo 4 zenith2
    match_and_fit_distortion(new_data_path)
