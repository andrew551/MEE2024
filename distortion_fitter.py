import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.spatial.transform import Rotation

def to_polar(v):
    theta = np.arcsin(v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])
    ret = np.array([theta*360/2/np.pi, phi*360/2/np.pi])
    ret[ret < 0] += 360
    return ret.T

def transform(x, plate, img_shape):
    pixel_scale = x[0] # radians per pixel
        

    # 2) non_linear terms

    #x[3 to 5, 6 to 8] : quadratic terms (x, y)#: conventention: value is the largest correction applied (in pixels)
    #x[9 to 12] : cubic terms

    # 3) position 
    
    ra, dec, roll = x[1], x[2], x[3]

    # step 1: linear transform

    plate_lin = np.copy(plate)
    plate_lin -= np.array([img_shape[0]/2, img_shape[1]/2])

    
    scale_x = x[4]
    shear_x = x[5]
    #plate_lin[:, 1] = scale_x*plate_lin[:, 1] + shear_x*plate_lin[:, 0]
    #plate_lin[:, 0] = (1/scale_x)*plate_lin[:, 0]
    # step 2: quadratic correction
    
    q0 = np.array([[4*x[6]/img_shape[0]**2, 2*x[7]/img_shape[0]/img_shape[1]], [2*x[7]/img_shape[0]/img_shape[1], 4*x[8]/img_shape[1]**2]])
    q1 = np.array([[4*x[9]/img_shape[0]**2, 2*x[10]/img_shape[0]/img_shape[1]], [2*x[10]/img_shape[0]/img_shape[1], 4*x[11]/img_shape[1]**2]])
    quadratic_0 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q0)#(plate_lin @ q0 @ plate_lin.T)
    quadratic_1 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q1)#(plate_lin @ q1 @ plate_lin.T)
    quad_correction = np.array([quadratic_0, quadratic_1]).T


    c0 = np.zeros((2, 2, 2))
    c0[0, 0, 0] = x[12]
    c0[1, 0, 0] = x[13] / 3
    c0[0, 1, 0] = c0[1, 0, 0]
    c0[0, 0, 1] = c0[1, 0, 0]
    c0[1, 1, 0] = x[14] / 3
    c0[0, 1, 1] = c0[1, 1, 0]
    c0[1, 0, 1] = c0[1, 1, 0]
    c0[1, 1, 1] = x[15]
    c1 = np.zeros((2, 2, 2))
    c1[0, 0, 0] = x[16]
    c1[1, 0, 0] = x[17] / 3
    c1[0, 1, 0] = c1[1, 0, 0]
    c1[0, 0, 1] = c1[1, 0, 0]
    c1[1, 1, 0] = x[18] / 3
    c1[0, 1, 1] = c1[1, 1, 0]
    c1[1, 0, 1] = c1[1, 1, 0]
    c1[1, 1, 1] = x[19]
    c1 = c1 * 8/img_shape[0]**3
    c0 = c0 * 8/img_shape[0]**3
    cubic_0 = np.einsum('ij,ik,il,jkl->i', plate_lin, plate_lin, plate_lin, c0)
    cubic_1 = np.einsum('ij,ik,il,jkl->i', plate_lin, plate_lin, plate_lin, c1)
    cubic_correction = np.array([cubic_0, cubic_1]).T
    corrected = plate_lin #+ quad_correction+cubic_correction # TODO: plus cubic term...




    # step 3: To spherical coordinates
    icoords = corrected * pixel_scale
    icoords[:, 1] = icoords[:, 1] / np.cos(icoords[:, 0]) # spherical coordinate curveture
    
    vector_positions_z = np.sin(icoords[:, 0])
    vector_positions_x = np.cos(icoords[:, 0]) * np.cos(icoords[:, 1])
    vector_positions_y = np.cos(icoords[:, 0]) * np.sin(icoords[:, 1])

    plate_vectors = np.array([vector_positions_x, vector_positions_y, vector_positions_z]).T
    
    # apply roll, then RA, then declination
    r = Rotation.from_euler('xyz', [roll, ra-np.pi/2, dec])
    rotated = r.apply(plate_vectors)
    return rotated

def qtransform(x, plate, img_shape):
    pixel_scale = x[0] # radians per pixel
        
    ra, dec, roll = x[1], x[2], x[3]

    # step 1: linear transform

    plate_lin = np.copy(plate)
    plate_lin -= np.array([img_shape[0]/2, img_shape[1]/2])

    # step 2: quadratic correction
    
    q0 = np.array([[4*x[4]/img_shape[0]**2, 2*x[5]/img_shape[0]/img_shape[1]], [2*x[5]/img_shape[0]/img_shape[1], 4*x[6]/img_shape[1]**2]])
    q1 = np.array([[4*x[7]/img_shape[0]**2, 2*x[8]/img_shape[0]/img_shape[1]], [2*x[8]/img_shape[0]/img_shape[1], 4*x[9]/img_shape[1]**2]])
    quadratic_0 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q0)#(plate_lin @ q0 @ plate_lin.T)
    quadratic_1 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q1)#(plate_lin @ q1 @ plate_lin.T)
    quad_correction = np.array([quadratic_0, quadratic_1]).T

    corrected = plate_lin + quad_correction # TODO: plus cubic term...

    # step 3: To spherical coordinates
    icoords = corrected * pixel_scale
    icoords[:, 1] = icoords[:, 1] / np.cos(icoords[:, 0]) # spherical coordinate curveture
    
    vector_positions_z = np.sin(icoords[:, 0])
    vector_positions_x = np.cos(icoords[:, 0]) * np.cos(icoords[:, 1])
    vector_positions_y = np.cos(icoords[:, 0]) * np.sin(icoords[:, 1])

    plate_vectors = np.array([vector_positions_x, vector_positions_y, vector_positions_z]).T
    
    # apply roll, then RA, then declination
    r = Rotation.from_euler('xyz', [roll, ra-np.pi/2, dec])
    rotated = r.apply(plate_vectors)
    return rotated

def get_fitfunc(plate, target, img_shape):

    def fitfunc(x):
        rotated = transform(x, plate, image_size)
        return np.linalg.norm(target-rotated)**2
    return fitfunc

#data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1705968683.6189432.csv"
data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1706029556.8293524.csv"
data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1706042215.7721128.csv"

data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1706042630.4431906.csv" # moon test
#data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1706042946.7686048.csv" # zwo 4 zenith2
image_size = (6388, 9576)
image_size = (3250, 4656)
image_size = (5644, 8288) # moon test
#image_size = (3250, 4656) # zwo 4

df = pd.read_csv(data_path)
df['vx'] = np.cos(np.radians(df['DEC'])) * np.cos(np.radians(df['RA']))
df['vy'] = np.cos(np.radians(df['DEC'])) * np.sin(np.radians(df['RA']))
df['vz'] = np.sin(np.radians(df['DEC']))

print(df)


target = np.array([df['vx'], df['vy'], df['vz']]).T
plate = np.array([df['py'], df['px']]).T
print(plate.shape)
print(target.shape)

f = get_fitfunc(plate, target, image_size)
#print(target)
#print(f((9e-6, 326.35/360*6.282, 44.81/360*6.282, 178/360*6.282)))

initial_guess = (7e-6, 27.33/360*6.282, 113.2/360*6.282, 4/360*6.282, 1, 0, 0,0,0,0,0,0,   0, 0, 0, 0, 0, 0, 0, 0) # moon test
#initial_guess = (9e-6, 44.81/360*6.282, 326.35/360*6.282, 178/360*6.282, 1, 0, 0,0,0,0,0,0,   0, 0, 0, 0, 0, 0, 0, 0) # zwo 4

#initial_guess = (9e-6, 44.81/360*6.282, 326.35/360*6.282, 178/360*6.282, 0,0,0,0,0,0)
result = scipy.optimize.minimize(get_fitfunc(plate, target, image_size), initial_guess, method = 'BFGS')  
print(result)
print(result.fun**0.5 / result.x[0])

resv = to_polar(transform(result.x, plate, image_size))
print(resv)

orig = to_polar(target)
print(orig)

plt.scatter(orig[:, 1], orig[:, 0])
plt.scatter(resv[:, 1], resv[:, 0])
plt.scatter(df['RA'], df['DEC'])
plt.show()





errors = resv - orig
if 0:
    plt.scatter((plate[:, 1]-image_size[1]/2), errors[:, 0])
    plt.show()

    plt.scatter((plate[:, 1]-image_size[1]/2), errors[:, 1])
    plt.show()

    plt.scatter((plate[:, 0]-image_size[0]/2), errors[:, 0])
    plt.show()

    plt.scatter((plate[:, 0]-image_size[0]/2), errors[:, 1])

    plt.show()



import database_lookup2
corners = to_polar(transform(result.x, np.array([[0,0], [image_size[0]-1., image_size[1]-1.], [0, image_size[1]-1.], [image_size[0]-1., 0]]), image_size))
dbs = database_lookup2.database_searcher("D:/tyc_dbase4/tyc_main.dat", debug_folder="D:/debugging")
print(corners)
#TODO: this will be broken if we wrap around 360 degrees
startable, starid = dbs.lookup_objects((np.min(corners[:, 1]), np.max(corners[:, 1])), (np.min(corners[:, 0]), np.max(corners[:, 0])))
other_path = "D:\output\STACKED_CENTROIDS_DATA1706042630.4431906.csv" # moon test
#other_path = "D:\output\STACKED_CENTROIDS_DATA1706042946.7686048.csv" # zwo 4
other_stars_df = pd.read_csv(other_path)
all_star_plate = np.array([other_stars_df['py'], other_stars_df['px']]).T

transformed_all = to_polar(transform(result.x, all_star_plate, image_size))

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


from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)

neigh.fit(candidate_stars)
distances, indices = neigh.kneighbors(transformed_all)
print(indices)
print(distances)

# find matches, but exclude ambiguity

match_threshhold = 1e-2 # in degrees
confusion_ratio = 2 # cloest match must be 2x closer than second place

keep = np.logical_and(distances[:, 0] < match_threshhold, distances[:, 1] / distances[:, 0] > confusion_ratio)
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


# remote RA, DEC, ROLL

# compute distortion coefficients


#result2 = scipy.optimize.minimize(get_fitfunc(plate, target, image_size), initial_guess, method = 'BFGS')  
#print(result)
