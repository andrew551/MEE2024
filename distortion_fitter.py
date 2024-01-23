import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.spatial.transform import Rotation

def to_polar(v):
    theta = np.arccos(v[:, 2])
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
    plate_lin[:, 1] = scale_x*plate_lin[:, 1] + shear_x*plate_lin[:, 0]
    plate_lin[:, 0] = (1/scale_x)*plate_lin[:, 0]
    # step 2: quadratic correction
    
    q0 = np.array([[4*x[6]/img_shape[0]**2, 2*x[7]/img_shape[0]/img_shape[1]], [2*x[7]/img_shape[0]/img_shape[1], 4*x[8]/img_shape[1]**2]])
    q1 = np.array([[4*x[9]/img_shape[0]**2, 2*x[10]/img_shape[0]/img_shape[1]], [2*x[10]/img_shape[0]/img_shape[1], 4*x[11]/img_shape[1]**2]])
    quadratic_0 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q0)#(plate_lin @ q0 @ plate_lin.T)
    quadratic_1 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q1)#(plate_lin @ q1 @ plate_lin.T)
    quad_correction = np.array([quadratic_0, quadratic_1]).T
    
    corrected = plate_lin  # TODO: plus cubic term...

    # step 3: To spherical coordinates
    icoords = corrected * pixel_scale
    icoords[:, 1] = icoords[:, 1] / np.cos(icoords[:, 0]) # spherical coordinate curveture
    icoords[:, 0] += np.pi/2
    
    vector_positions_z = np.cos(icoords[:, 0])
    vector_positions_x = np.sin(icoords[:, 0]) * np.cos(icoords[:, 1])
    vector_positions_y = np.sin(icoords[:, 0]) * np.sin(icoords[:, 1])

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

data_path = "D:\output\STACKED_CENTROIDS_MATCHED_ID1705968683.6189432.csv"
image_size = (6388, 9576)

df = pd.read_csv(data_path)
df['vz'] = np.cos(np.radians(df['DEC']))
df['vx'] = np.sin(np.radians(df['DEC'])) * np.cos(np.radians(df['RA']))
df['vy'] = np.sin(np.radians(df['DEC'])) * np.sin(np.radians(df['RA']))

print(df)


target = np.array([df['vx'], df['vy'], df['vz']]).T
plate = np.array([df['py'], df['px']]).T
print(plate.shape)
print(target.shape)

f = get_fitfunc(plate, target, image_size)
#print(target)
#print(f((9e-6, 326.35/360*6.282, 44.81/360*6.282, 178/360*6.282)))

result = scipy.optimize.minimize(get_fitfunc(plate, target, image_size), (9e-6, 44.81/360*6.282, 326.35/360*6.282, 178/360*6.282, 1, 0, 0,0,0,0,0,0))  
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
