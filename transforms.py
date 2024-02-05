import numpy as np
from scipy.spatial.transform import Rotation

def rotate_icoords(x, icoords):
    ra, dec, roll = x[0], x[1], x[2]
    icoords[:, 1] = icoords[:, 1] / np.cos(icoords[:, 0]) # spherical coordinate curveture
    
    vector_positions_z = np.sin(icoords[:, 0]) # z -> declination
    vector_positions_x = np.cos(icoords[:, 0]) * np.cos(icoords[:, 1])
    vector_positions_y = np.cos(icoords[:, 0]) * np.sin(icoords[:, 1]) # y -> right ascension

    plate_vectors = np.array([vector_positions_x, vector_positions_y, vector_positions_z]).T
    
    # apply roll, then RA, then declination
    #r = Rotation.from_euler('xyz', [roll, ra-np.pi/2, dec])
    r = Rotation.from_euler('xyz', [roll, dec-np.pi/2, ra])
    rotated = r.apply(plate_vectors)
    return rotated

# perform a general transform with rotation and (shearless) scaling
# so 3 + 1 = 4 degrees of freedom in x
def linear_transform(x, q, img_shape):

    pixel_scale = x[0] # radians per pixel
    # rotation  
    

    corrected = np.copy(q)
    corrected -= np.array([img_shape[0]/2, img_shape[1]/2])

    icoords = corrected * pixel_scale
    return rotate_icoords(x[1:4], icoords)

# perform a general transform with rotation and (shearless) scaling
# so 3 + 1 + 6 = 10 degrees of freedom in x

def quadratic_transform(x, coords, img_shape):
    plate_lin = np.copy(coords)
    plate_lin -= np.array([img_shape[0]/2, img_shape[1]/2])

    # step 2: quadratic correction
    
    q0 = np.array([[4*x[4]/img_shape[0]**2, 2*x[5]/img_shape[0]/img_shape[1]], [2*x[5]/img_shape[0]/img_shape[1], 4*x[6]/img_shape[1]**2]])
    q1 = np.array([[4*x[7]/img_shape[0]**2, 2*x[8]/img_shape[0]/img_shape[1]], [2*x[8]/img_shape[0]/img_shape[1], 4*x[9]/img_shape[1]**2]])
    quadratic_0 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q0)#(plate_lin @ q0 @ plate_lin.T)
    quadratic_1 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q1)#(plate_lin @ q1 @ plate_lin.T)
    quad_correction = np.array([quadratic_0, quadratic_1]).T

    corrected = plate_lin + quad_correction
    icoords = corrected * pixel_scale
    return rotate_icoords(x[1:4], icoords)


'''
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
'''
