import numpy as np
from scipy.spatial.transform import Rotation

'''
input: cartesian 3-unit-vectors
output: 2-vectors of polar coordinates in degrees
'''
def to_polar(v):
    v = v.reshape((-1, 3))
    theta = np.arcsin(v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])
    phi[phi < 0] += np.pi * 2
    ret = np.degrees(np.array([theta, phi]))
    
    return ret.T

'''
Transform back from celestial 3-vectors to pixel-like coordinates
inputs:
x: (platescale, coordinate) 4-tuple
v: array of shape (n, 3) : n 3-vectors of star positions
outputs: array of shape (n, 2): n 2-vectors of intermediate (i.e. pixel-like) coordinates
'''
def detransform_vectors(x, v):
    scale, ra, dec, roll = x[0], x[1], x[2], x[3]

    r = Rotation.from_euler('zyx', [-ra, dec, -roll])
    rotated = r.apply(v)

    icoord0 = np.arcsin(rotated[:, 2])

    icoord1 = np.arcsin(rotated[:, 1] / np.cos(icoord0))
    icoord1 *= np.cos(icoord0)

    return np.array([icoord0, icoord1]).T / scale

'''
transform from intermediate "rectilinear" coordinate system icoords to
3-vector coordinate system (with (0, 0) -> (1, 0, 0))
'''

def icoord_to_vector(icoords):
    initial_shape = icoords.shape
    if not initial_shape[-1] == 2:
        raise Exception("Last dimension of shape of input must be 2!")
    icoords = icoords.reshape((-1, 2))
    icoords[:, 1] = icoords[:, 1] / np.cos(icoords[:, 0]) # spherical coordinate curveture
    
    vector_positions_z = np.sin(icoords[:, 0]) # z -> declination
    vector_positions_x = np.cos(icoords[:, 0]) * np.cos(icoords[:, 1])
    vector_positions_y = np.cos(icoords[:, 0]) * np.sin(icoords[:, 1]) # y -> right ascension
    newshape = list(initial_shape)
    newshape[-1]  = 3
    return np.array([vector_positions_x, vector_positions_y, vector_positions_z]).T.reshape(tuple(newshape))
    
'''
transform from intermediate "rectilinear" coordinate system icoords to 
3-vector true coordinates given (ra, dec, roll) in x 
'''
def rotate_icoords(x, icoords):
    ra, dec, roll = x[0], x[1], x[2]
    plate_vectors = icoord_to_vector(icoords)
    # apply roll, then declination, then RA
    r = Rotation.from_euler('xyz', [roll, -dec, ra])
    rotated = r.apply(plate_vectors)
    return rotated

'''
perform a coordinate transform with rotation (ra, dec, roll) and (shearless) scaling
so 3 + 1 = 4 degrees of freedom in x
'''
def linear_transform(x, q, img_shape=None):

    pixel_scale = x[0] # radians per pixel
    icoords = q * pixel_scale
    return rotate_icoords(x[1:4], icoords)

'''
# all following functions are now unused

# allows for shear and stretch
def mixed_linear_transform(x, q, img_shape=None):
    #matrix = np.array([[x[0], x[1]], [x[2], x[3]]])
    matrix = np.array([[x[0]+1, x[1]], [0, 1 / (x[0] + 1)]])
    corrected = np.einsum('ij,kj->ki', matrix, q)
    return linear_transform(x[2:], corrected)


# add img_shape to use normalised quantities
def brown_distortion(x, q, img_shape):
    K1, K2, K3, P1, P2 = x[0], x[1], x[2], x[3], x[4]

    w = q / max(img_shape) * 2

    r2 = w[:, 0]**2 + w[:, 1]**2
    r4 = r2*r2
    r6 = r4*r2

    # the radial function is based on even Legendre polynomials, with the constant removed
    radial = K1 * (3*r2)/2 + K2 * (35*r4-30*r2)/8 + K3 * (231*r6-315*r4+105*r2)/16
    
    correction0 = w[:, 0] * radial + P1 * (r2 + 2 * w[:, 0]**2) + 2 * P2 * w[:, 0]*w[:, 1]
    correction1 = w[:, 1] * radial + P2 * (r2 + 2 * w[:, 1]**2) + 2 * P1 * w[:, 0]*w[:, 1]

    q_corrected=np.copy(q)
    q_corrected[:, 0] += correction0
    q_corrected[:, 1] += correction1
    return linear_transform(x[5:], q_corrected)

def skew_distortion(x, q, img_shape):

    w = q / max(img_shape) * 2
    correction0 = w[:, 0] ** 3 * x[0] + w[:, 0] * w[:, 1]**2 * x[1]
    correction1 = w[:, 1] ** 3 * x[2] + w[:, 1] * w[:, 0]**2 * x[3]

    q_corrected=np.copy(q)
    q_corrected[:, 0] += correction0
    q_corrected[:, 1] += correction1
    return linear_transform(x[4:], q_corrected)
    

def cubic_distortion(x, q, img_shape):

    w = q / max(img_shape) * 2
    c0 = np.zeros((2, 2, 2))
    c0[0, 0, 0] = x[0]
    c0[1, 0, 0] = x[1] / 3
    c0[0, 1, 0] = c0[1, 0, 0]
    c0[0, 0, 1] = c0[1, 0, 0]
    c0[1, 1, 0] = x[2] / 3
    c0[0, 1, 1] = c0[1, 1, 0]
    c0[1, 0, 1] = c0[1, 1, 0]
    c0[1, 1, 1] = x[3]
    c1 = np.zeros((2, 2, 2))
    c1[0, 0, 0] = x[4]
    c1[1, 0, 0] = x[5] / 3
    c1[0, 1, 0] = c1[1, 0, 0]
    c1[0, 0, 1] = c1[1, 0, 0]
    c1[1, 1, 0] = x[6] / 3
    c1[0, 1, 1] = c1[1, 1, 0]
    c1[1, 0, 1] = c1[1, 1, 0]
    c1[1, 1, 1] = x[7]
    
    cubic_0 = np.einsum('ij,ik,il,jkl->i', w, w, w, c0)
    cubic_1 = np.einsum('ij,ik,il,jkl->i', w, w, w, c1)
    cubic_correction = np.array([cubic_0, cubic_1]).T

    #matrix = np.array([[x[8], x[9]], [x[10], x[11]]])
    #lin_correction = np.einsum('ij,kj->ki', matrix, q)
    
    q_corrected = q + cubic_correction #+ lin_correction #+ quad_correction+cubic_correction # TODO: plus cubic term...
    return linear_transform(x[8:], q_corrected)

# perform a general transform with rotation and (shearless) scaling
# so 3 + 1 + 6 = 10 degrees of freedom in x

def quadratic_transform(x, coords, img_shape):
    plate_lin = np.copy(coords)
    #plate_lin -= np.array([img_shape[0]/2, img_shape[1]/2])

    # step 2: quadratic correction
    
    q0 = np.array([[4*x[4]/img_shape[0]**2, 2*x[5]/img_shape[0]/img_shape[1]], [2*x[5]/img_shape[0]/img_shape[1], 4*x[6]/img_shape[1]**2]])
    q1 = np.array([[4*x[7]/img_shape[0]**2, 2*x[8]/img_shape[0]/img_shape[1]], [2*x[8]/img_shape[0]/img_shape[1], 4*x[9]/img_shape[1]**2]])
    quadratic_0 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q0)#(plate_lin @ q0 @ plate_lin.T)
    quadratic_1 = np.einsum('ij,ik,jk->i', plate_lin, plate_lin, q1)#(plate_lin @ q1 @ plate_lin.T)
    quad_correction = np.array([quadratic_0, quadratic_1]).T

    corrected = plate_lin + quad_correction
    icoords = corrected * pixel_scale
    return rotate_icoords(x[1:4], icoords)

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
