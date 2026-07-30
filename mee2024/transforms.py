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
    # copy: reshape can return a view, and the line below writes to the array
    icoords = np.array(icoords, dtype=float).reshape((-1, 2))
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
