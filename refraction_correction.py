from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz
import numpy as np
import matplotlib.pyplot as plt
import erfa
import copy
import transforms




def as_unit_vector(c):
    return np.array([np.sin(c.alt.radian), np.cos(c.alt.radian) * np.cos(c.az.radian), np.cos(c.alt.radian) * np.sin(c.az.radian)]).T

# lifted from tetra3
def _find_rotation_matrix(image_vectors, catalog_vectors):
    """Calculate the least squares best rotation matrix between the two sets of vectors.
    image_vectors and catalog_vectors both Nx3. Must be ordered as matching pairs.
    """
    # find the covariance matrix H between the image and catalog vectors
    H = np.dot(image_vectors.T, catalog_vectors)
    # use singular value decomposition to find the rotation matrix
    (U, S, V) = np.linalg.svd(H)
    return np.dot(U, V)

has_hacked = False

class AstroCorrect:

    def __init__(self):
        global has_hacked
        if not has_hacked:  
            #HACK to remove light deflection calculation from astropy
            # the first argument of the erfa.ld function is the mass of the light-deflecting object ... we now set the mass to zero whenever this function is called by astropy
            origin = erfa.ld
            def no_grav_ld(bm, *args):
                return origin(0, *args)
            erfa.ld = no_grav_ld
        has_hacked = True

    # TODO: add distance to compute parallax
    def correct_ra_dec(self, stardata, options):
        #print(lat, lon)
        observing_location = EarthLocation(lat=options['observation_lat'], lon=options['observation_long'], height=options['observation_height']*u.m)  
        observing_time = Time(options['observation_date'] + ' ' + options['observation_time'])
        icrs_v = stardata.get_vectors()
        if options['enable_corrections_ref']:
            aa = AltAz(location=observing_location, obstime=observing_time,
                       obswl=options['observation_wavelength']*u.micron, pressure=options['observation_pressure']*u.hPa,
                       relative_humidity=options['observation_humidity']*u.m/u.m, temperature=options['observation_temp']*u.deg_C)
        else:
            aa = AltAz(location=observing_location, obstime=observing_time)
        coord = SkyCoord(stardata.get_ra() * u.rad, stardata.get_dec() * u.rad)
        local = coord.transform_to(aa)
        print('sky mean position alt/az:', np.mean(local.alt.degree), np.mean(local.az.degree)) 
        local_v = as_unit_vector(local)
        rot = _find_rotation_matrix(local_v, icrs_v)
        corrected = (rot.T @ local_v.T).T
        delta = corrected - icrs_v
        print('rms diff of corrections (arcsec)', np.degrees(np.linalg.norm(delta)/delta.shape[0])*3600)
        print(repr(stardata.data[:5, :5]))
        ret = copy.copy(stardata) # note this is shallow copy
        ret.data[:, 0] = np.arctan2(corrected[:, 1], corrected[:, 0]) # RA
        ret.data[:, 0] += (ret.data[:, 0] < 0) * 2 * np.pi # 0 to 2pi convention
        ret.data[:, 1] = np.arctan(corrected[:, 2] / np.sqrt(corrected[:, 0]**2 + corrected[:, 1]**2)) # DEC
        ret.data[:, 2:5] = corrected
        print(repr(ret.data[:5, :5]))
        return ret, np.mean(local.alt.degree), np.mean(local.az.degree)
        

# https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html
# AltAz can correct for aberration (always), parallax (if distance is given), refraction (given P, T, humidity, and lambda), and light defection
# we don't want light deflection correction
# https://github.com/astropy/astropy/blob/228452378ef7473d18ec967165996a8167bb6015/astropy/coordinates/builtin_frames/icrs_cirs_transforms.py#L26
# ICRS -> (parallax, aberration, deflection) -> GCRS -> AltAz
# TODO: figure out how to go back from AltAz to "apparent" dec, ra
if __name__ == '__main__':
    def square(x):
        return x ** 2

    import math

    func = math.sqrt
    def test():
        print(math.sqrt(4))
    def test2():
        print(func(4))
    test()
    test2()
    math.sqrt = square
    test()
    test2()
    print(help(erfa.ld))

    #HACK to remove light deflection calculation from astropy
    # the first argument of the erfa.ld function is the mass of the light-deflecting object ... we now set the mass to zero whenever this function is called by astropy
    origin = erfa.ld
    def no_grav_ld(bm, *args):
        return origin(0, *args)
    erfa.ld = no_grav_ld
    
    
    
    mm = list(range(1, 10))
    xx = []
    for m in mm:
    
        observing_location = EarthLocation(lat='52.2532', lon='351.63910339111703', height=100*u.m)  
        observing_time = Time(f'2017-0{m}-05 20:12:18')  
        aa = AltAz(location=observing_location, obstime=observing_time)
        
        coord = SkyCoord('4h42m', '-38d6m50.8s')
        coord2 = SkyCoord('10h42m', '-18d6m50.8s')
        print(coord, coord2)
        local = coord.transform_to(aa)
        local2 = coord2.transform_to(aa)
        print(local)
        #print(local.ra)
        #print(local2)
        #print(local.az.radian, local.alt.radian)
        #print(local2.az.radian, local2.alt.radian)

        v1 = as_unit_vector(local)
        v2 = as_unit_vector(local2)
        #print(v1, v2)
        x = np.arccos(np.dot(v1, v2))
        xx.append(np.degrees(x)*3600)
    plt.plot(mm, xx)
    plt.show()
    

    
