from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz
import numpy as np
import matplotlib.pyplot as plt
import erfa

def as_unit_vector(c):
    return np.array([np.sin(c.alt.radian), np.cos(c.alt.radian) * np.cos(c.az.radian), np.cos(c.alt.radian) * np.sin(c.az.radian)])



class AstroCorrect:

    def __init__(self):
        pass

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
    

    
