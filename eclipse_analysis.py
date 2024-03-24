import pandas as pd
import numpy as np
import json
import zipfile
import matplotlib.pyplot as plt
from refraction_correction import _find_rotation_matrix
from transforms import to_polar

from astropy.coordinates import EarthLocation,SkyCoord, Distance, get_body, AltAz
from astropy.time import Time
from astropy import units as u

def as_unit_vector(dec, ra):
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T 

def eclipse_analysis(path_data, options):
    print(path_data)
    archive = zipfile.ZipFile(path_data, 'r')

    data = json.load(archive.open('distortion/distortion_results.txt'))
    #image_size = data['img_shape']
    df = pd.read_csv(archive.open('distortion/CATALOGUE_MATCHED_ERRORS.csv'))
    df = df.astype({'px':float, 'py':float, 'RA(catalog)':float, 'RA(obs)':float, 'DEC(catalog)':float, 'DEC(obs)':float,}) # fix datatypes

    print(data)
    print(df)

    
    if data['gravitational correction enabled?']:
        print('warning: grav enabled')
        #raise Exception("expected 'gravitational correction enabled?':False")

    observing_location = EarthLocation(lat=data['observation_lat (degrees)'], lon=data['observation_long (degrees)'], height=data['observation_height (m)']*u.m)  
    observing_time = Time(data['observation_date'] + ' ' + data['observation_time (UTC)'])
    sun = get_body("sun", observing_time, observing_location)
    moon = get_body("moon", observing_time, observing_location)
    print(sun)
    print(moon)

    aa = AltAz(location=observing_location, obstime=observing_time)

    local_sun = sun.transform_to(aa)
    local_moon = moon.transform_to(aa)
    print(local_sun)
    print(local_moon)
    #print(local_sun.ra, local_sun.dec)
    #print(local_moon.ra, local_moon.dec)
    fig, ax = plt.subplots()
    ax.scatter(df['RA(catalog)'], df['DEC(catalog)'], color='blue', label = 'catalog')
    ax.scatter(df['RA(obs)'], df['DEC(obs)'], marker='+', color='orange', label = 'observation')

    sun_circle = plt.Circle((sun.ra.degree, sun.dec.degree), 32/60/2, color='yellow') # NOTE: sun and moon are not actually circles in RA/DEC space!
    moon_circle = plt.Circle((moon.ra.degree, moon.dec.degree), 34/60/2, color='black')
    ax.add_patch(sun_circle)
    ax.add_patch(moon_circle)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    sun_v = as_unit_vector(sun.dec.radian, sun.ra.radian)
    moon_v = as_unit_vector(moon.dec.radian, sun.ra.radian)

    stars_obs_v = as_unit_vector(np.radians(df['DEC(obs)']), np.radians(df['RA(obs)']))
    stars_cata_v = as_unit_vector(np.radians(df['DEC(catalog)']), np.radians(df['RA(catalog)']))
    radial_distances_obs = np.arcsin(np.linalg.norm(stars_obs_v - sun_v, axis=1) / 2) * 2
    radial_distances_catalog = np.arcsin(np.linalg.norm(stars_cata_v - sun_v, axis=1) / 2) * 2
    print(stars_obs_v)
    print(stars_cata_v)
    print(radial_distances_obs, radial_distances_catalog)

    rad_dist = np.degrees(radial_distances_catalog) * (32/60/2)**-1
    deflection = np.degrees(radial_distances_obs - radial_distances_catalog)*3600
    if data['gravitational correction enabled?']:
        deflection += 1.751 / rad_dist
    plt.scatter(rad_dist, deflection)
    plt.ylabel("radial deflection (arcsec)")
    plt.xlabel("radial position (sun radii)")

    xx = np.linspace(1, 5)
    yy = 1.751 / xx
    plt.plot(xx, yy)
    
    plt.show()

    delta_vectors = stars_cata_v - sun_v
    
    e0 = np.degrees(np.sqrt(np.linalg.norm(stars_obs_v - stars_cata_v)**2 / stars_cata_v.shape[0]))*3600
    print(e0)
    print(rad_dist)
    def error_function1(deflection_const):
        #stars_obs_v = as_unit_vector(np.radians(df['DEC(obs)']+ddec), np.radians(df['RA(obs)']+dra))
        #radial_distances_obs = np.arcsin(np.linalg.norm(stars_obs_v - sun_v, axis=1) / 2) * 2
        
        #radial_distances_catalog = np.arcsin(np.linalg.norm(delta_vectors, axis=1) / 2) * 2
        #rad_dist = np.degrees(radial_distances_catalog) * (32/60/2)**-1
        #deflection = np.degrees(radial_distances_obs - radial_distances_catalog)*3600
        deflection = np.radians(deflection_const / rad_dist / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        #print(deflection)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        '''
        q1 = to_polar(cata_vectors_corrected)
        q2 = to_polar(stars_cata_v)
        plt.scatter(q1[:, 1], q1[:, 0], color='blue', marker='+')
        plt.scatter(q2[:, 1], q2[:, 0], color='orange', marker='+')
        plt.show()
        '''
        
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        return rms
    print(error_function1(0))
    
    xxx = np.linspace(-1, 3)
    yyy = [error_function1(_)for _ in xxx]
    plt.plot(xxx, yyy)
    plt.show()
    
        
if __name__ == '__main__':
    #eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323214311__data (2)20240317002546/distortion.zip', {}) # no cheat
    eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323224137__data (2)20240317002546/distortion.zip', {}) # yes cheat
    #eclipse_analysis('
