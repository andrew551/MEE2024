import pandas as pd
import numpy as np
import json
import zipfile
import matplotlib.pyplot as plt
from refraction_correction import _find_rotation_matrix
from transforms import to_polar
import scipy
from pathlib import Path
import datetime
from MEE2024util import output_path, _version

import astropy
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

    #print(data)
    #print(df)

    
    if data['gravitational correction enabled?']:
        print('warning: grav enabled')
        #raise Exception("expected 'gravitational correction enabled?':False")

    observing_location = EarthLocation(lat=data['observation_lat (degrees)'], lon=data['observation_long (degrees)'], height=data['observation_height (m)']*u.m)  
    observing_time = Time(data['observation_date'] + ' ' + data['observation_time (UTC)'])
    sun = get_body("sun", observing_time, observing_location)
    moon = get_body("moon", observing_time, observing_location)
    print(sun)
    print(moon)

    sun_apparent_angular_radius = np.degrees((astropy.constants.R_sun / sun.distance).to(u.dimensionless_unscaled).value)
    moon_apparent_angular_radius = np.degrees((1737.4*u.km / moon.distance).to(u.dimensionless_unscaled).value)
    print(sun_apparent_angular_radius)
    print(moon_apparent_angular_radius)
    aa = AltAz(location=observing_location, obstime=observing_time)

    local_sun = sun.transform_to(aa)
    local_moon = moon.transform_to(aa)
    print(local_sun)
    print(local_moon)
    #print(local_sun.ra, local_sun.dec)
    #print(local_moon.ra, local_moon.dec)
    if options['flag_display3']:
        fig, ax = plt.subplots()
        ax.scatter(df['RA(catalog)'], df['DEC(catalog)'], color='blue', label = 'catalog')
        ax.scatter(df['RA(obs)'], df['DEC(obs)'], marker='+', color='orange', label = 'observation')
        ax.set_title(f"Eclipse Field with {df.shape[0]} chosen stars")
        ax.set_xlabel("RA (degrees)")
        ax.set_ylabel("DEC (degrees)")
        sun_circle = plt.Circle((sun.ra.degree, sun.dec.degree), sun_apparent_angular_radius, color='yellow') # NOTE: sun and moon are not actually circles in RA/DEC space!
        moon_circle = plt.Circle((moon.ra.degree, moon.dec.degree), moon_apparent_angular_radius, color='black')
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
    rad_dist = np.degrees(radial_distances_catalog) / sun_apparent_angular_radius
    #print(stars_obs_v)
    #print(stars_cata_v)
    #print(radial_distances_obs, radial_distances_catalog)

    '''
    
    deflection_obs = np.degrees(radial_distances_obs - radial_distances_catalog)*3600
    if data['gravitational correction enabled?']:
        deflection_obs += 1.751 / rad_dist
    plt.scatter(rad_dist, deflection_obs)
    plt.ylabel("radial deflection (arcsec)")
    plt.xlabel("radial position (sun radii)")

    xx = np.linspace(1, 5)
    yy = 1.751 / xx
    plt.plot(xx, yy)
    
    plt.show()
    '''

    delta_vectors = stars_cata_v - sun_v
    
    #e0 = np.degrees(np.sqrt(np.linalg.norm(stars_obs_v - stars_cata_v)**2 / stars_cata_v.shape[0]))*3600
    #print(e0)
    print("radial distances (in solar radii)", rad_dist)

    # deflection: d = A / r
    def error_function1(deflection_const):
        deflection = np.radians(deflection_const / rad_dist / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        return rms

    # allow power-law relationship for deflection: d = A / r^B
    def error_function2(deflection_const):
        deflection = np.radians(deflection_const[0] / rad_dist**deflection_const[1] / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        return rms
    
    result1 = scipy.optimize.minimize(error_function1, 0, method = 'Nelder-Mead')
    print(result1)

    result2 = scipy.optimize.minimize(error_function2, (0, 1), method = 'Nelder-Mead')
    print(result2)

    if options['flag_display3']:    
        xxx = np.linspace(-0.25, 3)
        yyy = [error_function1(_)for _ in xxx]
        plt.plot(xxx, yyy)
        plt.xlabel("deflection constant (arcsec)")
        plt.ylabel("rms (arcsec)")

    naive_error = result1.fun/np.sqrt(df.shape[0])
    string = f"deflection constant = {result1.x[0]:.5f}\ndifference vs. accepted value: {100*(result1.x[0]-1.751)/1.751:.3f}%\n\ndeflected star position rms = {result1.fun:.3f} arcsec\nrms / sqrt(nstars) = {naive_error:.5f} arcsec\nnaive error estimate = {100*naive_error/1.751:.1f}%\n"
    if options['flag_display3']:
        plt.annotate(string, xy = (result1.x[0], result1.fun), xytext=(result1.x[0]-0.3, result1.fun+0.2), fontsize=14, arrowprops=dict(facecolor='black', shrink=0.05))
        plt.title("Least-squares deflection fit")
        plt.show()


    starttime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f'ECLIPSE_OUTPUT{starttime}.txt'
    output_file = Path(output_path(output_name, options))
    print(output_file)
    with open(output_file, 'w') as f:
        f.write(f"MEE2024 version: {_version()}\n")
        f.write(f"input file: {path_data}\n")
        f.write(string)
        f.write(f"\n\na/R^b fit: a = {result2.x[0]:.3f}, b = {result2.x[1]:.3f}, rms = {result2.fun:.3f} arcsec\n")
        
if __name__ == '__main__':
    pass
    #eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323214311__data (2)20240317002546/distortion.zip', {}) # no cheat
    #eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323224137__data (2)20240317002546/distortion.zip', {}) # yes cheat
    eclipse_analysis('D:/Don 2017 eclipse data/DISTORTION_OUTPUT20240324141609__data_eclipse20240317002546/distortion.zip', {'output_dir':'D:/output4', 'flag_display3':True})
