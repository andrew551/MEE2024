"""
@author: Andrew Smith
Version 6 May 2024
"""

import pandas as pd
import numpy as np
import json
import zipfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    starttime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(path_data)
    archive = zipfile.ZipFile(path_data, 'r')
    data = json.load(archive.open('distortion_results.txt'))
    #image_size = data['img_shape']
    df = pd.read_csv(archive.open('CATALOGUE_MATCHED_ERRORS.csv'))
    print(df)
    df = df.astype({'px':float, 'py':float, 'RA(catalog)':float, 'RA(obs)':float, 'DEC(catalog)':float, 'DEC(obs)':float, 'magV':float}) # fix datatypes
    df = df.loc[df['magV'] <= options['eclipse_limiting_mag']]
    print(df)
    if options['remove_double_stars_eclipse']:
        df = df.loc[~df['flag_is_double']]
    #df = df.drop(df[df['ID'] == 'gaia:2577063532462714368'].index)
    #print(data)
    #print(df)

    
    if data['gravitational correction enabled?']:
        print('WARNING: gravity was enabled in the input data')
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
    

    sun_v = as_unit_vector(sun.dec.radian, sun.ra.radian)
    moon_v = as_unit_vector(moon.dec.radian, moon.ra.radian)

    reference_v = moon_v if options['object_centre_moon'] else sun_v

    stars_obs_v = as_unit_vector(np.radians(df['DEC(obs)']), np.radians(df['RA(obs)']))
    stars_cata_v = as_unit_vector(np.radians(df['DEC(catalog)']), np.radians(df['RA(catalog)']))
    radial_distances_catalog = np.arcsin(np.linalg.norm(stars_cata_v - reference_v, axis=1) / 2) * 2
    rad_dist = np.degrees(radial_distances_catalog) / sun_apparent_angular_radius
    delta_vectors = stars_cata_v - reference_v

    
    print("radial distances (in solar radii)", rad_dist)


    if options['limit_radial_sun_radii']:
        print("limiting radii to",  options['limit_radial_sun_radii_value'])
        mask = rad_dist < options['limit_radial_sun_radii_value']
        df_rem = df.loc[~mask]
        df = df.loc[mask]
        stars_obs_v  =stars_obs_v[mask]
        stars_cata_v = stars_cata_v[mask]
        radial_distances_catalog = radial_distances_catalog[mask]
        rad_dist = rad_dist[mask]
        delta_vectors = delta_vectors[mask]
    else:
        print("not limiting radii")

    field_describe = f"Eclipse Field with {df.shape[0]} chosen stars with magnitudes {np.min(df['magV']):.1f} to {np.max(df['magV']):.1f}"
    if options['flag_display3']:
        fig, ax = plt.subplots()
        ax.scatter(df['RA(catalog)'], df['DEC(catalog)'], color='blue', label = 'catalog')
        ax.scatter(df['RA(obs)'], df['DEC(obs)'], marker='+', color='orange', label = 'observation (used)')
        if options['limit_radial_sun_radii']:
            ax.scatter(df_rem['RA(obs)'], df_rem['DEC(obs)'], marker='x', color='red', label = 'observation (excluded)')
        ax.set_title(field_describe, fontsize=16)
        ax.set_xlabel("RA (degrees)", fontsize=16)
        ax.set_ylabel("DEC (degrees)", fontsize=16)
        sun_circle = plt.Circle((sun.ra.degree, sun.dec.degree), sun_apparent_angular_radius, color='yellow') # NOTE: sun and moon are not actually circles in RA/DEC space!
        moon_circle = plt.Circle((moon.ra.degree, moon.dec.degree), moon_apparent_angular_radius, color='black')
        if not options['object_centre_moon']:
            ax.add_patch(sun_circle)
        ax.add_patch(moon_circle)
        ax.legend()
        ax.set_aspect('equal')
        for id_i, ra_i, dec_i, mag_i in zip(df['ID'], df['RA(catalog)'], df['DEC(catalog)'], df['magV']):
            ax.annotate(f'    {id_i[5:]} mag={mag_i:.1f}', (ra_i, dec_i), fontsize=6)

        '''
        # inset Axes....
        x1, x2, y1, y2 = 18.4, 18.5, 7.7, 7.8  # subregion of the original image
        axins = ax.inset_axes(
            [0.85, 1.05, 0.3, 0.5],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        #axins.imshow(Z2, extent=extent, origin="lower")
        axins.scatter(df['RA(catalog)'], df['DEC(catalog)'], color='blue', label = 'catalog')
        axins.scatter(df['RA(obs)'], df['DEC(obs)'], marker='+', color='orange', label = 'observation (used)')
        for id_i, ra_i, dec_i, mag_i in zip(df['ID'], df['RA(catalog)'], df['DEC(catalog)'], df['magV']):
            axins.annotate(f'    {id_i[5:]} mag={mag_i:.1f}', (ra_i, dec_i), fontsize=6)
        ax.indicate_inset_zoom(axins, edgecolor="black")
        '''
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    
    # deflection: d = A / r
    def error_function1(deflection_const, return_rotation=False):
        deflection = np.radians(deflection_const / rad_dist / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        if return_rotation:
            return rms, rot
        return rms

    # allow power-law relationship for deflection: d = A / r^B
    def error_function2(deflection_const, return_rotation=False):
        deflection = np.radians(deflection_const[0] / rad_dist**deflection_const[1] / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        if return_rotation:
            return rms, rot
        return rms

        # deflection: d = A / r + B * r
    def error_function3(deflection_const, return_rotation=False):
        deflection = np.radians(deflection_const[0] / rad_dist / 3600 + deflection_const[1] * rad_dist / 3600)
        delta_vectors_unit = delta_vectors / np.linalg.norm(delta_vectors, axis = 1).reshape(delta_vectors.shape[0], 1)
        cata_vectors_corrected = stars_cata_v + deflection.reshape(delta_vectors.shape[0], 1) * delta_vectors_unit
        cata_vectors_corrected = cata_vectors_corrected / np.linalg.norm(cata_vectors_corrected, axis = 1).reshape(delta_vectors.shape[0], 1)
        rot = _find_rotation_matrix(stars_obs_v, cata_vectors_corrected)
        corrected = (rot.T @ stars_obs_v.T).T
        rms = np.degrees(np.sqrt(np.linalg.norm(corrected - cata_vectors_corrected)**2 / stars_cata_v.shape[0]))*3600
        if return_rotation:
            return rms, rot
        return rms
    
    
    result1 = scipy.optimize.minimize(error_function1, 0, method = 'Nelder-Mead')
    print(result1)

    result2 = scipy.optimize.minimize(error_function2, (0, 1), method = 'Nelder-Mead')
    print(result2)

    result3 = scipy.optimize.minimize(error_function3, (0, 0), method = 'Nelder-Mead')
    print(result3)
    ### rms minimisation curve
    if options['flag_display3']:    
        xxx = np.linspace(-0.25, 3)
        yyy = [error_function1(_)for _ in xxx]
        plt.plot(xxx, yyy)
        plt.xlabel("deflection constant (arcsec)")
        plt.ylabel("rms (arcsec)")
        
    rms, rot = error_function1(result1.x[0], return_rotation=True)
    naive_error = result1.fun/np.sqrt(df.shape[0])
    string = f"deflection constant = {result1.x[0]:.5f}\ndifference vs. accepted value: {100*(result1.x[0]-1.751)/1.751:.3f}%\n\ndeflected star position rms = {result1.fun:.3f} arcsec\nrms / sqrt(nstars) = {naive_error:.5f} arcsec\nnaive uncertainty estimate = {100*naive_error/1.751:.1f}%\n"
    obs_rot = (rot.T @ stars_obs_v.T).T
    radial_distances_obs = np.arcsin(np.linalg.norm(obs_rot - reference_v, axis=1) / 2) * 2

    deflection_obs = np.degrees(radial_distances_obs - radial_distances_catalog)*3600
    #if data['gravitational correction enabled?']:
    #    deflection_obs += 1.751 / rad_dist
    if options['flag_display3']:        
        plt.annotate(string, xy = (result1.x[0], result1.fun), xytext=(result1.x[0]-0.3, result1.fun+0.05), fontsize=14, arrowprops=dict(facecolor='black', shrink=0.05))
        plt.title(f"Least-squares deflection fit for {df.shape[0]} chosen stars with magnitudes {np.min(df['magV']):.1f} to {np.max(df['magV']):.1f}")
        plt.show()
    ### scatter of deflection vs. radius
    plt.axhline(0, color='black')
    plt.scatter(rad_dist, deflection_obs)
    plt.ylabel("radial deflection (arcsec)")
    plt.xlabel("radial position (solar radii)")
    
    plt.annotate(f"L = {result1.x[0]:.5f}", (3, result1.x/3+0.3), fontsize=11)
    plt.title(field_describe)
    xx = np.linspace(np.min(rad_dist)-0.5, np.max(rad_dist))
    yy = result1.x[0] / xx
    plt.plot(xx, yy, color='black')
    plt.savefig(output_path(f'ECLIPSE_DEFLECTIONS{starttime}.png', options), dpi=400)
    if options['flag_display3']:      
        plt.show()
    plt.close()
    
    plt.scatter(1/rad_dist, deflection_obs)
    plt.ylabel("radial deflection (arcsec)")
    plt.xlabel("1/radial position (solar radii)")
    
    plt.annotate(f"L = {result1.x[0]:.5f}", (3, result1.x/3+0.3), fontsize=11)
    plt.title(field_describe)
    xx = np.linspace(np.min(rad_dist)-0.5, np.max(rad_dist))

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False).fit(1/rad_dist.reshape(-1, 1), deflection_obs)
    
    #yy = result1.x[0] / xx
    yy = reg.coef_[0] / xx
    print(reg)
    plt.annotate(f"L = {reg.coef_[0]:.5f}", (1/3, 0.5), fontsize=11)
    plt.plot(1/xx, yy, color='black')
    #plt.savefig(output_path(f'ECLIPSE_DEFLECTIONS{starttime}.png', options), dpi=400)
    if options['flag_display3']:      
        plt.show()


    output_name = f'ECLIPSE_OUTPUT{starttime}.txt'
    output_file = Path(output_path(output_name, options))
    print(output_file)
    with open(output_file, 'w') as f:
        f.write(f"MEE2024 version: {_version()}\n")
        f.write(f"input file: {path_data}\n\n")
        f.write(f"limiting magnitude: {options['eclipse_limiting_mag']}\n")
        if options['limit_radial_sun_radii']:
            f.write(f"cutoff sun radii: {options['limit_radial_sun_radii_value']}")
        else:
            f.write(f"cutoff sun radii: None")
        f.write(f"remove double stars: {options['remove_double_stars_eclipse']}\n")
        f.write(f"number of stars used: {df.shape[0]}\n")
        f.write(string)
        f.write(f"\na/R^b fit: a = {result2.x[0]:.3f}, b = {result2.x[1]:.3f}, rms = {result2.fun:.3f} arcsec\n\n")
        f.write("radial distances: " + str(rad_dist) + "\n\n")
        f.write("deflection (arcsec): " + str(deflection_obs)+"\n")

    
    r_0 = np.arcsin(np.linalg.norm(stars_obs_v - reference_v, axis=1) / 2) * 2
    deflection_obs_0 = np.degrees(radial_distances_obs - radial_distances_catalog)*3600
    plt.scatter(rad_dist, deflection_obs_0)

    xx = np.linspace(np.min(rad_dist)-0.5, np.max(rad_dist))
    '''
    reg = LinearRegression(fit_intercept=False).fit(1/rad_dist.reshape(-1, 1), deflection_obs_0)
    samples=[]
    for _ in range(1000):
        sample_ind = np.random.choice(rad_dist.size, rad_dist.size)
        sample_r = rad_dist[sample_ind]
        sample_d = deflection_obs_0[sample_ind]
        reg2 = LinearRegression(fit_intercept=False).fit(np.c_[1/sample_r.reshape(-1, 1), sample_r.reshape(-1, 1)], sample_d)
        print(reg2.coef_)
        samples.append(reg2.coef_)
    samples = np.array(samples)
    
    #yy = result1.x[0] / xx
    yy = reg.coef_[0] / xx
    print(reg.coef_)
    
    plt.annotate(f"L = {reg.coef_[0]:.5f}", (3, 1), fontsize=11)
    plt.plot(xx, yy, color='black')

    
    plt.show()
    plt.clf()

    plt.scatter(samples[:, 0], samples[:, 1])
    plt.show()
    '''
#def bootstrap(

    ###

    ###
    #https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    def confidence_ellipse(cov, mu, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The Axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        #if x.size != y.size:
        #    raise ValueError("x and y must be the same size")

        
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        print(pearson)
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mu[0]

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mu[1]
        #print(scale_x, scale_y, ell_radius_x, ell_radius_y)
        #print(mu)
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    '''

    fig, ax_nstd = plt.subplots(figsize=(6, 6))

    dependency_nstd = [[0.8, 0.75],
                       [-0.2, 0.35]]
    mu = 0, 0
    scale = 8, 5

    #ax_nstd.axvline(c='grey', lw=1)
    #ax_nstd.axhline(c='grey', lw=1)

    x, y = samples[:, 0], samples[:, 1] #get_correlated_dataset(500, dependency_nstd, mu, scale)
    ax_nstd.scatter(x, y, s=0.5)
    cov = np.cov(x, y)
    mu = [np.mean(x), np.mean(y)]
    confidence_ellipse(cov, mu, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(cov, mu, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(cov, mu, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    
    #ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.set_title('Different standard deviations')
    ax_nstd.legend()
    plt.show()
    '''

    import statsmodels.api as sm
    platescale0 = data['platescale (arcseconds/pixel)']
    model = sm.OLS(deflection_obs_0,np.c_[1/rad_dist.reshape(-1, 1), rad_dist.reshape(-1, 1)])
    results = model.fit()
    print(results.params)
    print(results.bse)
    print(results.summary())
    cov = results.cov_params()
    mu = results.params
    #mu[1] += platescale0
    print(results.cov_params())
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    print(mu)

    factor = 3600 / platescale0 * sun_apparent_angular_radius
    mu[1] /= factor
    cov[:, 1] /= factor
    cov[1, :] /= factor
    mu[1] += platescale0
    print("shifted and scaled mu")
    print(mu)
    print("scaled cov")
    print(cov)
    confidence_ellipse(cov, mu, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(cov, mu, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia')
    confidence_ellipse(cov, mu, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue')
    plt.scatter([mu[0]], [mu[1]], marker='+', s=100)
    ax_nstd.text(mu[0]+0.04, mu[1], f"({mu[0]:.3f}, {mu[1]:.6f})", fontsize=12, bbox=dict(boxstyle='square,pad=.3', facecolor='white', edgecolor='white'))
    ax_nstd.set_xlabel('L (arcsec / solar radius)', fontsize = 16)
    ax_nstd.set_ylabel('Plate Scale (arcsec / pixel)', fontsize=16)
    ax_nstd.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5f'))
    ax_nstd.set_title('Covariance of Deflection Constant and Plate Scale', fontsize=16)
    ax_nstd.legend(prop={'size': 12})
    plt.xlim(mu[0] - 4 * cov[0,0]**0.5, mu[0] + 4 * cov[0,0]**0.5)
    plt.ylim(mu[1] - 4 * cov[1,1]**0.5, mu[1] + 4 * cov[1,1]**0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


    deflections_corrected = deflection_obs_0 - (mu[1]-platescale0) * factor * rad_dist


    plt.clf()
    plt.axhline(0, color='black')
    plt.scatter(rad_dist, deflections_corrected)
    plt.ylabel("radial deflection (arcsec)", fontsize=16)
    plt.xlabel("radial position (solar radii)", fontsize=16)
    plt.xticks(np.arange(2, 12, 1.0), fontsize=14)
    plt.yticks(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.annotate(f"L = {mu[0]:.3f}", (3, 1.5), fontsize=16)
    plt.title('Deflections for ' + field_describe, fontsize=16)
    xx = np.linspace(np.min(rad_dist)-0.5, np.max(rad_dist))
    yy = mu[0] / xx
    plt.plot(xx, yy, color='black')
    plt.savefig(output_path(f'ECLIPSE_DEFLECTIONS_corrected{starttime}.png', options), dpi=400)
    if options['flag_display3']:      
        plt.show()
    plt.close()

    
    
if __name__ == '__main__':
    pass
    #eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323214311__data (2)20240317002546/distortion.zip', {}) # no cheat
    #eclipse_analysis('D:/eclipsetest/DISTORTION_OUTPUT20240323224137__data (2)20240317002546/distortion.zip', {}) # yes cheat
    eclipse_analysis('D:/Don 2017 eclipse data/DISTORTION_OUTPUT20240324141609__data_eclipse20240317002546/distortion.zip', {'output_dir':'D:/output4', 'flag_display3':True})
