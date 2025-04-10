"""
@author: Andrew Smith
Version 6 May 2024
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import transforms
import matplotlib.pyplot as plt
import scipy
import datetime
from MEE2024util import date_string_to_float, date_from_float
from scipy.special import legendre
import copy
import json
from collections import defaultdict
import zipfile
import statsmodels.api as sm

mapping = {'constant':0, 'linear':1, 'quadratic':2, 'cubic':3, 'quartic': 4, 'quintic':5, 'sextic': 6, 'septic':7}
legendre_polies = [legendre(i) for i in range(max(mapping.values())+1)]

def get_basis(y, x, img_shape, options, use_special=False):
    w = max(img_shape)/2
    basis = []
    order = mapping[options['distortionOrder']]
    if options['basis_type'] == 'polynomial':
        for i in range(1, order+1): # up to nth order binomials
            for j in range(i+1):
                basis.append(y ** j * x ** (i-j) / w**i)
        return np.array(basis).T
    elif options['basis_type'] == 'legendre':
        x_scaled = 2 * x / img_shape[1]
        y_scaled = 2 * y / img_shape[0]
        print("LEGENDRE BASIS MINMAX XY", np.max(x_scaled), np.min(x_scaled), np.max(y_scaled), np.min(y_scaled))
        
        for i in range(1, order+1): # up to nth order legendre binomials
            for j in range(i+1):
                if i == 1:
                    basis.append(y ** j * x ** (i-j) / w) # special case: the linear terms need to be like this or else code elsewhere will break
                else:
                    basis.append(legendre_polies[j](y_scaled) * legendre_polies[i-j](x_scaled))
        return np.array(basis).T
    else:
        raise Exception("invalid basis_type")

# TODO test the legendre case of this function more
def get_constant_and_linear_terms(coeff_x, coeff_y, img_shape, options):
    w = max(img_shape)/2
    '''
    given the coefficients and basis type get the four linear terms in the expansion about (0,0)
    '''
    constant = [coeff_x[0], coeff_y[0]]
    x_deri = [coeff_x[1] / w, coeff_x[2] / w]
    y_deri = [coeff_y[1] / w, coeff_y[2] / w]
    order = mapping[options['distortionOrder']]
    print("BASIS XY DERI", x_deri, y_deri)
    if options['basis_type'] == 'legendre': # other coefficients in legendre basis also contribute linearly
        ind = 3
        for i in range(2, order+1): # up to nth order legendre binomials
            for j in range(i+1):
                const_ind = legendre_polies[j].coef[-1] * legendre_polies[i-j].coef[-1]
                deriv_y = (legendre_polies[j].coef[-1] * legendre_polies[i-j].coef[-2]) if len(legendre_polies[i-j].coef) > 1 else 0
                deriv_x = (legendre_polies[j].coef[-2] * legendre_polies[i-j].coef[-1]) if len(legendre_polies[j].coef) > 1 else 0
                constant[0] += const_ind * coeff_x[ind]
                constant[1] += const_ind * coeff_y[ind]
                x_deri[0] += deriv_x * coeff_x[ind] * 2 / img_shape[1]
                x_deri[1] += deriv_y * coeff_x[ind] * 2 / img_shape[1]
                y_deri[0] += deriv_x * coeff_y[ind] * 2 / img_shape[0]
                y_deri[1] += deriv_y * coeff_y[ind] * 2 / img_shape[0]
                ind += 1
    print(x_deri, y_deri)
    return constant, x_deri, y_deri

def get_coeff_names(options):
    names = ['1']
    # TODO: check basis type
    for i in range(1, mapping[options['distortionOrder']]+1): # up to nth order binomials
        for j in range(i+1):
            if j == 0:
                names.append(f'x^{i-j}')
            elif i - j == 0:
                names.append(f'y^{j}')
            else:
                names.append(f'x^{i-j} * y^{j}')
    names = [name.replace('x^1', 'x').replace('y^1', 'y') for name in names]
    return names

'''
performs linear regression on errors, return the rms residual error
'''

def _regression_helper(errors, basis_x, basis_y):
    reg_x = LinearRegression().fit(basis_x, errors[:, 1])
    reg_y = LinearRegression().fit(basis_y, errors[:, 0])
    res_x = reg_x.predict(basis_x) - errors[:, 1]
    res_y = reg_y.predict(basis_y) - errors[:, 0]
    rms = np.mean(res_x**2+res_y**2)**0.5
    return rms    

'''
absorb two constant and two linear degrees of freedom in (reg_x, reg_y) into shifts in
shifts in q
returns: corrected q
'''
def _get_corrected_q(q, coeff_x, coeff_y, img_shape, options):
    constant, x_deri, y_deri = get_constant_and_linear_terms(coeff_x, coeff_y, img_shape, options)
    w = (max(img_shape)/2)
    platescale_multiplier = ((1 + x_deri[0]) * (1 + y_deri[1]))**0.5
    new_platescale = q[0] * platescale_multiplier
    theta = q[3]
    shiftRA_DEC = q[0] * np.array([[1/np.cos(q[2]), 0], [0, 1]]) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]]) @ np.array(constant)
    shift_roll_angle = x_deri[1] # small angle appromixation
    corrected_q = (new_platescale, q[1] + shiftRA_DEC[0], q[2] + shiftRA_DEC[1], q[3]-shift_roll_angle)
    return corrected_q

'''
stardata.epoch : the date that was requested from the catalogue (not the true observation date)
date_guess : the guessed date which we want to now improve
return : improved date guess, pmotion correction
'''
def _date_guess(date_guess, q, plate, stardata, img_shape, options):
    '''
    target = stardata.get_vectors()
    pmotion = stardata.get_pmotion()
    

    
    detransformed = transforms.detransform_vectors(q, target)
    errors = detransformed - plate
    basis = get_basis(plate[:, 0], plate[:, 1], w, m, options)
    #print('pshape', pmotion.shape)
    theta = q[3]

    rmatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]]) / (np.degrees(q[0])*3600*1000) # divide by 1000 to get arcsec from milli-arcsec
    pmotion[np.isnan(pmotion)] = 0
    pm_pixel = np.einsum('ij, ...j-> ...j', rmatrix, pmotion)
    pm_pixel[:, [0, 1]] = pm_pixel[:, [1, 0]] # swap columns of pm_pixel
    # apply date_guess to correct pmotion
    errors_p = errors + pm_pixel * (date_string_to_float(date_guess) - stardata.get_epoch_float())

    basis_x = np.c_[basis, pm_pixel[:, 1]]
    basis_y = np.c_[basis, pm_pixel[:, 0]]
    
    reg_x = LinearRegression().fit(basis_x, errors_p[:, 1]*m)
    reg_y = LinearRegression().fit(basis_y, errors_p[:, 0]*m)
    plate_corrected = plate + np.array([reg_y.predict(basis_x), reg_x.predict(basis_y)]).T / m
    #print(reg_x.coef_, reg_x.intercept_)
    #print(reg_y.coef_, reg_y.intercept_)
    print('dt guess x/y:', reg_x.coef_[-1], reg_y.coef_[-1])
    t0=datetime.datetime.fromisoformat(date_guess)
    t_guess = (t0 + datetime.timedelta(days=-int((reg_x.coef_[-1]+ reg_y.coef_[-1])*365.25/2))).date().isoformat()
    print('I guess image was taken on date:', date_guess, t_guess, int((reg_x.coef_[-1]+ reg_y.coef_[-1])*365.25/2))
    pmotion_correction = pm_pixel * (date_string_to_float(t_guess) - date_string_to_float(options['observation_date']))
    '''
    # show plot of rms vs t
   

    dtt = np.linspace(-15, 15, num=40)
    rmss = []
    basis = get_basis(plate[:, 0], plate[:, 1], img_shape, options)
    t0 = date_string_to_float(date_guess)
    for dt in dtt:
        stardata_copy = copy.copy(stardata)
        stardata_copy.update_epoch(dt+t0)
        target_t = stardata_copy.get_vectors()

        detransformed = transforms.detransform_vectors(q, target_t)
        errors = detransformed - plate
        rms = np.degrees(_regression_helper(errors, basis, basis)*q[0])*3600
        rmss.append(rms)
    plt.plot(dtt+t0, rmss)
    plt.ylabel('rms / arcsec')
    plt.xlabel('date (years)')
    if options['flag_display2']:
        plt.show()
    plt.close()

    def rms_func(t):
        stardata_copy = copy.copy(stardata)
        stardata_copy.update_epoch(t)
        target_t = stardata_copy.get_vectors()
        detransformed = transforms.detransform_vectors(q, target_t)
        errors = detransformed - plate
        rms = _regression_helper(errors, basis, basis)
        return rms

    min_result = scipy.optimize.minimize_scalar(rms_func, bounds = (t0-50, t0+50), method='bounded')

    print('min_result', min_result)

    min_date = date_from_float(min_result.x)
    print('min_date', min_date)
    return min_date

'''
perform requested linear regression with general
polynomial in x and y of the requested order (1, 3 or 5)

q : initial guess of (platescale, ra, dec, roll)
plate: (x, y) coordinates of stars
target: corresponding(x', y', z') of star true positions according to catalogue
new Oct'24: option for weighted centroids 
'''
def _cubic_helper(q, plate, target, img_shape, fix_coeff_x, fix_coeff_y, options, use_special=False, weights=1):
    detransformed = transforms.detransform_vectors(q, target)
    errors = detransformed - plate
    basis = get_basis(plate[:, 0], plate[:, 1], img_shape, options, use_special)

    '''
    new: if requested, use "fixed" higher order contributions
    '''
    #partition basis into "fixed" and "free" components
    order_total = mapping[options['distortionOrder']]
    order_free = mapping[options['distortion_fixed_coefficients']] if not options['distortion_fixed_coefficients'] == 'None' else order_total

    n_free = (order_free+2) * (order_free+1) // 2 - 1
    n_total = (order_total+2) * (order_total+1) // 2 - 1
    print(n_free, n_total)
    print(basis.shape)
    basis_free = basis[:, :n_free]
    basis_fixed = basis[:, n_free:]
    errors_fixed = np.copy(errors)
    fixed_correction = np.zeros(plate.shape, plate.dtype)
    coefficients_x = []
    coefficients_y = []
    if n_free < n_total:
        coefficients_x = np.array(list(fix_coeff_x.values()))[n_free+1:]
        coefficients_y = np.array(list(fix_coeff_y.values()))[n_free+1:]

        
        fixed_correction_x = np.einsum('ik,k->i', basis_fixed, coefficients_x)
        fixed_correction_y = np.einsum('ik,k->i', basis_fixed, coefficients_y)
  
        errors_fixed[:, 1] -= fixed_correction_x
        errors_fixed[:, 0] -= fixed_correction_y
        fixed_correction[:, 1] += fixed_correction_x
        fixed_correction[:, 0] += fixed_correction_y

    if weights == 1:
        weights = np.ones(plate.shape[0]) # TODO: unused remove!
    
    ols_result_x = sm.OLS(errors_fixed[:, 1], sm.add_constant(basis_free)).fit()
    ols_result_y = sm.OLS(errors_fixed[:, 0], sm.add_constant(basis_free)).fit()
    
    #print("OLS_X_SE", ols_result_x.HC0_se, '\n', "coeff", ols_result_x.params)
    #print("OLS_Y_SE", ols_result_y.HC0_se, '\n', "coeff", ols_result_y.params)

    plate_corrected = plate + np.array([ols_result_y.predict(sm.add_constant(basis_free)), ols_result_x.predict(sm.add_constant(basis_free))]).T + fixed_correction
    
    coeff_x = list(ols_result_x.params) + list(coefficients_x)
    coeff_y = list(ols_result_y.params) + list(coefficients_y)


    platescale_stdrelerror = (ols_result_x.HC0_se[1]**2 + ols_result_y.HC0_se[2]**2)**0.5 / (max(img_shape)/2)
    #print("PTSCALE STDERR:", platescale_stderror, ' vs. ', q[0])
    
    return _get_corrected_q(q, coeff_x, coeff_y, img_shape, options), plate_corrected, coeff_x, coeff_y, basis, errors_fixed, ols_result_x, ols_result_y, platescale_stdrelerror

def apply_corrections(q, plate, coeff_x, coeff_y, img_shape, options):
    basis = get_basis(plate[:, 0], plate[:, 1], img_shape, options)
    print(basis.shape)
    corr_x = np.einsum('ji,i->j', basis, coeff_x[1:]) # 1: to remove constant (which should be near-zero)
    corr_y = np.einsum('ji,i->j', basis, coeff_y[1:])
    return plate + np.c_[corr_y, corr_x]
                      
def _do_3D_plot(plate, errors, reg_x, reg_y, img_shape, options):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')    
    ax.scatter(plate[:,1], plate[:, 0], errors[:, 1], marker='+')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('x-error (pixels)')
    ax.set_title("x-error fit")
    X = np.linspace(-img_shape[1]/2, img_shape[1]/2, 20)
    Y = np.linspace(-img_shape[0]/2, img_shape[0]/2, 20)
    X, Y = np.meshgrid(X, Y)

    basis = get_basis(Y.flatten(), X.flatten(), img_shape, options)
    
    ### fix for fixed coeffs
    order_total = mapping[options['distortionOrder']]
    order_free = mapping[options['distortion_fixed_coefficients']] if not options['distortion_fixed_coefficients'] == 'None' else order_total
    n_free = (order_free+2) * (order_free+1) // 2 - 1                
    basis_free = basis[:, :n_free]
    
    Z_x = reg_x.predict(sm.add_constant(basis_free)).reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z_x, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.4)


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')    
    ax2.scatter(plate[:,1], plate[:, 0], errors[:, 0], marker='+')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('y-error (pixls)')
    ax2.set_title("y-error fit")
    Z_y = reg_y.predict(sm.add_constant(basis_free)).reshape(X.shape)
    surf = ax2.plot_surface(X, Y, Z_y, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.4)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')    
    ax3.scatter(plate[:,1], plate[:, 0], np.linalg.norm(errors, axis=1), marker='+')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('norm(error)')
    ax3.set_title("norm(error) fit")
    Z_n = (Z_x**2+Z_y**2)**0.5
    surf = ax3.plot_surface(X, Y, Z_n, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.4)
    
    if options['flag_display2']:
        plt.show()
    plt.close()

def do_cubic_fit(plate, stardata, initial_guess, img_shape, options, weights=1):
    target = stardata.get_vectors()
    fix_coeff_x, fix_coeff_y, fix_platescale, combined_platescale_uncertainty = _open_distortion_files(options)
    order_total = mapping[options['distortionOrder']]
    order_free = mapping[options['distortion_fixed_coefficients']] if not options['distortion_fixed_coefficients'] == 'None' else order_total

    if order_free == 0: # special case for only constant degree of freedom: use a linear fit, then discard the stretch/skew coefficients
        q_corrected = _cubic_helper(initial_guess, plate, target, img_shape, fix_coeff_x, fix_coeff_y, dict(options, **{'distortion_fixed_coefficients':'linear'}))[0]
        q_corrected = _cubic_helper(q_corrected, plate, target, img_shape, fix_coeff_x, fix_coeff_y, dict(options, **{'distortion_fixed_coefficients':'linear'}))[0]
        q_corrected = tuple([np.radians(fix_platescale/3600)]+list(q_corrected[1:4]))
        plate_corrected = apply_corrections(q_corrected, plate, list(fix_coeff_x.values()), list(fix_coeff_y.values()), img_shape, options)
        detransformed = transforms.detransform_vectors(q_corrected, target)
        errors = detransformed - plate_corrected
        mean_error = np.mean(errors, axis=0)
        #if not options['no_plot']:
        #    print('mean error:', mean_error)
        return q_corrected, plate_corrected, list(fix_coeff_x.values()), list(fix_coeff_y.values()), combined_platescale_uncertainty
    
    q_corrected = _cubic_helper(initial_guess, plate, target, img_shape, fix_coeff_x, fix_coeff_y, options, weights=weights)[0]
    q_corrected = _cubic_helper(q_corrected, plate, target, img_shape, fix_coeff_x, fix_coeff_y, options, weights=weights)[0]
    q_corrected, plate_corrected, coeff_x, coeff_y, basis, errors, reg_x, reg_y, platescale_stdrelerror = _cubic_helper(q_corrected, plate, target, img_shape, fix_coeff_x, fix_coeff_y, options, weights=weights) # apply for third time to really shrink the unwanted coefficients

    
    '''
    now if needed, apply the special basis functions
    if not options['basis_type'] == 'polynomial':
        print("now using special basis")
        q_corrected, plate_corrected, reg_x, reg_y, basis, errors = _cubic_helper(q_corrected, plate, target, w, m, options, use_special=True) # apply for third time to really shrink the unwanted coefficients
    '''

    #print('residuals_x\n', reg_x.predict(basis) / m - errors[:, 1])
    #print('residuals_y\n', reg_y.predict(basis) / m - errors[:, 0])
    if not ('no_plot' in options and options['no_plot']):
        print(reg_x.params)
        print(reg_y.params)

        _do_3D_plot(plate, errors, reg_x, reg_y, img_shape, options)
 
    return q_corrected, plate_corrected, coeff_x, coeff_y, platescale_stdrelerror


def show_coef_boxplot(loaded):
    coeff_x = defaultdict(list)
    coeff_y = defaultdict(list)
    for data in loaded:
        for k, v in data["distortion coeffs x"].items():
            if k in ['1']:#, 'x^2', 'y^2', 'x * y']:
                continue
            coeff_x[k].append(v)
        for k, v in data["distortion coeffs y"].items():
            if k in ['1']:#, 'x^2', 'y^2', 'x * y']:
                continue
            coeff_y[k].append(v)
    fig, ax = plt.subplots(2, 1)
    data = [coeff_x[k] for k in coeff_x]
    ax[0].axhline()
    ax[0].set_title("Distortion Coefficients X", fontsize=16)
    ax[0].boxplot(data)
    ax[0].set_ylabel('distortion coefficient (pixels)', fontsize=14)
    ax[0].set_xticklabels(['$'+k.replace('*', '')+'$' for k in coeff_x], fontsize=14)
    ax[0].set_ylim(-15, 15)
    data = [coeff_y[k] for k in coeff_y]
    ax[1].set_title("Distortion Coefficients Y", fontsize=16)
    ax[1].axhline()
    ax[1].boxplot(data)
    ax[1].set_xticklabels(['$'+k.replace('*', '')+'$' for k in coeff_y], fontsize=14)
    ax[1].set_ylabel('distortion coefficient (pixels)', fontsize=14)
    ax[1].set_ylim(-15, 15)
    num_boxes = len(coeff_x)
    pos = np.arange(num_boxes) + 1
    upper_labels = [f'{np.mean(coeff_x[k]):.2f}' for k in coeff_x]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax[0].get_xticklabels()):
        k = tick % 2
        ax[0].text(pos[tick], .95, upper_labels[tick],
                 transform=ax[0].get_xaxis_transform(),
                 horizontalalignment='center', size='small')
                 #weight=weights[k], color=box_colors[k])

    num_boxes = len(coeff_y)
    pos = np.arange(num_boxes) + 1
    upper_labels = [f'{np.mean(coeff_y[k]):.2f}' for k in coeff_y]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax[1].get_xticklabels()):
        k = tick % 2
        ax[1].text(pos[tick], .95, upper_labels[tick],
                 transform=ax[1].get_xaxis_transform(),
                 horizontalalignment='center', size='small')
                 #weight=weights[k], color=box_colors[k])

    
    plt.show()
        

def _open_distortion_files(options):
    files = options['distortion_reference_files'].split(';')
    loaded = []
    for file in files:
        if file == '':
            continue
        if file.endswith('.zip'):
            archive = zipfile.ZipFile(file, 'r')
            loaded.append(json.load(archive.open('distortion_results.txt')))
        else:
            with open(file) as fp:
                loaded.append(json.load(fp))
    n = len(loaded)
    coeff_x = defaultdict(float)
    coeff_y = defaultdict(float)
    orders = []
    platescales = []
    platescale_uncertainties = []
    for data in loaded:
        #print(data, options)
        platescales.append(data["platescale (arcseconds/pixel)"])
        if "platescale_relative_uncertainty" in data:
            platescale_uncertainties.append(data["platescale_relative_uncertainty"])
        if "distortion order" in data and not data["distortion order"] == options["distortionOrder"]:
            raise Exception(f'input distortion order not consistent: {options["distortionOrder"]} was requested but input files have order {data["distortion order"]}')
        for k, v in data["distortion coeffs x"].items():
            coeff_x[k] += v/n
        for k, v in data["distortion coeffs y"].items():
            coeff_y[k] += v/n
        if "distortion order" in data: # legacy compatible
            orders.append(data["distortion order"])
    if len(set(orders)) > 1:
        raise Exception("input distortion files are not same order: " + str(orders))
    #show_coef_boxplot(loaded)
    coeff_x, coeff_y = dict(coeff_x), dict(coeff_y)
    #print(coeff_x)
    #print(coeff_y)
    if platescale_uncertainties:
        combined_platescale_uncertainty = np.linalg.norm(platescale_uncertainties) / len(platescale_uncertainties)
    elif len(platescales) >= 3:
        print("WARNING: no platescale uncertainty found in files, using variance of observations")
        combined_platescale_uncertainty = np.std(platescales) * (len(platescales) / (len(platescales) - 1))**0.5
    else:
        if platescales:
            print("WARNING: no platescale uncertainty could be made")
        combined_platescale_uncertainty = -1
    
    return coeff_x, coeff_y, np.mean(platescales) if platescales else -1, combined_platescale_uncertainty
    
