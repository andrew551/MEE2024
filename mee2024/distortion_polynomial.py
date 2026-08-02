"""
@author: Andrew Smith
Version 6 May 2024
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from mee2024 import transforms
import matplotlib.pyplot as plt
import scipy
from mee2024.MEE2024util import date_string_to_float, date_from_float
import copy
import json
from collections import defaultdict
import zipfile
import statsmodels.api as sm

mapping = {'constant':0, 'linear':1, 'quadratic':2, 'cubic':3, 'quartic': 4, 'quintic':5, 'sextic': 6, 'septic':7}

'''
Polynomial basis in the centred plate coordinates (y, x), normalised by w so that
coefficients are in pixels at the image edge. Returns an (nstars, nterms) array; the
constant term is added separately by sm.add_constant in the fit.
'''
def get_basis(y, x, w, m, options):
    basis = []
    order = mapping[options['distortionOrder']]
    for i in range(1, order+1): # up to nth order binomials
        for j in range(i+1):
            basis.append(y ** j * x ** (i-j) / w**i)
    return np.array(basis).T

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
def _get_corrected_q(q, reg_x, reg_y, w):
    platescale_multiplier = ((1 + reg_x.params[1] / w) * (1 + reg_y.params[2] / w))**0.5
    new_platescale = q[0] * platescale_multiplier
    theta = q[3]
    shiftRA_DEC = q[0] * np.array([[1/np.cos(q[2]), 0], [0, 1]]) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]]) @ np.array([reg_x.params[0], reg_y.params[0]])
    shift_roll_angle = reg_x.params[2] / w # small angle appromixation
    corrected_q = (new_platescale, q[1] + shiftRA_DEC[0], q[2] + shiftRA_DEC[1], q[3]-shift_roll_angle)
    return corrected_q

'''
stardata.epoch : the date that was requested from the catalogue (not the true observation date)
date_guess : the guessed date which we want to now improve
return : improved date guess, pmotion correction
'''
def _date_guess(date_guess, q, plate, stardata, img_shape, options):
    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #q[0] # for astrometrica convention
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
    basis = get_basis(plate[:, 0], plate[:, 1], w, m, options)
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
def _cubic_helper(q, plate, target, w, m, fix_coeff_x, fix_coeff_y, options, weights=1):
    detransformed = transforms.detransform_vectors(q, target)
    errors = detransformed - plate
    basis = get_basis(plate[:, 0], plate[:, 1], w, m, options)

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
  
        errors_fixed[:, 1] -= fixed_correction_x / m
        errors_fixed[:, 0] -= fixed_correction_y / m
        fixed_correction[:, 1] += fixed_correction_x / m
        fixed_correction[:, 0] += fixed_correction_y / m

    ols_result_x = sm.OLS(errors_fixed[:, 1]*m, sm.add_constant(basis_free)).fit()
    ols_result_y = sm.OLS(errors_fixed[:, 0]*m, sm.add_constant(basis_free)).fit()
    
    #print("OLS_X_SE", ols_result_x.HC0_se, '\n', "coeff", ols_result_x.params)
    #print("OLS_Y_SE", ols_result_y.HC0_se, '\n', "coeff", ols_result_y.params)

    plate_corrected = plate + np.array([ols_result_y.predict(sm.add_constant(basis_free)), ols_result_x.predict(sm.add_constant(basis_free))]).T / m + fixed_correction
    
    coeff_x = list(ols_result_x.params) + list(coefficients_x)
    coeff_y = list(ols_result_y.params) + list(coefficients_y)

    platescale_stdrelerror = (ols_result_x.HC0_se[1]**2 + ols_result_y.HC0_se[2]**2)**0.5 / w
    #print("PTSCALE STDERR:", platescale_stderror, ' vs. ', q[0])
    
    return _get_corrected_q(q, ols_result_x, ols_result_y, w), plate_corrected, coeff_x, coeff_y, basis, errors_fixed, ols_result_x, ols_result_y, platescale_stdrelerror

def apply_corrections(q, plate, coeff_x, coeff_y, img_shape, options):
    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #result.x[0] # for astrometrica convention
    basis = get_basis(plate[:, 0], plate[:, 1], w, m, options)
    print(basis.shape)
    corr_x = np.einsum('ji,i->j', basis, coeff_x[1:]) # 1: to remove constant (which should be near-zero)
    corr_y = np.einsum('ji,i->j', basis, coeff_y[1:])
    return plate + np.c_[corr_y, corr_x]
                      
def _do_3D_plot(plate, errors, reg_x, reg_y, img_shape, w, m, options):
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

    basis = get_basis(Y.flatten(), X.flatten(), w, m, options)
    
    ### fix for fixed coeffs
    order_total = mapping[options['distortionOrder']]
    order_free = mapping[options['distortion_fixed_coefficients']] if not options['distortion_fixed_coefficients'] == 'None' else order_total
    n_free = (order_free+2) * (order_free+1) // 2 - 1                
    basis_free = basis[:, :n_free]
    
    Z_x = reg_x.predict(sm.add_constant(basis_free)).reshape(X.shape)
    ax.plot_surface(X, Y, Z_x, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.4)


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')    
    ax2.scatter(plate[:,1], plate[:, 0], errors[:, 0], marker='+')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('y-error (pixls)')
    ax2.set_title("y-error fit")
    Z_y = reg_y.predict(sm.add_constant(basis_free)).reshape(X.shape)
    ax2.plot_surface(X, Y, Z_y, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                     linewidth=0, antialiased=False, alpha=0.4)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')    
    ax3.scatter(plate[:,1], plate[:, 0], np.linalg.norm(errors, axis=1), marker='+')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('norm(error)')
    ax3.set_title("norm(error) fit")
    Z_n = (Z_x**2+Z_y**2)**0.5
    ax3.plot_surface(X, Y, Z_n, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                     linewidth=0, antialiased=False, alpha=0.4)
    
    if options['flag_display2']:
        plt.show()
    plt.close()

def do_cubic_fit(plate, stardata, initial_guess, img_shape, options, weights=1):
    target = stardata.get_vectors()
    # Least squares with no rows (or fewer rows than free coefficients) fails deep
    # inside the regression with a message about array dimensions, which says nothing
    # about the cause: the matching stage found too few stars. Say that instead.
    n_terms = (mapping[options['distortionOrder']] + 1) * (
        mapping[options['distortionOrder']] + 2) // 2
    if len(plate) < n_terms:
        raise ValueError(
            f'only {len(plate)} star(s) matched the catalogue, but a '
            f'{options["distortionOrder"]} distortion fit needs at least {n_terms}. '
            f'Nothing can be fitted from this. Common causes: the plate solve landed '
            f'on the wrong field, the observation date is far from the truth, the '
            f'magnitude limit excludes the stars that were detected, or the catalogue '
            f'lists stars twice (check `mee2024 catalogue` for overlapping archives).')
    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #result.x[0] # for astrometrica convention
    #w = 1
    #m = max(img_shape)
    fix_coeff_x, fix_coeff_y, fix_platescale, combined_platescale_uncertainty = _open_distortion_files(options)
    order_total = mapping[options['distortionOrder']]
    order_free = mapping[options['distortion_fixed_coefficients']] if not options['distortion_fixed_coefficients'] == 'None' else order_total

    if order_free == 0: # special case for only constant degree of freedom: use a linear fit, then discard the stretch/skew coefficients
        q_corrected = _cubic_helper(initial_guess, plate, target, w, m, fix_coeff_x, fix_coeff_y, dict(options, **{'distortion_fixed_coefficients':'linear'}))[0]
        q_corrected = _cubic_helper(q_corrected, plate, target, w, m, fix_coeff_x, fix_coeff_y, dict(options, **{'distortion_fixed_coefficients':'linear'}))[0]
        q_corrected = tuple([np.radians(fix_platescale/3600)]+list(q_corrected[1:4]))
        plate_corrected = apply_corrections(q_corrected, plate, list(fix_coeff_x.values()), list(fix_coeff_y.values()), img_shape, options)
        return q_corrected, plate_corrected, list(fix_coeff_x.values()), list(fix_coeff_y.values()), combined_platescale_uncertainty

    q_corrected = _cubic_helper(initial_guess, plate, target, w, m, fix_coeff_x, fix_coeff_y, options, weights=weights)[0]
    q_corrected = _cubic_helper(q_corrected, plate, target, w, m, fix_coeff_x, fix_coeff_y, options, weights=weights)[0]
    q_corrected, plate_corrected, coeff_x, coeff_y, basis, errors, reg_x, reg_y, platescale_stdrelerror = _cubic_helper(q_corrected, plate, target, w, m, fix_coeff_x, fix_coeff_y, options, weights=weights) # apply for third time to really shrink the unwanted coefficients

    if not ('no_plot' in options and options['no_plot']):
        print(reg_x.params)
        print(reg_y.params)

        _do_3D_plot(plate, errors, reg_x, reg_y, img_shape, w, m, options)
 
    return q_corrected, plate_corrected, coeff_x, coeff_y, platescale_stdrelerror


def distortion_field(coeff_x, coeff_y, img_shape, options, n=22):
    """Sample the fitted distortion on a grid over the image.

    Returns (X, Y, DX, DY) in pixels, where (DX, DY) is the correction ``apply_corrections``
    would add at each point -- i.e. how far the optics displaced a star from where an ideal
    gnomonic projection would put it.
    """
    w = (max(img_shape) / 2)
    m = 1
    ys = np.linspace(-img_shape[0] / 2, img_shape[0] / 2, n)
    xs = np.linspace(-img_shape[1] / 2, img_shape[1] / 2, n)
    X, Y = np.meshgrid(xs, ys)
    basis = get_basis(Y.flatten(), X.flatten(), w, m, options)
    # coeff[0] is the constant term, which _get_corrected_q already folded into the
    # pointing, so it is deliberately excluded here
    DX = np.einsum('ji,i->j', basis, np.asarray(coeff_x[1:])).reshape(X.shape)
    DY = np.einsum('ji,i->j', basis, np.asarray(coeff_y[1:])).reshape(Y.shape)
    return X, Y, DX, DY


def suggest_residual_bins(n_stars, configured=0, lo=4, hi=24, per_cell=8):
    """Bins per axis for the residual-correlation map, chosen from the star count.

    A single star's nearest-neighbour correlation is a cosine spanning -1..1, so a cell
    holding two or three of them is mostly noise and a map of such cells reads as
    structure that is not there. Aiming at ``per_cell`` stars keeps a warm patch
    meaningful: a typical 430-star field lands on 7 bins, and 4600 stars earns the
    24-bin ceiling.

    ``configured`` (``residual_bins`` in the options) overrides the choice when set, for
    anyone who wants a particular resolution regardless.
    """
    if configured and configured > 0:
        return int(max(lo, min(32, configured)))
    if n_stars <= 0:
        return lo
    return int(max(lo, min(hi, round((n_stars / per_cell) ** 0.5))))


def analysis_payload(plate, corrections, residuals, coeff_x, coeff_y, img_shape, options,
                     platescale_arcsec=None, n=22):
    """The data behind the app's advanced analysis views, as plain JSON-able lists.

    Two things the flat field map cannot show:

    * the fitted displacement as a *surface* with each star's measured displacement
      plotted against it. A star sits off the surface by exactly its residual, so an
      order that is too low shows up as the scatter undulating coherently above and
      below rather than peppering it evenly -- the old three-panel 3-D view.
    * where on the detector the residuals are spatially correlated, which localises an
      optical imperfection instead of averaging it into a single number.

    Positions are pixels from the image centre, matching ``distortion_field``. Columns
    rather than a list of records, which roughly halves the JSON. Residuals are given
    separately from the measured displacement so the frontend need not subtract them
    back out.
    """
    X, Y, DX, DY = distortion_field(coeff_x, coeff_y, img_shape, options, n=n)
    # the polynomial was fitted to (ideal - measured), which is the correction it applies
    # less whatever it failed to absorb; recovering it this way avoids re-evaluating the
    # basis and cannot drift out of step with the fit that actually ran
    measured = np.asarray(corrections) - np.asarray(residuals)

    def col(a):
        return [round(float(v), 4) for v in np.asarray(a).ravel()]

    return {
        'image_size': [int(img_shape[0]), int(img_shape[1])],
        'platescale': float(platescale_arcsec) if platescale_arcsec else None,
        'order': options['distortionOrder'],
        'bins': suggest_residual_bins(len(plate), options.get('residual_bins', 0)),
        'stars': {
            'x': col(plate[:, 1]), 'y': col(plate[:, 0]),
            'dx': col(measured[:, 1]), 'dy': col(measured[:, 0]),
            'rx': col(residuals[:, 1]), 'ry': col(residuals[:, 0]),
        },
        'surface': {
            'x': col(X[0, :]), 'y': col(Y[:, 0]),
            'dx': [col(row) for row in DX], 'dy': [col(row) for row in DY],
        },
    }


def render_distortion_field(coeff_x, coeff_y, img_shape, options, platescale_arcsec=None,
                            save_to=None):
    """Draw the distortion field: arrows plus a magnitude map. Returns the figure.

    Replaces the three rotatable 3-D matplotlib windows the old code opened, which showed
    the same information but could not be read at a glance or saved usefully.
    """
    X, Y, DX, DY = distortion_field(coeff_x, coeff_y, img_shape, options)
    magnitude = np.hypot(DX, DY)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    unit = 'pixels'
    scale = 1.0
    if platescale_arcsec:
        scale = platescale_arcsec
        unit = 'arcsec'

    ax = axes[0]
    ax.quiver(X, Y, DX, DY, magnitude * scale, cmap='viridis', angles='xy',
              pivot='middle', width=0.004)
    ax.set_title(f'Distortion displacement ({options["distortionOrder"]} fit)')
    ax.set_xlabel('x (pixels from centre)')
    ax.set_ylabel('y (pixels from centre)')
    ax.set_aspect('equal')
    # y is a row offset, so it increases downward. Left to itself matplotlib puts it the
    # other way up, and the field then mirrors the frame it is describing.
    ax.invert_yaxis()
    ax.grid(alpha=0.25)

    ax = axes[1]
    mesh = ax.pcolormesh(X, Y, magnitude * scale, cmap='magma', shading='auto')
    contours = ax.contour(X, Y, magnitude * scale, colors='white', linewidths=0.7,
                          alpha=0.75)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%.2f')
    fig.colorbar(mesh, ax=ax, label=f'displacement ({unit})')
    ax.set_title('Distortion magnitude')
    ax.set_xlabel('x (pixels from centre)')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    peak = float(np.max(magnitude) * scale)
    fig.suptitle(f'peak displacement {peak:.2f} {unit}'
                 + ('' if platescale_arcsec else ' (pixels)'), fontsize=10)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=200, bbox_inches='tight')
    return fig


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
    for axis, coeffs in ((ax[0], coeff_x), (ax[1], coeff_y)):
        pos = np.arange(len(coeffs)) + 1
        upper_labels = [f'{np.mean(coeffs[k]):.2f}' for k in coeffs]
        for tick in range(len(coeffs)):
            axis.text(pos[tick], .95, upper_labels[tick],
                      transform=axis.get_xaxis_transform(),
                      horizontalalignment='center', size='small')

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
    print(coeff_x)
    print(coeff_y)
    if platescale_uncertainties:
        combined_platescale_uncertainty = np.linalg.norm(platescale_uncertainties) / len(platescale_uncertainties)
    elif len(platescales) >= 3:
        print("WARNING: no platescale uncertainty found in files, using variance of observations")
        combined_platescale_uncertainty = np.std(platescales) * (len(platescales) / (len(platescales) - 1))**0.5
    else:
        print("WARNING: no platescale uncertainty could be made")
        combined_platescale_uncertainty = -1
    # no reference files at all is the normal case: the caller only uses the mean
    # platescale when higher-order coefficients are actually being held fixed
    mean_platescale = np.mean(platescales) if platescales else float('nan')
    return coeff_x, coeff_y, mean_platescale, combined_platescale_uncertainty
    
