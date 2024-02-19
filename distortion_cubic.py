from sklearn.linear_model import LinearRegression
import numpy as np
import transforms
import matplotlib.pyplot as plt
import scipy
import datetime
from MEE2024util import date_string_to_float

mapping = {'linear':1, 'cubic':3, 'quintic':5, 'septic':7}

def get_basis(y, x, w, m, options):
    basis = []
    for i in range(1, mapping[options['distortionOrder']]+1): # up to nth order binomials
        for j in range(i+1):
            basis.append(y ** j * x ** (i-j) / w**i)
    return np.array(basis).T

def get_coeff_names(options):
    names = ['1']
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
absorb two constant and two linear degrees of freedom in (reg_x, reg_y) into shifts in
shifts in q
returns: corrected q
'''
def _get_corrected_q(q, reg_x, reg_y, w):
    platescale_multiplier = ((1 + reg_x.coef_[0] / w) * (1 + reg_y.coef_[1] / w))**0.5
    new_platescale = q[0] * platescale_multiplier
    theta = q[3]
    shiftRA_DEC = q[0] * np.array([[1/np.cos(q[2]), 0], [0, 1]]) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]]) @ np.array([reg_x.intercept_, reg_y.intercept_])
    shift_roll_angle = reg_x.coef_[1] / w # small angle appromixation
    corrected_q = (new_platescale, q[1] + shiftRA_DEC[0], q[2] + shiftRA_DEC[1], q[3]-shift_roll_angle)
    return corrected_q

'''
stardata.epoch : the date that was requested from the catalogue (not the true observation date)
date_guess : the guessed date which we want to now improve
return : improved date guess, pmotion correction
'''
def _date_guess(date_guess, q, plate, stardata, img_shape, options):
    target = stardata.get_vectors()
    pmotion = stardata.get_pmotion()
    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #q[0] # for astrometrica convention

    
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
    errors += pm_pixel * (date_string_to_float(date_guess) - stardata.epoch)

    basis_x = np.c_[basis, pm_pixel[:, 1]]
    basis_y = np.c_[basis, pm_pixel[:, 0]]
    
    reg_x = LinearRegression().fit(basis_x, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basis_y, errors[:, 0]*m)
    plate_corrected = plate + np.array([reg_y.predict(basis_x), reg_x.predict(basis_y)]).T / m
    #print(reg_x.coef_, reg_x.intercept_)
    #print(reg_y.coef_, reg_y.intercept_)
    print('dt guess x/y:', reg_x.coef_[-1], reg_y.coef_[-1])
    t0=datetime.datetime.fromisoformat(date_guess)
    t_guess = (t0 + datetime.timedelta(days=-int((reg_x.coef_[-1]+ reg_y.coef_[-1])*365.25/2))).date().isoformat()
    print('I guess image was taken on date:', date_guess, t_guess, int((reg_x.coef_[-1]+ reg_y.coef_[-1])*365.25/2))
    pmotion_correction = pm_pixel * (date_string_to_float(t_guess) - date_string_to_float(options['observation_date']))
    return t_guess, pmotion_correction

'''
perform requested linear regression with general
polynomial in x and y of the requested order (1, 3 or 5)

q : initial guess of (platescale, ra, dec, roll)
plate: (x, y) coordinates of stars
target: corresponding(x', y', z') of star true positions according to catalogue
'''
def _cubic_helper(q, plate, target, w, m, options):
    detransformed = transforms.detransform_vectors(q, target)
    errors = detransformed - plate
    basis = get_basis(plate[:, 0], plate[:, 1], w, m, options)
    reg_x = LinearRegression().fit(basis, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basis, errors[:, 0]*m)
    plate_corrected = plate + np.array([reg_y.predict(basis), reg_x.predict(basis)]).T / m
    return _get_corrected_q(q, reg_x, reg_y, w), plate_corrected, reg_x, reg_y, basis, errors

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
    Z_x = reg_x.predict(get_basis(Y.flatten(), X.flatten(), w, m, options)).reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z_x, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.4)


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')    
    ax2.scatter(plate[:,1], plate[:, 0], errors[:, 0], marker='+')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('y-error (pixls)')
    ax2.set_title("y-error fit")
    Z_y = reg_y.predict(get_basis(Y.flatten(), X.flatten(), w, m, options)).reshape(X.shape)
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

def do_cubic_fit(plate, stardata, initial_guess, img_shape, options):
    target = stardata.get_vectors()
    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #result.x[0] # for astrometrica convention
    
    q_corrected = _cubic_helper(initial_guess, plate, target, w, m, options)[0]
    q_corrected = _cubic_helper(q_corrected, plate, target, w, m, options)[0]
    q_corrected, plate_corrected, reg_x, reg_y, basis, errors = _cubic_helper(q_corrected, plate, target, w, m, options) # apply for third time to really shrink the unwanted coefficients

    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print('residuals_x\n', reg_x.predict(basis) / m - errors[:, 1])
    print('residuals_y\n', reg_y.predict(basis) / m - errors[:, 0])
    
    _do_3D_plot(plate, errors, reg_x, reg_y, img_shape, w, m, options)
 
    return q_corrected, plate_corrected, reg_x, reg_y
