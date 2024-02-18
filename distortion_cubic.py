from sklearn.linear_model import LinearRegression
import numpy as np
import transforms
import matplotlib.pyplot as plt
import scipy

def to_polar(v):
    theta = np.arcsin(v[:, 2])
    phi = np.arctan2(v[:, 1], v[:, 0])
    phi[phi < 0] += np.pi * 2
    ret = np.degrees(np.array([theta, phi]))
    
    return ret.T

def get_fitfunc(plate, target, transform_function=transforms.linear_transform, img_shape=None):

    def fitfunc(x):
        rotated = transform_function(x, plate, img_shape)
        return np.linalg.norm(target-rotated)**2 / plate.shape[0] # mean square error
    return fitfunc

###### the distortion fitting and matching function #####

mapping = {'linear':1, 'cubic':3, 'quintic':5}

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



def do_cubic_fit(plate2, target2, initial_guess, img_shape, options):
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error linear solve: ', result.fun**0.5)

    resv = to_polar(transforms.linear_transform(result.x, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(result.x, target2)
    errors = detransformed - plate2

    w = (max(img_shape)/2) # 1 # for astrometrica convention
    m = 1 #result.x[0] # for astrometrica convention
    
    basis = get_basis(plate2[:, 0], plate2[:, 1], w, m, options)
    
    
    
    reg_x = LinearRegression().fit(basis, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basis, errors[:, 0]*m)
    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print(reg_x.predict(basis)/ m - errors[:, 1])
    print(reg_y.predict(basis)/ m - errors[:, 0])
    print('mean errors', np.mean(errors, axis=0))
    plate2_corrected = plate2 + np.array([reg_y.predict(basis), reg_x.predict(basis)]).T / m
    '''
    initial_guess = result.x
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2_corrected, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error corrected solve (arcseconds): ', np.degrees(result.fun**0.5) * 3600)
    '''
    # TODO: re-absorb the linear and constant coefficients into changes in plate_scale, roll, dec and ra

    # platescale adjustment:
    platescale_multiplier = ((1 + reg_x.coef_[0] / (max(img_shape)/2)) * (1 + reg_y.coef_[1] / (max(img_shape)/2)))**0.5

    new_platescale = result.x[0] * platescale_multiplier
    theta = result.x[3]
    shiftRA_DEC = result.x[0] * np.array([[1/np.cos(result.x[2]), 0], [0, 1]]) @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]]) @ np.array([reg_x.intercept_, reg_y.intercept_])

    shift_roll_angle = reg_x.coef_[1] / (max(img_shape)/2) # small angle appromixation
    #shift_roll_angle  = np.arctan(reg_x.coef_[1] / (max(img_shape)/2) / (1 + reg_y.coef_[1] / (max(img_shape)/2)))

    new_result = (new_platescale, result.x[1] + shiftRA_DEC[0], result.x[2] + shiftRA_DEC[1], result.x[3]-shift_roll_angle)
    print("old platescale:", result.x)
    print("modified platescale:", new_result)

    resv = to_polar(transforms.linear_transform(new_result, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(new_result, target2)
    errors = detransformed - plate2

    reg_x = LinearRegression().fit(basis, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basis, errors[:, 0]*m)
    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print(reg_x.predict(basis)/ m - errors[:, 1])
    print(reg_y.predict(basis)/ m - errors[:, 0])
    print('mean errors', np.mean(errors, axis=0))
    plate2_corrected = plate2 + np.array([reg_y.predict(basis), reg_x.predict(basis)]).T / m

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')    
    ax.scatter(plate2[:,1], plate2[:, 0], errors[:, 1], marker='+')

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
    ax2.scatter(plate2[:,1], plate2[:, 0], errors[:, 0], marker='+')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('y-error (pixls)')
    ax2.set_title("y-error fit")
    Z_y = reg_y.predict(get_basis(Y.flatten(), X.flatten(), w, m, options)).reshape(X.shape)
    surf = ax2.plot_surface(X, Y, Z_y, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.4)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')    
    ax3.scatter(plate2[:,1], plate2[:, 0], np.linalg.norm(errors, axis=1), marker='+')

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

    
    return new_result, plate2_corrected, reg_x, reg_y

