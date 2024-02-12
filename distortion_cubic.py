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

def do_cubic_fit(plate2, target2, initial_guess, img_shape, options):
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error linear solve: ', result.fun**0.5)

    #resi = to_polar(transforms.linear_transform(initial_guess, plate2, image_size))

    resv = to_polar(transforms.linear_transform(result.x, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(result.x, target2)
    errors = detransformed - plate2

    plt.scatter(detransformed[:, 1], detransformed[:, 0], marker='+')
    plt.scatter(plate2[:, 1], plate2[:, 0], marker='+')
    if options['flag_display']:
        plt.show()
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.scatter(plate2[:,1], plate2[:, 0], errors[:, 1], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('x-error')
    if options['flag_display']:
        plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.scatter(plate2[:,1], plate2[:, 0], errors[:, 0], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('y-error')

    if options['flag_display']:
        plt.show()
    plt.close()
    


    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(plate2[:, 1], errors[:, 0], label = 'px-ey')
    axs[0,0].legend()
    axs[1,0].scatter(plate2[:, 1], errors[:, 1], label = 'px-ex')
    axs[1,0].legend()
    axs[1,1].scatter(plate2[:, 0], errors[:, 0], label = 'py-ey')
    axs[1,1].legend()
    axs[0,1].scatter(plate2[:, 0], errors[:, 1], label = 'py-ex')
    axs[0,1].legend()
    plt.title('errors before correction')
    if options['flag_display']:
        plt.show()
    plt.close()
    
    basis = []
    for i in range(1, 4): # up to cubic order
        for j in range(i+1):
            basis.append(plate2[:, 0] ** j * plate2[:, 1] ** (i-j) / (max(img_shape)/2)**i)
    basis = np.array(basis).T
    m = result.x[0]
    m = 1
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

    new_result = (new_platescale, result.x[1] + shiftRA_DEC[0], result.x[2] + shiftRA_DEC[1], result.x[3]-shift_roll_angle)
    print("old platescale:", result.x)
    print("modified platescale:", new_result)

    resv = to_polar(transforms.linear_transform(new_result, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(new_result, target2)
    errors = detransformed - plate2

    plt.scatter(detransformed[:, 1], detransformed[:, 0], marker='+')
    plt.scatter(plate2[:, 1], plate2[:, 0], marker='+')
    if options['flag_display']:
        plt.show()
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.scatter(plate2[:,1], plate2[:, 0], errors[:, 1], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('x-error')
    if options['flag_display']:
        plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')    
    ax.scatter(plate2[:,1], plate2[:, 0], errors[:, 0], marker='+')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('y-error')

    if options['flag_display']:
        plt.show()
    plt.close()
    


    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(plate2[:, 1], errors[:, 0], label = 'px-ey')
    axs[0,0].legend()
    axs[1,0].scatter(plate2[:, 1], errors[:, 1], label = 'px-ex')
    axs[1,0].legend()
    axs[1,1].scatter(plate2[:, 0], errors[:, 0], label = 'py-ey')
    axs[1,1].legend()
    axs[0,1].scatter(plate2[:, 0], errors[:, 1], label = 'py-ex')
    axs[0,1].legend()
    plt.title('errors before correction')
    if options['flag_display']:
        plt.show()
    plt.close()

    w = (max(img_shape)/2) # 1 # for astrometric convention
    m = 1 #result.x[0] # for astrometric convention
    basis = []
    for i in range(1, 4): # up to cubic order
        for j in range(i+1):
            basis.append(plate2[:, 0] ** j * plate2[:, 1] ** (i-j) / w**i)
    basis = np.array(basis).T
    
    reg_x = LinearRegression().fit(basis, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basis, errors[:, 0]*m)
    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print(reg_x.predict(basis)/ m - errors[:, 1])
    print(reg_y.predict(basis)/ m - errors[:, 0])
    print('mean errors', np.mean(errors, axis=0))
    plate2_corrected = plate2 + np.array([reg_y.predict(basis), reg_x.predict(basis)]).T / m
    
    return new_result, plate2_corrected, reg_x, reg_y

def do_cubic_fit_r(plate2, target2, initial_guess, img_shape, options):
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error linear solve: ', result.fun**0.5)

    #resi = to_polar(transforms.linear_transform(initial_guess, plate2, image_size))

    resv = to_polar(transforms.linear_transform(result.x, plate2))
    orig = to_polar(target2)

    detransformed = transforms.detransform_vectors(result.x, target2)

    plt.scatter(detransformed[:, 1], detransformed[:, 0], marker='+')
    plt.scatter(plate2[:, 1], plate2[:, 0], marker='+')
    if options['flag_display']:
        plt.show()
    plt.close()
    
    errors = detransformed - plate2


    fig, axs = plt.subplots(2, 2)
    axs[0,0].scatter(plate2[:, 1], errors[:, 0], label = 'px-ey')
    axs[0,0].legend()
    axs[1,0].scatter(plate2[:, 1], errors[:, 1], label = 'px-ex')
    axs[1,0].legend()
    axs[1,1].scatter(plate2[:, 0], errors[:, 0], label = 'py-ey')
    axs[1,1].legend()
    axs[0,1].scatter(plate2[:, 0], errors[:, 1], label = 'py-ex')
    axs[0,1].legend()
    plt.title('errors before correction')
    if options['flag_display']:
        plt.show()
    plt.close()
    
    basis = []
    for i in (1, 3): # up to cubic order
        for j in range(i+1):
            if i == 1 or j % 2 == 0:
                basis.append(plate2[:, 0] ** j * plate2[:, 1] ** (i-j) / (max(img_shape)/2)**i)
    basis = np.array(basis).T
    basisY = []
    for i in (1, 3): # up to cubic order
        for j in range(i+1):
            if i == 1 or j % 2 == 1:
                basisY.append(plate2[:, 0] ** j * plate2[:, 1] ** (i-j) / (max(img_shape)/2)**i)
    basisY = np.array(basisY).T
    m = result.x[0]
    m = 1
    reg_x = LinearRegression().fit(basis, errors[:, 1]*m)
    reg_y = LinearRegression().fit(basisY, errors[:, 0]*m)
    print(reg_x.coef_, reg_x.intercept_)
    print(reg_y.coef_, reg_y.intercept_)

    print(reg_x.predict(basis)/ m - errors[:, 1])
    print(reg_y.predict(basis)/ m - errors[:, 0])
    print('mean errors', np.mean(errors, axis=0))
    plate2_corrected = plate2 + np.array([reg_y.predict(basisY), reg_x.predict(basis)]).T / m

    initial_guess = result.x
    print('initial guess:', initial_guess) 
    result = scipy.optimize.minimize(get_fitfunc(plate2_corrected, target2), initial_guess, method = 'Nelder-Mead')  # BFGS doesn't like something here
    print(result)
    print('rms error corrected solve (arcseconds): ', np.degrees(result.fun**0.5) * 3600)
    # TODO: re-absorb the linear and constant coefficients into changes in plate_scale, roll, dec and ra
    return result, plate2_corrected, reg_x, reg_y
