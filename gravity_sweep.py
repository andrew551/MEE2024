import transforms
import numpy as np
from MEE2024util import output_path, date_string_to_float
import refraction_correction
import distortion_polynomial
import matplotlib.pyplot as plt
import scipy



def gravity_sweep(stardata0, plate2, initial_guess, image_size, mask_select, starttime, basename, options):
    options = options.copy()
    options['no_plot'] = True
    rmses = []
    Ls = []
    astrocorrect = refraction_correction.AstroCorrect()

    def error_func(g):
        stardata, alt, az = astrocorrect.correct_ra_dec(stardata0, options, var_grav=g/1.751)
        stardata.select_indices(mask_select)
        result, plate2_corrected, coeff_x, coeff_y = distortion_polynomial.do_cubic_fit(plate2, stardata, initial_guess, image_size, options)
        transformed_final = transforms.linear_transform(result, plate2_corrected, image_size)
        mag_errors = np.linalg.norm(transformed_final - stardata.get_vectors(), axis=1)
        errors_arcseconds = np.degrees(mag_errors)*3600
        mean_rms = np.degrees(np.mean(mag_errors**2)**0.5)*3600
        return mean_rms
    
    for g in np.linspace(-0.5, 3, 20):
        mean_rms = error_func(g)
        rmses.append(mean_rms)
        Ls.append(g)

    result1 = scipy.optimize.minimize(error_func, 0, method = 'Nelder-Mead')
    print(result1)
        
    plt.plot(Ls, rmses)
    plt.xlabel('L (arcsec)')
    plt.ylabel('mean rms (arcsec)')
    plt.title('L constant rms sweep')
    naive_error = result1.fun/np.sqrt(plate2.shape[0])
    string = f"deflection constant = {result1.x[0]:.5f}\ndifference vs. accepted value: {100*(result1.x[0]-1.751)/1.751:.3f}%\n\ndeflected star position rms = {result1.fun:.3f} arcsec\nrms / sqrt(nstars) = {naive_error:.5f} arcsec\nnaive uncertainty estimate = {100*naive_error/1.751:.1f}%\n"

    plt.annotate(string, xy = (result1.x[0], result1.fun), xytext=(result1.x[0]-0.3, result1.fun+0.02), fontsize=14, arrowprops=dict(facecolor='black', shrink=0.05))
    plt.savefig(output_path(f'ECLIPSE_L_SWEEP{starttime}__{basename}.png', options), dpi=400)
    plt.show()
