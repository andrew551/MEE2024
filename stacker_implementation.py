"""
@author: Andrew Smith
Version 3 January 2024
"""
from astropy.io import fits
from pathlib import Path
import glob
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import tetra3
import math
from scipy.optimize import minimize
import MEE2024util
import time
from MEE2024util import output_path, logme, clearlog, write_complete
import datetime
import pandas as pd
import PySimpleGUI as sg


# TODO: replace usage of np.roll with something which only translates (currently pixels near the edges of the image will be messed up)

# return fit file image as np array
def open_image(file):
    with fits.open(file) as hdul:
        if 'PRIMARY' in hdul:
            return hdul['PRIMARY'].data
        else:
            return hdul[0].data

def open_images(files):
    return [open_image(file) for file in files]
    
# apply a min-filter to the image
# effective at removing random speckles of bright noise,
# while maintaining acceptable errors in centroid positions
def filter_min(img, d2=4):
    return img # currently disabled
    result = np.copy(img)
    a = math.floor(d2**0.5)
    for i in range(-a, a+1):
        for j in range(-a, a+1):
            if i*i+j*j > d2:
                continue
            result = np.minimum(result, np.roll(img, (i, j), axis=(0, 1)))
    return result

# try to find the optimal alignment vector between two sets of centroids
# two-step implementation (first rough, then more accurate)
def attempt_align(c1, c2, options, guess = (0,0)):
    m = min(min(c1.shape[0], c2.shape[0]), options['m'])
    c1a = c1[:m, :]
    c2a = c2[:m, :]
    a = np.ones((m, m, 2))
    def loss_fxn(b):
        d = c1a*a - np.swapaxes(c2a*a, 0, 1) - b
        norms = np.minimum(np.linalg.norm(d, axis=2)**1.5, options['cutoff']) # 1.5 power norms of distances (capped?)
        return np.sum(np.min(norms, axis = 0)) / c1.shape[0]
    result = minimize(loss_fxn, guess)
    print(result)

    #plt.scatter(c1a[:, 1], c1a[:, 0])
    #plt.scatter(c2a[:, 1], c2a[:, 0])
    #plt.show()
    
    def enumerate_matches(b, eps=2):
        d = np.reshape(c1, (c1.shape[0], 1, -1)) - np.swapaxes(np.reshape(c2, (c2.shape[0], 1, -1)), 0, 1) - b
        norms = np.linalg.norm(d, axis=2)
        matches1 = {}
        matches2 = {}
        #print(norms[:5, :5])
        norms[options['n']:, options['n']:] = 99999
        while 1:
            ind = np.unravel_index(np.argmin(norms), norms.shape)
            print('info', ind, norms[ind])
            if norms[ind] > eps:
                break
            i, j = tuple(ind)
            if not i in matches1 and not j in matches2:
                matches1[i] = j
                matches2[j] = i
                norms[i, :] = 999999
                norms[:, j] = 999999
        return matches1, matches2
    matches1, matches2 = enumerate_matches(result.x, eps=options['pxl_tol'])
    if len(matches1) == 0:
        print("ERROR: no matched stars between images ... problably this means failure")
        return None, None, None, None, None
    vec1 = np.array([c1[i, :] for i in matches1 if i < options['n']])
    vec2 = np.array([c2[matches1[i], :] for i in matches1 if i < options['n']])
    #print(vec1, vec2)
    def loss_fxn2(b):
        return np.linalg.norm(vec1 - vec2 - b) ** 2

    result2 = minimize(loss_fxn2, guess)
    print(result2)
    print(vec1.shape)
    return result.x, matches1, matches2, result2.x, (result2.fun/vec1.shape[0])**0.5

def do_loop_with_progress_bar(items, fxn, message='Progress', **kwargs):
    layout = [[sg.Text(message)], [sg.ProgressBar(max_value=len(items), orientation='h', size=(20, 20), key='progress')]]
    window = sg.Window('Progress Meter', layout, finalize=True)
    progress_bar = window['progress']
    ret = []
    progress_bar.update_bar(0)
    for i in range(len(items)): 
        ret.append(fxn(items[i], **kwargs))
        progress_bar.update_bar(i+1)
    window.close()
    return ret

def do_stack(files, darkfiles, flatfiles, options):
    starttime = str(time.time())
    logpath = 'LOG'+starttime+'.txt'
    clearlog(logpath, options)
    logme(logpath, options, 'using options:'+str(options))
    logme(logpath, options, 'stacking files:'+str(files))
    logme(logpath, options, 'using darks:'+str(darkfiles))
    logme(logpath, options, 'using flats:'+str(flatfiles))
    print('using options:'+str(options))
    print('stacking files:'+str(files))
    print('using darks:'+str(darkfiles))
    print('using flats:'+str(flatfiles))


    imgs = do_loop_with_progress_bar(files, open_image, message='Opening files...')
    dark = np.mean(np.array(open_images(darkfiles)), axis=0) if darkfiles else np.zeros(imgs[0].shape, dtype=imgs[0].dtype)
    flat = np.mean(np.array(open_images(flatfiles)), axis=0) if flatfiles else np.ones(imgs[0].shape, dtype=float)

    if options['save_dark_flat']:
        if darkfiles:
            fits.writeto(output_path('DARK_STACK'+starttime+'.fit', options), dark)
        if flatfiles:
            fits.writeto(output_path('FLAT_STACK'+starttime+'.fit', options), flat)

    #reg_imgs = [(img-dark)/flat for img in imgs]
    reg_imgs = do_loop_with_progress_bar(imgs, lambda img: (img-dark)/flat, message='Processing images (1)...')
    filtered_imgs = do_loop_with_progress_bar(reg_imgs, filter_min, message='Processing images(2)...', d2=4)
    #filtered_imgs = [filter_min(img, d2=4) for img in reg_imgs]
    centroids = do_loop_with_progress_bar(filtered_imgs, tetra3.get_centroids_from_image, message='Finding centroids...')

    #centroids = [tetra3.get_centroids_from_image(img) for img in filtered_imgs]
    # simple stacking: use the first image as the "key" and fit all others to it
    shifts = []
    rms_errors = []
    deltas = []
    prev = (0, 0)
    for i in range(1, len(filtered_imgs)):
        shift, matches1, matches2, shift2, fun2 = attempt_align(centroids[0], centroids[i], options, guess=prev)
        print(shift, shift2, fun2)
        shifts.append(shift2)
        if shift2 is None:
            print(f'NOTE: failure to find centroid match on frame # {i}')
            rms_errors.append(None)
            deltas.append(None)
            continue
        prev = shift2
        rms_errors.append(fun2)
        deltas.append(np.array([centroids[0][j] - centroids[i][matches1[j]] for j in matches1 if j < options['n']]))
    print(rms_errors)
    print(shifts)
    # show residual 2D errors
    plt.clf()
    for i in range(1, len(filtered_imgs)):
        if shifts[i-1] is None:
            continue
        lbl = '$\\Delta_{0' + str(i) + ',rms} = ' + format(rms_errors[i-1], '.3f') + '$'
        plt.scatter(deltas[i-1][:, 1], deltas[i-1][:, 0], label = lbl)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('2D residuals between centroids')
    plt.grid()
    plt.savefig(output_path('TWOD_RESIDUALS'+starttime+'.png', options))
    if options['flag_display']:
        plt.show()
    #TODO: can add linear correlation of Dx, Dy to {px, py}. If it is non-zero it may indicate a rotation
    plt.clf()
    plt.scatter(centroids[0][:, 1], centroids[0][:, 0], label = str(0))
    for i in range(1, len(filtered_imgs)):
        if shifts[i-1] is None:
            continue
        plt.scatter(centroids[i][:, 1]+shifts[i-1][1], centroids[i][:, 0]+shifts[i-1][0], label = str(i))
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Centroids found on each image')
    plt.grid()
    plt.savefig(output_path('CentroidsALL'+starttime+'.png', options))
    if options['flag_display']:
        plt.show()

    # now do actual stacking

    shifted_images = [reg_imgs[0]] + [np.roll(img, shift.astype(int), axis = (0, 1)) for img, shift in zip(reg_imgs[1:], shifts) if not shift is None]

    stacked = np.mean(np.array(shifted_images), axis = 0)
    #plt.clf()
    #plt.imshow(stacked, cmap='gray_r', vmin=np.percentile(stacked, 50), vmax=np.percentile(stacked, 95))
    #if options['flag_display']:
    #    plt.show()
    # rescale stacked to 16 bit integers
    stacked16 = ((stacked-np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 65535).astype(np.uint16)
    fits.writeto(output_path('STACKED'+starttime+'.fit', options), stacked16)

    # plate solve
    centroids_stacked = tetra3.get_centroids_from_image(stacked)
    plt.clf()
    plt.title('Largest 20 stars found on stacked image')
    plt.imshow(stacked, cmap='gray_r', vmin=np.percentile(stacked, 50), vmax=np.percentile(stacked, 95))
    plt.scatter(centroids_stacked[:20, 1], centroids_stacked[:20, 0], marker='x')
    plt.savefig(output_path('CentroidsStackGood'+starttime+'.png', options))
    if options['flag_display']:
        plt.show()
        
    
    np.savetxt(output_path('STACKED_CENTROIDS'+starttime+'.txt', options), centroids_stacked)
    logme(logpath, options, f'saving {centroids_stacked.shape[0]} centroid pixel coordinates')
    t3 = tetra3.Tetra3(load_database='hip_database938') #tyc_dbase_test3 #hip_database938
    solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True)
    #solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True, fov_estimate=5, fov_max_error=1, distortion = (-0.0020, -0.0005))
    print(solution)
    logme(logpath, options, str(solution))
    # TODO identify stars using catalogue
    # and save catalogue ids of each identified star (currently only around 20 are matched by the tetra software "for free"

    df = pd.DataFrame({'px': np.array(solution['matched_centroids'])[:, 1],
                       'py': np.array(solution['matched_centroids'])[:, 0],
                       'ID': solution['matched_catID']})
    
    df.to_csv(output_path('STACKED_CENTROIDS_MATCHED_ID'+starttime+'.csv', options))
    write_complete(logpath, options)

