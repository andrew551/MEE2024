"""
@author: Andrew Smith
Version 6 January 2024
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
from collections import Counter
from skimage import measure
import cv2
from skimage.morphology import convex_hull_image
from skimage.transform import downscale_local_mean, resize
import skimage.data._fetchers # fix py2exe bug

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
# upon further investigation, this filter has been found to not be so useful
# and has been disabled
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

# find the largest connected region of saturated pixels
# and set it to a dark value

def remove_saturated_blob(img, sat_val=65535, radius=100, min_size=20000, downscale=8, perform=True):
    if not perform:
        return img, np.zeros(img.shape, dtype=int)
    if sat_val is None:
        sat_val = np.max(img)
    down_downscaled = downscale_local_mean(img, (downscale, downscale))
    
    is_sat = down_downscaled==sat_val
    #print(np.max(img),np.max(down_downscaled))

    labels = measure.label(is_sat, connectivity=1)
    areas = [region.area for region in measure.regionprops(labels)]
    #print(areas)
    if not areas or max(areas)*downscale**2 < min_size:
        return img, np.zeros(img.shape, dtype=int)
    mask = labels == (np.argmax(areas)+1)
    chull = convex_hull_image(mask)
    #contours_mask = measure.find_contours(mask) # alternative method could use contours...
    mask_expand = np.copy(chull)
    radius = radius // downscale
    for i in range(-1, 2):
        for j in range(-1, 2):
            mask_t = np.roll(chull, (i*radius, j*radius), axis=(0,1))
            if j > 0:
                mask_t[:, :j*radius] = 0
            elif j < 0:
                mask_t[:, j*radius:] = 0
            if i > 0:
                mask_t[:i*radius, :] = 0
            elif i < 0:
                mask_t[i*radius:, :] = 0
            mask_expand = np.logical_or(mask_expand, mask_t)
    
    #plt.imshow(mask_expand^chull)
    #plt.show()
    #print(np.sum(mask_expand), np.sum(chull))
    #plt.show()
    
    up_scaled = resize(mask_expand, img.shape).astype(bool)
    img = np.copy(img) # deep copy
    img[up_scaled] = np.percentile(img, 5) # make it dark
    return (img, up_scaled)


# try to find the optimal alignment vector between two sets of centroids
# two-step implementation (first rough, then more accurate)
def attempt_align(c1, c2, options, guess = (0,0)):
    if not c1.size or not c2.size:
        print("ERROR: no star centroids found")
        return None, None, None, None, None
    m = min(min(c1.shape[0], c2.shape[0]), options['m'])
    c1 = c1.reshape((c1.shape[0], -1))
    c2 = c2.reshape((c2.shape[0], -1))

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

def filter_bad_centroids(centroids, mask, r=20):
    ret = []
    for centroid in centroids:
        x0, x1 = int(centroid[0]), int(centroid[1])
        flag = True
        for i in range(x0-r, x0+r+1):
            for j in range(x1-r, x1+r+1):
                if i < 0 or j < 0 or i >= mask.shape[0] or j >= mask.shape[0]:
                    continue
                if mask[i, j]:
                    flag = False
        if flag:
            ret.append(centroid)
    return np.array(ret)
        

def do_stack(files, darkfiles, flatfiles, options):
    starttime = str(time.time())
    logpath = 'LOG'+starttime+'.txt'
    clearlog(logpath, options)
    logme(logpath, options, 'using options:'+str(options))
    logme(logpath, options, 'stacking files:'+str(files))
    logme(logpath, options, 'using darks:'+str(darkfiles))
    logme(logpath, options, 'using flats:'+str(flatfiles))
    logme(logpath, options, 'using database:'+str(options['database']))
    print('using options:'+str(options))
    print('stacking files:'+str(files))
    print('using darks:'+str(darkfiles))
    print('using flats:'+str(flatfiles))
    print('using database:'+str(options['database']))

    imgs = do_loop_with_progress_bar(files, open_image, message='Opening files...')
    dark = np.mean(np.array(open_images(darkfiles)), axis=0) if darkfiles else np.zeros(imgs[0].shape, dtype=imgs[0].dtype)
    flat = np.mean(np.array(open_images(flatfiles)), axis=0) if flatfiles else np.ones(imgs[0].shape, dtype=float)

    if options['save_dark_flat']:
        if darkfiles:
            fits.writeto(output_path('DARK_STACK'+starttime+'.fit', options), dark)
        if flatfiles:
            fits.writeto(output_path('FLAT_STACK'+starttime+'.fit', options), flat)

    desatblob = do_loop_with_progress_bar(imgs, remove_saturated_blob, message='Processing images (step 1)...', sat_val=None, radius = options['blob_radius_extra'], perform=options['delete_saturated_blob'])
    deblobbed_imgs = [t[0] for t in desatblob]
    masks = [t[1] for t in desatblob]
    reg_imgs = do_loop_with_progress_bar(deblobbed_imgs, lambda img: (img-dark)/flat, message='Processing images (step 2)...')
    #filtered_imgs = do_loop_with_progress_bar(reg_imgs, remove_saturated_blob, message='Processing images(2)...', d2=4)
    centroids = do_loop_with_progress_bar(reg_imgs, tetra3.get_centroids_from_image, message='Finding centroids...')
    centroids = [filter_bad_centroids(x, mask, r=options['centroid_gap_blob']) for x, mask in zip(centroids, masks)]

    # simple stacking: use the first image as the "key" and fit all others to it
    shifts = []
    rms_errors = []
    deltas = []
    prev = (0, 0)
    used_stars_stacking = Counter()
    for i in range(1, len(imgs)):
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
        used_stars_stacking.update(matches1.keys())
        print(matches1)
    print(rms_errors)
    print(shifts)
    # show stars used in stacking
    used_centroids = np.array([centroids[0][s] for s in used_stars_stacking])
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.scatter(used_centroids[:, 1], used_centroids[:, 0], marker='x')
    plt.title('Used stars for stacking')
    plt.gca().invert_yaxis()
    plt.grid()
    for k, v in used_stars_stacking.items():
        plt.gca().annotate(str(v), tuple(reversed(centroids[0][k])))
    plt.savefig(output_path('USEDSTARS'+starttime+'.png', options), dpi=600)
    if options['flag_display']:
        plt.show()    

    
    # show residual 2D errors
    plt.clf()
    for i in range(1, len(imgs)):
        if shifts[i-1] is None:
            continue
        lbl = '$\\Delta_{0' + str(i) + ',rms} = ' + format(rms_errors[i-1], '.3f') + '$'
        plt.scatter(deltas[i-1][:, 1], deltas[i-1][:, 0], label = lbl)
    plt.gca().set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('2D residuals between centroids')
    plt.grid()
    plt.savefig(output_path('TWOD_RESIDUALS'+starttime+'.png', options), dpi=600)
    if options['flag_display']:
        plt.show()
    #TODO: can add linear correlation of Dx, Dy to {px, py}. If it is non-zero it may indicate a rotation
    plt.clf()
    plt.scatter(centroids[0][:, 1], centroids[0][:, 0], label = str(0))
    for i in range(1, len(imgs)):
        if shifts[i-1] is None:
            continue
        plt.scatter(centroids[i][:, 1]+shifts[i-1][1], centroids[i][:, 0]+shifts[i-1][0], label = str(i))
    plt.gca().set_aspect('equal')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('Centroids found on each image')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.savefig(output_path('CentroidsALL'+starttime+'.png', options), bbox_inches="tight", dpi=600)
    if options['flag_display']:
        plt.show()

    # now do actual stacking
    shifted_images = [reg_imgs[0]] + [np.roll(img, shift.astype(int), axis = (0, 1)) for img, shift in zip(reg_imgs[1:], shifts) if not shift is None]
    stacked = np.mean(np.array(shifted_images), axis = 0)
    
    # rescale stacked to 16 bit integers
    stacked16 = ((stacked-np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 65535).astype(np.uint16)
    fits.writeto(output_path('STACKED'+starttime+'.fit', options), stacked16)
    # find centroids on the stacked image
    centroids_stacked = tetra3.get_centroids_from_image(stacked)
    centroids_stacked = filter_bad_centroids(centroids_stacked, masks[0], r=options['centroid_gap_blob']) # use 0th mask here
    
    np.savetxt(output_path('STACKED_CENTROIDS'+starttime+'.txt', options), centroids_stacked)
    logme(logpath, options, f'saving {centroids_stacked.shape[0]} centroid pixel coordinates')
    # plate solve
    flag_found_IDs = False
    if options['database']:
        t3 = tetra3.Tetra3(load_database=options['database']) #tyc_dbase_test3 #hip_database938
        solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True)
        #solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True, fov_estimate=5, fov_max_error=1, distortion = (-0.0020, -0.0005))
        print(solution)
        logme(logpath, options, str(solution))
        # TODO identify stars using catalogue
        # and save catalogue ids of each identified star (currently only around 20 are matched by the tetra software "for free"
        if not solution['RA'] is None:
            df = pd.DataFrame({'px': np.array(solution['matched_centroids'])[:, 1],
                               'py': np.array(solution['matched_centroids'])[:, 0],
                               'ID': solution['matched_catID'],
                               'RA': np.array(solution['matched_stars'])[:, 0],
                               'DEC': np.array(solution['matched_stars'])[:, 1],
                               'magV': np.array(solution['matched_stars'])[:, 2]})
            
            df.to_csv(output_path('STACKED_CENTROIDS_MATCHED_ID'+starttime+'.csv', options))
            flag_found_IDs = True
        else:
            print("ERROR: platesolve failed to identify location")
    else:
        print('no database provided, so skipping platesolve')
        logme(logpath, options, 'no database provided, so skipping platesolve')

    plt.clf()
    plt.title(f'Largest {min(options["d"], len(centroids_stacked))} of {len(centroids_stacked)} stars found on stacked image')
    plt.imshow(stacked, cmap='gray_r', vmin=np.percentile(stacked, 50), vmax=np.percentile(stacked, 95))
    plt.scatter(centroids_stacked[:options["d"], 1]-0.5, centroids_stacked[:options["d"], 0]-0.5, marker='x') # subtract half pixel to align with image properly
    if flag_found_IDs:
        for index, row in df.iterrows():
            plt.gca().annotate(str(int(row['ID']) if isinstance(row['ID'], float) else row['ID']) + f'\nMag={row["magV"]:.1f}', (row['px'], row['py']), color='r')
    plt.savefig(output_path('CentroidsStackGood'+starttime+'.png', options), bbox_inches="tight", dpi=600)
    if options['flag_display']:
        plt.show()
    
        
    write_complete(logpath, options)

