"""
@author: Andrew Smith
Version 6 May 2024
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
from MEE2024util import output_path, _version, setup_logger
import datetime
import pandas as pd
import PySimpleGUI as sg
from collections import Counter
from skimage import measure
import cv2
from skimage.morphology import convex_hull_image
from skimage.transform import downscale_local_mean, resize
import skimage.data._fetchers # fix py2exe bug
import scipy
import database_cache
import os
import shutil
import json
import logging
import platesolve_triangle
import multiprocessing
import cProfile
import warnings

# return fit file image as np array
def open_image(file):
    with fits.open(file) as hdul:
        if 'PRIMARY' in hdul:
            return hdul['PRIMARY'].data
        else:
            return hdul[0].data

def open_images(files):
    return [open_image(file) for file in files]

def roll_fillzero(src, shift):
    rolled = np.roll(src, shift=shift, axis=(0,1))
    i, j = shift
    if j > 0:
        rolled[:, :j] = 0
    elif j < 0:
        rolled[:, j:] = 0
    if i > 0:
        rolled[:i, :] = 0
    elif i < 0:
        rolled[i:, :] = 0
    return rolled

def expand_mask(src, radius, target_size=None):
    mask_expand = np.copy(src).astype(bool)
    for i in range(-1, 2):
        for j in range(-1, 2):
            mask_t = roll_fillzero(src, (i*radius, j*radius))
            mask_expand = np.logical_or(mask_expand, mask_t)
    if not target_size is None:
        mask_expand = resize(mask_expand, target_size)
    return mask_expand.astype(bool)

def expand_labels(labels):
    ret = np.copy(labels)
    for i in range(-1, 2):
        for j in range(-1, 2):
            ret = np.maximum(ret, roll_fillzero(labels, (i, j)))
    return ret

# find the largest connected region of saturated pixels
# and set it to a dark value

def remove_saturated_blob(img, sat_val=65535, radius=100, radius2=150, min_size=20000, downscale=8, blob_saturation=1, perform=True):
    if not perform:
        return img, np.zeros(img.shape, dtype=int), np.zeros(img.shape, dtype=int)
    if sat_val is None:
        sat_val = np.max(img)*blob_saturation # change from maximum to 99th percentile times 0.97
    down_downscaled = downscale_local_mean(img, (downscale, downscale))
    
    is_sat = down_downscaled>=sat_val
    #print(np.max(img),np.max(down_downscaled))

    labels = measure.label(is_sat, connectivity=1)
    areas = [region.area for region in measure.regionprops(labels)]
    #print(areas)
    if not areas or max(areas)*downscale**2 < min_size:
        return img, np.zeros(img.shape, dtype=int), np.zeros(img.shape, dtype=int)
    mask = labels == (np.argmax(areas)+1)
    chull = convex_hull_image(mask)
    #contours_mask = measure.find_contours(mask) # alternative method could use contours...
    mask_1 = expand_mask(chull, radius//downscale, img.shape)
    mask_2 = expand_mask(chull, radius2//downscale, img.shape)
    
    #plt.imshow(mask_expand^chull)
    #plt.show()
    #print(np.sum(mask_expand), np.sum(chull))
    #plt.show()
    
    img = np.copy(img) # deep copy
    img[mask_1] = np.percentile(img, 5) # make it dark
    return (img, mask_1, mask_2)


# try to find the optimal alignment vector between two sets of centroids
# two-step implementation (first rough, then more accurate)
def attempt_align(c1, c2, options, guess = (0,0), framenum=-1):
    if not c1.size or not c2.size:
        print("ERROR: no star centroids found")
        raise Exception(f"The stacking procedure failed to match stars between frame 0 and {framenum}! No centroids found! Check that all frames are okay,\nin the same field, \
and that you have chosen appropriate centroid detection threshholds")
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
        raise Exception(f"The stacking procedure failed to match stars between frame 0 and {framenum}! Check that all frames are okay,\nin the same field, \
and that you have chosen appropriate centroid detection threshholds")
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

def multiprocessing_fxn(q, fxn, item, i, **kwargs):
    #print(item, kwargs)
    q.put((i, fxn(item, **kwargs)))

def do_loop_with_progress_bar_multiprocessing(items, fxn, message='Progress', nthreads=4, **kwargs):
    layout = [[sg.Text(message)], [sg.ProgressBar(max_value=len(items), orientation='h', size=(20, 20), key='progress')]]
    window = sg.Window('Progress Meter', layout, finalize=True)
    progress_bar = window['progress']
    ret = [None for _ in items]
    progress_bar.update_bar(0)
    q = multiprocessing.Queue()
    procs = []
    for i, item in enumerate(items[:nthreads]):
        p = multiprocessing.Process(target=multiprocessing_fxn, args = (q, fxn, item, i), kwargs=kwargs)
        p.start()
        procs.append(p)
    n_ret = 0
    i = nthreads
    while n_ret < len(items):
        x = q.get()
        ret[x[0]] = x[1] # this way order of inputs is preserved
        n_ret += 1
        progress_bar.update_bar(n_ret)
        if i < len(items):
            p = multiprocessing.Process(target=multiprocessing_fxn, args = (q, fxn, items[i], i), kwargs=kwargs)
            p.start()
            procs.append(p)
            i += 1
    for p in procs:
        p.join()
    window.close()
    return ret

def filter_bad_centroids(centroids_data, mask2, shape):
    ret = []
    for data in centroids_data:
        x0, x1 = int(data[2][0]), int(data[2][1])
        if not x0 <= 0 and not x0 >= shape[0]-1 and not x1 <= 0 and not x1 >= shape[1]-1 and not mask2[x0, x1]:
            ret.append(data)
    return ret

# remove centroids within f pixels of image edge
def filter_very_edgy_centroids(centroids_data, img, f=5):
    ret = []
    for data in centroids_data:
        x0, x1 = int(data[2][0]), int(data[2][1])
        if x0 >= f and x0 <= img.shape[0] - f - 1 and x1 >= f and x1 <= img.shape[1] - f - 1:
            ret.append(data)
    return ret

# this function thies to remove 'centroids' that are actually
# edge artifacts by looking for an anomaly in the gradients distributions near the centroid
# also removes all points within 3 pixels of image edge
def filter_edgy_centroids(centroids_data, img, f=3, d=16, thresh=2, edge_threshold=20):
    ds = d * np.array([[0,0], [1,0], [-1, 0], [0, 1], [0, -1]])
    ret = []
    for data in centroids_data:
        x0, x1 = int(data[2][0]), int(data[2][1])
        if x0 < d or x0 > img.shape[0] - d - 1 or x1 < d or x1 > img.shape[1] - d - 1:
            if x0 >= f and x0 <= img.shape[0] - f - 1 and x1 >= f and x1 <= img.shape[1] - f - 1:
                ret.append(data) # pass on filtering points near image edge, but remove points really close to edge
            continue

        field = img[x0-d:x0+d+1, x1-d:x1+d+1]
        
        diff0 = np.abs(np.diff(field, axis=0))
        diff1 = np.abs(np.diff(field, axis=1))
        
        max0 = np.max(diff0, axis=0)
        max1 = np.max(diff1, axis=1)


        median_max = np.median([max0, max1])

        joined = np.concatenate((diff0.flatten(), diff1.flatten()))
        
        lq = np.percentile(joined, 40)
        uq = np.percentile(joined, 60)

        if uq-lq==0 or (median_max - (lq+uq)/2) / (uq-lq) > edge_threshold:
            print('deleting edgy centroid: ', x0, x1)
        else:
            ret.append(data)
        
        
        '''    
        dvals = [np.mean(img[x0-r+d_i[0]:x0+r+d_i[0], x1-r+d_i[1]:x1+r+1+d_i[1]]) for d_i in ds]
        ratio_x = (dvals[0] - dvals[1]) / (dvals[0] - dvals[2])
        ratio_y = (dvals[0] - dvals[3]) / (dvals[0] - dvals[4])
        #print('ratio_xy', ratio_x, ratio_y) 
        if ratio_x < 0 or ratio_y < 0 or np.abs(np.log(ratio_x)) > thresh or np.abs(np.log(ratio_y)) > thresh:
            print('deleting edgy centroid: ', x0, x1)
        else:
            ret.append(data)
        '''
    return ret
            
    

def get_centroids_blur(img_mask2, ksize=17, r_max=10, options={}, gauss=False, debug_display=False):
    t_start = time.time()
    img, mask, mask2 = img_mask2
    if not options['centroid_gaussian_subtract']:
        centroids = tetra3.get_centroids_from_image(img)
        return [(-1, -1, x) for x in centroids] # return tetra centroids
    if options['background_subtraction_mode'] =='Gaussian':
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        inner = 3
        blur = (cv2.blur(img, (ksize, ksize)) - cv2.blur(img, (inner, inner)) * (inner**2/ksize**2)) * (ksize**2 / (ksize**2-inner**2))
    sub = img-blur
    sub[mask2] = 0

    squared = sub*sub
    large = np.percentile(squared, 95)
    squared[mask2] = large
    squared[squared > large*10] = large*10
    local_variance = scipy.ndimage.filters.uniform_filter(squared, size=(50, 50))

    #plt.imshow(local_variance)
    #plt.show()

    data = np.maximum(sub / np.sqrt(local_variance) - options['sigma_subtract'], 0)

    passed = data > options['centroid_gaussian_thresh']
    passed[expand_mask(mask2, 8)] = 0 # TODO: reflect on this quick fix to edge problems
    #plt.imshow(data, cmap='gray_r', vmin=4, vmax=5)
    #plt.show()
    print("--- %s seconds for centroid finding (prepare)---" % (time.time() - t_start))
    centroid_labels = measure.label(passed, connectivity=1)
    centroid_labels_exp = expand_labels(centroid_labels) # expand by one more ring of pixels
    properties = measure.regionprops(centroid_labels, data)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice') # RuntimeWarning: invalid value encountered in scalar divide
        warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide')
        properties_exp = measure.regionprops(centroid_labels_exp, data)

    print("--- %s seconds for centroid finding (labelling)---" % (time.time() - t_start))

    
    areas = [region.area for region in properties]
    centroids = [region.centroid_weighted for region in properties_exp]
    fluxes = []
    for i in range(len(centroids)):
        if np.isnan(centroids[i][0]):
            fluxes.append(None)
            continue
        around_data = data[int(centroids[i][0])-r_max:int(centroids[i][0])+r_max+1, int(centroids[i][1])-r_max:int(centroids[i][1])+r_max+1]
        around_labels = centroid_labels_exp[int(centroids[i][0])-r_max:int(centroids[i][0])+r_max+1, int(centroids[i][1])-r_max:int(centroids[i][1])+r_max+1]
        fluxes.append(np.sum(around_data[around_labels==i+1]))


    if debug_display:
        sz = 10
        for i in range(len(centroids)):
            x0, x1 = int(centroids[i][0]), int(centroids[i][1])
            data_near = data[x0-sz:x0+sz+1,x1-sz:x1+sz+1]
            diffences = np.diff(data_near, axis = 0)
            if (np.count_nonzero(diffences==0) > 10 or areas[i] < options['min_area']) and not abs(x0-1291) < 20:
                print('assume fake:', centroids[i],np.count_nonzero(diffences==0))
                continue
            #if not data_near.shape == (sz*2+1, sz*2+1) or areas[i] < 10:
            #    continue
            if 1:
                print(centroids[i])
                fig, ax = plt.subplots()
                plt.imshow(data_near)
                show_scanlines(data_near, fig, ax)
                plt.show()
    
    '''
    acc_centroids = []
    sz = 10
    for i in range(len(centroids)):
        x0, x1 = int(centroids[i][0]), int(centroids[i][1])
        data_near = data[x0-sz:x0+sz+1,x1-sz:x1+sz+1]
        if not data_near.shape == (sz*2+1, sz*2+1) or areas[i] < 10:
            acc_centroids.append(centroids[i])
            continue
        if 1:
            fig, ax = plt.subplots()
            plt.imshow(data_near)
            show_scanlines(data_near, fig, ax)
            plt.show()
        correction = photutils.centroids.centroid_2dg(data_near) - (np.array(data_near.shape)-1) / 2
        print(correction)
        acc_centroids.append((x0+correction[0], x1+correction[1]))
    '''

    

    

    sorted_c = sorted([(f, a, c) for f, c, a in zip(fluxes, centroids, areas) if a >= options['min_area'] and not np.isnan(c[0])], reverse=True)
    print(f"n centroids initial {len(sorted_c)}")
    # sanity check: mean(3x3 around centroid) > mean(5x5 around centroid) > mean(7x7) > mean(9x9) around centroid in raw img
    # this should help heal with fake centroids due to artifacts like dead pixels

    def sanity_check(centroid):
        x0, x1 = int(centroid[0]), int(centroid[1])
        mean_sequence = [np.mean(img[x0-r:x0+r+1, x1-r:x1+r+1]) for r in range(1, 5)]
        for i in range(len(mean_sequence) - 1):
            if mean_sequence[i] < mean_sequence[i+1]:
                return False
        return True
    if options['sanity_check_centroids']:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice') # RuntimeWarning: invalid value encountered in scalar divide
            warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide')
            sorted_c = [cc for cc in sorted_c if sanity_check(cc[2])]
            print(f"n centroids sanity-filtered {len(sorted_c)}")
    #sorted_c = [(f, c) for f,c in zip(fluxes, centroids)], reverse=True)
    print("--- %s seconds for centroid finding (all)---" % (time.time() - t_start))
    print('found:', sorted_c)
    return sorted_c
    
    
    if 1:
        fig, ax = plt.subplots()
        plt.imshow(sub, vmin=np.percentile(sub,80), vmax=np.percentile(sub, 95))
        show_scanlines(sub, fig, ax)
        plt.show(block=True)
    centroids = tetra3.get_centroids_from_image(sub, sigma=2.5, min_area=1)
    print(centroids)
    return centroids

def show_scanlines(src_img, fig, ax):
    fig2, ax2 = plt.subplots(dpi=100, figsize=(5, 5))
    fig3, ax3 = plt.subplots(dpi=100, figsize=(5, 5))
    ax2.set_title('X-transcept')
    ax3.set_title('Y-transcept')
    line_x, = ax2.plot([], [], label='x-line')
    line_y, = ax3.plot([], [], label='y-line', color='orange')
    def plot_lines(x, y, xlim, ylim):
        x1, x2 = int(xlim[0]), int(xlim[1])
        ax2.set_xlim((x1, x2))
        data = src_img[int(y), x1:x2]
        if not data.size:
            return
        line_x.set_data(np.arange(x1, x2), data)
        ax2.set_ylim(np.min(data)*0.7, np.max(data)*1.3)

        y1, y2 = int(ylim[0]), int(ylim[1])
        y1, y2 = min(y1, y2), max(y1, y2)
        ax3.set_xlim((y1, y2))
        data2 = src_img[y1:y2, int(x)]
        if not data2.size:
            return
        line_y.set_data(np.arange(y1, y2), data2)
        ax3.set_ylim(np.min(data2)*0.7, np.max(data2)*1.3)
    def mouse_move(event):
        x = event.xdata
        y = event.ydata
        if x is not None and y is not None and x >= 0 and x < src_img.shape[1] and y > 0 and y < src_img.shape[0]:
            plot_lines(x, y, ax.get_xlim(), ax.get_ylim())
            fig2.canvas.draw_idle()
            fig3.canvas.draw_idle()
    cid = fig.canvas.mpl_connect('motion_notify_event', mouse_move)

def add_img_to_stack(data, output_array=None, count_array=None):
    img, shift = data # unpack tuple
    shift = (round(shift[0]), round(shift[1]))
    a1 = np.ones(count_array.shape, dtype=int)
    output_array += roll_fillzero(img, shift)
    count_array += roll_fillzero(a1, shift)

def open_img_and_preprocess(file, options = {}, dark=0, flat=1):
    img = open_image(file)
    desatblob_img, mask, mask2 = remove_saturated_blob(img, sat_val=None, radius = options['blob_radius_extra'], radius2 = options['blob_radius_extra']+options['centroid_gap_blob'], blob_saturation=options['blob_saturation_level']/100, perform=options['delete_saturated_blob'])
    reg_img = (desatblob_img - dark) / flat
    return reg_img, mask, mask2

def open_img_and_find_centroids(file, options = {}, dark=0, flat=1):
    reg_img, mask, mask2 = open_img_and_preprocess(file, options, dark, flat)
    centroids = get_centroids_blur((reg_img, mask, mask2), options=options)
    centroids_filtered = filter_bad_centroids(centroids, mask2, reg_img.shape)
    return centroids_filtered

def open_img_and_add_to_stack(data, output_array=None, count_array=None, options = {}, dark=0, flat=1):
    file, shift = data # unpack tuple
    reg_img, _, _ = open_img_and_preprocess(file, options, dark, flat)
    add_img_to_stack((reg_img, shift), output_array, count_array)
    
def do_stack(files, darkfiles, flatfiles, options):
    starttime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f'CENTROID_OUTPUT{starttime}'
    output_dir = Path(output_path(output_name, options))
    logpath = output_dir / f'LOG{starttime}.txt'
    data_dir = Path(output_dir) / 'data'
    os.mkdir(output_dir)
    os.mkdir(data_dir)
    print(f'logpath {logpath}')
    logger = setup_logger('logger'+starttime, logpath)
    logger.info('start time: ' + str(datetime.datetime.now()) + '\n')
    logger.info('using version:'+_version())
    logger.info('using options:'+str(options))
    logger.info('stacking files:'+str(files))
    logger.info('using darks:'+str(darkfiles))
    logger.info('using flats:'+str(flatfiles))
    print('using version:'+_version())
    print('using options:'+str(options))
    print('stacking files:'+str(files))
    print('using darks:'+str(darkfiles))
    print('using flats:'+str(flatfiles))

    

    imgs_0 = open_image(files[0])#do_loop_with_progress_bar([files[0]], open_image, message='Opening files...')
    _, masks_0, masks2_0 = remove_saturated_blob(imgs_0, sat_val=None, radius = options['blob_radius_extra'], radius2 = options['blob_radius_extra']+options['centroid_gap_blob'], blob_saturation=options['blob_saturation_level']/100, perform=options['delete_saturated_blob'])
    dark = np.mean(np.array(open_images(darkfiles)), axis=0) if darkfiles else np.zeros(imgs_0.shape, dtype=imgs_0.dtype)
    flat = np.mean(np.array(open_images(flatfiles)), axis=0) if flatfiles else np.ones(imgs_0.shape, dtype=float)

    print('image size:'+str(imgs_0.shape))
    logger.info('image size:'+str(imgs_0.shape))
    
    if options['save_dark_flat']:
        if darkfiles:
            fits.writeto(output_dir / ('DARK_STACK'+starttime+'.fit'), dark.astype(np.float32))
        if flatfiles:
            fits.writeto(output_dir / ('FLAT_STACK'+starttime+'.fit'), flat.astype(np.float32))
    t_start_c = time.time()
    #cProfile.runctx("do_loop_with_progress_bar(files, open_img_and_find_centroids, message='Finding all centroids...', dark = dark, flat=flat, options=options)", globals(), locals(), sort='cumtime')
    centroids_data = do_loop_with_progress_bar(files, open_img_and_find_centroids, message='Finding all centroids...', dark = dark, flat=flat, options=options)
    print("--- %s seconds for centroid finding---" % (time.time() - t_start_c))
    centroids = [np.array([x[2] for x in y]) for y in centroids_data]
    
    # simple stacking: use the first image as the "key" and fit all others to it
    shifts = [(0,0)]
    rms_errors = []
    deltas = []
    prev = (0, 0)
    used_stars_stacking = Counter()
    for i in range(1, len(files)):
        shift, matches1, matches2, shift2, fun2 = attempt_align(centroids[0], centroids[i], options, guess=prev, framenum=i)
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
    used_centroids = np.array([centroids[0][s] for s in used_stars_stacking]).reshape((-1, 2))
    plt.clf()
    plt.gca().set_aspect('equal')
    plt.scatter(used_centroids[:, 1], used_centroids[:, 0], marker='x')
    plt.title('Used stars for stacking')
    plt.xlim((0, imgs_0.shape[1]))
    plt.ylim((0, imgs_0.shape[0]))
    plt.gca().invert_yaxis()
    plt.grid()
    for k, v in used_stars_stacking.items():
        plt.gca().annotate(str(v), tuple(reversed(centroids[0][k])))
    plt.savefig(output_dir / ('USEDSTARS'+starttime+'.png'), dpi=600)
    if options['flag_display']:
        plt.show()    

    
    # show residual 2D errors
    plt.clf()
    for i in range(1, len(files)):
        if shifts[i-1] is None:
            continue
        lbl = '$\\Delta_{0' + str(i) + ',rms} = ' + format(rms_errors[i-1], '.3f') + '$'
        plt.scatter(deltas[i-1][:, 1], deltas[i-1][:, 0], label = lbl)
    plt.gca().set_aspect('equal')
    if len(files) < 30:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('2D residuals between centroids')
    plt.grid()
    plt.savefig(output_dir / ('TWOD_RESIDUALS'+starttime+'.png'), dpi=600)
    if options['flag_display']:
        plt.tight_layout()
        plt.show()
    #TODO: can add linear correlation of Dx, Dy to {px, py}. If it is non-zero it may indicate a rotation
    plt.clf()
    for i in range(len(files)):
        if shifts[i] is None:
            continue
        #print(centroids, shifts)
        plt.scatter(centroids[i][:, 1]+shifts[i][1], centroids[i][:, 0]+shifts[i][0], label = str(i))
    plt.gca().set_aspect('equal')
    if len(files) < 30:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title('Centroids found on each image')
    plt.xlim((0, imgs_0.shape[1]))
    plt.ylim((0, imgs_0.shape[0]))
    plt.gca().invert_yaxis()
    plt.grid()
    plt.savefig(output_dir / ('CentroidsALL'+starttime+'.png'), bbox_inches="tight", dpi=600)
    if options['flag_display']:
        #plt.tight_layout()
        plt.show()
    plt.close()
    # now do actual stacking
    #shifted_images = [reg_imgs[0]] + [np.roll(img, shift.astype(int), axis = (0, 1)) for img, shift in zip(reg_imgs[1:], shifts) if not shift is None]
    #stacked = np.mean(np.array(shifted_images), axis = 0)

    stack_array = np.zeros(imgs_0.shape)
    count_array = np.zeros(imgs_0.shape, dtype=int)
    #do_loop_with_progress_bar(list(zip(reg_imgs, shifts)), add_img_to_stack, message='Stacking images...', output_array=stack_array, count_array=count_array)
    do_loop_with_progress_bar(list(zip(files, shifts)), open_img_and_add_to_stack, message='Stacking images...',
                              output_array=stack_array, count_array=count_array, options = options, dark=dark, flat=flat)
    stacked = stack_array / count_array
    
    # rescale stacked to 16 bit integers
    stacked16 = ((stacked-np.min(stacked)) / (np.max(stacked) - np.min(stacked)) * 65535).astype(np.uint16)
    fits.writeto(output_dir / ('STACKED'+starttime+'.fit'), stacked16)
    if options['float_fits']:
        fits.writeto(output_dir / ('STACKED_FLOAT'+starttime+'.fit'), stacked.astype(np.float32))
    # find centroids on the stacked image
    centroids_stacked_data = get_centroids_blur((stacked, masks_0, masks2_0),
                        options=dict(options, **{'centroid_gaussian_subtract':options['centroid_gaussian_subtract'] or options['sensitive_mode_stack']}), # use sensitive mode if requested only for the stack
                        debug_display=False)
    centroids_stacked_data = filter_bad_centroids(centroids_stacked_data, masks2_0, imgs_0.shape) # use 0th mask here
    centroids_stacked_data = filter_very_edgy_centroids(centroids_stacked_data, stacked, f=options['img_edge_distance'])
    if options['remove_edgy_centroids']:
        centroids_stacked_data = filter_edgy_centroids(centroids_stacked_data, stacked)
    centroids_stacked = np.array([x[2] for x in centroids_stacked_data])

    df_detection = pd.DataFrame({'px': np.array(centroids_stacked)[:, 1],
                               'py': np.array(centroids_stacked)[:, 0],
                       'area (pixels)':[x[1] for x in centroids_stacked_data],
                       'flux (noise-normed)': [x[0] for x in centroids_stacked_data]})
    df_detection.to_csv(data_dir / ('STACKED_CENTROIDS_DATA'+'.csv'))
    
    logger.info(f'saving {centroids_stacked.shape[0]} centroid pixel coordinates')
    # plate solve
    flag_found_IDs = False
    df_identification = None
    solution = {'ra':None, 'dec':None, 'roll':None, 'FOV':None, 'platescale/arcsec':None}
    #if options['database'] and options['do_tetra_platesolve']:
    if 1:
        #t3 = database_cache.open_database(options['database'])
        #t3 = tetra3.Tetra3(load_database=options['database']) #tyc_dbase_test3 #hip_database938
        #solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True)
        #solution = t3.solve_from_centroids(centroids_stacked, size=stacked.shape, pattern_checking_stars=options['k'], return_matches=True, fov_estimate=5, fov_max_error=1, distortion = (-0.0020, -0.0005))
        solution = platesolve_triangle.platesolve(centroids_stacked, stacked.shape, options = options, output_dir = output_dir)
        print(solution)
        logger.info(str(solution))
        # TODO identify stars using catalogue
        # and save catalogue ids of each identified star (currently only around 20 are matched by the tetra software "for free"
        if not solution['ra'] is None:
            df_identification = pd.DataFrame({'px': np.array(solution['matched_centroids'])[:, 1],
                               'py': np.array(solution['matched_centroids'])[:, 0],
                               #'ID': solution['matched_catID'],
                               'RA': np.degrees(np.array(solution['matched_stars'])[:, 0]),
                               'DEC': np.degrees(np.array(solution['matched_stars'])[:, 1]),
                               'magV': np.array(solution['matched_stars'])[:, 5]})
            
            df_identification.to_csv(data_dir / ('STACKED_CENTROIDS_MATCHED_ID'+'.csv'))
            flag_found_IDs = True
        else:
            logger.error("ERROR: platesolve failed to identify location")
            print("ERROR: platesolve failed to identify location")
    else:
        #print('no database provided or platesolve not requested, so skipping platesolve')
        logger.info('no database provided or platesolve not requested, so skipping platesolve')

    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title(f'Largest {min(options["d"], len(centroids_stacked))} of {len(centroids_stacked)} stars found on stacked image')
    plt.imshow(stacked, cmap='gray_r', vmin=np.percentile(stacked, 50), vmax=np.percentile(stacked, 95))
    shift = 0 if options['centroid_gaussian_subtract'] else 0.5
    plt.scatter(centroids_stacked[:options["d"], 1]-shift, centroids_stacked[:options["d"], 0]-shift, marker='x') # subtract half pixel to align with image properly
    if flag_found_IDs:
        
        for ind, (index, row) in enumerate(df_identification.iterrows()):
            if ind >= options["d"]:
                break
            plt.gca().annotate((str(int(row['ID']) if isinstance(row['ID'], float) else row['ID']) if 'ID' in row else '') + f'\nMag={row["magV"]:.1f}', (row['px'], row['py']), color='r')
    plt.savefig(output_dir / ('CentroidsStackGood'+starttime+'.png'), bbox_inches="tight", dpi=600)
    if options['flag_display']:
        show_scanlines(stacked, fig, ax)
        #plt.legend()
        plt.show(block=True)
    plt.clf()
    #if flag_found_IDs:
    #    df_identification.drop('ID', axis=1) # ID is problematic as it is not a numeric datatype ... turns the array into an object which is bad for safety
    identification_arr = df_identification.to_numpy() if flag_found_IDs else None
    identification_arr_cols = df_identification.columns.values if flag_found_IDs else None

    results_dict = {
                         'MEE2024 version': _version(),
                         'platesolved' : flag_found_IDs,
                         'n_centroids' : centroids_stacked.shape[0],
                         'img_shape' : imgs_0.shape,
                         'RA' : solution['ra'],
                         'DEC' : solution['dec'],
                         'roll' : solution['roll'],
                         'platescale/arcsec' : solution['platescale/arcsec'],#solution['FOV'] / max(imgs_0.shape) if flag_found_IDs else None,
                         '#frames stacked':len(files),
                         'source_files' : str(files),
                         'starttime':starttime,
                         'remove saturated blob?':options['delete_saturated_blob'],
                         'blob saturation level':options['blob_saturation_level'],
                         'blob_radius_extra':options['blob_radius_extra'],
                         'centroid_gap_blob':options['centroid_gap_blob'],
                         'sensitive stacking mode?':options['centroid_gaussian_subtract'],
                         'use sensitive on stacked result?':options['sensitive_mode_stack'],
                         'background stubtraction mode':options['background_subtraction_mode'],
                    }
    if options['centroid_gaussian_subtract'] or options['sensitive_mode_stack']:
        results_dict.update({'sigma threshold detection':options['centroid_gaussian_thresh'], 'min_area':options['min_area'], 'sigma_subtract':options['sigma_subtract']})
    with open(data_dir / 'results.txt', 'w', encoding="utf-8") as fp:
            json.dump(results_dict, fp, sort_keys=False, indent=4)
    
    print('making archive', output_dir, Path(output_dir).parent)                                           
    shutil.make_archive(data_dir,
                    'zip',
                    Path(data_dir))
                    #'data')
    zipfilepath = Path(data_dir).parent / 'data.zip'
    shutil.move(zipfilepath, Path(output_dir).parent / f'centroid_data{starttime}.zip')
    
    logger.info('end time: ' + str(datetime.datetime.now()) + '\n')
    print('Done!')
