
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

def open_images(path):
    files = glob.glob(mypath + '\*.fit')
    images = []
    dark = None
    bias = None
    for file in files[:6]:
        if not 'MEE2024' in file and not 'Dark' in file and not 'Bias' in file:
            continue
        print(file)
        with fits.open(file) as hdul:
            print(hdul.info())
            print(hdul['PRIMARY'].data) # get the image data
            if 'Dark' in file:
                dark = hdul['PRIMARY'].data
            elif 'Bias' in file:
                bias = hdul['PRIMARY'].data
            else:
                images.append(hdul['PRIMARY'].data)
    return np.array(images), dark, bias

def regularise(img, dark, bias):
    #blur = gaussian_filter(img, sigma=100)
    #plt.hist(dark.flatten(), bins=256, log=True)
    #plt.show()
    fix = (img - dark) / bias
    #fix[img == 65535] = 0
    #fix[fix == 65535] = 0
    return fix
    #return heal_zeros(fix)

def heal_zeros(img):
    arr = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [ 1, 1, 1]])
    means =  signal.convolve2d(img, arr, boundary='symm', mode='same')
    img[img==0] = means[img==0]
    print(f'healed {np.sum(img==0)} zeros')
    return img

def rescale(img, p1=50, p2=95):
    assert(p2 > p1)
    lo = np.percentile(img, p1)
    hi = np.percentile(img, p2)
    img[img > hi] = hi
    img[img < lo] = lo
    img = (img-lo)/(hi-lo)
    return img

def try_clean(img, d2=4):
    result = np.copy(img)
    a = math.floor(d2**0.5)
    for i in range(-a, a+1):
        for j in range(-a, a+1):
            if i*i+j*j > d2:
                continue
            result = np.minimum(result, np.roll(img, (i, j), axis=(0, 1)))
    return result

def attempt_align(c1, c2, n = 10, m = 30, cutoff_dist=250):

    n = min(min(c1.shape[0], c2.shape[0]), n)
    c1a = c1[:n, :]
    c2a = c2[:n, :]
    a = np.ones((n, n, 2))
    def loss_fxn(b):
        d = c1a*a - np.swapaxes(c2a*a, 0, 1) - b
        norms = np.minimum(np.linalg.norm(d, axis=2), cutoff_dist)**2 # square norms of distances, capped
        return np.sum(np.min(norms, axis = 0)) / c1.shape[0]
    result = minimize(loss_fxn, (0, 0))
    print(result)

    def enumerate_matches(b, eps=2):
        d = np.reshape(c1, (c1.shape[0], 1, -1)) - np.swapaxes(np.reshape(c2, (c2.shape[0], 1, -1)), 0, 1) - b
        norms = np.linalg.norm(d, axis=2)
        matches1 = {}
        matches2 = {}
        for i in range(norms.shape[0]):
            for j in range(norms.shape[1]):
                if norms[i, j] < eps and not i in matches1 and not j in matches2:
                    matches1[i] = j
                    matches2[j] = i
        return matches1, matches2
    matches1, matches2 = enumerate_matches(result.x)

    vec1 = np.array([c1[i, :] for i in matches1 if i < m])
    vec2 = np.array([c2[matches1[i], :] for i in matches1 if i < m])
    print(vec1, vec2)
    def loss_fxn2(b):
        return np.linalg.norm(vec1 - vec2 - b) ** 2

    result2 = minimize(loss_fxn2, (0,0))
    print(result2)
    print(vec1.shape)
    return result.x, matches1, matches2, result2.x, (result2.fun/vec1.shape[0])**0.5


    
    
        

mypath = 'E:\stardata'
if __name__ == '__main__':
    imgs, dark, bias = open_images(mypath)
    imgs = [try_clean(regularise(img, dark, bias)) for img in imgs]
    img1, img2 = imgs[0], imgs[3]
    print('doing corr')
    
    '''
    corr = signal.correlate2d(img1[:500, :500], img2[:500, :500], boundary='fill', mode='same')
    plt.imshow(corr)
    plt.show()
    print(corr)
    '''
    '''
    z = corr
    nrows, ncols = z.shape
    x = np.linspace(-nrows//2, nrows - nrows//2, nrows)
    y = np.linspace(-nrows//2, nrows - nrows//2, nrows)
    x, y = np.meshgrid(x, y)

  
    
    # Set up plot
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    plt.show()
    '''
    '''
    corrb = signal.correlate2d(img1[:500, :500], bias[:500, :500], boundary='fill', mode='same')
    plt.imshow(corrb)
    plt.show()
    print(corrb)
    
    plt.imshow(corr-corrb)
    plt.show()
    '''
    
    print(np.min(img1), np.max(img1), np.median(img1))
    centroids1 = tetra3.get_centroids_from_image(img1, sigma=2, filtsize=31)
    centroids2 = tetra3.get_centroids_from_image(img2, sigma=2, filtsize=31)
    print(centroids1, centroids2)
    fig, (ax1, ax2) = plt.subplots(2)
    print(img1)
    #ax1.imshow(img1, cmap='gray_r', vmin = np.percentile(img1, 50), vmax=np.percentile(img1, 95))
    #ax2.imshow(img2, cmap='gray_r', vmin = np.percentile(img2, 50), vmax=np.percentile(img2, 95))
    ax1.imshow(rescale(img1), cmap='gray_r')
    ax1.scatter(centroids1[:10, 1], centroids1[:10, 0])
    ax2.imshow(rescale(img2), cmap='gray_r')
    ax2.scatter(centroids2[:10, 1], centroids2[:10, 0])
    plt.show()
    print(centroids1.shape, centroids2.shape)
    shift, matches1, matches2, shift2, fun2 = attempt_align(centroids1, centroids2)

    stacked = (img1 + np.roll(img2, shift.astype(int), axis=(0, 1)))/2 # shift by integer number of pixelscen

    plt.imshow(stacked, cmap='gray_r')
    plt.scatter(centroids1[:10, 1], centroids1[:10, 0], marker='x')
    plt.scatter((centroids2+shift)[:10, 1], (centroids2+shift)[:10, 0], marker='+')
    plt.show()

    deltas = np.array([centroids1[i] - centroids2[matches1[i]] for i in matches1 if i < 30])
    plt.scatter(deltas[:, 1], deltas[:, 0])
    plt.show()

    plt.scatter(deltas[:, 1], np.array([centroids1[i] for i in matches1 if i < 30])[:, 1])
    plt.show()
    
    t3 = tetra3.Tetra3(load_database='hip_database938', debug_folder=Path(__file__).parent)
    solution = t3.solve_from_centroids(centroids1, size=img1.shape)
    print(solution)
