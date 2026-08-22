"""
@author: Andrew Smith
Version 6 May 2024
"""

from astropy.io import fits
from pathlib import Path
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from mee2024.MEE2024util import (output_path, _version, environment_line, setup_logger,
                                 close_logger, date_string_to_float)
from mee2024 import calibration
from mee2024 import events
from mee2024 import framescan
from mee2024 import hotpixels
from mee2024 import ser
from mee2024 import star_labels
from mee2024.progress import NullProgress
import datetime
import pandas as pd
from collections import Counter
from skimage import measure
import cv2
from skimage.morphology import convex_hull_image
from skimage.transform import downscale_local_mean, resize
import skimage.data._fetchers # noqa: F401 -- fix py2exe bug (forces PyInstaller to bundle it)
import scipy
import os
import shutil
import json
from mee2024 import platesolve_triangle
import warnings

# return fit file image as np array
def open_image(file):
    # a SER frame reference (`capture.ser#42`) names one frame inside a container. Checked
    # first because such a path does not exist as a file and every reader below would fail
    # on it with a message about the wrong thing
    if ser.is_ser(file):
        return ser.read_frame(file)
    try:
        with fits.open(file) as hdul:
            if 'PRIMARY' in hdul:
                image = hdul['PRIMARY'].data
            else:
                image = hdul[0].data
    except Exception as fits_error:
        # not a FITS file, or not a readable file at all: try the ordinary image formats
        img_bgr = cv2.imread(str(file))
        if img_bgr is None:
            # cv2.imread signals every failure by returning None, and passing that to
            # cvtColor only yields an assertion about an empty matrix that names
            # neither the file nor the reason. Say which it was instead.
            if not os.path.exists(file):
                raise FileNotFoundError(f'no such image file: {file}') from fits_error
            raise ValueError(f'could not read {file} as FITS or as an ordinary image '
                             f'(the FITS reader said: {fits_error})') from fits_error
        # (Optional) Convert BGR to RGB if needed for compatibility with other libraries like Matplotlib
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]*0.299 + image[..., 1]*0.587 + image[..., 2]*0.114
    assert image.ndim == 2
    return image

def open_images(files):
    return [open_image(file) for file in files]


def read_bit_depth(file):
    """How many bits the camera's ADC actually produced, or None if it does not say.

    FITS stores everything in one of a few container sizes, so ``BITPIX`` describes the
    container and not the sensor: a 12-bit camera routinely writes ``BITPIX = 16``, and
    may scale its values up inside it. Cameras that know better also write ``BITDEPTH``,
    which is the number we want -- it is what makes a dark comparable to a light.

    A SER container states its depth outright, in ``PixelDepthPerPlane``.
    """
    if ser.is_ser(file):
        try:
            return int(ser.read_header(file)['depth'])
        except Exception:
            return None
    try:
        with fits.open(file) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
    except Exception:
        return None
    for key in ('BITDEPTH', 'BIT_DEPTH', 'BITSPERSAMPLE'):
        value = header.get(key)
        if value not in (None, ''):
            try:
                return int(str(value).strip())
            except ValueError:
                pass
    bitpix = header.get('BITPIX')
    # negative BITPIX is floating point, which has no ADC depth to report
    return int(bitpix) if isinstance(bitpix, int) and bitpix > 0 else None


def assert_matching_bit_depth(lights, darks=(), flats=()):
    """Refuse to calibrate frames of one bit depth with frames of another.

    A dark is subtracted from a light pixel for pixel, so the two have to be counting in
    the same units. Mixing depths -- 12-bit lights with darks a driver has stretched into
    16 bits, say -- subtracts numbers that are a fixed factor too large, which does not
    look like an error: the background goes negative or the stars go flat, and the run
    completes. Better to stop and say so.

    Frames that do not declare a depth are skipped rather than assumed.
    """
    by_depth = {}
    for label, files in (('light', lights), ('dark', darks), ('flat', flats)):
        for path in files or []:
            depth = read_bit_depth(path)
            if depth is not None:
                by_depth.setdefault(depth, []).append((label, path))
    if len(by_depth) <= 1:
        return sorted(by_depth)[0] if by_depth else None
    summary = '; '.join(
        f'{depth}-bit: ' + ', '.join(sorted({label for label, _ in entries}))
        + f' (e.g. {Path(entries[0][1]).name})'
        for depth, entries in sorted(by_depth.items()))
    raise ValueError(
        f'the frames do not share a bit depth -- {summary}. A dark or flat is combined '
        f'with a light pixel for pixel, so they must count in the same units. Re-export '
        f'the calibration frames at the same depth as the lights, or leave them out.')


#: Hot pixels from a master dark. Lives in mee2024.hotpixels alongside the dark-free
#: search; re-exported here because this is where callers have always found it.
hot_pixel_mask = hotpixels.dark_mask

#: How much of a dark-subtracted stack has to be below zero before it is worth a warning.
#: Noise alone puts a handful of pixels under -- on 26 megapixels the minimum of a
#: 4 ADU-sigma distribution sits five or six sigma low by construction, and dividing by a
#: vignetted flat deepens it -- so the old "any pedestal at all" test fired on every
#: correctly calibrated frame. A genuinely mismatched dark is not subtle: it puts a large
#: share of the background under, which 1% detects and noise does not reach.
NEGATIVE_FRACTION_WARN = 0.01


def read_observation_date(file):
    """The DATE-OBS calendar date from a FITS header, as 'YYYY-MM-DD', or None.

    Recorded alongside the stage-1 results so that stage 2 can report how close a blind
    date guess came. Non-FITS inputs and headers without a date simply return None.

    For a SER frame the date comes from the container and its sidecar: the per-frame
    timestamp trailer if the capture recorded one, else the header's UTC stamp, else the
    sidecar's start time. All three are UTC, which is what this wants.
    """
    if ser.is_ser(file):
        try:
            container, index = ser.parse_ref(file)
            when = ser.open_ser(container).frame_time(index or 0)
            return when.date().isoformat() if when else None
        except Exception:
            return None
    try:
        with fits.open(file) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
        for key in ('DATE-OBS', 'DATE_OBS', 'DATE'):
            value = header.get(key)
            if value:
                return str(value).strip().replace('/', '-')[:10]
    except Exception:
        pass
    return None

def read_pointing(file):
    """Where the telescope thought it was pointing, in degrees, or None.

    Capture software writes this: RA/DEC as degrees, or OBJCTRA/OBJCTDEC as
    sexagesimal hours and degrees. It is the mount's own claim, not a measurement,
    which is exactly what makes it worth comparing against a solved position.
    """
    def sexagesimal(text, unit):
        parts = [float(p) for p in str(text).replace(':', ' ').split()]
        value = sum(p / 60 ** i for i, p in enumerate(parts))
        return value * (15.0 if unit == 'hours' else 1.0)

    if ser.is_ser(file):
        # the container has no pointing; the sidecar records the mount's claim, e.g.
        #   ASI Mount=RA=09:13:06.0,Dec=+13:56:48 (JNOW)
        try:
            container, index = ser.parse_ref(file)
            header = ser.open_ser(container).fits_header(index or 0)
            ra, dec = header.get('OBJCTRA'), header.get('OBJCTDEC')
            if ra is None or dec is None:
                return None
            ra, dec = sexagesimal(ra, 'hours'), sexagesimal(dec, 'degrees')
            return (float(ra % 360), float(dec)) if -90 <= dec <= 90 else None
        except Exception:
            return None
    try:
        with fits.open(file) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
        for ra_key, dec_key, unit in (('RA', 'DEC', 'degrees'),
                                      ('OBJCTRA', 'OBJCTDEC', 'hours'),
                                      ('CRVAL1', 'CRVAL2', 'degrees')):
            ra, dec = header.get(ra_key), header.get(dec_key)
            if ra is None or dec is None:
                continue
            try:
                ra = float(ra) if unit == 'degrees' else sexagesimal(ra, 'hours')
                dec = float(dec) if unit == 'degrees' else sexagesimal(dec, 'degrees')
            except (TypeError, ValueError):
                ra, dec = sexagesimal(ra, 'hours'), sexagesimal(dec, 'degrees')
            if -90 <= dec <= 90:
                return float(ra % 360), float(dec)
    except Exception:
        pass
    return None


def pointing_comment(header_pointing, solved_ra, solved_dec):
    """(separation in degrees, plain-language verdict) against the header's claim.

    The mount's own pointing is an independent check on the whole chain: a solve that
    lands where the telescope was aimed says the alignment and configuration were
    right, and one that lands elsewhere says something upstream is wrong -- which is
    worth saying at the moment of the solve rather than leaving to be noticed later.
    """
    if not header_pointing or solved_ra is None or solved_dec is None:
        return None, None
    ra, dec = header_pointing
    dra = ((solved_ra - ra + 180) % 360 - 180) * np.cos(np.radians(dec))
    separation = float(np.hypot(dra, solved_dec - dec))
    if separation < 0.5:
        verdict = 'agrees with the telescope pointing -- alignment and setup look good'
    elif separation < 5:
        verdict = ('close to the telescope pointing -- fine for the analysis, though '
                   'the mount alignment could be better')
    elif separation < 30:
        verdict = ('well away from the telescope pointing -- check the mount alignment '
                   'and that these frames are the field you meant')
    else:
        verdict = ('nowhere near the telescope pointing -- the header, the frames or '
                   'the setup disagree; something is wrong upstream')
    return separation, verdict


def _short(value, limit=68):
    """A FITS card holds 80 characters for keyword, value and comment together."""
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + '~'


def _stack_header(bit_depth=None, n_frames=None, source_header=None,
                  dark_info=None, flat_info=None, hot_pixels=None):
    """What a stacked frame needs to say about itself.

    The stack used to carry ``BITDEPTH``, ``NCOMBINE`` and a version, which describes the
    *stacking* and nothing about the exposure it came from. Anyone reading a STACKED file
    a month later -- or matching it against a calibration library -- needs the exposure,
    gain and epoch, and needs to know whether a dark and a flat were actually applied.
    Absence of a keyword is not evidence: a stack with no ``CALDARK`` could equally be
    uncalibrated or written by a version that never recorded it, so both are written
    always, with an explicit ``none``.
    """
    header = fits.Header()
    if bit_depth:
        header['BITDEPTH'] = (int(bit_depth), 'ADC bits of the source frames')
    if n_frames:
        header['NCOMBINE'] = (int(n_frames), 'light frames stacked')
    header['MEE2024'] = (_version(), 'version that wrote this stack')

    source = source_header or {}
    for key, comment in (('EXPTIME', 'exposure of one light frame, seconds'),
                         ('GAIN', 'camera gain of the light frames'),
                         ('EGAIN', 'e- per ADU'),
                         ('OFFSET', 'camera offset'),
                         ('CCD-TEMP', 'sensor temperature, C'),
                         ('SET-TEMP', 'requested sensor temperature, C'),
                         ('CAMID', 'camera identifier'),
                         ('INSTRUME', 'camera'),
                         ('TELESCOP', 'optical train'),
                         ('FOCALLEN', 'focal length, mm'),
                         ('XPIXSZ', 'pixel size, micron'),
                         ('XBINNING', 'binning'),
                         ('FILTER', 'filter')):
        if source.get(key) is not None:
            header[key] = (source[key], comment)
    if source.get('DATE-OBS'):
        # the first frame, not the middle: the epoch a stack should carry is a v1.4.0
        # question (it moves the fit), so this records what is true today rather than
        # quietly changing it
        header['DATE-OBS'] = (source['DATE-OBS'], 'start of the first light frame')

    header['CALDARK'] = (_short((dark_info or {}).get('source') or 'none'),
                         'master dark applied')
    header['CALFLAT'] = (_short((flat_info or {}).get('source') or 'none'),
                         'master flat applied')
    if hot_pixels is not None:
        header['CALHOT'] = (int(hot_pixels), 'hot pixels excluded from the stack')
    return header


def write_stacked_fits(path, stacked, bit_depth=None, n_frames=None,
                       source_header=None, dark_info=None, flat_info=None,
                       hot_pixels=None):
    """Write the stack in the input frames' own ADU, not stretched to fill the container.

    This used to be

        stacked16 = ((stacked - min) / (max - min) * 65535).astype(np.uint16)

    which rescaled every output to fill 16 bits regardless of what went in: 12-bit data
    came back 16-bit, the black point moved to wherever the darkest pixel happened to be,
    and the numbers no longer meant ADU at all -- a display stretch saved as science data.
    The gain is part of the measurement, so it is kept.

    Dark subtraction can leave the background below zero, which an unsigned image cannot
    hold. Rather than clip it away (a mismatched dark would silently flatten most of the
    frame to zero) a pedestal is added and recorded in ``PEDESTAL``, so subtracting that
    one number recovers the calibrated ADU exactly.

    Returns the pedestal used; how many pixels sat above the 16-bit container -- which is
    what drove the float32 fallback, not a loss; and what fraction of the frame was below
    zero, which is how the caller decides whether the pedestal is worth a warning.

    That over-container count used to be taken in the *other* branch, where
    ``shifted.max() <= 65535`` holds by construction, so it was always zero and the warning
    it fed could never fire. Reporting it from the branch it describes makes it say
    something true: the values were kept, in a wider dtype.
    """
    values = np.nan_to_num(np.asarray(stacked, dtype=np.float64), nan=0.0,
                           posinf=0.0, neginf=0.0)
    low = float(values.min())
    pedestal = int(np.ceil(-low)) if low < 0 else 0
    # what fraction of the frame is actually below zero, which is a different question from
    # whether *any* pixel is. On 26 million pixels with a few ADU of read noise the minimum
    # is several sigma negative by construction, so "any" is always true on a correctly
    # calibrated frame; a mismatched dark pushes the whole background under instead.
    negative_fraction = float(np.count_nonzero(values < 0)) / max(values.size, 1)
    shifted = np.rint(values + pedestal)

    header = _stack_header(bit_depth=bit_depth, n_frames=n_frames,
                           source_header=source_header, dark_info=dark_info,
                           flat_info=flat_info, hot_pixels=hot_pixels)
    header['COMMENT'] = 'pixel values are the input frames ADU, not rescaled'

    over = 0
    if shifted.max() > 65535:
        # too wide for the 16-bit container: keep the values rather than the dtype
        over = int(np.sum(shifted > 65535))
        data = shifted.astype(np.float32)
        header['COMMENT'] = 'stored as float32: the values exceed a 16-bit container'
    else:
        data = shifted.astype(np.uint16)
    if pedestal:
        header['PEDESTAL'] = (pedestal, 'added to keep values non-negative; subtract it')
    fits.writeto(path, data, header=header, overwrite=True)
    return pedestal, over, negative_fraction


def write_float_stack(path, stacked, **provenance):
    """The same stack as float32, with no pedestal and no container to overflow.

    This used to be written only when the classic UI's ``float_32_fits`` box was ticked --
    off by default, and absent from the app window and the CLI -- and with **no header at
    all**: no version, no exposure, not even the frame count, so the one output holding the
    unaltered calibrated values was also the one that could not say where it came from.

    It is now always written, because it is cheap (one file, same array already in memory)
    and because it is the only form that survives dark subtraction without arithmetic. The
    16-bit ``STACKED`` file adds a pedestal to keep values non-negative and rounds to
    integers; both are recoverable, but only if the reader knows to do it. This one needs
    no correction, which makes it the right input for anything measuring flux.
    """
    header = _stack_header(**provenance)
    header['CALPED'] = (0, 'no pedestal: float32 holds negatives directly')
    header['COMMENT'] = 'calibrated ADU as measured: no pedestal, no rounding, not rescaled'
    fits.writeto(path, np.asarray(stacked, dtype=np.float32), header=header,
                 overwrite=True)
    return path


def _stopped_early(message, files):
    """A centroid-phase failure, with what the early exit saved.

    Worth saying: someone watching a hundred-frame run die two frames in should know that
    is deliberate and that the remaining frames were not silently skipped for some other
    reason.
    """
    return Exception(
        f'{message}\n\nStopped after two frames rather than centroiding the other '
        f'{max(0, len(files) - 2)} first: every frame is aligned against frame 0, so a set '
        f'whose first two frames do not match cannot be stacked at all.')


def _align_frames(centroids, files, options):
    """Fit every frame's offset against the first. Extracted so it can be run twice.

    The dark-free hot-pixel search needs the shifts, and cleaning the centroid lists
    afterwards invalidates the alignment those shifts came from -- so this runs, the lists
    are filtered, and it runs again. It is cheap: an optimisation over about thirty points.

    Returns the shifts, per-frame rms and deltas, a tally of which stars were used, and the
    FRAME_ALIGNED payloads *without emitting them*, so a discarded first pass does not
    report frames the run did not end up using.
    """
    shifts = [(0, 0)]
    rms_errors = []
    deltas = []
    aligned = []
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
        aligned.append({'frame': i, 'shift': [float(shift2[0]), float(shift2[1])],
                        'rms': float(fun2), 'n_matched': len(matches1)})
        used_stars_stacking.update(matches1.keys())
        print(matches1)
    return shifts, rms_errors, deltas, used_stars_stacking, aligned


def save_calibration_stacks(output_dir, starttime, darkfiles, dark, flatfiles, flat):
    """Keep the combined dark and flat beside the results, for reuse.

    A master dark or flat is worth more than the frames it came from: it can calibrate a
    later session without hauling every original around. It is only written when there was
    something to combine, though -- a single input copied back out under a new name is
    clutter that looks like a product, and re-deriving it costs nothing.

    The header used to carry ``NCOMBINE``, ``COMBTYPE`` and a version number, which says a
    master exists and nothing about whether it may be used. Reuse is the *only* reason to
    save one, and reuse needs the exposure, gain, offset, temperature and camera -- so the
    provenance keys are copied from the frames it was built from. A master that cannot
    identify itself is a file someone will apply to the wrong lights.

    Returns the paths written, so a caller can report them.
    """
    written = []
    for label, frames, stack in (('DARK', darkfiles, dark), ('FLAT', flatfiles, flat)):
        if len(frames or []) < 2:
            continue
        path = Path(output_dir) / f'{label}_STACK{starttime}.fit'
        header = calibration._master_header(
            label.lower(), calibration.read_cal_header(frames[0]),
            {'n_frames': len(frames), 'combine': 'mean'}, data=stack,
            extra={'CALSRC': (str(Path(frames[0]).parent)[:68], 'folder it was built from')})
        fits.writeto(path, np.asarray(stack, dtype=np.float32), header=header,
                     overwrite=True)
        events.log(f'saved the combined {label.lower()} of {len(frames)} frames '
                   f'as {path.name}')
        written.append(path)
    return written


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
    return ret
            
def simple_get_centroids(image):
    """
    Simplified, faithful Tetra-style centroid extraction using fixed defaults.
    Returns centroids as (y, x) pixel coordinates.
    """

    # ---- 1. Ensure float32 mono image ----
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]*0.299 + image[..., 1]*0.587 + image[..., 2]*0.114
    assert image.ndim == 2

    # ---- 2. Background subtraction (local mean) ----
    image = image - scipy.ndimage.uniform_filter(image, size=25)

    # ---- 3. Threshold (sigma * global root-square noise) ----
    img_std = np.sqrt(np.mean(image**2))
    image_th = 2 * img_std

    # ---- 4. Binary mask + opening ----
    bin_mask = image > image_th
    bin_mask = scipy.ndimage.binary_opening(bin_mask)

    # ---- 5. Label regions ----
    labels, num_labels = scipy.ndimage.label(bin_mask)
    if num_labels == 0:
        return np.empty((0, 2))

    # ---- 6. Extract centroids (unbiased) ----
    centroids = []
    total_weights = []
    slices = scipy.ndimage.find_objects(labels)
    
    for i, slc in enumerate(slices, start=1):
        region_mask = (labels[slc] == i)
        area = np.count_nonzero(region_mask)
        if area < 5 or area > 100:
            continue

        region = image[slc]
        weights = region[region_mask]
        m0 = weights.sum()
        if m0 <= 0:
            continue

        yy, xx = np.nonzero(region_mask)

        # Pixel centers
        yy = yy + slc[0].start + 0.5
        xx = xx + slc[1].start + 0.5

        cy = np.sum(weights * yy) / m0
        cx = np.sum(weights * xx) / m0

        centroids.append((cy, cx))
        total_weights.append(m0)

    s = np.argsort(total_weights)[::-1]
    return np.asarray(centroids)[s]

def windowed_centroid(sub, y0, x0, sigma, R=8, iters=12, tol=1e-5):
    """Flux-weighted centroid under a fixed Gaussian window, iterated to convergence.

    Returns (y, x), or None if the star is too near the edge or the window holds no flux.

    Why this exists: the detection footprint is defined by a signal-to-noise threshold, so
    its *size* -- and the weighting inside it -- scale with the star's brightness. Where the
    PSF is asymmetric, bright and faint stars are then measured over different parts of the
    same profile and disagree with each other. Measured on the Leon zenith set that was
    172 and 299 mas beyond r = 2500 px, in twelve fields out of twelve, and it made the
    fitted cubic a function of the magnitude limit at the 4.5% level
    (docs/LEON_2026-08-11.md section 18.3).

    A fixed window does not assume the star is round and does not remove the asymmetry: the
    centroid still moves with the flare. What it removes is the *brightness dependence*, so
    the displacement becomes a smooth function of field position -- which is precisely what
    the distortion polynomial absorbs. The aberration stops leaking into the answer.

    Not SExtractor's XWIN: no factor-2 correction, so this converges to a fixed point that
    is brightness-independent rather than to the unbiased photocentre. That is the property
    that matters here; any residual offset is smooth in field position. Use the XWIN form if
    absolute positions ever matter.
    """
    H, W = sub.shape
    i0, j0 = int(round(y0)), int(round(x0))
    if i0 < R + 1 or j0 < R + 1 or i0 >= H - R - 1 or j0 >= W - R - 1:
        return None
    s = np.clip(sub[i0-R:i0+R+1, j0-R:j0+R+1], 0, None)
    yy, xx = np.mgrid[-R:R+1, -R:R+1]
    cx, cy = x0 - j0, y0 - i0
    for _ in range(iters):
        w = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)) * s
        total = w.sum()
        if total <= 0:
            return None
        nx, ny = (w * xx).sum() / total, (w * yy).sum() / total
        converged = abs(nx - cx) < tol and abs(ny - cy) < tol
        cx, cy = nx, ny
        if converged:
            break
    return i0 + cy, j0 + cx


def get_centroids_blur(img_mask2, ksize=17, r_max=10, options={}, gauss=False, debug_display=False):
    t_start = time.time()
    img, mask, mask2 = img_mask2
    if not options['centroid_gaussian_subtract']:
        centroids = simple_get_centroids(img)
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
    if options.get('centroid_refine_window'):
        # Same stars, better positions: only the estimator changes. See windowed_centroid.
        sigma = float(options.get('centroid_window_sigma', 2.0))
        refined, dropped = [], 0
        for f, a, c in sorted_c:
            r = windowed_centroid(sub, c[0], c[1], sigma)
            if r is None:
                dropped += 1
                continue
            refined.append((f, a, r))
        print(f"n centroids window-refined {len(refined)} (sigma {sigma} px, {dropped} dropped)")
        sorted_c = refined
    print("--- %s seconds for centroid finding (all)---" % (time.time() - t_start))
    print('found:', sorted_c)
    return sorted_c

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
    fig.canvas.mpl_connect('motion_notify_event', mouse_move)

def add_img_to_stack(data, output_array=None, count_array=None, valid=None):
    img, shift = data # unpack tuple
    shift = (round(shift[0]), round(shift[1]))
    # `valid` excludes pixels that carry no measurement -- hot pixels, which are fixed to
    # the detector. Dropping them from the count as well as the sum is what makes them
    # disappear instead of being averaged in at reduced strength: each sky position simply
    # loses whichever frames had a bad pixel under it, and keeps the rest.
    contributes = np.ones(count_array.shape, dtype=int) if valid is None else valid.astype(int)
    if valid is not None:
        img = img * valid
    output_array += roll_fillzero(img, shift)
    count_array += roll_fillzero(contributes, shift)

def open_img_and_preprocess(file, options = {}, dark=0, flat=1, hot=None):
    img = open_image(file)
    desatblob_img, mask, mask2 = remove_saturated_blob(img, sat_val=None, radius = options['blob_radius_extra'], radius2 = options['blob_radius_extra']+options['centroid_gap_blob'], blob_saturation=options['blob_saturation_level']/100, perform=options['delete_saturated_blob'])
    reg_img = (desatblob_img - dark) / flat
    if hot is not None and np.any(hot):
        # flatten them into the background so they cannot be detected as stars; the stack
        # excludes them outright via `valid`, but centroid finding runs on this array
        reg_img = np.where(hot, np.median(reg_img[::8, ::8]), reg_img)
    return reg_img, mask, mask2

def open_img_and_find_centroids(file, options = {}, dark=0, flat=1, hot=None):
    reg_img, mask, mask2 = open_img_and_preprocess(file, options, dark, flat, hot)
    # Never refine per frame: these centroids feed the alignment, which measures a
    # difference (so a static centroid bias cancels) and then rounds the shift to whole
    # pixels. Refining here would cost time and change nothing.
    centroids = get_centroids_blur((reg_img, mask, mask2),
                                   options=dict(options, **{'centroid_refine_window': False}))
    centroids_filtered = filter_bad_centroids(centroids, mask2, reg_img.shape)
    return centroids_filtered

def open_img_and_add_to_stack(data, output_array=None, count_array=None, options = {}, dark=0, flat=1, hot=None):
    file, shift = data # unpack tuple
    reg_img, _, _ = open_img_and_preprocess(file, options, dark, flat, hot)
    add_img_to_stack((reg_img, shift), output_array, count_array,
                     valid=None if hot is None else ~hot)
    
def select_frames(files, options):
    """Apply the frame range, and say what the sequence looks like before stacking it.

    Three things happen here, all of them reporting rather than deciding:

    * an explicit ``frame_range`` is applied -- the trim is a run parameter, recorded in the
      results, rather than a second copy of the data on disk;
    * the sequence is measured and a usable range *suggested*, so a capture that opens on
      the uneclipsed Sun or ends in blank frames says so before it is stacked into an
      average with them;
    * every frame's level is checked against the exposure its header claims.

    None of it changes a frame. The suggestion is a line in the log the user can act on by
    setting ``frame_range``; the exposure check names files to look at. That is deliberate:
    the alternative is silently processing data nobody has inspected, and on eclipse data
    there is too little of it to have any of it quietly dropped or relabelled.
    """
    files = [str(f) for f in files]
    if len(files) < 2:
        return files

    scanned = None
    if options.get('scan_frames', True) or options.get('check_exposures', True):
        try:
            scanned = framescan.scan(files)
        except Exception as exc:      # never fail a run over a measurement
            events.log(f'could not measure the frames: {exc}', level='warning')

    if scanned and options.get('check_exposures', True):
        try:
            for message in framescan.check_exposures(files, scanned):
                events.log(message, level='warning')
        except Exception as exc:
            events.log(f'could not check the exposures: {exc}', level='warning')

    if scanned and options.get('scan_frames', True):
        try:
            start, stop, info = framescan.suggest(scanned)
            summary = framescan.describe(start, stop, info)
            if start is not None and (info['dropped_leading'] or info['dropped_trailing']):
                events.log(summary + f'. Set frame_range to {start}-{stop} to use only '
                                     f'those; nothing has been dropped automatically.',
                           level='warning')
            else:
                events.log(summary)
        except Exception as exc:
            events.log(f'could not suggest a frame range: {exc}', level='warning')

    chosen = options.get('frame_range') or ''
    if not chosen:
        return files
    try:
        start, stop = framescan.parse_range(chosen, len(files))
    except ValueError as exc:
        raise ValueError(f'{exc}. The sequence has {len(files)} frames.') from exc
    kept = framescan.apply_range(files, start, stop)
    events.log(f'using frames {start}-{stop} of {len(files)} '
               f'({len(kept)} frame(s)), as frame_range asked')
    if not kept:
        raise ValueError(f'frame_range {chosen!r} selects no frames from a '
                         f'{len(files)}-frame sequence')
    return kept


def do_stack(files, darkfiles, flatfiles, options, progress=None):
    """Stage 1: stack the light frames and find + platesolve centroids on the result.

    progress: a mee2024.progress.ProgressReporter. Defaults to NullProgress, so the
    pipeline runs headless unless a caller explicitly asks for progress reporting.

    A thin wrapper around :func:`_do_stack`, so the run's log file is released the moment
    the run ends rather than when the program exits -- including when the run raises. The
    handler is attached to a process-global named logger, so nothing else would ever let
    go of it, and on Windows that left the user unable to move or delete their own log.
    """
    if progress is None:
        progress = NullProgress()
    files = select_frames(files, options)
    starttime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f'CENTROID_OUTPUT{starttime}'
    output_dir = Path(output_path(output_name, options))
    logpath = output_dir / f'LOG{starttime}.txt'
    data_dir = Path(output_dir) / 'data'
    # makedirs, not mkdir: the chosen output folder may not exist yet, and failing
    # several directory levels deep gives the user a baffling error
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f'logpath {logpath}')
    logger = setup_logger('logger'+starttime, logpath)
    logger.info(f'MEE2024 {_version()}')
    logger.info(f'environment: {environment_line()}')
    try:
        return _do_stack(files, darkfiles, flatfiles, options, progress,
                         starttime, output_dir, data_dir, logger)
    finally:
        close_logger(logger)


def _do_stack(files, darkfiles, flatfiles, options, progress,
              starttime, output_dir, data_dir, logger):
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

    

    # a dark is subtracted pixel for pixel, so it has to be counting in the same units
    bit_depth = assert_matching_bit_depth(files, darkfiles, flatfiles)
    if bit_depth:
        logger.info(f'bit depth: {bit_depth}')

    imgs_0 = open_image(files[0])
    _, masks_0, masks2_0 = remove_saturated_blob(imgs_0, sat_val=None, radius = options['blob_radius_extra'], radius2 = options['blob_radius_extra']+options['centroid_gap_blob'], blob_saturation=options['blob_saturation_level']/100, perform=options['delete_saturated_blob'])
    # Streamed, not stacked into one array. `np.mean(np.array(open_images(files)))` holds
    # every frame at once: fifty frames of the 26 MP ASI2600MM is 5.2 GB as float32, and
    # 10.4 GB while the list and the array both exist, so a full dark tier could not be
    # combined at all on an ordinary machine. It also accepts a master straight from the
    # calibration library, which is the usual case now.
    dark, dark_info = calibration.load_or_combine(darkfiles, progress=None)
    # captured before the None is replaced by a zero array, so the header can say
    # "none" rather than describing a dark that was never there
    dark_applied = dark_info if dark is not None else None
    if dark is None:
        dark = np.zeros(imgs_0.shape, dtype=imgs_0.dtype)
    else:
        logger.info(f'dark: {dark_info["source"]} ({dark_info.get("combine", "")})')
        events.log(f'dark from {dark_info["source"]}'
                   + (f' ({dark_info["combine"]})' if dark_info.get('combine') else ''))
    flat, flat_info = calibration.load_or_combine(flatfiles, progress=None)
    flat_applied = flat_info if flat is not None else None
    if flat is None:
        flat = np.ones(imgs_0.shape, dtype=float)
    elif flat_info.get('normalised'):
        # a library master flat is already divided by its own level; doing it again divides
        # by a number near 1, which does almost nothing and looks like it worked
        logger.info(f'flat: {flat_info["source"]}, already normalised')
        events.log(f'flat from {flat_info["source"]} (already normalised)')
    else:
        # A flat corrects *relative* sensitivity, so it has to be about 1. Dividing by raw
        # flat ADU (thousands) scaled the whole frame away; the old output stretch hid it,
        # and now that the stack keeps its ADU it would not.
        flat_level = float(np.median(flat[::8, ::8]))
        if flat_level > 0:
            # the same question the library build asks, on the path that skipped it: a
            # hand-picked flat set was normalised and applied with nothing checking how
            # it was exposed
            try:
                fill_warning, _, _ = calibration.flat_fill_check(
                    flat_level, calibration.read_cal_header(flatfiles[0]))
            except Exception:
                fill_warning = None
            if fill_warning:
                logger.warning(fill_warning)
                events.log(fill_warning, level='warning')
            flat = flat / flat_level
            logger.info(f'flat normalised by its median level {flat_level:.1f}')
            events.log(f'flat from {flat_info["source"]}, normalised by {flat_level:.1f}')

    print('image size:'+str(imgs_0.shape))
    logger.info('image size:'+str(imgs_0.shape))

    # Hot pixels survive dark subtraction -- they clip, and clipping is not linear -- and
    # then smear across the stack as fake stars, because they are fixed to the detector
    # while the field is dithered. Find them once from the master dark and drop them.
    hot = hot_pixel_mask(dark, options['hot_pixel_sigmas'],
                         min_adu=options.get('hot_pixel_min_adu',
                                             hotpixels.MIN_DARK_ADU)) if darkfiles else None
    if hot is not None and np.any(hot):
        n_hot = int(np.sum(hot))
        message = (f'{n_hot} hot pixel(s) found in the master dark '
                   f'({100 * n_hot / hot.size:.4f}% of the frame); excluded from the '
                   f'stack rather than subtracted')
        print(message)
        logger.info(message)
        events.log(message)

    if options['save_dark_flat']:
        save_calibration_stacks(output_dir, starttime, darkfiles, dark, flatfiles, flat)
    t_start_c = time.time()
    # Fail fast. Centroid finding is the expensive part -- seconds per frame on a full
    # sensor -- and if frames 0 and 1 cannot be matched then nothing downstream can work:
    # every later frame is aligned against frame 0, so the run was going to die anyway,
    # just after paying for every frame first. Doing the first pair up front costs nothing
    # on good data (each frame is still centroided exactly once) and turns a doomed
    # hundred-frame run into a two-frame one.
    if len(files) > 1:
        pair = progress.loop(files[:2], open_img_and_find_centroids,
                             message='Checking the first two frames...',
                             dark=dark, flat=flat, options=options, hot=hot)
        first_two = [np.array([x[2] for x in y]) for y in pair]
        for index, found in enumerate(first_two):
            if not len(found):
                raise _stopped_early(
                    f'No star centroids were found on frame {index} '
                    f'({Path(files[index]).name}). Nothing can be stacked or solved from '
                    f'it. Check the frame is not blank or wildly out of focus, and that '
                    f'the centroid detection threshold suits this data.', files)
        try:
            attempt_align(first_two[0], first_two[1], options, framenum=1)
        except Exception as exc:
            # attempt_align already says what went wrong; what it cannot know is that this
            # was the early probe, so add what was skipped and why the run stopped here
            raise _stopped_early(str(exc), files) from exc
        rest = (progress.loop(files[2:], open_img_and_find_centroids,
                              message='Finding all centroids...',
                              dark=dark, flat=flat, options=options, hot=hot)
                if len(files) > 2 else [])
        centroids_data = list(pair) + list(rest)
    else:
        centroids_data = progress.loop(files, open_img_and_find_centroids, message='Finding all centroids...', dark = dark, flat=flat, options=options, hot=hot)
    print("--- %s seconds for centroid finding---" % (time.time() - t_start_c))
    centroids = [np.array([x[2] for x in y]) for y in centroids_data]

    shifts, rms_errors, deltas, used_stars_stacking, aligned = _align_frames(
        centroids, files, options)

    # Without darks, hot pixels can now be found from the dither itself: a star is fixed to
    # the sky, a hot pixel to the detector. That needs the shifts, which is why it happens
    # here rather than before centroid finding -- and why the centroid lists are *filtered*
    # afterwards rather than detected again, which would cost more than everything else in
    # this function put together. The alignment above is good enough to derive shifts from:
    # centroids rank by integrated flux and a hot pixel has almost no area, so on the
    # measured example the one hot centroid of 388 ranked 106th, far outside the brightest
    # 30 the aligner uses. That is circumstantial, though -- a hot cluster or a sparse field
    # would put one in reach -- so the lists are cleaned and the alignment redone.
    if hot is None and options['hot_pixel_dark_free']:
        hot, info = hotpixels.persistence_mask(files, shifts, blob_mask=masks2_0)
        if info['declined']:
            message = f'no dark-free hot-pixel search: {info["declined"]}'
            print(message)
            logger.info(message)
        else:
            message = (f'{info["n_flagged"]} hot pixel(s) identified from the dither '
                       f'({info["dither_px"]:.1f} px) out of {info["n_candidates"]} bright '
                       f'candidates, without a dark frame')
            print(message)
            logger.info(message)
            events.log(message)
        if hot is not None and hot.any():
            spoiled = hotpixels.spoiled_by(hot)
            dropped_total = 0
            for i, data in enumerate(centroids_data):
                centroids_data[i], dropped = hotpixels.drop_masked_centroids(data, spoiled)
                dropped_total += dropped
            if dropped_total:
                centroids = [np.array([x[2] for x in y]) for y in centroids_data]
                message = (f'{dropped_total} centroid(s) dropped as hot pixels; '
                           f'realigning without them')
                print(message)
                logger.info(message)
                events.log(message)
                shifts, rms_errors, deltas, used_stars_stacking, aligned = _align_frames(
                    centroids, files, options)
    # emitted once, from whichever alignment turned out to be the final one
    for event in aligned:
        events.emit(events.FRAME_ALIGNED, **event)
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
        # rms_errors/deltas hold one entry per *aligned* frame, so frame i is at i-1
        if deltas[i-1] is None:
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
    stack_array = np.zeros(imgs_0.shape)
    count_array = np.zeros(imgs_0.shape, dtype=int)
    progress.loop(list(zip(files, shifts)), open_img_and_add_to_stack, message='Stacking images...',
                  output_array=stack_array, count_array=count_array, options = options, dark=dark, flat=flat, hot=hot)
    # a pixel can have no contributions at all: the dither leaves the frame edges uncovered,
    # and a hot pixel is excluded everywhere it lands. 0/0 is nan, which then poisons the
    # centroid pass, so those pixels are left at zero instead
    stacked = np.divide(stack_array, count_array, out=np.zeros_like(stack_array),
                        where=count_array > 0)

    # the light frames' own provenance, so the stack can be matched to a calibration
    # tier or an epoch without the originals being present
    try:
        source_header = calibration.read_cal_header(files[0])
    except Exception:
        source_header = {}
    provenance = dict(bit_depth=bit_depth, n_frames=len(files),
                      source_header=source_header, dark_info=dark_applied,
                      flat_info=flat_applied,
                      hot_pixels=int(np.sum(hot)) if hot is not None else None)
    pedestal, over_16bit, negative_fraction = write_stacked_fits(
        output_dir / ('STACKED'+starttime+'.fit'), stacked, **provenance)
    if pedestal:
        # Recorded always -- subtracting it recovers the calibrated ADU exactly -- but only
        # *warned* about when the negativity is systematic. On 26 million pixels with a few
        # ADU of read noise the single lowest pixel is always some way below zero, so
        # warning whenever a pedestal exists fired on every correctly calibrated frame,
        # which is the fastest way to teach someone to ignore a warning. A mismatched dark
        # looks different: it puts a large share of the frame under, not one extreme pixel.
        message = (f'the calibrated stack dips {pedestal} ADU below zero '
                   f'({100 * negative_fraction:.3f}% of pixels), so the saved image '
                   f'carries a PEDESTAL of {pedestal}; subtract it to recover the '
                   f'calibrated ADU.')
        print(message)
        logger.info(message)
        if negative_fraction > NEGATIVE_FRACTION_WARN:
            events.log(message + (f' Over {100 * NEGATIVE_FRACTION_WARN:.0f}% of the frame '
                                  f'is negative, which usually means the darks do not '
                                  f'match the lights (different temperature or exposure).'),
                       level='warning')
        else:
            events.log(message)
    if over_16bit:
        events.log(f'{over_16bit} pixel(s) exceeded the 16-bit container, so the stack '
                   f'was saved as float32 rather than clipped')
    write_float_stack(output_dir / ('STACKED_FLOAT'+starttime+'.fit'), stacked,
                      **provenance)
    # find centroids on the stacked image
    centroids_stacked_data = get_centroids_blur((stacked, masks_0, masks2_0),
                        options=dict(options, **{'centroid_gaussian_subtract':options['centroid_gaussian_subtract'] or options['sensitive_mode_stack']}), # use sensitive mode if requested only for the stack
                        debug_display=False)
    centroids_stacked_data = filter_bad_centroids(centroids_stacked_data, masks2_0, imgs_0.shape) # use 0th mask here
    centroids_stacked_data = filter_very_edgy_centroids(centroids_stacked_data, stacked, f=options['img_edge_distance'])
    if options['remove_edgy_centroids']:
        centroids_stacked_data = filter_edgy_centroids(centroids_stacked_data, stacked)
    centroids_stacked = np.array([x[2] for x in centroids_stacked_data])
    # No detections at all means every later step gets an empty or 1-D array and
    # fails somewhere unrecognisable. Name the real problem here.
    if centroids_stacked.size == 0:
        raise ValueError(
            'no stars were found on the stacked image, so there is nothing to plate '
            'solve. Check that these are light frames of a star field, and that the '
            'detection settings suit them: a lower centroid threshold '
            '(centroid_gaussian_thresh) or a smaller minimum area (min_area) finds '
            'fainter stars, and "remove big bright object" may be masking the frame '
            'if the saturation level is set too low.')
    centroids_stacked = centroids_stacked.reshape(-1, 2)

    df_detection = pd.DataFrame({'px': np.array(centroids_stacked)[:, 1],
                               'py': np.array(centroids_stacked)[:, 0],
                       'area (pixels)':[x[1] for x in centroids_stacked_data],
                       'flux (noise-normed)': [x[0] for x in centroids_stacked_data]})
    df_detection.to_csv(data_dir / ('STACKED_CENTROIDS_DATA'+'.csv'))
    
    logger.info(f'saving {centroids_stacked.shape[0]} centroid pixel coordinates')
    events.emit(events.CENTROIDS_FOUND, stage='stack', n=int(centroids_stacked.shape[0]),
                image_shape=[int(imgs_0.shape[0]), int(imgs_0.shape[1])])
    # wider than the pane it lands in, so the zoom control has real detail to reveal
    # rather than magnified PNG pixels
    events.png_event('stack_preview', image=stacked, max_width=1600)
    # plate solve
    flag_found_IDs = False
    df_identification = None
    pointing_metrics = {}
    solution = platesolve_triangle.platesolve(centroids_stacked, stacked.shape, options = options, output_dir = output_dir)
    # how the solve went, not just whether: a failure that records nothing cannot be
    # diagnosed afterwards, and the interesting failures are the marginal ones
    _diag = solution.get('diagnostics') or {}
    solve_provenance = {
        'solver': solution.get('solver'),
        'pattern_db': solution.get('pattern_db'),
        'verify_catalogue': solution.get('verify_catalogue'),
        'n_candidates': _diag.get('n_candidates'),
        'n_clusters_checked': _diag.get('n_clusters_checked'),
        'threshold': _diag.get('threshold'),
        'anchor_rounds_used': _diag.get('anchor_rounds_used'),
        'layers_used': _diag.get('layers_used'),
        'noise_px_used': _diag.get('noise_px_used'),
        'time_s': _diag.get('time_s'),
        'mirror': solution.get('mirror'),
        'n_centroids_offered': int(centroids_stacked.shape[0]),
    }
    print(solution)
    logger.info(str(solution))
    if not solution['ra'] is None:
        df_identification = pd.DataFrame({'px': np.array(solution['matched_centroids'])[:, 1],
                           'py': np.array(solution['matched_centroids'])[:, 0],
                           #'ID': solution['matched_catID'],
                           'RA': np.degrees(np.array(solution['matched_stars'])[:, 0]),
                           'DEC': np.degrees(np.array(solution['matched_stars'])[:, 1]),
                           'magV': np.array(solution['matched_stars'])[:, 5]})
        
        df_identification.to_csv(data_dir / ('STACKED_CENTROIDS_MATCHED_ID'+'.csv'))
        flag_found_IDs = True

        # the epoch matters for naming by position: these are the fastest-moving stars
        try:
            label_epoch = date_string_to_float(options['observation_date'])
        except Exception:
            label_epoch = 2024.0
        star_labels.emit_from_solution(solution, stacked.shape, epoch=label_epoch)

        header_pointing = read_pointing(files[0])
        separation, verdict = pointing_comment(header_pointing, solution['ra'],
                                               solution['dec'])
        if verdict:
            message = (f'solved position is {separation:.2f}° from the header '
                       f'(RA {header_pointing[0]:.3f}°, Dec {header_pointing[1]:.3f}°): '
                       f'{verdict}')
            print(message)
            logger.info(message)
            events.log(message,
                       level='info' if separation < 5 else 'warning')
            # carried on the stage's own metrics event below, so the UI finds it
            # where it looks for everything else about stage 1
            pointing_metrics = {'header_pointing_separation_deg': separation,
                                'header_pointing_verdict': verdict}
    else:
        logger.error("ERROR: platesolve failed to identify location")
        print("ERROR: platesolve failed to identify location")
    

    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_title(f'Largest {min(options["d"], len(centroids_stacked))} of {len(centroids_stacked)} stars found on stacked image')
    # Draw a strided copy, mapped back onto the full pixel grid with `extent`, so the
    # scatter and annotations below still work in original pixel coordinates.
    # imshow applies the colormap at the resolution of the array it is given: on a
    # 3520x4656 frame that is a 499 MiB float64 RGBA intermediate, which is enough to
    # fail outright on a memory-pressured machine. A strided copy is visually identical
    # at any sane figure size and costs step^2 less.
    display_step = max(1, int(np.ceil(max(stacked.shape) / 1400)))
    plt.imshow(stacked[::display_step, ::display_step], cmap='gray_r',
               vmin=np.percentile(stacked, 50), vmax=np.percentile(stacked, 95),
               extent=(0, stacked.shape[1], stacked.shape[0], 0))
    shift = 0 if options['centroid_gaussian_subtract'] else 0.5
    plt.scatter(centroids_stacked[:options["d"], 1]-shift, centroids_stacked[:options["d"], 0]-shift, marker='x') # subtract half pixel to align with image properly
    if flag_found_IDs:
        # the stage-1 platesolve does not carry catalogue IDs, so annotate with magnitude only
        for ind, (index, row) in enumerate(df_identification.iterrows()):
            if ind >= options["d"]:
                break
            plt.gca().annotate(f'Mag={row["magV"]:.1f}', (row['px'], row['py']), color='r')
    plt.savefig(output_dir / ('CentroidsStackGood'+starttime+'.png'), bbox_inches="tight", dpi=600)
    if options['flag_display']:
        show_scanlines(stacked, fig, ax)
        plt.show(block=True)
    plt.clf()

    results_dict = {
                         'MEE2024 version': _version(),
                         'platesolved' : flag_found_IDs,
                         'solve': solve_provenance,
                         'n_centroids' : centroids_stacked.shape[0],
                         'img_shape' : imgs_0.shape,
                         'RA' : solution['ra'],
                         'DEC' : solution['dec'],
                         'roll' : solution['roll'],
                         'platescale/arcsec' : solution['platescale/arcsec'],#solution['FOV'] / max(imgs_0.shape) if flag_found_IDs else None,
                         '#frames stacked':len(files),
                         'source_files' : str(files),
                         # the folder the frames came from, as its own key: collating a
                         # batch afterwards should not mean parsing a stringified list
                         'source_folder': str(Path(files[0]).parent) if files else None,
                         # what was actually applied, so two runs can be compared rather
                         # than assumed identical. Recovering this from the log meant
                         # opening every archive; several of these change the star set
                         'calibration': {
                             'darks': [str(p) for p in (darkfiles or [])],
                             'flats': [str(p) for p in (flatfiles or [])],
                             'n_darks': len(darkfiles or []),
                             'n_flats': len(flatfiles or []),
                         },
                         'effective_options': {k: v for k, v in options.items()
                                               if not k.startswith('_')},
                         # per-frame alignment: the drift, the residual scatter and the
                         # dither span. All three were computed and then reached only a
                         # print() and a plot, so drift could not be recovered afterwards
                         'alignment': {
                             'shifts_px': [None if s is None else [float(s[0]), float(s[1])]
                                           for s in shifts],
                             'rms_px': [None if r is None else float(r) for r in rms_errors],
                             'dither_span_px': float(hotpixels.dither_span(shifts)),
                             'frames': [e['frame'] for e in aligned],
                             'n_matched': [e['n_matched'] for e in aligned],
                         },
                         'starttime':starttime,
                         # from the first frame's FITS header; lets stage 2 score a blind
                         # date guess against the truth. None for inputs without a header.
                         'observation_date_header': read_observation_date(files[0]),
                         # the mount's own claim about where it was pointing, so the
                         # solved position can be scored against it here and in stage 2
                         'header_pointing': read_pointing(files[0]),
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
    # The PSF of the stacked image, for the UI: median FWHM (the seeing number), the
    # sampling verdict that decides which centroiding algorithm is honest, ellipticity as
    # the optics/tracking diagnostic, and the columns behind the panel's maps. Measured on
    # every run because sampling is a property of the setup, not of the software: the three
    # bundled datasets span FWHM 1.2 to 3.0 px (docs/PSF_REVIEW.md §6). Never fatal.
    psf_metrics = {}
    try:
        from mee2024 import psf as psf_module
        psf_payload = psf_module.event_payload(
            stacked, centroids_stacked, platescale_arcsec=solution['platescale/arcsec'])
        if psf_payload:
            events.emit(events.PSF, **psf_payload)
            psf_summary = psf_payload['summary']
            psf_metrics = {'fwhm_px': psf_summary.get('fwhm_px'),
                           'fwhm_arcsec': psf_summary.get('fwhm_arcsec'),
                           'psf_ellipticity': psf_summary.get('ellipticity'),
                           'undersampled': psf_summary.get('undersampled')}
            if psf_summary.get('undersampled'):
                events.log(
                    f'the star images are undersampled (FWHM '
                    f'{psf_summary["fwhm_px"]:.1f} px, under 2): centroid positions '
                    f'carry pixel-phase bias no averaging can remove', level='warning')
    except Exception as exc:
        events.log(f'PSF measurement skipped: {exc}', level='warning')

    events.emit(events.METRICS, stage='stack', n_centroids=int(centroids_stacked.shape[0]),
                n_frames=len(files), platesolved=bool(flag_found_IDs),
                dither_span_px=float(hotpixels.dither_span(shifts)),
                solver=solve_provenance.get('solver'),
                ra=solution['ra'], dec=solution['dec'], roll=solution['roll'],
                platescale=solution['platescale/arcsec'],
                stack_rms_px=[None if r is None else float(r) for r in rms_errors],
                **psf_metrics, **pointing_metrics)
    
    print('making archive', output_dir, Path(output_dir).parent)
    shutil.make_archive(data_dir, 'zip', Path(data_dir))
    zipfilepath = Path(data_dir).parent / 'data.zip'
    final_zip = Path(output_dir).parent / f'centroid_data{starttime}.zip'
    shutil.move(zipfilepath, final_zip)

    logger.info('end time: ' + str(datetime.datetime.now()) + '\n')
    print('Done!')
    return final_zip
