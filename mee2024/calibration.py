"""
The calibration library: master darks keyed by gain and exposure, and a master flat.

Before this there was no way to build a master dark or flat except as a side effect of a
light run -- `save_dark_flat` wrote one beside the results, keyed to nothing, headed with
`NCOMBINE` and little else. So a master could not be found again, could not be checked
against the frames it was about to calibrate, and could not be reused without hauling the
whole session folder around. This module builds them deliberately, records what they are,
and matches them to lights by the two things that actually have to agree.

**Gain and exposure, and nothing else.** A dark is `pedestal + rate x t` (measured: the
pedestal is 1.7% of a defect at 10 s but 63% at 0.1 s), so a tier cannot be extrapolated
from another tier in either direction, and this module never tries. It matches a tier or
reports that it has none. Capture every tier you mean to use; at eclipse-ladder exposures
they are nearly free.

**Never `IMAGETYP`.** SharpCap's Sequencer has no frame-type command, so scripted darks
and flats are recorded as `FRAMETYP = IMAGETYP = 'Light'`. The Leon capture scripts put the
type in `TARGETNAME`, which lands in `OBJECT` (`DARK_G0_0p1s`, `DARK_G101_4s`, `FLATS`),
and say so in their own header comments. Classifying on `IMAGETYP` would silently treat
fifty darks as a light field.

**Temperature is recorded, not keyed.** For a cooled body the setpoint is in `SET-TEMP` and
is what makes a dark comparable; for an uncooled one the temperature is measured rather
than chosen, so keying on it means never matching. Either way the observer is the one who
segregates by temperature, and this warns when the frames disagree with the master.

**No flat-darks.** Measured at 87 ppm on the normalised flat -- 1.2% of the PRNU, worth
0.1-0.25 mas on a centroid -- against one more input to pick and nine more minutes of a
session. What makes that valid is the mid-range fill of the flat itself, so
:func:`build_master_flat` checks the fill and says so when it is not there.

**Everything streams.** `np.mean(np.array(open_images(files)))` materialises the whole
cube: 50 frames of the 26 MP ASI2600MM is 5.2 GB as float32, and 10.4 GB while the list
and the array both exist -- more than the machine has, so a full dark tier could not be
combined at all. :func:`combine` makes one pass with four frame-sized accumulators (sum,
sum of squares, running minimum, running maximum), which is independent of frame count and
yields the per-pixel standard deviation for nothing -- the only thing that finds
telegraph/RTS pixels, since the mean finds only the hot ones.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from mee2024 import events
from mee2024.MEE2024util import _version

#: Format marker in every ``meta.json``, so a future layout change is detectable.
FORMAT = 'mee2024-callib'
FORMAT_VERSION = 1

INDEX_FILE = 'library.json'
MASTER_FILE = 'master.fits'
SIGMA_FILE = 'sigma.fits'
META_FILE = 'meta.json'

#: Drop each pixel's own highest and lowest frame from the mean. A cosmic ray or a
#: satellite is a handful of pixels in one frame out of fifty, and a plain mean carries
#: 1/50th of it into every light the master ever calibrates. See :func:`combine` for why
#: this rather than sigma-clipping, which cannot work in a streaming pass.
REJECT_MINMAX = 'minmax'

#: Below this many frames, min-max rejection costs more information than it saves: at five
#: frames it discards 40% of the data, and the noise penalty sqrt(n/(n-2)) reaches 1.29.
MIN_FRAMES_TO_REJECT = 5

#: How closely a dark's exposure must match the light's, as a fraction. Tight on purpose:
#: this is a match test, not a scaling factor, and a 4 s dark has no business calibrating
#: a 6 s light however close 4 and 6 look on a ladder.
EXPOSURE_TOLERANCE = 0.01

#: Setpoint mismatch worth a warning, in Celsius. A cooled body holds its setpoint to a
#: few tenths, so any real difference here means two different thermal regimes.
SETPOINT_WARN_C = 0.5

#: Measured-temperature mismatch worth a warning, in Celsius, for an uncooled body where
#: there is no setpoint to compare. Wider, because nothing was promised.
MEASURED_WARN_C = 5.0

#: Where a flat should sit in its container for the no-flat-dark argument to hold. The
#: unsubtracted offset pedestal is diluted by the signal, so at half scale it leaves
#: ~0.3% at a vignetted corner -- smooth, multiplicative and centroid-irrelevant. At a
#: tenth of scale it is five times that and no longer ignorable.
FLAT_FILL_MIN = 0.25
FLAT_FILL_MAX = 0.75

#: Header keys read from every calibration frame, and carried into the master.
PROVENANCE_KEYS = ('EXPTIME', 'GAIN', 'EGAIN', 'OFFSET', 'BLKLEVEL', 'CCD-TEMP',
                   'SET-TEMP', 'CAMID', 'INSTRUME', 'TELESCOP', 'FOCALLEN', 'XPIXSZ',
                   'XBINNING', 'YBINNING', 'FILTER', 'FOCUSPOS', 'BIASADU', 'RDNOISE',
                   'ADCBITS', 'BITPIX', 'OBJECT', 'DATE-OBS')

_UNSAFE = re.compile(r'[^A-Za-z0-9_.-]+')


class CalibrationError(Exception):
    """A calibration set that cannot be combined, with the reason."""


# ------------------------------------------------------------------ reading headers

def read_cal_header(path):
    """The header keys that identify a calibration frame. Missing keys are simply absent.

    Returns a plain dict so it can go straight into ``meta.json``; FITS values are cast to
    ``int``/``float``/``str`` because a ``fits.Card`` value does not round-trip through
    JSON.

    A SER frame is described by its container plus the capture software's sidecar, which
    carries the gain, exposure, offset, binning, camera and both temperatures -- everything
    the library keys on.
    """
    from mee2024 import ser

    out = {}
    if ser.is_ser(path):
        try:
            container, index = ser.parse_ref(path)
            handle = ser.open_ser(container)
            header = handle.fits_header(index or 0)
            for key in PROVENANCE_KEYS:
                value = header.get(key)
                if value not in (None, ''):
                    out[key] = value
            out['shape'] = [handle.header['height'], handle.header['width']]
        except Exception:
            return {}
        return out
    try:
        with fits.open(path) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
            shape = (hdul['PRIMARY'] if 'PRIMARY' in hdul else hdul[0]).shape
    except Exception:
        return out
    for key in PROVENANCE_KEYS:
        value = header.get(key)
        if value in (None, ''):
            continue
        if isinstance(value, bool):
            out[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = float(value)
        else:
            out[key] = str(value).strip()
    if shape and len(shape) == 2:
        out['shape'] = [int(shape[0]), int(shape[1])]
    return out


def camera_token(header):
    """A short filesystem-safe camera name for a folder name.

    ``CAMID`` is a 16-digit serial -- correct but unreadable, and these folder names sit
    inside paths that are already close to the Windows limit. ``INSTRUME`` reads better
    ('ZWO ASI2600MM Pro' -> 'ASI2600MM'), and the serial is kept in ``meta.json`` where
    identity actually has to be exact.
    """
    name = str(header.get('INSTRUME') or header.get('CAMID') or 'camera')
    match = re.search(r'ASI\s*(\d+\w*)', name, re.IGNORECASE)
    if match:
        return f'ASI{match.group(1).upper()}'
    return _UNSAFE.sub('', name.replace(' ', ''))[:16] or 'camera'


def _exposure_token(seconds):
    """0.1 -> '0p100s'. A decimal point in a folder name is legal but reads as a suffix."""
    return f'{float(seconds):.3f}'.replace('.', 'p') + 's'


def dark_key(header):
    """What has to agree for a dark to calibrate a light: the folder name that encodes it.

    Camera, gain, exposure and binning. Not temperature -- see the module docstring. Not
    ``OBJECT``, which names the capture run rather than the configuration, so two nights
    of the same tier land in the same entry and the second simply supersedes the first.
    """
    missing = [k for k in ('GAIN', 'EXPTIME') if header.get(k) is None]
    if missing:
        raise CalibrationError(
            f'the frames do not say what they are: no {" or ".join(missing)} in the FITS '
            f'header. A dark is matched to a light by gain and exposure, so a dark that '
            f'cannot state either cannot be filed.')
    binning = int(header.get('XBINNING') or 1)
    return (f'dark_{camera_token(header)}_g{int(header["GAIN"])}'
            f'_{_exposure_token(header["EXPTIME"])}_bin{binning}')


def flat_key(header):
    """What has to agree for a flat: the optical train, not the gain.

    Deliberately gain-free. One gain-0 flat set corrects the gain-101 night data too, on
    the first-order assumption that PRNU and vignetting do not depend on gain -- which is
    what the Leon capture scripts assume and state. The gain that was actually used is
    recorded, and :func:`match_flat` warns when it differs, so the assumption is visible
    rather than buried.

    ``FOCUSPOS`` is in the key because dust-donut geometry depends on focus, quantised so
    that ordinary focuser jitter does not split one flat set into several entries.
    """
    binning = int(header.get('XBINNING') or 1)
    train = _UNSAFE.sub('', str(header.get('TELESCOP') or 'scope').replace(' ', ''))[:20]
    parts = [f'flat_{camera_token(header)}', train or 'scope', f'bin{binning}']
    if header.get('FILTER'):
        parts.append(_UNSAFE.sub('', str(header['FILTER']))[:8])
    if header.get('FOCUSPOS') is not None:
        parts.append(f'f{int(round(float(header["FOCUSPOS"]) / 100.0)) * 100}')
    return '_'.join(parts)


# ------------------------------------------------------------------ combining frames

def combine(files, reject=REJECT_MINMAX, progress=None):
    """Mean and per-pixel standard deviation of ``files``, without holding them all.

    One streaming pass, four frame-sized accumulators: sum, sum of squares, running
    minimum and running maximum. The alternative,
    ``np.mean(np.array(open_images(files)))``, is 5.2 GB for fifty frames of the 26 MP
    ASI2600MM and 10.4 GB while the list and the array both exist -- more than the machine
    has, so a full dark tier could not be combined at all.

    ``reject='minmax'`` drops each pixel's own extreme high and low frame from the mean,
    which is what keeps a cosmic ray or a satellite trail out of a master that will then
    calibrate every light of the session -- a mean of fifty carries a fiftieth of the hit
    into all of them. It comes free from the same pass:
    ``mean = (sum - min - max) / (n - 2)``.

    **Sigma-clipping was tried first and is the wrong tool here.** Clipping needs a spread
    to measure against, and the only spread available in a streaming pass is the per-pixel
    standard deviation -- which the outlier itself inflates. One frame at 60 000 ADU among
    nine at 500 gives that pixel a sigma near 17 800, so even a 5-sigma cut keeps the
    spike. Min-max rejection needs no scale estimate at all, and it discriminates on the
    right axis: a cosmic ray is extreme in *one* frame, while a hot or telegraph pixel is
    high in *every* frame and so survives with only 2 of N samples trimmed.

    The reported sigma is deliberately the **untrimmed** spread. It is the map that finds
    telegraph/RTS pixels -- the defect class that defeats subtraction entirely -- and
    trimming each pixel's extremes is precisely what would hide them.

    Sums are accumulated about a pedestal taken from the first frame rather than about
    zero. A flat sits near 30 000 ADU, so the squared sum reaches ~5e10 while the variance
    being recovered from it is ~1e3 -- fine in float64, but free to make exact.

    Returns ``(mean, sigma, info)``.
    """
    files = [str(f) for f in files]
    if not files:
        raise CalibrationError('no frames to combine')
    from mee2024.stacker_implementation import open_image

    def read(path, shape):
        frame = open_image(path)
        if shape is not None and frame.shape != shape:
            raise CalibrationError(
                f'{Path(path).name} is {frame.shape[1]}x{frame.shape[0]} but the first '
                f'frame is {shape[1]}x{shape[0]}. A master is combined pixel for pixel, '
                f'so every frame has to be the same size.')
        return frame

    first = read(files[0], None)
    shape = first.shape
    pedestal = float(np.median(first[::8, ::8]))
    del first

    if progress is not None and hasattr(progress, 'start'):
        progress.start(len(files), 'Combining frames')
    total = np.zeros(shape, dtype=np.float64)
    total_sq = np.zeros(shape, dtype=np.float64)
    lowest = np.full(shape, np.inf)
    highest = np.full(shape, -np.inf)
    for index, path in enumerate(files):
        frame = np.asarray(read(path, shape), dtype=np.float64)
        frame -= pedestal                      # in place: no extra frame-sized buffer
        total += frame
        total_sq += frame * frame
        np.minimum(lowest, frame, out=lowest)
        np.maximum(highest, frame, out=highest)
        del frame
        if progress is not None:
            progress.update(index + 1)
    if progress is not None and hasattr(progress, 'finish'):
        progress.finish()

    n = len(files)
    info = {'n_frames': n, 'reject': reject, 'combine': 'mean', 'n_trimmed': 0}
    # the untrimmed spread, which is where RTS pixels show themselves
    variance = np.maximum((total_sq - total * total / n) / max(n - 1, 1), 0.0)
    sigma = np.sqrt(variance)
    if reject == REJECT_MINMAX and n >= MIN_FRAMES_TO_REJECT:
        mean = (total - lowest - highest) / (n - 2)
        info.update(combine=f'mean of {n} less the high and low frame per pixel',
                    n_trimmed=2 * int(np.prod(shape)))
    else:
        mean = total / n
        if reject == REJECT_MINMAX:
            info['note'] = (f'only {n} frame(s): min-max rejection needs '
                            f'{MIN_FRAMES_TO_REJECT}, so a plain mean was taken and a '
                            f'transient in any one frame is in the master')
    mean = mean + pedestal
    return mean.astype(np.float32), sigma.astype(np.float32), info


def is_master(path):
    """Is this file already a master built by :func:`write_entry`?

    ``CALKIND`` is written by nothing else, so its presence is a reliable answer. A master
    handed to the pipeline must not be re-combined with itself, and a *normalised* master
    flat must not be normalised twice -- the second division is by a number near 1, so it
    does almost nothing and looks like it worked.
    """
    try:
        with fits.open(path) as hdul:
            header = hdul['PRIMARY'].header if 'PRIMARY' in hdul else hdul[0].header
        return str(header.get('CALKIND') or '') in ('dark', 'flat'), dict(
            kind=str(header.get('CALKIND') or ''),
            normalised=header.get('CALNORM') is not None,
            n_frames=int(header.get('NCOMBINE') or 0))
    except Exception:
        return False, {}


def load_or_combine(files, reject=REJECT_MINMAX, progress=None):
    """The calibration frame the pipeline should use, from whatever it was given.

    Three cases, and the point is that the caller does not have to know which: one master
    from a library (read as-is), one ordinary frame (read as-is), or many frames (combined
    by streaming). Returns ``(array, info)`` where ``info['source']`` says which happened,
    so the run can record it.
    """
    from mee2024.stacker_implementation import open_image

    files = [str(f) for f in files or []]
    if not files:
        return None, {'source': 'none'}
    if len(files) == 1:
        master, detail = is_master(files[0])
        image = open_image(files[0])
        if master:
            return image, {'source': 'library master', 'n_frames': detail['n_frames'],
                           'normalised': detail['normalised'], 'path': files[0],
                           'combine': 'none (already combined)'}
        return image, {'source': 'single frame', 'n_frames': 1, 'normalised': False,
                       'path': files[0], 'combine': 'none'}
    mean, _, info = combine(files, reject=reject, progress=progress)
    info.update(source=f'{len(files)} frames', normalised=False, path=files[0])
    return mean, info


# ------------------------------------------------------------------ building masters

def _master_header(kind, header, info, extra=None):
    """The provenance a master needs to identify itself.

    The old `save_dark_flat` wrote `NCOMBINE`, `COMBTYPE` and a version, which is enough
    to say a master exists and nothing about whether it may be used. Reuse is the only
    reason to save one, and reuse needs exposure, gain, temperature and the camera.
    """
    out = fits.Header()
    out['MEE2024'] = (_version(), 'version that built this master')
    out['CALKIND'] = (kind, 'dark or flat')
    out['NCOMBINE'] = (int(info['n_frames']), 'frames combined')
    # a FITS card holds 80 characters for keyword, value and comment together, so the
    # short form goes here and meta.json carries the sentence
    out['COMBTYPE'] = ('mean', 'how they were combined')
    out['CALREJ'] = (str(info.get('reject') or 'none'),
                     'per-pixel outlier rejection used')
    out['CALDATE'] = (datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      'when this master was built')
    for key in ('EXPTIME', 'GAIN', 'EGAIN', 'OFFSET', 'CCD-TEMP', 'SET-TEMP', 'CAMID',
                'INSTRUME', 'TELESCOP', 'FOCALLEN', 'XPIXSZ', 'XBINNING', 'FILTER',
                'FOCUSPOS', 'BIASADU', 'RDNOISE', 'ADCBITS'):
        if header.get(key) is not None:
            out[key] = header[key]
    if header.get('DATE-OBS'):
        out['DATE-OBS'] = (header['DATE-OBS'], 'first frame of the set')
    for key, value in (extra or {}).items():
        out[key] = value
    return out


def _bias_note(mean, header, compare=True):
    """Compare the measured bias against the header's ``BIASADU``, and say so.

    ``BIASADU`` has been wrong before -- on the 533MM the header said 393.59 against a
    measured 94.66, a factor of 4.16, and subtracting it as a scalar from a flat would
    have injected a 3.6% error. It is worth checking rather than trusting, and worth
    recording either way. (On the cooled 2600MM it checks out: 500.9 claimed against
    502.4 measured, the difference being the dark current.)

    ``compare=False`` records the level without judging it. That is the right thing for a
    **flat**, whose median is the illumination level and has no business being compared to
    a bias: 33 223 against a claimed 500.9 is not a broken header, it is a correctly
    exposed flat, and warning about it would train the reader to ignore the warning that
    matters.
    """
    measured = float(np.median(mean[::8, ::8]))
    note = {'measured_level_adu' if not compare else 'measured_bias_adu': measured}
    claimed = header.get('BIASADU')
    if claimed is not None:
        note['header_bias_adu'] = float(claimed)
    if not compare or claimed is None:
        return note, None
    if measured > 0 and not 0.8 <= float(claimed) / measured <= 1.25:
        return note, (
            f'the header claims BIASADU {float(claimed):.1f} but the frames measure '
            f'{measured:.1f} -- a factor of {float(claimed) / measured:.2f}. Use the '
            f'measured value; the header one is not usable as written on this camera.')
    return note, None


def build_master_dark(files, reject=REJECT_MINMAX, progress=None):
    """A master dark, its per-pixel sigma, and what the set turned out to be.

    Returns ``(mean, sigma, meta)``. Defect *finding* is left to
    :func:`mee2024.hotpixels.dark_mask` at use time, so the threshold is a decision of the
    run rather than baked into the library.
    """
    files = [str(f) for f in files]
    header = read_cal_header(files[0])
    key = dark_key(header)                       # raises if gain or exposure is missing
    mean, sigma, info = combine(files, reject=reject, progress=progress)
    bias, warning = _bias_note(mean, header)
    meta = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'kind': 'dark',
            'key': key, 'header': header, **info, **bias,
            'source_folder': str(Path(files[0]).parent),
            'files': [Path(f).name for f in files],
            'mean_sigma_adu': float(np.median(sigma[::8, ::8])),
            'warnings': [warning] if warning else []}
    return mean, sigma, meta


def build_master_flat(files, reject=REJECT_MINMAX, progress=None, normalise=True):
    """A master flat, normalised to about 1, plus the fill check that justifies no flat-dark.

    The fill level is not advice. The reason a flat-dark can be skipped is that the
    unsubtracted offset pedestal is diluted by the signal, and the dilution is only small
    while the flat is near mid-range. A flat exposed to a tenth of full scale carries five
    times the residual, and nothing downstream would notice.
    """
    files = [str(f) for f in files]
    header = read_cal_header(files[0])
    key = flat_key(header)
    mean, sigma, info = combine(files, reject=reject, progress=progress)

    full_scale = float(2 ** int(header.get('ADCBITS') or header.get('BITPIX') or 16) - 1)
    level = float(np.median(mean[::8, ::8]))
    fill = level / full_scale if full_scale > 0 else 0.0
    warnings = []
    if not FLAT_FILL_MIN <= fill <= FLAT_FILL_MAX:
        warnings.append(
            f'the flat sits at {100 * fill:.0f}% of full scale ({level:.0f} of '
            f'{full_scale:.0f} ADU), outside the {100 * FLAT_FILL_MIN:.0f}-'
            f'{100 * FLAT_FILL_MAX:.0f}% this pipeline assumes. Mid-range fill is what '
            f'makes it safe to take no flat-darks: the offset pedestal is left in, and '
            f'its share of the signal grows as the fill falls. Re-take the flats nearer '
            f'mid-range, or take flat-darks for this set.')
    # the flat's median is its illumination level, not a bias: recorded, not compared
    bias, _ = _bias_note(mean, header, compare=False)

    if normalise and level > 0:
        # a flat corrects *relative* sensitivity, so it has to be about 1
        mean = (mean / level).astype(np.float32)
        sigma = (sigma / level).astype(np.float32)
    prnu = float(np.std(mean[::8, ::8]))
    meta = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'kind': 'flat',
            'key': key, 'header': header, **info, **bias,
            'source_folder': str(Path(files[0]).parent),
            'files': [Path(f).name for f in files],
            'level_adu': level, 'fill_fraction': fill, 'full_scale_adu': full_scale,
            'prnu_fraction': prnu, 'normalised': bool(normalise and level > 0),
            'gain_note': ('recorded, not keyed: one flat set is assumed to correct every '
                          'gain, since PRNU and vignetting are gain-independent to first '
                          'order'),
            'warnings': warnings}
    return mean, sigma, meta


# ------------------------------------------------------------------ the library on disk

def write_entry(library_root, mean, sigma, meta):
    """Write one master into the library and return its directory.

    ``master.fits`` and ``sigma.fits`` beside a ``meta.json``: the FITS files so any other
    tool can read them, the JSON because a header cannot hold a file list or a warning.
    """
    directory = Path(library_root) / meta['key']
    directory.mkdir(parents=True, exist_ok=True)
    kind = meta['kind']
    header = _master_header(kind, meta['header'], meta)
    if kind == 'flat' and meta.get('normalised'):
        header['CALNORM'] = (meta['level_adu'], 'ADU the flat was divided by')
        header['CALFILL'] = (meta['fill_fraction'], 'fraction of full scale before that')
    fits.writeto(directory / MASTER_FILE, np.asarray(mean, dtype=np.float32),
                 header=header, overwrite=True)
    sigma_header = _master_header(kind, meta['header'], meta,
                                 extra={'CALDATA': ('sigma',
                                                    'per-pixel standard deviation')})
    fits.writeto(directory / SIGMA_FILE, np.asarray(sigma, dtype=np.float32),
                 header=sigma_header, overwrite=True)
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, default=str),
                                       encoding='utf-8')
    return directory


def read_index(library_root):
    """Every entry in a library, read from the per-entry ``meta.json`` files.

    Derived from the folders rather than trusted from ``library.json``: an entry copied in
    by hand, or a library assembled from two machines, should still be usable, and an index
    that can disagree with the data is worse than no index at all. ``library.json`` is
    written for the human reading the folder.
    """
    root = Path(library_root)
    entries = []
    if not root.is_dir():
        return entries
    for meta_path in sorted(root.glob(f'*/{META_FILE}')):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if meta.get('format') != FORMAT:
            continue
        meta['directory'] = str(meta_path.parent)
        meta['master'] = str(meta_path.parent / MASTER_FILE)
        meta['sigma'] = str(meta_path.parent / SIGMA_FILE)
        if Path(meta['master']).exists():
            entries.append(meta)
    return entries


def write_index(library_root, entries=None):
    """Refresh ``library.json``: what is in here, one line of fact per entry."""
    root = Path(library_root)
    root.mkdir(parents=True, exist_ok=True)
    entries = entries if entries is not None else read_index(root)
    index = {
        'format': FORMAT, 'format_version': FORMAT_VERSION,
        'mee2024_version': _version(),
        'written': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'entries': [{
            'key': e['key'], 'kind': e['kind'],
            'exptime': e['header'].get('EXPTIME'), 'gain': e['header'].get('GAIN'),
            'camera': e['header'].get('INSTRUME'), 'camid': e['header'].get('CAMID'),
            'set_temp': e['header'].get('SET-TEMP'),
            'ccd_temp': e['header'].get('CCD-TEMP'),
            'binning': e['header'].get('XBINNING'), 'shape': e['header'].get('shape'),
            'n_frames': e.get('n_frames'), 'source_folder': e.get('source_folder'),
            'warnings': e.get('warnings') or [],
        } for e in entries],
    }
    (root / INDEX_FILE).write_text(json.dumps(index, indent=2, default=str),
                                    encoding='utf-8')
    return root / INDEX_FILE


# ------------------------------------------------------------------ discovering sets

#: Folder names that mark calibration rather than science, checked case-insensitively
#: against any path component. The Leon scripts make the capture folders self-labelling
#: (`DARK_G0_0p1s`, `FLATS`) precisely so this is possible without opening a frame.
CALIBRATION_WORDS = ('dark', 'flat', 'bias', 'offset')


def looks_like_calibration(path, root=None):
    """Does this folder hold calibration frames rather than a field?

    Name first, because it is free and the capture scripts made the names carry the type.
    ``OBJECT`` is the fallback and the authority -- it is where ``TARGETNAME`` lands -- but
    reading it means opening a frame, so it is only consulted when the name is silent.

    ``IMAGETYP`` is never consulted. On every scripted set examined it says ``'Light'`` on
    darks and flats alike, because the Sequencer cannot set it.
    """
    path = Path(path)
    parts = [p.lower() for p in path.parts]
    if root is not None:
        try:
            parts = [p.lower() for p in path.relative_to(Path(root)).parts]
        except ValueError:
            pass
    for part in parts:
        for word in CALIBRATION_WORDS:
            # a word boundary, so 'Darkfield_survey' is not caught by 'dark'
            if re.search(rf'(^|[^a-z]){word}s?([^a-z]|$)', part):
                return True
    return False


def classify_frames(files):
    """'dark', 'flat' or None for a folder of frames, from ``OBJECT``.

    Consulted when the folder name says nothing. ``OBJECT`` carries ``TARGETNAME``, which
    is the only place a scripted capture can record the frame type at all.
    """
    for path in list(files)[:1]:
        obj = str(read_cal_header(path).get('OBJECT') or '').lower()
        if re.search(r'(^|[^a-z])(bias|offset)', obj):
            return 'dark'
        if re.search(r'(^|[^a-z])dark', obj):
            return 'dark'
        if re.search(r'(^|[^a-z])flat', obj):
            return 'flat'
    return None


def find_calibration_sets(root):
    """Folders of calibration frames under ``root``, one entry per set.

    Walks the way the batch scanner does -- a folder that directly holds frames is a set,
    and is not descended into -- so the usual `DARKS/DARK_G0_0p1s/22_34_05/*.fits` nesting
    resolves to one set per tier without the caller knowing the layout.
    """
    from mee2024.ui.batch import find_fields

    fields, info = find_fields(root, max_fields=10_000, min_frames=2,
                              skip_calibration=False)
    sets = []
    for field in fields:
        kind = classify_frames(field['frames'])
        if kind is None and looks_like_calibration(field['folder'], root):
            # named like calibration but the header does not say which: darks are the
            # commoner case and a flat is caught by its fill check downstream
            kind = 'dark'
        if kind is None:
            continue
        sets.append({'kind': kind, 'folder': field['folder'],
                     'frames': field['frames'],
                     'name': set_name(field['frames'], field['folder'])})
    return sets, info


def set_name(frames, folder):
    """What to call a calibration set in the log.

    The folder that directly holds the frames is a capture timestamp (`22_34_05`), which
    identifies nothing to a reader. ``OBJECT`` carries ``TARGETNAME`` and is exactly the
    tier name the capture script chose (`DARK_G0_0p1s`), so prefer it; fall back to the
    nearest folder above that is not itself a bare timestamp.
    """
    obj = str(read_cal_header(frames[0]).get('OBJECT') or '').strip() if frames else ''
    if obj:
        return obj
    path = Path(folder)
    for candidate in (path, *path.parents):
        if not re.fullmatch(r'[\d_.\-]+', candidate.name or ''):
            return candidate.name
    return path.name


# ------------------------------------------------------------------ building a library

def build_library(library_root, darks_root=None, flats_root=None, reject=REJECT_MINMAX,
                  progress_for=None, on_note=None):
    """Build or extend a calibration library from folders of frames.

    Every folder of frames beneath ``darks_root``/``flats_root`` becomes one entry, keyed
    by what it is rather than where it came from -- so re-running a night's darks
    supersedes the previous copy of that tier instead of accumulating near-duplicates
    nobody can choose between.

    A set that cannot be combined is reported and skipped: one unusable tier must not cost
    the other seven.
    """
    note = on_note or (lambda message, **kw: events.log(message, **kw))
    root = Path(library_root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for label, source in (('dark', darks_root), ('flat', flats_root)):
        if not source:
            continue
        sets, info = find_calibration_sets(source)
        sets = [s for s in sets if s['kind'] == label]
        if not sets:
            note(f'no {label} sets found under {source}'
                 + (f' -- {info["truncated"]}' if info.get('truncated') else ''),
                 level='warning')
            continue
        note(f'{len(sets)} {label} set(s) under {source}')
        for entry in sets:
            frames = entry['frames']
            try:
                progress = progress_for(entry['name'], len(frames)) if progress_for else None
                builder = build_master_dark if label == 'dark' else build_master_flat
                mean, sigma, meta = builder(frames, reject=reject,
                                            progress=progress)
                directory = write_entry(root, mean, sigma, meta)
                for warning in meta.get('warnings') or []:
                    note(warning, level='warning')
                note(f'{meta["key"]}: {meta["combine"]}'
                     + (f' -- {meta["note"]}' if meta.get('note') else '')
                     + f' -> {directory.name}')
                results.append(meta)
                del mean, sigma
            except Exception as exc:
                note(f'{entry["name"]}: could not build a master {label} -- '
                     f'{type(exc).__name__}: {exc}', level='warning')
    write_index(root)
    return results


# ------------------------------------------------------------------ matching to lights

def _temperature_warning(light, master):
    """A note when the master's temperature does not match the light's, or None.

    **Both** the setpoint and the measured temperature are checked, not the setpoint alone.
    A matching setpoint is what makes two cooled sets comparable, but it is a *request*, and
    a cooler that failed to hold it still reports the setpoint it was asked for -- so
    stopping at the setpoint would stay silent on exactly the case worth catching. The
    measured comparison is the wider of the two, because for a cooled body a couple of
    degrees of cooler ripple is normal (the Leon darks span 9.6-12.4 C on one +10 C
    setpoint) and nothing was promised on an uncooled one.

    Defect amplitudes track temperature and no subtraction follows that, so this is a
    warning about the *mask* being the useful half, not a reason to refuse the tier.
    """
    notes = []
    for key, limit, what in (('SET-TEMP', SETPOINT_WARN_C, 'setpoint'),
                             ('CCD-TEMP', MEASURED_WARN_C, 'measured sensor temperature')):
        a, b = light.get(key), master.get(key)
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > limit:
            notes.append(f'the master was taken at a {what} of {float(b):.1f} C and these '
                         f'frames are at {float(a):.1f} C')
    if not notes:
        return None
    return ('; '.join(notes) + '. Defect amplitudes track temperature, which no '
            'subtraction follows -- segregate the library by temperature, or expect the '
            'mask to be the useful half.')


def match_dark(entries, light_header):
    """The library dark for these lights, or ``(None, reason)``.

    Gain and exposure must agree; camera, binning and shape must agree; temperature is
    reported. **Nothing is scaled.** Defect amplitude is ``pedestal + rate x t`` -- the
    pedestal is 63% of a defect at 0.1 s and 1.7% at 10 s -- so a tier interpolated from
    its neighbours would be wrong in a way that looks plausible. A missing tier is
    reported as missing.
    """
    if light_header.get('GAIN') is None or light_header.get('EXPTIME') is None:
        return None, ('these frames do not record GAIN and EXPTIME, so no dark can be '
                      'matched to them by the only two things that have to agree')
    gain = int(light_header['GAIN'])
    exptime = float(light_header['EXPTIME'])
    camera = camera_token(light_header)
    binning = int(light_header.get('XBINNING') or 1)
    shape = light_header.get('shape')

    near = []
    for entry in entries:
        if entry.get('kind') != 'dark':
            continue
        header = entry.get('header') or {}
        if camera_token(header) != camera:
            continue
        if int(header.get('XBINNING') or 1) != binning:
            continue
        if shape and header.get('shape') and list(header['shape']) != list(shape):
            continue
        if int(header.get('GAIN', -1)) != gain:
            near.append(entry)
            continue
        master_exp = float(header.get('EXPTIME', -1))
        if abs(master_exp - exptime) <= EXPOSURE_TOLERANCE * max(exptime, 1e-6):
            return entry, _temperature_warning(light_header, header)
        near.append(entry)

    if not near:
        return None, (f'the library has no dark for this camera at bin {binning}'
                      + (f' and {shape[1]}x{shape[0]}' if shape else ''))
    available = ', '.join(sorted({f'g{int(e["header"].get("GAIN", -1))}/'
                                  f'{float(e["header"].get("EXPTIME", 0)):g}s'
                                  for e in near}))
    return None, (f'the library has no dark at gain {gain} and {exptime:g} s. It has '
                  f'{available}. Tiers are not interpolated -- a defect is '
                  f'pedestal + rate x time, so a neighbouring tier is the wrong answer '
                  f'rather than an approximate one. Capture this tier.')


def match_flat(entries, light_header):
    """The library flat for these lights, or ``(None, reason)``.

    Matched on the optical train and geometry, not the gain -- see :func:`flat_key`. A
    gain difference is a note, not a rejection, because that is the documented assumption
    the flats were taken under.
    """
    camera = camera_token(light_header)
    binning = int(light_header.get('XBINNING') or 1)
    shape = light_header.get('shape')
    train = str(light_header.get('TELESCOP') or '')
    candidates = []
    for entry in entries:
        if entry.get('kind') != 'flat':
            continue
        header = entry.get('header') or {}
        if camera_token(header) != camera:
            continue
        if int(header.get('XBINNING') or 1) != binning:
            continue
        if shape and header.get('shape') and list(header['shape']) != list(shape):
            continue
        if train and str(header.get('TELESCOP') or '') != train:
            continue
        candidates.append(entry)
    if not candidates:
        return None, 'the library has no flat for this camera and optical train'
    # the closest focus wins: dust-donut geometry is what a flat gets wrong when focus moves
    light_focus = light_header.get('FOCUSPOS')
    if light_focus is not None:
        candidates.sort(key=lambda e: abs(float(e['header'].get('FOCUSPOS', 0))
                                          - float(light_focus)))
    best = candidates[0]
    header = best.get('header') or {}
    notes = []
    if (header.get('GAIN') is not None and light_header.get('GAIN') is not None
            and int(header['GAIN']) != int(light_header['GAIN'])):
        notes.append(f'the flat was taken at gain {int(header["GAIN"])} and these frames '
                     f'are at gain {int(light_header["GAIN"])}; it is applied on the '
                     f'assumption that PRNU and vignetting do not depend on gain')
    if (header.get('FOCUSPOS') is not None and light_focus is not None
            and abs(float(header['FOCUSPOS']) - float(light_focus)) > 200):
        notes.append(f'the flat was focused at {int(float(header["FOCUSPOS"]))} and these '
                     f'frames at {int(float(light_focus))}; dust shadows move with focus')
    return best, '; '.join(notes) or None


def resolve_for_field(library_root, frames, want_dark=True, want_flat=True, on_note=None):
    """Which masters a field should use, and what to say about them.

    The point of the library: a batch points at one folder and every field gets the tier
    that matches *its own* frames, reported per field. That replaces the darks-and-flats
    picker, whose selection was invisible in folder mode and applied to every field
    whether it fitted or not.

    Returns ``{'dark': path or None, 'flat': path or None, 'notes': [...]}``.
    """
    note = on_note or (lambda message, **kw: events.log(message, **kw))
    out = {'dark': None, 'flat': None, 'notes': []}
    if not library_root:
        return out
    entries = read_index(library_root)
    if not entries:
        out['notes'].append(f'no calibration library at {library_root}')
        note(out['notes'][-1], level='warning')
        return out
    header = read_cal_header(frames[0]) if frames else {}
    for kind, wanted, matcher in (('dark', want_dark, match_dark),
                                  ('flat', want_flat, match_flat)):
        if not wanted:
            continue
        entry, reason = matcher(entries, header)
        if entry is None:
            out['notes'].append(f'no {kind}: {reason}')
            note(f'no master {kind} applied -- {reason}', level='warning')
            continue
        out[kind] = entry['master']
        message = f'master {kind} {entry["key"]} ({entry.get("n_frames", "?")} frames)'
        if reason:
            message += f' -- {reason}'
            note(message, level='warning')
        else:
            note(message)
        out['notes'].append(message)
    return out
