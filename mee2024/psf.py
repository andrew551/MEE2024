"""
Measuring the point spread function from detected stars.

The pipeline's product is astrometry, so the PSF matters here for exactly three reasons:
it sets the centroid precision the photons permit (~FWHM/2.355/SNR per star), its sampling
(FWHM in pixels) decides which centroiding algorithm is honest to use, and its *asymmetry*
displaces every centroid by a shape-dependent fraction of a pixel — harmless while the
pattern is static, a bias when it changes between nights. docs/PSF_REVIEW.md carries the
survey behind those claims.

This module measures rather than assumes: cutouts at detected centroids, Gaussian-windowed
moments (SExtractor-style), pixel-integrated elliptical Gaussian fits, and a Moffat fit to
the stacked high-SNR profile. It is used by the exploration tool and, for the cheap subset,
by stage 1 to report seeing to the UI.
"""

import numpy as np

#: cutout half-width. 10 px covers FWHM up to ~5 px with sky to spare on every dataset
#: we have; bigger costs isolation (more neighbours excluded).
CUT = 10

#: a star whose peak reaches this fraction of the detector's full scale is treated as
#: saturated: its core carries no shape information and biases every fit
SATURATION_FRACTION = 0.9


# ---------------------------------------------------------------------- cutouts

def full_scale_of(image):
    """The clipping level of the frame, or None when nothing in it clips.

    Saturation is only *detectable* as a plateau: several pixels sharing the frame's top
    value. Guessing an ADC ceiling instead was tried and misfires — a frame whose
    brightest star merely comes near a plausible ceiling gets its best stars flagged as
    saturated, which is exactly backwards.
    """
    peak = float(np.max(image))
    if peak <= 0:
        return None
    if int(np.sum(np.asarray(image) >= peak * 0.999)) >= 4:
        return peak
    return None


def extract_cutouts(image, positions, cut=CUT, isolation_px=None):
    """Background-subtracted cutouts around integer-rounded positions, with quality flags.

    Returns a list of dicts: data (2*cut+1 square), origin (row, col of the corner),
    peak, flux, background, noise, saturated, isolated. Positions too near the edge are
    dropped outright — a truncated cutout would bias every measurement made on it.
    """
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    isolation = isolation_px if isolation_px is not None else 1.5 * cut
    full_scale = full_scale_of(image)
    saturation = SATURATION_FRACTION * full_scale if full_scale else np.inf

    rounded = np.round(np.asarray(positions, dtype=float)).astype(int)
    out = []
    for index, (row, col) in enumerate(rounded):
        if not (cut <= row < height - cut and cut <= col < width - cut):
            continue
        data = image[row - cut:row + cut + 1, col - cut:col + cut + 1].copy()
        # local background from the cutout's border ring: nearer than a global estimate,
        # cheaper than an annulus fit, and robust to a neighbour in one corner
        ring = np.concatenate([data[0, :], data[-1, :], data[1:-1, 0], data[1:-1, -1]])
        background = float(np.median(ring))
        noise = 1.4826 * float(np.median(np.abs(ring - background)))
        data -= background
        others = np.delete(rounded, index, axis=0)
        if len(others):
            gap = np.min(np.hypot(others[:, 0] - row, others[:, 1] - col))
        else:
            gap = np.inf
        out.append({
            'data': data, 'origin': (row - cut, col - cut), 'index': index,
            'peak': float(data.max()), 'flux': float(data.sum()),
            'background': background, 'noise': max(noise, 1e-9),
            'saturated': bool((data + background).max() >= saturation),
            'isolated': bool(gap >= isolation),
        })
    return out


# ------------------------------------------------------- moments and derived shape

def windowed_moments(data, sigma_window=2.5, iterations=6):
    """Gaussian-windowed first and second moments, iterated to the centroid.

    The window suppresses the corner pixels whose variance would otherwise dominate plain
    moments (the reason plain COM cannot reach the noise limit). The measured second
    moments are shrunk by the window; for a Gaussian source the true moment follows from
    M_true = M·σw²/(σw² − M), which is applied per eigenvalue. Good to a few percent for
    real (Moffat-ish) profiles, and the LS fit is the reference where it matters.

    Returns (cy, cx, m_yy, m_xx, m_xy) in cutout coordinates, or None if it diverged.
    """
    data = np.asarray(data, dtype=np.float64)
    size = data.shape[0]
    grid = np.arange(size, dtype=np.float64)
    xs, ys = np.meshgrid(grid, grid)
    cy = cx = (size - 1) / 2.0
    sw2 = float(sigma_window) ** 2
    for _ in range(iterations):
        weight = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sw2))
        w = data * weight
        total = w.sum()
        if not total > 0:
            return None
        cy_new = float((w * ys).sum() / total)
        cx_new = float((w * xs).sum() / total)
        if abs(cy_new - cy) < 1e-4 and abs(cx_new - cx) < 1e-4:
            cy, cx = cy_new, cx_new
            break
        cy, cx = cy_new, cx_new
        if not (0 <= cy < size and 0 <= cx < size):
            return None
    weight = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sw2))
    w = data * weight
    total = w.sum()
    if not total > 0:
        return None
    m_yy = float((w * (ys - cy) ** 2).sum() / total)
    m_xx = float((w * (xs - cx) ** 2).sum() / total)
    m_xy = float((w * (ys - cy) * (xs - cx)).sum() / total)

    # undo the window's shrinkage along each principal axis
    tensor = np.array([[m_yy, m_xy], [m_xy, m_xx]])
    values, vectors = np.linalg.eigh(tensor)
    corrected = []
    for value in values:
        if not 0 < value < sw2:            # window narrower than the source: no dice
            return None
        corrected.append(value * sw2 / (sw2 - value))
    tensor = vectors @ np.diag(corrected) @ vectors.T
    return cy, cx, float(tensor[0, 0]), float(tensor[1, 1]), float(tensor[0, 1])


def shape_from_moments(m_yy, m_xx, m_xy):
    """(fwhm, ellipticity, angle_rad, e1, e2) from second moments.

    fwhm is the geometric mean of the principal axes (the single 'size' number);
    ellipticity is 1 − b/a; (e1, e2) are the lensing-convention components, whose mean
    over stars is meaningful where a mean of angles is not.
    """
    tensor = np.array([[m_yy, m_xy], [m_xy, m_xx]])
    values = np.linalg.eigvalsh(tensor)
    if values[0] <= 0:
        return None
    sigma_minor, sigma_major = np.sqrt(values[0]), np.sqrt(values[1])
    fwhm = 2.3548 * float(np.sqrt(sigma_major * sigma_minor))
    ellipticity = float(1.0 - sigma_minor / sigma_major)
    angle = 0.5 * float(np.arctan2(2 * m_xy, m_xx - m_yy))
    denominator = m_xx + m_yy
    e1 = float((m_xx - m_yy) / denominator)
    e2 = float(2 * m_xy / denominator)
    return fwhm, ellipticity, angle, e1, e2


# ------------------------------------------------------------- model fits

def _oversampled_gaussian(params, size, oversample=3):
    """A pixel-integrated elliptical Gaussian, via oversampled evaluation.

    At FWHM ~2-3 px, evaluating the model at pixel centres instead of integrating over
    pixels puts a phase-dependent kink into the fit; 3x3 subsampling reduces that error to
    well below the noise on any real cutout.
    """
    amplitude, cy, cx, sigma_y, sigma_x, theta, offset = params
    step = 1.0 / oversample
    fine = np.arange(-0.5 + step / 2, size - 0.5, step)
    xs, ys = np.meshgrid(fine, fine)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (xs - cx) * ct + (ys - cy) * st
    yr = -(xs - cx) * st + (ys - cy) * ct
    model = amplitude * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    binned = model.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    return binned + offset


def fit_gaussian(data, noise=1.0):
    """Least-squares pixel-integrated elliptical Gaussian.

    Returns a dict (cy, cx, fwhm, ellipticity, angle, amplitude, rms) or None. The rms is
    per-pixel residual over noise, so >>1 flags a profile the Gaussian cannot describe
    (wings, saturation, a neighbour).
    """
    from scipy.optimize import least_squares

    data = np.asarray(data, dtype=np.float64)
    size = data.shape[0]
    start = windowed_moments(data)
    if start is None:
        return None
    cy, cx, m_yy, m_xx, m_xy = start
    shape = shape_from_moments(m_yy, m_xx, m_xy)
    if shape is None:
        return None
    fwhm0 = shape[0]
    sigma0 = max(fwhm0 / 2.3548, 0.6)
    params0 = [max(data.max(), 1.0), cy, cx, sigma0, sigma0, shape[2], 0.0]

    def residual(params):
        return (_oversampled_gaussian(params, size) - data).ravel()

    try:
        result = least_squares(
            residual, params0, method='lm', max_nfev=400)
    except Exception:
        return None
    if not result.success and result.status <= 0:
        return None
    amplitude, cy, cx, sigma_y, sigma_x, theta, _ = result.x
    sigma_y, sigma_x = abs(sigma_y), abs(sigma_x)
    if not (0.3 < sigma_x < size and 0.3 < sigma_y < size):
        return None
    major, minor = max(sigma_x, sigma_y), min(sigma_x, sigma_y)
    return {
        'cy': float(cy), 'cx': float(cx),
        'fwhm': 2.3548 * float(np.sqrt(sigma_x * sigma_y)),
        'fwhm_major': 2.3548 * float(major), 'fwhm_minor': 2.3548 * float(minor),
        'ellipticity': float(1.0 - minor / major),
        'angle': float(theta if sigma_x >= sigma_y else theta + np.pi / 2),
        'amplitude': float(amplitude),
        'rms': float(np.sqrt(np.mean(result.fun ** 2))) / max(noise, 1e-9),
    }


def fit_radial_moffat(radii, values):
    """Fit I(r) = I0·(1+(r/alpha)^2)^(-beta) to a radial profile. Returns (I0, alpha, beta,
    fwhm) or None. beta is the wing index: 4.765 for pure Kolmogorov seeing, lower when the
    optics add wings, infinity recovers the Gaussian."""
    from scipy.optimize import least_squares

    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values) & (values > 0)
    if keep.sum() < 6:
        return None
    radii, values = radii[keep], values[keep]
    i0 = float(values.max())
    half = radii[np.argmin(np.abs(values - i0 / 2))]

    def residual(params):
        log_i0, alpha, beta = params
        model = log_i0 - beta * np.log1p((radii / alpha) ** 2)
        return model - np.log(values)

    try:
        result = least_squares(residual, [np.log(i0), max(half, 0.7), 3.0],
                               bounds=([-np.inf, 0.2, 0.7], [np.inf, 50.0, 50.0]),
                               max_nfev=400)
    except Exception:
        return None
    log_i0, alpha, beta = result.x
    fwhm = 2.0 * alpha * np.sqrt(2 ** (1.0 / beta) - 1.0)
    return float(np.exp(log_i0)), float(alpha), float(beta), float(fwhm)


# -------------------------------------------------- the stacked high-SNR profile

def stacked_profile(image, stars, cut=CUT, top_n=40):
    """A high-SNR mean PSF from the brightest clean stars, recentred to subpixel accuracy.

    Shifting by the *fitted* centres before averaging is what makes the wings believable:
    averaging integer-aligned cutouts convolves the profile with the centroid scatter.
    Returns (profile, n_used); profile is None when too few clean bright stars exist.
    """
    from scipy.ndimage import shift as subpixel_shift

    bright = [s for s in stars if s.get('fit_rms') is not None]
    bright.sort(key=lambda s: -s['flux'])
    chosen = bright[:top_n]
    if len(chosen) < 5:
        return None, 0
    image = np.asarray(image)
    size = 2 * cut + 1
    accumulated = np.zeros((size, size))
    used = 0
    for star in chosen:
        row, col = int(round(star['y'])), int(round(star['x']))
        if not (cut <= row < image.shape[0] - cut and cut <= col < image.shape[1] - cut):
            continue
        cutout = np.asarray(
            image[row - cut:row + cut + 1, col - cut:col + cut + 1], dtype=float)
        ring = np.concatenate([cutout[0, :], cutout[-1, :], cutout[1:-1, 0],
                               cutout[1:-1, -1]])
        cutout = cutout - np.median(ring)
        total = cutout.sum()
        if total <= 0:
            continue
        recentred = subpixel_shift(cutout / total,
                                   ((row - star['y']), (col - star['x'])), order=3)
        accumulated += recentred
        used += 1
    return (accumulated / used if used else None), used


def radial_of(profile):
    """(radii, values) of a square profile about its centre, sorted by radius."""
    centre = (profile.shape[0] - 1) / 2
    ys, xs = np.indices(profile.shape)
    radii = np.hypot(ys - centre, xs - centre).ravel()
    values = profile.ravel()
    order = np.argsort(radii)
    return radii[order], values[order]


def event_payload(image, positions, platescale_arcsec=None, fit_top_n=80):
    """Everything the UI's PSF panel draws, as plain JSON-able columns.

    Per-star columns (position, FWHM, ellipticity components) let the frontend draw the
    field map and whiskers at any binning without another run; the binned radial profile
    plus the Moffat and Gaussian parameters draw the wings plot. Kept deliberately cheap:
    moments for every star, fits only for the brightest ``fit_top_n`` (which also feed the
    stacked profile). Returns None when there is nothing worth showing.
    """
    stars, summary = measure_field(image, positions,
                                   platescale_arcsec=platescale_arcsec,
                                   fit_top_n=fit_top_n)
    if not stars or summary['n_stars'] < 5:
        return None
    profile, n_stacked = stacked_profile(image, stars)
    moffat = None
    radial = None
    if profile is not None:
        radii, values = radial_of(profile)
        fit = fit_radial_moffat(radii, values)
        if fit:
            i0, alpha, beta, fwhm_moffat = fit
            moffat = {'alpha': alpha, 'beta': beta, 'fwhm': fwhm_moffat}
            summary['moffat_beta'] = beta
        # binned to ~80 points: the scatter cloud is the tool's job, not the UI's
        bins = np.linspace(0, radii.max(), 81)
        which = np.digitize(radii, bins)
        radial = {'r': [], 'value': []}
        for b in range(1, len(bins)):
            sel = which == b
            if sel.any():
                radial['r'].append(float(radii[sel].mean()))
                radial['value'].append(float(np.median(values[sel])))

    def col(name):
        return [round(float(s[name]), 4) for s in stars]

    return {
        'summary': summary,
        'n_stacked': n_stacked,
        'radial': radial,
        'moffat': moffat,
        'stars': {'x': col('x'), 'y': col('y'), 'fwhm': col('fwhm'),
                  'e1': col('e1'), 'e2': col('e2')},
        'image_size': [int(image.shape[0]), int(image.shape[1])],
    }


# ---------------------------------------------------------- the survey of a frame

def measure_field(image, positions, platescale_arcsec=None, max_stars=2000,
                  fit_top_n=None):
    """PSF measurements for every usable star on a frame.

    Returns (stars, summary): per-star dicts with position, flux, SNR, moments shape and
    Gaussian-fit shape; and a summary with the constant-PSF numbers the UI shows — median
    FWHM (px and arcsec), median ellipticity, the sampling verdict, and counts of what was
    excluded and why.

    ``fit_top_n`` bounds the least-squares fits to the N brightest stars, with windowed
    moments (a few ms each) covering the rest: a dense field can hold 1500+ usable stars,
    and 1500 fits is tens of seconds a live run should not pay when moments agree with the
    fits to a few percent. The exploration tool fits everything; stage 1 passes a cap.
    """
    cutouts = extract_cutouts(image, positions)
    stars = []
    excluded = {'saturated': 0, 'crowded': 0, 'failed': 0, 'edge':
                len(positions) - len(cutouts)}
    usable = []
    for cutout in cutouts[:max_stars]:
        if cutout['saturated']:
            excluded['saturated'] += 1
            continue
        if not cutout['isolated']:
            excluded['crowded'] += 1
            continue
        usable.append(cutout)
    fit_worthy = set()
    if fit_top_n is not None:
        by_flux = sorted(range(len(usable)), key=lambda i: -usable[i]['flux'])
        fit_worthy = set(by_flux[:fit_top_n])
    for rank, cutout in enumerate(usable):
        moments = windowed_moments(cutout['data'])
        if moments is None:
            excluded['failed'] += 1
            continue
        cy, cx, m_yy, m_xx, m_xy = moments
        shape = shape_from_moments(m_yy, m_xx, m_xy)
        if shape is None:
            excluded['failed'] += 1
            continue
        fwhm_m, ell_m, angle_m, e1, e2 = shape
        fit = (fit_gaussian(cutout['data'], noise=cutout['noise'])
               if fit_top_n is None or rank in fit_worthy else None)
        row0, col0 = cutout['origin']
        stars.append({
            'y': row0 + (fit['cy'] if fit else cy),
            'x': col0 + (fit['cx'] if fit else cx),
            'flux': cutout['flux'], 'peak': cutout['peak'],
            'snr': cutout['flux'] / (cutout['noise'] * (2 * CUT + 1)),
            'fwhm_moments': fwhm_m, 'ellipticity_moments': ell_m,
            'e1': e1, 'e2': e2,
            'fwhm': fit['fwhm'] if fit else fwhm_m,
            'ellipticity': fit['ellipticity'] if fit else ell_m,
            'angle': fit['angle'] if fit else angle_m,
            'fit_rms': fit['rms'] if fit else None,
        })

    summary = {'n_stars': len(stars), 'excluded': excluded,
               'platescale_arcsec': platescale_arcsec}
    if stars:
        # Summary numbers come from the *fitted* stars when enough exist: windowed
        # moments run ~15% wide on Moffat wings, so a median over a fit/moments mixture
        # would depend on where the fit cap happened to fall — measured as 2.93 px
        # against 2.41 px on the same frame. The fitted subset is the brightest stars,
        # which are also the best-measured ones.
        fitted = [s for s in stars if s['fit_rms'] is not None]
        basis = fitted if len(fitted) >= 20 else stars
        fwhms = np.array([s['fwhm'] for s in basis])
        ells = np.array([s['ellipticity'] for s in basis])
        summary['fwhm_px'] = float(np.median(fwhms))
        summary['fwhm_px_scatter'] = 1.4826 * float(
            np.median(np.abs(fwhms - np.median(fwhms))))
        summary['ellipticity'] = float(np.median(ells))
        summary['n_fitted'] = len(fitted)
        if platescale_arcsec:
            summary['fwhm_arcsec'] = summary['fwhm_px'] * platescale_arcsec
        # the number that decides which centroiding algorithm is honest (PSF_REVIEW.md §4)
        summary['undersampled'] = bool(summary['fwhm_px'] < 2.0)
    return stars, summary
