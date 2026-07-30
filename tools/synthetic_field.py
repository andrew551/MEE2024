"""
Synthesize realistic star-field centroid lists from the offline Gaia catalogue.

A synthetic field is the ground-truth instrument for the plate solver: pointing, roll,
plate scale, distortion and noise are all known exactly, so solve success, wrongness and
timing can be measured over any part of parameter space -- FOV sweeps, noise sweeps,
sparse fields, dense galactic-plane fields, or pure-junk fields that must be rejected.

The forward model mirrors the real pipeline in reverse:

    catalogue stars -> gnomonic projection at (ra, dec, roll, platescale)
                    -> polynomial optical distortion (same basis the fitter uses)
                    -> detection incompleteness + magnitude-ordering scatter
                    -> Gaussian centroid noise
                    -> brightest-first centroid list, exactly what platesolve() expects
"""

import numpy as np

from mee2024 import transforms


def synthesize_field(catalogue, ra_deg, dec_deg, roll_deg, fov_width_deg,
                     shape=(2000, 3000), mag_limit=12.0, epoch=2023.84,
                     noise_px=0.3, distortion_px=3.0, n_detect=120,
                     mag_order_scatter=0.3, dropout=0.05, seed=0):
    """Returns (centroids, truth). centroids is an (n, 2) array of (y, x), brightest first.

    catalogue: anything with .lookup(ra_range, dec_range, max_magnitude, epoch) -> StarTable
    fov_width_deg: the width of the long axis (shape[1]) in degrees
    noise_px: 1-sigma Gaussian centroid error per axis, in pixels
    distortion_px: approximate size of the cubic distortion at the field edge, in pixels
    n_detect: how many of the brightest stars are 'detected'
    mag_order_scatter: magnitude noise applied before brightness ordering, so the
        detected-brightest list is realistically imperfect
    dropout: fraction of stars randomly not detected
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    platescale_rad = np.radians(fov_width_deg) / width
    x = (platescale_rad, np.radians(ra_deg), np.radians(dec_deg), np.radians(roll_deg))

    # query a bounding box comfortably containing the field
    radius_deg = fov_width_deg * np.hypot(height, width) / width / 2 * 1.2
    cos_dec = max(np.cos(np.radians(dec_deg)), 0.05)
    ra_lo = (ra_deg - radius_deg / cos_dec) % 360
    ra_hi = (ra_deg + radius_deg / cos_dec) % 360
    dec_lo = max(dec_deg - radius_deg, -89.99)
    dec_hi = min(dec_deg + radius_deg, 89.99)
    stars = catalogue.lookup((ra_lo, ra_hi), (dec_lo, dec_hi), mag_limit, epoch)

    # gnomonic projection to centred (y, x) pixel coordinates
    plate = transforms.detransform_vectors(x, stars.get_vectors())

    inside = ((np.abs(plate[:, 0]) < height / 2 - 2) &
              (np.abs(plate[:, 1]) < width / 2 - 2))
    plate, mags = plate[inside], stars.get_mags()[inside]

    # cubic optical distortion, same functional form the fitter models
    w = max(shape) / 2
    yn, xn = plate[:, 0] / w, plate[:, 1] / w
    c = rng.uniform(-1, 1, size=6)
    dx = distortion_px * (c[0] * xn**3 + c[1] * xn * yn**2 + c[2] * xn**2 * yn)
    dy = distortion_px * (c[3] * yn**3 + c[4] * yn * xn**2 + c[5] * yn**2 * xn)
    plate = plate + np.c_[dy, dx]

    # detection: imperfect brightness ordering, random dropouts, centroid noise
    keep = rng.random(len(plate)) > dropout
    plate, mags = plate[keep], mags[keep]
    order = np.argsort(mags + rng.normal(0, mag_order_scatter, len(mags)))
    plate = plate[order][:n_detect]
    plate = plate + rng.normal(0, noise_px, plate.shape)

    centroids = plate + np.array([height / 2, width / 2])
    truth = {'ra': ra_deg, 'dec': dec_deg, 'roll': roll_deg,
             'platescale_arcsec': np.degrees(platescale_rad) * 3600,
             'fov_width_deg': fov_width_deg, 'n_stars': len(centroids),
             'n_in_field': int(np.sum(inside))}
    return centroids, truth


def junk_field(shape=(2000, 3000), n=120, seed=0):
    """Uniform random centroids: contains no sky. A solver must reject it."""
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(2, shape[0] - 2, n), rng.uniform(2, shape[1] - 2, n)]


def solution_matches_truth(result, truth, pos_tol_deg=0.05, scale_rtol=0.02):
    """Did the solver find the right answer? (roll is checked via scale+position,
    because the pipeline carries two internal +90/+180 roll conventions.)"""
    if not result.get('success'):
        return False
    cos_dec = np.cos(np.radians(truth['dec']))
    dra = abs((result['ra'] - truth['ra'] + 180) % 360 - 180) * cos_dec
    ddec = abs(result['dec'] - truth['dec'])
    dscale = abs(result['platescale/arcsec'] - truth['platescale_arcsec'])
    return (np.hypot(dra, ddec) < pos_tol_deg
            and dscale < scale_rtol * truth['platescale_arcsec'])
