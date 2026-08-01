"""
Candidate-solution verification for the v2 solver.

A faithful port of v1's ``match_centroids`` (mutual nearest neighbour with a 2x
confusion margin, local catalogue density for the acceptance threshold) with one
deliberate change: the comparison catalogue is **the one named in the pattern
database's manifest** -- the same Gaia archive the patterns were built from -- rather
than the bundled Tycho npz that v1 opens on its own. Verifying a Gaia-built solve
against Tycho would leave Tycho's ~2.5 arcsec propagated errors inside the acceptance
statistics, which is exactly what the rebuild removes.
"""

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors

from mee2024 import transforms
from mee2024.MEE2024util import get_bbox
from mee2024.platesolve_triangle import estimate_acceptance_threshold  # noqa: F401  (re-exported for solve.py)


#: one provider per release set for the process lifetime (mmap handles are cheap but
#: manifest reads and object setup are not free at one-per-candidate rates)
_CATALOGUE_CACHE = {}


def open_verify_catalogue(verify_spec):
    """The provider named by a pattern DB manifest. Tests inject their own instead."""
    from mee2024.starcat import providers
    if verify_spec.get('provider') != 'gaia_offline':
        raise ValueError(f"unknown verification provider in manifest: "
                         f"{verify_spec.get('provider')!r}")
    key = tuple(verify_spec.get('releases') or [])
    if key not in _CATALOGUE_CACHE:
        try:
            provider = providers.GaiaOfflineProvider.from_installed(list(key) or None)
        except Exception:
            # The archive this database was built against may have been merged into a
            # deeper one (`mee2024 catalogue --merge`) and removed. Verifying against
            # whatever is installed is correct as long as it covers the same stars,
            # and a deeper archive strictly does.
            provider = providers.GaiaOfflineProvider.from_installed()
            print(f'note: {" + ".join(key) or "the named archive"} is no longer '
                  f'installed; verifying against {provider.describe()} instead')
        _CATALOGUE_CACHE[key] = provider
    return _CATALOGUE_CACHE[key]


def catalogue_size(catalogue, mag_limit):
    """How many stars the verification catalogue holds to the given depth.

    Feeds the acceptance estimator's hypothesis count. For the offline archives every
    stored star is within the limit, so the manifest count is exact; any other object
    just needs to answer len().
    """
    if hasattr(catalogue, 'catalogues'):
        return int(sum(len(c) for c in catalogue.catalogues))
    return int(len(catalogue))


def _bbox_solid_angle(ra_range, dec_range):
    """Solid angle of a (possibly RA-wrapping) bounding box, in steradians."""
    dra = (ra_range[1] - ra_range[0]) % 360 or 360
    sin_hi = math.sin(math.radians(max(dec_range)))
    sin_lo = math.sin(math.radians(min(dec_range)))
    return math.radians(dra) * (sin_hi - sin_lo)


def match_centroids(centroids, platescale_fit, image_size, options, catalogue,
                    mag_limit, epoch, adapt_depth=False):
    """Mutually match observed centroids against catalogue stars for one candidate.

    Returns (stardata, matched_plate, max_error, local_density, ids) with ``stardata``
    in v1's (n, 6) layout: [ra_rad, dec_rad, vx, vy, vz, mag] and ``ids`` the catalogue
    identifier of each matched star, which is what lets the UI label them by name or
    HIP number. ``adapt_depth`` enables
    the S3 detection-count-aware comparison set; the S1 ratio path keeps the full
    depth so its behaviour stays frozen.
    """
    corners = transforms.to_polar(transforms.linear_transform(
        platescale_fit,
        np.array([[0, 0],
                  [image_size[0] - 1., image_size[1] - 1.],
                  [0, image_size[1] - 1.],
                  [image_size[0] - 1., 0]])
        - np.array([image_size[0] / 2, image_size[1] / 2])))
    bbox = get_bbox(corners)
    # A corner-derived box cannot describe a field that reaches the celestial pole:
    # the field's declination extreme is the pole itself, between corners, and its
    # corner RAs are arbitrary -- the box then silently excludes the very stars
    # nearest the pole, and verification starves. (v1 shares this defect; it was
    # invisible while no solver brought polar fields as far as verification.)
    half_diag_deg = np.degrees(platescale_fit[0]) * np.hypot(*image_size) / 2
    boresight_dec_deg = np.degrees(platescale_fit[2])
    if 90.0 - abs(boresight_dec_deg) < half_diag_deg * 1.1:
        if boresight_dec_deg >= 0:
            bbox = ((0.0, 360.0), (min(bbox[1]), 90.0))
        else:
            bbox = ((0.0, 360.0), (-90.0, max(bbox[1])))
    table = catalogue.lookup(bbox[0], bbox[1], mag_limit, epoch=epoch)
    stardata = np.zeros((len(table), 6))
    ids = np.zeros(len(table), dtype=np.int64)
    if len(table):
        stardata[:, 0] = table.ra
        stardata[:, 1] = table.dec
        stardata[:, 2:5] = table.get_vectors()
        stardata[:, 5] = table.get_mags()
        ids = np.asarray(table.get_ids(), dtype=np.int64)

    # Verification depth tracks the detection count (S3): the detections are the
    # field's brightest stars, so catalogue stars much fainter than that population
    # can never be matched -- they only add confusers and inflate the local density
    # (which sets the acceptance threshold). A 10-star field verified against the
    # full G<12 set needs a threshold no 10 detections can reach. The factor of 8
    # covers the bounding box exceeding the frame (~2x in area) plus detection
    # ordering scatter, so every plausible counterpart stays in the comparison set;
    # for a normal ~100-detection field the cap exceeds the bbox count and nothing
    # changes.
    if adapt_depth:
        depth_cap = max(8 * len(centroids), 60)
        if stardata.shape[0] > depth_cap:
            keep_brightest = np.argsort(stardata[:, 5], kind='stable')[:depth_cap]
            stardata = stardata[keep_brightest]
            ids = ids[keep_brightest]

    # local catalogue density of the comparison set actually used: the false-match
    # rate scales with it, and the galactic plane runs 3-10x the all-sky mean, so
    # the acceptance threshold must know it
    local_density = stardata.shape[0] / max(_bbox_solid_angle(*bbox), 1e-12)

    match_threshhold = np.radians(options['rough_match_threshhold'] / 3600)
    if stardata.shape[0] < 2:
        # nothing (or one star) to match against: no verification is possible
        return (stardata[:0], np.zeros((0, 2)), match_threshhold, local_density,
                ids[:0])

    all_star_plate = centroids - np.array([image_size[0] / 2, image_size[1] / 2])
    all_vectors = transforms.linear_transform(platescale_fit, all_star_plate)

    candidate_star_vectors = stardata[:, 2:5]
    # nearest two catalogue stars to each observed star (3-vector metric)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(candidate_star_vectors)
    distances, indices = neigh.kneighbors(all_vectors)
    # nearest observed star to each catalogue star, for the reflexivity check
    neigh_bar = NearestNeighbors(n_neighbors=1)
    neigh_bar.fit(all_vectors)
    distances_bar, indices_bar = neigh_bar.kneighbors(candidate_star_vectors)

    confusion_ratio = 2  # closest match must be 2x closer than second place
    keep = np.logical_and(distances[:, 0] < match_threshhold,
                          distances[:, 1] / distances[:, 0] > confusion_ratio)
    # is the nearest-neighbour relation reflexive? [eliminates 1-to-many matching]
    keep = np.logical_and(
        keep, indices_bar[indices[:, 0]].flatten() == np.arange(indices.shape[0]))
    keep_i = np.nonzero(keep)

    chosen = indices[keep_i, 0].flatten()
    stardata = stardata[chosen, :]
    ids = ids[chosen]
    plate2 = all_star_plate[keep_i, :][0]
    matched_vectors = all_vectors[keep_i, :][0]
    errors = np.linalg.norm(stardata[:, 2:5] - matched_vectors, axis=1)
    max_error = np.max(errors) if errors.size else match_threshhold
    return stardata, plate2, max_error, local_density, ids
