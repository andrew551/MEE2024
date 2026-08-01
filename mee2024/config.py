"""
Default configuration for MEE2024.

These defaults used to live in ``main.py``, which meant that anything wanting to know
them -- the CLI, the tests -- had to import ``main``, pulling in FreeSimpleGUI and
starting the triangle-database subprocess as a side effect. They live here instead so
that ``options`` can be obtained without touching the GUI or spawning anything.
"""

from copy import deepcopy

from mee2024 import MEE2024util

# default values for all options
DEFAULT_OPTIONS = {
    'flag_display': True,
    'flag_display2': True,
    'flag_debug': False,
    'save_dark_flat': False,
    'sensitive_mode_stack': True,
    'workDir': '',
    'workDir2': '',
    '-DARK-': '',
    '-FLAT-': '',
    'output_dir': '',
    # 'gaia' is the installed offline archive plus the bright fill Gaia itself lacks,
    # and falls back to the online archive only until an archive is installed.
    # 'gaia_online' queries the ESA archive every run: minutes per field
    'catalogue': 'gaia',
    'm': 30,
    'n': 30,
    'd': 100,  # how many stacked found stars to display
    'img_edge_distance': 5,  # how many pixels away from edge
    'pxl_tol': 10,  # for stacking centroid matching
    'cutoff': 100,  # for stacking centroid matching, penalty saturation distance
    'delete_saturated_blob': True,
    'blob_saturation_level': 100,
    'blob_radius_extra': 100,  # delete pixels near saturated moon/sun region
    'centroid_gap_blob': 30,  # ignore centroids within this distance of saturated region + radius_extra
    'centroid_gaussian_subtract': False,  # use the "sensitive mode" of custom centroid detection
    'centroid_gaussian_thresh': 5.0,  # threshhold for detecting centroids (sensitive mode)
    'min_area': 4,  # minimum area for found centroids (sensitive mode)
    'sanity_check_centroids': True,
    'float_fits': False,  # output fits files with float type
    'max_star_mag_dist': 12.0,
    'observation_date': '2023-12-01',
    'distortion_fit_tol': 1.0,  # arcseconds tolerance
    'remove_edgy_centroids': True,
    'sigma_subtract': 3.0,
    'distortionOrder': 'cubic',
    'guess_date': False,
    'DEFAULT_DATE': '2020-01-01',  # the default date for date guessing
    'double_star_cutoff': 10.0,  # within how many arcseconds to consider near_neighbour
    'double_star_mag': 17.0,  # max mag of double stars
    'rough_match_threshhold': 36.0,  # (in arcsec) (0.01 degrees)
    'enable_corrections': False,
    'observation_time': '',
    'observation_lat': '',
    'observation_long': '',
    'enable_corrections_ref': False,
    'enable_gravitational_def': False,
    'observation_temp': 10.0,
    'observation_pressure': 1010.0,
    'observation_humidity': 0.0,
    'observation_height': 0.0,
    'observation_wavelength': 0.65,
    'distortion_reference_files': '',
    'distortion_fixed_coefficients': 'None',
    'flag_display3': True,
    'background_subtraction_mode': 'annular',
    'eclipse_limiting_mag': 11.0,
    'remove_double_stars_eclipse': False,
    'safety_limit_mag': 13.0,
    'object_centre_moon': False,
    'gravity_sweep': False,
    'limit_radial_sun_radii': False,
    'limit_radial_sun_radii_value': 9.0,
    'crop_circle': False,
    'crop_circle_thresh': 1.0,
    'remove_double_tab2': False,
    'eclipse_method': 'Method 1 & 2',
    # draw the fitted distortion field (arrows + magnitude map) and emit it as an event
    'distortion_field_plot': True,
    # --- watch mode: process frames as they land in a folder -------------------
    'watch_folder': '',
    # a frame is only read once its last modification is this many seconds old, so a
    # file still being written by the capture software is never opened half-complete
    'watch_settle_seconds': 10.0,
    # run once this many settled frames have accumulated
    'watch_batch_size': 5,
    # ...or sooner, if nothing new has arrived for this long and at least 2 frames are held
    'watch_quiet_seconds': 60.0,
    'watch_poll_seconds': 2.0,
    # where to download prebuilt catalogues from, as {name: {url, sha256, size_bytes}}.
    # Overrides the published URLs; set with `mee2024 catalogue --set-source`, so moving
    # from GitHub releases to Zenodo is a config change, not a code change.
    'catalogue_sources': {},
    # fetch a missing offline catalogue when a run needs one, rather than failing. Set
    # False on a metered connection: the base archive is a 138 MB download.
    'auto_download_catalogue': True,
    # bins per axis for the spatially resolved residual-correlation map.
    # 0 chooses from the star count, aiming at ~8 stars per cell; set a number to force it
    'residual_bins': 0,
    # which interface a no-argument launch (or a double-clicked .exe) opens:
    # 'app' for the new window, 'classic' for the original FreeSimpleGUI one
    'default_interface': 'app',
    # which plate solver runs. 'v2' is the Gaia/Kendall rebuild (mee2024/platesolve2;
    # measured record in docs/bench/BENCH.md) and falls back to 'triangle' -- the
    # classic Tycho solver -- automatically when its pattern database or catalogue
    # is not installed
    'platesolver': 'v2',
    # which pattern database the v2 solver reads; '' auto-selects the best installed
    'pattern_db': '',
    # 1-sigma centroid noise the v2 solver assumes when sizing its match radius, in
    # pixels. 0.3 suits stacked images; failed solves escalate this automatically,
    # so single noisy frames cost a retry rather than a permanently wider search
    'platesolve_noise_px': 0.3,
}


def get_default_options():
    """A fresh copy of the defaults. Never hand out the module-level dict itself."""
    return deepcopy(DEFAULT_OPTIONS)


def load_options(config_path=None):
    """Defaults with the user's saved config merged over the top."""
    options = get_default_options()
    MEE2024util.read_ini(options, path=config_path)
    return options
