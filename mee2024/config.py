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
    # write the combined dark and flat alongside the results, so a master calibration
    # frame can be reused without keeping every original. Only written when two or more
    # frames were actually combined.
    'save_dark_flat': True,
    'sensitive_mode_stack': True,
    'workDir': '',
    'workDir2': '',
    '-DARK-': '',
    '-FLAT-': '',
    # a folder of master darks and flats built by `mee2024 calibrate`. When set, each field
    # is matched to the master with its own gain and exposure, per field, and told so --
    # which is what replaces picking darks and flats by hand for a folder run, where one
    # chosen set was applied to every field whether it fitted or not
    'calibration_library': '',
    # which frames of a sequence to use, as '50-172' (0-based, inclusive). Empty means all.
    # A capture rarely starts and stops on the science -- the Sun is there before totality
    # and back afterwards -- and this records the trim as a run parameter rather than as a
    # second copy of a 15 GB file
    'frame_range': '',
    # measure every frame's level before stacking, and say which range looks usable. Cheap:
    # a strip through each frame, about a second for 180 frames of a 22 GB container
    'scan_frames': True,
    # warn when a frame's brightness disagrees with the exposure its header states. Catches
    # capture software writing a new exposure into the header of a frame that still holds
    # the previous one, which nothing downstream can detect
    'check_exposures': True,
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
    # how far above the master dark's own noise a pixel must sit to be called hot and
    # excluded from the stack. The bulk of a dark is tight, so 20 sigma is far outside it
    # -- on the measured example it selects 299 pixels of 46.8 million. Raise it to keep
    # more, or set it very high to disable the exclusion entirely.
    'hot_pixel_sigmas': 20.0,
    # the floor under that cut, in ADU above the dark's own bias. A multiple of sigma is
    # not a fixed threshold -- the master is a mean of N darks, so its sigma falls as
    # 1/sqrt(N) and taking more darks silently masked more pixels. 10 ADU is what moves a
    # centroid enough to matter (~37 mas at 2 px from a 1000 ADU star). Set 0 for the old
    # pure-sigma behaviour
    'hot_pixel_min_adu': 10.0,
    # With no darks, find hot pixels from the dither instead: a star is fixed to the sky, a
    # hot pixel to the detector (docs/bench/HOTPIX.md -- 96.3% of the dark-confirmed hot
    # pixels, no false positives, about 4% of stage 1's runtime). Declines and says so when
    # the field barely moved between frames, since then the two are indistinguishable.
    'hot_pixel_dark_free': True,
    'blob_radius_extra': 100,  # delete pixels near saturated moon/sun region ('blob' mode only)
    'centroid_gap_blob': 30,  # ignore centroids within this distance of the painted mask
    # The shape of the Sun/Moon mask. 'disk' is a circle on the saturated core -- what the
    # geometry actually is. 'blob' is the pre-v1.4.0 convex hull, which followed streamers
    # into lobes that masked sky at some azimuths while leaving others at the core's edge;
    # kept only so an older reduction can be reproduced. Switching the default is
    # results-changing on any field with a saturated Sun or Moon (docs/ROADMAP.md F26).
    'eclipse_mask_mode': 'disk',  # 'disk' | 'blob'
    # which standard configuration produced a reduction: 'zenith', 'eclipse', or 'custom'
    # when the settings were assembled by hand. Written to results.txt so an archive says
    # which standard it was reduced under. See mee2024/field_presets.py.
    'field_preset': 'custom',
    # how far outside the measured saturated core the disk is painted. 10 px is Douglas'
    # 2026-08-31 choice, halved from 20 after a Bruns 2017 star at 1.49 R_sun cleared the
    # edge by only 11 px. Note the centre itself carries ~4-11 px of streamer bias, so a
    # star within ~15 px of the painted edge deserves a per-frame check.
    'eclipse_disk_margin_px': 10,
    # Subtract a heavily blurred copy of each frame, flattening the coronal gradient before
    # detection -- Bruns' 2017 method, and what makes stars inside ~2 R_sun measurable. Off
    # by default because it is results-changing and only wanted on eclipse fields.
    'coronal_subtract': False,
    'coronal_subtract_sigma_px': 10.0,
    'coronal_pedestal_adu': 2000.0,  # keeps the subtracted frame positive for integer output
    'centroid_gaussian_subtract': False,  # use the "sensitive mode" of custom centroid detection
    'centroid_gaussian_thresh': 5.0,  # threshhold for detecting centroids (sensitive mode)
    'min_area': 4,  # minimum area for found centroids (sensitive mode)
    # Re-centroid the stacked image under a fixed Gaussian window instead of over the
    # threshold-defined footprint, whose size scales with brightness. Off by default: it
    # moves measured numbers, so it needs its own validation (ROADMAP F15). Applies to the
    # stacked image only -- per-frame alignment is differential and integer-rounded.
    'centroid_refine_window': False,
    'centroid_window_sigma': 2.0,  # px; near the PSF sigma is about right
    # Reject stars whose peak reaches the sensor's full scale. Nothing else at any stage
    # tests a peak value -- `sanity_check_centroids` only checks the radial profile
    # decreases, which a flat-topped star passes -- so a clipped star is rejected only if
    # its position error happens to exceed `distortion_fit_tol`. On the eclipse field that
    # tolerance is 999 by design, so nothing stands in the way at all (ROADMAP F16).
    # On by default since 2026-08-28, with the validation this needed. Measured on the six
    # 08-12 zenith fields, stage 1 run once and stage 2 twice off the same centroids so the
    # rejection is the only variable: d(3000) moved -0.012 % against a 1.18 % field-to-field
    # scatter, and the plate scale 0.01 ppm. 23 clipped stars removed across six fields, 13
    # of them in the fit. tools/f16_zenith_ab.py; ROADMAP F16.
    # It matters where the tolerance is loose, not here: at step 3's tol 999 nothing else
    # rejects a clipped star at all.
    'reject_saturated_stars': True,
    'saturation_fraction': 0.95,  # of full scale; below 1 to catch the shoulder of a clip
    'sanity_check_centroids': True,
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
    # With the higher orders fixed from the reference files ('constant' above), fit the
    # plate scale from this field instead of importing the reference's. Off by default,
    # which is the published behaviour. Exists because an imported scale that is wrong --
    # Station 1 2024 refocused between calibration and eclipse, ~600 ppm -- puts 5-6" of
    # residual at the corners of a 9576 px frame and forces a 20" fit tolerance, and a
    # 20" gate admits mis-matches near the Sun (one at 1.87 R_sun, 15.9" off). With the
    # scale fitted the tolerance can be 2-3": the largest genuine deflection in a science
    # set is under 1". The deflection fit downstream refits the scale jointly with L, so a
    # scale fitted here without L absorbs nothing that fit does not give back
    # (docs/STEP3_2026.md, "The two-pass match"). Ignored unless the fixed order is
    # 'constant'.
    'distortion_free_scale': False,
    # A coarser gate for the FIRST fit, in arcseconds; 0 means "the same as
    # distortion_fit_tol", the published single-gate behaviour. The first fit is made
    # on everything the rough match admitted, and a handful of wrong assignments
    # inside the rough threshold can pull it far enough that a tight gate applied to
    # ITS residuals throws out the good stars with the bad: on the Station 1 2024
    # 0.4 s block the pre-outlier rms was 13", and a 3" gate there kept 27 stars of
    # 189. With this set (20" there) the first gate removes the gross mis-matches, the
    # refit is sound, and distortion_fit_tol is then applied to the refitted
    # residuals and refitted once more (docs/STEP3_2026.md, "The two-pass match").
    'distortion_fit_tol_initial': 0.0,
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
    # drop stars with a close companion from the distortion fit: the companion pulls
    # the measured centroid away from the catalogue position. On by default -- a blended
    # pair is a systematically wrong position, not a noisy one, so keeping it costs more
    # than the star is worth. Matches the app's default, so the CLI and the window do
    # not quietly disagree about what a fit of the same frames means
    'remove_double_tab2': True,
    # drop stars the catalogue has no proper motion for: they cannot be propagated to
    # the observation epoch, so their position carries the catalogue epoch's error.
    # Off by default: the motion is now borrowed from Hipparcos where Gaia lacks it, and
    # the stars this would drop are disproportionately the bright ones worth keeping
    'remove_missing_pm': False,
    # having offered to tidy away the superseded g12 archives once, do not keep asking
    'catalogue_cleanup_dismissed': False,
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
    # the app window's last processing preset ('auto', 'quick', 'deep'), so a new
    # session starts where the last one left off
    'ui_preset': 'auto',
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
