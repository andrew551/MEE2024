"""The reduction defaults set on 2026-09-05, and the way back from them.

Windowed centroids, a 0.2" calibration-fit tolerance and a cubic are the defaults because of
what three instruments measured (docs/STEP3_2026.md, "The PSF on three instruments"): coma
points outward on one reducer system and inward on another, the moment estimator's radial
offset then depends on magnitude and leaks into L, and the windowed estimator removes the
magnitude dependence whichever way the tail points while costing nothing on a clean optic.
These defaults move measured numbers, so they are pinned here with their reasons -- and so is
the route back, because the Bruns 2017 reproduction (footprint moments, Gaussian background)
must stay one step away: the 'eclipse' field preset is that convention, and --set recovers any
single value.
"""
from mee2024 import cli
from mee2024 import field_presets as fp
from mee2024.config import get_default_options


def test_windowed_centroids_are_the_default():
    assert get_default_options()['centroid_refine_window'] is True


def test_the_window_is_narrower_than_the_psfs_it_is_meant_for():
    """2 px against PSF sigmas of 2.5-3.5 px on the matrix's instruments; a window wider than
    the PSF is flat over it and degenerates to the moment estimator."""
    assert get_default_options()['centroid_window_sigma'] == 2.0


def test_the_calibration_fit_tolerance_is_a_fifth_of_an_arcsecond():
    assert get_default_options()['distortion_fit_tol'] == 0.2


def test_a_cubic_is_the_default_order():
    assert get_default_options()['distortionOrder'] == 'cubic'


def test_the_background_default_is_annular():
    assert get_default_options()['background_subtraction_mode'] == 'annular'


def test_the_bruns_convention_is_one_preset_away():
    """The way back to the estimator that reproduced Bruns 2018 end to end."""
    o = fp.apply_field_preset(get_default_options(), 'eclipse')
    assert o['centroid_refine_window'] is False
    assert o['background_subtraction_mode'] == 'Gaussian'


def test_the_previous_defaults_are_one_set_away():
    o = cli.apply_sets(get_default_options(), ['centroid_refine_window=False', 'distortion_fit_tol=1.0'])
    assert o['centroid_refine_window'] is False
    assert o['distortion_fit_tol'] == 1.0
