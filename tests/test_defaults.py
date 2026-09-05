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
    """2 px against PSF sigmas of 2.5-3.5 px on the matrix's instruments. The window must be
    narrower than the PSF: wider, it weights the background gradient around the star -- on
    Bruns' 0.7 px PSF a 2 px window took the residual near the Sun from 0.15" to 0.71" and L
    from 1.78 to 1.28. Stage 1 warns when the measured PSF is not wider than the window."""
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


def test_stage_one_warns_when_the_window_is_not_narrower_than_the_psf():
    """The condition on the windowed default, found on Bruns' field: a 2 px window on a 0.7 px
    PSF (FWHM 1.65) fails near the Sun; on Station 1's 7.5 px FWHM it is fine."""
    from mee2024.stacker_implementation import window_wider_than_psf
    o = get_default_options()
    assert window_wider_than_psf(o, 7.5) is None
    msg = window_wider_than_psf(o, 1.65)
    assert msg and 'not narrower' in msg and 'eclipse preset' in msg
    assert window_wider_than_psf(dict(o, centroid_window_sigma=0.5), 1.65) is None
    assert window_wider_than_psf(dict(o, centroid_refine_window=False), 1.65) is None
    assert window_wider_than_psf(o, None) is None
