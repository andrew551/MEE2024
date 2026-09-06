"""Stage 3: the deflection fit.

There were no tests here at all, which is how a units error in the Method 1 covariance
survived. These pin the piece that was wrong plus the invariant that would have caught it.
"""

import numpy as np
import pytest

from mee2024.eclipse_analysis import plate_scale_covariance

# Leon geometry: ASI2600MM at 350 mm, and the solar radius at that eclipse.
SUN_RADIUS_DEG = 961.0 / 3600
LEON_PLATESCALE = 2.216


def leon_annulus(n=400, rmin=2.0, rmax=9.0, seed=1):
    """Star radii in solar radii, the range section 17.2 says is usable."""
    return np.random.default_rng(seed).uniform(rmin, rmax, n)


def test_plate_scale_covariance_does_not_depend_on_the_pixel_scale():
    """THE INVARIANT THAT WOULD HAVE CAUGHT THE BUG.

    `rad_dist` is in solar radii and the answer is in arcseconds, so the same field
    expressed at a different plate scale is the same angular measurement and must cost L
    the same. The previous implementation scaled with 1/platescale0 and therefore claimed
    that photographing the identical sky through a different focal length changed how much
    a 10 ppm scale error hurt.
    """
    rad_dist = leon_annulus()
    a = plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG)
    b = plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG)
    assert a == b
    # the signature no longer accepts a plate scale at all -- that is the fix
    with pytest.raises(TypeError):
        plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG, LEON_PLATESCALE)


def test_plate_scale_covariance_matches_the_closed_form():
    rad_dist = leon_annulus()
    got = plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG)
    lever = np.dot(1 / rad_dist, rad_dist) / np.dot(1 / rad_dist, 1 / rad_dist)
    assert got == pytest.approx(10e-6 * 961.0 * lever)


def test_plate_scale_covariance_is_linear_in_the_uncertainty():
    rad_dist = leon_annulus()
    one = plate_scale_covariance(rad_dist, 1e-6, SUN_RADIUS_DEG)
    ten = plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG)
    assert ten == pytest.approx(10 * one)


def test_ten_ppm_costs_about_ten_per_cent_of_L():
    """The sensitivity, pinned. This is the number that makes the term matter.

    L = 1.7512 arcsec at the limb, and a 10 ppm plate-scale error puts about a tenth of it
    onto the fitted value -- because at 9 solar radii the spurious displacement is 86 mas
    against 195 mas of real deflection.

    The exact figure depends on how the stars are distributed in radius, so the two in
    circulation are not in conflict: **9.9 % for stars uniform in r**, which is what this
    test constructs, and **7.7-8.4 % for stars uniform on the sensor**, which is the real
    eclipse-field case (`tools/cubic_into_deflection.py`, and Andrew's independent
    +7.7 %/10 ppm). A narrower annulus gives more, not less.
    """
    dL = plate_scale_covariance(leon_annulus(), 10e-6, SUN_RADIUS_DEG)
    assert dL / 1.7512 == pytest.approx(0.099, abs=0.004)


def test_the_old_expression_understated_it_by_the_plate_scale():
    """PINS THE DEFECT ITSELF, so a revert cannot pass silently.

    The old line multiplied by `3600 / platescale0 * sun_apparent_angular_radius` -- the
    solar radius in pixels. The ratio to the correct value is exactly the plate scale.
    """
    rad_dist = leon_annulus()
    lever = np.dot(1 / rad_dist, rad_dist) / np.dot(1 / rad_dist, 1 / rad_dist)
    old = lever * 10e-6 * (3600 / LEON_PLATESCALE * SUN_RADIUS_DEG)
    new = plate_scale_covariance(rad_dist, 10e-6, SUN_RADIUS_DEG)
    assert new / old == pytest.approx(LEON_PLATESCALE, rel=1e-9)


def test_a_narrow_annulus_costs_more_than_a_wide_one():
    """Sanity on the lever arm: less radial range, less separation from the 1/r signal."""
    wide = plate_scale_covariance(leon_annulus(rmin=2.0, rmax=9.0), 10e-6, SUN_RADIUS_DEG)
    narrow = plate_scale_covariance(leon_annulus(rmin=5.0, rmax=10.0), 10e-6, SUN_RADIUS_DEG)
    assert narrow > wide
