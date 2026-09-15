"""The tangent-plane export, and the gauge it exists to make explicit.

Douglas, 2026-09-15: "perfect optics with no distortion should be perfectly flat. Does that
occur now?" -- it does not, and these tests pin the size of what a flawless telescope draws in
MEE's frame, because that number is what makes the export worth having.

The control matters as much as the measurement: a true ARC projection against the tangent plane
must give the textbook tan(t) - t = 0.3655 "/deg^3. It is here so that if the transform code
ever changes, the failure shows up as a broken control rather than as a plausible wrong number.
MEE's own frame is NOT ARC, which is why it gives ~0.434 instead.
"""
import json

import numpy as np
import pytest

from mee2024 import distortion_polynomial as dp
from mee2024 import transforms

ARCSEC = 206264.806247
#: (label, ny, nx, arcsec/pixel) -- the three instruments in the matrix
GEOMETRY = [('bruns2017', 2472, 3296, 2.0868),
            ('mexico2024_station2', 3520, 4656, 1.8673),
            ('husillos2026', 6388, 9576, 2.2028)]


def opts(order='cubic'):
    return {'distortionOrder': order, 'distortion_fixed_coefficients': 'None'}


def test_arc_against_tangent_is_the_textbook_term():
    """The control. A projection whose radius IS the angle differs from the tangent plane by
    exactly tan(t) - t; if this drifts, the machinery below is not measuring what it claims."""
    t = np.radians(np.linspace(0.05, 1.2, 300))
    k = np.polyfit(np.degrees(t) ** 3, (np.tan(t) - t) * ARCSEC, 1)[0]
    assert k == pytest.approx(0.3655, abs=0.001)


def test_mee_frame_is_not_an_arc_projection():
    """MEE's frame is (dec, RA*cos(dec)). Its departure from the tangent plane is ~0.434
    "/deg^3, not the 0.3655 an ARC projection would give -- the correction of 2026-09-15."""
    ny, nx, ps = 2472, 3296, 2.0868
    scale = np.radians(ps / 3600.0)
    zero = [0.0] * len(dp.get_coeff_names(opts()))
    out = dp.tangent_plane_coefficients((scale, 0, 0, 0), zero, zero, (ny, nx), opts())
    k = out['gauge term radial cubic (arcsec/deg^3)']
    assert 0.40 < k < 0.47, k
    assert abs(k - 0.3655) > 0.05, 'the ARC value must not be what this frame gives'


@pytest.mark.parametrize('label,ny,nx,ps', GEOMETRY)
def test_a_perfect_optic_is_not_flat(label, ny, nx, ps):
    """With no distortion of its own, the field MEE would still draw. Grows as the cube of the
    field radius, which is why it is negligible on Bruns and 8+ pixels on Husillos."""
    scale = np.radians(ps / 3600.0)
    order = 'quintic' if label == 'husillos2026' else 'cubic'
    zero = [0.0] * len(dp.get_coeff_names(opts(order)))
    out = dp.tangent_plane_coefficients((scale, 0, 0, 0), zero, zero, (ny, nx), opts(order))
    corner = out['gauge term at the corner (pixels)']
    expect = {'bruns2017': 0.37, 'mexico2024_station2': 0.84, 'husillos2026': 8.79}[label]
    assert corner == pytest.approx(expect, rel=0.05)
    # a polynomial of the fitted order represents the exact transform essentially perfectly
    assert out['refit residual rms (pixels)'] < 0.01


def test_export_is_serialisable_and_marked():
    scale = np.radians(2.0868 / 3600.0)
    zero = [0.0] * len(dp.get_coeff_names(opts()))
    out = dp.tangent_plane_coefficients((scale, 0, 0, 0), zero, zero, (2472, 3296), opts())
    assert out['gauge'] == dp.TAN_GAUGE_MARK
    json.dumps(out)                       # must survive the write in distortion_fitter
    assert set(out['distortion coeffs x']) == set(dp.get_coeff_names(opts()))


def test_a_tan_export_is_refused_as_a_reference(tmp_path):
    """The one way the gauge bites a measurement: a TAN coefficient frozen into a fit."""
    scale = np.radians(2.0868 / 3600.0)
    zero = [0.0] * len(dp.get_coeff_names(opts()))
    out = dp.tangent_plane_coefficients((scale, 0, 0, 0), zero, zero, (2472, 3296), opts())
    p = tmp_path / 'distortion_results_TAN.txt'
    p.write_text(json.dumps(out), encoding='utf-8')
    with pytest.raises(ValueError, match='TANGENT-PLANE'):
        dp._open_distortion_files({'distortion_reference_files': str(p),
                                   'distortionOrder': 'cubic',
                                   'distortion_fixed_coefficients': 'None'})


def test_round_trip_a_known_field():
    """Give MEE a distortion that is exactly the gauge term with the sign flipped -- i.e. a
    telescope whose optics happen to cancel the projection -- and the tangent-plane export must
    come back flat. It is the same arithmetic run backwards, and it closes the loop."""
    ny, nx, ps = 2472, 3296, 2.0868
    scale = np.radians(ps / 3600.0)
    o = opts()
    zero = [0.0] * len(dp.get_coeff_names(o))
    gauge = dp.tangent_plane_coefficients((scale, 0, 0, 0), zero, zero, (ny, nx), o)
    names = dp.get_coeff_names(o)
    cx = [-gauge['distortion coeffs x'][n] for n in names]
    cy = [-gauge['distortion coeffs y'][n] for n in names]
    out = dp.tangent_plane_coefficients((scale, 0, 0, 0), cx, cy, (ny, nx), o)
    assert out['gauge term at the corner (pixels)'] == pytest.approx(
        gauge['gauge term at the corner (pixels)'], rel=0.05)
    flat = max(abs(v) for n, v in out['distortion coeffs x'].items() if n != '1')
    assert flat < 0.01, 'optics cancelling the gauge must export as flat'
