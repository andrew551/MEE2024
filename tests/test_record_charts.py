"""The shared chart constructions: the arithmetic pinned, the conventions enforced, one copy.

tools/record_charts.py exists because four chart tools carried four copies of the same
constructions and diverged four ways in one week (its docstring has the list). These tests pin
what the copies got wrong.
"""
import os

import numpy as np
import pytest

from tools.record_charts import (SkyFrame, covariance_chart, draw_ellipse, field_chart,
                                 joint_plate_scale, ppm_from)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cell 2's four blocks as its record summary stores them: stage-2 scale, the S the pooled
# fit found (ppm), and the joint scale the record quotes (arcsec/px, PS = 1.84847).
CELL2_BLOCKS = [
    (1.8472819, -45.6, 1.8473662),
    (1.8472793, -43.1, 1.8473589),
    (1.8472778, -45.3, 1.8473615),
    (1.8472783, -46.2, 1.8473636),
]


@pytest.mark.parametrize('stage2, s_ppm, joint', CELL2_BLOCKS)
def test_joint_scale_reproduces_cell_2(stage2, s_ppm, joint):
    """joint = stage2 - S*PS, and a NEGATIVE S is a LARGER scale. The other sign, and the
    zenith reference as the base, put Station 2's chart 840 ppm out."""
    assert joint_plate_scale(stage2, s_ppm * 1e-6, 1.84847) == pytest.approx(joint, abs=1.5e-7)


def test_joint_scale_sign():
    assert joint_plate_scale(2.0, -100e-6, 2.0) > 2.0     # negative S -> larger scale
    assert joint_plate_scale(2.0, +100e-6, 2.0) < 2.0
    assert ppm_from(1.0001, 1.0) == pytest.approx(100.0)


def _frame():
    """A sensor at 2 arcsec/px, rolled 30 degrees, centred at RA 17.5, Dec 7.9.

    The stars sit on a grid symmetric about the centre, so the frame's fitted ra0/de0 -- the
    sample means -- coincide with the centre the sky was generated about. With random points
    the means fell 0.017 deg off and the constant cos(dec) of the two conventions differed by
    3e-5, which read as a round-trip error that was not one.
    """
    ps, roll = 2.0, np.radians(30.0)
    ra0, de0 = 17.5, 7.9
    gx, gy = np.meshgrid(np.linspace(200, 3800, 10), np.linspace(300, 2700, 6))
    px, py = gx.ravel(), gy.ravel()
    x = (px - 2000) * ps / 3600; y = (py - 1500) * ps / 3600            # degrees on the sensor
    xa = np.cos(roll) * x - np.sin(roll) * y                            # sky, RA*cos(dec)
    ya = np.sin(roll) * x + np.cos(roll) * y
    ra = ra0 + xa / np.cos(np.radians(de0)); dec = de0 + ya
    return SkyFrame.from_stars(ra, dec, px, py, ps), ps, px, py, ra, dec


def test_sky_frame_round_trips():
    sf, ps, px, py, ra, dec = _frame()
    r, d = sf.px_to_sky(px, py)
    assert np.allclose(r, ra, atol=1e-9) and np.allclose(d, dec, atol=1e-9)
    # a unit sensor displacement is one arcsec of sky (the Bruns revision-10 defect was 2.087x)
    vx, vy = sf.sensor_vec_to_sky(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.allclose(np.hypot(vx, vy), 1.0, atol=1e-9)


def test_sky_frame_refuses_a_bad_scale():
    sf, ps, px, py, ra, dec = _frame()
    with pytest.raises(AssertionError):
        SkyFrame.from_stars(ra, dec, px, py, ps * 2.087)


def test_field_chart_axes_ascend_and_arrows_stay_inside():
    """RA ascends to the right whatever the sky convention says; every arrow end is inside."""
    import matplotlib.pyplot as plt
    sf, ps, px, py, ra, dec = _frame()
    x, y = sf.px_to_sky(px, py)
    vx, vy = sf.sensor_vec_to_sky(np.full(60, 0.5), np.full(60, -0.3))
    fig, ax = plt.subplots()
    lo_x, hi_x, lo_y, hi_y = field_chart(
        ax, x, y, vx, vy, sf.corners(4000, 3000), (float(x.mean()), float(y.mean()), 0.26),
        0.4, sf.cos0, groups=[(np.ones(60, bool), dict(s=20, color='tab:blue', label='stars'))],
        arrow_color='tab:blue')
    assert lo_x < hi_x and lo_y < hi_y
    assert ax.get_xlim()[0] < ax.get_xlim()[1], 'the RA axis is reversed'
    assert ax.get_xlabel() == 'RA (degrees)'
    plt.close(fig)


def test_covariance_chart_draws_two_ellipses_with_the_set_labels():
    import matplotlib.pyplot as plt
    C1 = np.array([[0.01, -0.5], [-0.5, 100.0]])
    C2 = np.array([[0.04, -1.0], [-1.0, 400.0]])
    fig, ax = covariance_chart(C1, (1.8, 0.0), C2, (1.9, -10.0), [('a line', 'black')], 'title')
    labels = [h.get_label() for h in ax.get_legend().legend_handles]
    assert any(l.startswith('1$\\sigma$') for l in labels)
    assert ax.get_ylabel() == 'Plate scale (ppm difference from imported value)'
    plt.close(fig)


@pytest.mark.parametrize('tool', ['matrix_bruns/b17_charts_record.py', 'step3_charts_record.py',
                                  'matrix_station1/s1_charts_record.py',
                                  'matrix_station2/s2_charts_record.py'])
def test_every_record_chart_tool_uses_the_shared_module(tool):
    src = open(os.path.join(REPO, 'tools', tool), encoding='utf-8').read()
    assert 'from tools.record_charts import' in src, (
        '%s must draw its charts through tools/record_charts.py, not its own copy' % tool)
    for private in ('def px_to_sky(', 'def sensor_vec_to_sky(', 'def draw(cov, mu'):
        assert private not in src, '%s still carries a private %s' % (tool, private)
