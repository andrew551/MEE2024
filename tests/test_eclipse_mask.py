"""The Sun/Moon mask and the coronal subtraction (F26).

The disk replaced the convex-hull blob because the hull followed streamers: it wrapped
whatever saturated, so a bright streamer pulled the mask into a lobe that covered sky far
from the Sun while another azimuth was left at the core's edge. These tests pin the two
properties that motivated the change -- the disk's radius does not depend on a streamer,
and its geometry comes from the raw frame rather than from whatever preprocessing ran
first -- plus the on/off switch that gives a plain stack.
"""

import numpy as np
import pytest

from mee2024 import stacker_implementation as si
from mee2024.config import DEFAULT_OPTIONS


SHAPE = (600, 800)
CENTRE = (300.0, 400.0)      # (y, x)
CORE_R = 90.0


def frame_with_saturated_disc(streamer=False, background=1000.0):
    """A saturated circle, optionally with a long thin saturated streamer off one side."""
    img = np.full(SHAPE, background, dtype=np.float64)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    r = np.hypot(yy - CENTRE[0], xx - CENTRE[1])
    img[r <= CORE_R] = 65535.0
    if streamer:
        # 12 px wide, reaching 250 px to the right -- far outside any honest mask
        img[int(CENTRE[0]) - 6:int(CENTRE[0]) + 6,
            int(CENTRE[1]):int(CENTRE[1]) + 250] = 65535.0
    return img


def opts(**over):
    o = dict(DEFAULT_OPTIONS)
    o.update(over)
    return o


def test_disk_covers_the_core_and_stops_just_outside_it():
    img = frame_with_saturated_disc()
    o = opts(eclipse_mask_mode='disk', eclipse_disk_margin_px=10, centroid_gap_blob=30)
    out, paint, detect = si.mask_bright_object(img, img, o)
    assert paint.any()
    yy, xx = np.nonzero(paint)
    radii = np.hypot(yy - CENTRE[0], xx - CENTRE[1])
    # every saturated pixel is painted, and the mask does not reach far beyond the core
    assert (img[paint] >= 65535).sum() >= (img >= 65535).sum() * 0.99
    assert radii.max() < CORE_R + 25          # margin 10 px, plus downscale-8 granularity
    assert detect.sum() > paint.sum()         # the detection mask is the wider one


def test_a_streamer_does_not_inflate_the_disk():
    """The property the convex hull failed: a bright streamer must not grow the mask."""
    plain = si.mask_bright_object(frame_with_saturated_disc(False), frame_with_saturated_disc(False),
                                  opts(eclipse_mask_mode='disk'))[1]
    streaked_img = frame_with_saturated_disc(True)
    streaked = si.mask_bright_object(streaked_img, streaked_img, opts(eclipse_mask_mode='disk'))[1]
    grew = (streaked.sum() - plain.sum()) / plain.sum()
    assert abs(grew) < 0.25, f'the streamer changed the disk area by {100 * grew:.0f}%'
    # and the far end of the streamer is left unmasked, unlike the hull
    assert not streaked[int(CENTRE[0]), int(CENTRE[1]) + 240]


def test_the_hull_by_contrast_does_follow_the_streamer():
    """Kept as the reason the default changed, not as an endorsement."""
    streaked = frame_with_saturated_disc(True)
    hull = si.mask_bright_object(streaked, streaked, opts(eclipse_mask_mode='blob'))[1]
    assert hull[int(CENTRE[0]), int(CENTRE[1]) + 240]


def test_masking_off_leaves_the_frame_untouched():
    img = frame_with_saturated_disc()
    out, paint, detect = si.mask_bright_object(img, img, opts(delete_saturated_blob=False))
    assert np.array_equal(out, img)
    assert not paint.any() and not detect.any()


def test_the_disk_is_a_logical_gate_and_never_touches_a_pixel():
    """The reason fake centroids clustered on the mask perimeter: painting the region flat
    puts a hard edge in the data, and the 17 px background blur straddling that edge makes
    `img - blur` spuriously positive just outside it. The disk gates detection instead."""
    img = frame_with_saturated_disc(streamer=True)
    before = img.copy()
    out, paint, detect = si.mask_bright_object(img, img, opts(eclipse_mask_mode='disk'))
    assert out is img or np.array_equal(out, before)   # not a pixel changed
    assert np.array_equal(img, before)                 # and the input was not mutated
    assert paint.any() and detect.sum() > paint.sum()  # the gate is still there


def test_the_legacy_blob_still_paints():
    """`blob` exists to reproduce pre-v1.4.0 reductions, so it must keep its behaviour."""
    img = frame_with_saturated_disc()
    out, paint, _ = si.mask_bright_object(img, img, opts(eclipse_mask_mode='blob'))
    assert not np.array_equal(out, img)
    assert out[paint].max() < 65535


def test_no_saturated_object_means_no_mask():
    img = np.full(SHAPE, 1000.0)
    img[300, 400] = 5000.0                     # a star, not a Sun
    out, paint, detect = si.mask_bright_object(img, img, opts())
    assert not paint.any()
    assert np.array_equal(out, img)


def test_mask_geometry_is_read_from_the_raw_frame():
    """Saturation is a statement about the sensor, so the mask is measured on the raw
    frame. Coronal subtraction rescales everything and moves the apparent full-scale
    level, so reading the geometry off the subtracted frame gives a different -- and
    wrong -- disk even when it finds one at all."""
    raw = frame_with_saturated_disc()
    subtracted = si.subtract_coronal_background(raw, opts(coronal_subtract=True))
    _, paint_raw, _ = si.mask_bright_object(subtracted, raw, opts())
    _, paint_self, _ = si.mask_bright_object(subtracted, subtracted, opts())
    assert paint_raw.any()
    # the raw-measured disk covers the true saturated core
    assert (raw[paint_raw] >= 65535).sum() >= (raw >= 65535).sum() * 0.99
    # the self-measured one is materially different -- it is not the same mask
    assert abs(int(paint_self.sum()) - int(paint_raw.sum())) > 0.1 * paint_raw.sum()


def test_coronal_subtraction_flattens_a_steep_gradient():
    """A 1/r^2 corona plus a flat sky: after subtraction the background is flat to a few
    ADU, which is the condition an annular or Gaussian local background needs."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    r = np.hypot(yy - CENTRE[0], xx - CENTRE[1]) + 1.0
    corona = 4.0e6 / r**2
    img = 1000.0 + corona
    o = opts(coronal_subtract=True, coronal_subtract_sigma_px=10.0, coronal_pedestal_adu=2000.0)
    out = si.subtract_coronal_background(img, o)
    ring = (r > 200) & (r < 280)
    before = img[ring].std()
    after = out[ring].std()
    assert after < before / 20, f'gradient not flattened: {before:.0f} -> {after:.0f} ADU'
    assert abs(np.median(out[ring]) - 2000.0) < 60      # sits on the pedestal


def test_coronal_subtraction_does_not_carve_a_ring_outside_a_saturated_core():
    """The defect that put fake detections on the mask perimeter.

    Blurring the frame with the saturated core included leaves the model near full scale
    for ~3 sigma outside the core, so the subtraction digs a trench there -- right where
    the inner stars are, and wider than the forbidden disk's margin. The masked blur must
    hand back a flat background all the way to the core's edge.
    """
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    r = np.hypot(yy - CENTRE[0], xx - CENTRE[1]) + 1.0
    img = 1000.0 + 4.0e6/r**2
    img[r <= CORE_R] = 65535.0                      # the saturated core
    out = si.subtract_coronal_background(img, opts(coronal_subtract=True))
    # just outside the core, the background must sit at the pedestal, not in a trench
    for lo, hi in ((CORE_R + 5, CORE_R + 20), (CORE_R + 20, CORE_R + 50),
                   (CORE_R + 50, CORE_R + 100)):
        ring = (r > lo) & (r < hi)
        med = float(np.median(out[ring]))
        assert abs(med - 2000.0) < 250, (
            'ring %.0f-%.0f px outside the core sits at %.0f ADU, not the 2000 pedestal'
            % (lo, hi, med))


def test_coronal_subtraction_keeps_a_star_measurable():
    """The subtraction must not eat the stars it exists to reveal: a narrow source is far
    smaller than the blur, so its peak above background survives."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    r = np.hypot(yy - CENTRE[0], xx - CENTRE[1]) + 1.0
    img = 1000.0 + 4.0e6 / r**2
    star_r = np.hypot(yy - 150.0, xx - 650.0)
    img = img + 3000.0 * np.exp(-star_r**2 / (2 * 1.5**2))
    o = opts(coronal_subtract=True)
    out = si.subtract_coronal_background(img, o)
    local = out[130:170, 630:670]
    peak_above_bg = local.max() - np.median(local)
    assert peak_above_bg > 3000.0 * 0.9
