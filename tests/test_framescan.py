"""
Measuring a sequence: which frames are usable, and does each one match its stated exposure.

The shapes tested here are the ones real eclipse captures take. A sequence can open on the
uneclipsed Sun, settle into the corona, and then end saturated again as the Sun returns --
so nothing may assume the transient is at the start. And capture software that changes
exposure mid-sequence can write the new exposure into the header of a frame that still
holds the previous one, which no downstream stage can detect.
"""

import numpy as np
import pytest
from astropy.io import fits

from mee2024 import framescan


def _levels(medians, saturated=None, blank=None):
    """Hand-built scan output, so the decision logic can be tested without any files."""
    n = len(medians)
    saturated = saturated or [0.0] * n
    blank = blank or [0.0] * n
    return [{'index': i, 'frame': f'f{i}.fits', 'median': float(medians[i]),
             'mean': float(medians[i]), 'max': float(medians[i]),
             'saturated': float(saturated[i]), 'blank': float(blank[i])}
            for i in range(n)]


# --------------------------------------------------------------- suggesting a range

def test_a_stable_sequence_is_kept_whole():
    start, stop, info = framescan.suggest(_levels([1000 + i for i in range(20)]))
    assert (start, stop) == (0, 19)
    assert info['dropped_leading'] == info['dropped_trailing'] == 0


def test_a_saturated_head_is_trimmed():
    """The frames either side of second contact are not all-white -- they decay smoothly
    through every value in between -- so a blank/white test cannot find this boundary."""
    medians = [65535] * 5 + [40000, 20000, 9000, 5200, 4300] + [4000 - i * 5 for i in range(20)]
    saturated = [1.0] * 5 + [0.5, 0.3, 0.12, 0.06, 0.05] + [0.046] * 20
    start, stop, info = framescan.suggest(_levels(medians, saturated))
    assert start >= 10, 'the decaying frames should be trimmed, not just the pinned ones'
    assert stop == len(medians) - 1


def test_a_blank_tail_is_trimmed():
    medians = [3000 - i for i in range(20)] + [0] * 6
    blank = [0.0] * 20 + [1.0] * 6
    start, stop, _ = framescan.suggest(_levels(medians, blank=blank))
    assert (start, stop) == (0, 19)


def test_the_sun_may_be_at_the_END_and_is_still_trimmed():
    """The case that breaks any rule assuming saturated-at-start. A calibration sequence
    that runs through third contact goes from totality into full saturation in seconds."""
    medians = [5000 + i * 100 for i in range(8)] + [20000, 33000, 49000, 65535, 65535]
    saturated = [0.0] * 8 + [0.0, 0.0, 0.0, 0.7, 1.0]
    start, stop, info = framescan.suggest(_levels(medians, saturated))
    assert start == 0
    assert stop <= 8, 'the returning Sun must be trimmed from the end'
    assert info['dropped_trailing'] >= 4


def test_saturation_is_not_mistaken_for_stability():
    """A saturated frame has a pinned median, so its frame-to-frame change is exactly zero.
    A rule that looks only for "settled" keeps the entire saturated run."""
    medians = [65535] * 10 + [4000, 3000] + [2000 - i for i in range(15)]
    saturated = [1.0] * 10 + [0.4, 0.1] + [0.02] * 15
    start, _, _ = framescan.suggest(_levels(medians, saturated))
    assert start >= 10, 'the pinned run is stable but useless'


def test_a_legitimate_saturation_floor_is_not_trimmed():
    """A totality exposure clips the inner corona in every frame. That is not a defect and
    must not cause every frame to be dropped."""
    medians = [3000 - i for i in range(20)]
    saturated = [0.046] * 20
    start, stop, _ = framescan.suggest(_levels(medians, saturated))
    assert (start, stop) == (0, 19)


def test_the_change_threshold_adapts_to_the_cadence():
    """At 315 ms a stable stretch changes well under 1% a frame; at 2 s the same sky changes
    1.3-2.6%. A fixed threshold rejected every frame of a real 2 s sequence."""
    medians = [5058, 5126, 5214, 5310, 5421, 5553, 5697]     # measured, all usable
    start, stop, _ = framescan.suggest(_levels(medians))
    assert (start, stop) == (0, 6)


def test_an_all_blank_sequence_says_so():
    start, stop, info = framescan.suggest(_levels([0] * 5, blank=[1.0] * 5))
    assert start is None and 'blank' in info['reason']


def test_describe_names_what_was_dropped_and_why():
    medians = [65535] * 4 + [3000 - i for i in range(10)] + [0, 0]
    saturated = [1.0] * 4 + [0.0] * 10 + [0.0, 0.0]
    blank = [0.0] * 14 + [1.0, 1.0]
    text = framescan.describe(*framescan.suggest(_levels(medians, saturated, blank)))
    assert 'usable' in text and 'blank' in text and 'saturated' in text


# ------------------------------------------------------------------- parsing a range

@pytest.mark.parametrize('text, expected', [
    ('50-172', (50, 172)), ('50:172', (50, 172)), ('7', (7, 7)),
    ('', (None, None)), (None, (None, None)), ('all', (None, None)),
])
def test_ranges_parse(text, expected):
    assert framescan.parse_range(text) == expected


def test_a_range_is_clipped_to_the_sequence():
    assert framescan.parse_range('5-999', n_frames=20) == (5, 19)


@pytest.mark.parametrize('bad', ['abc', '10-2', '1-2-3'])
def test_a_bad_range_is_refused_rather_than_guessed(bad):
    with pytest.raises(ValueError):
        framescan.parse_range(bad, n_frames=20)


def test_a_range_past_the_end_is_refused():
    with pytest.raises(ValueError, match='past the end'):
        framescan.parse_range('50-60', n_frames=20)


def test_apply_range_selects_inclusively():
    frames = [f'f{i}' for i in range(10)]
    assert framescan.apply_range(frames, 2, 4) == ['f2', 'f3', 'f4']
    assert framescan.apply_range(frames, None, None) == frames


# ------------------------------------------------------ the exposure-consistency check

def _write(path, value, exptime, shape=(24, 32)):
    fits.writeto(path, np.full(shape, value, dtype=np.uint16),
                 header=fits.Header({'EXPTIME': exptime}), overwrite=True)
    return str(path)


def test_a_clean_ladder_says_nothing(tmp_path):
    frames = ([_write(tmp_path / f'a{i}.fits', 1000, 0.1) for i in range(4)]
              + [_write(tmp_path / f'b{i}.fits', 3000, 0.3) for i in range(4)]
              + [_write(tmp_path / f'c{i}.fits', 6000, 0.6) for i in range(4)])
    assert framescan.check_exposures(frames) == []


def test_the_first_frame_after_a_change_carrying_old_pixels_is_caught(tmp_path):
    """The real fault: header says 0.3 s, pixels are still the 0.1 s exposure."""
    frames = [_write(tmp_path / f'a{i}.fits', 1000, 0.1) for i in range(4)]
    frames.append(_write(tmp_path / 'bad.fits', 1010, 0.3))     # says 0.3, looks like 0.1
    frames += [_write(tmp_path / f'b{i}.fits', 3000, 0.3) for i in range(4)]
    messages = framescan.check_exposures(frames)
    assert len(messages) == 1
    assert 'bad.fits' in messages[0]
    assert '0.3' in messages[0] and '0.1' in messages[0]


def test_a_drifting_sky_does_not_hide_the_fault(tmp_path):
    """A whole-group comparison fails here: the group's own spread is wide enough to
    swallow the bad frame. The transition test is local, so it does not care."""
    frames = [_write(tmp_path / f'a{i}.fits', 1000 + i * 40, 0.1) for i in range(5)]
    frames.append(_write(tmp_path / 'bad.fits', 1170, 1.0))     # says 1.0, looks like 0.1
    frames += [_write(tmp_path / f'b{i}.fits', 2800 + i * 400, 1.0) for i in range(5)]
    messages = framescan.check_exposures(frames)
    assert any('bad.fits' in m for m in messages)


def test_a_single_exposure_field_is_silent(tmp_path):
    frames = [_write(tmp_path / f'z{i}.fits', 1000 + i, 4.0) for i in range(10)]
    assert framescan.check_exposures(frames) == []


def test_an_outlier_without_a_transition_is_still_reported(tmp_path):
    """Cloud, or a frame at some other exposure entirely."""
    frames = [_write(tmp_path / f'z{i}.fits', 1000, 4.0) for i in range(8)]
    frames.insert(4, _write(tmp_path / 'cloudy.fits', 9000, 4.0))
    messages = framescan.check_exposures(frames)
    assert any('cloudy.fits' in m for m in messages)


def test_frames_without_an_exposure_keyword_are_not_a_failure(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f'n{i}.fits'
        fits.writeto(path, np.full((8, 8), 100, dtype=np.uint16), overwrite=True)
        paths.append(str(path))
    assert framescan.check_exposures(paths) == []


def test_a_steadily_brightening_sky_is_a_trend_not_a_hundred_faults(tmp_path):
    """Against the whole group this drowned a real capture in warnings: the sky brightened
    from first frame to last, so most frames sat far from the group median and every one
    was reported. A trend is not a fault."""
    frames = [_write(tmp_path / f'r{i:03d}.fits', 1800 + i * 60, 0.315) for i in range(120)]
    assert framescan.check_exposures(frames) == []


def test_one_bad_frame_inside_a_brightening_sequence_is_still_found(tmp_path):
    """...and the local comparison must not have thrown away the sensitivity that matters."""
    frames = [_write(tmp_path / f'r{i:03d}.fits', 1800 + i * 60, 0.315) for i in range(60)]
    frames.insert(30, _write(tmp_path / 'odd.fits', 12000, 0.315))
    messages = framescan.check_exposures(frames)
    assert any('odd.fits' in m for m in messages)


def test_a_wall_of_messages_is_summarised_rather_than_printed(tmp_path):
    """Past a certain count the sequence is the problem, not the frames.

    Isolated spikes, not an alternating pattern -- an alternation is a consistent shape and
    a local median rightly sees no outlier in it.
    """
    frames = []
    for i in range(66):
        value = 9000 if i % 3 == 0 else 1000
        frames.append(_write(tmp_path / f'x{i:03d}.fits', value, 0.5))
    messages = framescan.check_exposures(frames)
    assert len(messages) <= framescan.MAX_MESSAGES + 1
    assert any('suggested frame range' in m for m in messages)
