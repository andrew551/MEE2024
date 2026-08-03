"""
Labels for the identified stars the UI draws over the stacked preview.

The interesting behaviour is the tier assignment -- the slider only ever *adds* labels, so
a star must land in the tightest tier it qualifies for -- and that a star eliminated as a
double is still reported, flagged, so the frontend can cross it out.
"""

import numpy as np
import pytest

from mee2024 import events, star_labels


class _Index:
    """Stands in for the bundled name index: star 1 is named, star 2 has a HIP number."""

    def name_for(self, ids):
        return ['Vega' if i == 1 else '' for i in ids]

    def hip_for(self, ids):
        return [91262 if i == 2 else 0 for i in ids]


@pytest.fixture
def named_index(monkeypatch):
    import mee2024.starcat.labels as labels_module
    monkeypatch.setattr(labels_module.LabelIndex, 'try_bundled',
                        classmethod(lambda cls: _Index()))


def test_tiers_prefer_name_then_hip_then_magnitude(named_index):
    labels, tiers = star_labels.build_labels([0.0, 2.2, 7.5, 12.0], ids=[1, 2, 3, 4])
    assert labels == ['Vega', 'HIP 91262', 'G 7.5', 'G 12.0']
    assert tiers == ['named', 'hip', 'bright', 'all']


def test_tiers_fall_back_to_magnitudes_without_ids():
    labels, tiers = star_labels.build_labels([4.0, 11.0])
    assert labels == ['G 4.0', 'G 11.0']
    assert tiers == ['bright', 'all']


def test_bright_limit_is_the_tier_boundary():
    _, tiers = star_labels.build_labels([9.0, 9.1], bright_limit=9.0)
    assert tiers == ['bright', 'all']


def test_a_missing_name_index_still_labels_by_magnitude(monkeypatch):
    import mee2024.starcat.labels as labels_module
    monkeypatch.setattr(labels_module.LabelIndex, 'try_bundled',
                        classmethod(lambda cls: (_ for _ in ()).throw(OSError('gone'))))
    labels, tiers = star_labels.build_labels([3.0], ids=[1])
    assert labels == ['G 3.0'] and tiers == ['bright']


def test_emit_sends_columns_in_image_pixels():
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        star_labels.emit(np.array([[10.0, 20.0], [30.0, 40.0]]), [5.0, 6.0], (100, 200))
    event = sink.latest(events.STARS)
    assert event['image_size'] == [100, 200]
    assert event['y'] == [10.0, 30.0] and event['x'] == [20.0, 40.0]
    assert event['dropped'] == [False, False]
    assert event['stage'] == 'stack'


def test_emit_flags_eliminated_doubles():
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        star_labels.emit(np.array([[1.0, 2.0], [3.0, 4.0]]), [5.0, 6.0], (10, 10),
                         dropped=np.array([False, True]), stage='distortion')
    event = sink.latest(events.STARS)
    assert event['dropped'] == [False, True]
    assert event['stage'] == 'distortion'


def test_emit_is_a_no_op_with_no_stars():
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        assert star_labels.emit(np.zeros((0, 2)), [], (10, 10)) is None
    assert sink.latest(events.STARS) is None


def test_emit_from_solution_takes_magnitudes_from_column_five():
    stars = np.zeros((2, 6))
    stars[:, 5] = [1.5, 10.0]
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        star_labels.emit_from_solution(
            {'matched_stars': stars, 'matched_centroids': np.array([[1., 2.], [3., 4.]])},
            (50, 60))
    event = sink.latest(events.STARS)
    assert event['mag'] == [1.5, 10.0]
    assert event['tier'] == ['bright', 'all']


def test_emit_from_solution_ignores_a_failed_solve():
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        assert star_labels.emit_from_solution({'matched_stars': None}, (50, 60)) is None
    assert sink.latest(events.STARS) is None


# ------------------------------------------------- names found by position, not by id

class _PositionIndex:
    """A stand-in whose crossmatch is empty, as Gaia's largely is for named stars."""

    def name_for(self, ids):
        return [''] * len(list(ids))

    def hip_for(self, ids):
        return [0] * len(list(ids))

    def __init__(self):
        self.asked = []

    def names_by_position(self, ra, dec, epoch=2024.0):
        self.asked.append((list(np.atleast_1d(ra)), float(epoch)))
        return ['Rasalhague'] + [None] * (len(np.atleast_1d(ra)) - 1)


@pytest.fixture
def position_index(monkeypatch):
    import mee2024.starcat.labels as labels_module
    index = _PositionIndex()
    monkeypatch.setattr(labels_module.LabelIndex, 'try_bundled',
                        classmethod(lambda cls: index))
    return index


def test_a_name_is_found_by_position_when_the_id_cannot_reach_one(position_index):
    """Gaia's crossmatch to Hipparcos misses 46 of the 49 named stars, so without this
    the brightest star in a frame is the one least likely to be labelled."""
    labels, tiers = star_labels.build_labels(
        [2.11, 8.0], ids=[4493746564376875520, 2], ra=[0.1, 0.2], dec=[0.3, 0.4])
    assert labels[0] == 'Rasalhague' and tiers[0] == 'named'
    assert labels[1] == 'G 8.0'


def test_the_positional_lookup_is_given_the_observation_epoch(position_index):
    """These stars move; comparing against the wrong epoch is what loses them."""
    star_labels.build_labels([2.11], ids=[1], ra=[0.1], dec=[0.3], epoch=2022.63)
    assert position_index.asked[0][1] == pytest.approx(2022.63)


def test_without_positions_nothing_is_asked(position_index):
    star_labels.build_labels([2.11], ids=[1])
    assert position_index.asked == []


def test_a_failing_positional_lookup_does_not_break_labelling(monkeypatch):
    import mee2024.starcat.labels as labels_module

    class _Broken(_PositionIndex):
        def names_by_position(self, ra, dec, epoch=2024.0):
            raise RuntimeError('no hipparcos')

    monkeypatch.setattr(labels_module.LabelIndex, 'try_bundled',
                        classmethod(lambda cls: _Broken()))
    labels, _ = star_labels.build_labels([2.11], ids=[1], ra=[0.1], dec=[0.3])
    assert labels == ['G 2.1']


def test_a_hip_label_is_upgraded_to_a_proper_name(monkeypatch):
    """'HIP 86032' is correct but 'Rasalhague' is better, so a position wins over it."""
    import mee2024.starcat.labels as labels_module

    class _HipOnly(_PositionIndex):
        def hip_for(self, ids):
            return [86032] * len(list(ids))

    monkeypatch.setattr(labels_module.LabelIndex, 'try_bundled',
                        classmethod(lambda cls: _HipOnly()))
    labels, tiers = star_labels.build_labels([2.11], ids=[1], ra=[0.1], dec=[0.3])
    assert labels == ['Rasalhague'] and tiers == ['named']
