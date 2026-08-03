"""
Lending a proper motion to Gaia's brightest stars.

Gaia's brightest stars often get two-parameter solutions -- a position and nothing else --
because they saturate its detectors: 21% of the catalogue brighter than G=4 has no proper
motion, against 0.8% at G=10-13. Those positions cannot be propagated to the observation
epoch, go stale, and the distortion fit discards them as outliers. Measured on the bundled
Rasalhague frame: the brightest star in the field missed by 1.845 arcsec after 6.6 years of
unpropagated motion, where every other star of the brightest fifteen sat at 0.01-0.25.

Hipparcos has good proper motions for exactly those stars and is already merged in to fill
the bright end, so the fix is to borrow one.
"""

import numpy as np
import pytest

from mee2024.starcat import providers
from mee2024.starcat.table import ORIGIN_GAIA, ORIGIN_HIPPARCOS, StarTable


def _table(ra_deg, dec_deg, mags, pmra, pmdec, epoch=2016.0, origin=ORIGIN_GAIA):
    n = len(ra_deg)
    return StarTable(ra=np.radians(np.asarray(ra_deg, dtype=float)),
                     dec=np.radians(np.asarray(dec_deg, dtype=float)),
                     mag=np.asarray(mags, dtype=float),
                     ids=np.arange(n, dtype=np.int64), epoch=epoch, origin=origin,
                     pmra=np.asarray(pmra, dtype=float),
                     pmdec=np.asarray(pmdec, dtype=float),
                     parallax=np.full(n, np.nan),
                     radial_velocity=np.full(n, np.nan))


class _Donor:
    """A stand-in fill provider returning a fixed table, propagated on request."""

    name = 'donor'
    is_offline = True

    def __init__(self, table):
        self.table = table
        self.asked_for = []

    def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
        self.asked_for.append(epoch)
        return self.table.at_epoch(epoch) if epoch is not None else self.table


RA, DEC = 100.0, 20.0


def test_a_star_with_no_proper_motion_gets_one():
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    donor = _Donor(_table([RA], [DEC], [2.1], [108.0], [-221.0],
                          origin=ORIGIN_HIPPARCOS))
    filled = providers.fill_proper_motion(target, [(donor, 8.0)],
                                          (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert filled == 1
    assert target.has_proper_motion()[0]
    assert target.pmra[0] == pytest.approx(108.0)
    assert target.pmdec[0] == pytest.approx(-221.0)


def test_the_position_is_left_alone_only_the_motion_is_borrowed():
    """Gaia's position is the better one at its own epoch; only the motion is missing."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    before = (target.ra[0], target.dec[0])
    donor = _Donor(_table([RA + 0.0001], [DEC], [2.1], [108.0], [-221.0]))
    providers.fill_proper_motion(target, [(donor, 8.0)], (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert (target.ra[0], target.dec[0]) == before


def test_a_star_that_already_has_one_is_untouched():
    target = _table([RA], [DEC], [2.0], [5.0], [6.0])
    donor = _Donor(_table([RA], [DEC], [2.1], [108.0], [-221.0]))
    assert providers.fill_proper_motion(target, [(donor, 8.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 0
    assert target.pmra[0] == 5.0


def test_a_donor_too_far_away_is_not_believed():
    """Both sides are at a common epoch, so a real pair is milliarcseconds apart."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    donor = _Donor(_table([RA + 0.01], [DEC], [2.0], [108.0], [-221.0]))   # 34 arcsec
    assert providers.fill_proper_motion(target, [(donor, 8.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 0
    assert not target.has_proper_motion()[0]


def test_a_donor_of_quite_different_brightness_is_not_believed():
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    donor = _Donor(_table([RA], [DEC], [9.0], [108.0], [-221.0]))
    assert providers.fill_proper_motion(target, [(donor, 12.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 0


def test_a_donor_without_a_motion_of_its_own_is_no_help():
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    donor = _Donor(_table([RA], [DEC], [2.0], [np.nan], [np.nan]))
    assert providers.fill_proper_motion(target, [(donor, 8.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 0


def test_the_donor_is_asked_at_the_targets_epoch():
    """Comparing positions from different epochs would reject real pairs and accept wrong
    ones -- the whole point is that these stars move."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan], epoch=2016.0)
    donor = _Donor(_table([RA], [DEC], [2.0], [108.0], [-221.0], epoch=1991.25))
    providers.fill_proper_motion(target, [(donor, 8.0)], (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert donor.asked_for == [2016.0]


def test_the_first_fill_source_wins():
    """Hipparcos before Tycho: its astrometry is the better of the two."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    first = _Donor(_table([RA], [DEC], [2.0], [1.0], [2.0]))
    second = _Donor(_table([RA], [DEC], [2.0], [99.0], [99.0]))
    providers.fill_proper_motion(target, [(first, 8.0), (second, 11.0)],
                                 (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert target.pmra[0] == pytest.approx(1.0)


def test_a_second_source_fills_what_the_first_could_not():
    target = _table([RA, RA + 0.5], [DEC, DEC], [2.0, 3.0],
                    [np.nan, np.nan], [np.nan, np.nan])
    first = _Donor(_table([RA], [DEC], [2.0], [1.0], [2.0]))
    second = _Donor(_table([RA + 0.5], [DEC], [3.0], [7.0], [8.0]))
    filled = providers.fill_proper_motion(target, [(first, 8.0), (second, 11.0)],
                                          (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert filled == 2
    assert target.pmra[0] == pytest.approx(1.0) and target.pmra[1] == pytest.approx(7.0)


def test_a_parallax_is_taken_too_when_gaia_has_none():
    """It only curves the propagation, but a bright star's parallax is often large."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    donor = _Donor(_table([RA], [DEC], [2.0], [108.0], [-221.0]))
    donor.table.parallax = np.array([67.0])
    providers.fill_proper_motion(target, [(donor, 8.0)], (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    assert target.get_parallax()[0] == pytest.approx(67.0)


def test_an_empty_table_is_not_a_problem():
    assert providers.fill_proper_motion(providers.empty_table(), [(None, 8.0)],
                                        (0, 1), (0, 1)) == 0


def test_no_fill_sources_is_not_a_problem():
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    assert providers.fill_proper_motion(target, [], (0, 1), (0, 1)) == 0


def test_a_donor_that_raises_is_skipped_not_fatal():
    """An online fill source can fail; the run should carry on with what it has."""
    class _Broken:
        name = 'broken'
        def lookup(self, *a, **k):
            raise RuntimeError('no network')

    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
    good = _Donor(_table([RA], [DEC], [2.0], [108.0], [-221.0]))
    assert providers.fill_proper_motion(target, [(_Broken(), 8.0), (good, 8.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 1


def test_filling_makes_the_star_propagate():
    """The whole point: without a motion the position stays at the catalogue epoch."""
    target = _table([RA], [DEC], [2.0], [np.nan], [np.nan], epoch=2016.0)
    stale = target.at_epoch(2022.63)
    assert stale.ra[0] == target.ra[0], 'a star with no motion should not move'

    donor = _Donor(_table([RA], [DEC], [2.0], [108.0], [-221.0]))
    providers.fill_proper_motion(target, [(donor, 8.0)], (RA - 1, RA + 1), (DEC - 1, DEC + 1))
    moved = target.at_epoch(2022.63)
    shift = 3600 * np.degrees(np.hypot(
        (moved.ra[0] - target.ra[0]) * np.cos(target.dec[0]), moved.dec[0] - target.dec[0]))
    # 6.63 years at 246 mas/yr is about 1.6 arcsec, which is what the 1 arcsec fit
    # tolerance was rejecting
    assert 1.3 < shift < 2.0, f'moved {shift:.2f} arcsec'


# ---------------------------------------------------- the unpropagated lookup contract

def test_the_offline_provider_can_return_positions_unpropagated():
    """The fill has to happen before propagation, which needs this mode."""
    from mee2024.starcat import providers as p
    provider = p.GaiaOfflineProvider.from_installed()
    native = provider.lookup((RA - 0.2, RA + 0.2), (DEC - 0.2, DEC + 0.2), 9.0, epoch=None)
    assert native.epoch == pytest.approx(p.GAIA_DR3_EPOCH, abs=0.5)


def test_the_merged_provider_can_be_asked_not_to_fill():
    provider = providers.MergedProvider(fill_proper_motions=False)
    assert provider.fill_proper_motions is False


def test_a_source_is_not_used_past_its_magnitude_ceiling():
    """Tycho's positions reach ~2.5 arcsec by V=11, so beyond its ceiling a 2-arcsec match
    radius would be picking neighbours at random rather than identifying the same star."""
    target = _table([RA], [DEC], [10.5], [np.nan], [np.nan])
    donor = _Donor(_table([RA], [DEC], [10.5], [3.0], [4.0]))
    assert providers.fill_proper_motion(target, [(donor, 9.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 0
    assert donor.asked_for == [], 'should not even query past the ceiling'


def test_a_faint_star_is_skipped_while_a_bright_one_is_filled():
    target = _table([RA, RA + 0.01], [DEC, DEC], [2.0, 10.5],
                    [np.nan, np.nan], [np.nan, np.nan])
    donor = _Donor(_table([RA, RA + 0.01], [DEC, DEC], [2.0, 10.5],
                          [1.0, 3.0], [2.0, 4.0]))
    assert providers.fill_proper_motion(target, [(donor, 9.0)],
                                        (RA - 1, RA + 1), (DEC - 1, DEC + 1)) == 1
    assert target.has_proper_motion()[0]
    assert not target.has_proper_motion()[1]


def test_a_primary_that_ignores_the_unpropagated_mode_still_works():
    """A provider may not implement epoch=None. Returning a table with epoch None would
    make everything downstream unpropagatable, so the merge falls back to a plain lookup
    rather than failing -- the fill can only ever be an improvement."""
    class _Literal:
        """Stores whatever epoch it is handed, including None."""
        name = 'literal'
        is_offline = True
        magnitude_limit = 12.0

        def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
            out = _table([RA], [DEC], [2.0], [np.nan], [np.nan])
            out.epoch = epoch
            return out

    donor = _Donor(_table([RA], [DEC], [2.0], [108.0], [-221.0]))
    merged = providers.MergedProvider(primary=_Literal(), fills=[(donor, 8.0)])
    result = merged.lookup((RA - 1, RA + 1), (DEC - 1, DEC + 1), 12.0, epoch=2024.0)
    assert result.epoch == pytest.approx(2024.0)
    assert len(result) >= 1
