"""Catalogue providers: the Tycho adapter, the merge policy, and the registry."""

import numpy as np
import pytest

from mee2024.starcat import ORIGIN_GAIA, ORIGIN_TYCHO, StarTable
from mee2024.starcat import providers as prov


# ---------------------------------------------------------------- Tycho ids

def test_tycho_id_round_trip():
    tyc1, tyc2, tyc3 = 9537, 12121, 4
    packed = prov.encode_tycho_id(tyc1, tyc2, tyc3)
    assert prov.decode_tycho_id(packed) == (tyc1, tyc2, tyc3)


def test_tycho_ids_are_unique_and_vectorised():
    tyc1 = np.array([1, 1, 2, 9537])
    tyc2 = np.array([1, 2, 1, 12121])
    tyc3 = np.array([1, 1, 1, 4])
    packed = prov.encode_tycho_id(tyc1, tyc2, tyc3)
    assert len(np.unique(packed)) == 4
    back = prov.decode_tycho_id(packed)
    assert np.array_equal(back[0], tyc1)
    assert np.array_equal(back[1], tyc2)
    assert np.array_equal(back[2], tyc3)


# ------------------------------------------------------------ Tycho provider

def test_tycho_provider_returns_a_star_table():
    provider = prov.TychoProvider()
    table = provider.lookup((354.2, 358.1), (44.0, 46.3), max_magnitude=11.0)
    assert len(table) > 50
    assert np.all(table.origin == ORIGIN_TYCHO)
    assert table.band == 'G_est_from_V'
    assert table.epoch == prov.TychoProvider.CATALOGUE_EPOCH
    assert np.all(table.mag < 11.0)


def test_tycho_provider_reports_no_proper_motion():
    """Tycho positions are pre-propagated and carry no PM, per star."""
    table = prov.TychoProvider().lookup((354.2, 358.1), (44.0, 46.3), max_magnitude=10.0)
    assert not table.has_proper_motion().any()
    assert not table.is_gaia().any()


def test_tycho_provider_can_keep_native_v_magnitudes():
    box = ((354.2, 358.1), (44.0, 46.3))
    as_v = prov.TychoProvider(magnitude_to_g=False).lookup(*box, max_magnitude=11.0)
    as_g = prov.TychoProvider(magnitude_to_g=True).lookup(*box, max_magnitude=11.0)
    assert as_v.band == 'V'
    assert as_g.band == 'G_est_from_V'
    # the G estimate is the V magnitude plus the measured offset
    assert np.median(as_g.mag) < np.median(as_v.mag)


def test_tycho_provider_describes_itself():
    text = prov.TychoProvider().describe()
    assert 'tycho' in text and 'offline' in text


def test_lookup_objects_alias_still_works():
    """The pre-refactor call site says dbs.lookup_objects(...)."""
    table = prov.TychoProvider().lookup_objects((354.2, 358.1), (44.0, 46.3),
                                                star_max_magnitude=10.0)
    assert len(table) > 0


# ------------------------------------------------------------- merge policy

def _fake_provider(table, offline=True, limit=12.0):
    class Fake(prov.CatalogueProvider):
        name = 'fake'
        is_offline = offline
        magnitude_limit = limit

        def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
            keep = table.mag < max_magnitude
            out = table.select(keep)
            # epoch=None means "leave it at the catalogue's own epoch", which is what the
            # proper-motion fill needs; honour that rather than storing a None
            if epoch is not None:
                out.epoch = epoch
            return out
    return Fake()


def _table(ras, decs, mags, origin, epoch=2024.0, band='G'):
    return StarTable(ra=np.radians(ras), dec=np.radians(decs), mag=mags,
                     ids=np.arange(1, len(ras) + 1, dtype=np.int64),
                     epoch=epoch, origin=origin, band=band)


def test_merge_drops_tycho_stars_that_gaia_already_has():
    gaia = _table([10.0, 11.0], [20.0, 20.0], [8.0, 8.5], ORIGIN_GAIA)
    # the first Tycho star coincides with a Gaia star; the second does not
    tycho = _table([10.00001, 12.0], [20.0, 20.0], [5.0, 5.5], ORIGIN_TYCHO,
                   band='G')
    merged = prov.MergedProvider(primary=_fake_provider(gaia),
                                 secondary=_fake_provider(tycho)).lookup(
        (0.0, 30.0), (10.0, 30.0), max_magnitude=12.0)

    assert len(merged) == 3, 'the duplicate Tycho star should have been dropped'
    assert int(np.sum(merged.origin == ORIGIN_TYCHO)) == 1
    kept_tycho_ra = np.degrees(merged.ra[merged.origin == ORIGIN_TYCHO])[0]
    assert kept_tycho_ra == pytest.approx(12.0)


def test_merge_rejects_tycho_stars_fainter_than_the_fill_limit():
    """Tycho positions are ~2.5 arcsec by V=11, so faint fills are worse than nothing."""
    gaia = _table([10.0], [20.0], [8.0], ORIGIN_GAIA)
    tycho = _table([50.0, 51.0], [20.0, 20.0], [7.0, 10.5], ORIGIN_TYCHO, band='G')
    merged = prov.MergedProvider(primary=_fake_provider(gaia),
                                 secondary=_fake_provider(tycho),
                                 bright_fill_limit=9.0).lookup(
        (0.0, 60.0), (10.0, 30.0), max_magnitude=12.0)
    assert len(merged) == 2
    assert np.max(merged.mag[merged.origin == ORIGIN_TYCHO]) < 9.0


def test_merge_preserves_per_star_provenance_for_the_precision_fit():
    """The whole point: the distortion fit must be able to exclude Tycho stars."""
    gaia = _table([10.0, 11.0], [20.0, 20.0], [8.0, 8.5], ORIGIN_GAIA)
    tycho = _table([50.0], [20.0], [6.0], ORIGIN_TYCHO, band='G')
    merged = prov.MergedProvider(primary=_fake_provider(gaia),
                                 secondary=_fake_provider(tycho)).lookup(
        (0.0, 60.0), (10.0, 30.0))
    gaia_only = merged.select(merged.is_gaia())
    assert len(gaia_only) == 2
    assert np.all(gaia_only.origin == ORIGIN_GAIA)


def test_merge_with_no_tycho_contribution_returns_gaia_unchanged():
    gaia = _table([10.0, 11.0], [20.0, 20.0], [8.0, 8.5], ORIGIN_GAIA)
    empty = prov.empty_table(epoch=2024.0)
    merged = prov.MergedProvider(primary=_fake_provider(gaia),
                                 secondary=_fake_provider(empty)).lookup(
        (0.0, 30.0), (10.0, 30.0))
    assert len(merged) == 2


def test_merge_is_offline_only_when_both_halves_are():
    gaia = _table([10.0], [20.0], [8.0], ORIGIN_GAIA)
    both_offline = prov.MergedProvider(primary=_fake_provider(gaia, offline=True),
                                       secondary=_fake_provider(gaia, offline=True))
    mixed = prov.MergedProvider(primary=_fake_provider(gaia, offline=False),
                                secondary=_fake_provider(gaia, offline=True))
    assert both_offline.is_offline
    assert not mixed.is_offline


# ------------------------------------------------------------------ defaults

def test_provider_without_neighbour_support_flags_no_doubles():
    """A provider that cannot answer the question must not flag everything."""
    table = _table([10.0, 11.0], [20.0, 20.0], [8.0, 9.0], ORIGIN_GAIA)
    neighbours = prov.CatalogueProvider().lookup_neighbours(table, 10.0, 17.0)
    assert len(neighbours) == 0


def test_empty_table_is_usable():
    empty = prov.empty_table()
    assert len(empty) == 0
    assert empty.get_vectors().shape == (0, 3)
    assert not empty.has_proper_motion().any()


# ------------------------------------------------------------------ registry

def test_registry_lists_the_known_catalogues():
    assert {'gaia', 'tycho', 'merged'} <= set(prov.known_catalogues())


def test_registry_builds_providers_by_name():
    assert isinstance(prov.build('tycho'), prov.TychoProvider)
    assert isinstance(prov.build('gaia_online'), prov.GaiaOnlineProvider)
    assert isinstance(prov.build('merged'), prov.MergedProvider)


def test_registry_passes_the_gaia_limit_through():
    assert prov.build('gaia_online', gaia_limit=11.5).magnitude_limit == 11.5


def test_registry_rejects_an_unknown_name():
    with pytest.raises(KeyError, match='unknown catalogue'):
        prov.build('sloan')


# ------------------------------------------------- Gaia query construction

def test_gaia_query_uses_an_or_clause_when_the_field_wraps_ra_zero():
    provider = prov.GaiaOnlineProvider()
    captured = {}

    def fake_query(ra_range, dec_range, max_magnitude):
        captured['ra_range'] = ra_range
        raise RuntimeError('stop here')

    provider._query = fake_query
    with pytest.raises(RuntimeError):
        provider.lookup((350.0, 10.0), (0.0, 5.0))
    assert captured['ra_range'] == (350.0, 10.0)


def test_gaia_provider_has_no_bright_star_floor():
    """The old query said 'BETWEEN 3 AND max', dropping the 150 sources with G<3."""
    import inspect
    source = inspect.getsource(prov.GaiaOnlineProvider._query)
    assert 'BETWEEN 3' not in source
    assert 'phot_g_mean_mag <' in source


def test_the_default_catalogue_follows_what_is_installed(monkeypatch):
    """'gaia' is the offline archive plus the bright fill, or the online archive
    until one is installed -- the v1.2.0 offline-first default."""
    from mee2024.starcat import download
    monkeypatch.setattr(download, 'installed_catalogues', lambda: [])
    assert isinstance(prov.build('gaia'), prov.GaiaOnlineProvider)

    built = {}
    monkeypatch.setattr(download, 'installed_catalogues', lambda: ['pretend'])
    monkeypatch.setattr(prov.GaiaOfflineProvider, 'from_installed',
                        classmethod(lambda cls, *a, **k: built.setdefault(
                            'offline', object.__new__(prov.GaiaOfflineProvider))))
    merged = prov.build('gaia')
    assert isinstance(merged, prov.MergedProvider)
    assert merged.primary is built['offline']


def test_user_catalogues_are_a_curated_pair():
    names = [name for name, _, _ in prov.USER_CATALOGUES]
    assert names == ['gaia', 'gaia_online']
    # the building blocks stay reachable by name for tests and advanced use
    for name in ('tycho', 'hipparcos', 'gaia_offline', 'merged', 'merged_offline'):
        assert name in prov.known_catalogues()


# ------------------------------- overlapping archives (the v1.2.0 duplicate bug)

class _FakeArchive:
    """Just enough of OfflineCatalogue for the selection rule."""

    def __init__(self, name, bright, faint):
        self.manifest = {'name': name}
        self.magnitude_min = bright
        self.magnitude_limit = faint


def test_a_superset_archive_wins_over_the_parts_it_contains():
    """The bug this rule exists for: reading g13 alongside the g12 + 12<G<13 pair it
    was merged from lists every star twice, and duplicate entries make every match
    look ambiguous -- so a good field matches nothing at all."""
    standard = _FakeArchive('gaia_dr3_g13', 1.7, 13.0)
    base = _FakeArchive('gaia_dr3_g12', 1.7, 12.0)
    extension = _FakeArchive('gaia_dr3_g12_13', 12.0, 13.0)
    compact = _FakeArchive('gaia_dr3_g10', 1.7, 10.0)

    chosen = prov.choose_non_overlapping([standard, base, extension, compact])
    assert [c.manifest['name'] for c in chosen] == ['gaia_dr3_g13']


def test_the_disjoint_pair_is_still_read_together():
    """Without the standard archive, base + extension are complementary and both
    belong: the extension alone holds nothing brighter than G=12."""
    base = _FakeArchive('gaia_dr3_g12', 1.7, 12.0)
    extension = _FakeArchive('gaia_dr3_g12_13', 12.0, 13.0)
    chosen = prov.choose_non_overlapping([extension, base])
    assert [c.manifest['name'] for c in chosen] == ['gaia_dr3_g12', 'gaia_dr3_g12_13']


def test_a_lone_compact_archive_is_used():
    compact = _FakeArchive('gaia_dr3_g10', 1.7, 10.0)
    assert prov.choose_non_overlapping([compact]) == [compact]


def test_a_deeper_archive_extends_a_compact_one():
    compact = _FakeArchive('gaia_dr3_g10', 1.7, 10.0)
    deep = _FakeArchive('gaia_dr3_g15', 1.7, 15.0)
    chosen = prov.choose_non_overlapping([compact, deep])
    assert [c.manifest['name'] for c in chosen] == ['gaia_dr3_g15']
