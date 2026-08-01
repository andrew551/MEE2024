"""Resolving which archives a catalogue needs, and how deep it reaches.

The two published archives are disjoint magnitude slices rather than a base and a
superset, which is the source of nearly every way this can be got wrong -- so most of
these tests are about that distinction.
"""

import pytest

from mee2024.starcat import download


@pytest.fixture
def nothing_installed(monkeypatch):
    monkeypatch.setattr(download, 'installed_catalogues', lambda: [])
    for release in download.RELEASES.values():
        monkeypatch.setattr(release, 'is_installed', lambda: False)


@pytest.fixture
def base_installed(monkeypatch):
    base = download.RELEASES['gaia_dr3_g12']
    monkeypatch.setattr(download, 'installed_catalogues', lambda: [base])
    monkeypatch.setattr(base, 'is_installed', lambda: True)
    monkeypatch.setattr(download.RELEASES['gaia_dr3_g12_13'], 'is_installed',
                        lambda: False)


# ------------------------------------------------------------ releases_needed

def test_an_online_catalogue_needs_no_download(nothing_installed):
    assert download.releases_needed('gaia_online') == []
    assert download.releases_needed('tycho') == []
    assert download.releases_needed('hipparcos') == []


def test_the_default_catalogue_wants_an_archive_but_does_not_require_one(
        nothing_installed):
    """'gaia' prefers the archive -- so selecting it triggers the first-use
    download -- but falls back to the online archive rather than failing."""
    assert download.releases_needed('gaia') == [download.preferred_release()]
    warnings = download.prepare_catalogue('gaia', allow_download=False)
    assert any('online' in w for w in warnings)


def test_the_offline_catalogue_needs_the_standard_archive(nothing_installed):
    assert download.releases_needed('gaia_offline') == ['gaia_dr3_g13']
    assert download.releases_needed('merged_offline') == ['gaia_dr3_g13']


def test_the_offline_catalogue_needs_nothing_once_an_archive_is_present(base_installed):
    """It reads whatever is installed, so the base alone is enough to run."""
    assert download.releases_needed('gaia_offline') == []


def test_naming_an_archive_directly_needs_that_archive(base_installed):
    assert download.releases_needed('gaia_dr3_g12') == []
    assert download.releases_needed('gaia_dr3_g12_13') == ['gaia_dr3_g12_13']


def test_an_unknown_catalogue_name_needs_nothing(nothing_installed):
    """Resolution must not raise on a name it does not own; the provider reports that."""
    assert download.releases_needed('something_else') == []


# -------------------------------------------------- effective_magnitude_limit

def test_the_online_archive_reports_no_practical_limit():
    assert download.effective_magnitude_limit('gaia_online') is None


def test_the_offline_depth_is_the_deepest_installed_archive(base_installed):
    assert download.effective_magnitude_limit('gaia_offline') == 12.0


def test_installing_the_extension_deepens_the_offline_catalogue(monkeypatch):
    monkeypatch.setattr(download, 'installed_catalogues',
                        lambda: [download.RELEASES['gaia_dr3_g12'],
                                 download.RELEASES['gaia_dr3_g12_13']])
    assert download.effective_magnitude_limit('gaia_offline') == 13.0
    assert download.effective_magnitude_limit('gaia') == 13.0


def test_the_offline_depth_is_unknown_with_nothing_installed(nothing_installed):
    assert download.effective_magnitude_limit('gaia_offline') is None


# ----------------------------------------------------------- the depth warning

def test_no_warning_when_the_request_fits():
    assert download.magnitude_warning('gaia_offline', 12.0, 13.0) is None
    assert download.magnitude_warning('gaia_offline', 13.0, 13.0) is None


def test_no_warning_when_the_depth_is_unknown():
    """An unlimited catalogue must not produce a warning about a limit it does not have."""
    assert download.magnitude_warning('gaia', 20.0, None) is None


def test_a_shallow_catalogue_warns_and_names_the_remedy():
    text = download.magnitude_warning('gaia_offline', 14.0, 12.0)
    assert 'G<12' in text and 'magnitude 14' in text
    assert 'gaia_dr3_g13' in text


def test_at_full_depth_the_advice_is_to_go_online():
    """Nothing local goes past G=13, so recommending another archive would be wrong."""
    text = download.magnitude_warning('gaia_offline', 15.0, 13.0)
    assert 'gaia_dr3_g13' not in text
    assert 'online' in text


def test_the_warning_does_not_render_limits_as_floats():
    assert '13.0' not in download.magnitude_warning('gaia_offline', 14.0, 13.0)


# ------------------------------------------------------------ prepare_catalogue

def test_prepare_downloads_what_is_missing(monkeypatch, nothing_installed):
    fetched, notes = [], []
    monkeypatch.setattr(download, 'ensure_available',
                        lambda name, **kw: fetched.append(name))
    warnings = download.prepare_catalogue(
        'gaia_offline', options={'max_star_mag_dist': 12.0},
        on_note=notes.append)
    assert fetched == ['gaia_dr3_g13']
    assert any('downloading' in n for n in notes)
    assert warnings == []


def test_prepare_passes_a_progress_reporter_per_archive(monkeypatch, nothing_installed):
    seen = {}

    def fake_ensure(name, progress=None, **kw):
        seen[name] = progress

    monkeypatch.setattr(download, 'ensure_available', fake_ensure)
    sentinel = object()
    download.prepare_catalogue('gaia_offline', options={},
                               progress_for=lambda name: sentinel)
    assert seen == {'gaia_dr3_g13': sentinel}


def test_prepare_refuses_when_downloading_is_switched_off(monkeypatch, nothing_installed):
    monkeypatch.setattr(download, 'ensure_available',
                        lambda name, **kw: pytest.fail('must not download'))
    with pytest.raises(RuntimeError, match='--fetch gaia_dr3_g13'):
        download.prepare_catalogue('gaia_offline', options={}, allow_download=False)


def test_prepare_reports_a_depth_problem_it_cannot_fix(base_installed):
    warnings = download.prepare_catalogue(
        'gaia_offline', options={'max_star_mag_dist': 14.0})
    assert len(warnings) == 1 and 'G<12' in warnings[0]


def test_prepare_is_quiet_for_an_online_catalogue(nothing_installed):
    assert download.prepare_catalogue('gaia_online',
                                      options={'max_star_mag_dist': 20.0}) == []


# ----------------------------------------------------------------- the registry

def test_one_archive_is_the_recommended_setup():
    """The standard depth is a single archive; the rest are tiers or legacy."""
    assert download.RECOMMENDED_SETUP == ('gaia_dr3_g13',)
    assert download.DEFAULT_RELEASE == 'gaia_dr3_g13'
    assert all(download.RELEASES[n].recommended for n in download.RECOMMENDED_SETUP)


def test_a_fresh_install_is_offered_a_publishable_archive():
    """The first-use download must name an asset that actually exists."""
    name = download.preferred_release()
    release = download.RELEASES[name]
    assert release.is_published or release.is_installed()


def test_the_deep_archive_is_marked_as_an_extension():
    """Its role is what stops it being offered as a catalogue usable on its own."""
    assert download.RELEASES['gaia_dr3_g13'].role == 'base'
    assert download.RELEASES['gaia_dr3_g12'].role == 'legacy'
    assert download.RELEASES['gaia_dr3_g12_13'].role == 'extension'


def test_every_published_archive_carries_a_checksum_and_size():
    published = [r for r in download.RELEASES.values() if r.is_published]
    assert published, 'at least one archive must be fetchable'
    for release in published:
        assert release.sha256 and len(release.sha256) == 64
        assert release.size_bytes and release.size_bytes > 0
        assert release.url.endswith(f'/{release.name}.zip')
