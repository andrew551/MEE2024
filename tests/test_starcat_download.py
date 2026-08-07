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


# ------------------------------------------------ opening a bundled catalogue by name

def _fake_manifests(monkeypatch):
    """Treat any directory holding a manifest.json as a readable catalogue.

    database_cache imports `store` inside the function, so the real module is what has to
    be patched rather than an attribute of database_cache.
    """
    from mee2024.starcat import store

    def read_manifest(directory):
        from pathlib import Path
        if (Path(directory) / 'manifest.json').exists():
            return {}
        raise FileNotFoundError(str(directory))

    monkeypatch.setattr(store, 'read_manifest', read_manifest)


def test_a_bundled_catalogue_can_be_opened_by_name(monkeypatch, tmp_path):
    """The executable ships gaia_dr3_g10 inside itself and lists it as installed, so it
    must also be openable. It was not: only the user's data directory was searched, so
    selecting the bundled archive fell through to the legacy Tycho reader and died on
    open('gaia_dr3_g10')."""
    from mee2024 import database_cache
    from mee2024.starcat import download as dl

    bundled = tmp_path / 'bundled' / 'gaia_dr3_g10'
    bundled.mkdir(parents=True)
    (bundled / 'manifest.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(dl.RELEASES['gaia_dr3_g10'], 'bundled_directory',
                        lambda: bundled)
    monkeypatch.setattr(dl.RELEASES['gaia_dr3_g10'], 'directory',
                        lambda: tmp_path / 'absent')
    _fake_manifests(monkeypatch)
    assert database_cache._installed_catalogue_dir('gaia_dr3_g10') == bundled


def test_an_installed_copy_wins_over_a_bundled_one(monkeypatch, tmp_path):
    """A downloaded archive is the deeper one, so it should be preferred."""
    from mee2024 import database_cache
    from mee2024.starcat import download as dl

    for name in ('installed', 'bundled'):
        (tmp_path / name / 'gaia_dr3_g10').mkdir(parents=True)
        (tmp_path / name / 'gaia_dr3_g10' / 'manifest.json').write_text('{}',
                                                                       encoding='utf-8')
    monkeypatch.setattr(dl.RELEASES['gaia_dr3_g10'], 'directory',
                        lambda: tmp_path / 'installed' / 'gaia_dr3_g10')
    monkeypatch.setattr(dl.RELEASES['gaia_dr3_g10'], 'bundled_directory',
                        lambda: tmp_path / 'bundled' / 'gaia_dr3_g10')
    _fake_manifests(monkeypatch)
    assert database_cache._installed_catalogue_dir('gaia_dr3_g10') == \
        tmp_path / 'installed' / 'gaia_dr3_g10'


def test_an_unknown_name_is_still_none():
    from mee2024 import database_cache
    assert database_cache._installed_catalogue_dir('not_a_catalogue_at_all') is None


def test_the_depth_advice_does_not_tell_you_to_install_what_you_have(monkeypatch):
    """The reported log advised installing gaia_dr3_g13 while it was installed and being
    used to build the pattern database."""
    monkeypatch.setattr(download.RELEASES['gaia_dr3_g13'], 'is_installed', lambda: True)
    text = download.magnitude_warning('gaia_dr3_g10', 13.0, 10.0)
    assert 'already installed' in text and 'Install gaia_dr3_g13' not in text


def test_the_depth_advice_says_install_when_it_really_is_missing(monkeypatch):
    monkeypatch.setattr(download.RELEASES['gaia_dr3_g13'], 'is_installed', lambda: False)
    text = download.magnitude_warning('gaia_dr3_g10', 13.0, 10.0)
    assert 'Install gaia_dr3_g13' in text


# ------------------------------------------------- large downloads need agreeing to

def test_only_the_multi_gigabyte_archive_needs_confirming():
    """The threshold has to separate the standard archive from the deep one, or it is
    either useless (everything gated) or pointless (nothing gated)."""
    assert not download.RELEASES['gaia_dr3_g13'].needs_confirmation, \
        'the recommended 320 MB archive must download without ceremony'
    assert download.RELEASES['gaia_dr3_g15'].needs_confirmation, \
        'a 1.57 GB download must be agreed to first'


def test_the_size_warning_gives_the_exact_size_and_names_the_alternative():
    """Vague warnings get clicked through. An exact byte count is a decision."""
    release = download.RELEASES['gaia_dr3_g15']
    message = release.size_warning()
    assert f'{release.size_bytes:,}' in message, 'the exact size must be quoted'
    assert '1.57 GB' in message
    assert 'gaia_dr3_g13' in message, 'must name the archive most users should take'
    # the archive and its unpacked copy coexist, so the disk cost is the sum of both
    assert release.disk_needed_bytes() > release.size_bytes
    assert download.RELEASES['gaia_dr3_g13'].size_warning() is None


def test_a_large_download_does_not_start_without_confirmation(monkeypatch):
    """The guard lives at the choke point, so no caller can bypass it by accident."""
    release = download.RELEASES['gaia_dr3_g15']
    monkeypatch.setattr(release, 'is_installed', lambda: False)

    def explode(*a, **k):
        raise AssertionError('the download started without being agreed to')

    monkeypatch.setattr(download, '_download', explode)
    with pytest.raises(download.ConfirmationRequired) as caught:
        download.ensure_available('gaia_dr3_g15')
    assert caught.value.size_bytes == release.size_bytes
    assert 'gaia_dr3_g13' in str(caught.value)

    # declining through a callback is the same as never agreeing
    with pytest.raises(download.ConfirmationRequired):
        download.ensure_available('gaia_dr3_g15', confirm=lambda r: False)


def test_confirming_lets_the_large_download_proceed(monkeypatch):
    """...and the same guard must not stand in the way once the user has said yes."""
    release = download.RELEASES['gaia_dr3_g15']
    monkeypatch.setattr(release, 'is_installed', lambda: False)
    started = []

    def record(url, destination, expected_sha256=None, progress=None):
        started.append(url)
        raise RuntimeError('stop here: the guard is what is under test, not the network')

    monkeypatch.setattr(download, '_download', record)
    for agreement in (True, lambda r: True):
        started.clear()
        with pytest.raises(RuntimeError, match='stop here'):
            download.ensure_available('gaia_dr3_g15', confirm=agreement)
        assert started, 'confirming did not let the download through'


def test_a_run_never_starts_a_multi_gigabyte_download_by_itself(monkeypatch):
    """Auto-download exists so a run does not fail for want of the standard archive.
    Silently spending an hour and 3.5 GB mid-run is a different thing entirely."""
    monkeypatch.setattr(download, 'releases_needed', lambda *a, **k: ['gaia_dr3_g15'])
    monkeypatch.setattr(download.RELEASES['gaia_dr3_g15'], 'is_installed', lambda: False)

    def explode(*a, **k):
        raise AssertionError('a run started a 1.57 GB download on its own')

    monkeypatch.setattr(download, 'ensure_available', explode)
    warnings = download.prepare_catalogue('gaia', options={})
    assert any('too large to fetch automatically' in w for w in warnings), warnings
    assert any('--fetch gaia_dr3_g15' in w for w in warnings), 'must say how to opt in'
