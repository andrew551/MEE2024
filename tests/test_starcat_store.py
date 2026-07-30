"""The offline catalogue on-disk format: write, memory-mapped read, verify, package.

Exercised against a synthetic catalogue and against the real captured Gaia fixture, so
the format is proven end to end before a 139 MB archive is ever built.
"""

import functools
import json

import numpy as np
import pytest

from mee2024.starcat import ORIGIN_GAIA, StarTable
from mee2024.starcat import providers as prov
from mee2024.starcat import store
from tests.fixture_catalogue import load_gaia_rows


def synthetic_table(n=2000, seed=0, epoch=2016.0):
    rng = np.random.default_rng(seed)
    return StarTable(
        ra=np.radians(rng.uniform(0, 360, n)),
        dec=np.radians(np.degrees(np.arcsin(rng.uniform(-1, 1, n)))),
        mag=rng.uniform(3, 12, n),
        ids=rng.integers(1, 2**62, n, dtype=np.int64),
        epoch=epoch, origin=ORIGIN_GAIA, band='G',
        pmra=rng.normal(0, 20, n), pmdec=rng.normal(0, 20, n),
        parallax=np.abs(rng.normal(5, 3, n)),
        radial_velocity=rng.normal(0, 30, n),
        nn_sep=rng.uniform(0.5, 60, n), nn_mag=rng.uniform(8, 17, n))


def table_from_fixture(name='zwo3_zenith'):
    rows = load_gaia_rows(name)
    return StarTable(
        ra=np.radians(np.array(rows['ra'], dtype=float)),
        dec=np.radians(np.array(rows['dec'], dtype=float)),
        mag=np.array(rows['phot_g_mean_mag'], dtype=float),
        ids=np.array(rows['source_id'], dtype=np.int64),
        epoch=float(np.median(rows['ref_epoch'])),
        origin=ORIGIN_GAIA, band='G',
        pmra=np.array(rows['pmra'], dtype=float),
        pmdec=np.array(rows['pmdec'], dtype=float),
        parallax=np.array(rows['parallax'], dtype=float),
        radial_velocity=np.array(rows['radial_velocity'], dtype=float))


# ------------------------------------------------------------- dec band index

def test_dec_index_has_one_entry_per_band_plus_one():
    dec = np.radians(np.linspace(-89.5, 89.5, 500))
    index = store.build_dec_index(dec)
    assert index.shape == (store.N_DEC_BANDS + 1,)
    assert index[0] == 0
    assert index[-1] == 500


def test_dec_index_is_monotonic():
    dec = np.sort(np.radians(np.random.default_rng(1).uniform(-90, 90, 1000)))
    index = store.build_dec_index(dec)
    assert np.all(np.diff(index) >= 0)


def test_dec_index_rejects_unsorted_input():
    with pytest.raises(ValueError, match='sorted ascending'):
        store.build_dec_index(np.radians([10.0, -10.0]))


def test_dec_index_bands_contain_the_right_stars():
    dec_deg = np.array([-89.5, -0.5, 0.5, 44.2, 44.8, 89.5])
    index = store.build_dec_index(np.radians(dec_deg))
    # band for +44 degrees is k = 134, covering [44, 45)
    start, stop = index[134], index[135]
    assert stop - start == 2
    assert np.all((dec_deg[start:stop] >= 44) & (dec_deg[start:stop] < 45))


# ----------------------------------------------------------------- round trip

def test_write_then_read_round_trips_every_column(tmp_path):
    table = synthetic_table(500)
    store.write_catalogue(tmp_path, table, name='synthetic', magnitude_limit=12.0)
    catalogue = store.OfflineCatalogue(tmp_path)

    assert len(catalogue) == 500
    assert catalogue.epoch == table.epoch
    assert catalogue.band == 'G'

    everything = catalogue.lookup((0.0, 360.0), (-90.0, 90.0))
    assert len(everything) == 500
    # compare as sets keyed on id, since writing sorts by declination
    order_out = np.argsort(everything.ids)
    order_in = np.argsort(table.ids)
    assert np.array_equal(everything.ids[order_out], table.ids[order_in])
    assert np.allclose(everything.ra[order_out], table.ra[order_in])
    assert np.allclose(everything.dec[order_out], table.dec[order_in])
    assert np.allclose(everything.pmra[order_out], table.pmra[order_in], atol=1e-4)


def test_written_catalogue_is_sorted_by_declination(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(300), name='synthetic')
    dec = np.load(tmp_path / 'dec.npy')
    assert np.all(np.diff(dec) >= 0)


def test_19_digit_source_ids_survive_the_round_trip(tmp_path):
    """The whole catalogue is keyed on these; a float would corrupt them."""
    ids = np.array([1926193296392975104, 2577063532462714368, 4472832130942575872],
                   dtype=np.int64)
    table = StarTable(ra=np.radians([1.0, 2.0, 3.0]), dec=np.radians([1.0, 2.0, 3.0]),
                      mag=[8.0, 9.0, 10.0], ids=ids, epoch=2016.0)
    store.write_catalogue(tmp_path, table, name='ids')
    out = store.OfflineCatalogue(tmp_path).lookup((0.0, 10.0), (0.0, 10.0))
    assert set(out.ids.tolist()) == set(ids.tolist())


def test_columns_are_memory_mapped_not_loaded(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(1000), name='synthetic')
    catalogue = store.OfflineCatalogue(tmp_path)
    assert isinstance(catalogue._open('ra'), np.memmap)


# --------------------------------------------------------------------- lookup

def test_lookup_restricts_to_the_box(tmp_path):
    table = synthetic_table(5000, seed=3)
    store.write_catalogue(tmp_path, table, name='synthetic')
    catalogue = store.OfflineCatalogue(tmp_path)

    found = catalogue.lookup((100.0, 110.0), (20.0, 30.0))
    ra_deg, dec_deg = np.degrees(found.ra), np.degrees(found.dec)
    assert np.all((ra_deg >= 100.0) & (ra_deg <= 110.0))
    assert np.all((dec_deg >= 20.0) & (dec_deg <= 30.0))

    expected = np.sum((np.degrees(table.ra) >= 100) & (np.degrees(table.ra) <= 110)
                      & (np.degrees(table.dec) >= 20) & (np.degrees(table.dec) <= 30))
    assert len(found) == expected


def test_lookup_handles_the_ra_zero_wrap(tmp_path):
    ra = np.radians([1.0, 5.0, 180.0, 355.0, 359.0])
    dec = np.radians([10.0] * 5)
    table = StarTable(ra=ra, dec=dec, mag=[8.0] * 5,
                      ids=np.arange(1, 6, dtype=np.int64), epoch=2016.0)
    store.write_catalogue(tmp_path, table, name='wrap')
    found = store.OfflineCatalogue(tmp_path).lookup((350.0, 6.0), (0.0, 20.0))
    assert sorted(np.degrees(found.ra).round().astype(int)) == [1, 5, 355, 359]


def test_lookup_applies_the_magnitude_cut(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(2000, seed=5), name='synthetic')
    found = store.OfflineCatalogue(tmp_path).lookup(
        (0.0, 360.0), (-90.0, 90.0), max_magnitude=8.0)
    assert len(found) > 0
    assert np.all(found.mag < 8.0)


def test_lookup_propagates_to_the_requested_epoch(tmp_path):
    table = synthetic_table(200, seed=7, epoch=2016.0)
    store.write_catalogue(tmp_path, table, name='synthetic')
    catalogue = store.OfflineCatalogue(tmp_path)
    at_ref = catalogue.lookup((0.0, 360.0), (-90.0, 90.0))
    moved = catalogue.lookup((0.0, 360.0), (-90.0, 90.0), epoch=2026.0)
    assert at_ref.epoch == 2016.0
    assert moved.epoch == 2026.0
    assert not np.allclose(at_ref.ra, moved.ra)


def test_lookup_of_an_empty_region(tmp_path):
    table = StarTable(ra=np.radians([10.0]), dec=np.radians([10.0]), mag=[8.0],
                      ids=np.array([1], dtype=np.int64), epoch=2016.0)
    store.write_catalogue(tmp_path, table, name='sparse')
    found = store.OfflineCatalogue(tmp_path).lookup((200.0, 210.0), (-80.0, -70.0))
    assert len(found) == 0


def test_lookup_only_reads_the_bands_it_needs(tmp_path):
    """The point of the dec index: a small box must not scan the whole catalogue."""
    table = synthetic_table(20000, seed=11)
    store.write_catalogue(tmp_path, table, name='synthetic')
    catalogue = store.OfflineCatalogue(tmp_path)
    start, stop = catalogue._dec_rows((44.0, 46.0))
    assert stop - start < len(table) / 10, (
        f'a 2-degree band touched {stop - start} of {len(table)} rows')


# ---------------------------------------------------------------- verification

def test_manifest_records_provenance_and_checksums(tmp_path):
    manifest = store.write_catalogue(
        tmp_path, synthetic_table(100), name='synthetic',
        catalogue='Gaia DR3', provenance='SELECT ... test', magnitude_limit=12.0)
    assert manifest['format'] == store.FORMAT
    assert manifest['n_stars'] == 100
    assert manifest['magnitude_limit'] == 12.0
    assert manifest['provenance'] == 'SELECT ... test'
    for entry in manifest['columns'].values():
        assert len(entry['sha256']) == 64


def test_verify_passes_on_a_freshly_written_catalogue(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(100), name='synthetic')
    assert store.verify(tmp_path) == []


def test_verify_detects_a_corrupted_column(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(100), name='synthetic')
    corrupt = np.load(tmp_path / 'mag.npy')
    corrupt[0] += 1.0
    np.save(tmp_path / 'mag.npy', corrupt)
    problems = store.verify(tmp_path)
    assert any('mag.npy' in p and 'checksum' in p for p in problems)


def test_verify_detects_a_missing_column(tmp_path):
    store.write_catalogue(tmp_path, synthetic_table(100), name='synthetic')
    (tmp_path / 'parallax.npy').unlink()
    assert any('missing' in p for p in store.verify(tmp_path))


def test_reading_a_foreign_directory_is_refused(tmp_path):
    (tmp_path / store.MANIFEST_FILE).write_text(json.dumps({'format': 'something-else'}),
                                                encoding='utf-8')
    with pytest.raises(ValueError, match='not a mee2024-starcat'):
        store.read_manifest(tmp_path)


def test_reading_a_future_format_version_is_refused(tmp_path):
    (tmp_path / store.MANIFEST_FILE).write_text(
        json.dumps({'format': store.FORMAT, 'format_version': 99}), encoding='utf-8')
    with pytest.raises(ValueError, match='format version'):
        store.read_manifest(tmp_path)


def test_missing_manifest_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='manifest'):
        store.read_manifest(tmp_path)


# -------------------------------------------------------------- packaging

def test_pack_and_unpack_round_trip(tmp_path):
    source = tmp_path / 'built'
    store.write_catalogue(source, synthetic_table(400), name='synthetic')
    archive = store.pack(source, tmp_path / 'synthetic.zip')
    assert archive.exists()

    extracted = store.unpack(archive, tmp_path / 'installed', verify_checksums=True)
    assert store.verify(extracted) == []
    assert len(store.OfflineCatalogue(extracted)) == 400


# ------------------------------------------------- the offline provider itself

def test_offline_provider_serves_the_real_captured_field(tmp_path):
    """Round-trip the actual Gaia response for zwo3 through the offline format."""
    table = table_from_fixture('zwo3_zenith')
    store.write_catalogue(tmp_path, table, name='zwo3', magnitude_limit=12.0)
    provider = prov.GaiaOfflineProvider(tmp_path)

    assert provider.is_offline
    assert provider.magnitude_limit == 12.0

    found = provider.lookup((354.2567, 358.0621), (44.0301, 46.2267),
                            max_magnitude=12.0, epoch=2023.839777)
    assert len(found) == len(table), 'the whole field should come back'
    assert found.epoch == pytest.approx(2023.839777)
    assert np.all(found.is_gaia())


@pytest.mark.parametrize('field,box', [
    ('zwo3_zenith', ((354.2567, 358.0621), (44.0301, 46.2267))),
    ('zwo1_zenith', ((26.0048, 31.1285), (42.7480, 47.8709))),
])
def test_offline_propagation_reproduces_gaias_own_answer(tmp_path, field, box):
    """The gate that decides whether going offline is acceptable.

    The fixture carries Gaia's server-side ESDC_EPOCH_PROP_POS positions at the
    observation epoch. Propagating the reference-epoch positions locally must land in the
    same place -- otherwise an offline catalogue would quietly shift every star.
    """
    from tests.fixture_catalogue import load_gaia_fixture

    rows, epoch = load_gaia_fixture(field)
    table = table_from_fixture(field)
    store.write_catalogue(tmp_path, table, name=field, magnitude_limit=12.0)
    offline = prov.GaiaOfflineProvider(tmp_path).lookup(*box, 12.0, epoch)

    truth = {int(sid): (ra, dec) for sid, ra, dec
             in zip(rows['source_id'], rows['ra_prop'], rows['dec_prop'])
             if not np.isnan(ra)}
    assert len(truth) > 500, 'fixture is missing its propagated reference positions'

    ra_truth = np.array([truth[int(i)][0] for i in offline.ids])
    dec_truth = np.array([truth[int(i)][1] for i in offline.ids])
    dra = (np.degrees(offline.ra) - ra_truth) * np.cos(offline.dec) * 3.6e6
    ddec = (np.degrees(offline.dec) - dec_truth) * 3.6e6
    sep_mas = np.hypot(dra, ddec)

    assert np.median(sep_mas) < 0.01
    assert np.max(sep_mas) < 0.1, (
        f'{field}: max disagreement with Gaia {np.max(sep_mas):.4f} mas')


def test_startable_propagation_beats_the_old_stardata_path():
    """StarTable passes radial velocity to apply_space_motion; StarData did not.

    For one star in zwo3 that is worth 2.45 mas. Small next to our 100 mas RMS, but the
    new path is the one that matches Gaia exactly, so this documents the improvement
    rather than treating it as a discrepancy.
    """
    from tests.fixture_catalogue import FixtureCatalogue, load_gaia_fixture

    rows, epoch = load_gaia_fixture('zwo3_zenith')
    box = ((354.2567, 358.0621), (44.0301, 46.2267))
    truth = {int(sid): (ra, dec) for sid, ra, dec
             in zip(rows['source_id'], rows['ra_prop'], rows['dec_prop'])
             if not np.isnan(ra)}

    def worst_error(ids, ra_rad, dec_rad):
        ra_truth = np.array([truth[int(i)][0] for i in ids])
        dec_truth = np.array([truth[int(i)][1] for i in ids])
        dra = (np.degrees(ra_rad) - ra_truth) * np.cos(dec_rad) * 3.6e6
        ddec = (np.degrees(dec_rad) - dec_truth) * 3.6e6
        return np.max(np.hypot(dra, ddec))

    new = table_from_fixture('zwo3_zenith').at_epoch(epoch)
    old = FixtureCatalogue('zwo3_zenith').lookup_objects(*box, 12.0, epoch)

    new_error = worst_error(new.ids, new.ra, new.dec)
    old_error = worst_error(old.get_ids(), old.get_ra(), old.get_dec())
    assert new_error < 0.1, f'StarTable worst error {new_error:.4f} mas'
    assert old_error > 1.0, f'expected the no-RV path to be worse, got {old_error:.4f} mas'


def test_offline_provider_caps_requests_at_its_own_depth(tmp_path, capsys):
    """Asking deeper than the archive goes must say so rather than quietly return less."""
    table = table_from_fixture('zwo3_zenith')
    store.write_catalogue(tmp_path, table, name='zwo3', magnitude_limit=12.0)
    provider = prov.GaiaOfflineProvider(tmp_path)
    provider.lookup((354.0, 358.0), (44.0, 46.0), max_magnitude=13.0)
    printed = capsys.readouterr().out
    assert 'G<12' in printed and 'magnitude 13' in printed
    # and it must name the way out, not merely complain
    assert 'gaia_dr3_g12_13' in printed


def test_depth_warning_reaches_the_event_bus(tmp_path):
    """The app window learns about truncation through events, not through stdout."""
    from mee2024 import events
    table = table_from_fixture('zwo3_zenith')
    store.write_catalogue(tmp_path, table, name='zwo3', magnitude_limit=12.0)
    provider = prov.GaiaOfflineProvider(tmp_path)
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        provider.lookup((354.0, 358.0), (44.0, 46.0), max_magnitude=13.0)
    warnings = [e for e in sink.events
                if e['type'] == events.LOG and e.get('level') == 'warning']
    assert len(warnings) == 1
    assert 'G<12' in warnings[0]['text']


def test_no_warning_when_the_request_fits_the_archive(tmp_path, capsys):
    table = table_from_fixture('zwo3_zenith')
    store.write_catalogue(tmp_path, table, name='zwo3', magnitude_limit=12.0)
    provider = prov.GaiaOfflineProvider(tmp_path)
    provider.lookup((354.0, 358.0), (44.0, 46.0), max_magnitude=11.0)
    assert 'note:' not in capsys.readouterr().out


def test_offline_provider_uses_precomputed_neighbour_flags(tmp_path):
    """Double-star flagging offline relies on nn_sep/nn_mag, not a fresh mag-17 query."""
    table = synthetic_table(50, seed=13)
    table.nn_sep[:5] = 1.0        # five tight pairs
    table.nn_mag[:5] = 12.0
    table.nn_sep[5:] = 100.0
    store.write_catalogue(tmp_path, table, name='synthetic')
    provider = prov.GaiaOfflineProvider(tmp_path)
    everything = provider.lookup((0.0, 360.0), (-90.0, 90.0))
    neighbours = provider.lookup_neighbours(everything, radius_arcsec=10.0,
                                            max_magnitude=17.0)
    assert len(neighbours) == 5


def test_offline_provider_stacks_a_base_and_a_deep_archive(tmp_path):
    """A G<12 base plus an optional 12<G<13 extension must read as one catalogue."""
    rng = np.random.default_rng(17)

    def band(n, lo, hi):
        return StarTable(ra=np.radians(rng.uniform(0, 360, n)),
                         dec=np.radians(rng.uniform(-10, 10, n)),
                         mag=rng.uniform(lo, hi, n),
                         ids=rng.integers(1, 2**62, n, dtype=np.int64),
                         epoch=2016.0, band='G')

    store.write_catalogue(tmp_path / 'base', band(100, 6, 12), name='base',
                          magnitude_limit=12.0)
    store.write_catalogue(tmp_path / 'deep', band(200, 12, 13), name='deep',
                          magnitude_limit=13.0)

    provider = prov.GaiaOfflineProvider([tmp_path / 'base', tmp_path / 'deep'])
    assert provider.magnitude_limit == 13.0
    everything = provider.lookup((0.0, 360.0), (-20.0, 20.0), max_magnitude=13.0)
    assert len(everything) == 300
    shallow = provider.lookup((0.0, 360.0), (-20.0, 20.0), max_magnitude=12.0)
    assert len(shallow) == 100


def test_offline_provider_requires_a_directory():
    with pytest.raises(ValueError, match='at least one'):
        prov.GaiaOfflineProvider([])


# ------------------------------------------------------- download and install

def test_release_urls_and_checksums_are_configured():
    """The registry must carry a real URL and hash, or auto-download cannot work."""
    from mee2024.starcat import download
    for name, release in download.RELEASES.items():
        assert release.is_published, f'{name} has no URL'
        assert release.url.startswith('https://'), f'{name}: {release.url}'
        assert release.sha256 and len(release.sha256) == 64, f'{name} has no sha256'
        assert release.n_stars > 1_000_000
        assert release.size_bytes > 1_000_000


def test_the_two_archives_are_disjoint_magnitude_slices():
    """G<12 and 12<G<13 are complementary, not a base and a superset.

    The provider concatenates whichever are installed, so installing both gives G<13.
    """
    from mee2024.starcat import download
    base = download.RELEASES['gaia_dr3_g12']
    deep = download.RELEASES['gaia_dr3_g12_13']
    assert base.magnitude_limit == 12.0
    assert deep.magnitude_limit == 13.0
    assert '12 < G < 13' in deep.description


def test_download_rejects_an_html_error_page(tmp_path, monkeypatch):
    """Drive and similar hosts serve interstitial or quota pages as HTTP 200 HTML.

    Saving that as a .zip fails much later with a baffling error, so refuse it up front.
    """
    import io
    from mee2024.starcat import download

    class FakeResponse(io.BytesIO):
        headers = {'Content-Type': 'text/html', 'Content-Length': '100'}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(download.urllib.request, 'urlopen',
                        lambda *a, **k: FakeResponse(b'<!DOCTYPE html><html>quota</html>'))
    with pytest.raises(ValueError, match='returned a web page'):
        download._download('https://example.test/x.zip', tmp_path / 'x.zip')
    assert not (tmp_path / 'x.zip.part').exists(), 'the partial file must be removed'


def test_download_verifies_the_checksum_and_installs(tmp_path, monkeypatch):
    """End to end over a real local HTTP server: fetch, verify, unpack, use."""
    import http.server
    import threading
    from mee2024.starcat import download

    served = tmp_path / 'served'
    served.mkdir()
    store.write_catalogue(tmp_path / 'built', synthetic_table(80), name='served',
                          magnitude_limit=12.0)
    archive = store.pack(tmp_path / 'built', served / 'served.zip')
    digest = store.sha256(archive)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(served))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{httpd.server_address[1]}/served.zip'
        target = tmp_path / 'fetched.zip'
        download._download(url, target, expected_sha256=digest)
        assert target.exists()

        # a wrong checksum must be rejected and leave nothing behind
        other = tmp_path / 'bad.zip'
        with pytest.raises(ValueError, match='checksum mismatch'):
            download._download(url, other, expected_sha256='0' * 64)
        assert not other.exists()
    finally:
        httpd.shutdown()
        httpd.server_close()

    installed = store.unpack(target, tmp_path / 'installed', verify_checksums=True)
    assert len(prov.GaiaOfflineProvider(installed).lookup((0, 360), (-90, 90))) == 80


def test_a_missing_asset_is_reported_as_actionable(tmp_path, monkeypatch):
    """Until the release assets exist the URLs 404; that must not be a traceback."""
    import urllib.error
    from mee2024.starcat import download

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError('u', 404, 'Not Found', {}, None)

    monkeypatch.setattr(download.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(RuntimeError, match='has not been uploaded'):
        download._download('https://example.test/gaia_dr3_g12.zip',
                           tmp_path / 'gaia_dr3_g12.zip')
