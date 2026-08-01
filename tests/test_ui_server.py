"""The UI server, its API, and the run controller.

None of these open a window: the API is plain methods, and the HTTP layer is exercised
with urllib against a real server on an ephemeral localhost port.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from mee2024 import events
from mee2024.MEE2024util import _version
from mee2024.ui.runner import CancellableProgress, Cancelled, PipelineRunner
from mee2024.ui.server import Api, UiServer


@pytest.fixture
def api():
    return Api()


@pytest.fixture
def server():
    srv = UiServer(api=Api()).start()
    try:
        yield srv
    finally:
        srv.stop()


def get(server, path, token=None):
    request = urllib.request.Request(f'http://127.0.0.1:{server.port}{path}')
    request.add_header('X-MEE-Token', token if token is not None else server.token)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def post(server, path, payload, token=None):
    request = urllib.request.Request(
        f'http://127.0.0.1:{server.port}{path}',
        data=json.dumps(payload).encode(), method='POST')
    request.add_header('Content-Type', 'application/json')
    request.add_header('X-MEE-Token', token if token is not None else server.token)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


# ------------------------------------------------------------------ the API

def test_hello_describes_the_app(api):
    hello = api.hello()
    assert hello['version'].startswith('v')
    assert 'auto' in hello['presets']
    assert 'gaia' in hello['known_catalogues']
    assert hello['roots'], 'expected at least one browsable root'


def test_default_catalogue_prefers_offline_when_anything_is_installed(api):
    """'gaia_offline' rather than an archive name: it reads every archive present, so
    installing the deep extension deepens the default instead of being ignored."""
    from mee2024.starcat import download
    expected = 'gaia_offline' if download.installed_catalogues() else 'gaia'
    assert api._default_catalogue() == 'gaia'      # always: it reads what is installed


def test_browse_lists_directories_and_image_files(api, tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'frame1.fit').write_bytes(b'x' * 10)
    (tmp_path / 'frame2.FITS').write_bytes(b'x' * 20)
    (tmp_path / 'notes.txt').write_text('ignored')
    (tmp_path / '.hidden').write_text('ignored')

    listing = api.browse(str(tmp_path))
    assert [d['name'] for d in listing['directories']] == ['sub']
    assert sorted(f['name'] for f in listing['files']) == ['frame1.fit', 'frame2.FITS']
    assert listing['parent'] is not None
    assert listing['files'][0]['size'] > 0


def test_browse_rejects_a_file_path(api, tmp_path):
    target = tmp_path / 'a.fit'
    target.write_bytes(b'x')
    with pytest.raises(ValueError, match='not a directory'):
        api.browse(str(target))


def test_browse_rejects_a_missing_directory(api, tmp_path):
    with pytest.raises(ValueError):
        api.browse(str(tmp_path / 'nope'))


def test_state_is_idle_before_any_run(api):
    state = api.state()
    assert state['status'] == 'idle'
    assert state['events'] == []


def test_cancel_is_a_no_op_when_idle(api):
    assert api.cancel() == {'ok': False}


def test_start_requires_light_frames(api):
    with pytest.raises(ValueError, match='at least one light frame'):
        api.start({'lights': []})


def test_start_rejects_a_missing_file(api, tmp_path):
    with pytest.raises(ValueError, match='not found'):
        api.start({'lights': [str(tmp_path / 'ghost.fit')]})


# ------------------------------------------------------------------- options

@pytest.mark.parametrize('preset,order,guess', [
    ('auto', 'cubic', True), ('quick', 'cubic', False), ('deep', 'septic', True),
])
def test_presets_map_to_options(preset, order, guess):
    options = PipelineRunner().build_options({'preset': preset})
    assert options['distortionOrder'] == order
    assert options['guess_date'] is guess


def test_options_never_open_a_plot_window():
    """Everything the user should see travels as an event, not a matplotlib popup."""
    options = PipelineRunner().build_options({'preset': 'auto'})
    assert not options['flag_display']
    assert not options['flag_display2']
    assert not options['flag_display3']


def test_an_explicit_date_switches_off_guessing():
    options = PipelineRunner().build_options({'observation_date': '2023-10-29'})
    assert options['observation_date'] == '2023-10-29'
    assert options['guess_date'] is False


def test_raw_option_overrides_win():
    options = PipelineRunner().build_options(
        {'preset': 'auto', 'options': {'distortionOrder': 'linear', 'min_area': 9}})
    assert options['distortionOrder'] == 'linear'
    assert options['min_area'] == 9


def test_darks_and_flats_are_recorded_for_the_config():
    options = PipelineRunner().build_options({'darks': ['d1.fit', 'd2.fit'],
                                              'flats': ['f.fit']})
    assert options['-DARK-'] == 'd1.fit;d2.fit'
    assert options['-FLAT-'] == 'f.fit'


# ------------------------------------------------------------- cancellation

def test_cancellable_progress_raises_once_the_flag_is_set():
    flag = threading.Event()
    progress = CancellableProgress(flag)
    progress.update(1)          # fine
    flag.set()
    with pytest.raises(Cancelled):
        progress.update(2)


def test_a_cancelled_loop_stops_early():
    flag = threading.Event()
    progress = CancellableProgress(flag)
    seen = []

    def work(item):
        seen.append(item)
        if item == 2:
            flag.set()
        return item

    with pytest.raises(Cancelled):
        progress.loop([1, 2, 3, 4, 5], work)
    assert seen == [1, 2], 'should have stopped at the item that set the flag'


# --------------------------------------------------------------- run plumbing

def test_start_creates_a_missing_output_folder(api, tmp_path, monkeypatch):
    """A folder that does not exist yet must not fail deep inside the pipeline.

    Regression: do_stack used os.mkdir, which cannot create a parent, so choosing a
    not-yet-existing output folder died with a bare WinError 3 several layers down.
    """
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'x')
    target = tmp_path / 'new' / 'nested'

    import mee2024.stacker_implementation as si
    monkeypatch.setattr(si, 'do_stack', lambda *a, **k: tmp_path / 'z.zip')

    api.start({'lights': [str(frame)], 'output_dir': str(target), 'stages': ['stack']})
    api.runner.thread.join(timeout=30)
    assert target.is_dir()
    assert api.runner.snapshot()['status'] == 'done'


def test_start_reports_an_unusable_output_folder_immediately(api, tmp_path):
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'x')
    blocker = tmp_path / 'blocker'
    blocker.write_text('I am a file, not a folder')
    with pytest.raises(ValueError, match='output folder'):
        api.start({'lights': [str(frame)], 'output_dir': str(blocker / 'sub')})


def test_a_failed_run_does_not_look_like_an_api_error(api, tmp_path, monkeypatch):
    """The state payload reports a run failure in 'run_error', never in 'error'.

    Regression: the frontend treats a top-level 'error' key as a transport failure, so
    reusing that name for the run's own error broke polling exactly when a run failed.
    """
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'x')
    import mee2024.stacker_implementation as si
    monkeypatch.setattr(si, 'do_stack',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    api.start({'lights': [str(frame)], 'stages': ['stack']})
    api.runner.thread.join(timeout=30)

    state = api.state()
    assert state['status'] == 'failed'
    assert 'boom' in state['run_error']
    assert 'error' not in state


def test_a_failing_run_is_reported_as_failed(monkeypatch, tmp_path):
    """The worker must record the error rather than die silently in its thread."""
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'not really a fits file')

    runner = PipelineRunner()
    import mee2024.stacker_implementation as si
    monkeypatch.setattr(si, 'do_stack',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))

    runner.start({'lights': [str(frame)]})
    runner.thread.join(timeout=30)
    snapshot = runner.snapshot()
    assert snapshot['status'] == 'failed'
    assert 'boom' in snapshot['run_error']
    assert any(e['type'] == events.ERROR for e in snapshot['events'])


def test_a_run_emits_events_into_the_snapshot(monkeypatch, tmp_path):
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'x')
    runner = PipelineRunner()

    import mee2024.stacker_implementation as si

    def fake_stack(lights, darks, flats, options, progress=None):
        progress.loop([1, 2], lambda x: x, message='Finding all centroids...')
        return tmp_path / 'centroid_data.zip'

    monkeypatch.setattr(si, 'do_stack', fake_stack)
    runner.start({'lights': [str(frame)], 'stages': ['stack']})
    runner.thread.join(timeout=30)

    snapshot = runner.snapshot()
    assert snapshot['status'] == 'done'
    kinds = [e['type'] for e in snapshot['events']]
    assert events.STAGE_STARTED in kinds and events.PROGRESS in kinds
    assert snapshot['outputs']['centroid_zip'].endswith('centroid_data.zip')


def test_two_runs_cannot_overlap(monkeypatch, tmp_path):
    frame = tmp_path / 'a.fit'
    frame.write_bytes(b'x')
    runner = PipelineRunner()
    release = threading.Event()

    import mee2024.stacker_implementation as si
    monkeypatch.setattr(si, 'do_stack',
                        lambda *a, **k: (release.wait(10), tmp_path / 'z.zip')[1])

    runner.start({'lights': [str(frame)], 'stages': ['stack']})
    try:
        with pytest.raises(RuntimeError, match='already in progress'):
            runner.start({'lights': [str(frame)]})
    finally:
        release.set()
        runner.thread.join(timeout=30)


def test_snapshot_since_only_returns_new_events():
    runner = PipelineRunner()
    with events.using(runner.bus):
        events.log('first')
        events.log('second')
    first_seq = runner.snapshot()['events'][0]['seq']
    assert len(runner.snapshot(since=first_seq)['events']) == 1


# --------------------------------------------------------------------- HTTP

def test_frontend_is_served_with_the_token_substituted(server):
    url = f'http://127.0.0.1:{server.port}/?token={server.token}'
    with urllib.request.urlopen(url, timeout=5) as r:
        html = r.read().decode('utf-8')
    assert r.status == 200
    assert '__MEE_TOKEN__' not in html, 'token placeholder was not substituted'
    assert server.token in html
    assert '<title>MEE2024</title>' in html
    assert '__MEE_AUTHORS__' not in html
    assert 'Andrew Smith and Douglas Smith' in html


def test_the_frontend_itself_needs_the_token(server):
    """The page embeds the session token, so serving it unauthenticated would leak it
    to any other local process that can reach the port."""
    for url in (f'http://127.0.0.1:{server.port}/',
                f'http://127.0.0.1:{server.port}/?token=wrong'):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(url, timeout=5)
        assert excinfo.value.code == 403


def test_api_requires_the_token(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, '/api/hello', token='wrong')
    assert excinfo.value.code == 403


def test_api_accepts_the_token_in_the_query_string(server):
    """The frontend is opened with ?token=..., so that path must work too."""
    url = f'http://127.0.0.1:{server.port}/api/hello?token={server.token}'
    with urllib.request.urlopen(url, timeout=5) as response:
        assert json.loads(response.read())['version'].startswith('v')


def test_hello_and_state_over_http(server):
    status, hello = get(server, '/api/hello')
    assert status == 200 and 'presets' in hello
    status, state = get(server, '/api/state?since=0')
    assert status == 200 and state['status'] == 'idle'


def test_browse_over_http(server, tmp_path):
    (tmp_path / 'x.fit').write_bytes(b'x')
    import urllib.parse
    status, listing = get(
        server, '/api/browse?path=' + urllib.parse.quote(str(tmp_path)))
    assert status == 200
    assert [f['name'] for f in listing['files']] == ['x.fit']


def test_run_over_http_reports_a_bad_request(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        post(server, '/api/run', {'lights': []})
    assert excinfo.value.code == 400
    assert 'light frame' in excinfo.value.read().decode()


def test_unknown_endpoints_are_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, '/api/nope')
    assert excinfo.value.code == 404


def test_server_binds_only_to_loopback():
    srv = UiServer(api=Api())
    try:
        assert srv.httpd.server_address[0] == '127.0.0.1'
        assert len(srv.token) > 20
    finally:
        srv.stop()


def test_server_can_be_used_as_a_context_manager():
    with UiServer(api=Api()) as srv:
        status, hello = get(srv, '/api/hello')
        assert status == 200


def test_url_carries_the_token():
    srv = UiServer(api=Api())
    try:
        assert srv.token in srv.url
        assert srv.url.startswith('http://127.0.0.1:')
    finally:
        srv.stop()


# ------------------------------------------------------------------ frontend

def test_frontend_is_self_contained():
    """No CDN, no external fetches: the app must work with no internet."""
    from mee2024.ui.server import FRONTEND
    html = FRONTEND.read_text(encoding='utf-8')
    for forbidden in ('http://', 'https://', '//cdn', 'integrity='):
        assert forbidden not in html.replace('http://127.0.0.1', ''), \
            f'frontend references {forbidden}'


def test_frontend_handles_every_event_type():
    """A new event type should not silently vanish in the UI."""
    from mee2024.ui.server import FRONTEND
    html = FRONTEND.read_text(encoding='utf-8')
    for event_type in events.ALL_TYPES:
        assert f"case '{event_type}'" in html, f'frontend ignores {event_type}'


# ------------------------------------------------------- watch mode over the API

def test_hello_reports_authors_and_watch_defaults(api):
    hello = api.hello()
    assert hello['authors'] == 'Andrew Smith and Douglas Smith'
    # against _version() rather than a literal, so a release does not need a test edit
    assert hello['version'] == _version()
    assert hello['version'].startswith('v')
    defaults = hello['watch_defaults']
    assert defaults['settle_seconds'] == 10.0
    assert defaults['batch_size'] >= 1


def test_watch_start_requires_a_real_folder(api):
    with pytest.raises(ValueError, match='choose a folder'):
        api.watch_start({})
    with pytest.raises(ValueError, match='not a folder'):
        api.watch_start({'folder': 'no/such/place'})


def test_watch_start_stop_and_state(api, tmp_path):
    api.watch_start({'folder': str(tmp_path), 'settle_seconds': 0.01,
                     'batch_size': 9, 'poll_seconds': 0.01})
    watch = api.state()['watch']
    assert watch['running'] is True
    assert watch['folder'] == str(tmp_path)
    assert watch['batch_size'] == 9
    assert api.watch_stop() == {'ok': True}
    assert api.state()['watch']['running'] is False


def test_watch_refuses_two_watches_at_once(api, tmp_path):
    api.watch_start({'folder': str(tmp_path), 'poll_seconds': 0.01})
    try:
        with pytest.raises(RuntimeError, match='already watching'):
            api.watch_start({'folder': str(tmp_path)})
    finally:
        api.watch_stop()


def test_watch_flush_is_harmless_with_nothing_pending(api, tmp_path):
    assert api.watch_flush() == {'ok': False}
    api.watch_start({'folder': str(tmp_path), 'poll_seconds': 0.01})
    try:
        assert api.watch_flush() == {'ok': False}
    finally:
        api.watch_stop()


def test_state_has_no_watch_block_before_watching(api):
    assert api.state()['watch'] is None


def test_watch_endpoints_are_routed_over_http(server, tmp_path):
    status, payload = post(server, '/api/watch/start',
                           {'folder': str(tmp_path), 'poll_seconds': 0.01})
    assert status == 200 and payload['ok'] is True
    try:
        status, state = get(server, '/api/state?since=0')
        assert state['watch']['running'] is True
    finally:
        post(server, '/api/watch/stop', {})


# --------------------------------------------------------- catalogue download

def test_catalogues_report_whether_they_can_be_downloaded(api):
    for entry in api.hello()['catalogues']:
        assert 'downloadable' in entry
        assert 'installed' in entry


def test_fetch_without_a_configured_url_explains_what_to_do(api, monkeypatch):
    from mee2024.starcat import download

    release = download.RELEASES['gaia_dr3_g12']
    monkeypatch.setattr(release, 'url', None)
    monkeypatch.setattr(release, 'is_installed', lambda: False)
    with pytest.raises(ValueError, match='--set-source'):
        api.fetch_catalogue('gaia_dr3_g12')


def test_fetch_of_an_installed_catalogue_is_a_no_op(api, monkeypatch):
    from mee2024.starcat import download

    release = download.RELEASES['gaia_dr3_g12']
    monkeypatch.setattr(release, 'is_installed', lambda: True)
    assert api.fetch_catalogue('gaia_dr3_g12') == {'ok': True, 'already': True}


def test_catalogues_carry_their_depth_and_recommendation(api):
    entries = {c['name']: c for c in api.hello()['catalogues']}
    assert entries['gaia_dr3_g13']['role'] == 'base'
    assert entries['gaia_dr3_g13']['magnitude_limit'] == 13.0
    assert entries['gaia_dr3_g10']['role'] == 'compact'
    # exactly the standard archive is badged; the other tiers are not
    assert [c['name'] for c in entries.values() if c['recommended']] == ['gaia_dr3_g13']


def test_superseded_archives_are_not_offered_for_download(api, monkeypatch):
    """Listing them beside the archive that replaced them invites the same stars twice."""
    from mee2024.starcat import download

    for name in ('gaia_dr3_g12', 'gaia_dr3_g12_13'):
        monkeypatch.setattr(download.RELEASES[name], 'is_installed', lambda: False)
    assert 'gaia_dr3_g12' not in {c['name'] for c in api.hello()['catalogues']}


def test_a_superseded_archive_still_shows_while_it_is_installed(api, monkeypatch):
    """It can be selected as the catalogue for a run; hiding one in use would be worse."""
    from mee2024.starcat import download

    monkeypatch.setattr(download.RELEASES['gaia_dr3_g12'], 'is_installed', lambda: True)
    entries = {c['name']: c for c in api.hello()['catalogues']}
    assert entries['gaia_dr3_g12']['role'] == 'legacy'


def test_hello_reports_how_deep_each_catalogue_reaches(api):
    limits = api.hello()['catalogue_limits']
    # the online archive has no practical limit, so it must not read as a shallow one
    assert limits['gaia_online'] is None
    assert limits['gaia_dr3_g12'] == 12.0
    assert api.hello()['recommended_catalogue'] == 'gaia'


def test_the_default_catalogue_uses_every_installed_archive(api, monkeypatch):
    """Naming one archive would cap the run at that archive's own depth."""
    from mee2024.starcat import download

    monkeypatch.setattr(download, 'installed_catalogues',
                        lambda: [download.RELEASES['gaia_dr3_g12']])
    assert api._default_catalogue() == 'gaia'


def test_no_installed_archive_falls_back_to_the_online_archive(api, monkeypatch):
    from mee2024.starcat import download

    monkeypatch.setattr(download, 'installed_catalogues', lambda: [])
    assert api._default_catalogue() == 'gaia'


def test_download_progress_is_labelled_as_bytes():
    """A byte count rendered as an item count reads as gibberish in the progress bar."""
    from mee2024.progress import EventProgress

    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        progress = EventProgress(stage='download:x', label='Downloading x', unit='bytes')
        progress.start(137_952_319, 'Downloading x')
        progress.update(1_048_576)
        progress.finish()
    kinds = [e['type'] for e in sink.events]
    assert kinds == [events.STAGE_STARTED, events.PROGRESS, events.STAGE_FINISHED]
    assert all(e.get('unit') == 'bytes' for e in sink.events[:2])
    assert sink.events[0]['n_items'] == 137_952_319


# ------------------------------------------------------ catalogue preflight

def test_a_run_fetches_a_missing_catalogue_before_stacking(monkeypatch):
    """The download must happen up front, not after minutes of stacking work."""
    from mee2024.starcat import download

    fetched = []
    monkeypatch.setattr(download, 'releases_needed',
                        lambda catalogue, options=None: ['gaia_dr3_g12'])
    monkeypatch.setattr(download, 'ensure_available',
                        lambda name, **kw: fetched.append(name))
    monkeypatch.setattr(download, 'effective_magnitude_limit',
                        lambda catalogue, options=None: 12.0)

    runner = PipelineRunner()
    with events.using(runner.bus):
        runner.prepare_catalogue({'catalogue': 'gaia_offline', 'max_star_mag_dist': 12.0,
                                  'auto_download_catalogue': True})
    assert fetched == ['gaia_dr3_g12']


def test_a_run_refuses_rather_than_downloading_when_told_not_to(monkeypatch):
    from mee2024.starcat import download

    monkeypatch.setattr(download, 'releases_needed',
                        lambda catalogue, options=None: ['gaia_dr3_g12'])
    monkeypatch.setattr(download, 'ensure_available',
                        lambda name, **kw: pytest.fail('must not download'))

    runner = PipelineRunner()
    with pytest.raises(RuntimeError, match='--fetch gaia_dr3_g12'):
        runner.prepare_catalogue({'catalogue': 'gaia_offline',
                                  'auto_download_catalogue': False})


def test_a_run_warns_when_asked_for_stars_the_catalogue_lacks(monkeypatch):
    from mee2024.starcat import download

    monkeypatch.setattr(download, 'releases_needed',
                        lambda catalogue, options=None: [])
    monkeypatch.setattr(download, 'effective_magnitude_limit',
                        lambda catalogue, options=None: 13.0)

    runner = PipelineRunner()
    with events.using(runner.bus):
        runner.prepare_catalogue({'catalogue': 'gaia_offline', 'max_star_mag_dist': 14.0})
    warnings = [e for e in runner.sink.events
                if e['type'] == events.LOG and e.get('level') == 'warning']
    assert len(warnings) == 1 and 'G<13' in warnings[0]['text']


def test_no_warning_when_the_catalogue_is_deep_enough(monkeypatch):
    from mee2024.starcat import download

    monkeypatch.setattr(download, 'releases_needed', lambda catalogue, options=None: [])
    monkeypatch.setattr(download, 'effective_magnitude_limit',
                        lambda catalogue, options=None: 13.0)

    runner = PipelineRunner()
    with events.using(runner.bus):
        runner.prepare_catalogue({'catalogue': 'gaia_offline', 'max_star_mag_dist': 12.0})
    assert not [e for e in runner.sink.events if e.get('level') == 'warning']


# ------------------------------------------------- session lifetime and pickers

def test_the_wait_ends_when_the_page_says_goodbye():
    """Closing a browser tab used to leave the process running until the terminal
    was killed, because nothing told the server the page had gone."""
    import threading
    from mee2024.ui.server import Api, UiServer

    server = UiServer(api=Api())
    server.api.page_open = True          # a page is here
    outcome = {}

    def wait():
        outcome['reason'] = server.wait_until_closed(idle_seconds=30,
                                                     grace_seconds=0.2, poll=0.05)

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    time.sleep(0.2)
    assert waiter.is_alive(), 'must keep waiting while the page is open'
    server.api.goodbye()
    waiter.join(timeout=5)
    assert outcome['reason'] == 'closed'
    server.httpd.server_close()


def test_a_pagehide_that_is_not_a_close_does_not_end_the_session():
    """pagehide also fires for navigation and for the back/forward cache, from which
    a page can come back -- acting on it at once froze a live page."""
    import threading
    from mee2024.ui.server import Api, UiServer
    server = UiServer(api=Api())
    server.api.page_open = True
    outcome = {}
    waiter = threading.Thread(
        target=lambda: outcome.setdefault('reason', server.wait_until_closed(
            idle_seconds=30, grace_seconds=1.0, poll=0.05)), daemon=True)
    waiter.start()
    server.api.goodbye()
    time.sleep(0.3)
    server.api.ping()                 # the page came back (pageshow)
    time.sleep(1.5)
    assert waiter.is_alive() and 'reason' not in outcome
    server.httpd.server_close()


def test_a_heartbeat_keeps_an_idle_page_alive():
    """The page only polls during a run, so an idle one must be kept alive by its
    heartbeat -- otherwise the server shuts down under a live page, which is exactly
    what left a frozen tab behind."""
    import threading
    from mee2024.ui.server import Api, UiServer
    server = UiServer(api=Api())
    server.api.page_open = True
    outcome = {}
    waiter = threading.Thread(
        target=lambda: outcome.setdefault('reason', server.wait_until_closed(
            idle_seconds=0.6, grace_seconds=1.0, poll=0.05)), daemon=True)
    waiter.start()
    for _ in range(6):
        time.sleep(0.2)
        server.api.ping()
        server.api.last_seen = time.time()
    assert waiter.is_alive() and 'reason' not in outcome
    server.httpd.server_close()


def test_the_wait_gives_up_if_no_page_ever_arrives():
    from mee2024.ui.server import Api, UiServer
    server = UiServer(api=Api())
    assert server.wait_until_closed(idle_seconds=0.3, poll=0.05) == 'never opened'
    server.httpd.server_close()


def test_a_running_pipeline_keeps_the_session_alive(monkeypatch):
    """Tidiness must never cut a run short."""
    import threading
    from mee2024.ui.server import Api, UiServer
    server = UiServer(api=Api())
    monkeypatch.setattr(type(server.api.runner), 'is_running', property(lambda s: True))
    server.api.page_open = False
    outcome = {}
    waiter = threading.Thread(
        target=lambda: outcome.setdefault(
            'reason', server.wait_until_closed(idle_seconds=0.2, poll=0.05)),
        daemon=True)
    waiter.start()
    time.sleep(0.5)
    assert waiter.is_alive() and 'reason' not in outcome
    server.httpd.server_close()


def test_pick_reports_whether_a_native_dialog_exists(api):
    """Browser mode has no file dialog, and 'unavailable' must stay distinguishable
    from 'the user cancelled' -- the frontend falls back only for the first."""
    assert api.pick({'multiple': True}) == {'available': False, 'paths': []}

    api.native_dialog = lambda multiple=True, directory=False: ['/frames/a.fits']
    assert api.pick({'multiple': True}) == {'available': True,
                                           'paths': ['/frames/a.fits']}

    api.native_dialog = lambda multiple=True, directory=False: None
    assert api.pick({}) == {'available': True, 'paths': []}


def test_hello_offers_where_the_last_session_left_off(api):
    last = api.hello()['last']
    for key in ('work_dir', 'output_dir', 'catalogue', 'preset', 'distortion_order'):
        assert key in last
    assert api.hello()['config_path'].endswith('MEE_config.txt')
