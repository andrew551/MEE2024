"""The UI server, its API, and the run controller.

None of these open a window: the API is plain methods, and the HTTP layer is exercised
with urllib against a real server on an ephemeral localhost port.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from mee2024 import events
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


def test_default_catalogue_prefers_an_installed_offline_archive(api):
    from mee2024.starcat import download
    default = api._default_catalogue()
    installed = [r.name for r in download.RELEASES.values() if r.is_installed()]
    assert default in (installed or ['gaia'])


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
    ('auto', 'quintic', True), ('quick', 'cubic', False), ('deep', 'septic', True),
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
    with urllib.request.urlopen(f'http://127.0.0.1:{server.port}/', timeout=5) as r:
        html = r.read().decode('utf-8')
    assert r.status == 200
    assert '__MEE_TOKEN__' not in html, 'token placeholder was not substituted'
    assert server.token in html
    assert '<title>MEE2024</title>' in html


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
