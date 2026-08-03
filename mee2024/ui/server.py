"""
The local HTTP server behind the app.

Bound to 127.0.0.1 on an ephemeral port, and every API call must carry a per-session
token, so nothing outside this process can drive the pipeline or browse the filesystem.

The API is deliberately thin, and each endpoint is a plain method on ``Api`` taking and
returning JSON-able dicts, so tests exercise the behaviour without going through HTTP.
"""

import json
import os
import platform
import secrets
import string
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mee2024 import events
from mee2024.MEE2024util import AUTHORS, _version
from mee2024.ui.runner import PipelineRunner

FRONTEND = Path(__file__).parent / 'frontend.html'

IMAGE_SUFFIXES = {'.fit', '.fits', '.fts', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}


class Api:
    """The application logic. No HTTP in here."""

    def __init__(self, runner=None):
        self.runner = runner or PipelineRunner()
        self._fetching = None
        #: liveness, for the browser-mode watchdog (see UiServer.wait_until_closed)
        self.last_seen = time.time()
        self.page_open = False
        #: when the page said goodbye, if it has; any later request clears it
        self.closing_since = None
        self.native_dialog = None

    # ------------------------------------------------------------------ meta

    def hello(self):
        from mee2024.MEE2024util import get_config_path, read_ini
        from mee2024.config import get_default_options
        from mee2024.starcat import providers
        defaults = get_default_options()
        saved = get_default_options()
        read_ini(saved)          # the user's own settings, if they have run before
        return {
            # where the session should pick up from: the folder last used, the
            # catalogue and preset last chosen. Absent on a first run, and the
            # frontend keeps its own defaults then.
            'last': {
                'work_dir': saved.get('workDir') or '',
                'output_dir': saved.get('output_dir') or '',
                'catalogue': saved.get('catalogue') or defaults['catalogue'],
                'preset': saved.get('ui_preset') or 'auto',
                'distortion_order': saved.get('distortionOrder')
                or defaults['distortionOrder'],
            },
            'config_path': str(get_config_path()),
            # context for a bug report, gathered where it is actually known
            'platform': f'{platform.platform()} python {platform.python_version()}',
            'solver_info': self._solver_info(),
            'version': _version(),
            'authors': AUTHORS,
            'presets': self.runner.PRESETS,
            'catalogues': self._catalogues(),
            'default_catalogue': self._default_catalogue(),
            # the curated pair, not every registered provider: Tycho and Hipparcos
            # alone are building blocks, and 'merged' is what 'gaia' now means
            'known_catalogues': [name for name, _, _ in providers.USER_CATALOGUES],
            'catalogue_labels': {name: label
                                 for name, label, _ in providers.USER_CATALOGUES},
            'catalogue_notes': {name: note
                                for name, _, note in providers.USER_CATALOGUES},
            'catalogue_limits': self._catalogue_limits(),
            'recommended_catalogue': self._recommended_catalogue(),
            'roots': self.roots(),
            'watch_defaults': {
                'settle_seconds': defaults['watch_settle_seconds'],
                'batch_size': defaults['watch_batch_size'],
                'quiet_seconds': defaults['watch_quiet_seconds'],
            },
        }

    def _solver_info(self):
        """Which solver and which databases a run would actually use."""
        from mee2024.config import get_default_options
        from mee2024.MEE2024util import read_ini
        options = get_default_options()
        read_ini(options)
        which = options.get('platesolver', 'v2')
        if which != 'v2':
            return 'triangle (classic)'
        try:
            from mee2024.platesolve2 import pattern_db
            layers = [db.name for db in pattern_db.resolve_layers(options)]
            return 'v2 [' + ', '.join(layers) + ']'
        except Exception as exc:
            return f'v2 (no pattern database: {exc.__class__.__name__})'

    def _catalogues(self):
        from mee2024.config import get_default_options
        from mee2024.MEE2024util import read_ini
        from mee2024.starcat import download
        options = get_default_options()
        read_ini(options)
        out = []
        for name in download.RELEASES:
            release = download.get_release(name, options=options)
            # a superseded archive is still listed if it is installed -- it can be
            # selected as the catalogue for a run, and hiding an archive someone is
            # actually using would be worse than the clutter -- but it is never offered
            # as a download
            if not (release.offered or release.is_installed()):
                continue
            out.append({'name': release.name, 'description': release.description,
                        'installed': release.is_installed(),
                        'size': release.human_size(),
                        'downloadable': release.is_published,
                        'magnitude_limit': release.magnitude_limit,
                        'role': release.role,
                        'recommended': release.recommended})
        return out

    def _catalogue_limits(self):
        """How deep each selectable catalogue reaches, so the UI can warn as you type.

        None means "no practical limit" -- the online archive. Only the *installed* depth
        is reported for the offline catalogues, since that is what a run would see.
        """
        from mee2024.config import get_default_options
        from mee2024.MEE2024util import read_ini
        from mee2024.starcat import download, providers
        options = get_default_options()
        read_ini(options)
        limits = {}
        for name in providers.known_catalogues():
            limits[name] = download.effective_magnitude_limit(name, options=options)
        for name in download.RELEASES:
            limits[name] = download.get_release(name, options=options).magnitude_limit
        return limits

    def fetch_catalogue(self, name):
        """Download a prebuilt catalogue. Runs in a thread so the UI stays responsive."""
        from mee2024.config import get_default_options
        from mee2024.MEE2024util import read_ini
        from mee2024.starcat import download

        if self._fetching:
            raise ValueError('a download is already in progress')
        options = get_default_options()
        read_ini(options)
        release = download.get_release(name, options=options)
        if release.is_installed():
            return {'ok': True, 'already': True}
        if not release.is_published:
            raise ValueError(
                f'no download URL is configured for {name}. Set one with '
                f'"mee2024 catalogue --set-source {name} --url URL --sha256 HASH", '
                f'or build it locally with tools/build_gaia_offline.py.')

        def work():
            from mee2024.progress import EventProgress

            with events.using(self.runner.bus):
                try:
                    progress = EventProgress(stage=f'download:{name}',
                                             label=f'Downloading {name}', unit='bytes')
                    directory = download.ensure_available(
                        name, progress=progress, options=options)
                    events.log(f'{name} installed at {directory}')
                except Exception as exc:
                    events.emit(events.ERROR, text=f'{type(exc).__name__}: {exc}')
                    events.log(f'download failed: {exc}', level='error')
                finally:
                    self._fetching = None

        self._fetching = threading.Thread(target=work, daemon=True)
        self._fetching.start()
        return {'ok': True, 'started': True}

    # ------------------------------------------------------------ watch mode

    def watch_start(self, spec):
        self.runner.start_watch(spec)
        return {'ok': True}

    def watch_stop(self):
        return {'ok': self.runner.stop_watch()}

    def watch_flush(self):
        return {'ok': self.runner.flush_watch()}

    def _default_catalogue(self):
        """Always 'gaia': it reads every installed archive, and falls back to the
        online archive only until one is installed. Naming a specific archive would
        cap the run at that archive's own depth."""
        return 'gaia'

    def _recommended_catalogue(self):
        return 'gaia'

    def roots(self):
        """Sensible starting points for the file picker."""
        home = Path.home()
        candidates = [home, home / 'Documents', home / 'Pictures', home / 'Desktop',
                      home / 'Downloads']
        roots = [{'label': p.name or str(p), 'path': str(p)}
                 for p in candidates if p.is_dir()]
        if os.name == 'nt':
            for letter in string.ascii_uppercase:
                drive = Path(f'{letter}:/')
                if drive.exists():
                    roots.append({'label': f'{letter}:', 'path': str(drive)})
        else:
            roots.append({'label': '/', 'path': '/'})
        return roots

    # ------------------------------------------------------------------ files

    def browse(self, path=None):
        """List one directory: its subdirectories and any image files in it."""
        target = Path(path).expanduser() if path else Path.home()
        try:
            target = target.resolve()
        except OSError:
            raise ValueError(f'cannot open {target}')
        if not target.is_dir():
            raise ValueError(f'not a directory: {target}')

        directories, files = [], []
        try:
            entries = sorted(target.iterdir(), key=lambda p: p.name.lower())
        except PermissionError:
            raise ValueError(f'permission denied: {target}')
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            try:
                if entry.is_dir():
                    directories.append({'name': entry.name, 'path': str(entry)})
                elif entry.suffix.lower() in IMAGE_SUFFIXES:
                    files.append({'name': entry.name, 'path': str(entry),
                                  'size': entry.stat().st_size})
            except OSError:
                continue
        parent = str(target.parent) if target.parent != target else None
        return {'path': str(target), 'parent': parent,
                'directories': directories, 'files': files}

    # -------------------------------------------------------------------- run

    def start(self, spec):
        self.runner.start(spec)
        return {'ok': True}

    def cancel(self):
        return {'ok': self.runner.cancel()}

    def scan_fields(self, spec=None):
        """Preview the fields a batch would process, without starting anything.

        Separate from starting the run so the user sees what a folder actually contains --
        and, more to the point, sees the refusal *before* twenty runs begin rather than
        after. Returns the field list, the frame counts, and any reason the walk stopped.
        """
        from mee2024.ui import batch

        spec = spec or {}
        folder = spec.get('folder')
        if not folder:
            return {'fields': [], 'info': {'truncated': 'no folder chosen'}}
        limit = int(spec.get('max_fields') or batch.DEFAULT_MAX_FIELDS)
        fields, info = batch.find_fields(folder, max_fields=limit)
        return {
            # the frame lists themselves are not sent: for twenty fields of a few hundred
            # frames that is a lot of JSON to render a count from
            'fields': [{'name': f['name'], 'relative': f['relative'],
                        'folder': f['folder'], 'n_frames': len(f['frames'])}
                       for f in fields],
            'info': info,
            'summary': batch.describe(fields, info),
        }

    def start_batch(self, spec):
        """Discover the fields under a folder and run each of them."""
        from mee2024.ui import batch

        folder = (spec or {}).get('folder')
        if not folder:
            raise ValueError('choose a folder of fields to process')
        limit = int(spec.get('max_fields') or batch.DEFAULT_MAX_FIELDS)
        fields, info = batch.find_fields(folder, max_fields=limit)
        if info.get('truncated'):
            raise ValueError(info['truncated'])
        self.runner.start(dict(spec, fields=fields, batch_root=str(folder)))
        return {'ok': True, 'n_fields': len(fields),
                'summary': batch.describe(fields, info)}

    def state(self, since=0):
        return self.runner.snapshot(since=int(since))

    def results(self):
        """Parsed contents of whatever the run produced, for the score cards."""
        import zipfile
        out = {}
        for key, member in (('centroid_zip', 'results.txt'),
                            ('distortion_zip', 'distortion_results.txt')):
            path = self.runner.outputs.get(key)
            if not path:
                continue
            try:
                with zipfile.ZipFile(path) as archive:
                    out[key] = json.load(archive.open(member))
            except Exception as exc:
                out[key] = {'error': str(exc)}
        return out

    def pick(self, spec=None):
        """Open the platform's own file dialog, if this session has one.

        Available in the native window (app.py installs it); in browser mode there is
        no such thing, so the answer is ``{'available': False}`` and the frontend
        falls back to its own picker. Cancelling returns no paths, which is different
        from being unavailable and must stay distinguishable.
        """
        spec = spec or {}
        if self.native_dialog is None:
            return {'available': False, 'paths': []}
        chosen = self.native_dialog(multiple=bool(spec.get('multiple', True)),
                                    directory=bool(spec.get('directory')))
        return {'available': True, 'paths': [str(p) for p in (chosen or [])]}

    def ping(self):
        """The page is still here. Cheapest possible call; cancels a pending close.

        The frontend beats this while it is open, because it does *not* poll when
        idle -- polling only runs during a run -- and an open page that makes no
        requests is indistinguishable from a closed one.
        """
        self.closing_since = None
        return {'ok': True}

    def goodbye(self):
        """The page says it is going away (tab closed, navigated off, browser quit).

        Sent as a beacon on pagehide. Deliberately *not* an immediate shutdown:
        pagehide also fires when a page is merely navigated within or frozen into
        the browser's back/forward cache, after which it can come back alive -- and
        acting on it at once left the user looking at a live page whose server had
        gone. So it only starts a countdown, which any later request cancels.
        """
        self.page_open = False
        self.closing_since = time.time()
        return {'ok': True}

    def reveal(self, path):
        """Open a folder in the platform file manager."""
        target = Path(path)
        directory = target if target.is_dir() else target.parent
        if not directory.is_dir():
            raise ValueError(f'no such folder: {directory}')
        import subprocess
        import sys
        if os.name == 'nt':
            os.startfile(str(directory))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(directory)])
        else:
            subprocess.Popen(['xdg-open', str(directory)])
        return {'ok': True}


class _Handler(BaseHTTPRequestHandler):
    server_version = 'MEE2024UI'
    api = None
    token = None

    def log_message(self, *args):
        pass  # the app has its own log; do not spam the console

    # ------------------------------------------------------------- utilities

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self, query):
        supplied = (self.headers.get('X-MEE-Token')
                    or (query.get('token', [None])[0]))
        return supplied == self.token

    # ---------------------------------------------------------------- routing

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path in ('/', '/index.html'):
            # The served page carries the session token so its own API calls can
            # authenticate. That makes the page itself a credential, so it must not be
            # handed out unauthenticated: any other local process could otherwise simply
            # GET / and read the token out of the HTML.
            if not self._authorised(query):
                self._send_json({'error': 'bad or missing token'}, 403)
                return
            self._serve_frontend()
            return
        if not parsed.path.startswith('/api/'):
            self._send_json({'error': 'not found'}, 404)
            return
        if not self._authorised(query):
            self._send_json({'error': 'bad token'}, 403)
            return
        # any authorised call means the page is still there, and cancels a countdown
        # started by a pagehide that turned out not to be a close
        self.api.last_seen = time.time()
        self.api.page_open = True
        self.api.closing_since = None
        try:
            if parsed.path == '/api/ping':
                self._send_json(self.api.ping())
            elif parsed.path == '/api/hello':
                self._send_json(self.api.hello())
            elif parsed.path == '/api/state':
                self._send_json(self.api.state(query.get('since', ['0'])[0]))
            elif parsed.path == '/api/browse':
                self._send_json(self.api.browse(query.get('path', [None])[0]))
            elif parsed.path == '/api/results':
                self._send_json(self.api.results())
            else:
                self._send_json({'error': 'not found'}, 404)
        except Exception as exc:
            self._send_json({'error': str(exc)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authorised(query):
            self._send_json({'error': 'bad token'}, 403)
            return
        length = int(self.headers.get('Content-Length') or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            self._send_json({'error': 'bad JSON'}, 400)
            return
        try:
            if parsed.path == '/api/run':
                self._send_json(self.api.start(payload))
            elif parsed.path == '/api/cancel':
                self._send_json(self.api.cancel())
            elif parsed.path == '/api/batch/scan':
                self._send_json(self.api.scan_fields(payload))
            elif parsed.path == '/api/batch/run':
                self._send_json(self.api.start_batch(payload))
            elif parsed.path == '/api/reveal':
                self._send_json(self.api.reveal(payload.get('path')))
            elif parsed.path == '/api/watch/start':
                self._send_json(self.api.watch_start(payload))
            elif parsed.path == '/api/watch/stop':
                self._send_json(self.api.watch_stop())
            elif parsed.path == '/api/watch/flush':
                self._send_json(self.api.watch_flush())
            elif parsed.path == '/api/catalogue/fetch':
                self._send_json(self.api.fetch_catalogue(payload.get('name')))
            elif parsed.path == '/api/goodbye':
                self._send_json(self.api.goodbye())
            elif parsed.path == '/api/pick':
                self._send_json(self.api.pick(payload))
            else:
                self._send_json({'error': 'not found'}, 404)
        except Exception as exc:
            self._send_json({'error': str(exc)}, 400)

    def _serve_frontend(self):
        try:
            html = FRONTEND.read_text(encoding='utf-8')
        except OSError:
            self._send_json({'error': 'frontend missing'}, 500)
            return
        html = html.replace('__MEE_TOKEN__', self.token)
        html = html.replace('__MEE_VERSION__', _version())
        html = html.replace('__MEE_AUTHORS__', AUTHORS)
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


class UiServer:
    """Serves the frontend and API on localhost."""

    def __init__(self, api=None, host='127.0.0.1', port=0):
        self.api = api or Api()
        self.token = secrets.token_urlsafe(24)
        handler = type('Handler', (_Handler,), {'api': self.api, 'token': self.token})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.thread = None

    @property
    def port(self):
        return self.httpd.server_address[1]

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/?token={self.token}'

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
        return self

    def wait_until_closed(self, idle_seconds=150.0, grace_seconds=3.0, poll=0.5):
        """Block until the page goes away, so `mee2024` returns to the prompt.

        Closing a browser tab tells the server nothing by itself, which is why the
        process used to outlive it. Two signals end the wait, and both are
        deliberately forgiving, because exiting under a page that is still open is
        far worse than exiting a few seconds late:

        * the page's goodbye beacon starts a ``grace_seconds`` countdown, which any
          later request cancels -- ``pagehide`` also fires for navigation and for
          the back/forward cache, from which a page can return;
        * silence for ``idle_seconds`` is the backstop for a browser that was killed
          outright. It is long because the page's heartbeat is throttled to about
          once a minute while its tab sits in the background, and a backgrounded tab
          is still very much open.

        A run or an active folder watch always keeps the server alive.
        """
        page_seen_ever = False
        while True:
            time.sleep(poll)
            if self.api.page_open:
                page_seen_ever = True
            if self.api.runner.is_running or getattr(
                    self.api.runner.watcher, 'running', False):
                self.api.last_seen = time.time()
                self.api.closing_since = None
                continue
            closing = self.api.closing_since
            if closing is not None and time.time() - closing > grace_seconds:
                return 'closed'
            if time.time() - self.api.last_seen > idle_seconds:
                return 'idle' if page_seen_ever else 'never opened'

    def stop(self):
        # shutdown() blocks for ever if serve_forever() was never entered, so only ask
        # for it when there is actually a serving loop to interrupt
        if self.thread is not None:
            self.httpd.shutdown()
            self.thread.join(timeout=5)
            self.thread = None
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
