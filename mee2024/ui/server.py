"""
The local HTTP server behind the app.

Bound to 127.0.0.1 on an ephemeral port, and every API call must carry a per-session
token, so nothing outside this process can drive the pipeline or browse the filesystem.

The API is deliberately thin, and each endpoint is a plain method on ``Api`` taking and
returning JSON-able dicts, so tests exercise the behaviour without going through HTTP.
"""

import json
import os
import secrets
import string
import threading
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

    # ------------------------------------------------------------------ meta

    def hello(self):
        from mee2024.config import get_default_options
        from mee2024.starcat import providers
        defaults = get_default_options()
        return {
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

    def _catalogues(self):
        from mee2024.config import get_default_options
        from mee2024.MEE2024util import read_ini
        from mee2024.starcat import download
        options = get_default_options()
        read_ini(options)
        out = []
        for name in download.RELEASES:
            release = download.get_release(name, options=options)
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
        try:
            if parsed.path == '/api/hello':
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
