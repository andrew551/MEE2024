"""
Launching the app window.

Preference order:

1. **pywebview** -- a native window using the platform's own web view (WebView2 on
   Windows 10/11, WKWebView on macOS, webkit2gtk on Linux). No bundled browser engine,
   so packaging stays small.
2. **the default browser** -- same server, same frontend. The fallback for machines with
   no web view, and for remote or headless use.

Because both talk to the same local HTTP server, there is only one frontend to maintain.
"""

import sys
import threading
import webbrowser

from mee2024.MEE2024util import _version
from mee2024.ui.server import Api, UiServer

WINDOW_TITLE = f'MEE2024 {_version()}'
MIN_SIZE = (1024, 680)
DEFAULT_SIZE = (1320, 880)


def have_webview():
    try:
        import webview  # noqa: F401  -- probe only: is a native web view available?
        return True
    except Exception:
        return False


def launch(prefer_browser=False, host='127.0.0.1', port=0, block=True):
    """Start the server and open the UI. Returns the server (already running).

    prefer_browser: skip the native window even if pywebview is installed.
    block: when True, return only once the window (or the user's Ctrl-C) closes.
    """
    server = UiServer(api=Api(), host=host, port=port).start()
    print(f'MEE2024 UI serving at {server.url}')

    if not prefer_browser and have_webview():
        _run_native(server)
        server.stop()
        return server

    webbrowser.open(server.url)
    if block:
        print('Close this window or press Ctrl-C to quit.')
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print('\nshutting down')
        finally:
            server.stop()
    return server


def _run_native(server):
    import webview

    window = webview.create_window(
        WINDOW_TITLE, server.url,
        width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1],
        min_size=MIN_SIZE, background_color='#0b0e14')

    # A native file dialog is nicer than the built-in browser when one is available;
    # the frontend's own picker remains the portable path and the only one in browser mode.
    server.api.native_dialog = lambda multiple=True, directory=False: window.create_file_dialog(
        webview.FOLDER_DIALOG if directory
        else (webview.OPEN_DIALOG if not multiple else webview.OPEN_DIALOG),
        allow_multiple=multiple)

    gui = None
    if sys.platform.startswith('linux'):
        gui = 'gtk'  # webkit2gtk; qt is the other option if gtk is missing
    try:
        webview.start(gui=gui)
    except Exception as exc:
        print(f'native window failed ({exc}); falling back to the browser')
        webbrowser.open(server.url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
