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


def launch(prefer_browser=False, host='127.0.0.1', port=0, block=True,
           keep_alive=False):
    """Start the server and open the UI. Returns the server (already running).

    prefer_browser: skip the native window even if pywebview is installed.
    block: when True, return only once the window (or the user's Ctrl-C) closes.
    keep_alive: stay up even after the page goes away -- for serving a browser on
        another machine, or reloading the tab freely while developing.
    """
    server = UiServer(api=Api(), host=host, port=port).start()
    # flush: a frozen (PyInstaller) build buffers stdout when it is redirected or
    # has no console, and this URL is the only way to reach the UI
    print(f'MEE2024 UI serving at {server.url}', flush=True)

    if not prefer_browser and have_webview():
        _run_native(server)
        server.stop()
        return server

    # Say which interface this is and why, because the difference is visible: the
    # native window can open the platform's file dialogs and a browser tab cannot.
    if prefer_browser:
        print('opening in your browser (--browser)', flush=True)
    else:
        print('opening in your browser: pywebview is not installed, so there is no '
              'native window. `pip install pywebview` for the app window and native '
              'file dialogs.', flush=True)
    webbrowser.open(server.url)
    if block:
        print('Close the browser tab (or press Ctrl-C) to quit.', flush=True)
        try:
            if keep_alive:
                threading.Event().wait()
            else:
                reason = server.wait_until_closed()
                print(f'browser session ended ({reason}); shutting down', flush=True)
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
        # pywebview disables text selection by default, which made the activity log
        # impossible to select or copy -- so bug reports arrived as screenshots
        text_select=True,
        min_size=MIN_SIZE, background_color='#0b0e14')

    # The platform's own file dialog, which the frontend asks for through /api/pick.
    # Browser mode has no equivalent, so the built-in picker stays as the fallback.
    # `start` seeds where the dialog opens. Without it the OS falls back to its own
    # per-process last-visited folder, which is *shared* between the file and folder
    # dialogs -- so choosing an output folder re-aimed the next input dialog and
    # vice versa. The server passes a per-purpose folder instead (see Api.pick).
    def native_dialog(multiple=True, directory=False, start=None):
        return window.create_file_dialog(
            webview.FOLDER_DIALOG if directory else webview.OPEN_DIALOG,
            directory=start or '',
            allow_multiple=bool(multiple) and not directory,
            file_types=() if directory else (
                'Image frames (*.fit;*.fits;*.fts;*.tif;*.tiff;*.png;*.jpg;*.jpeg)',
                'All files (*.*)'))

    server.api.native_dialog = native_dialog

    gui = None
    if sys.platform.startswith('linux'):
        gui = 'gtk'  # webkit2gtk; qt is the other option if gtk is missing
    try:
        webview.start(gui=gui)
    except Exception as exc:
        print(f'native window failed ({exc}); falling back to the browser', flush=True)
        webbrowser.open(server.url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
