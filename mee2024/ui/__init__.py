"""
The MEE2024 desktop application.

A local HTTP server (``server``) drives the pipeline and streams typed events; a
self-contained HTML frontend renders them. The same server serves both the native
window (pywebview) and a plain browser, so there is one transport and one frontend
regardless of how it is launched.

    from mee2024.ui import launch
    launch()                      # native window if available, else the browser
"""

from mee2024.ui.app import launch

__all__ = ['launch']
