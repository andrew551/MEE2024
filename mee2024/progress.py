"""
Progress reporting, decoupled from any particular UI.

The pipeline used to call FreeSimpleGUI directly to put a progress bar on screen, which
meant the pipeline could not run without a display. It now takes a ``ProgressReporter``
instead: ``NullProgress`` for tests, ``TextProgress`` for the CLI, ``GuiProgress`` for
the GUI. FreeSimpleGUI is imported lazily inside ``GuiProgress`` so importing this module
never pulls in tkinter.
"""

import multiprocessing
import sys

from mee2024 import events


def _stage_name(message):
    """A short stable key for a loop, derived from its human-readable message."""
    return message.strip().rstrip('.').lower().replace(' ', '_')[:40] or 'loop'


class ProgressReporter:
    """Runs a loop over ``items`` and reports how far through it is.

    Subclasses override ``start`` / ``update`` / ``finish`` for display. The loop bodies
    are shared, and they also emit events on the ambient bus, so every reporter -- and a
    frontend watching the bus -- stays in step without any subclass doing extra work.
    """

    def start(self, total, message):
        pass

    def update(self, completed):
        pass

    def finish(self):
        pass

    def loop(self, items, fxn, message='Progress', **kwargs):
        """Apply ``fxn(item, **kwargs)`` to each item in order, reporting progress."""
        items = list(items)
        stage = _stage_name(message)
        self.start(len(items), message)
        events.emit(events.STAGE_STARTED, stage=stage, label=message, n_items=len(items))
        ok = False
        try:
            ret = []
            for i, item in enumerate(items):
                ret.append(fxn(item, **kwargs))
                self.update(i + 1)
                events.emit(events.PROGRESS, stage=stage, label=message,
                            done=i + 1, of=len(items))
            ok = True
            return ret
        finally:
            self.finish()
            events.emit(events.STAGE_FINISHED, stage=stage, ok=ok)

    def parallel_loop(self, items, fxn, message='Progress', nthreads=4, **kwargs):
        """As ``loop``, but across ``nthreads`` processes. Input order is preserved.

        ``fxn`` and every item and kwarg must be picklable.
        """
        items = list(items)
        self.start(len(items), message)
        try:
            ret = [None] * len(items)
            q = multiprocessing.Queue()
            procs = []

            def spawn(i):
                p = multiprocessing.Process(
                    target=_worker, args=(q, fxn, items[i], i), kwargs=kwargs)
                p.start()
                procs.append(p)

            for i in range(min(nthreads, len(items))):
                spawn(i)
            next_index = min(nthreads, len(items))
            for n_done in range(1, len(items) + 1):
                i, value = q.get()
                ret[i] = value
                self.update(n_done)
                if next_index < len(items):
                    spawn(next_index)
                    next_index += 1
            for p in procs:
                p.join()
            return ret
        finally:
            self.finish()


def _worker(q, fxn, item, i, **kwargs):
    q.put((i, fxn(item, **kwargs)))


class NullProgress(ProgressReporter):
    """Reports nothing. The default, so the pipeline is silent unless asked otherwise."""


class TextProgress(ProgressReporter):
    """A single rewritten line on stderr. No dependencies, works in any terminal."""

    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stderr
        self.total = 0
        self.message = ''

    def start(self, total, message):
        self.total = total
        self.message = message
        self.update(0)

    def update(self, completed):
        if not self.total:
            return
        width = 30
        filled = int(width * completed / self.total)
        bar = '#' * filled + '-' * (width - filled)
        self.stream.write(f'\r{self.message} [{bar}] {completed}/{self.total}')
        self.stream.flush()

    def finish(self):
        self.stream.write('\n')
        self.stream.flush()


class GuiProgress(ProgressReporter):
    """A FreeSimpleGUI progress-meter window, one per loop."""

    def __init__(self):
        self.window = None
        self.bar = None

    def start(self, total, message):
        import FreeSimpleGUI as sg
        layout = [[sg.Text(message)],
                  [sg.ProgressBar(max_value=max(total, 1), orientation='h',
                                  size=(20, 20), key='progress')]]
        self.window = sg.Window('Progress Meter', layout, finalize=True)
        self.bar = self.window['progress']
        self.bar.update_bar(0)

    def update(self, completed):
        if self.bar is not None:
            self.bar.update_bar(completed)

    def finish(self):
        if self.window is not None:
            self.window.close()
            self.window = None
            self.bar = None
