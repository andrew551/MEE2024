"""
Typed pipeline events.

The pipeline is already headless (see mee2024.progress): nothing in it imports a GUI
toolkit. What was missing was a richer contract than "progress bar percent", so that a
frontend can show *what* is happening rather than only how far along it is.

Events travel over an ambient bus held in a ContextVar, so emitting needs no extra
argument threaded through five call layers:

    bus = EventBus([JsonlSink(path), ListSink()])
    with events.using(bus):
        do_stack(...)          # emits stage/progress/frame/solve/metrics events

Note ContextVars do not propagate into threads started elsewhere, so a worker thread must
open its own ``using()`` block -- which is what mee2024.ui.server does.

Every event is a plain JSON-serialisable dict with ``seq``, ``t`` (seconds since the bus
was created), ``type``, and type-specific fields. Keeping them plain dicts means a sink can
be a file, a queue, a websocket or a test list with no adaptation.
"""

import contextvars
import json
import threading
import time
from contextlib import contextmanager

# ---------------------------------------------------------------- event types

STAGE_STARTED = 'stage_started'      # stage, label, n_items, unit
STAGE_FINISHED = 'stage_finished'    # stage, ok
PROGRESS = 'progress'                # stage, done, of, label, unit
FRAME_ALIGNED = 'frame_aligned'      # frame, shift, rms, n_matched
CENTROIDS_FOUND = 'centroids_found'  # stage, n, image_shape
SOLVE_CANDIDATE = 'solve_candidate'  # n_triangles, n_matched, threshold, accepted, ra, dec
SOLVE_RESULT = 'solve_result'        # success, ra, dec, roll, platescale, n_matched, mirror
METRICS = 'metrics'                  # any subset of the quality numbers
IMAGE = 'image'                      # name, png (base64), width, height
ANALYSIS = 'analysis'                # stars, surface, image_size, platescale, order
STARS = 'stars'                      # identified stars over the stacked preview:
                                     # stage, x, y, mag, label, tier, dropped,
                                     # image_size
LOG = 'log'                          # level, text
ERROR = 'error'                      # text, traceback

ALL_TYPES = (STAGE_STARTED, STAGE_FINISHED, PROGRESS, FRAME_ALIGNED, CENTROIDS_FOUND,
             SOLVE_CANDIDATE, SOLVE_RESULT, METRICS, IMAGE, ANALYSIS, STARS, LOG,
             ERROR)


# --------------------------------------------------------------------- sinks

class Sink:
    """Receives events. Must not raise: a broken sink may not break a pipeline run."""

    def handle(self, event):
        raise NotImplementedError

    def close(self):
        pass


class ListSink(Sink):
    """Keeps events in memory. Used by the UI server and by tests."""

    def __init__(self, limit=None):
        self.events = []
        self.limit = limit
        self._lock = threading.Lock()

    def handle(self, event):
        with self._lock:
            self.events.append(event)
            if self.limit is not None and len(self.events) > self.limit:
                del self.events[:len(self.events) - self.limit]

    def since(self, seq):
        """Events with seq > the given value, oldest first."""
        with self._lock:
            return [e for e in self.events if e['seq'] > seq]

    def latest(self, event_type):
        with self._lock:
            for event in reversed(self.events):
                if event['type'] == event_type:
                    return event
        return None

    def clear(self):
        with self._lock:
            self.events.clear()


class JsonlSink(Sink):
    """One JSON object per line. The machine-readable record of a run."""

    def __init__(self, path):
        self.path = path
        self._fp = open(path, 'w', encoding='utf-8')
        self._lock = threading.Lock()

    def handle(self, event):
        with self._lock:
            json.dump(event, self._fp)
            self._fp.write('\n')
            self._fp.flush()

    def close(self):
        with self._lock:
            if not self._fp.closed:
                self._fp.close()


class CallbackSink(Sink):
    def __init__(self, callback):
        self.callback = callback

    def handle(self, event):
        self.callback(event)


class TextSink(Sink):
    """Human-readable one-liners, for `--events stderr`."""

    def __init__(self, stream=None):
        import sys
        self.stream = stream if stream is not None else sys.stderr

    def handle(self, event):
        kind = event['type']
        if kind == PROGRESS:
            return  # too chatty for a text log; the progress bar covers it
        detail = {k: v for k, v in event.items()
                  if k not in ('seq', 't', 'type') and k != 'png'}
        self.stream.write(f"[{event['t']:7.2f}s] {kind}: {detail}\n")
        self.stream.flush()


# ----------------------------------------------------------------------- bus

class EventBus:
    """Fans events out to sinks, stamping each with a sequence number and timestamp."""

    def __init__(self, sinks=()):
        self.sinks = list(sinks)
        self._seq = 0
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()

    def add_sink(self, sink):
        self.sinks.append(sink)
        return sink

    def reset(self):
        """Restart the clock and the sequence, so timestamps are relative to this run."""
        with self._lock:
            self._seq = 0
            self._t0 = time.perf_counter()

    def emit(self, event_type, **fields):
        with self._lock:
            self._seq += 1
            event = {'seq': self._seq, 't': round(time.perf_counter() - self._t0, 3),
                     'type': event_type, **fields}
        for sink in list(self.sinks):
            try:
                sink.handle(event)
            except Exception:
                # a frontend that has gone away must not take the pipeline down
                pass
        return event

    def close(self):
        for sink in list(self.sinks):
            try:
                sink.close()
            except Exception:
                pass


_current = contextvars.ContextVar('mee2024_event_bus', default=None)


@contextmanager
def using(bus):
    """Make ``bus`` the ambient bus for the duration of the block."""
    token = _current.set(bus)
    try:
        yield bus
    finally:
        _current.reset(token)


def current():
    return _current.get()


def emit(event_type, **fields):
    """Emit on the ambient bus. A no-op when nothing is listening, so it is always safe."""
    bus = _current.get()
    if bus is not None:
        return bus.emit(event_type, **fields)
    return None


def log(text, level='info'):
    return emit(LOG, level=level, text=str(text))


def png_event(name, figure=None, image=None, max_width=900):
    """Emit an IMAGE event from a matplotlib figure or a 2-D array.

    Encoded as base64 PNG so it can travel over the same JSON channel as everything
    else -- no shared filesystem needed between pipeline and frontend.

    Science frames are large (a 3520x4656 stack is 125 MiB as float64), so the array is
    **strided down before any dtype conversion** and turned straight into 8-bit
    greyscale. Converting first and shrinking afterwards is what makes this run out of
    memory on real data.
    """
    if _current.get() is None:
        return None
    import base64
    import io

    import numpy as np

    buffer = io.BytesIO()
    if figure is not None:
        figure.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        width = height = None
    else:
        source = np.asarray(image)
        height, width = int(source.shape[0]), int(source.shape[1])
        step = max(1, int(np.ceil(width / max_width)))
        small = np.asarray(source[::step, ::step], dtype=np.float32)  # stride, then cast
        # A star field is almost entirely sky, so a percentile stretch anchored low in
        # the distribution renders that sky mid-grey and the stars barely brighter. Put
        # the black point just above the sky (most pixels ARE sky, so the median is it)
        # and the white point well down the bright tail: the background goes dark and
        # the stars stand out, which is what this preview is for.
        lo, hi = np.percentile(small, (55.0, 99.9))
        if hi <= lo:                                  # a flat or nearly empty frame
            lo, hi = float(small.min()), float(max(small.max(), small.min() + 1e-9))
        # gamma < 1 lifts the faint stars back up without lifting the sky with them
        normalised = np.clip((small - lo) / (hi - lo), 0.0, 1.0)
        grey = (255.0 * normalised ** 0.65).astype(np.uint8)
        _write_png_greyscale(grey, buffer)
    return emit(IMAGE, name=name, width=width, height=height,
                png=base64.b64encode(buffer.getvalue()).decode('ascii'))


def _write_png_greyscale(grey, buffer):
    """Write an 8-bit greyscale PNG. Uses Pillow when present, else the stdlib.

    The stdlib path keeps IMAGE events working in a minimal install: a PNG is just a
    zlib-compressed scanline stream with CRC-checked chunks.
    """
    try:
        from PIL import Image
        Image.fromarray(grey, mode='L').save(buffer, format='PNG')
        return
    except ImportError:
        pass
    import struct
    import zlib

    height, width = grey.shape
    raw = b''.join(b'\x00' + grey[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))

    buffer.write(b'\x89PNG\r\n\x1a\n')
    buffer.write(chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)))
    buffer.write(chunk(b'IDAT', zlib.compress(raw, 6)))
    buffer.write(chunk(b'IEND', b''))
