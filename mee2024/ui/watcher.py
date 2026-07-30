"""
Watch a folder and process frames as they arrive.

The point is instant feedback while the telescope is still on the field: point, capture,
and see within seconds whether the pointing, focus and data quality are good.

The hard part is not noticing files -- it is *not reading a file that is still being
written*. Capture software creates the file, then streams tens of megabytes into it, and a
FITS reader pointed at it half way through either fails or, worse, succeeds on a truncated
frame. So a frame is only considered ready once:

  * its last modification is at least ``settle_seconds`` old, and
  * its size has not changed since the previous poll.

The size check catches the case where a writer is slow enough that mtime looks stale
between two writes, which a pure mtime rule would let through.

Batching: stacking needs several frames, so settled frames accumulate until either
``batch_size`` are held, or nothing new has arrived for ``quiet_seconds`` and at least two
are held. Frames are never processed twice.
"""

import threading
import time
from pathlib import Path

from mee2024 import events

IMAGE_SUFFIXES = {'.fit', '.fits', '.fts', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}

WATCH_STARTED = 'watch_started'
WATCH_STOPPED = 'watch_stopped'
WATCH_SEEN = 'watch_seen'          # a new file has appeared but is not settled yet
WATCH_READY = 'watch_ready'        # a file has settled and joined the pending batch
WATCH_BATCH = 'watch_batch'        # a batch is being handed to the pipeline


class FolderWatcher:
    """Polls a folder and calls ``on_batch(paths)`` when a batch of frames is ready.

    Polling rather than OS notifications: it is a handful of lines, needs no extra
    dependency, behaves identically on Windows, macOS and Linux, and works on network
    shares where notification APIs are unreliable -- which is exactly where capture
    software often writes.
    """

    def __init__(self, folder, on_batch, settle_seconds=10.0, batch_size=5,
                 quiet_seconds=60.0, poll_seconds=2.0, clock=time.time,
                 suffixes=None, process_existing=False):
        self.folder = Path(folder)
        self.on_batch = on_batch
        self.settle_seconds = float(settle_seconds)
        self.batch_size = max(1, int(batch_size))
        self.quiet_seconds = float(quiet_seconds)
        self.poll_seconds = float(poll_seconds)
        self.clock = clock
        self.suffixes = {s.lower() for s in (suffixes or IMAGE_SUFFIXES)}
        self.process_existing = process_existing

        self.pending = []              # settled, not yet handed over
        self.processed = set()         # handed over at some point
        self.batches_run = 0
        self._sizes = {}               # path -> size at the previous poll
        self._known = set()            # every path we have ever noticed
        self._last_arrival = None      # when a frame most recently became ready
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ status

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self):
        with self._lock:
            return {'folder': str(self.folder), 'running': self.running,
                    'pending': [str(p) for p in self.pending],
                    'n_pending': len(self.pending),
                    'n_processed': len(self.processed),
                    'batches_run': self.batches_run,
                    'settle_seconds': self.settle_seconds,
                    'batch_size': self.batch_size}

    # ------------------------------------------------------------------- loop

    def start(self):
        if self.running:
            return self
        if not self.folder.is_dir():
            raise ValueError(f'not a folder: {self.folder}')
        if not self.process_existing:
            # everything already present predates the watch, so adopt it as known but
            # never process it -- otherwise starting a watch reprocesses the whole night
            for path in self._candidates():
                self._known.add(path)
                self.processed.add(path)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        events.emit(WATCH_STARTED, folder=str(self.folder),
                    settle_seconds=self.settle_seconds, batch_size=self.batch_size)
        return self

    def stop(self, timeout=5):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        events.emit(WATCH_STOPPED, folder=str(self.folder))

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as exc:      # a transient IO error must not end the watch
                events.log(f'watch error: {exc}', level='warning')
            self._stop.wait(self.poll_seconds)

    # ------------------------------------------------------------------- poll

    def _candidates(self):
        try:
            entries = sorted(self.folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return []
        return [p for p in entries
                if p.is_file() and p.suffix.lower() in self.suffixes
                and not p.name.startswith('.')]

    def poll(self, dispatch=True):
        """One pass. Returns the batch handed over, or None. Called directly by tests."""
        now = self.clock()
        for path in self._candidates():
            if path in self.processed or path in self.pending:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            previous_size = self._sizes.get(path)
            self._sizes[path] = stat.st_size
            if path not in self._known:
                self._known.add(path)
                events.emit(WATCH_SEEN, path=str(path), size=stat.st_size)
                continue           # never settle on the very first sighting
            if previous_size != stat.st_size:
                continue           # still growing
            if now - stat.st_mtime < self.settle_seconds:
                continue           # written recently; it may not be finished
            with self._lock:
                self.pending.append(path)
                self._last_arrival = now
            events.emit(WATCH_READY, path=str(path), n_pending=len(self.pending))

        return self._maybe_dispatch(now) if dispatch else None

    def _maybe_dispatch(self, now):
        with self._lock:
            held = len(self.pending)
            quiet = self._last_arrival is not None and \
                (now - self._last_arrival) >= self.quiet_seconds
            if held >= self.batch_size or (quiet and held >= 2):
                batch = list(self.pending)
                self.pending.clear()
                self.processed.update(batch)
                self.batches_run += 1
                self._last_arrival = None
            else:
                batch = None
        if batch:
            events.emit(WATCH_BATCH, paths=[str(p) for p in batch], n=len(batch),
                        batch_index=self.batches_run)
            self.on_batch(batch)
        return batch

    def flush(self):
        """Hand over whatever is pending, regardless of batch size. For a 'process now'."""
        with self._lock:
            if not self.pending:
                return None
            batch = list(self.pending)
            self.pending.clear()
            self.processed.update(batch)
            self.batches_run += 1
            self._last_arrival = None
        events.emit(WATCH_BATCH, paths=[str(p) for p in batch], n=len(batch),
                    batch_index=self.batches_run, forced=True)
        self.on_batch(batch)
        return batch
