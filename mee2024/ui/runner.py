"""
Runs the pipeline for the UI, in a worker thread, reporting through the event bus.

Kept free of any HTTP or window concern so it can be driven directly from a test.
"""

import threading
import traceback
from pathlib import Path

from mee2024 import events
from mee2024.config import get_default_options
from mee2024.progress import ProgressReporter

IDLE = 'idle'
RUNNING = 'running'
DONE = 'done'
FAILED = 'failed'
CANCELLED = 'cancelled'


class Cancelled(Exception):
    """Raised inside the worker when the user asks to stop."""


class CancellableProgress(ProgressReporter):
    """Checks a cancel flag between items, so a run can be stopped promptly.

    Cancellation is cooperative: the flag is examined between loop items, which is the
    finest granularity available without killing a thread mid-computation.
    """

    def __init__(self, cancel_event):
        self.cancel_event = cancel_event

    def update(self, completed):
        if self.cancel_event.is_set():
            raise Cancelled()


class PipelineRunner:
    """Owns at most one run at a time and the events it produced."""

    #: presets the Simple mode offers, chosen from the measured behaviour of the pipeline
    PRESETS = {
        'auto': {'label': 'Auto (recommended)',
                 'note': 'sensitive centroids, quintic distortion, guessed date'},
        'quick': {'label': 'Quick look',
                  'note': 'plain centroids, cubic distortion -- fastest'},
        'deep': {'label': 'Deep',
                 'note': 'sensitive centroids, septic distortion -- for wide fields'},
    }

    def __init__(self):
        self.status = IDLE
        self.sink = events.ListSink(limit=20000)
        self.bus = events.EventBus([self.sink])
        self.cancel_event = threading.Event()
        self.thread = None
        self.error = None
        self.outputs = {}
        self.spec = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state

    def snapshot(self, since=0):
        with self._lock:
            return {
                'status': self.status,
                # deliberately not called 'error': the frontend reserves that key for
                # transport-level failures, and a failed run must not look like one
                'run_error': self.error,
                'outputs': {k: str(v) for k, v in self.outputs.items()},
                'spec': self.spec,
                'events': self.sink.since(since),
                'seq': self.sink.events[-1]['seq'] if self.sink.events else 0,
            }

    @property
    def is_running(self):
        return self.status == RUNNING

    def cancel(self):
        if self.status == RUNNING:
            self.cancel_event.set()
            return True
        return False

    # -------------------------------------------------------------------- run

    def start(self, spec):
        """Begin a run. spec keys: lights, darks, flats, output_dir, preset, stages,
        catalogue, options (a dict of raw option overrides)."""
        if self.is_running:
            raise RuntimeError('a run is already in progress')
        lights = [str(p) for p in spec.get('lights') or []]
        if not lights:
            raise ValueError('choose at least one light frame')
        missing = [p for p in lights if not Path(p).exists()]
        if missing:
            raise ValueError(f'file not found: {missing[0]}')
        # create the output folder now, so a bad choice is reported here rather than
        # several directory levels deep inside the pipeline
        if spec.get('output_dir'):
            try:
                Path(spec['output_dir']).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f'cannot use that output folder: {exc}')

        with self._lock:
            self.status = RUNNING
            self.error = None
            self.outputs = {}
            self.spec = dict(spec)
            self.sink.clear()
            self.bus.reset()   # event times should be relative to this run, not to boot
            self.cancel_event = threading.Event()

        self.thread = threading.Thread(target=self._work, args=(dict(spec),), daemon=True)
        self.thread.start()
        return True

    def build_options(self, spec):
        options = get_default_options()
        preset = spec.get('preset', 'auto')
        # never open a plot window: everything the user should see travels as an event
        options.update(flag_display=False, flag_display2=False, flag_display3=False)
        if preset == 'auto':
            options.update(sensitive_mode_stack=True, distortionOrder='quintic',
                           guess_date=True)
        elif preset == 'quick':
            options.update(sensitive_mode_stack=False, distortionOrder='cubic',
                           guess_date=False)
        elif preset == 'deep':
            options.update(sensitive_mode_stack=True, distortionOrder='septic',
                           guess_date=True)
        if spec.get('output_dir'):
            options['output_dir'] = str(spec['output_dir'])
        if spec.get('darks'):
            options['-DARK-'] = ';'.join(str(p) for p in spec['darks'])
        if spec.get('flats'):
            options['-FLAT-'] = ';'.join(str(p) for p in spec['flats'])
        if spec.get('catalogue'):
            options['catalogue'] = spec['catalogue']
        if spec.get('observation_date'):
            options['observation_date'] = spec['observation_date']
            options['guess_date'] = False
        options.update(spec.get('options') or {})
        return options

    def _work(self, spec):
        import matplotlib
        matplotlib.use('Agg')
        from mee2024 import database_cache, distortion_fitter, stacker_implementation

        # a worker thread does not inherit the ambient bus, so open one here
        with events.using(self.bus):
            try:
                options = self.build_options(spec)
                stages = spec.get('stages') or ['stack', 'distortion']
                progress = CancellableProgress(self.cancel_event)
                events.log(f'starting: {len(spec["lights"])} light frame(s), '
                           f'preset={spec.get("preset", "auto")}')

                centroid_zip = spec.get('centroid_zip')
                if 'stack' in stages:
                    centroid_zip = stacker_implementation.do_stack(
                        [str(p) for p in spec['lights']],
                        [str(p) for p in spec.get('darks') or []],
                        [str(p) for p in spec.get('flats') or []],
                        options, progress=progress)
                    self._record_output('centroid_zip', centroid_zip)
                    events.log(f'stage 1 complete: {Path(centroid_zip).name}')

                if 'distortion' in stages and centroid_zip:
                    if self.cancel_event.is_set():
                        raise Cancelled()
                    distortion_zip = distortion_fitter.match_and_fit_distortion(
                        str(centroid_zip), options, None)
                    self._record_output('distortion_zip', distortion_zip)
                    events.log(f'stage 2 complete: {Path(distortion_zip).name}')

                self._finish(DONE)
            except Cancelled:
                events.log('run cancelled', level='warning')
                self._finish(CANCELLED)
            except Exception as exc:
                text = f'{type(exc).__name__}: {exc}'
                events.emit(events.ERROR, text=text, traceback=traceback.format_exc())
                with self._lock:
                    self.error = text
                self._finish(FAILED)
            finally:
                try:
                    database_cache.shutdown_triangles()
                except Exception:
                    pass

    def _record_output(self, key, value):
        with self._lock:
            self.outputs[key] = value

    def _finish(self, status):
        with self._lock:
            self.status = status
