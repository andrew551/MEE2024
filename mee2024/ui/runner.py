"""
Runs the pipeline for the UI, in a worker thread, reporting through the event bus.

Kept free of any HTTP or window concern so it can be driven directly from a test.
"""

import threading
import traceback
from pathlib import Path

from mee2024 import events
from mee2024.config import get_default_options
from mee2024.progress import EventProgress, ProgressReporter

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

    #: How to process: let the pipeline decide, or decide yourself. There used to be three
    #: presets, but each was only a frozen combination of controls the settings panel
    #: already exposes, so choosing between them meant guessing which frozen combination
    #: was nearest what you wanted. 'quick' and 'deep' are still honoured so a saved
    #: config from an older version keeps running.
    PRESETS = {
        'auto': {'label': 'Auto (recommended)',
                 'note': 'no settings needed -- sensitive centroids, cubic distortion, '
                         'date recovered from proper motions'},
        'custom': {'label': 'Custom',
                   'note': 'choose everything yourself in Settings below'},
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
        self.watcher = None
        self.watch_spec = {}
        self.watch_history = []
        self.batch_results = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state

    def snapshot(self, since=0):
        with self._lock:
            watch = None
            if self.watcher is not None:
                watch = dict(self.watcher.snapshot(), history=list(self.watch_history))
            return {
                'status': self.status,
                # deliberately not called 'error': the frontend reserves that key for
                # transport-level failures, and a failed run must not look like one
                'run_error': self.error,
                'outputs': {k: str(v) for k, v in self.outputs.items()},
                'spec': self.spec,
                'watch': watch,
                'batch': list(self.batch_results),
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
        if spec.get('fields'):
            # batch mode: the frames are discovered per field, not chosen by hand
            if not spec.get('output_dir'):
                raise ValueError('batch mode needs an output folder, so each field\'s '
                                 'results can be written beside a copy of its own '
                                 'folder layout')
        else:
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
            self.batch_results = []
            self.sink.clear()
            self.bus.reset()   # event times should be relative to this run, not to boot
            self.cancel_event = threading.Event()

        self.remember(spec)
        self.thread = threading.Thread(target=self._work, args=(dict(spec),), daemon=True)
        self.thread.start()
        return True

    #: what a run should carry over to the next session, and where it lives in the
    #: options dict the config file round-trips
    REMEMBERED = {'output_dir': 'output_dir', 'catalogue': 'catalogue',
                  'preset': 'ui_preset', 'observation_date': 'observation_date'}

    def remember(self, spec):
        """Save the choices this run was given, so the next session starts there.

        Without this the app forgot everything between launches -- the folder you
        last used most of all -- because nothing in the UI ever wrote the config
        file that the CLI and the classic interface have always used.
        """
        from mee2024.MEE2024util import read_ini, write_ini

        try:
            options = get_default_options()
            read_ini(options)
            lights = [str(p) for p in spec.get('lights') or []]
            if lights:
                options['workDir'] = str(Path(lights[0]).parent)
            for key, option in self.REMEMBERED.items():
                value = spec.get(key)
                if value:
                    options[option] = str(value)
            for key, value in (spec.get('options') or {}).items():
                options[key] = value
            write_ini(options)
        except Exception as exc:      # never let bookkeeping break a run
            events.log(f'could not save settings: {exc}', level='warning')

    def build_options(self, spec):
        options = get_default_options()
        preset = spec.get('preset', 'auto')
        # never open a plot window: everything the user should see travels as an event
        options.update(flag_display=False, flag_display2=False, flag_display3=False)
        # 'custom' takes the defaults and lets spec['options'] below say everything;
        # every other preset pins the choices it names
        if preset == 'auto':
            options.update(sensitive_mode_stack=True, distortionOrder='cubic',
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
        elif spec.get('date_from_header'):
            # take it from the frames themselves rather than guessing or asking
            from mee2024.stacker_implementation import read_observation_date
            lights = [str(p) for p in spec.get('lights') or []]
            header_date = read_observation_date(lights[0]) if lights else None
            if header_date:
                options['observation_date'] = header_date
                options['guess_date'] = False
                events.log(f'observation date {header_date} read from the FITS header')
            else:
                options['guess_date'] = True
                events.log('no date in the FITS header; recovering it from proper '
                           'motions instead', level='warning')
        options.update(spec.get('options') or {})
        return options

    def prepare_catalogue(self, options):
        """Fetch a missing catalogue and report depth problems, onto the event bus."""
        from mee2024.starcat import download

        for warning in download.prepare_catalogue(
                options.get('catalogue') or 'gaia', options=options,
                allow_download=options.get('auto_download_catalogue', True),
                on_note=events.log,
                progress_for=lambda name: EventProgress(
                    stage=f'download:{name}', label=f'Downloading {name}', unit='bytes')):
            events.log(warning, level='warning')

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
                if spec.get('fields'):
                    events.log(f'starting batch: {len(spec["fields"])} field(s), '
                               f'preset={spec.get("preset", "auto")}')
                else:
                    events.log(f'starting: {len(spec["lights"])} light frame(s), '
                               f'preset={spec.get("preset", "auto")}')
                if 'distortion' in stages:
                    # only stage 2 consults the catalogue, but check before stage 1 so a
                    # missing download is not discovered after minutes of stacking
                    self.prepare_catalogue(options)
                if options.get('platesolver', 'v2') == 'v2':
                    # the plate-solving database is derived from the catalogue, so it
                    # is built here rather than downloaded -- once, before any work
                    from mee2024 import platesolve2
                    platesolve2.ensure_pattern_db(
                        options, on_note=events.log,
                        progress=EventProgress(stage='patterndb',
                                               label='Preparing the plate solver'))

                if spec.get('fields'):
                    self._run_fields(spec, options, progress)
                else:
                    self._run_one(spec, options, progress)

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

    def _run_one(self, spec, options, progress, stages=None):
        """Stage 1 and stage 2 for one field. Returns the two output paths."""
        from mee2024 import distortion_fitter, stacker_implementation

        stages = stages or spec.get('stages') or ['stack', 'distortion']
        centroid_zip = spec.get('centroid_zip')
        distortion_zip = None
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
        return centroid_zip, distortion_zip

    def _field_metrics(self, since_seq):
        """The few numbers worth showing beside a finished field.

        Read from this field's own slice of the event stream rather than from the zips: the
        pipeline already reports them, and re-opening two archives to recover numbers that
        just went past would be work for nothing. Scoped by sequence number because in a
        batch the previous field's metrics are still in the sink.
        """
        found = {}
        for event in self.sink.since(since_seq):
            if event['type'] == events.METRICS:
                found[event.get('stage')] = event
        out = {}
        stack = found.get('stack') or {}
        distortion = found.get('distortion') or {}
        if stack.get('n_centroids') is not None:
            out['n_centroids'] = int(stack['n_centroids'])
        if stack.get('platesolved') is not None:
            out['platesolved'] = bool(stack['platesolved'])
        if distortion.get('rms_mas') is not None:
            out['rms_mas'] = float(distortion['rms_mas'])
        if distortion.get('n_stars') is not None:
            out['n_stars'] = int(distortion['n_stars'])
        if distortion.get('nn_corr') is not None:
            out['nn_corr'] = float(distortion['nn_corr'])
        return out

    def _run_fields(self, spec, options, progress):
        """Run every discovered field, one after another.

        A failure in one field must not abandon the rest -- a night of observing is too
        expensive to lose to one bad folder -- so each is caught, recorded and reported, and
        the batch carries on. Cancellation is different: it stops at the field boundary,
        because that is what the user asked for.

        The catalogue and the pattern database are prepared once by the caller rather than
        per field, which is most of why this is not simply a loop over start().
        """
        from mee2024.ui import batch

        fields = spec['fields']
        output_root = spec.get('output_dir')
        results = []
        events.emit(events.BATCH_STARTED, n_fields=len(fields),
                    root=spec.get('batch_root', ''))
        for number, field in enumerate(fields, start=1):
            if self.cancel_event.is_set():
                events.log(f'batch stopped after {number - 1} of {len(fields)} field(s)',
                           level='warning')
                break
            label = field.get('relative') or field.get('name') or field['folder']
            events.emit(events.BATCH_FIELD, index=number, of=len(fields), name=label,
                        n_frames=len(field['frames']), status='running')
            events.log(f'[{number}/{len(fields)}] {label}: '
                       f'{len(field["frames"])} frame(s)')
            field_options = dict(options)
            if output_root:
                field_options['output_dir'] = batch.output_dir_for(field, output_root)
            entry = {'name': label, 'folder': field['folder'],
                     'n_frames': len(field['frames']),
                     'output_dir': field_options.get('output_dir', '')}
            # where this field's events begin, so its numbers are not confused with the
            # previous field's when they are collected below
            with self._lock:
                mark = self.sink.events[-1]['seq'] if self.sink.events else 0
            try:
                field_spec = dict(spec, lights=field['frames'], centroid_zip=None)
                centroid_zip, distortion_zip = self._run_one(
                    field_spec, field_options, progress)
                entry.update(status='done', centroid_zip=str(centroid_zip or ''),
                             distortion_zip=str(distortion_zip or ''),
                             **self._field_metrics(mark))
            except Cancelled:
                entry.update(status='cancelled')
                results.append(entry)
                events.emit(events.BATCH_FIELD, index=number, of=len(fields), name=label,
                            n_frames=len(field['frames']), status='cancelled')
                events.log(f'batch stopped during {label}', level='warning')
                break
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                events.emit(events.ERROR, text=f'{label}: {entry["error"]}',
                            traceback=traceback.format_exc())
                events.log(f'[{number}/{len(fields)}] {label} FAILED: {entry["error"]}',
                           level='warning')
            results.append(entry)
            events.emit(events.BATCH_FIELD, index=number, of=len(fields), name=label,
                        n_frames=len(field['frames']), status=entry['status'],
                        error=entry.get('error', ''), folder=field['folder'],
                        output_dir=entry.get('output_dir', ''),
                        rms_mas=entry.get('rms_mas'), n_stars=entry.get('n_stars'),
                        nn_corr=entry.get('nn_corr'),
                        n_centroids=entry.get('n_centroids'))
        with self._lock:
            self.batch_results = results
        done = sum(1 for r in results if r['status'] == 'done')
        failed = sum(1 for r in results if r['status'] == 'failed')
        events.emit(events.BATCH_FINISHED, n_done=done, n_failed=failed,
                    n_fields=len(fields))
        events.log(f'batch finished: {done} of {len(fields)} field(s) succeeded'
                   + (f', {failed} failed' if failed else ''),
                   level='warning' if failed else 'info')

    def _record_output(self, key, value):
        with self._lock:
            self.outputs[key] = value

    def _finish(self, status):
        with self._lock:
            self.status = status

    # ------------------------------------------------------------- watch mode

    def start_watch(self, spec):
        """Watch a folder and run each settled batch of frames automatically."""
        from mee2024.config import get_default_options
        from mee2024.ui.watcher import FolderWatcher

        if self.watcher is not None and self.watcher.running:
            raise RuntimeError('already watching a folder')
        folder = spec.get('folder')
        if not folder:
            raise ValueError('choose a folder to watch')
        if not Path(folder).is_dir():
            raise ValueError(f'not a folder: {folder}')

        defaults = get_default_options()
        settle = float(spec.get('settle_seconds', defaults['watch_settle_seconds']))
        batch = int(spec.get('batch_size', defaults['watch_batch_size']))
        quiet = float(spec.get('quiet_seconds', defaults['watch_quiet_seconds']))
        poll = float(spec.get('poll_seconds', defaults['watch_poll_seconds']))

        with self._lock:
            self.watch_spec = dict(spec)
            self.watch_history = []
        self.watcher = FolderWatcher(
            folder, self._on_watch_batch, settle_seconds=settle, batch_size=batch,
            quiet_seconds=quiet, poll_seconds=poll,
            process_existing=bool(spec.get('process_existing')))
        # the watcher thread does not inherit the ambient bus, so give it one
        with events.using(self.bus):
            self.watcher.start()
        return True

    def stop_watch(self):
        if self.watcher is None:
            return False
        with events.using(self.bus):
            self.watcher.stop()
        return True

    def flush_watch(self):
        """Process whatever frames are held, without waiting for a full batch."""
        if self.watcher is None:
            return False
        with events.using(self.bus):
            return self.watcher.flush() is not None

    def _on_watch_batch(self, paths):
        """Called from the watcher thread when a batch of frames has settled."""
        if self.is_running:
            # a run is still going; put the frames back so they are not silently lost
            with self.watcher._lock:
                self.watcher.pending = list(paths) + self.watcher.pending
                self.watcher.processed.difference_update(paths)
            events.log(f'{len(paths)} frame(s) held: previous run still in progress',
                       level='warning')
            return
        spec = dict(self.watch_spec)
        spec['lights'] = [str(p) for p in paths]
        spec.pop('folder', None)
        with self._lock:
            self.watch_history.append({'n': len(paths), 'status': 'running'})
        try:
            self.start(spec)
        except Exception as exc:
            events.log(f'could not start batch: {exc}', level='warning')
            return
        # wait for it here, in the watcher thread, so batches never overlap
        if self.thread is not None:
            self.thread.join()
        with self._lock:
            if self.watch_history:
                self.watch_history[-1]['status'] = self.status
                self.watch_history[-1]['outputs'] = {k: str(v)
                                                     for k, v in self.outputs.items()}
