"""The typed pipeline event bus."""

import json
import threading

import pytest

from mee2024 import events
from mee2024.progress import NullProgress, TextProgress


# --------------------------------------------------------------------- the bus

def test_events_are_stamped_with_sequence_and_time():
    sink = events.ListSink()
    bus = events.EventBus([sink])
    bus.emit(events.LOG, text='one')
    bus.emit(events.LOG, text='two')
    assert [e['seq'] for e in sink.events] == [1, 2]
    assert all(isinstance(e['t'], float) for e in sink.events)
    assert [e['text'] for e in sink.events] == ['one', 'two']


def test_since_returns_only_newer_events():
    sink = events.ListSink()
    bus = events.EventBus([sink])
    for i in range(5):
        bus.emit(events.LOG, text=str(i))
    assert [e['text'] for e in sink.since(3)] == ['3', '4']
    assert sink.since(5) == []


def test_latest_finds_the_most_recent_of_a_type():
    sink = events.ListSink()
    bus = events.EventBus([sink])
    bus.emit(events.METRICS, rms_mas=200.0)
    bus.emit(events.LOG, text='noise')
    bus.emit(events.METRICS, rms_mas=110.0)
    assert sink.latest(events.METRICS)['rms_mas'] == 110.0
    assert sink.latest(events.ERROR) is None


def test_list_sink_honours_its_limit():
    sink = events.ListSink(limit=3)
    bus = events.EventBus([sink])
    for i in range(10):
        bus.emit(events.LOG, text=str(i))
    assert [e['text'] for e in sink.events] == ['7', '8', '9']


def test_a_broken_sink_cannot_break_a_run():
    """A frontend that has gone away must not take the pipeline down with it."""
    class Broken(events.Sink):
        def handle(self, event):
            raise RuntimeError('frontend vanished')

    good = events.ListSink()
    bus = events.EventBus([Broken(), good])
    bus.emit(events.LOG, text='still delivered')
    assert len(good.events) == 1


def test_jsonl_sink_writes_one_object_per_line(tmp_path):
    path = tmp_path / 'run.jsonl'
    bus = events.EventBus([events.JsonlSink(path)])
    bus.emit(events.STAGE_STARTED, stage='stack', n_items=5)
    bus.emit(events.PROGRESS, stage='stack', done=1, of=5)
    bus.close()

    lines = path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first['type'] == events.STAGE_STARTED and first['n_items'] == 5


def test_events_are_json_serialisable():
    """Everything must survive the trip to a frontend."""
    sink = events.ListSink()
    bus = events.EventBus([sink])
    bus.emit(events.FRAME_ALIGNED, frame=2, shift=[1.5, -2.5], rms=0.06, n_matched=28)
    bus.emit(events.SOLVE_RESULT, success=True, ra=28.46, dec=45.33, roll=86.3,
             platescale=1.85, mirror=False, n_matched=99)
    for event in sink.events:
        json.loads(json.dumps(event))


def test_text_sink_skips_progress_spam():
    import io
    stream = io.StringIO()
    bus = events.EventBus([events.TextSink(stream)])
    bus.emit(events.PROGRESS, stage='stack', done=1, of=5)
    bus.emit(events.STAGE_FINISHED, stage='stack', ok=True)
    output = stream.getvalue()
    assert 'progress' not in output
    assert 'stage_finished' in output


def test_callback_sink_forwards_events():
    seen = []
    bus = events.EventBus([events.CallbackSink(seen.append)])
    bus.emit(events.LOG, text='hello')
    assert seen[0]['text'] == 'hello'


# ------------------------------------------------------------- the ambient bus

def test_emit_is_a_no_op_with_no_bus():
    assert events.current() is None
    assert events.emit(events.LOG, text='nobody listening') is None


def test_using_installs_and_restores_the_bus():
    sink = events.ListSink()
    bus = events.EventBus([sink])
    with events.using(bus):
        assert events.current() is bus
        events.emit(events.LOG, text='inside')
    assert events.current() is None
    assert len(sink.events) == 1


def test_nested_buses_restore_the_outer_one():
    outer_sink, inner_sink = events.ListSink(), events.ListSink()
    outer, inner = events.EventBus([outer_sink]), events.EventBus([inner_sink])
    with events.using(outer):
        with events.using(inner):
            events.emit(events.LOG, text='inner')
        events.emit(events.LOG, text='outer')
    assert len(inner_sink.events) == 1
    assert len(outer_sink.events) == 1


def test_a_worker_thread_must_open_its_own_bus():
    """Documents the ContextVar boundary the UI server has to respect."""
    sink = events.ListSink()
    bus = events.EventBus([sink])
    seen_in_thread = []

    def worker():
        seen_in_thread.append(events.current())
        with events.using(bus):
            events.emit(events.LOG, text='from the worker')

    with events.using(bus):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen_in_thread == [None], 'a thread does not inherit the ambient bus'
    assert [e['text'] for e in sink.events] == ['from the worker']


def test_bus_is_threadsafe_under_concurrent_emitters():
    sink = events.ListSink()
    bus = events.EventBus([sink])

    def worker(n):
        with events.using(bus):
            for i in range(n):
                events.emit(events.LOG, text=str(i))

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sink.events) == 200
    assert sorted(e['seq'] for e in sink.events) == list(range(1, 201))


# --------------------------------------------------- progress emits events too

def test_progress_loop_emits_stage_and_progress_events():
    """Every reporter gets events for free, because loop() emits them."""
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        NullProgress().loop([1, 2, 3], lambda x: x * 2, message='Finding all centroids...')

    kinds = [e['type'] for e in sink.events]
    assert kinds[0] == events.STAGE_STARTED
    assert kinds[-1] == events.STAGE_FINISHED
    progress = [e for e in sink.events if e['type'] == events.PROGRESS]
    assert [(p['done'], p['of']) for p in progress] == [(1, 3), (2, 3), (3, 3)]
    assert sink.events[0]['stage'] == 'finding_all_centroids'
    assert sink.events[0]['label'] == 'Finding all centroids...'


def test_progress_loop_reports_failure_in_stage_finished():
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        with pytest.raises(ValueError):
            NullProgress().loop([1], lambda x: (_ for _ in ()).throw(ValueError('boom')))
    assert sink.latest(events.STAGE_FINISHED)['ok'] is False


def test_progress_loop_still_returns_results_and_works_without_a_bus():
    assert NullProgress().loop([1, 2], lambda x: x + 1) == [2, 3]


def test_text_progress_also_emits(capsys):
    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        TextProgress().loop([1, 2], lambda x: x, message='Stacking images...')
    assert any(e['type'] == events.PROGRESS for e in sink.events)


# ------------------------------------------------------------------- CLI wiring

def test_cli_event_bus_yields_nothing_when_not_requested():
    from mee2024 import cli
    args = cli.build_parser().parse_args(['stack', 'a.fit', '--no-config'])
    with cli.event_bus(args) as bus:
        assert bus is None


def test_cli_event_bus_writes_jsonl_when_asked(tmp_path):
    from mee2024 import cli
    path = tmp_path / 'events.jsonl'
    args = cli.build_parser().parse_args(
        ['stack', 'a.fit', '--no-config', '--events-jsonl', str(path)])
    with cli.event_bus(args) as bus:
        assert bus is not None
        events.emit(events.LOG, text='recorded')
    assert json.loads(path.read_text(encoding='utf-8').strip())['text'] == 'recorded'
