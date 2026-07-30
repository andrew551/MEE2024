"""Folder watching: the settle rule that stops us reading a half-written frame."""

import os
import time

import pytest

from mee2024.ui.watcher import FolderWatcher


class FakeClock:
    """A clock the test drives, so settle behaviour is checked without real waiting."""

    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def write_frame(folder, name, size=64, mtime=None):
    path = folder / name
    path.write_bytes(b'x' * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def make_watcher(folder, clock, **kwargs):
    batches = []
    options = dict(settle_seconds=10.0, batch_size=3, quiet_seconds=60.0, clock=clock)
    options.update(kwargs)
    watcher = FolderWatcher(folder, batches.append, **options)
    return watcher, batches


def test_rejects_a_folder_that_does_not_exist(tmp_path):
    watcher, _ = make_watcher(tmp_path / 'nope', FakeClock())
    with pytest.raises(ValueError, match='not a folder'):
        watcher.start()


def test_a_brand_new_file_is_not_ready_on_first_sight(tmp_path):
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock)
    write_frame(tmp_path, 'a.fit', mtime=clock.now)

    watcher.poll()
    assert watcher.pending == [], 'a file must never settle on the poll that finds it'


def test_a_file_still_being_written_is_not_ready(tmp_path):
    """The size check catches a slow writer that a pure mtime rule would let through."""
    clock = FakeClock()
    watcher, _ = make_watcher(tmp_path, clock)
    path = write_frame(tmp_path, 'a.fit', size=64, mtime=clock.now - 999)
    watcher.poll()                      # first sighting

    path.write_bytes(b'x' * 4096)       # grew since the last poll
    os.utime(path, (clock.now - 999, clock.now - 999))
    watcher.poll()
    assert watcher.pending == [], 'a file whose size changed must not be considered ready'

    watcher.poll()                      # size now stable, mtime old
    assert [p.name for p in watcher.pending] == ['a.fit']


def test_a_recently_modified_file_waits_for_the_settle_period(tmp_path):
    clock = FakeClock()
    watcher, _ = make_watcher(tmp_path, clock, settle_seconds=10.0)
    write_frame(tmp_path, 'a.fit', mtime=clock.now)
    watcher.poll()
    watcher.poll()
    assert watcher.pending == [], 'modified 0 s ago: too fresh'

    clock.advance(9.0)
    watcher.poll()
    assert watcher.pending == [], 'modified 9 s ago: still inside the 10 s settle window'

    clock.advance(2.0)
    watcher.poll()
    assert [p.name for p in watcher.pending] == ['a.fit']


def test_a_batch_is_handed_over_once_it_is_full(tmp_path):
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock, batch_size=3)
    for i in range(3):
        write_frame(tmp_path, f'f{i}.fit', mtime=clock.now - 999)

    watcher.poll()                  # first sighting of all three
    assert batches == []
    watcher.poll()                  # settled -> batch full -> dispatched
    assert len(batches) == 1
    assert [p.name for p in batches[0]] == ['f0.fit', 'f1.fit', 'f2.fit']
    assert watcher.pending == []


def test_a_partial_batch_goes_after_a_quiet_period(tmp_path):
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock, batch_size=5, quiet_seconds=60.0)
    for i in range(2):
        write_frame(tmp_path, f'f{i}.fit', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    assert batches == [], 'two frames is below the batch size, and nothing is quiet yet'

    clock.advance(61)
    watcher.poll()
    assert len(batches) == 1 and len(batches[0]) == 2


def test_a_single_frame_never_goes_on_quiet_alone(tmp_path):
    """One frame cannot be stacked, so holding it is better than a run that must fail."""
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock, batch_size=5, quiet_seconds=30.0)
    write_frame(tmp_path, 'only.fit', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    clock.advance(120)
    watcher.poll()
    assert batches == []
    assert len(watcher.pending) == 1


def test_flush_hands_over_a_partial_batch_on_demand(tmp_path):
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock, batch_size=10)
    write_frame(tmp_path, 'a.fit', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    assert watcher.flush() is not None
    assert len(batches) == 1 and len(batches[0]) == 1
    assert watcher.flush() is None, 'nothing pending: flush must be a no-op'


def test_a_frame_is_never_processed_twice(tmp_path):
    clock = FakeClock()
    watcher, batches = make_watcher(tmp_path, clock, batch_size=1)
    write_frame(tmp_path, 'a.fit', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    assert len(batches) == 1
    for _ in range(4):
        watcher.poll()
    assert len(batches) == 1, 'an already-processed frame must not be handed over again'


def test_non_image_files_are_ignored(tmp_path):
    clock = FakeClock()
    watcher, _ = make_watcher(tmp_path, clock, batch_size=1)
    write_frame(tmp_path, 'notes.txt', mtime=clock.now - 999)
    write_frame(tmp_path, 'log.csv', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    assert watcher.pending == []


def test_files_already_present_are_skipped_by_default(tmp_path):
    """Starting a watch must not reprocess a whole night of existing data."""
    clock = FakeClock()
    write_frame(tmp_path, 'old.fit', mtime=clock.now - 999)
    watcher, batches = make_watcher(tmp_path, clock, batch_size=1)
    watcher.start()
    try:
        watcher.poll()
        watcher.poll()
    finally:
        watcher.stop()
    assert batches == []
    # it is counted as already dealt with, so it can never be picked up later either
    assert watcher.snapshot()['n_processed'] == 1
    assert watcher.snapshot()['n_pending'] == 0


def test_process_existing_picks_up_what_is_already_there(tmp_path):
    clock = FakeClock()
    write_frame(tmp_path, 'old.fit', mtime=clock.now - 999)
    watcher, batches = make_watcher(tmp_path, clock, batch_size=1, process_existing=True)
    watcher.poll()
    watcher.poll()
    assert len(batches) == 1


def test_snapshot_reports_useful_state(tmp_path):
    clock = FakeClock()
    watcher, _ = make_watcher(tmp_path, clock, batch_size=4, settle_seconds=7.5)
    write_frame(tmp_path, 'a.fit', mtime=clock.now - 999)
    watcher.poll()
    watcher.poll()
    snap = watcher.snapshot()
    assert snap['n_pending'] == 1 and snap['batch_size'] == 4
    assert snap['settle_seconds'] == 7.5
    assert snap['folder'] == str(tmp_path)
    assert snap['running'] is False


def test_start_and_stop_with_the_real_clock(tmp_path):
    """A smoke test of the actual thread, with a settle time short enough to be quick."""
    batches = []
    watcher = FolderWatcher(tmp_path, batches.append, settle_seconds=0.05,
                            batch_size=1, poll_seconds=0.02, process_existing=True)
    write_frame(tmp_path, 'a.fit')
    watcher.start()
    try:
        deadline = time.time() + 5
        while not batches and time.time() < deadline:
            time.sleep(0.05)
    finally:
        watcher.stop()
    assert not watcher.running
    assert len(batches) == 1, 'the polling thread should have dispatched one batch'


def test_watch_events_are_emitted(tmp_path):
    from mee2024 import events
    clock = FakeClock()
    bus = events.EventBus([events.ListSink()])
    sink = bus.sinks[0]
    watcher, _ = make_watcher(tmp_path, clock, batch_size=1)
    write_frame(tmp_path, 'a.fit', mtime=clock.now - 999)
    with events.using(bus):
        watcher.poll()
        watcher.poll()
    kinds = [e['type'] for e in sink.events]
    assert 'watch_seen' in kinds
    assert 'watch_ready' in kinds
    assert 'watch_batch' in kinds
