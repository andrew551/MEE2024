"""
Finding the fields in a folder tree, and running them.

Capture software writes one folder per field, so the useful unit of work is a tree rather
than a list of frames. The risk this carries is the walk: pointed at a drive root it would
visit every directory on the machine and then start hundreds of runs, so most of these tests
are about the bounds and about *refusing* rather than silently truncating.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from mee2024.ui import batch


def _frames(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        fits.writeto(folder / name, np.zeros((4, 4), dtype=np.int16))


@pytest.fixture
def session(tmp_path):
    """The layout the request described: session / field / time / frames."""
    root = tmp_path / '2026-07-27'
    _frames(root / 'H1_eclipse_altaz' / '22_02_22',
            [f'H1_{i:05d}.fits' for i in range(1, 5)])
    _frames(root / 'H2_plus2deg_alt' / '22_07_46',
            [f'H2_{i:05d}.fits' for i in range(1, 5)])
    _frames(root / 'Z1_base' / '21_27_17', [f'Z1_{i:05d}.fits' for i in range(1, 4)])
    # the sidecar text files capture software leaves must not be mistaken for frames
    (root / 'H1_eclipse_altaz' / '22_02_22' / 'H1.CameraSettings.txt').write_text('x')
    return root


# --------------------------------------------------------------------- discovery

def test_every_folder_of_frames_is_a_field(session):
    fields, info = batch.find_fields(session)
    assert [f['name'] for f in fields] == ['22_02_22', '22_07_46', '21_27_17']
    assert info['found'] == 3
    assert info['truncated'] is None


def test_the_relative_path_is_what_the_output_mirrors(session):
    fields, _ = batch.find_fields(session)
    assert {f['relative'].replace('\\', '/') for f in fields} == {
        'H1_eclipse_altaz/22_02_22', 'H2_plus2deg_alt/22_07_46', 'Z1_base/21_27_17'}


def test_frame_counts_exclude_the_sidecar_files(session):
    fields, _ = batch.find_fields(session)
    by_name = {f['name']: f for f in fields}
    assert len(by_name['22_02_22']['frames']) == 4
    assert all(p.endswith('.fits') for p in by_name['22_02_22']['frames'])


def test_frames_are_sorted_so_a_rerun_matches(session):
    fields, _ = batch.find_fields(session)
    for field in fields:
        assert field['frames'] == sorted(field['frames'])


def test_a_folder_of_frames_is_itself_a_field(tmp_path):
    """Pointing this at a single capture folder should behave as anyone would expect."""
    _frames(tmp_path / 'one', ['a.fits', 'b.fits'])
    fields, info = batch.find_fields(tmp_path / 'one')
    assert len(fields) == 1
    assert fields[0]['relative'] == ''
    assert batch.output_dir_for(fields[0], tmp_path / 'out') == str(tmp_path / 'out')


def test_a_field_stops_the_walk_going_deeper(tmp_path):
    """A folder of frames is a field, not a container of fields -- otherwise a stray
    subfolder inside a capture folder would be processed as a second field."""
    _frames(tmp_path / 'field', ['a.fits'])
    _frames(tmp_path / 'field' / 'thumbnails', ['t.fits'])
    fields, _ = batch.find_fields(tmp_path)
    assert [f['name'] for f in fields] == ['field']


def test_other_image_formats_count_too(tmp_path):
    (tmp_path / 'tifs').mkdir()
    (tmp_path / 'tifs' / 'a.tif').write_bytes(b'0')
    fields, _ = batch.find_fields(tmp_path)
    assert len(fields) == 1


def test_a_tree_with_no_frames_says_so(tmp_path):
    (tmp_path / 'empty' / 'deeper').mkdir(parents=True)
    fields, info = batch.find_fields(tmp_path)
    assert fields == []
    assert 'no image frames' in info['truncated']


def test_a_missing_folder_is_reported_not_raised(tmp_path):
    fields, info = batch.find_fields(tmp_path / 'nope')
    assert fields == [] and 'not a folder' in info['truncated']


# ------------------------------------------------------------------ the bounds

def test_too_many_fields_refuses_rather_than_truncating(tmp_path):
    """Silently processing the first twenty of two hundred would be worse than stopping:
    the user would think the job was done."""
    for i in range(25):
        _frames(tmp_path / f'field{i:03d}', ['a.fits'])
    fields, info = batch.find_fields(tmp_path, max_fields=20)
    assert fields == [], 'must not start a partial batch'
    assert 'more than 20' in info['truncated']


def test_the_field_limit_is_adjustable(tmp_path):
    for i in range(25):
        _frames(tmp_path / f'field{i:03d}', ['a.fits'])
    fields, info = batch.find_fields(tmp_path, max_fields=30)
    assert len(fields) == 25 and info['truncated'] is None


def test_exactly_the_limit_is_allowed(tmp_path):
    for i in range(20):
        _frames(tmp_path / f'field{i:03d}', ['a.fits'])
    fields, info = batch.find_fields(tmp_path, max_fields=20)
    assert len(fields) == 20 and info['truncated'] is None


def test_a_huge_tree_stops_scanning_even_with_few_frames(tmp_path):
    """The walk itself is the cost when a tree is enormous and nearly empty -- which is
    exactly what a drive root looks like."""
    for i in range(40):
        (tmp_path / f'dir{i:03d}' / 'sub').mkdir(parents=True)
    fields, info = batch.find_fields(tmp_path, max_scanned=10)
    assert fields == []
    assert 'stopped after looking at 10 folders' in info['truncated']
    assert info['scanned'] <= 11


def test_a_real_session_tree_is_well_inside_the_bounds(session):
    fields, info = batch.find_fields(session)
    assert info['scanned'] < batch.DEFAULT_MAX_SCANNED
    assert len(fields) < batch.DEFAULT_MAX_FIELDS


# ------------------------------------------------------------- output mirroring

def test_the_output_mirrors_the_input_layout(session, tmp_path):
    fields, _ = batch.find_fields(session)
    out = tmp_path / 'results'
    for field in fields:
        target = batch.output_dir_for(field, out)
        assert target.startswith(str(out))
        assert target.endswith(field['relative'])


def test_two_fields_named_the_same_stay_apart(tmp_path):
    """Capture software reuses timestamps across nights; flattening would collide."""
    _frames(tmp_path / 'nightA' / '22_02_22', ['a.fits'])
    _frames(tmp_path / 'nightB' / '22_02_22', ['a.fits'])
    fields, _ = batch.find_fields(tmp_path)
    targets = {batch.output_dir_for(f, tmp_path / 'out') for f in fields}
    assert len(targets) == 2


# ------------------------------------------------------- naming the run's folder

def test_the_run_folder_is_named_after_the_input(session, tmp_path):
    """An output folder should say which input produced it, without the user naming it."""
    root = batch.run_output_root(tmp_path / 'results', session)
    assert root == str(tmp_path / 'results' / '2026-07-27')


def test_a_fresh_folder_gets_no_timestamp(session, tmp_path):
    """The paths are already close to the Windows limit; do not lengthen them for a
    collision that has not happened."""
    (tmp_path / 'results' / '2026-07-27').mkdir(parents=True)
    root = batch.run_output_root(tmp_path / 'results', session)
    assert root == str(tmp_path / 'results' / '2026-07-27')


@pytest.mark.parametrize('record', batch.RUN_RECORDS)
def test_a_second_run_does_not_overwrite_the_first_run_s_records(session, tmp_path,
                                                                 record):
    """The per-field archives are timestamped and survive. The summary and the activity
    log are not: a second run into the same folder used to destroy the only account of
    what the first one did."""
    first = tmp_path / 'results' / '2026-07-27'
    first.mkdir(parents=True)
    (first / record).write_text('the first run')
    second = batch.run_output_root(tmp_path / 'results', session, stamp='20260810_0151')
    assert second == str(tmp_path / 'results' / '2026-07-27_20260810_0151')
    assert (first / record).read_text() == 'the first run'


def test_a_source_with_no_folder_name_leaves_the_root_alone(tmp_path):
    """A batch assembled without a root, and a path that is nothing but a separator."""
    assert batch.run_output_root(tmp_path, '') == str(tmp_path)
    assert batch.run_output_root(tmp_path, '/') == str(tmp_path)


def test_the_label_is_bounded_and_free_of_separators():
    assert batch.source_label('J:/data/Zenith') == 'Zenith'
    assert len(batch.source_label('J:/data/' + 'x' * 200)) == batch.MAX_LABEL
    assert '/' not in batch.source_label('a/b:c*d')


def test_describe_reports_counts(session):
    fields, info = batch.find_fields(session)
    text = batch.describe(fields, info)
    assert '3 field(s)' in text and '11 frame(s)' in text


def test_describe_passes_the_refusal_through(tmp_path):
    fields, info = batch.find_fields(tmp_path / 'nope')
    assert batch.describe(fields, info) == info['truncated']


# ---------------------------------------------------------- running the batch

@pytest.fixture
def runner():
    from mee2024.ui.runner import PipelineRunner
    return PipelineRunner()


def _wait(runner, timeout=20.0):
    import time
    deadline = time.time() + timeout
    while runner.status == 'running' and time.time() < deadline:
        time.sleep(0.02)
    return runner.status


def test_each_field_runs_into_its_own_mirrored_output(session, tmp_path, runner,
                                                      monkeypatch):
    from mee2024 import stacker_implementation as si
    from mee2024.ui import runner as runner_module

    seen = []

    def fake_stack(lights, darks, flats, options, progress=None):
        seen.append({'n': len(lights), 'out': options['output_dir']})
        return tmp_path / 'fake.zip'

    monkeypatch.setattr(si, 'do_stack', fake_stack)
    fields, _ = batch.find_fields(session)
    out = tmp_path / 'results'
    runner.start({'fields': fields, 'output_dir': str(out), 'preset': 'auto',
                  'stages': ['stack'], 'batch_root': str(session)})
    assert _wait(runner) == 'done'
    assert len(seen) == 3
    for entry, field in zip(seen, fields):
        assert entry['n'] == len(field['frames'])
        assert entry['out'].endswith(field['relative'])


def test_two_batches_into_one_folder_both_keep_their_records(session, tmp_path,
                                                             monkeypatch):
    """I18: the summary and the activity log have fixed names, so the second run used to
    destroy the first one's -- the very records that say what happened."""
    from mee2024 import stacker_implementation as si
    from mee2024.ui.runner import PipelineRunner

    monkeypatch.setattr(si, 'do_stack', lambda *a, **k: tmp_path / 'fake.zip')
    fields, _ = batch.find_fields(session)
    out = tmp_path / 'results'
    roots = []
    for _ in range(2):
        run = PipelineRunner()
        run.start({'fields': fields, 'output_dir': str(out), 'preset': 'auto',
                   'stages': ['stack'], 'batch_root': str(session)})
        assert _wait(run) == 'done'
        roots.append(run.spec['output_dir'])

    assert roots[0] != roots[1]
    assert roots[0] == str(out / '2026-07-27')
    assert roots[1].startswith(str(out / '2026-07-27_'))
    for root in roots:
        assert (Path(root) / 'batch_summary.csv').exists()
        assert (Path(root) / 'activity.jsonl').exists()


def test_the_remembered_folder_is_the_one_the_user_picked(session, tmp_path, runner,
                                                          monkeypatch):
    """Remembering the subfolder instead would nest a second copy of the name inside it
    on the next run, and again on the one after that."""
    from mee2024 import stacker_implementation as si
    from mee2024.ui import runner as runner_module

    saved = {}
    monkeypatch.setattr(si, 'do_stack', lambda *a, **k: tmp_path / 'fake.zip')
    # capture the settings write rather than letting a test edit the user's own config
    monkeypatch.setattr('mee2024.MEE2024util.write_ini', saved.update)
    fields, _ = batch.find_fields(session)
    out = tmp_path / 'results'
    runner.start({'fields': fields, 'output_dir': str(out), 'preset': 'auto',
                  'stages': ['stack'], 'batch_root': str(session)})
    assert _wait(runner) == 'done'
    assert saved['output_dir'] == str(out)
    assert runner.spec['output_dir'] == str(out / '2026-07-27')


def test_one_bad_field_does_not_abandon_the_rest(session, tmp_path, runner, monkeypatch):
    """A night of observing is too expensive to lose to one bad folder."""
    from mee2024 import stacker_implementation as si

    calls = []

    def flaky(lights, darks, flats, options, progress=None):
        calls.append(options['output_dir'])
        if len(calls) == 2:
            raise RuntimeError('bad frame')
        return tmp_path / 'fake.zip'

    monkeypatch.setattr(si, 'do_stack', flaky)
    fields, _ = batch.find_fields(session)
    runner.start({'fields': fields, 'output_dir': str(tmp_path / 'out'),
                  'preset': 'auto', 'stages': ['stack']})
    assert _wait(runner) == 'done', 'the batch itself should not fail'
    assert len(calls) == 3, 'every field should have been attempted'
    statuses = [r['status'] for r in runner.batch_results]
    assert statuses == ['done', 'failed', 'done']
    assert 'bad frame' in runner.batch_results[1]['error']


def test_cancelling_stops_at_the_next_field(session, tmp_path, runner, monkeypatch):
    from mee2024 import stacker_implementation as si

    def stack_then_cancel(lights, darks, flats, options, progress=None):
        runner.cancel_event.set()
        return tmp_path / 'fake.zip'

    monkeypatch.setattr(si, 'do_stack', stack_then_cancel)
    fields, _ = batch.find_fields(session)
    runner.start({'fields': fields, 'output_dir': str(tmp_path / 'out'),
                  'preset': 'auto', 'stages': ['stack']})
    _wait(runner)
    assert len(runner.batch_results) == 1, 'should not begin a second field'


def test_the_batch_reports_a_field_per_event(session, tmp_path, runner, monkeypatch):
    from mee2024 import events
    from mee2024 import stacker_implementation as si

    monkeypatch.setattr(si, 'do_stack',
                        lambda *a, **k: tmp_path / 'fake.zip')
    fields, _ = batch.find_fields(session)
    runner.start({'fields': fields, 'output_dir': str(tmp_path / 'out'),
                  'preset': 'auto', 'stages': ['stack']})
    _wait(runner)
    kinds = [e['type'] for e in runner.sink.events]
    assert events.BATCH_STARTED in kinds and events.BATCH_FINISHED in kinds
    finished = [e for e in runner.sink.events if e['type'] == events.BATCH_FINISHED][-1]
    assert finished['n_done'] == 3 and finished['n_failed'] == 0


def test_a_batch_without_an_output_folder_is_refused(session, runner):
    """Every field's results go to a mirror of its own folder, so there has to be a root
    to mirror into -- and saying so up front beats writing three fields into one place."""
    fields, _ = batch.find_fields(session)
    with pytest.raises(ValueError, match='output folder'):
        runner.start({'fields': fields, 'preset': 'auto'})


# ------------------------------------------------------------------ fail fast

def _count_centroid_calls(monkeypatch):
    """Count how many frames get centroided, so the saving can be measured not assumed."""
    from mee2024 import stacker_implementation as si

    calls = []
    real = si.open_img_and_find_centroids

    def counting(file, options=None, dark=0, flat=1, hot=None):
        calls.append(file)
        return real(file, options or {}, dark, flat, hot)

    monkeypatch.setattr(si, 'open_img_and_find_centroids', counting)
    return calls


def test_unmatchable_frames_stop_after_two(tmp_path, monkeypatch):
    """Every frame is aligned against frame 0, so if the first two do not match the run was
    always going to die -- the point is not paying for the other frames first."""
    import numpy as np
    from mee2024 import stacker_implementation as si
    from mee2024.config import get_default_options

    # ten frames of pure noise: no centroids anywhere, so nothing can be matched
    rng = np.random.default_rng(0)
    files = []
    for i in range(10):
        path = tmp_path / f'noise_{i:03d}.fits'
        fits.writeto(path, rng.normal(100, 3, (64, 64)).astype(np.float32))
        files.append(str(path))

    calls = _count_centroid_calls(monkeypatch)
    options = get_default_options()
    options.update(flag_display=False, flag_display2=False, output_dir=str(tmp_path / 'o'),
                   centroid_gaussian_subtract=True)
    with pytest.raises(Exception) as exc:
        si.do_stack(files, [], [], options)
    assert len(calls) == 2, f'centroided {len(calls)} frames before giving up, wanted 2'
    assert 'Stopped after two frames' in str(exc.value)
    assert '8 first' in str(exc.value), 'should say how many frames it skipped'


def test_a_blank_first_frame_names_the_frame(tmp_path, monkeypatch):
    import numpy as np
    from mee2024 import stacker_implementation as si
    from mee2024.config import get_default_options

    files = []
    for i in range(4):
        path = tmp_path / f'flat_{i:03d}.fits'
        fits.writeto(path, np.full((64, 64), 100.0, dtype=np.float32))
        files.append(str(path))
    calls = _count_centroid_calls(monkeypatch)
    options = get_default_options()
    options.update(flag_display=False, flag_display2=False, output_dir=str(tmp_path / 'o'),
                   centroid_gaussian_subtract=True)
    with pytest.raises(Exception) as exc:
        si.do_stack(files, [], [], options)
    assert 'flat_000.fits' in str(exc.value), 'should name the offending frame'
    assert len(calls) == 2


def test_good_data_still_centroids_every_frame_exactly_once(tmp_path, monkeypatch):
    """The early probe must not cost anything on data that works: frames 0 and 1 are done
    in the probe and must not be done again in the main pass."""
    import numpy as np
    from mee2024 import stacker_implementation as si

    calls = []

    def fake(file, options=None, dark=0, flat=1, hot=None):
        calls.append(file)
        # a fixed pattern of 'centroids' that aligns with itself
        return [(100.0, 9.0, (10.0 + i, 20.0 + 2 * i)) for i in range(8)]

    monkeypatch.setattr(si, 'open_img_and_find_centroids', fake)
    files = [str(tmp_path / f'f{i}.fits') for i in range(6)]
    # only the centroid phase matters here, so stop before anything reads a file
    monkeypatch.setattr(si, 'attempt_align',
                        lambda c1, c2, options, guess=(0, 0), framenum=-1:
                        ((0, 0), {0: 0}, {0: 0}, (0.0, 0.0), 0.1))
    monkeypatch.setattr(si, 'open_image',
                        lambda f: np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(Exception):
        # it will fail later (no real frames to stack), but the centroid count is the point
        si.do_stack(files, [], [], _stack_options(tmp_path))
    assert sorted(calls) == sorted(files), 'every frame exactly once, none twice'
    assert len(calls) == len(files)


def _stack_options(tmp_path):
    from mee2024.config import get_default_options
    options = get_default_options()
    options.update(flag_display=False, flag_display2=False,
                   output_dir=str(tmp_path / 'out'), centroid_gaussian_subtract=True)
    return options
