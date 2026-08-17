"""One test per bug fixed during the 2026-07 refactor.

Each of these fails against the code as it was before the fix.
"""

import inspect

import numpy as np
import pytest

from mee2024 import distortion_fitter, eclipse_analysis, gaia_search, transforms
from mee2024 import stacker_implementation as si


def test_rough_match_threshold_converts_arcsec_to_degrees():
    """Bug 1: the divisor was 33600, making the tolerance 9.33x tighter than the UI said.

    rough_match_threshhold is in arcseconds and the arrays being compared are in
    degrees, so the conversion must be /3600. platesolve_triangle.match_centroids
    already did it correctly; distortion_fitter.match_centroids did not.
    """
    source = inspect.getsource(distortion_fitter.match_centroids)
    assert '/ 33600' not in source and '/33600' not in source
    assert "options['rough_match_threshhold'] / 3600" in source


def test_double_star_query_takes_the_cosine_of_radians():
    """Bug 2: np.cos() was applied to a declination in degrees.

    The RA half-width of the double-star search box must widen by 1/cos(dec); feeding
    degrees to np.cos gives a nonsensical (and sometimes negative) box.
    """
    source = inspect.getsource(gaia_search.lookup_nearby)
    assert 'np.cos(np.radians(dec))' in source
    assert 'np.cos(dec)' not in source


class _MinimalStarData:
    """Enough of StarData to exercise __copy__."""

    def __init__(self):
        self.epoch = 'EPOCH-SENTINEL'
        self.mags = np.array([1.0])
        self.vectors = np.zeros((1, 3))
        self.ids = np.array([1])
        self.c = None
        self.pm = np.zeros((1, 2))
        self.parallax = np.ones(1)
        self.has_pm = True


def test_stardata_copy_preserves_the_epoch():
    """Bug 3: __copy__ assigned to `newone.epch`, so copies had no `.epoch` at all."""
    from mee2024.StarData import StarData
    original = _MinimalStarData()
    duplicate = StarData.__copy__(original)
    assert hasattr(duplicate, 'epoch'), '__copy__ dropped the epoch (the `epch` typo)'
    assert duplicate.epoch == 'EPOCH-SENTINEL'
    assert not hasattr(duplicate, 'epch')


def test_residual_plot_guards_the_right_frame_index():
    """Bug 4: the guard read shifts[i-1] while indexing deltas[i-1] and rms_errors[i-1].

    shifts has one entry per frame (including frame 0); deltas and rms_errors have one
    per *aligned* frame. Checking shifts[i-1] tested the previous frame's success.
    """
    # _do_stack holds the body; do_stack is the wrapper that releases the log file
    source = inspect.getsource(si._do_stack)
    assert 'if shifts[i-1] is None' not in source
    assert 'if deltas[i-1] is None' in source


def test_eclipse_report_labels_method_two_correctly():
    """Bug 5: the Method 2 line in the report was labelled 'Method 1 results'."""
    source = inspect.getsource(eclipse_analysis.eclipse_analysis)
    assert source.count('Method 2 results') == 1
    assert source.count('Method 1 results') == 1


def test_eclipse_analysis_never_shows_plots_unconditionally():
    """Bug 6: two bare plt.show() calls made headless/batch runs impossible."""
    source = inspect.getsource(eclipse_analysis.eclipse_analysis)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == 'plt.show()':
            indent = len(line) - len(line.lstrip())
            assert indent >= 8, f'unguarded plt.show() at indent {indent}'


def test_eclipse_analysis_does_not_reference_a_conditional_axis():
    """Bug 6b: show_deflection_scatter used `ax`, only defined when flag_display3 was on.

    With graphics off, that raised NameError before any result was written.
    """
    source = inspect.getsource(eclipse_analysis.eclipse_analysis)
    assert 'ax.tick_params' not in source


def test_tab3_validates_its_own_output_directory():
    """Bug 7: tab 3 checked and opened output_dir2, which belongs to tab 2."""
    from mee2024 import UI_handler
    source = inspect.getsource(UI_handler.inputUI)
    assert "values['output_dir3']" in source


def test_icoord_to_vector_leaves_its_argument_alone():
    """Bug 8: reshape returned a view, so the function wrote into the caller's array."""
    icoords = np.array([[0.1, 0.2], [0.3, -0.4]])
    before = icoords.copy()
    transforms.icoord_to_vector(icoords)
    assert np.array_equal(icoords, before)


def test_icoord_to_vector_is_idempotent_across_calls():
    """The practical consequence of bug 8: calling twice gave different answers."""
    icoords = np.array([[0.1, 0.2], [0.3, -0.4]])
    first = transforms.icoord_to_vector(icoords)
    second = transforms.icoord_to_vector(icoords)
    assert np.allclose(first, second)


def test_write_ini_error_path_does_not_itself_raise(tmp_path):
    """write_ini's failure branch concatenated a str with a Path, raising TypeError."""
    from mee2024 import MEE2024util
    # a directory is not writable as a file: exercise the error path
    MEE2024util.write_ini({'a': 1}, path=tmp_path)  # must not raise


def test_open_distortion_files_with_no_references_is_quiet(options):
    """np.mean([]) emitted a RuntimeWarning on every ordinary run."""
    from mee2024 import distortion_polynomial as dp
    options['distortion_reference_files'] = ''
    with np.errstate(all='raise'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            coeff_x, coeff_y, platescale, uncertainty = dp._open_distortion_files(options)
    assert coeff_x == {} and coeff_y == {}
    assert np.isnan(platescale)
    assert uncertainty == -1


def test_stack_preview_plot_downsamples_before_the_colormap():
    """imshow applies its colormap at the resolution of the array it is handed.

    On a 3520x4656 stack that is a 499 MiB float64 RGBA intermediate, which fails outright
    on a memory-pressured machine -- observed as a MemoryError during a watch-mode run.
    The plot must therefore stride the image and use `extent` to keep the star overlay in
    original pixel coordinates.
    """
    # _do_stack holds the body; do_stack is the wrapper that releases the log file
    source = inspect.getsource(si._do_stack)
    assert 'display_step' in source, 'the stack preview no longer downsamples'
    assert 'extent=' in source, 'downsampling without extent would misplace the overlay'
    assert 'plt.imshow(stacked,' not in source, 'full-resolution imshow is back'


def test_extent_keeps_overlay_coordinates_after_striding():
    """The mechanism itself: a strided image plus extent spans the original grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    height, width = 400, 600
    image = np.zeros((height, width))
    step = 4
    figure, axes = plt.subplots()
    try:
        axes.imshow(image[::step, ::step], extent=(0, width, height, 0))
        assert axes.get_xlim() == (0, width)
        assert axes.get_ylim() == (height, 0)
    finally:
        plt.close(figure)


def test_observation_date_is_recorded_for_the_guess_check():
    """Stage 1 must record the header date, or stage 2 cannot score its blind guess."""
    # _do_stack holds the body; do_stack is the wrapper that releases the log file
    source = inspect.getsource(si._do_stack)
    assert 'observation_date_header' in source
    assert 'read_observation_date' in source


def test_config_migration_is_version_targeted():
    """A future version bump must not silently reset settings the user chose.

    Before v1.0.0 the migration reset rough_match_threshhold on *every* version change.
    """
    from mee2024.MEE2024util import migrate_config

    old = {'__version__': 'v0.6.0', 'rough_match_threshhold': 200.0}
    notes = migrate_config(old)
    assert old['rough_match_threshhold'] == 36
    assert notes and 'units bug' in notes[0]

    current = {'__version__': 'v1.0.0', 'rough_match_threshhold': 120.0}
    assert migrate_config(current) == []
    assert current['rough_match_threshhold'] == 120.0, 'must not touch a current config'


def test_both_star_label_emitters_pass_sky_positions():
    """A proper name is usually only reachable by position: Gaia's crossmatch to Hipparcos
    misses 46 of the 49 named stars. The resolver was added and unit-tested, but the
    stage-2 call site was not updated -- and since that event supersedes stage 1's in the
    frontend, every label fell back to a magnitude and the fix looked like it had failed.

    Testing the resolver in isolation cannot catch this; testing the wiring can.
    """
    import inspect

    from mee2024 import distortion_fitter, stacker_implementation

    for module in (distortion_fitter, stacker_implementation):
        source = inspect.getsource(module)
        for call in ('star_labels.emit(', 'star_labels.emit_from_solution('):
            start = source.find(call)
            while start != -1:
                # the argument list ends at the matching close bracket
                depth, i = 0, start + len(call) - 1
                while i < len(source):
                    if source[i] == '(':
                        depth += 1
                    elif source[i] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                args = source[start:i]
                assert 'epoch=' in args, (
                    f'{module.__name__}: {call} without an epoch -- naming by position '
                    f'against the wrong epoch loses the fastest-moving stars')
                if call == 'star_labels.emit(':
                    assert 'ra=' in args and 'dec=' in args, (
                        f'{module.__name__}: {call} without ra/dec, so no proper name can '
                        f'be resolved by position')
                start = source.find(call, i)


def test_close_logger_releases_the_log_file(tmp_path):
    """The run's LOG file was held open until the program exited.

    setup_logger attaches a logging.FileHandler to a *named* logger, which logging keeps
    in a process-global registry, so nothing ever closed it: one leaked descriptor per
    run, and on Windows an exclusive lock -- the user could not move or delete their own
    log while the app was still open. Deleting the file is the honest test of the lock;
    on Windows it raises PermissionError while a handle is open.
    """
    import logging

    from mee2024.MEE2024util import close_logger, setup_logger

    path = tmp_path / 'LOG20260807000000.txt'
    logger = setup_logger('test-logger-release', path)
    logger.info('a line, so the file is really written to')
    assert path.exists()

    close_logger(logger)

    assert logger.handlers == [], 'handlers still attached'
    assert 'test-logger-release' not in logging.Logger.manager.loggerDict, \
        'the named logger lingers in the registry, one per run'
    assert path.read_text(encoding='utf-8').strip(), 'the log lost its contents'
    path.unlink()                      # PermissionError here on Windows if still open
    close_logger(logger)               # idempotent: must not raise on a second call


def test_do_stack_releases_its_log_file_even_when_the_run_fails(tmp_path):
    """A failed run must let go of its log too -- that is when you want to read it."""
    from mee2024 import stacker_implementation
    from mee2024.config import get_default_options

    options = get_default_options()
    options['output_dir'] = str(tmp_path)
    with pytest.raises(Exception):
        stacker_implementation.do_stack([str(tmp_path / 'no_such_frame.fits')], [], [],
                                        options)
    logs = list(tmp_path.glob('CENTROID_OUTPUT*/LOG*.txt'))
    assert logs, 'the run left no log to check'
    for log in logs:
        log.unlink()                   # fails with PermissionError if the handle leaked


def test_erfa_ld_is_restored_when_correct_ra_dec_raises():
    """The erfa.ld monkey-patch was only reverted on the success path.

    correct_ra_dec disables (or rescales) astropy's gravitational light deflection by
    patching the module-global erfa.ld, and restored the original at the end of its
    body. Any exception in between -- a malformed observation date is enough -- left
    the patch installed, silently zeroing the deflection of every later astropy
    computation in the process. For a deflection experiment that is the worst
    available failure mode, so the restore must survive the error path too.
    """
    import erfa

    from mee2024.refraction_correction import AstroCorrect

    original = erfa.ld
    corrector = AstroCorrect()
    options = {'enable_gravitational_def': False,
               'observation_lat': 51.5, 'observation_long': 0.0,
               'observation_height': 0.0,
               'observation_date': 'not-a-date', 'observation_time': '99:99:99'}
    try:
        with pytest.raises(Exception):
            corrector.correct_ra_dec(None, options)
        assert erfa.ld is original, 'erfa.ld left patched after an exception'
    finally:
        erfa.ld = original  # never poison the rest of the suite, even if this fails


def test_open_image_names_a_missing_file(tmp_path):
    """open_image blamed cvtColor for a path that was never openable.

    A path matching no file falls out of the FITS branch, and cv2.imread reports that
    by returning None; the None went straight into cv2.cvtColor, so the run died with
    '(-215:Assertion failed) !_src.empty() in function cv::cvtColor', naming neither
    the file nor the reason. An unexpanded shell glob is the usual way to get here.
    """
    missing = tmp_path / 'zwo3' / 'not_a_real_frame.fits'
    with pytest.raises(FileNotFoundError, match='not_a_real_frame'):
        si.open_image(str(missing))


def test_open_image_names_a_file_it_cannot_decode(tmp_path):
    """The other half: the file is there, but is neither FITS nor a readable image."""
    junk = tmp_path / 'truncated.fits'
    junk.write_bytes(b'this is not a FITS header')
    with pytest.raises(ValueError, match='truncated') as caught:
        si.open_image(str(junk))
    assert 'FITS reader said' in str(caught.value), (
        'the swallowed astropy error is the only clue to what is wrong with the file')


def test_the_stage2_archive_name_states_each_timestamp_once():
    """`Path(path_data).stem + data['starttime']` recited the stage-1 start time twice.

    The result was an 89-character file name, and a working folder of the same length
    beside it, before the field tree a batch puts above them -- against a 260-character
    limit on Windows.
    """
    label = distortion_fitter.stage1_label('D:/out/centroid_data20260808013718.zip',
                                           '20260808013718')
    assert label == '20260808013718'
    assert len(f'distortion_data20260808013735__{label}.zip') == 49   # was 89


def test_the_stage2_archive_name_still_identifies_an_unconventional_input():
    """A renamed or hand-made stage-1 archive keeps its name in the label, because that
    is the only thing tying the fit to its input."""
    assert distortion_fitter.stage1_label('D:/out/rerun_of_field_7.zip',
                                          '20260808013718') == \
        'rerun_of_field_720260808013718'


# ------------------------------------------------- the epoch, decided where the data is (I9)

def test_the_fitter_prefers_the_header_epoch_over_guessing():
    """The header-date fix used to live in one front end's options assembly, so whether it
    applied depended on which of three interfaces was used -- and the CLI, the one most
    likely to be scripted and left unattended, fell back to a 2023 default. Stage 1 already
    writes observation_date_header into the archive, so stage 2 can decide it from the data."""
    from mee2024.distortion_fitter import resolve_epoch

    options = {'guess_date': True, 'observation_date': '2023-12-01'}
    resolved = resolve_epoch(options, {'observation_date_header': '2026-08-11'})
    assert resolved['observation_date'] == '2026-08-11'
    assert resolved['guess_date'] is False
    # the caller's dict is not mutated underneath it
    assert options['guess_date'] is True


def test_an_explicit_date_outranks_the_header_but_the_disagreement_is_reported():
    """An explicit instruction wins -- but one of the two is wrong and the fit cannot tell
    which, so it says so rather than choosing quietly."""
    from mee2024 import events
    from mee2024.distortion_fitter import resolve_epoch

    sink = events.ListSink()
    with events.using(events.EventBus([sink])):
        resolved = resolve_epoch({'guess_date': False, 'observation_date': '2026-01-01'},
                                 {'observation_date_header': '2026-08-11'})
    assert resolved['observation_date'] == '2026-01-01'
    messages = [e.get('text', '') for e in sink.events]
    assert any('disagree' in m for m in messages), messages


def test_no_header_date_leaves_the_guesser_alone():
    from mee2024.distortion_fitter import resolve_epoch

    options = {'guess_date': True, 'observation_date': '2023-12-01'}
    assert resolve_epoch(options, {}) is options
    assert resolve_epoch(options, {'observation_date_header': None}) is options
