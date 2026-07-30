"""One test per bug fixed during the 2026-07 refactor.

Each of these fails against the code as it was before the fix.
"""

import inspect

import numpy as np

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
    source = inspect.getsource(si.do_stack)
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
