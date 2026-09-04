"""Argument parsing and option resolution for the command-line interface."""

import json
from pathlib import Path

import pytest

from mee2024 import cli
from mee2024.config import DEFAULT_OPTIONS


def parse(argv):
    return cli.build_parser().parse_args(argv)


# ------------------------------------------------------------------ parsing

def test_every_subcommand_is_reachable():
    for name in ['gui', 'config', 'build-triangle-db']:
        assert parse([name]).command == name
    assert parse(['stack', 'a.fit']).command == 'stack'
    assert parse(['distortion', 'd.zip']).command == 'distortion'
    assert parse(['eclipse', 'e.zip']).command == 'eclipse'
    assert parse(['run', 'a.fit']).command == 'run'


def test_stack_collects_lights_darks_and_flats():
    args = parse(['stack', 'a.fit', 'b.fit', '--dark', 'd1.fit', 'd2.fit', '--flat', 'f.fit'])
    assert [str(p) for p in args.lights] == ['a.fit', 'b.fit']
    assert [str(p) for p in args.dark] == ['d1.fit', 'd2.fit']
    assert [str(p) for p in args.flat] == ['f.fit']


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        parse([])


def test_unknown_distortion_order_is_rejected():
    with pytest.raises(SystemExit):
        parse(['distortion', 'd.zip', '--order', 'octic'])


# ------------------------------------------------------- input file resolution

def _frames(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b'')
    return directory


def test_a_missing_light_frame_is_named_and_the_run_refused(tmp_path, capsys):
    """The whole point: no run starts, and the message says which path was wrong."""
    missing = tmp_path / 'nope.fits'
    assert cli.main(['stack', str(missing), '--no-config']) == 1
    error = capsys.readouterr().err
    assert 'no such light frame' in error
    assert str(missing) in error


def test_an_empty_light_frame_list_is_refused():
    """argparse's nargs='+' cannot see a glob that expanded to nothing, so check here."""
    with pytest.raises(RuntimeError, match='no light frames given'):
        cli.resolve_input_files([], 'light frame', required=True)


def test_cmd_stack_refuses_an_empty_light_frame_list_before_doing_any_work():
    args = parse(['stack', 'a.fits', '--no-config'])
    args.lights = []
    with pytest.raises(RuntimeError, match='no light frames given'):
        cli.cmd_stack(args)


def test_a_pattern_matching_only_subfolders_is_refused(tmp_path):
    """The reported reproduction: tests/data/fits/zwo3 holds only darks/ and field/.

    The shell passes the pattern through unexpanded because nothing matches it, and
    every frame path in the run is then that literal pattern.
    """
    zwo3 = tmp_path / 'zwo3'
    (zwo3 / 'darks').mkdir(parents=True)
    (zwo3 / 'field').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='matched no files'):
        cli.resolve_input_files([str(zwo3 / '*.fits')], 'light frame', required=True)


def test_a_pattern_matching_directories_says_so(tmp_path):
    zwo3 = tmp_path / 'zwo3'
    (zwo3 / 'darks').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='matched only directories'):
        cli.resolve_input_files([str(zwo3 / '*')], 'light frame')


def test_a_directory_given_as_a_frame_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match='is a directory, not a file'):
        cli.resolve_input_files([str(tmp_path)], 'light frame')


def test_the_cli_expands_a_glob_the_shell_left_alone(tmp_path):
    """cmd.exe never globs, so on Windows this is the normal path, not a corner case."""
    directory = _frames(tmp_path / 'lights', 'b.fits', 'a.fits')
    (directory / 'subfolder').mkdir()
    resolved = cli.resolve_input_files([str(directory / '*.fits')], 'light frame',
                                       required=True)
    assert [Path(p).name for p in resolved] == ['a.fits', 'b.fits'], (
        'expanded frames must be sorted, so a stack is reproducible run to run')


def test_an_already_expanded_list_survives_resolution(tmp_path):
    """The POSIX case: the shell globbed, and these are ordinary paths."""
    directory = _frames(tmp_path / 'lights', 'a.fits', 'b.fits')
    given = [directory / 'a.fits', directory / 'b.fits']
    assert cli.resolve_input_files(given, 'light frame') == [str(p) for p in given]


def test_darks_and_flats_are_checked_too(tmp_path):
    _frames(tmp_path, 'light.fits')
    with pytest.raises(RuntimeError, match='no such dark frame'):
        cli.resolve_input_files([tmp_path / 'missing_dark.fits'], 'dark frame')
    with pytest.raises(RuntimeError, match='no such flat frame'):
        cli.resolve_input_files([tmp_path / 'missing_flat.fits'], 'flat frame')


def test_no_darks_or_flats_is_not_an_error():
    assert cli.resolve_input_files([], 'dark frame') == []


def test_frames_are_checked_before_the_pattern_database_is_prepared(tmp_path, monkeypatch):
    """Preparing the database takes minutes; checking a path takes none of them."""
    def fail(*args, **kwargs):
        raise AssertionError('the pattern database was prepared before the inputs '
                             'were checked')

    monkeypatch.setattr(cli, '_prepare_pattern_db', fail)
    args = parse(['stack', str(tmp_path / 'nope.fits'), '--no-config'])
    with pytest.raises(RuntimeError, match='no such light frame'):
        cli.cmd_stack(args)


def test_run_checks_its_frames_before_preparing_the_catalogue(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('the catalogue was prepared before the inputs were checked')

    monkeypatch.setattr(cli, '_prepare_catalogue', fail)
    args = parse(['run', str(tmp_path / 'nope.fits'), '--no-config'])
    with pytest.raises(RuntimeError, match='no such light frame'):
        cli.cmd_run(args)


# ------------------------------------------------------------- type coercion

@pytest.mark.parametrize('raw,expected', [
    ('true', True), ('True', True), ('1', True), ('yes', True), ('on', True),
    ('false', False), ('False', False), ('0', False), ('no', False), ('off', False),
])
def test_coerce_bool_options(raw, expected):
    assert cli.coerce_option('flag_display', raw) is expected


def test_coerce_int_and_float_and_str_options():
    assert cli.coerce_option('min_area', '7') == 7
    assert isinstance(cli.coerce_option('min_area', '7'), int)
    assert cli.coerce_option('distortion_fit_tol', '0.5') == 0.5
    assert cli.coerce_option('distortionOrder', 'quintic') == 'quintic'
    assert cli.coerce_option('distortion_free_scale', 'True') is True
    assert cli.coerce_option('distortion_fit_tol_initial', '20') == 20.0


def test_coerce_rejects_an_unknown_option():
    with pytest.raises(ValueError, match='unknown option'):
        cli.coerce_option('not_a_real_option', '1')


def test_coerce_rejects_a_non_boolean_for_a_boolean_option():
    with pytest.raises(ValueError, match='expects a boolean'):
        cli.coerce_option('flag_display', 'maybe')


def test_coerce_rejects_a_fractional_value_for_an_integer_option():
    with pytest.raises(ValueError, match='whole number'):
        cli.coerce_option('min_area', '3.5')


def test_coerce_accepts_an_integral_float_for_an_integer_option():
    assert cli.coerce_option('min_area', '4.0') == 4


def test_apply_sets_requires_key_equals_value():
    with pytest.raises(ValueError, match='key=value'):
        cli.apply_sets({}, ['just_a_key'])


# --------------------------------------------------------- option resolution

def test_resolve_options_starts_from_defaults_with_no_config():
    args = parse(['stack', 'a.fit', '--no-config'])
    options = cli.resolve_options(args)
    assert options['distortionOrder'] == DEFAULT_OPTIONS['distortionOrder']


def test_no_display_switches_off_all_three_display_flags():
    args = parse(['stack', 'a.fit', '--no-config', '--no-display'])
    options = cli.resolve_options(args)
    assert not options['flag_display']
    assert not options['flag_display2']
    assert not options['flag_display3']


def test_set_overrides_win_over_flags():
    args = parse(['distortion', 'd.zip', '--no-config',
                  '--order', 'cubic', '--set', 'distortionOrder=septic'])
    assert cli.resolve_options(args)['distortionOrder'] == 'septic'


def test_flags_win_over_the_config_file(tmp_path):
    cfg = tmp_path / 'cfg.txt'
    cfg.write_text(json.dumps({'distortionOrder': 'linear'}), encoding='utf-8')
    args = parse(['distortion', 'd.zip', '--config', str(cfg), '--order', 'quintic'])
    assert cli.resolve_options(args)['distortionOrder'] == 'quintic'


def test_config_file_wins_over_defaults(tmp_path):
    cfg = tmp_path / 'cfg.txt'
    cfg.write_text(json.dumps({'max_star_mag_dist': 9.5}), encoding='utf-8')
    args = parse(['distortion', 'd.zip', '--config', str(cfg)])
    assert cli.resolve_options(args)['max_star_mag_dist'] == 9.5


def test_output_dir_flag_is_applied():
    from pathlib import Path
    args = parse(['stack', 'a.fit', '--no-config', '-o', 'some/dir'])
    assert cli.resolve_options(args)['output_dir'] == str(Path('some/dir'))


def test_fix_distortion_files_are_joined_with_semicolons():
    args = parse(['distortion', 'd.zip', '--no-config',
                  '--fix-distortion', 'a.zip', 'b.zip'])
    assert cli.resolve_options(args)['distortion_reference_files'] == 'a.zip;b.zip'


def test_multiple_set_flags_all_apply():
    args = parse(['stack', 'a.fit', '--no-config',
                  '--set', 'min_area=9', '--set', 'sigma_subtract=1.5'])
    options = cli.resolve_options(args)
    assert options['min_area'] == 9
    assert options['sigma_subtract'] == 1.5


# ---------------------------------------------------------------- config cmd

def test_config_set_writes_the_file(tmp_path, capsys):
    cfg = tmp_path / 'cfg.txt'
    args = parse(['config', '--config', str(cfg), '--set', 'min_area=6'])
    cli.cmd_config(args)
    assert json.loads(cfg.read_text(encoding='utf-8'))['min_area'] == 6


def test_config_show_prints_valid_json(tmp_path, capsys):
    args = parse(['config', '--no-config'])
    cli.cmd_config(args)
    printed = capsys.readouterr().out
    # the last JSON object printed is the options dump
    options = json.loads(printed[printed.index('{'):])
    assert options['distortionOrder'] == DEFAULT_OPTIONS['distortionOrder']


def test_main_reports_a_bad_set_as_a_usage_error():
    with pytest.raises(SystemExit):
        cli.main(['config', '--no-config', '--set', 'nonsense=1'])


# ------------------------------------------------------- catalogue pack/install

def test_catalogue_accepts_pack_and_install_options():
    assert parse(['catalogue', '--pack', 'gaia_dr3_g12']).pack == 'gaia_dr3_g12'
    args = parse(['catalogue', '--install', 'a.zip', '--name', 'b', '--force'])
    assert args.install == 'a.zip' and args.name == 'b' and args.force


def _tiny_catalogue(directory, n=40, seed=0):
    """A minimal but valid catalogue on disk."""
    import numpy as np
    from mee2024.starcat import StarTable, store
    rng = np.random.default_rng(seed)
    table = StarTable(ra=np.radians(rng.uniform(0, 360, n)),
                      dec=np.radians(rng.uniform(-30, 30, n)),
                      mag=rng.uniform(4, 11, n),
                      ids=np.arange(1, n + 1, dtype=np.int64), epoch=2016.0)
    store.write_catalogue(directory, table, name=directory.name, magnitude_limit=12.0)
    return table


def test_pack_then_install_round_trips(tmp_path, monkeypatch, capsys):
    """The whole point: move a catalogue between machines with two commands."""
    from mee2024.starcat import store

    root = tmp_path / 'catalogues'
    (root / 'source_cat').mkdir(parents=True)
    _tiny_catalogue(root / 'source_cat')
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root', lambda: root)

    archive = tmp_path / 'moved.zip'
    assert cli.main(['catalogue', '--no-config', '--pack', 'source_cat',
                     '--out', str(archive)]) == 0
    assert archive.exists()
    assert 'sha256' in capsys.readouterr().out

    assert cli.main(['catalogue', '--no-config', '--install', str(archive),
                     '--name', 'arrived']) == 0
    assert store.verify(root / 'arrived') == []
    assert store.read_manifest(root / 'arrived')['n_stars'] == 40


def test_install_refuses_to_clobber_without_force(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'catalogues'
    (root / 'cat').mkdir(parents=True)
    _tiny_catalogue(root / 'cat')
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root', lambda: root)

    archive = tmp_path / 'cat.zip'
    cli.main(['catalogue', '--no-config', '--pack', 'cat', '--out', str(archive)])
    capsys.readouterr()

    assert cli.main(['catalogue', '--no-config', '--install', str(archive)]) == 1
    assert 'already exists' in capsys.readouterr().out
    assert cli.main(['catalogue', '--no-config', '--install', str(archive),
                     '--force']) == 0


def test_pack_of_a_missing_catalogue_reports_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root',
                        lambda: tmp_path / 'catalogues')
    (tmp_path / 'catalogues').mkdir()
    assert cli.main(['catalogue', '--no-config', '--pack', 'nope']) == 1
    assert 'not installed' in capsys.readouterr().out


def test_install_of_a_missing_archive_reports_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root',
                        lambda: tmp_path / 'catalogues')
    (tmp_path / 'catalogues').mkdir()
    assert cli.main(['catalogue', '--no-config', '--install', str(tmp_path / 'x.zip')]) == 1
    assert 'no such archive' in capsys.readouterr().out


def test_install_rejects_a_corrupted_archive(tmp_path, monkeypatch):
    """Checksums in the manifest are what make an unattended transfer trustworthy."""
    import zipfile
    root = tmp_path / 'catalogues'
    (root / 'cat').mkdir(parents=True)
    _tiny_catalogue(root / 'cat')
    monkeypatch.setattr('mee2024.MEE2024util.get_catalogue_root', lambda: root)

    archive = tmp_path / 'cat.zip'
    cli.main(['catalogue', '--no-config', '--pack', 'cat', '--out', str(archive)])

    # rewrite one column with the wrong bytes, keeping the manifest untouched
    corrupt = tmp_path / 'corrupt.zip'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(corrupt, 'w') as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith('mag.npy'):
                data = data[:-8] + b'\x00' * 8
            dst.writestr(item, data)

    assert cli.main(['catalogue', '--no-config', '--install', str(corrupt),
                     '--name', 'bad']) == 1
    assert not (root / 'bad').exists(), (
        'a failed transfer must not leave a half-installed catalogue behind')


# ---------------------------------------------------- default interface & remote check

def test_no_arguments_opens_the_app_by_default(monkeypatch):
    """From v1.0.0 a double-clicked exe opens the new interface, not the classic one."""
    from mee2024 import main as main_module

    called = []
    monkeypatch.setattr(main_module, 'default_interface', lambda: 'app')
    # patch the function on the module itself: `from mee2024.ui import app` binds the
    # package attribute, so swapping sys.modules would not be seen and the real UI
    # would launch and block
    monkeypatch.setattr('mee2024.ui.app.launch',
                        lambda *a, **k: called.append('app') or 0)
    assert main_module.main([]) == 0
    assert called == ['app']


def test_the_classic_interface_can_be_made_the_default(monkeypatch):
    from mee2024 import main as main_module

    called = []
    monkeypatch.setattr(main_module, 'default_interface', lambda: 'classic')
    monkeypatch.setattr(main_module, 'run_gui', lambda: called.append('classic'))
    assert main_module.main([]) == 0
    assert called == ['classic']


def test_a_failing_app_window_falls_back_to_the_classic_interface(monkeypatch):
    """A double-clicked exe has no console, so a failure must still leave a usable app."""
    from mee2024 import main as main_module

    called = []
    monkeypatch.setattr(main_module, 'default_interface', lambda: 'app')

    def explode(*args, **kwargs):
        raise RuntimeError('no display')

    monkeypatch.setattr('mee2024.ui.app.launch', explode)
    monkeypatch.setattr(main_module, 'run_gui', lambda: called.append('classic'))
    assert main_module.main([]) == 0
    assert called == ['classic']


@pytest.mark.parametrize('value,expected', [
    ('app', 'app'), ('classic', 'classic'), ('legacy', 'classic'), ('gui', 'classic'),
    ('anything else', 'app'),
])
def test_default_interface_reads_the_config(tmp_path, monkeypatch, value, expected):
    import json as _json
    from mee2024 import main as main_module

    config = tmp_path / 'cfg.txt'
    config.write_text(_json.dumps({'default_interface': value}), encoding='utf-8')
    monkeypatch.setattr('mee2024.MEE2024util.get_config_path', lambda: config)
    assert main_module.default_interface() == expected


def test_check_remote_reports_a_missing_asset_clearly(monkeypatch, capsys):
    """The command exists so a release can be verified without downloading 138 MB."""
    import urllib.error
    from mee2024.starcat import download

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError('u', 404, 'Not Found', {}, None)

    monkeypatch.setattr(download.urllib.request, 'urlopen', fake_urlopen)
    assert cli.main(['catalogue', '--no-config', '--check-remote']) == 1
    output = capsys.readouterr().out
    assert 'FAIL' in output
    assert 'catalogues-v1' in output, 'the message should name the expected tag'
    assert 'draft' in output, 'a draft release is the most likely cause'


def test_check_remote_accepts_a_correctly_published_asset(monkeypatch, capsys):
    """Every published asset serves the size its registry entry claims."""
    from mee2024.starcat import download

    # keyed on the URL, so this stays honest as more archives are published: a fixed
    # size would pass only while exactly one release had one
    by_url = {r.url: r.size_bytes for r in download.RELEASES.values() if r.url}

    class FakeResponse:
        def __init__(self, request):
            self.headers = {'Content-Length': str(by_url[request.full_url])}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(download.urllib.request, 'urlopen',
                        lambda request, **k: FakeResponse(request))
    assert cli.main(['catalogue', '--no-config', '--check-remote']) == 0
    assert 'reachable' in capsys.readouterr().out


def test_check_remote_notices_a_different_file_was_uploaded(monkeypatch, capsys):
    """A re-zipped upload has the wrong size, and would fail hash verification later."""
    from mee2024.starcat import download

    class WrongSize:
        headers = {'Content-Length': '999'}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: WrongSize())
    assert cli.main(['catalogue', '--no-config', '--check-remote']) == 1
    assert 'size mismatch' in capsys.readouterr().out


def test_a_successful_stack_exits_zero(tmp_path, monkeypatch):
    """`stack` and `distortion` return the archive they wrote, which is useful in Python and
    wrong as an exit code: sys.exit() treats a non-integer as an error message and exits 1.
    Every successful run therefore reported failure, so `mee2024 stack ... && next` never ran
    the next step."""
    from mee2024 import cli

    monkeypatch.setattr(cli, 'cmd_stack', lambda args: tmp_path / 'centroid_data.zip')
    parser = cli.build_parser()
    monkeypatch.setattr(cli, 'build_parser', lambda: parser)
    for action in parser._subparsers._group_actions[0].choices.values():
        if action.get_default('func') is not None and 'stack' in str(action.prog):
            action.set_defaults(func=cli.cmd_stack)
    assert cli.main(['stack', str(tmp_path / 'x.fits')]) == 0


def test_a_command_that_returns_an_int_keeps_it(monkeypatch):
    from mee2024 import cli

    parser = cli.build_parser()
    monkeypatch.setattr(cli, 'build_parser', lambda: parser)
    for action in parser._subparsers._group_actions[0].choices.values():
        if 'config' in str(action.prog):
            action.set_defaults(func=lambda args: 3)
    assert cli.main(['config']) == 3
