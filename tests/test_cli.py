"""Argument parsing and option resolution for the command-line interface."""

import json

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
