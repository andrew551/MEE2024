"""
Command-line interface for MEE2024.

Every pipeline stage is reachable headlessly:

    mee2024 stack       LIGHTS...          -> centroid_data<ts>.zip
    mee2024 distortion  DATA.zip           -> distortion_data<ts>__<name>.zip
    mee2024 eclipse     DISTORTION.zip     -> ECLIPSE_OUTPUT<ts>.txt
    mee2024 run         LIGHTS...          -> stage 1 then stage 2 (then 3 with --eclipse)
    mee2024 config      --show / --set k=v
    mee2024 gui

Options are resolved in this order, later winning:
    built-in defaults  ->  config file  ->  named flags  ->  --set key=value
"""

import argparse
import json
import sys
from pathlib import Path

from mee2024 import MEE2024util
from mee2024.config import DEFAULT_OPTIONS, get_default_options
from mee2024.progress import NullProgress, TextProgress

_TRUE = {'1', 'true', 'yes', 'on'}
_FALSE = {'0', 'false', 'no', 'off'}


def coerce_option(key, raw):
    """Coerce a string from --set to the type of that option's default."""
    if key not in DEFAULT_OPTIONS:
        raise ValueError(f"unknown option {key!r} (see `mee2024 config --show`)")
    default = DEFAULT_OPTIONS[key]
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"option {key!r} expects a boolean, got {raw!r}")
    if isinstance(default, int):
        value = float(raw)
        if value != int(value):
            raise ValueError(f"option {key!r} expects a whole number, got {raw!r}")
        return int(value)
    if isinstance(default, float):
        return float(raw)
    return raw


def apply_sets(options, assignments):
    for assignment in assignments or []:
        if '=' not in assignment:
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        key, raw = assignment.split('=', 1)
        options[key.strip()] = coerce_option(key.strip(), raw)
    return options


def resolve_options(args):
    """Defaults, then the config file, then flags, then --set."""
    options = get_default_options()
    if not getattr(args, 'no_config', False):
        MEE2024util.read_ini(options, path=getattr(args, 'config', None))

    if getattr(args, 'output_dir', None) is not None:
        options['output_dir'] = str(args.output_dir)
    if getattr(args, 'no_display', False):
        options['flag_display'] = False
        options['flag_display2'] = False
        options['flag_display3'] = False
    if getattr(args, 'catalogue', None) is not None:
        options['catalogue'] = args.catalogue
    if getattr(args, 'order', None) is not None:
        options['distortionOrder'] = args.order
    if getattr(args, 'fix_distortion', None):
        options['distortion_reference_files'] = ';'.join(str(p) for p in args.fix_distortion)
    if getattr(args, 'limiting_mag', None) is not None:
        options['eclipse_limiting_mag'] = args.limiting_mag

    apply_sets(options, getattr(args, 'set', None))
    return options


def make_progress(args):
    return NullProgress() if getattr(args, 'quiet', False) else TextProgress()


def _use_headless_backend(options):
    """Select a non-interactive matplotlib backend when nothing will be shown."""
    if not (options['flag_display'] or options['flag_display2'] or options['flag_display3']):
        import matplotlib
        matplotlib.use('Agg')


# --------------------------------------------------------------------------- commands

def cmd_stack(args):
    options = resolve_options(args)
    _use_headless_backend(options)
    from mee2024 import database_cache, stacker_implementation
    try:
        lights = [str(p) for p in args.lights]
        darks = [str(p) for p in (args.dark or [])]
        flats = [str(p) for p in (args.flat or [])]
        zip_path = stacker_implementation.do_stack(lights, darks, flats, options,
                                                   progress=make_progress(args))
    finally:
        database_cache.shutdown_triangles()
    print(f'stage 1 output: {zip_path}')
    return zip_path


def cmd_distortion(args):
    options = resolve_options(args)
    _use_headless_backend(options)
    from mee2024 import database_cache, distortion_fitter
    try:
        zip_path = distortion_fitter.match_and_fit_distortion(str(args.data), options, None)
    finally:
        database_cache.shutdown_triangles()
    print(f'stage 2 output: {zip_path}')
    return zip_path


def cmd_eclipse(args):
    options = resolve_options(args)
    _use_headless_backend(options)
    from mee2024 import eclipse_analysis
    eclipse_analysis.eclipse_analysis(str(args.distortion), options)
    print('stage 3 complete')
    return 0


def cmd_run(args):
    """Stage 1 then stage 2, and stage 3 too if --eclipse was given."""
    options = resolve_options(args)
    _use_headless_backend(options)
    from mee2024 import database_cache, distortion_fitter, stacker_implementation
    try:
        lights = [str(p) for p in args.lights]
        darks = [str(p) for p in (args.dark or [])]
        flats = [str(p) for p in (args.flat or [])]
        progress = make_progress(args)
        centroid_zip = stacker_implementation.do_stack(lights, darks, flats, options,
                                                       progress=progress)
        print(f'stage 1 output: {centroid_zip}')
        distortion_zip = distortion_fitter.match_and_fit_distortion(str(centroid_zip), options, None)
        print(f'stage 2 output: {distortion_zip}')
    finally:
        database_cache.shutdown_triangles()
    if args.eclipse:
        from mee2024 import eclipse_analysis
        eclipse_analysis.eclipse_analysis(str(distortion_zip), options)
        print('stage 3 complete')
    return 0


def cmd_config(args):
    if args.set:
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        apply_sets(options, args.set)
        MEE2024util.write_ini(options, path=args.config)
        print(f'wrote {args.config or MEE2024util.get_config_path()}')
        return 0
    options = resolve_options(args)
    if args.show_path:
        print(MEE2024util.get_config_path())
    else:
        print(json.dumps(options, sort_keys=True, indent=4))
    return 0


def cmd_gui(args):
    from mee2024 import main as main_module
    main_module.run_gui()
    return 0


def cmd_catalogue(args):
    """Inspect, verify or fetch the offline star catalogues."""
    from mee2024.starcat import download, store

    if args.verify:
        release = download.get_release(args.verify)
        directory = release.directory()
        if not release.is_installed():
            print(f'{args.verify} is not installed at {directory}')
            return 1
        problems = store.verify(directory)
        if problems:
            print(f'{args.verify}: FAILED')
            for problem in problems:
                print(f'  {problem}')
            return 1
        manifest = store.read_manifest(directory)
        print(f'{args.verify}: OK -- {manifest["n_stars"]} stars, '
              f'epoch {manifest["epoch"]}, {manifest["band"]}<{manifest["magnitude_limit"]}')
        return 0

    if args.fetch:
        directory = download.ensure_available(args.fetch, progress=make_progress(args))
        print(f'{args.fetch} ready at {directory}')
        return 0

    print(download.status())
    print()
    from mee2024.starcat import providers
    print('catalogue names accepted by --catalogue: '
          + ', '.join(providers.known_catalogues()))
    return 0


def cmd_build_triangle_db(args):
    """Regenerate the triangle plate-solving database (takes several minutes)."""
    from mee2024 import platesolve_new
    platesolve_new.generate()
    print(f'triangle database written to {MEE2024util.get_triangle_db_path()}')
    return 0


# ---------------------------------------------------------------------------- parser

def _add_common(parser):
    parser.add_argument('--config', type=Path, default=None,
                        help='config file to read (default: the user config file)')
    parser.add_argument('--no-config', action='store_true',
                        help='ignore the saved config file and start from built-in defaults')
    parser.add_argument('--set', action='append', metavar='KEY=VALUE',
                        help='override any option; repeatable')


def _add_pipeline_common(parser):
    _add_common(parser)
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                        help='where to write results (default: alongside the input)')
    parser.add_argument('--no-display', action='store_true',
                        help='never open a plot window; run fully headless')
    parser.add_argument('--quiet', action='store_true', help='suppress the progress bar')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='mee2024',
        description='Modern Eddington Experiment analysis pipeline. '
                    'Run with no arguments to open the GUI.')
    parser.add_argument('--version', action='version', version=MEE2024util._version())
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('gui', help='open the graphical interface')
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser('stack', help='stage 1: stack frames, find centroids, platesolve')
    p.add_argument('lights', nargs='+', type=Path, help='light frames')
    p.add_argument('--dark', nargs='*', type=Path, help='dark frames')
    p.add_argument('--flat', nargs='*', type=Path, help='flat frames')
    _add_pipeline_common(p)
    p.set_defaults(func=cmd_stack)

    p = sub.add_parser('distortion', help='stage 2: match a catalogue and fit distortion')
    p.add_argument('data', type=Path, help='centroid_data<ts>.zip from stage 1')
    p.add_argument('--order', choices=['linear', 'cubic', 'quintic', 'septic'], default=None,
                   help='distortion polynomial order')
    p.add_argument('--catalogue', default=None, help="catalogue to match against (e.g. 'gaia')")
    p.add_argument('--fix-distortion', nargs='*', type=Path, default=None,
                   help='reference distortion file(s) whose high-order terms are held fixed')
    _add_pipeline_common(p)
    p.set_defaults(func=cmd_distortion)

    p = sub.add_parser('eclipse', help='stage 3: fit the gravitational deflection constant')
    p.add_argument('distortion', type=Path, help='distortion_data<ts>__<name>.zip from stage 2')
    p.add_argument('--limiting-mag', type=float, default=None, help='faintest star to use')
    _add_pipeline_common(p)
    p.set_defaults(func=cmd_eclipse)

    p = sub.add_parser('run', help='stages 1 and 2 back to back')
    p.add_argument('lights', nargs='+', type=Path, help='light frames')
    p.add_argument('--dark', nargs='*', type=Path, help='dark frames')
    p.add_argument('--flat', nargs='*', type=Path, help='flat frames')
    p.add_argument('--order', choices=['linear', 'cubic', 'quintic', 'septic'], default=None)
    p.add_argument('--catalogue', default=None)
    p.add_argument('--eclipse', action='store_true', help='also run stage 3')
    _add_pipeline_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('config', help='show or edit the saved configuration')
    p.add_argument('--show-path', action='store_true', help='print the config file path only')
    _add_common(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser('catalogue', help='list, verify or fetch offline star catalogues')
    p.add_argument('--verify', metavar='NAME',
                   help='check an installed catalogue against its checksums')
    p.add_argument('--fetch', metavar='NAME',
                   help='download a catalogue if it is not already installed')
    p.add_argument('--quiet', action='store_true', help='suppress the progress bar')
    _add_common(p)
    p.set_defaults(func=cmd_catalogue)

    p = sub.add_parser('build-triangle-db',
                       help='regenerate the plate-solving triangle database')
    _add_common(p)
    p.set_defaults(func=cmd_build_triangle_db)

    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, KeyError) as exc:
        # bad --set, unknown option or unknown catalogue: a user error, not a crash
        parser.error(str(exc).strip('"\''))
    except RuntimeError as exc:
        # an actionable situation, e.g. a catalogue that has to be built first
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
