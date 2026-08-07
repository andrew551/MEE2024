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
import contextlib
import glob
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


_GLOB_CHARS = '*?['


def resolve_input_files(entries, kind, *, required=False):
    """Expand the frame paths from the command line, and refuse a list we cannot stack.

    Two jobs. First, expand wildcards ourselves: cmd.exe does no globbing at all and
    PowerShell only globs for cmdlets, so on Windows ``stack lights/*.fits`` arrives
    here as the unexpanded pattern and there is no file by that name to open.

    Second, fail here rather than deep inside the pipeline. A path that matches nothing
    reaches ``cv2.imread``, which reports every failure by returning None, and the run
    dies in ``cvtColor`` with an assertion about an empty matrix that names neither the
    file nor the reason -- after the pattern database has been prepared, which is minutes.

    Unlike ``main.precheck_files``, which drops bad frames and stacks the rest because a
    GUI user can see the list they picked, this refuses the whole run: a headless run has
    nobody watching, and silently stacking a subset is a wrong answer rather than an error.
    """
    resolved = []
    for entry in entries:
        raw = str(entry)
        if not raw:
            raise RuntimeError(f'{kind} path is empty')
        # a plain path globs to itself when it exists, and to nothing when it does not,
        # so patterns and ordinary paths take the same route through here
        matches = sorted(glob.glob(raw))
        if not matches and Path(raw).exists():
            matches = [raw]  # a real name that happens to contain * ? or [
        files = [match for match in matches if Path(match).is_file()]
        if not files:
            if any(char in raw for char in _GLOB_CHARS):
                reason = 'matched only directories' if matches else 'matched no files'
                raise RuntimeError(f'{kind} pattern {reason}: {raw}')
            if matches:
                raise RuntimeError(f'{kind} is a directory, not a file: {raw}')
            raise RuntimeError(f'no such {kind}: {raw}')
        resolved.extend(files)
    if required and not resolved:
        raise RuntimeError(f'no {kind}s given')
    return resolved


@contextlib.contextmanager
def event_bus(args):
    """An ambient event bus for the run, wired to whatever the flags asked for."""
    from mee2024 import events

    sinks = []
    if getattr(args, 'events_jsonl', None):
        sinks.append(events.JsonlSink(args.events_jsonl))
    if getattr(args, 'events_text', False):
        sinks.append(events.TextSink())
    if not sinks:
        yield None
        return
    bus = events.EventBus(sinks)
    try:
        with events.using(bus):
            yield bus
    finally:
        bus.close()


def _use_headless_backend(options):
    """Select a non-interactive matplotlib backend when nothing will be shown."""
    if not (options['flag_display'] or options['flag_display2'] or options['flag_display3']):
        import matplotlib
        matplotlib.use('Agg')


def _prepare_catalogue(args, options):
    """Fetch a missing offline catalogue and warn about depth, before stage 2 runs."""
    from mee2024.starcat import download

    for warning in download.prepare_catalogue(
            options.get('catalogue') or 'gaia', options=options,
            allow_download=options.get('auto_download_catalogue', True),
            on_note=print, progress_for=lambda name: make_progress(args)):
        print(f'warning: {warning}')
    _prepare_pattern_db(args, options)


def _prepare_pattern_db(args, options):
    """Build the plate-solving database if it is missing: derived data, not a download."""
    if options.get('platesolver', 'v2') != 'v2':
        return
    from mee2024 import platesolve2
    platesolve2.ensure_pattern_db(options, on_note=print,
                                  progress=make_progress(args))


# --------------------------------------------------------------------------- commands

def cmd_stack(args):
    options = resolve_options(args)
    # before the pattern database is prepared: checking the inputs is instant, and
    # preparing the database is not
    lights = resolve_input_files(args.lights, 'light frame', required=True)
    darks = resolve_input_files(args.dark or [], 'dark frame')
    flats = resolve_input_files(args.flat or [], 'flat frame')
    _use_headless_backend(options)
    # stage 1 plate-solves too, so the solver's database must exist before it starts
    _prepare_pattern_db(args, options)
    from mee2024 import database_cache, stacker_implementation
    try:
        with event_bus(args):
            zip_path = stacker_implementation.do_stack(lights, darks, flats, options,
                                                       progress=make_progress(args))
    finally:
        database_cache.shutdown_triangles()
    print(f'stage 1 output: {zip_path}')
    return zip_path


def cmd_distortion(args):
    options = resolve_options(args)
    _use_headless_backend(options)
    _prepare_catalogue(args, options)
    from mee2024 import database_cache, distortion_fitter
    try:
        with event_bus(args):
            zip_path = distortion_fitter.match_and_fit_distortion(str(args.data), options, None)
    finally:
        database_cache.shutdown_triangles()
    print(f'stage 2 output: {zip_path}')
    return zip_path


def cmd_eclipse(args):
    options = resolve_options(args)
    _use_headless_backend(options)
    from mee2024 import eclipse_analysis
    with event_bus(args):
        eclipse_analysis.eclipse_analysis(str(args.distortion), options)
    print('stage 3 complete')
    return 0


def cmd_run(args):
    """Stage 1 then stage 2, and stage 3 too if --eclipse was given."""
    options = resolve_options(args)
    lights = resolve_input_files(args.lights, 'light frame', required=True)
    darks = resolve_input_files(args.dark or [], 'dark frame')
    flats = resolve_input_files(args.flat or [], 'flat frame')
    _use_headless_backend(options)
    # before stage 1, so a missing catalogue is not discovered after minutes of stacking
    _prepare_catalogue(args, options)
    from mee2024 import database_cache, distortion_fitter, stacker_implementation
    try:
        progress = make_progress(args)
        with event_bus(args):
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
    """The classic FreeSimpleGUI interface, kept as the legacy mode."""
    from mee2024 import main as main_module
    main_module.run_gui()
    return 0


def cmd_ui(args):
    """The app window: a native web view when available, otherwise the browser."""
    from mee2024.ui import app
    app.launch(prefer_browser=args.browser, port=args.port,
               keep_alive=args.keep_alive)
    return 0


def _confirm_large_download(release):
    """Ask, at a terminal, before starting a multi-gigabyte download.

    Declines rather than blocks when there is no terminal to ask at -- a scripted or
    headless run must not hang on a prompt nobody can see. ``--yes`` is the way through
    in that case, which keeps the decision the user's either way.
    """
    print()
    print(release.size_warning())
    try:
        answer = input(f'Download {release.name} anyway? [y/N] ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        # no one there to answer -- a scripted run must decline, not block. The isatty
        # check this replaces is unreliable here: on Windows the null device reports as
        # a character device, so `< /dev/null` still looks interactive
        print(f'\nnot running interactively, so not starting a '
              f'{release.size_bytes / 1e9:.2f} GB download. Re-run with --yes to agree '
              f'to it in advance.')
        return False
    return answer in ('y', 'yes')


def cmd_catalogue(args):
    """Inspect, verify, fetch, pack or install the offline star catalogues."""
    from mee2024.MEE2024util import get_catalogue_root
    from mee2024.starcat import download, store

    if args.pack:
        directory = _resolve_catalogue_dir(args.pack)
        if directory is None:
            print(f'{args.pack} is not installed; nothing to pack')
            return 1
        destination = Path(args.out) if args.out else Path.cwd() / f'{args.pack}.zip'
        print(f'packing {directory} -> {destination}')
        archive = store.pack(directory, destination)
        size = archive.stat().st_size
        print(f'wrote {archive} ({size / 1e6:.0f} MB)')
        print(f'sha256 {store.sha256(archive)}')
        print('\nCopy it to the other machine and run:')
        print(f'    mee2024 catalogue --install {archive.name}')
        return 0

    if args.merge:
        name = args.merge if isinstance(args.merge, str) else download.DEFAULT_RELEASE
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        sources = args.from_ or None
        print(f'merging installed archives into {name} '
              f'(this reads and rewrites every star, and recomputes neighbour flags)')
        directory, manifest = download.merge_installed(
            name, sources=sources, options=options, force=args.force,
            on_note=lambda t: print(f'  {t}'))
        print(f"\n{name}: {manifest['n_stars']} stars to G<"
              f"{manifest['magnitude_limit']}, written to {directory}")
        print('The parts it was merged from are still installed; remove them with '
              '`mee2024 catalogue --remove NAME` once you have verified this one.')
        return 0

    if args.repair:
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        download.repair_catalogue(args.repair, options=options,
                                  on_note=lambda t: print(f'  {t}'))
        return 0

    if args.remove:
        directory = _resolve_catalogue_dir(args.remove)
        if directory is None:
            print(f'{args.remove} is not installed; nothing to remove')
            return 1
        import shutil
        try:
            shutil.rmtree(directory)
        except PermissionError as exc:
            # a running MEE2024 holds the catalogue memory-mapped, and Windows will
            # not delete a mapped file. Say which program to close rather than
            # showing a traceback about one .npy file.
            print(f'cannot remove {directory}: {exc.strerror or exc}.\n'
                  f'Close any running MEE2024 window (it keeps the catalogue open) '
                  f'and try again.', file=sys.stderr)
            return 1
        print(f'removed {directory}')
        return 0

    if args.set_source:
        if not args.url:
            print('--set-source needs --url', file=sys.stderr)
            return 1
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        sources = dict(options.get('catalogue_sources') or {})
        entry = {'url': args.url}
        if args.sha256:
            entry['sha256'] = args.sha256
        sources[args.set_source] = entry
        options['catalogue_sources'] = sources
        MEE2024util.write_ini(options, path=args.config)
        print(f'{args.set_source} will be downloaded from {args.url}')
        if not args.sha256:
            print('note: no --sha256 given, so the download cannot be verified. '
                  'Get one with `mee2024 catalogue --pack NAME` on the source machine.')
        return 0

    if args.install:
        source = Path(args.install)
        if not source.exists():
            print(f'no such archive: {source}')
            return 1
        name = args.name or source.stem
        directory = get_catalogue_root() / name
        if directory.exists() and not args.force:
            print(f'{directory} already exists; pass --force to replace it')
            return 1
        print(f'installing {source} -> {directory}')
        try:
            store.unpack(source, directory, verify_checksums=True)
        except Exception as exc:
            # a failed transfer must not leave a half-installed catalogue that later
            # looks usable; remove it and say plainly what went wrong
            import shutil as _shutil
            _shutil.rmtree(directory, ignore_errors=True)
            print(f'install failed, nothing was kept: {exc}', file=sys.stderr)
            print('the archive is corrupt or truncated -- copy it again', file=sys.stderr)
            return 1
        manifest = store.read_manifest(directory)
        print(f'installed {name}: {manifest["n_stars"]} stars, epoch {manifest["epoch"]}, '
              f'{manifest["band"]}<{manifest["magnitude_limit"]} -- checksums verified')
        return 0

    if args.check_remote:
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        failures = 0
        for result in download.check_remote(options=options):
            state = 'OK  ' if result['ok'] else (
                'skip' if result.get('skipped') else 'FAIL')
            print(f"[{state}] {result['name']}")
            print(f"       {result['url']}")
            print(f"       {result['detail']}")
            failures += 0 if (result['ok'] or result.get('skipped')) else 1
        if failures:
            print('\nSee RELEASING.md for the exact tag and asset names expected.')
        else:
            print('\nAll catalogue assets are reachable; '
                  '`mee2024 catalogue --fetch NAME` will work.')
        return 1 if failures else 0

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
        options = get_default_options()
        MEE2024util.read_ini(options, path=args.config)
        if args.url:      # a one-off URL, without saving it to the config
            options.setdefault('catalogue_sources', {})[args.fetch] = {
                'url': args.url, 'sha256': args.sha256}
        try:
            directory = download.ensure_available(
                args.fetch, progress=make_progress(args), options=options,
                confirm=args.yes or _confirm_large_download)
        except download.ConfirmationRequired:
            # _confirm_large_download has already shown the size and asked; repeating
            # the whole paragraph here just buries the one line that matters
            print('cancelled: nothing was downloaded.')
            return 1
        print(f'{args.fetch} ready at {directory}')
        return 0

    print(download.status())
    locally_built = sorted(
        d.name for d in get_catalogue_root().iterdir()
        if d.is_dir() and not d.name.startswith('.') and _resolve_catalogue_dir(d.name))
    if locally_built:
        print('\nlocally built catalogues:')
        for name in locally_built:
            manifest = store.read_manifest(get_catalogue_root() / name)
            print(f'  {name}: {manifest["n_stars"]} stars, '
                  f'{manifest["band"]}<{manifest["magnitude_limit"]}')
    print()
    from mee2024.starcat import providers
    print('catalogue names accepted by --catalogue: '
          + ', '.join(providers.known_catalogues()) + ', or any name listed above')
    return 0


def _resolve_catalogue_dir(name):
    """The directory of an installed catalogue, whether registered or locally built."""
    from mee2024 import database_cache
    return database_cache._installed_catalogue_dir(name)


def cmd_build_triangle_db(args):
    """Regenerate the triangle plate-solving database (takes several minutes)."""
    from mee2024 import platesolve_new
    platesolve_new.generate()
    print(f'triangle database written to {MEE2024util.get_triangle_db_path()}')
    return 0


def cmd_build_pattern_db(args):
    """Build a v2 pattern database from the installed offline Gaia catalogue."""
    from mee2024.platesolve2 import build
    from mee2024.progress import NullProgress, TextProgress

    params = {}
    if args.theta_pat is not None:
        params['theta_pat_deg'] = args.theta_pat
    if args.theta_sep is not None:
        params['theta_sep_deg'] = args.theta_sep
    if args.depth is not None:
        params['d'] = args.depth
    if args.anchors is not None:
        params['a'], params['b'] = args.anchors
    if args.invariant is not None:
        params['invariant'] = args.invariant
    if args.tolerance is not None:
        params['tolerance'] = args.tolerance
    if args.dedupe is not None:
        params['dedupe_rule'] = args.dedupe
    progress = NullProgress() if args.quiet else TextProgress()
    out_dir, manifest = build.build_from_catalogue(
        name=args.name, catalogue_names=tuple(args.catalogue), params=params,
        progress=progress)
    print(f"pattern database {manifest['name']} written to {out_dir}: "
          f"{manifest['n_anchors']} anchors, {manifest['n_triangles']} triangles")
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
    parser.add_argument('--events-jsonl', type=Path, default=None, metavar='PATH',
                        help='write a machine-readable JSONL record of the run')
    parser.add_argument('--events-text', action='store_true',
                        help='print pipeline events to stderr as they happen')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='mee2024',
        description='Modern Eddington Experiment analysis pipeline. '
                    'Run with no arguments to open the GUI.')
    parser.add_argument('--version', action='version', version=MEE2024util._version())
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('ui', help='open the app window (new interface)')
    p.add_argument('--browser', action='store_true',
                   help='use the default browser instead of a native window')
    p.add_argument('--port', type=int, default=0,
                   help='serve on a fixed port instead of an ephemeral one')
    p.add_argument('--keep-alive', action='store_true',
                   help='keep serving after the page closes (for a browser on '
                        'another machine, or reloading the tab while developing)')
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser('gui', help='open the classic interface (legacy)')
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser('stack', help='stage 1: stack frames, find centroids, platesolve')
    p.add_argument('lights', nargs='+', type=Path,
                   help='light frames; wildcards are expanded here if the shell did not')
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
    p.add_argument('lights', nargs='+', type=Path,
                   help='light frames; wildcards are expanded here if the shell did not')
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

    p = sub.add_parser('catalogue',
                       help='list, verify, fetch, pack or install offline star catalogues')
    p.add_argument('--verify', metavar='NAME',
                   help='check an installed catalogue against its checksums')
    p.add_argument('--fetch', metavar='NAME',
                   help='download a catalogue if it is not already installed')
    p.add_argument('--pack', metavar='NAME',
                   help='zip an installed catalogue for copying to another machine')
    p.add_argument('--install', metavar='ARCHIVE',
                   help='install a catalogue from a zip made by --pack')
    p.add_argument('--set-source', metavar='NAME',
                   help='record where a catalogue can be downloaded from')
    p.add_argument('--merge', nargs='?', const=True, metavar='NAME',
                   help='merge the installed archives into one (default: '
                        'gaia_dr3_g13), recomputing double-star neighbour flags')
    p.add_argument('--from', dest='from_', nargs='+', metavar='NAME',
                   help='which installed archives --merge should read')
    p.add_argument('--repair', metavar='NAME',
                   help='rebuild a lost manifest for an archive whose install was '
                        'interrupted (validates the data first)')
    p.add_argument('--remove', metavar='NAME',
                   help='delete an installed catalogue directory')
    p.add_argument('--url', help='download URL, for --set-source or a one-off --fetch')
    p.add_argument('--sha256', help='expected checksum of the downloaded archive')
    p.add_argument('--out', metavar='FILE', help='where --pack writes its archive')
    p.add_argument('--name', metavar='NAME',
                   help='catalogue name for --install (default: the archive filename)')
    p.add_argument('--force', action='store_true',
                   help='let --install replace an existing catalogue, or let --merge '
                        'proceed with an unreadable or faint-only source')
    p.add_argument('--check-remote', action='store_true',
                   help='check the published archives are reachable, without downloading them')
    p.add_argument('--yes', '-y', action='store_true',
                   help='agree in advance to a large download, instead of being asked')
    p.add_argument('--quiet', action='store_true', help='suppress the progress bar')
    _add_common(p)
    p.set_defaults(func=cmd_catalogue)

    p = sub.add_parser('build-triangle-db',
                       help='regenerate the plate-solving triangle database')
    _add_common(p)
    p.set_defaults(func=cmd_build_triangle_db)

    p = sub.add_parser('build-pattern-db',
                       help='build a v2 pattern database from the offline catalogue')
    p.add_argument('--name', default='patdb_g12_t17',
                   help='variant name (default: patdb_g12_t17)')
    p.add_argument('--catalogue', nargs='+', default=['gaia_dr3_g12'],
                   help='installed offline catalogue(s) to build from')
    p.add_argument('--theta-pat', type=float, default=None, metavar='DEG',
                   help='pattern disc radius in degrees (default 1.7)')
    p.add_argument('--theta-sep', type=float, default=None, metavar='DEG',
                   help='anchor isolation radius in degrees (default 0.4)')
    p.add_argument('--depth', type=int, default=None, metavar='N',
                   help='star-list depth (default 700000)')
    p.add_argument('--anchors', type=int, nargs=2, default=None, metavar=('A', 'B'),
                   help='unconditional and gap-fill anchor counts '
                        '(default 80000 160000)')
    p.add_argument('--invariant', choices=['ratio_dphi', 'kendall'], default=None,
                   help='triangle invariant (default ratio_dphi)')
    p.add_argument('--tolerance', type=float, default=None,
                   help='calibrated invariant-space match radius, recorded in the '
                        'manifest for the solver to use')
    p.add_argument('--dedupe', choices=['none', 'dimmer_legs'], default=None,
                   help='dimmer_legs stores each star triple once, under its '
                        'brightest member (more distinct triples per byte)')
    p.add_argument('--quiet', action='store_true', help='suppress the progress bar')
    p.set_defaults(func=cmd_build_pattern_db)

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
