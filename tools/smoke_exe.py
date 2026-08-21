"""Run a built executable through the checks that only the bundle can fail.

    python tools/smoke_exe.py dist/MEE_v1.3.9.exe
    python tools/smoke_exe.py dist/MEE_v1.3.9.exe --expect-version v1.3.9
    python tools/smoke_exe.py dist/MEE_v1.3.9.exe --lights "I:/MEE test frames/fits/00_23_49/Zenith_*.fits"

The test suite exercises the source tree. Almost every user runs the executable, and the
two are not the same program: PyInstaller decides at build time which modules exist, so a
missing hidden import fails in one subcommand of the bundle and nowhere else. `pytest` is
structurally incapable of seeing it.

`inspect_exe.py` is the companion to this and answers a different question -- it reads the
archive to say what was *bundled*. This one runs the thing.

What it would have caught, from this project's own history:

* PyInstaller 6.12 against numpy 2.5 produced an exe that built perfectly and died on the
  first import with `No module named 'numpy._core._exceptions'`.
* `mee2024 stack` and `mee2024 distortion` returned their output archive to `sys.exit()`,
  which treats a non-integer as an error message -- so every successful run reported failure
  to the shell. It shipped in v1.3.5 and survived to v1.3.8, because both the source and the
  bundle had it and nothing checked an exit code. That is the `--lights` check below, and it
  is the reason a *successful* command is asserted to exit 0 rather than merely to run.

Two checks stay manual, because a build machine cannot honestly perform them: double-clicking
into the app window, and a plate solve on a machine with no catalogue in its data directory.
See RELEASING.md.
"""

import argparse
import glob as globmod
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: every subcommand the CLI is supposed to expose. Hard-coded on purpose: reading the list
#: out of --help and then checking it against itself would pass on a truncated CLI.
SUBCOMMANDS = ('ui', 'gui', 'stack', 'distortion', 'eclipse', 'run', 'calibrate',
               'config', 'catalogue', 'build-triangle-db', 'build-pattern-db')

VERSION_RE = re.compile(r'^v\d+\.\d+\.\d+')


class Smoke:
    def __init__(self, exe, timeout=180):
        self.exe = str(exe)
        self.timeout = timeout
        self.failures = []
        self.passes = 0

    def run(self, *args, timeout=None):
        """Invoke the exe. Returns (returncode, stdout, stderr); -1 on timeout."""
        try:
            done = subprocess.run((self.exe,) + args, capture_output=True, text=True,
                                  timeout=timeout or self.timeout,
                                  errors='replace')
            return done.returncode, done.stdout or '', done.stderr or ''
        except subprocess.TimeoutExpired:
            return -1, '', f'timed out after {timeout or self.timeout}s'

    def check(self, label, ok, detail=''):
        if ok:
            self.passes += 1
            print(f'  ok    {label}')
        else:
            self.failures.append((label, detail))
            print(f'  FAIL  {label}')
            for line in (detail or '').strip().splitlines()[:6]:
                print(f'          {line}')

    # ---------------------------------------------------------------- checks

    def version(self, expect=None):
        code, out, err = self.run('--version')
        text = out.strip().splitlines()[-1].strip() if out.strip() else ''
        self.check('--version exits 0', code == 0, err or f'exit {code}')
        self.check('--version prints a version', bool(VERSION_RE.match(text)),
                   f'got {text!r}')
        if expect:
            self.check(f'--version is {expect}', text == expect,
                       f'expected {expect!r}, got {text!r}')
        return text

    def top_help(self):
        code, out, err = self.run('--help')
        self.check('--help exits 0', code == 0, err or f'exit {code}')
        missing = [s for s in SUBCOMMANDS if s not in out]
        self.check(f'--help lists all {len(SUBCOMMANDS)} subcommands', not missing,
                   f'missing: {missing}')

    def subcommand_help(self):
        """The cheap catch for a hidden import missing from one code path only.

        argparse prints help and exits before the subcommand does any work, but the module
        holding it has already been imported by then -- so a bundle that omitted one
        subcommand's dependency fails here and passes everywhere else.
        """
        for sub in SUBCOMMANDS:
            code, out, err = self.run(sub, '--help')
            self.check(f'{sub} --help exits 0', code == 0,
                       (err or f'exit {code}')[-400:])

    def read_only_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = str(Path(tmp) / 'smoke_config.txt')
            # --config points somewhere disposable so a smoke run never reads or writes
            # the settings of whoever is running it
            code, out, err = self.run('config', '--show-path', '--config', cfg)
            self.check('config --show-path exits 0', code == 0, err or f'exit {code}')
            self.check('config --show-path prints a config path',
                       'MEE_config.txt' in out, f'got {out.strip()[-200:]!r}')

            code, out, err = self.run('catalogue', '--config', cfg)
            self.check('catalogue listing exits 0', code == 0, err or f'exit {code}')
            self.check('catalogue listing names the catalogue directory',
                       'catalogue directory' in out, f'got {out.strip()[:200]!r}')

    def failure_modes(self):
        """A program that always exits 0 is as broken as one that always exits 1."""
        code, _, _ = self.run('--definitely-not-a-flag')
        self.check('an unknown flag exits non-zero', code not in (0, -1), f'exit {code}')
        code, _, _ = self.run('notasubcommand')
        self.check('an unknown subcommand exits non-zero', code not in (0, -1),
                   f'exit {code}')

    def stack_end_to_end(self, lights, dark=None):
        """The strongest check, and the only one needing data: does success exit 0?

        Opt-in because it needs real frames, and because a plate solve can fail for
        legitimate reasons (no catalogue deep enough for the field) that say nothing about
        the bundle. When it does run, it is the check that would have caught the exit-code
        bug that shipped for four releases.
        """
        matched = globmod.glob(lights)
        if len(matched) < 2:
            self.check(f'stack: {lights} matched {len(matched)} file(s)', False,
                       'need at least 2 frames; pass a glob that matches the light frames')
            return
        with tempfile.TemporaryDirectory() as tmp:
            # -o not --output, --dark not --darks; --no-display so a smoke run
            # never opens a window, --quiet to keep the output readable
            args = ['stack', lights, '-o', tmp, '--no-config', '--no-display', '--quiet']
            if dark:
                args += ['--dark', dark]
            started = time.time()
            code, out, err = self.run(*args, timeout=1800)
            took = time.time() - started
            self.check(f'stack of {len(matched)} frames exits 0 ({took:.0f}s)', code == 0,
                       (err or out)[-600:])
            produced = list(Path(tmp).rglob('STACKED*.fit'))
            self.check('stack wrote a STACKED image', bool(produced),
                       f'nothing matching STACKED*.fit under {tmp}')

    # ---------------------------------------------------------------- report

    def report(self):
        total = self.passes + len(self.failures)
        print()
        if self.failures:
            print(f'{len(self.failures)} of {total} checks FAILED:')
            for label, detail in self.failures:
                print(f'  - {label}')
            print()
            print('A bundle that fails here would have passed the test suite: pytest runs')
            print('the source tree, and these checks run the program users actually get.')
            return 1
        print(f'all {total} checks passed')
        print()
        print('Still to do by hand, and not checkable on a build machine (see RELEASING.md):')
        print('  - double-click it: the app window opens, and `gui` opens the classic one')
        print('  - a plate solve on a machine with no catalogue in its data directory')
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('exe', type=Path, help='the built executable')
    ap.add_argument('--expect-version', default=None,
                    help='fail unless --version reports exactly this, e.g. v1.3.9')
    # named --lights, not --frames: the exe's own --frames means a frame *range*
    ap.add_argument('--lights', default=None,
                    help='glob of light frames for an end-to-end stack (opt-in)')
    ap.add_argument('--dark', default=None, help='glob of darks for --lights')
    ap.add_argument('--timeout', type=int, default=180,
                    help='per-invocation timeout in seconds (default 180)')
    args = ap.parse_args()

    if not args.exe.is_file():
        print(f'no such file: {args.exe}')
        return 2

    size_mb = args.exe.stat().st_size / 1e6
    print(f'{args.exe}  ({size_mb:.0f} MB)')
    print()

    smoke = Smoke(args.exe, timeout=args.timeout)
    print('runs at all')
    smoke.version(args.expect_version)
    print('command line is intact')
    smoke.top_help()
    smoke.subcommand_help()
    print('read-only commands work')
    smoke.read_only_commands()
    print('failures are reported as failures')
    smoke.failure_modes()
    if args.lights:
        print('end to end')
        smoke.stack_end_to_end(args.lights, args.dark)

    return smoke.report()


if __name__ == '__main__':
    sys.exit(main())
