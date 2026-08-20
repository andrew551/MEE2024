"""
What actually ended up inside a release executable.

    python tools/inspect_exe.py dist/MEE_v1.3.9.exe

Running the built exe and seeing `gaia_dr3_g10 ... installed` does *not* prove the bundle
arrived: the build machine has that catalogue installed in its own data directory, which is
exactly where the runtime looks first. Only reading the archive answers the question, so
this does that, and checks the other things a bundle can silently lose:

* the compact catalogue, without which a fresh install cannot solve a plate offline;
* the single-file UI frontend and the star-label index, which live at hand-written
  destinations in the spec and so break quietly when those are wrong;
* GPU/ML stacks, which nothing here imports and which once turned a build into 2.7 GB;
* **the bundled interpreter**, which is the one users actually run.

That last one is here because RELEASING.md claimed for a while that releases were built on
Python 3.9, which was stale prose and not a record of any build machine. Working from it, a
collaborator reasonably concluded the shipped binaries carried a 3.9-only bug in the SER
timestamp parser. They do not -- but settling that meant pulling `python3XX.dll` out of the
archive by hand, when the archive names it outright. No document can be trusted about the
build interpreter; the bundle can.

Exits non-zero if any of that is wrong, so it can gate a release.
"""

import argparse
import re
import sys

from PyInstaller.archive.readers import CArchiveReader

#: nothing in this project imports these; their presence means the build environment
#: carried packages the project does not declare
HEAVY = {'torch', 'torchvision', 'torchaudio', 'tensorflow', 'cupy', 'cupyx', 'jax',
         'jaxlib', 'nvidia', 'numba', 'llvmlite', 'triton', 'transformers', 'keras'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exe', help='the built one-file executable')
    ap.add_argument('--catalogue', default='gaia_dr3_g10',
                    help='the catalogue the spec should have bundled')
    ap.add_argument('--expect-python', metavar='X.Y',
                    help='fail unless the bundled interpreter is this version, e.g. 3.12')
    args = ap.parse_args()

    names = list(CArchiveReader(args.exe).toc)
    print(f'{len(names)} entries in {args.exe}')

    problems = []

    bundled = sorted(n for n in names if args.catalogue in n)
    print(f'\nbundled catalogue: {len(bundled)} file(s)')
    for name in bundled:
        print('   ', name)
    if 'manifest.json' not in ' '.join(bundled):
        problems.append(f'{args.catalogue} is missing or has no manifest -- a fresh '
                        f'install will have to download a catalogue before it can solve')

    for label, needle in (('UI frontend', 'frontend.html'),
                          ('star-label index', 'star_labels'),
                          ('Hipparcos', 'hipparcos2'),
                          ('Tycho', 'compressed_tycho')):
        found = [n for n in names if needle in n]
        print(f'{label}: {len(found)} file(s)' if found else f'{label}: MISSING')
        if not found:
            problems.append(f'{label} is not in the archive')

    # python312.dll, not python3.dll: the stable-ABI shim carries no version number
    matches = [re.fullmatch(r'python(\d)(\d{1,2})\.dll', n, re.I) for n in names]
    versions = sorted({f'{m.group(1)}.{m.group(2)}' for m in matches if m})
    print(f'\nbundled interpreter: {", ".join(versions) if versions else "NOT FOUND"}')
    if not versions:
        problems.append('no python3XX.dll in the archive -- either this is not a frozen '
                        'CPython bundle, or PyInstaller changed how it names one')
    elif len(versions) > 1:
        problems.append(f'more than one interpreter bundled: {", ".join(versions)}')
    elif args.expect_python and versions[0] != args.expect_python:
        problems.append(f'built on Python {versions[0]}, expected {args.expect_python} '
                        f'-- check which venv produced this')

    tops = {re.split(r'[\\/]', n)[0].split('.')[0].lower() for n in names}
    heavy = sorted(tops & HEAVY)
    print(f'\nheavy GPU/ML stacks: {", ".join(heavy) if heavy else "none"}')
    if heavy:
        problems.append(f'{", ".join(heavy)} were bundled; build from the project .venv')

    if problems:
        print('\nPROBLEMS:')
        for problem in problems:
            print(f'  - {problem}')
        return 1
    print('\nthe executable carries everything it should, and nothing it should not')
    return 0


if __name__ == '__main__':
    sys.exit(main())
