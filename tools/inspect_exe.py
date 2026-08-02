"""
What actually ended up inside a release executable.

    python tools/inspect_exe.py dist/MEE_2024_v1.2.2.exe

Running the built exe and seeing `gaia_dr3_g10 ... installed` does *not* prove the bundle
arrived: the build machine has that catalogue installed in its own data directory, which is
exactly where the runtime looks first. Only reading the archive answers the question, so
this does that, and checks the other things a bundle can silently lose:

* the compact catalogue, without which a fresh install cannot solve a plate offline;
* the single-file UI frontend and the star-label index, which live at hand-written
  destinations in the spec and so break quietly when those are wrong;
* GPU/ML stacks, which nothing here imports and which once turned a build into 2.7 GB.

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
