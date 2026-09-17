"""Collapse byte-identical stage-1 stacks in the output tree onto hard links.

WHY THEY EXIST. Stage 1 writes the stack before it centroids it --
`stacker_implementation.py` writes `STACKED<stamp>.fit` and `STACKED_FLOAT<stamp>.fit`, and
only then calls `get_centroids_blur` on it. Every A/B study in this project varies an option
consumed at or after that point: `centroid_refine_window` (moments vs windowed),
`background_subtraction_mode` (annular vs Gaussian), the corona/moments variants. So each arm
re-runs the whole of stage 1 and writes a bit-for-bit identical stack into its own folder. Two
axes gives four copies. Measured 2026-09-17: 97 sets, 13.5 GB on a 228 GB tree, of which 80 %
was this one mechanism; the rest was one reference field reduced independently by four
campaigns, plus a few probe runs later repeated for real.

WHY HARD LINKS RATHER THAN DELETION. Every tool globs a stack inside its own run folder
(`<run>/CENTROID_OUTPUT*/STACKED*.fit`), so a path has to keep existing. A hard link is the
same file under a second name: the glob finds it, the bytes are the same, and removing one
name later leaves the others intact. The one behaviour that changes is that an IN-PLACE write
through one name would be seen through all of them -- stage 1 never does that, since every run
creates a new timestamped `CENTROID_OUTPUT` directory rather than overwriting an existing
stack.

WHY THE FULL HASH MATTERS. A size-plus-both-edges fingerprint is not sufficient on this data.
On the 2026-09-17 run it proposed 99 sets and the full hash rejected 7 files from them: stacks
of the same field that agree on their first and last 4 MB and differ in between. Linking those
would have silently replaced one reduction's output with another's. The fingerprint is only
used to decide what is worth reading in full.

    .venv/Scripts/python.exe tools/dedupe_stacks.py               # report, change nothing
    .venv/Scripts/python.exe tools/dedupe_stacks.py --apply       # link, then verify
"""
import argparse
import collections
import hashlib
import os
import shutil
import time

DEFAULT_ROOT = r'F:\MEE_output'
DEFAULT_MIN_MB = 20
EDGE = 4 * 1024 * 1024


def full_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path, size):
    """size + first and last 4 MB -- a filter, never the decision (see the module docstring)."""
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, 'rb') as f:
        h.update(f.read(EDGE))
        if size > 2 * EDGE:
            f.seek(-EDGE, os.SEEK_END)
            h.update(f.read(EDGE))
    return h.hexdigest()


def candidate_sets(root, min_bytes):
    by_size = collections.defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size >= min_bytes:
                by_size[size].append(path)

    groups = collections.defaultdict(list)
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            try:
                groups[fingerprint(path, size)].append(path)
            except OSError:
                continue
    return [paths for paths in groups.values() if len(paths) > 1]


def confirm(sets):
    """Split each candidate set by FULL hash. Returns (confirmed sets, files rejected)."""
    confirmed, rejected, read = [], 0, 0
    for paths in sets:
        by_hash = collections.defaultdict(list)
        for path in paths:
            try:
                by_hash[full_hash(path)].append(path)
                read += os.path.getsize(path)
            except OSError:
                continue
        for members in by_hash.values():
            if len(members) > 1:
                confirmed.append(members)
            else:
                rejected += 1
    return confirmed, rejected, read


def plan(confirmed):
    todo, already, waste = [], 0, 0
    for members in confirmed:
        canonical = members[0]
        canonical_ino = os.stat(canonical).st_ino
        for duplicate in members[1:]:
            st = os.stat(duplicate)
            if st.st_ino == canonical_ino:
                already += 1
                continue
            todo.append((canonical, duplicate))
            waste += st.st_size
    return todo, already, waste


def link_one(canonical, duplicate):
    """Link beside the duplicate, then replace it. os.replace is atomic on Windows, so the
    path is never absent -- it is the old file or the new link, never nothing."""
    tmp = duplicate + '.hardlink-tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        os.link(canonical, tmp)
        os.replace(tmp, duplicate)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    a, b = os.stat(canonical), os.stat(duplicate)
    assert a.st_ino == b.st_ino and b.st_nlink > 1, duplicate


def verify(confirmed):
    checked = bad = unlinked = 0
    for members in confirmed:
        want = full_hash(members[0])
        base = os.stat(members[0]).st_ino
        for path in members:
            checked += 1
            if full_hash(path) != want:
                bad += 1
                print('  CONTENT CHANGED: %s' % path)
            elif os.stat(path).st_ino != base:
                unlinked += 1
    return checked, bad, unlinked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--min-mb', type=int, default=DEFAULT_MIN_MB)
    ap.add_argument('--apply', action='store_true', help='make the links (default: report only)')
    args = ap.parse_args()

    started = time.time()
    sets = candidate_sets(args.root, args.min_mb * 1024 * 1024)
    print('candidate duplicate sets: %d  (%.0f s)' % (len(sets), time.time() - started))

    confirmed, rejected, read = confirm(sets)
    print('confirmed by full hash  : %d sets  (read %.1f GB, %.0f s)'
          % (len(confirmed), read / 1e9, time.time() - started))
    if rejected:
        print('rejected by full hash   : %d file(s) that matched only on size and both edges'
              % rejected)

    todo, already, waste = plan(confirmed)
    print('links to make: %d   already linked: %d   reclaimable: %.2f GB'
          % (len(todo), already, waste / 1e9))
    if not args.apply:
        print('\nreport only -- nothing changed. Re-run with --apply.')
        return

    before = shutil.disk_usage(args.root).free
    made = failed = 0
    for canonical, duplicate in todo:
        try:
            link_one(canonical, duplicate)
            made += 1
        except Exception as exc:                                # noqa: BLE001
            failed += 1
            print('FAILED %s (%s)' % (duplicate, exc))
    after = shutil.disk_usage(args.root).free
    print('\nlinked %d, failed %d' % (made, failed))
    print('free space %.1f -> %.1f GB  (+%.2f GB)'
          % (before / 1e9, after / 1e9, (after - before) / 1e9))

    print('\nVERIFY')
    checked, bad, unlinked = verify(confirmed)
    print('  %d files re-hashed, %d content mismatches, %d still separate copies'
          % (checked, bad, unlinked))
    print('  total %.0f s' % (time.time() - started))


if __name__ == '__main__':
    main()
