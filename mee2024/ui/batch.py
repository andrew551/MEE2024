"""
Finding the fields in a folder tree, for batch processing.

Capture software writes one folder per field, usually with a session folder above and a
timestamp folder below:

    2026-07-27/H1_eclipse_altaz/22_02_22/H1_eclipse_altaz_00001.fits
    2026-07-27/Z1_base/21_27_17/Z1_base_00001.fits

so the thing to process is not the folder the user picks but every folder *below* it that
directly holds frames. This walks the tree, returns those folders, and mirrors their layout
into the output, under a folder named for the input.

The walk is the dangerous part. Pointed at a drive root it would visit every directory on
the machine and then start hundreds of runs, so it is bounded twice over -- by how many
fields it will accept and by how many directories it will look at -- and says which bound it
hit rather than quietly returning a truncated list.
"""

import datetime
import os
import re
from pathlib import Path

#: Frame extensions worth stacking. `.fit`/`.fits` cover the capture software; the image
#: formats are here because open_image falls back to cv2 for them.
FRAME_SUFFIXES = ('.fits', '.fit', '.fts', '.tif', '.tiff', '.png', '.jpg', '.jpeg')

#: How many fields a batch will accept before refusing. A night's observing is a handful of
#: fields; twenty is generous. Past this the user has almost certainly picked a folder
#: further up the tree than they meant, and starting a hundred runs is not a kind way to
#: find that out.
DEFAULT_MAX_FIELDS = 20

#: How many directories the walk will look at. Independent of the field count: a tree can
#: be enormous and still contain few frames, and the walk itself is the cost there. Set so
#: that a real session tree passes unnoticed and a drive root does not.
DEFAULT_MAX_SCANNED = 2000

#: A field needs at least this many frames. One stray frame beside a text file is more
#: likely a stray than a field, and stacking a single frame is rarely what was meant --
#: though it is allowed, since the Rasalhague example is exactly that.
MIN_FRAMES = 1


def is_frame(name):
    return Path(name).suffix.lower() in FRAME_SUFFIXES


def find_fields(root, max_fields=DEFAULT_MAX_FIELDS, max_scanned=DEFAULT_MAX_SCANNED,
                min_frames=MIN_FRAMES):
    """Folders under ``root`` that directly contain frames.

    Returns ``(fields, info)``. Each field is a dict with ``folder``, ``relative`` (its
    path relative to the root, which is what the output mirrors), ``frames`` (sorted full
    paths) and ``name``. ``info`` records what the walk did and, if it stopped early,
    which limit stopped it -- ``info['truncated']`` is a sentence to show the user.

    A root that holds frames itself counts as a field, so pointing this at a single
    capture folder behaves the way anyone would expect.
    """
    root = Path(root)
    info = {'scanned': 0, 'found': 0, 'truncated': None, 'root': str(root)}
    if not root.is_dir():
        info['truncated'] = f'{root} is not a folder'
        return [], info

    fields = []
    # sorted walk so the order matches what the user sees in a file manager, and so a
    # rerun processes the same tree in the same order
    for current, subdirs, files in os.walk(root):
        subdirs.sort()
        info['scanned'] += 1
        frames = sorted(f for f in files if is_frame(f))
        if len(frames) >= min_frames:
            folder = Path(current)
            fields.append({
                'folder': str(folder),
                'relative': '' if folder == root else str(folder.relative_to(root)),
                'name': folder.name or str(folder),
                'frames': [str(folder / f) for f in frames],
            })
            # a folder of frames is a field, not a container of fields
            subdirs[:] = []
        if len(fields) > max_fields:
            info['truncated'] = (
                f'more than {max_fields} folders of frames under {root}. That is usually a '
                f'sign of a folder higher up the tree than intended, so nothing has been '
                f'started. Pick a folder closer to the data, or raise the limit.')
            return [], info
        if info['scanned'] >= max_scanned:
            info['truncated'] = (
                f'stopped after looking at {max_scanned} folders under {root} without '
                f'finishing. Pick a folder closer to the data: this one has more '
                f'subfolders than a session tree should.')
            return [], info

    info['found'] = len(fields)
    if not fields:
        info['truncated'] = (f'no image frames anywhere under {root} '
                             f'({info["scanned"]} folder(s) looked at)')
    return fields, info


def output_dir_for(field, output_root):
    """Where a field's results go: the input layout, mirrored under the output root.

    Mirroring rather than flattening keeps the association between a field and its results
    obvious once there are twenty of them, and keeps two fields that a capture program named
    the same thing on different nights apart.
    """
    base = Path(output_root)
    return str(base / field['relative']) if field['relative'] else str(base)


#: The files a run writes at its own level rather than per field. Their names are fixed,
#: so two runs sharing one folder cannot both keep theirs.
RUN_RECORDS = ('batch_summary.csv', 'batch_summary.json', 'activity.jsonl')

#: Characters a folder name may not carry on Windows. A source folder name has already
#: passed the filesystem once, so this only ever matters for the odd separator.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Longest folder name taken from an input path. The field tree, the archive names and a
#: `DISTORTION_OUTPUT<ts>__...` working folder all sit below this, and Windows stops at 260
#: characters for the lot.
MAX_LABEL = 64


def source_label(source):
    """A folder name for the output, taken from the input path. '' if there is none."""
    if not source:
        return ''
    path = Path(str(source))
    name = path.name or path.drive.strip(':\\/ ')
    return _UNSAFE.sub('_', name).strip('. ')[:MAX_LABEL]


def run_output_root(output_root, source, stamp=None):
    """Where one run's results go: a subfolder of the chosen folder, named for the input.

    Writing straight into the chosen folder was wrong twice over. Nothing in an output
    folder said which input produced it, so the user had to make and name one by hand for
    every field set. And the run-level files -- the batch summary and the activity log --
    have fixed names, so a second run pointed at the same folder silently destroyed the
    first run's records: the per-field archives survived, being timestamped, but the
    account of what happened did not.

    A timestamp is appended **only** when a previous run's records are already sitting in
    the target. Appending one always would lengthen every path for a case that usually
    does not arise, and these paths are already close to the Windows limit.
    """
    root = Path(output_root)
    name = source_label(source)
    if not name:
        return str(root)
    target = root / name
    if not any((target / record).exists() for record in RUN_RECORDS):
        return str(target)
    stamp = stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(root / f'{name}_{stamp}')


def describe(fields, info):
    """A one-line summary for the log."""
    if info.get('truncated'):
        return info['truncated']
    total = sum(len(f['frames']) for f in fields)
    return (f'{len(fields)} field(s), {total} frame(s), from '
            f'{info["scanned"]} folder(s) under {info["root"]}')
