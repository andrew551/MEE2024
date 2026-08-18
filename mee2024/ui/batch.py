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
#: formats are here because open_image falls back to cv2 for them; `.ser` is a *container*
#: of many frames rather than one frame, and is expanded by `frames_of`.
FRAME_SUFFIXES = ('.fits', '.fit', '.fts', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.ser')

#: Extensions that hold many frames in one file. One of these *is* a field on its own -- a
#: 22 GB capture is not "one frame beside some others" -- so a folder holding one is a field
#: even under the usual minimum, and the frame list comes from inside the container.
CONTAINER_SUFFIXES = ('.ser',)

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

#: How far a folder's own frames must be outnumbered by the frames beneath it before it is
#: treated as a container rather than a field.
#:
#: A folder of frames is normally a field, and the walk stops there -- otherwise a
#: `thumbnails` subfolder inside a capture folder would be processed as a second field. But
#: a *session root* can hold something stray: the Leon root keeps one `.JPG` of the site
#: beside 1251 frames of data in `DARKS`, `Zenith`, `Horizon` and the rest, and treating
#: that photograph as the field pruned the whole tree, so a batch aimed at the session
#: found one field and processed none of the session.
#:
#: Ten to one separates the two cleanly and needs no knowledge of file types or naming: a
#: capture folder of 30 frames keeps its 5-frame thumbnails subfolder out, while 1 frame
#: against 1251 is plainly not the field.
CONTAINER_RATIO = 10


def is_frame(name):
    return Path(name).suffix.lower() in FRAME_SUFFIXES


def is_container(name):
    """Does this file hold many frames rather than one?"""
    return Path(name).suffix.lower() in CONTAINER_SUFFIXES


def frames_of(folder, names):
    """The frame list for a folder: paths for ordinary images, references for containers.

    A `.ser` file becomes one reference per frame (`capture.ser#0`, `#1`, ...), so
    everything downstream keeps working on a list of frames without knowing the difference.
    A container that cannot be read is left as a plain path, and fails later with a message
    about that file rather than being silently dropped here.
    """
    folder = Path(folder)
    out = []
    for name in names:
        path = folder / name
        if not is_container(name):
            out.append(str(path))
            continue
        try:
            from mee2024 import ser
            out.extend(ser.expand(path))
        except Exception:
            out.append(str(path))
    return out


def _is_calibration(folder, root, frames):
    """Is this leaf a folder of darks or flats rather than a field?

    Kept here as a thin wrapper so the walk does not import the calibration module unless
    it has a leaf to judge, and so the rule itself lives in one place. Name first, then
    ``OBJECT`` from a single frame -- never ``IMAGETYP``, which reads 'Light' on every
    scripted dark and flat because the capture software's sequencer cannot set it.
    """
    from mee2024 import calibration

    if calibration.looks_like_calibration(folder, root):
        return True
    return calibration.classify_frames([str(folder / f) for f in frames[:1]]) is not None


def find_fields(root, max_fields=DEFAULT_MAX_FIELDS, max_scanned=DEFAULT_MAX_SCANNED,
                min_frames=MIN_FRAMES, skip_calibration=True):
    """Folders under ``root`` that directly contain frames.

    Returns ``(fields, info)``. Each field is a dict with ``folder``, ``relative`` (its
    path relative to the root, which is what the output mirrors), ``frames`` (sorted full
    paths) and ``name``. ``info`` records what the walk did and, if it stopped early,
    which limit stopped it -- ``info['truncated']`` is a sentence to show the user.

    A root that holds frames itself counts as a field, so pointing this at a single
    capture folder behaves the way anyone would expect, and a folder of frames is a field
    rather than a container of fields -- a `thumbnails` subfolder inside a capture folder
    must not become a second field.

    The exception is a folder whose own frames are swamped by what lies beneath it. The
    Leon session root keeps one `.JPG` of the site beside `DARKS`, `Zenith`, `Horizon` and
    1251 frames of data; claiming it as a field pruned the entire tree, so a batch pointed
    at that root found one "field" holding one photograph and processed none of the
    session. :data:`CONTAINER_RATIO` decides: outnumbered that heavily, a folder is a
    container with something stray in it, and the walk goes on down.

    ``skip_calibration`` leaves out folders of darks, flats or bias frames, which are
    listed in ``info['calibration']``. Without it every tier of a dark library under the
    session root was stacked and plate-solved as though it were a field, and failed -- a
    capped-lens frame has no stars in it. A real session tree keeps them there: the Leon
    campaign has eight dark tiers and a flat set beside two nights of science. Pass False
    when the calibration folders are what you are looking for.
    """
    root = Path(root)
    info = {'scanned': 0, 'found': 0, 'truncated': None, 'root': str(root),
            'calibration': []}
    if not root.is_dir():
        info['truncated'] = f'{root} is not a folder'
        return [], info

    candidates = []
    # sorted walk so the order matches what the user sees in a file manager, and so a
    # rerun processes the same tree in the same order
    for current, subdirs, files in os.walk(root):
        subdirs.sort()
        info['scanned'] += 1
        frames = sorted(f for f in files if is_frame(f))
        if len(frames) >= min_frames:
            candidates.append((Path(current), frames))
        # the field count is checked against the candidates that survive the container
        # test below -- a container folder with a stray frame in it must not count
        # towards the limit it would otherwise trip
        if info['scanned'] >= max_scanned:
            info['truncated'] = (
                f'stopped after looking at {max_scanned} folders under {root} without '
                f'finishing. Pick a folder closer to the data: this one has more '
                f'subfolders than a session tree should.')
            return [], info

    # shallowest first, so a container is judged before the folders it contains
    candidates.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    below = {}
    for folder, frames in candidates:
        for parent in folder.parents:
            below[parent] = below.get(parent, 0) + len(frames)

    fields, claimed = [], []
    for folder, frames in candidates:
        if any(folder == owner or owner in folder.parents for owner in claimed):
            continue                     # a field already claimed this branch
        if below.get(folder, 0) >= CONTAINER_RATIO * len(frames):
            continue                     # a container with something stray in it
        claimed.append(folder)
        if skip_calibration and _is_calibration(folder, root, frames):
            info['calibration'].append(str(folder))
            continue
        fields.append({
            'folder': str(folder),
            'relative': '' if folder == root else str(folder.relative_to(root)),
            'name': folder.name or str(folder),
            'frames': frames_of(folder, frames),
        })
    fields.sort(key=lambda f: f['relative'])

    if len(fields) > max_fields:
        info['truncated'] = (
            f'more than {max_fields} folders of frames under {root}. That is usually a '
            f'sign of a folder higher up the tree than intended, so nothing has been '
            f'started. Pick a folder closer to the data, or raise the limit.')
        return [], info

    info['found'] = len(fields)
    if not fields:
        info['truncated'] = (f'no image frames anywhere under {root} '
                             f'({info["scanned"]} folder(s) looked at)')
    return fields, info


def find_fields_in(roots, max_fields=DEFAULT_MAX_FIELDS, max_scanned=DEFAULT_MAX_SCANNED,
                   min_frames=MIN_FRAMES, skip_calibration=True):
    """:func:`find_fields` over several roots at once.

    Reducing an arbitrary *subset* -- two fields out of eighteen, after a rerun -- used to
    mean one run each, because a batch took exactly one root and processed everything
    beneath it. Ctrl-clicking several folders is the ordinary way to say what you mean.

    ``relative`` is taken from each field's **own** root, and the roots' shared parent
    becomes the run's name, so mirroring the input layout still works when the selection
    spans folders that have no parent in common (different drives, say) -- in which case
    there is no shared parent and each field keeps its own root's name at the front.
    """
    roots = [Path(r) for r in (roots if isinstance(roots, (list, tuple)) else [roots]) if r]
    if not roots:
        return [], {'scanned': 0, 'found': 0, 'truncated': 'no folder chosen',
                    'root': '', 'calibration': []}
    if len(roots) == 1:
        return find_fields(roots[0], max_fields=max_fields, max_scanned=max_scanned,
                           min_frames=min_frames, skip_calibration=skip_calibration)

    shared = _common_parent(roots)
    fields, merged = [], {'scanned': 0, 'found': 0, 'truncated': None,
                          'root': str(shared) if shared else ', '.join(str(r) for r in roots),
                          'roots': [str(r) for r in roots], 'calibration': []}
    seen = set()
    for root in roots:
        found, info = find_fields(root, max_fields=max_fields, max_scanned=max_scanned,
                                  min_frames=min_frames,
                                  skip_calibration=skip_calibration)
        merged['scanned'] += info.get('scanned', 0)
        merged['calibration'] += info.get('calibration') or []
        if info.get('truncated'):
            # one bad root must not silently drop the others, but it must be said
            merged['truncated'] = info['truncated']
            return [], merged
        for field in found:
            if field['folder'] in seen:      # nested selections would process twice
                continue
            seen.add(field['folder'])
            # keep the chosen folder's own name at the front, so two fields called
            # `22_16_15` under different roots do not collide in the output
            prefix = root.name if shared is None or root != shared else ''
            relative = field['relative']
            field['relative'] = str(Path(prefix) / relative) if prefix else relative
            fields.append(field)
    if len(fields) > max_fields:
        merged['truncated'] = (
            f'more than {max_fields} folders of frames across the {len(roots)} folders '
            f'chosen, so nothing has been started. Choose fewer, or raise the limit.')
        return [], merged
    merged['found'] = len(fields)
    if not fields:
        merged['truncated'] = (f'no image frames in any of the {len(roots)} folders chosen '
                               f'({merged["scanned"]} folder(s) looked at)')
    return fields, merged


def _common_parent(roots):
    """The deepest folder every root sits under, or None if they share none."""
    try:
        import os.path
        shared = Path(os.path.commonpath([str(r) for r in roots]))
    except ValueError:                 # different drives on Windows
        return None
    return shared if str(shared) not in ('', '.') else None


def batch_root_for(roots):
    """The folder a multi-root run should be named after.

    Their shared parent when they have one -- `.../Zenith` for two fields chosen under it,
    which is what someone would call that run. When they share nothing (two drives), the
    first choice names it and the rest are recorded in the summary.
    """
    roots = [Path(r) for r in (roots if isinstance(roots, (list, tuple)) else [roots]) if r]
    if not roots:
        return ''
    if len(roots) == 1:
        return str(roots[0])
    shared = _common_parent(roots)
    return str(shared) if shared is not None else str(roots[0])


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
    """A one-line summary for the log.

    Says how many calibration folders were left out. A skip nobody is told about reads as
    "everything was processed", and the whole reason for the skip is that those folders
    used to appear as failed fields.
    """
    if info.get('truncated'):
        return info['truncated']
    total = sum(len(f['frames']) for f in fields)
    skipped = info.get('calibration') or []
    return (f'{len(fields)} field(s), {total} frame(s), from '
            f'{info["scanned"]} folder(s) under {info["root"]}'
            + (f'; {len(skipped)} calibration folder(s) skipped '
               f'({", ".join(Path(s).name for s in skipped[:4])}'
               f'{", ..." if len(skipped) > 4 else ""})' if skipped else ''))
