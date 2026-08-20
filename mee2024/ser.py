"""
Reading SER files: a whole capture in one container, one frame at a time.

SER is what capture software writes when FITS-per-frame cannot keep up. A 61-megapixel
camera at 3.2 fps is 388 MB/s and 122 MB per frame; asking a filesystem for 180 separate
files at that rate is asking a lot, and the frames are really a video. The format is
correspondingly plain: a 178-byte header, then frame after frame of raw pixels, then
optionally one 64-bit timestamp per frame. Reading frame N is a seek, which suits a pipeline
that already works one frame at a time and never wants a 15 GB file resident.

**A SER frame is addressed as ``path.ser#N``.** The pipeline's unit of work is a frame path,
and rewriting that assumption everywhere would be a far larger change than reading the
format. A reference carries its index instead, so `find_fields`, the aligner, the stacker and
the logs all keep working on "a list of frames" -- see :func:`parse_ref`.

Three things measured on real files, each of which will bite a reader that trusts the
specification:

* **The ``LittleEndian`` header field cannot be trusted.** On the file examined it reads 0 --
  big-endian by a literal reading -- while the pixels are unambiguously little-endian. This
  is a known ambiguity in the format's history and writers disagree. :func:`byte_order`
  measures it from the pixels instead, which takes a few milliseconds and is decisive.
* **The timestamps are often absent**, in two different ways. One file has a trailer of 180
  slots containing nothing but zeros -- the space is there and unwritten -- and it is the one
  capture of three that ended abnormally, finishing on eight blank frames. The trailer is
  written when a capture *completes*, so an interrupted one loses it. (Not a setting: all
  three sidecars say `Timestamp Frames=Off`, including the two whose trailers are full.)
  Another file, trimmed by an external tool, had lost even the header timestamp. So timing
  falls back through: the per-frame trailer, the header, then the sidecar.
* **The header strings are not zero-filled.** They carry leftover memory after the
  terminator, so a reader must cut at the first NUL rather than strip trailing spaces.

The `.CameraSettings.txt` sidecar the capture software writes alongside carries most of what
a FITS header would -- exposure, gain, offset, binning, camera, both temperatures, and UTC
start/middle/end -- plus something FITS does not have: explicit confirmation that no dark,
flat, background subtraction or banding suppression was applied. See :func:`read_sidecar`.
"""

import logging
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

#: Module logger, deliberately without a handler of its own. It inherits whatever the run
#: configured, and with nothing configured Python's last-resort handler still puts a warning
#: on stderr -- so an unreadable sidecar is visible either way. Attaching a FileHandler here
#: would repeat the leak that ``MEE2024util.close_logger`` exists to undo.
_LOGGER = logging.getLogger(__name__)

#: What the first 14 bytes of every SER file say.
FILE_ID = 'LUCAM-RECORDER'

HEADER_BYTES = 178
SUFFIX = '.ser'

#: Separates a container from the frame index within it: ``capture.ser#42``.
FRAME_SEPARATOR = '#'

#: ColorID values that mean three planes per pixel rather than one.
RGB_IDS = (100, 101)

#: .NET ticks are 100-nanosecond intervals since 0001-01-01.
TICKS_PER_MICROSECOND = 10
DOTNET_EPOCH = datetime(1, 1, 1)


class SerError(Exception):
    """A file that does not read as SER, with the reason."""


# --------------------------------------------------------------------- frame references

def parse_ref(ref):
    """``'a/b.ser#42'`` -> ``(Path('a/b.ser'), 42)``; a plain path -> ``(path, None)``.

    Deliberately tolerant of a bare SER path with no index, which means "the whole file" and
    is what a user types.
    """
    text = str(ref)
    if FRAME_SEPARATOR in text:
        head, _, tail = text.rpartition(FRAME_SEPARATOR)
        if tail.isdigit() and head:
            return Path(head), int(tail)
    return Path(text), None


def make_ref(path, index):
    """The reference for one frame of a container."""
    return f'{path}{FRAME_SEPARATOR}{int(index)}'


def is_ser(path):
    """Does this path name a SER container? Accepts a frame reference."""
    container, _ = parse_ref(path)
    return container.suffix.lower() == SUFFIX


def looks_like_ser(path):
    """Is this actually a SER file? Checks the magic rather than the extension."""
    container, _ = parse_ref(path)
    try:
        with open(container, 'rb') as fp:
            return fp.read(14).decode('latin-1', 'replace') == FILE_ID
    except OSError:
        return False


# --------------------------------------------------------------------------- the header

def _clean(raw):
    """A header string: cut at the NUL, because the field is not zero-filled."""
    return raw.split(b'\x00')[0].decode('latin-1', 'replace').strip()


def _ticks_to_datetime(ticks):
    if not ticks or ticks <= 0:
        return None
    try:
        return DOTNET_EPOCH + timedelta(microseconds=ticks // TICKS_PER_MICROSECOND)
    except (OverflowError, ValueError):
        return None


def read_header(path):
    """The SER header as a dict, with the derived geometry worked out.

    Raises :class:`SerError` if the file does not start with the SER magic, so a
    mis-detected file fails here with a sentence rather than deep inside numpy.
    """
    container, _ = parse_ref(path)
    try:
        size = container.stat().st_size
        with open(container, 'rb') as fp:
            raw = fp.read(HEADER_BYTES)
    except OSError as exc:
        raise SerError(f'cannot read {container}: {exc}') from exc
    if len(raw) < HEADER_BYTES:
        raise SerError(f'{container.name} is too short to be a SER file '
                       f'({len(raw)} bytes, a header alone is {HEADER_BYTES})')
    file_id = raw[0:14].decode('latin-1', 'replace')
    if file_id != FILE_ID:
        raise SerError(f'{container.name} does not start with {FILE_ID!r} '
                       f'(it starts with {file_id!r}), so it is not a SER file')

    lu_id, color_id, little, width, height, depth, frames = struct.unpack('<7i', raw[14:42])
    dt_local, dt_utc = struct.unpack('<2q', raw[162:178])
    planes = 3 if color_id in RGB_IDS else 1
    bytes_per_sample = 2 if depth > 8 else 1
    frame_bytes = width * height * planes * bytes_per_sample
    payload = frames * frame_bytes
    trailer = size - HEADER_BYTES - payload

    return {
        'path': str(container), 'size': size,
        'lu_id': lu_id, 'color_id': color_id, 'little_endian_flag': little,
        'width': width, 'height': height, 'depth': depth, 'frames': frames,
        'planes': planes, 'bytes_per_sample': bytes_per_sample,
        'frame_bytes': frame_bytes,
        'observer': _clean(raw[42:82]),
        'instrument': _clean(raw[82:122]),
        'telescope': _clean(raw[122:162]),
        'datetime_local': _ticks_to_datetime(dt_local),
        'datetime_utc': _ticks_to_datetime(dt_utc),
        'trailer_bytes': max(trailer, 0),
        'has_timestamps': trailer == frames * 8 and frames > 0,
    }


def read_timestamps(path):
    """The per-frame UTC timestamps, or None when the file carries none.

    A trailer of the right *size* is not enough: the space exists but may never have been
    written, which is what one real file shows for all 180 of its slots. That file is the one
    capture of three that ended abnormally -- its last eight frames are blank -- and the
    trailer is written when a capture completes. So the values are checked, not just the
    length.
    """
    header = read_header(path)
    if not header['has_timestamps']:
        return None
    offset = HEADER_BYTES + header['frames'] * header['frame_bytes']
    with open(header['path'], 'rb') as fp:
        fp.seek(offset)
        raw = fp.read(header['frames'] * 8)
    ticks = struct.unpack(f'<{header["frames"]}q', raw)
    stamps = [_ticks_to_datetime(t) for t in ticks]
    return stamps if any(s is not None for s in stamps) else None


# ------------------------------------------------------------------------- the byte order

#: How much better one byte order must score before the measurement is believed. Below this
#: the frame has nothing to judge by -- see :func:`byte_order`.
ORDER_MARGIN = 0.8


def byte_order(path, header=None):
    """``'<'`` or ``'>'``, measured from the pixels rather than read from the flag.

    The test counts how many distinct values each byte of the pair takes. Real 16-bit image
    data occupies a modest part of the container -- a sky at a few thousand ADU out of
    65535 -- so its **high byte varies slowly and takes few values**, while the low byte
    runs through the whole 0-255 with the noise. Whichever byte of the pair is the more
    repetitive is therefore the high one, and that fixes the order. On the file this was
    written for: 43 distinct values in one byte against 256 in the other.

    Two measures were tried first and both fail, which is worth recording because both look
    obviously right:

    * **spatial smoothness** (neighbouring pixels agree) -- fails on a frame with no
      large-scale structure, where the neighbour difference *is* the whole deviation
      whichever way it is read;
    * **the same, normalised by the range** -- fails whenever the data spans less than one
      step of the high byte, because then swapping is exactly a multiplication by 256 plus
      a constant, and *every* ratio is invariant under it.

    Counting distinct byte values survives both cases. When neither byte is clearly more
    repetitive the data genuinely does not say -- it fills the container -- and the answer
    falls back to little-endian, which is what writers produce in practice. 8-bit data has
    no byte order to get wrong.
    """
    header = header or read_header(path)
    if header['bytes_per_sample'] == 1:
        return '<'
    width = header['width']
    rows = min(64, header['height'])
    with open(header['path'], 'rb') as fp:
        fp.seek(HEADER_BYTES)
        raw = fp.read(width * rows * 2 * header['planes'])
    if len(raw) < 4:
        return '<'

    pairs = np.frombuffer(raw[:len(raw) // 2 * 2], dtype=np.uint8).reshape(-1, 2)
    first = int(np.unique(pairs[:, 0]).size)
    second = int(np.unique(pairs[:, 1]).size)
    if first and second:
        if first < ORDER_MARGIN * second:
            return '>'          # the first byte of each pair varies least: it is the high one
        if second < ORDER_MARGIN * first:
            return '<'
    return '<'


# ------------------------------------------------------------------------- the sidecar

#: How the capture software's setting names map onto the FITS keywords the pipeline reads.
#: The values it writes are strings with units attached, so each entry says how to parse it.
_SIDECAR_MAP = {
    'exposure': ('EXPTIME', 'seconds'),
    'analogue gain': ('GAIN', 'number'),
    'gain': ('GAIN', 'number'),
    'offset': ('OFFSET', 'number'),
    'binning': ('XBINNING', 'number'),
    'cameraserialnumber': ('CAMID', 'text'),
    'temperature': ('CCD-TEMP', 'number'),
    'target temperature': ('SET-TEMP', 'number'),
    'startcapture': ('DATE-OBS', 'time'),
    'midcapture': ('DATE-AVG', 'time'),
    'endcapture': ('DATE-END', 'time'),
    'jdstartcapture': ('JD_UTC', 'number'),
    'framecount': ('NFRAMES', 'number'),
    'frametype': ('FRAMETYP', 'text'),
    'timezone': ('TIMEZONE', 'text'),
}

#: Settings that say the pixels were *modified* by the capture software. FITS headers carry
#: nothing equivalent, and if any of these were on the frames are already calibrated and the
#: pipeline's assumptions break silently -- so they are read and reported.
_PROCESSING_KEYS = ('subtract dark', 'apply flat', 'background subtraction',
                    'banding suppression', 'dark scaling (experimental)',
                    'remove satellite trails')

_NOT_APPLIED = {'', 'none', 'off', '0', 'false', 'no'}


def sidecar_path(path):
    """Where the capture software's settings file sits for this container, or None."""
    container, _ = parse_ref(path)
    for candidate in (container.with_suffix('.CameraSettings.txt'),
                      container.parent / f'{container.stem}.CameraSettings.txt'):
        if candidate.exists():
            return candidate
    matches = sorted(container.parent.glob('*.CameraSettings.txt'))
    return matches[0] if len(matches) == 1 else None


def _parse_value(raw, kind):
    """One sidecar value, as the matching FITS header would carry it.

    ``None`` means the value could not be read, which drops the key. The raw text survives
    under ``'_raw'``, so nothing is lost -- but an unreadable *timestamp* is now logged
    rather than discarded in silence. It used to return ``None`` with nothing said, so a
    capture whose ``StartCapture`` would not parse lost ``DATE-OBS``, fell through to the
    date guesser, and left no record of why: indistinguishable from a sidecar that never
    carried a time at all.

    That silence hid a real interpreter dependency for a while. ``datetime.fromisoformat``
    before Python 3.11 accepts only 3 or 6 fractional digits, and these files write 7
    (``...T18:28:45.5937053Z``), so every SER capture reduced on 3.9 or 3.10 quietly lost its
    start time. The floor is 3.12 from 2026-08-20, which puts that cause out of reach;
    capture software that writes some other format is not.
    """
    raw = raw.strip()
    if kind == 'text':
        return raw
    if kind == 'seconds':
        match = re.match(r'([-\d.]+)\s*(ms|s)?', raw, re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value / 1000.0 if (match.group(2) or '').lower() == 'ms' else value
    if kind == 'number':
        match = re.match(r'[-\d.]+', raw)
        return float(match.group(0)) if match else None
    if kind == 'time':
        try:
            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
        except ValueError:
            _LOGGER.warning('SER sidecar: cannot read the timestamp %r, so this capture '
                            'falls back to the guessed date', raw)
            return None
    return raw


def read_sidecar(path):
    """The capture settings beside a SER file, as FITS-like keys.

    Returns ``{}`` when there is none. The raw settings are kept under ``'_raw'`` so nothing
    is lost, and the processing flags are summarised under ``'_modified'`` -- a list of any
    that were *on*, which is the answer to "are these pixels untouched?".
    """
    settings = sidecar_path(path)
    if settings is None:
        return {}
    try:
        text = settings.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return {}

    raw = {}
    out = {'_sidecar': str(settings)}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            # the section header is the camera model
            out.setdefault('INSTRUME', line[1:-1].strip())
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        raw[key] = value
        mapped = _SIDECAR_MAP.get(key.lower())
        if mapped:
            parsed = _parse_value(value, mapped[1])
            if parsed is not None:
                out[mapped[0]] = parsed

    # the mount's own claim about where it was pointing, e.g.
    #   ASI Mount=RA=09:13:06.0,Dec=+13:56:48 (JNOW)
    for key, value in raw.items():
        if 'mount' in key.lower() and 'RA=' in value:
            ra = re.search(r'RA=([\d:.+-]+)', value)
            dec = re.search(r'Dec=([\d:.+-]+)', value)
            if ra and dec:
                out['OBJCTRA'] = ra.group(1)
                out['OBJCTDEC'] = dec.group(1)

    if 'Capture Area' in raw:
        match = re.match(r'(\d+)\s*x\s*(\d+)', raw['Capture Area'])
        if match:
            out['NAXIS1'], out['NAXIS2'] = int(match.group(1)), int(match.group(2))
    if 'Colour Space' in raw:
        match = re.search(r'(\d+)', raw['Colour Space'])
        if match:
            out['BITDEPTH'] = int(match.group(1))

    # matched case-insensitively, since the file's capitalisation varies between versions
    lower = {k.lower(): v for k, v in raw.items()}
    modified = [key for key in _PROCESSING_KEYS
                if str(lower.get(key, '')).strip().lower() not in _NOT_APPLIED]
    out['_modified'] = modified
    out['_raw'] = raw
    return out


# ------------------------------------------------------------------------ reading frames

class SerFile:
    """One SER container, opened once and read by frame index.

    Holds no pixel data: every :meth:`read` seeks and returns one frame, so a 22 GB capture
    costs one frame of memory. The header, byte order and sidecar are resolved on
    construction because each is needed for every frame and none is expensive.
    """

    def __init__(self, path):
        self.path = parse_ref(path)[0]
        self.header = read_header(self.path)
        self.order = byte_order(self.path, self.header)
        self.sidecar = read_sidecar(self.path)
        self._timestamps = None
        self._timestamps_read = False

    # -------------------------------------------------------------- geometry and count

    @property
    def frames(self):
        return self.header['frames']

    @property
    def shape(self):
        return self.header['height'], self.header['width']

    @property
    def dtype(self):
        return f'{self.order}u{self.header["bytes_per_sample"]}'

    def refs(self, start=0, stop=None):
        """Frame references for a range, which is what the pipeline consumes."""
        stop = self.frames if stop is None else min(stop, self.frames)
        return [make_ref(self.path, i) for i in range(max(start, 0), stop)]

    # ------------------------------------------------------------------------ the pixels

    def read(self, index):
        """One frame as a 2-D array. Colour containers are reduced to luminance."""
        if not 0 <= index < self.frames:
            raise SerError(f'{self.path.name} has {self.frames} frames, '
                           f'so frame {index} does not exist')
        h = self.header
        with open(self.path, 'rb') as fp:
            fp.seek(HEADER_BYTES + index * h['frame_bytes'])
            raw = fp.read(h['frame_bytes'])
        if len(raw) < h['frame_bytes']:
            raise SerError(f'{self.path.name} ends early: frame {index} is incomplete '
                           f'({len(raw)} of {h["frame_bytes"]} bytes). The file is '
                           f'truncated, or the frame count in its header is wrong.')
        data = np.frombuffer(raw, dtype=self.dtype)
        if h['planes'] == 3:
            data = data.reshape(h['height'], h['width'], 3)
            # the same luminance weighting the ordinary image path uses
            data = (data[..., 0] * 0.299 + data[..., 1] * 0.587 + data[..., 2] * 0.114)
        else:
            data = data.reshape(h['height'], h['width'])
        return np.asarray(data, dtype=np.float32)

    # ---------------------------------------------------------------------- the metadata

    def timestamps(self):
        """Per-frame UTC datetimes, or None if the file carries none."""
        if not self._timestamps_read:
            self._timestamps = read_timestamps(self.path)
            self._timestamps_read = True
        return self._timestamps

    def frame_time(self, index):
        """The best UTC time available for one frame.

        Per-frame trailer if the file has one; otherwise the sidecar's start time plus the
        frame's share of the capture; otherwise the header's own UTC stamp. Returns None
        when the file has been through a tool that dropped all of them.
        """
        stamps = self.timestamps()
        if stamps and 0 <= index < len(stamps) and stamps[index] is not None:
            return stamps[index]
        start = self.sidecar.get('DATE-OBS') or self.header['datetime_utc']
        if start is None:
            return None
        end = self.sidecar.get('DATE-END')
        if end is not None and self.frames > 1:
            span = (end - start).total_seconds()
            return start + timedelta(seconds=span * index / (self.frames - 1))
        exptime = self.sidecar.get('EXPTIME')
        if exptime:
            return start + timedelta(seconds=exptime * index)
        return start

    def fits_header(self, index=0):
        """What a FITS header would have said, assembled from the container and sidecar.

        This is what lets the rest of the pipeline treat a SER frame like any other: the
        calibration library keys on GAIN and EXPTIME, stage 1 wants DATE-OBS, and the solver
        check wants the pointing. All of them are here or in the sidecar.
        """
        out = dict(self.sidecar)
        out.pop('_raw', None)
        header = self.header
        out.setdefault('INSTRUME', header['instrument'] or None)
        out.setdefault('TELESCOP', header['telescope'] or None)
        out.setdefault('OBSERVER', header['observer'] or None)
        out['NAXIS1'], out['NAXIS2'] = header['width'], header['height']
        out.setdefault('BITDEPTH', header['depth'])
        when = self.frame_time(index)
        if when is not None:
            out['DATE-OBS'] = when.replace(tzinfo=None).isoformat()
        return {k: v for k, v in out.items() if v is not None}

    def describe(self):
        h = self.header
        bits = [f'{h["frames"]} frames', f'{h["width"]}x{h["height"]}', f'{h["depth"]}-bit',
                'little-endian' if self.order == '<' else 'big-endian']
        if h['instrument']:
            bits.append(h['instrument'])
        if self.timestamps():
            bits.append('per-frame timestamps')
        elif h['datetime_utc']:
            bits.append('header UTC only')
        else:
            bits.append('no timestamps')
        return ', '.join(bits)


#: Containers opened more than once in a run -- the batch scanner asks for the frame count,
#: then the stacker reads every frame. Re-parsing the header each time is cheap but
#: re-measuring the byte order is a file read, so the object is kept.
_OPEN = {}


def open_ser(path):
    """A :class:`SerFile` for this path, reused across a run."""
    container, _ = parse_ref(path)
    key = str(container.resolve()) if container.exists() else str(container)
    if key not in _OPEN:
        _OPEN[key] = SerFile(container)
    return _OPEN[key]


def release():
    """Forget the cached containers. Nothing is held open, so this only frees memory."""
    _OPEN.clear()


def read_frame(ref):
    """One frame from a reference like ``capture.ser#42``. Frame 0 if no index is given."""
    container, index = parse_ref(ref)
    return open_ser(container).read(index or 0)


def expand(path, start=0, stop=None):
    """Every frame of a container as a list of references."""
    return open_ser(path).refs(start=start, stop=stop)
