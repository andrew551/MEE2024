"""
Reading SER containers, and addressing one frame inside one.

Every test writes a real SER file rather than mocking the reader, because the things that
went wrong on real data are all in the bytes: a header field that contradicts the pixels, a
timestamp trailer that is allocated and empty, header strings with leftover memory after the
terminator. A mock would agree with whatever the reader assumed.
"""

import struct
from datetime import datetime

import numpy as np
import pytest

from mee2024 import ser

DOTNET_EPOCH = datetime(1, 1, 1)


def _ticks(when):
    return int((when - DOTNET_EPOCH).total_seconds() * 10_000_000)


def write_ser(path, frames, *, width=16, height=12, depth=16, colour=0,
              little_endian_flag=0, byte_order='<', observer='Observer',
              instrument='ZWO ASI2600MM Pro', telescope='FRA500',
              utc=None, local=None, timestamps=None, trailing_junk=True):
    """A real SER file, with the awkward details of the real ones reproducible on demand."""
    header = bytearray(178)
    header[0:14] = b'LUCAM-RECORDER'
    struct.pack_into('<7i', header, 14, 0, colour, little_endian_flag,
                     width, height, depth, len(frames))

    def field(text):
        raw = text.encode('latin-1')[:39] + b'\x00'
        # real files do not zero-fill: they leave whatever was in memory after the NUL
        junk = bytes(range(1, 40 - len(raw) + 1)) if trailing_junk else bytes(40 - len(raw))
        return (raw + junk)[:40]

    header[42:82] = field(observer)
    header[82:122] = field(instrument)
    header[122:162] = field(telescope)
    struct.pack_into('<2q', header, 162,
                     _ticks(local) if local else 0, _ticks(utc) if utc else 0)

    dtype = f'{byte_order}u{2 if depth > 8 else 1}'
    with open(path, 'wb') as fp:
        fp.write(bytes(header))
        for frame in frames:
            fp.write(np.asarray(frame, dtype=dtype).tobytes())
        if timestamps is not None:
            fp.write(struct.pack(f'<{len(timestamps)}q',
                                 *[_ticks(t) if t else 0 for t in timestamps]))
    return path


def _smooth_frame(width, height, level=500, seed=0):
    """A frame that looks like sky: smooth, so the byte-order test has something to work on."""
    rng = np.random.default_rng(seed)
    return (level + rng.normal(0, 3, (height, width))).astype(np.uint16)


@pytest.fixture
def sample(tmp_path):
    frames = [_smooth_frame(16, 12, 500, seed=i) for i in range(5)]
    return write_ser(tmp_path / 'capture.ser', frames)


# ----------------------------------------------------------------- frame references

@pytest.mark.parametrize('text, index', [
    ('a/b.ser#42', 42), ('a/b.ser#0', 0), ('a/b.ser', None), ('a/b.fits', None),
])
def test_a_frame_reference_carries_its_index(text, index):
    path, got = ser.parse_ref(text)
    assert got == index
    assert path.name in ('b.ser', 'b.fits')


def test_a_windows_path_is_not_mistaken_for_a_reference():
    """Drive letters and folder names must survive; only a trailing #digits is an index."""
    path, index = ser.parse_ref(r'C:\Eclipse #3\capture.ser')
    assert index is None and path.name == 'capture.ser'


def test_make_and_parse_round_trip(tmp_path):
    ref = ser.make_ref(tmp_path / 'x.ser', 7)
    path, index = ser.parse_ref(ref)
    assert index == 7 and path.name == 'x.ser'


# --------------------------------------------------------------------------- the header

def test_the_header_reads_back(sample):
    header = ser.read_header(sample)
    assert header['frames'] == 5
    assert (header['width'], header['height']) == (16, 12)
    assert header['depth'] == 16


def test_header_strings_are_cut_at_the_nul_not_stripped(sample):
    """Real files leave memory after the terminator; rstrip is not enough."""
    header = ser.read_header(sample)
    assert header['instrument'] == 'ZWO ASI2600MM Pro'
    assert header['observer'] == 'Observer'
    assert '\x00' not in header['instrument']


def test_a_file_that_is_not_ser_says_so(tmp_path):
    path = tmp_path / 'notser.ser'
    path.write_bytes(b'this is not a SER file at all' * 20)
    with pytest.raises(ser.SerError, match='LUCAM-RECORDER'):
        ser.read_header(path)


def test_a_truncated_file_says_so(tmp_path):
    path = tmp_path / 'short.ser'
    path.write_bytes(b'LUCAM-RECORDER' + bytes(20))
    with pytest.raises(ser.SerError, match='too short'):
        ser.read_header(path)


# ----------------------------------------------------------------------- the byte order

def test_the_byte_order_is_measured_not_taken_from_the_flag(tmp_path):
    """The flag lies on real files: it reads 0 (big-endian) while the pixels are little.

    So the reader measures it, and this writes exactly that contradiction.
    """
    frames = [_smooth_frame(64, 8, 500, seed=1)]
    path = write_ser(tmp_path / 'lying.ser', frames,
                     width=64, height=8, little_endian_flag=0, byte_order='<')
    assert ser.read_header(path)['little_endian_flag'] == 0      # says big-endian
    assert ser.byte_order(path) == '<'                            # but is little-endian
    handle = ser.SerFile(path)
    assert float(np.median(handle.read(0))) == pytest.approx(500, abs=5)


def test_genuinely_big_endian_data_is_read_that_way(tmp_path):
    frames = [_smooth_frame(64, 8, 500, seed=2)]
    path = write_ser(tmp_path / 'big.ser', frames,
                     width=64, height=8, little_endian_flag=1, byte_order='>')
    assert ser.byte_order(path) == '>'
    assert float(np.median(ser.SerFile(path).read(0))) == pytest.approx(500, abs=5)


# --------------------------------------------------------------------- reading frames

def test_each_frame_reads_back_with_its_own_content(tmp_path):
    frames = [np.full((12, 16), value, dtype=np.uint16) for value in (100, 200, 300)]
    path = write_ser(tmp_path / 'ramp.ser', frames)
    handle = ser.SerFile(path)
    assert handle.frames == 3
    for index, value in enumerate((100, 200, 300)):
        assert float(np.median(handle.read(index))) == value


def test_asking_for_a_frame_that_is_not_there_says_how_many_there_are(sample):
    with pytest.raises(ser.SerError, match='5 frames'):
        ser.SerFile(sample).read(99)


def test_read_frame_goes_through_a_reference(tmp_path):
    frames = [np.full((12, 16), value, dtype=np.uint16) for value in (10, 20, 30)]
    path = write_ser(tmp_path / 'r.ser', frames)
    assert float(np.median(ser.read_frame(ser.make_ref(path, 1)))) == 20
    # a bare path means frame 0
    assert float(np.median(ser.read_frame(path))) == 10


def test_expand_gives_one_reference_per_frame(sample):
    refs = ser.expand(sample)
    assert len(refs) == 5
    assert all(ser.parse_ref(r)[1] == i for i, r in enumerate(refs))


# ------------------------------------------------------------------------- timestamps

def test_per_frame_timestamps_are_used_when_present(tmp_path):
    stamps = [datetime(2026, 8, 12, 18, 28, 45 + i) for i in range(3)]
    path = write_ser(tmp_path / 'stamped.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(3)],
                     timestamps=stamps)
    handle = ser.SerFile(path)
    assert handle.timestamps() is not None
    assert handle.frame_time(1) == stamps[1]


def test_an_allocated_but_empty_trailer_counts_as_no_timestamps(tmp_path):
    """The space can exist and never be written -- one real 180-frame file is exactly like
    that, being the one capture that ended abnormally. Size alone is not enough."""
    path = write_ser(tmp_path / 'empty.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(3)],
                     timestamps=[None, None, None],
                     utc=datetime(2026, 8, 12, 18, 28, 45))
    handle = ser.SerFile(path)
    assert handle.header['has_timestamps'] is True     # the space is there
    assert handle.timestamps() is None                 # but it holds nothing
    # so it falls back to the header's own UTC stamp
    assert handle.frame_time(0) == datetime(2026, 8, 12, 18, 28, 45)


def test_no_timestamps_at_all_is_not_an_error(tmp_path):
    """A file trimmed by an external tool can lose every timestamp it had."""
    path = write_ser(tmp_path / 'bare.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(3)])
    handle = ser.SerFile(path)
    assert handle.timestamps() is None
    assert handle.frame_time(0) is None


# --------------------------------------------------------------------------- the sidecar

SIDECAR = """[Zeus 455M PRO (IMX455)]
CameraSerialNumber=CAMD01825CE042109000
FrameType=Light
Binning=1
Colour Space=MONO16
Capture Area=9576x6388
Offset=200
Analogue Gain=125
Exposure=315.0000ms
Temperature=-0.30000001192092896
Target Temperature=0
Background Subtraction=Off
Banding Suppression=0
Apply Flat=None
Subtract Dark=None
ASI Mount=RA=09:13:06.0,Dec=+13:56:48 (JNOW)
StartCapture=2026-08-12T18:28:45.5937053Z
MidCapture=2026-08-12T18:29:13.9803398Z
EndCapture=2026-08-12T18:29:42.3669744Z
FrameCount=180
TimeZone=+2.00
"""


def test_the_sidecar_supplies_what_a_fits_header_would(tmp_path):
    path = write_ser(tmp_path / 'cap.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(3)])
    (tmp_path / 'cap.CameraSettings.txt').write_text(SIDECAR, encoding='utf-8')
    header = ser.SerFile(path).fits_header(0)
    assert header['EXPTIME'] == pytest.approx(0.315)     # 'ms' understood
    assert header['GAIN'] == 125
    assert header['OFFSET'] == 200
    assert header['XBINNING'] == 1
    assert header['INSTRUME'] == 'Zeus 455M PRO (IMX455)'
    assert header['CAMID'] == 'CAMD01825CE042109000'
    assert header['SET-TEMP'] == 0
    assert header['CCD-TEMP'] == pytest.approx(-0.3)
    assert header['OBJCTRA'] == '09:13:06.0'
    assert header['OBJCTDEC'] == '+13:56:48'


def test_the_sidecar_confirms_the_pixels_were_not_modified(tmp_path):
    """The one thing a FITS header cannot tell you: whether the capture software already
    applied a dark, a flat, or background subtraction."""
    path = write_ser(tmp_path / 'cap.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(2)])
    (tmp_path / 'cap.CameraSettings.txt').write_text(SIDECAR, encoding='utf-8')
    assert ser.read_sidecar(path)['_modified'] == []

    spoiled = SIDECAR.replace('Subtract Dark=None', 'Subtract Dark=master_dark.fit')
    (tmp_path / 'cap.CameraSettings.txt').write_text(spoiled, encoding='utf-8')
    assert 'subtract dark' in ser.read_sidecar(path)['_modified']


def test_the_sidecar_start_time_fills_in_for_a_missing_trailer(tmp_path):
    path = write_ser(tmp_path / 'cap.ser',
                     [_smooth_frame(16, 12, 500, seed=i) for i in range(3)])
    (tmp_path / 'cap.CameraSettings.txt').write_text(SIDECAR, encoding='utf-8')
    handle = ser.SerFile(path)
    first, last = handle.frame_time(0), handle.frame_time(2)
    assert first is not None and last is not None
    assert first < last                       # interpolated across the capture


def test_no_sidecar_is_not_an_error(sample):
    assert ser.read_sidecar(sample) == {}
    assert ser.SerFile(sample).fits_header(0)['NAXIS1'] == 16
