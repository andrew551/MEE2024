"""Inventory of a SharpCap SER capture tree from the headers: present, whole, and distinct.

Written for Joe Izen's Husillos 2026 data (matrix cell 4) on 2026-09-09, when the drive held 23 SER
against the 13 of the mid-download snapshot and two captures had byte-for-byte the same size as
two others. Folder names are local time; only the header carries UTC, and only the frames tell
two captures apart. So this reads:

  * the header as mee2024/ser.py reads it -- '<7i' at bytes 14:42 (luID, colorID, endian, width,
    height, depth, frames) and '<2q' at 162:178 (datetime, datetime_utc; 100 ns ticks since
    0001-01-01);
  * whether the file is whole: size must equal header + frames, plus 8 bytes per frame when the
    trailer of per-frame timestamps is present;
  * which CameraSettings.txt files sit beside no SER (captures announced, never delivered);
  * for any pair of files of identical size, an md5 of the first and last frames, so that one
    capture copied under two names cannot pass as a rehearsal.

  .venv/Scripts/python.exe tools/husillos2026/ser_inventory.py ["G:/Joe Izen Spain 2026"]
"""
import datetime
import glob
import hashlib
import os
import re
import struct
import sys
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"G:/Joe Izen Spain 2026"
EPOCH = datetime.datetime(1, 1, 1)


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, '/')


def header(p):
    with open(p, 'rb') as f:
        h = f.read(178)
    _luid, _color, _endian, w, hgt, depth, n = struct.unpack('<7i', h[14:42])
    dt, dt_utc = struct.unpack('<2q', h[162:178])
    bpp = 2 if depth > 8 else 1
    size = os.path.getsize(p)
    expected = 178 + n * w * hgt * bpp
    return dict(w=w, h=hgt, depth=depth, n=n, bpp=bpp, size=size, expected=expected,
                trailer=size - expected,
                utc=EPOCH + datetime.timedelta(microseconds=dt_utc / 10) if dt_utc > 0 else None,
                local=EPOCH + datetime.timedelta(microseconds=dt / 10) if dt > 0 else None)


def frame_hash(p, k, H):
    with open(p, 'rb') as f:
        f.seek(178 + k * H['w'] * H['h'] * H['bpp'])
        return hashlib.md5(f.read(H['w'] * H['h'] * H['bpp'])).hexdigest()[:10]


def settings(txt):
    s = open(txt, encoding='utf-8', errors='replace').read()

    def grab(pat):
        m = re.search(pat, s, re.I)
        return m.group(1).strip() if m else '?'

    return grab(r'Exposure\s*=\s*([^\r\n]+)'), grab(r'Gain\s*=\s*([^\r\n]+)'), grab(r'Offset\s*=\s*([^\r\n]+)')


sers = sorted(glob.glob(os.path.join(ROOT, '**', '*.ser'), recursive=True))
print('%-50s %6s %-10s %3s %19s %7s  %s' % ('file', 'frames', 'dims', 'bit', 'UTC start', 'GB', 'whole?'))
total, by_size = 0, defaultdict(list)
for p in sers:
    H = header(p)
    total += H['size']
    by_size[H['size']].append((p, H))
    if H['trailer'] in (0, 8 * H['n']):
        whole = 'yes' + (' (+timestamps)' if H['trailer'] else '')
    else:
        short = (H['expected'] - H['size']) / (H['w'] * H['h'] * H['bpp'])
        whole = 'SHORT: %.1f frames missing' % short if short > 0 else 'unexpected trailer of %d bytes' % H['trailer']
    print('%-50s %6d %-10s %3d %19s %7.2f  %s' % (
        rel(p), H['n'], '%dx%d' % (H['w'], H['h']), H['depth'],
        H['utc'].strftime('%Y-%m-%d %H:%M:%S') if H['utc'] else '?', H['size'] / 1e9, whole))
print('%d SER, %.1f GB' % (len(sers), total / 1e9))

print()
print('=== settings files with no SER beside them (announced, not delivered):')
none = True
for d in sorted({os.path.dirname(t) for t in glob.glob(os.path.join(ROOT, '**', '*.txt'), recursive=True)}):
    names = [os.path.basename(s) for s in glob.glob(os.path.join(d, '*.ser'))]
    for t in sorted(glob.glob(os.path.join(d, '*.txt'))):
        stem = os.path.basename(t).split('.CameraSettings')[0].split('_CameraSettings')[0]
        if not any(stem in s for s in names):
            e, g, o = settings(t)
            print('   %-46s exp %-10s gain %-4s offset %s' % (rel(os.path.join(d, stem)), e, g, o))
            none = False
if none:
    print('   none')

print()
print('=== files of identical size: the same capture under two names, or distinct?')
none = True
for size, group in by_size.items():
    if len(group) < 2:
        continue
    none = False
    hashes = []
    for p, H in group:
        hashes.append((frame_hash(p, 0, H), frame_hash(p, H['n'] - 1, H)))
        print('   %-50s UTC %s  first/last frame md5 %s / %s' % (rel(p), H['utc'], *hashes[-1]))
    print('      -> %s' % ('all DISTINCT' if len(set(hashes)) == len(hashes) else 'DUPLICATE FRAMES PRESENT'))
if none:
    print('   none')

print()
print('=== settings per capture (exposure, gain, offset):')
for t in sorted(glob.glob(os.path.join(ROOT, '**', '*.txt'), recursive=True)):
    e, g, o = settings(t)
    print('   %-60s exp %-10s gain %-4s offset %s' % (rel(t), e, g, o))
