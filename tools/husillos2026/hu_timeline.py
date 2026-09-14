"""The totality timeline, and the AM5's settling time measured from it.

Douglas, 2026-09-11: "This is actually a good opportunity to determine the settling time of the
AM5 mount.  Let's make a timeline for the start and finish of the three blocks taken during
totality.  Is there no gap between any of them?  Does the gap between blocks 2 and 3 include
the slew or does block 3 start after the slew has finished?"

Everything here is READ from the captures' own CameraSettings sidecars and from stage 1's
per-frame alignment record.  Nothing is derived from the folder names, which are local time.

WHY THIS IS MEASURABLE AT ALL.  Block 3 (CalibS) is 10.17 deg from blocks 1-2, so the mount
had to slew between them, and block 3's stage-1 alignment then records what the mount did
afterwards -- 81 frames at 3.17 fps of pure settling, sampled every 0.315 s.  That is a better
mount test than anything shot on purpose, because it is the real slew the real rig made under
the real load.

    .venv/Scripts/python.exe tools/husillos2026/hu_timeline.py
"""
import datetime
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

G = r'G:\Joe Izen Spain 2026\2026-08-12'                      # READ ONLY
HUS = r'F:\MEE_output\husillos2026'

#: (label, capture directory, file stem, stage-1 output, frames used)
BLOCKS = [
    ('1  Sun    ', 'SunJoe_20260812_182845', '20_28_45',
     os.path.join(HUS, 'eclipse', 's1_sun_dark'), '46-171'),
    ('2  Sn2    ', 'Sn2_Joe_20260812_182942', '20_29_43',
     os.path.join(HUS, 'eclipse', 's1_sn2_darkall'), '2-102'),
    ('3  CalibS ', 'CalibS_Joe_20260812_183018', '20_30_18',
     os.path.join(HUS, 'calibs', 's1_calibs_ecl'), '1-81'),
]

#: from the site card, `G:\Joe Izen Spain 2026\Husillos Spain.JPG`
C2 = datetime.datetime(2026, 8, 12, 18, 29, 0, 500000)
C3 = datetime.datetime(2026, 8, 12, 18, 30, 44, 200000)
#: CalibS' angular separation from the eclipse blocks' pointing, from the two plate solves
SLEW_DEG = 10.17
PS = 2.2030


def settings(folder, stem):
    kv = {}
    p = os.path.join(G, folder, stem + '.CameraSettings.txt')
    for line in open(p, encoding='utf-8', errors='replace'):
        if '=' in line:
            k, v = line.split('=', 1)
            kv[k.strip()] = v.strip()
    return kv


def t(v):
    return datetime.datetime.strptime(v[:23], '%Y-%m-%dT%H:%M:%S.%f')


def shifts(d):
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        return None
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z[0]).open('results.txt'),
                                   encoding='utf-8', errors='replace'))
    return np.array(r['alignment']['shifts_px'])


def main():
    rows = []
    for lbl, folder, stem, s1, frames in BLOCKS:
        kv = settings(folder, stem)
        rows.append(dict(lbl=lbl, start=t(kv['StartCapture']), end=t(kv['EndCapture']),
                         n=int(kv['FrameCount']), fps=float(kv['ActualFrameRate'].rstrip('fps')),
                         gain=kv['Analogue Gain'], exp=kv['Exposure'], used=frames,
                         drift=shifts(s1)))

    print('TOTALITY: C2 %s  ->  C3 %s  (%.1f s)'
          % (C2.strftime('%H:%M:%S.%f')[:12], C3.strftime('%H:%M:%S.%f')[:12],
             (C3 - C2).total_seconds()))
    print()
    print('%-11s %-13s %-13s %7s %6s %5s %9s %s'
          % ('block', 'start UTC', 'end UTC', 'dur (s)', 'frames', 'gain', 'interval', 'used'))
    for r in rows:
        print('%-11s %-13s %-13s %7.3f %6d %5s %9.4f %s'
              % (r['lbl'], r['start'].strftime('%H:%M:%S.%f')[:12],
                 r['end'].strftime('%H:%M:%S.%f')[:12],
                 (r['end'] - r['start']).total_seconds(), r['n'], r['gain'],
                 1.0 / r['fps'], r['used']))

    print()
    print('GAPS between consecutive blocks:')
    for a, b in zip(rows, rows[1:]):
        gap = (b['start'] - a['end']).total_seconds()
        iv = 1.0 / a['fps']
        print('   %s -> %s   %6.3f s  = %5.2f frame intervals   %s'
              % (a['lbl'].strip(), b['lbl'].strip(), gap, gap / iv,
                 'NO REAL GAP (one frame interval)' if gap < 1.5 * iv else 'A REAL GAP'))

    print()
    print('DOES THE 3.156 s GAP CONTAIN THE SLEW?')
    gap = (rows[2]['start'] - rows[1]['end']).total_seconds()
    d = rows[2]['drift']
    tot_as = float(np.hypot(*d[-1]) * PS)
    print('   the slew is %.2f deg = %.0f arcsec (blocks 1-2 on the Sun, block 3 %.2f deg away)'
          % (SLEW_DEG, SLEW_DEG * 3600, SLEW_DEG))
    print('   block 3 moves only %.1f arcsec across its whole 81-frame capture'
          % tot_as)
    print('   ratio: the slew is %.0fx larger than anything block 3 records'
          % (SLEW_DEG * 3600 / tot_as))
    print('   -> THE SLEW COMPLETED INSIDE THE GAP. Block 3 starts after it, and what its')
    print('      alignment records is the SETTLING TAIL, not the slew.')
    print('   mean slew rate if it filled the gap: %.2f deg/s' % (SLEW_DEG / gap))

    print()
    print('THE AM5 SETTLING TIME, from block 3\'s own alignment record')
    dist = np.hypot(d[:, 0], d[:, 1]) * PS
    tt = np.arange(len(d)) / rows[2]['fps']
    # rate from 10-frame windows, then an exponential fitted to the rate:
    #   rate(t) = (A/tau) exp(-t/tau)   ->   ln(rate) is linear in t with slope -1/tau
    # SUPERSEDED (2026-09-14, record section 3n): this gives 9.2 s where a fit on the
    # displacement itself gives 7.3 s (hu_settle_chart.py; 7.4 s on the RA axis alone).  The
    # windowed speed never falls to zero -- the RA axis is still creeping at ~0.5 "/s at 25 s
    # and the frame-to-frame jitter adds to every window -- so ln(rate) flattens late and the
    # slope comes out shallower than the settle's own decay.  The reason given below for not
    # fitting the displacement was wrong: an exponential that starts part-way through the
    # settle is still an exponential with the same tau, only a smaller A.  Kept so the
    # record's first number can be reproduced; do not quote it as the settling time.
    # (Original note: the displacement cannot be fitted directly because block 3 begins
    # part-way through the settle, so its zero is arbitrary; the rate does not care where the
    # clock started.)
    ta, ra = [], []
    for a in range(0, len(d) - 10, 5):
        b = a + 10
        ra.append((dist[b] - dist[a]) / (tt[b] - tt[a]))
        ta.append(0.5 * (tt[a] + tt[b]))
    ta, ra = np.array(ta), np.array(ra)
    ok = ra > 0.02
    sl, ic = np.polyfit(ta[ok], np.log(ra[ok]), 1)
    tau = -1.0 / sl
    print('   settled displacement across the capture: %.1f arcsec (%.1f px)'
          % (dist[-1], dist[-1] / PS))
    print('   exponential time constant  tau = %.1f s   (fit on %d windows, r = %.2f)'
          % (tau, ok.sum(), float(np.corrcoef(ta[ok], np.log(ra[ok]))[0, 1])))
    print('   [SUPERSEDED: the displacement fit gives 7.3 s (7.4 s on the RA axis) --'
          ' hu_settle_chart.py, record section 3n]')
    print('   so after a %.1f deg slew the AM5 needs about %.0f s to fall to 1/e,'
          % (SLEW_DEG, tau))
    print('   %.0f s to 5 %% (3 tau) and %.0f s to 1 %% (4.6 tau).' % (3 * tau, 4.6 * tau))
    print()
    print('   what that costs a 0.315 s exposure, as smear:')
    for a in range(0, 80, 10):
        b = a + 10
        print('      frames %2d-%2d  %5.2f arcsec/s  = %.2f px within one exposure'
              % (a, b, (dist[b] - dist[a]) / (tt[b] - tt[a]),
                 (dist[b] - dist[a]) / (tt[b] - tt[a]) / rows[2]['fps'] / PS))
    print('   the PSF is ~1.6 px FWHM, so the first ~20 frames are measurably trailed.')

    print()
    print('WHERE EACH BLOCK SITS RELATIVE TO TOTALITY:')
    for r in rows:
        print('   %s starts C2%+7.1f s, ends C3%+7.1f s%s'
              % (r['lbl'], (r['start'] - C2).total_seconds(),
                 (r['end'] - C3).total_seconds(),
                 '   <-- runs past C3' if r['end'] > C3 else ''))


if __name__ == '__main__':
    main()
