"""The two eclipse captures, frame by frame: where totality starts, and which frames belong.

Douglas, 2026-09-10, with two claims to check before anything is stacked:

  1. In `SunJoe_20260812_182845/20_28_45.ser` (180 frames, 315 ms, **gain 125**) totality does
     not begin until about **frame 50**.
  2. In `Sn2_Joe_20260812_182942/20_29_43.ser` (103 frames, 315 ms, **gain 0**) the **first two
     frames actually belong to the first file** -- a SharpCap frame-buffer carry-over, as was
     seen in the Leon data.

The second is testable to a certainty most such claims are not, because **the two captures were
shot at different GAINS**.  A frame carried over from the Sun capture is a gain-125 frame sitting
in a gain-0 file, and gain changes the ADU per electron by ~4.2x -- it moves the bias, the sky and
the saturation behaviour all at once.  So a carried-over frame does not merely look "anomalous",
it looks like the *other file*, and the two hypotheses make opposite predictions about which.

The sidecars already make the mechanism possible.  `20_28_45` ends at 18:29:42.367 and
`20_29_43` starts at 18:29:42.690: a gap of **0.323 s, one frame interval**, with no slew between
them.  That is exactly the situation in which a driver's ring buffer hands the first frames of the
new capture the last frames of the old.

Reading strategy: a 600-row band through the Sun rather than whole frames.  The Sun sits at about
(4600, 3400) px, so rows 3100-3700 carry the disk, the inner corona and, at both ends of the band,
sky far from either.  283 frames of band is 2.2 GB against 34.5 GB of whole frames, and nothing
measured here needs the rest.

  .venv/Scripts/python.exe tools/husillos2026/hu_eclipse_frames.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ser_track import read_header, read_timestamps  # noqa: E402

G = r"G:/Joe Izen Spain 2026/2026-08-12"
OUT = r"D:/MEE2024 output/MEE_output/husillos2026/eclipse"

SUN = os.path.join(G, 'SunJoe_20260812_182845', '20_28_45.ser')
SN2 = os.path.join(G, 'Sn2_Joe_20260812_182942', '20_29_43.ser')

#: the band of rows read from every frame, chosen to contain the Sun at y ~ 3400
ROW0, ROW1 = 3100, 3700
#: columns far from the Sun at x ~ 4600, for a sky level that the corona does not reach
SKY_COLS = 900
SAT = 65000


def scan(path, label):
    H = read_header(path)
    ts = read_timestamps(path, H)
    w, h, n = H['w'], H['h'], H['n']
    row_bytes = w * 2
    rows = []
    with open(path, 'rb') as f:
        for k in range(n):
            f.seek(178 + k * H['frame_bytes'] + ROW0 * row_bytes)
            band = np.frombuffer(f.read((ROW1 - ROW0) * row_bytes),
                                 dtype='<u2').reshape(ROW1 - ROW0, w).astype(np.float32)
            sky = np.concatenate([band[:, :SKY_COLS].ravel(), band[:, -SKY_COLS:].ravel()])
            bg = float(np.median(sky))
            above = band - bg
            bright = above > 1000
            rows.append(dict(
                frame=k, capture=label,
                t_s=(ts[k] - ts[0]).total_seconds() if ts else k * 0.3152,
                utc=ts[k].isoformat() if ts else '',
                sky_adu=bg,
                sky_sd=float(1.4826 * np.median(np.abs(sky - bg))),
                band_median=float(np.median(band)),
                max_adu=float(band.max()),
                sat_px=int((band >= SAT).sum()),
                px_over_1000=int(bright.sum()),
                flux_above_sky=float(above[above > 0].sum())))
            if k % 20 == 0:
                print('  %s frame %3d  sky %7.1f  max %7.0f  sat %7d  >1000 %8d'
                      % (label, k, bg, rows[-1]['max_adu'], rows[-1]['sat_px'],
                         rows[-1]['px_over_1000']), flush=True)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for path, label in ((SUN, 'sun_20_28_45'), (SN2, 'sn2_20_29_43')):
        cache = os.path.join(OUT, label + '_frames.csv')
        if os.path.exists(cache):
            frames.append(pd.read_csv(cache))
        else:
            print('scanning %s' % os.path.basename(path), flush=True)
            df = scan(path, label)
            df.to_csv(cache, index=False)
            frames.append(df)
    sun, sn2 = frames
    pd.set_option('display.width', 200)

    print()
    print('=' * 100)
    print('1. WHERE DOES TOTALITY START IN THE SUN CAPTURE?  (180 frames, 315 ms, gain 125)')
    print('=' * 100)
    print(sun[['frame', 't_s', 'sky_adu', 'band_median', 'max_adu', 'sat_px', 'px_over_1000']]
          .iloc[::5].to_string(index=False, float_format=lambda v: '%.1f' % v))

    print()
    print('=' * 100)
    print('2. DO THE FIRST FRAMES OF Sn2 BELONG TO THE SUN CAPTURE?')
    print('=' * 100)
    print('the last five frames of the SUN capture (gain 125):')
    print(sun[['frame', 't_s', 'sky_adu', 'sky_sd', 'band_median', 'max_adu', 'sat_px',
               'px_over_1000']].tail(5)
          .to_string(index=False, float_format=lambda v: '%.2f' % v))
    print()
    print('the first eight frames of Sn2 (gain 0):')
    print(sn2[['frame', 't_s', 'sky_adu', 'sky_sd', 'band_median', 'max_adu', 'sat_px',
               'px_over_1000']].head(8)
          .to_string(index=False, float_format=lambda v: '%.2f' % v))
    print()
    print('Sn2 frames 5-102, for what a settled gain-0 frame looks like:')
    body = sn2.iloc[5:]
    print('  sky %.2f +- %.2f ADU   sky sd %.3f +- %.3f   max %.0f-%.0f   sat %d-%d'
          % (body.sky_adu.mean(), body.sky_adu.std(), body.sky_sd.mean(), body.sky_sd.std(),
             body.max_adu.min(), body.max_adu.max(), body.sat_px.min(), body.sat_px.max()))
    print('->', OUT)


if __name__ == '__main__':
    main()
