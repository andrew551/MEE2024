"""Matrix cell 1 (Bruns 2017): S0 inventory of the raw eclipse ladder, folder-first.

The ladder (I:\\2017 eclipse images Don Bruns\\2017 Eclipse images\\eclipse, read-only):
EA 17 x 0.62 s -> E2 11 x 0.09 s -> EB 17 x 0.62 s, all full-frame 3296x2472 (the
1400x1400 subframe in Bruns' own VBS script was NOT used). Timestamps in the filenames
are local seconds-after-midnight (MDT = UTC-6); DATE-OBS is UTC and is the authority.
Leon's EXPTIME trap is checked anyway: header vs filename exposure, per frame.

Measured here, per tier: mid-time (UTC, from DATE-OBS), max ADU, saturated-pixel count,
rough Sun centre (centroid of saturated pixels -- known biased ~70 px by streamers, used
only to measure the saturation radius), the 99th-percentile saturation radius, and the
sky level in radial annuli (the coronal-subtraction design numbers).
"""
import glob, os, re
import numpy as np
from astropy.io import fits

RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017"
os.makedirs(OUT, exist_ok=True)
PS = 2.0868004                 # arcsec/px, the L/R bracket mean (bruns2017_lr canonical)
SATLEV = 65535

tiers = {'EA': sorted(glob.glob(os.path.join(RAW, 'EA_*.fit'))),
         'E2': sorted(glob.glob(os.path.join(RAW, 'E2_*.fit'))),
         'EB': sorted(glob.glob(os.path.join(RAW, 'EB_*.fit')))}

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

for t, files in tiers.items():
    say(f'== {t}: {len(files)} frames')
    times, exps, maxs, nsats = [], [], [], []
    cxs, cys = [], []
    sat_acc = None
    for f in files:
        with fits.open(f) as hd:
            img = hd[0].data
            hdr = hd[0].header
        fn_exp = float(re.search(r'exp_([\d.]+)_sec', os.path.basename(f)).group(1))
        if abs(hdr['EXPTIME'] - fn_exp) > 1e-6:
            say(f'  EXPTIME TRAP: {os.path.basename(f)} header {hdr["EXPTIME"]} vs filename {fn_exp}')
        times.append(hdr['DATE-OBS']); exps.append(hdr['EXPTIME'])
        maxs.append(int(img.max()))
        sat = img >= SATLEV
        nsats.append(int(sat.sum()))
        if sat.any():
            yy, xx = np.nonzero(sat)
            cxs.append(xx.mean()); cys.append(yy.mean())
            sat_acc = sat.astype(np.int32) if sat_acc is None else sat_acc + sat
    t0, t1 = times[0], times[-1]
    say(f'  exposures: {sorted(set(exps))} s; DATE-OBS {t0} .. {t1}')
    # mid-time: mean of start times + half the (uniform) exposure
    def sec(s):
        hh, mm, ss = s.split('T')[1].split(':')
        return int(hh)*3600 + int(mm)*60 + float(ss)
    mid = np.mean([sec(s) for s in times]) + exps[0]/2
    say(f'  mid-time {int(mid//3600):02d}:{int(mid%3600//60):02d}:{mid%60:04.1f} UTC')
    say(f'  max ADU {min(maxs)}..{max(maxs)}; saturated px/frame {min(nsats)}..{max(nsats)}')
    if cxs:
        cx, cy = float(np.mean(cxs)), float(np.mean(cys))
        say(f'  rough Sun centre (sat centroid, streamer-biased): ({cx:.0f}, {cy:.0f}) px')
        # 99th-pct saturation radius over the union of frames' saturated sets
        yy, xx = np.nonzero(sat_acc >= max(1, len(files)//2))
        r = np.hypot(xx-cx, yy-cy)
        say(f'  saturation radius (99th pct, >=half the frames): {np.percentile(r, 99):.0f} px'
            f' = {np.percentile(r, 99)*PS:.0f} arcsec')
    else:
        say('  no saturated pixels')
    # sky in radial annuli of the LAST frame (representative), around the rough centre
    with fits.open(files[-1]) as hd:
        img = hd[0].data.astype(np.float64)
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if cxs:
        rr = np.hypot(xx-cx, yy-cy)*PS/948.0        # ~R_sun 2017 in arcsec
        say('  sky by annulus (median ADU): ' + ', '.join(
            f'{a:.1f}-{b:.1f}Rs {np.median(img[(rr>a)&(rr<b)]):.0f}'
            for a, b in ((1.2,1.5),(1.5,2.0),(2.0,2.5),(2.5,3.0),(3.0,4.0),(4.0,6.0))))
open(os.path.join(OUT, 'inventory.txt'), 'w').write('\n'.join(lines))
print('written ->', os.path.join(OUT, 'inventory.txt'), flush=True)
