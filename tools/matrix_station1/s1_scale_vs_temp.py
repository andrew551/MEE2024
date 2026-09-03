"""Cell 2: the session's +50 ppm scale event is sensor temperature, and the eclipse is thermally
matched to its calibration.

Douglas found four of the seventeen raw zenith blocks on 2026-09-03. Their headers turn the
session's one plate-scale outlier from an unexplained event into a measurement.

`s1_zenith_floor.py` found sixteen of the seventeen 2024 zenith fields within −28 to +10 ppm
of their mean and one — 05:51:25Z, after a 13-minute gap and a 5° slew — at **+50 ppm**, and
recorded that the null pair spanning it cost Method 2 0.19 ″ where its neighbours cost
0.02–0.11 ″. That field is one of the four now available raw, and its `CCD-TEMP` is the
answer:

    05:32:53Z   -11.00 C      05:38:32Z    -9.90 C
    05:35:48Z    -9.67 C      05:51:25Z    +2.53 C   <- the +50 ppm field, 12.7 C warmer

The cooler evidently lost its setpoint over the 13-minute gap and was still coming back down
during the block (+4.0 C on the first frame, +1.2 C on the last). Silicon expands at about
2.6 ppm/K, so a warmer sensor has larger pixels and a larger plate scale in arcsec per pixel:
12.7 C predicts roughly 33 ppm against the 47 ppm observed, the remainder presumably the
optics and focus moving with the same warming.

This fits plate scale against sensor temperature on whatever fields have both, and reports:

  1. the slope in ppm per degree, from the modern windowed reductions (one convention
     throughout -- the 2024 moment fits differ from them by 4-10 ppm per field, which is the
     same size as the signal between the three cold fields, so the two must not be mixed);
  2. the eclipse tiers' own sensor temperature against the calibration fields', which is what
     decides whether any of this reaches L. It does not: both sit at -10 C.

The -600 ppm between the eclipse and the zenith calibration is NOT thermal -- it would need
160 C -- and remains the daytime refocus, as `s1_eclipse_method2_test.py` found.
"""
import glob, json, os
import numpy as np
from astropy.io import fits

RAW = r"I:/Mexico 2024/Station 1 Zenith"
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data/CapObj"
AB = r"D:/MEE2024 output/MEE_output/station1_record/zenith_raw_ab"
Z24 = r"D:/MEE2024 output/Station 1/zenith calibrations"
BLOCKS = [('f1', '2024-04-08_05_32_53Z', '20240417201719'),
          ('f2', '2024-04-08_05_35_48Z', '20240417202159'),
          ('f3', '2024-04-08_05_38_32Z', '20240417203538'),
          ('f4', '2024-04-08_05_51_25Z', '20240417203916')]
SI_CTE = 2.6          # ppm per K, silicon


def temp(block):
    fs = sorted(glob.glob(os.path.join(RAW, block, '*.FIT')))
    t = [fits.getheader(f).get('CCD-TEMP') for f in fs]
    return float(np.mean(t)), len(t)


def ps_modern(tag):
    hit = glob.glob(os.path.join(AB, 'windowed_annular', tag, 'stage2_free', '**', 'distortion_results.txt'), recursive=True)
    if not hit:
        return None, None, None
    j = json.load(open(hit[0], encoding='utf-8'))
    return j['platescale (arcseconds/pixel)'], j['#stars used'], j['final rms error (arcseconds)']


def ps_2024(stamp):
    hit = glob.glob(os.path.join(Z24, '*%s*' % stamp, '**', 'distortion_results.txt'), recursive=True)
    if not hit:
        return None
    return json.load(open(hit[0], encoding='utf-8'))['platescale (arcseconds/pixel)']


rows = []
print('%-4s %-22s %5s %9s %14s %8s %9s %14s' % ('', 'block', 'n', 'CCD-TEMP', 'ps (windowed)', 'stars', 'rms (")', 'ps (2024)'))
for tag, block, stamp in BLOCKS:
    T, n = temp(block)
    p, ns, rms = ps_modern(tag)
    p24 = ps_2024(stamp)
    print('%-4s %-22s %5d %8.2f C %14s %8s %9s %14s'
          % (tag, block, n, T,
             '%.7f' % p if p else 'not fitted',
             ns if ns else '-', '%.4f' % rms if rms else '-',
             '%.7f' % p24 if p24 else '-'))
    if p:
        rows.append((tag, T, p, p24))

if len(rows) >= 3:
    T = np.array([r[1] for r in rows]); P = np.array([r[2] for r in rows])
    cold = T < -5
    base = P[cold].mean()
    print('\n=== plate scale against sensor temperature (modern windowed reductions) ===')
    print('  reference: the mean of the %d fields below -5 C = %.7f "/px' % (cold.sum(), base))
    for tag, t, p, p24 in rows:
        print('    %-3s %+7.2f C  %+8.1f ppm from that mean' % (tag, t, 1e6*(p-base)/base))
    if len(rows) >= 2 and (T.max() - T.min()) > 1:      # ndarray.ptp() was removed in numpy 2
        c, cov = np.polyfit(T, 1e6*(P-base)/base, 1, cov=True)
        print('  fitted slope: %+.2f +- %.2f ppm per degree C  (silicon alone predicts %.1f)'
              % (c[0], np.sqrt(cov[0, 0]), SI_CTE))
        if (~cold).any():
            dp = 1e6*(P[~cold][0]-base)/base
            dt = T[~cold][0]-T[cold].mean()
            print('  the warm field alone: %+.1f ppm over %+.2f C = %.2f ppm/C' % (dp, dt, dp/dt))
            print('  the three cold fields scatter %.1f ppm over a %.2f C range, so they carry no'
                  % (1e6*(P[cold].max()-P[cold].min())/base, T[cold].max()-T[cold].min()))
            print('  independent leverage: the slope rests on the one warm field, and is quoted as')
            print('  an order of magnitude, not a coefficient.')

print('\n=== does any of it reach the eclipse? ===')
et = []
for d in sorted(os.listdir(G)):
    fs = sorted(glob.glob(os.path.join(G, d, '*.FIT')))
    if not fs:
        continue
    t = [fits.getheader(f).get('CCD-TEMP') for f in fs[::20]]
    et.append(np.mean(t))
    print('  eclipse tier %-12s %3d frames, CCD-TEMP %+.2f C' % (d[-9:], len(fs), np.mean(t)))
cold_T = np.mean([r[1] for r in rows if r[1] < -5]) if rows else float('nan')
if et:
    print('  eclipse mean %+.2f C against the cold zenith fields\' %+.2f C: a %.2f C difference,'
          % (np.mean(et), cold_T, abs(np.mean(et)-cold_T)))
    print('  worth about %.1f ppm of scale -- below the 0.7 ppm HC3 of a single field is not claimed,'
          % (abs(np.mean(et)-cold_T)*SI_CTE))
    print('  but it is far below the -600 ppm the refocus put there, and below the 8 ppm spread')
    print('  among the calibration fields themselves. The transfer is thermally matched.')
print('\nThe -600 ppm eclipse-to-zenith offset would need 160 C at this coefficient. It is the refocus.')
