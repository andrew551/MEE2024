"""Does the AM5 approach the sidereal rate slowly?  Two checks on the pointing-B captures.

Douglas, 2026-09-14: "My guess is the observed behaviour has to do with the fact that the RA
axis never comes to rest because it must always keep tracking at the sidereal rate.  This
means that by design the RA axis is not 'stopped' with maximum electromagnetic force but
rather is allowed to asymptotically approach the correct angular velocity to achieve SR
tracking.  This possibility could be tested by measuring the drift after a slew when the
tracking is turned off."

Two captures at pointing B bracket the moment tracking was switched on (record section 3w):
23_40_27 (14 frames, gain 0, untracked to its last frame at 21:40:44.8 UTC) and 23_41_01
(51 frames, gain 0, tracked from its first frame at 21:41:01.0).  So the RA motor went from
rest to the sidereal rate inside a 16.2 s window, and 23_41_01's alignment record shows what
was left of that approach when it opened.  A first-order approach with CalibS' RA time
constant of 7.4 s would still be short of the rate by 13.8 * exp(-16.2 / 7.4) = 1.6 "/s at
frame 1, i.e. at least 11 " of displacement still to come; the record says how much came.
Between the 23_37_17 and 23_41_01 solves the pointing also moved ~0.25 deg in RA beyond the
sidereal drift, so an RA-only nudge happened in one of the two gaps (21:39:28-21:40:27 or
21:40:45-21:41:01); nothing dates it, and the tool says what each case would mean.

Neither capture is the tracking-off-after-a-slew test Douglas proposed -- no slew is dated to
within 25 s of 23_40_27, whose 13 aligned frames are fitted here only to show what a
tracking-off record looks like when nothing is settling.  Units: arcsec, s, px.

    .venv/Scripts/python.exe tools/husillos2026/hu_switch_on.py
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hu_settle_chart as H  # noqa: E402

G = r'G:\Joe Izen Spain 2026\2026-08-12\10 deg'          # READ ONLY
HUS = r'F:\MEE_output\husillos2026\horizon'

SIDEREAL = 360.0 * 3600.0 / 86164.0905     # arcsec/s
DEC_B = 23.284                             # deg, pointing B (record section 3u, 23_37_17 deep solve)
TAU_CALIBS_RA = 7.4                        # s, record section 3n: CalibS RA axis, pure exponential
A_CALIBS_RA = 49.2                         # arcsec, same fit
GAP2_S = 16.2                              # s, 23_40_27 last frame 21:40:44.8 -> 23_41_01 frame 1 21:41:01.0

#: (label, SER stem, stage-1 output, stage-2 output or None)
CAPTURES = [
    ('23_40_27 (tracking OFF)', '23_40_27', os.path.join(HUS, 's1_h10_g0_short_23_40_27'), None),
    ('23_41_01 (tracking ON)', '23_41_01', os.path.join(HUS, 's1_h10_g0_c'),
     os.path.join(HUS, 's2_h10_g0_c')),
]


def frame_interval(stem):
    """Seconds per frame from the capture's own sidecar, never typed in."""
    for line in open(os.path.join(G, stem + '.CameraSettings.txt'), encoding='utf-8',
                     errors='replace'):
        if line.startswith('ActualFrameRate='):
            return 1.0 / float(line.split('=', 1)[1].strip().rstrip('fps'))
    raise KeyError(stem)


def line(t, v, c):
    return v * t + c


def fit_line_and_exp(name, tt, x, tau):
    p, cov = curve_fit(line, tt, x)
    r = x - line(tt, *p)
    print('   %-12s line: rate %+.3f +- %.3f arcsec/s = %+.1f arcsec/min, residual rms %.2f arcsec'
          % (name, p[0], np.sqrt(cov[0, 0]), p[0] * 60, r.std()))

    def le(t, v, c, A):
        return v * t + c + A * (1 - np.exp(-t / tau))
    q, cov = curve_fit(le, tt, x, p0=[p[0], p[1], 0.0])
    print('   %-12s line + exponential with tau fixed at %.1f s: A %+.1f +- %.1f arcsec'
          % ('', tau, q[2], np.sqrt(cov[2, 2])))
    return q[2], np.sqrt(cov[2, 2])


def main():
    sid_b = SIDEREAL * np.cos(np.radians(DEC_B))
    print('sidereal drift of an untracked pointing at Dec %+.3f: %.2f arcsec/s of arc'
          % (DEC_B, sid_b))
    print()
    for label, stem, s1, s2 in CAPTURES:
        dt = frame_interval(stem)
        tt, along, perp_rms = H.displacement(s1, dt)
        print('%s: %d aligned frames at %.4f s, %.1f s; along-drift %.1f arcsec, '
              'perpendicular rms %.2f arcsec' % (label, len(tt), dt, tt[-1], along[-1], perp_rms))
        if s2 is None:
            fit_line_and_exp('along', tt, along, TAU_CALIBS_RA)
            print('   -> %.2f arcsec/s against %.2f sidereal: the RA motor was at rest; no '
                  'exponential.' % (along[-1] / tt[-1], sid_b))
        else:
            tt, comps = H.axis_components(s1, s2, dt)
            for k, v in comps.items():
                A, eA = fit_line_and_exp(k, tt, np.asarray(v, float), TAU_CALIBS_RA)
                if k.startswith('RA'):
                    left = sid_b * np.exp(-GAP2_S / TAU_CALIBS_RA)
                    print('   -> a tau = %.1f s approach to the sidereal rate begun at the START of '
                          'the %.1f s switch-on window\n      would still be %.2f arcsec/s short at '
                          'frame 1 and add >= %.1f arcsec: the record allows %.1f +- %.1f.'
                          % (TAU_CALIBS_RA, GAP2_S, left, left * TAU_CALIBS_RA, A, eA))
                    for tau in (2.0, 3.0, 4.0, 5.0):
                        print('      (tau %.0f s from the start of the window would leave %.1f arcsec)'
                              % (tau, sid_b * tau * np.exp(-GAP2_S / tau)))
                    print('   -> and if the ~0.25 deg RA nudge fell in that window, a CalibS-like '
                          'settle (%.0f arcsec, tau %.1f s)\n      would have left >= %.1f arcsec: '
                          'the record allows %.1f +- %.1f.'
                          % (A_CALIBS_RA, TAU_CALIBS_RA,
                             A_CALIBS_RA * np.exp(-GAP2_S / TAU_CALIBS_RA), A, eA))
            k = 8
            for key, v in comps.items():
                v = np.asarray(v, float)
                print('   %-12s first %d frames %+.2f arcsec/s, last 20 frames %+.2f arcsec/s '
                      '(straight-line slopes, not fitted models)'
                      % (key, k, np.polyfit(tt[:k], v[:k], 1)[0], np.polyfit(tt[-20:], v[-20:], 1)[0]))
        print()
    print('NOT the proposed test: no slew is dated to within 25 s of 23_40_27 (the slew to '
          'pointing B preceded 23_37_17,\nthree minutes earlier), so a tracking-off record after '
          'a slew is still to be taken.')


if __name__ == '__main__':
    main()
