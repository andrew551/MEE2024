"""CalibS' drift against time, with the two settling fits overlaid.

Douglas, 2026-09-13: "Plot the CalibS drift against time with both fits overlaid."

WHAT IS PLOTTED.  The stage-1 alignment record of the frames-1-81 CalibS stack
(`calibs/s1_calibs_ecl`, `results.txt['alignment']['shifts_px']`): the global shift the
stacker applied to each 0.3153 s frame to align it to frame 1, i.e. where the field had
moved to.  The motion is nearly straight, so it is projected onto its own mean direction and
shown in arcsec (2.2028 "/px); the perpendicular scatter is quoted, not drawn.  Time is the
frame index times the frame interval, from frame 1 -- which began 3.156 s after the last
frame of the gain-0 coronal block, with the 10.17 deg slew completed inside that gap
(record section 3n).

THE TWO FITS, both least squares on the displacement:

    pure exponential      x(t) = A (1 - exp(-t / tau))
    exponential + drift   x(t) = A (1 - exp(-t / tau)) + v t

The record's tau = 9.2 s (section 3n) was fitted on the RATE over 15 windows, not on the
displacement, so the pure-exponential displacement fit here need not return 9.2 exactly; the
legend carries what each fit found and the subtitle carries the record's value.  A residual
panel shows what each model leaves behind against the frame-to-frame image motion at 8.8 deg.

Chart conventions follow the dataviz reference palette: data in the secondary text ink, the
two fits in categorical slots 1 and 2 (blue, orange -- an adjacent pair validated at CVD
Delta E 9.1), 2 px lines, >= 8 px markers with a surface ring, hairline solid grid, one y axis
per panel (arcsec; the pixel conversion is in the label, never a second axis), a legend
because there are two fitted series, text never in a series colour.

Writes calibs/calibs_settling.png (+ chart_versions/) and publishes it through
hu_record.publish so an earlier revision is superseded, never overwritten.

    .venv/Scripts/python.exe tools/husillos2026/hu_settle_chart.py
"""
import glob
import io
import json
import os
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from record_charts import ChartWriter  # noqa: E402
import hu_record  # noqa: E402

HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
SRC = os.path.join(HUS, 'calibs', 's1_calibs_ecl')
OUT = os.path.join(HUS, 'calibs')
REV = os.environ.get('HU_REV', 'rev01')
PS = 2.2028                 # arcsec per px, the cell's plate scale
DT = 0.3153                 # s per frame, CalibS (hu_timeline.py)
GAP_S = 3.156               # s from the last gain-0 coronal frame to CalibS frame 1
SLEW_DEG = 10.17
TAU_RECORD = 9.2            # s, section 3n, fitted on the rate
SETTLED_FIRST = 21          # first frame of the settled stack (hu_calibs.SETTLED_FIRST)

# dataviz reference palette, light mode
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
FIT1, FIT2 = '#2a78d6', '#eb6834'


def displacement():
    z = glob.glob(os.path.join(SRC, 'centroid_data*.zip'))[0]
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z).open('results.txt'), encoding='utf-8',
                                   errors='replace'))
    s = np.array(r['alignment']['shifts_px'], float)
    d = s - s[0]
    u = d[-1] / np.linalg.norm(d[-1])
    along = d @ u * PS
    perp = d - np.outer(d @ u, u)
    return np.arange(len(s)) * DT, along, float(np.hypot(*perp.T).std() * PS)


def m1(t, A, tau):
    return A * (1 - np.exp(-t / tau))


def m2(t, A, tau, v):
    return A * (1 - np.exp(-t / tau)) + v * t


def main():
    t, x, perp_rms = displacement()
    p1, c1 = curve_fit(m1, t, x, p0=(45, 9))
    p2, c2 = curve_fit(m2, t, x, p0=(35, 5, 0.5))
    e1, e2 = np.sqrt(np.diag(c1)), np.sqrt(np.diag(c2))
    r1, r2 = x - m1(t, *p1), x - m2(t, *p2)
    tt = np.linspace(0, t[-1], 400)
    for lab, p, e, r in (('pure exponential', p1, e1, r1), ('exponential + drift', p2, e2, r2)):
        print('%-22s ' % lab + '  '.join('%.2f +- %.2f' % (a, b) for a, b in zip(p, e))
              + '   residual rms %.2f "' % r.std())
    print('perpendicular scatter %.2f " rms; total along-drift displacement %.1f " (%.1f px)'
          % (perp_rms, x[-1], x[-1] / PS))

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=(3.2, 1.3), hspace=0.08))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axr):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=1.0, linestyle='-')
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9.5)
    # the data
    ax.plot(t, x, 'o', ms=6, color=INK2, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label='field displacement, one point per 0.3153 s frame', zorder=3)
    # the fits
    ax.plot(tt, m1(tt, *p1), '-', lw=2, color=FIT1, solid_capstyle='round',
            label='pure exponential:  A = %.1f ″,  τ = %.1f ± %.1f s' % (p1[0], p1[1], e1[1]),
            zorder=4)
    ax.plot(tt, m2(tt, *p2), '-', lw=2, color=FIT2, solid_capstyle='round',
            label='exponential + drift:  A = %.1f ″,  τ = %.1f ± %.1f s,  drift %.0f ″/min'
                  % (p2[0], p2[1], e2[1], p2[2] * 60), zorder=4)
    # the settled-stack cut
    tcut = (SETTLED_FIRST - 1) * DT
    for a in (ax, axr):
        a.axvline(tcut, color=INK2, lw=1, alpha=0.6)
    ax.text(tcut + 0.25, 1.5, 'settled stack begins\n(frame %d, %.1f s)' % (SETTLED_FIRST, tcut),
            color=INK2, fontsize=9, va='bottom')
    # direct labels at the line ends
    ax.text(t[-1] + 0.3, m1(t[-1], *p1), 'exp.', color=INK2, fontsize=9, va='center')
    ax.text(t[-1] + 0.3, m2(t[-1], *p2), 'exp. + drift', color=INK2, fontsize=9, va='center')
    ax.set_ylabel('displacement along the drift direction (″)\n1 px = %.4f ″' % PS, color=INK,
                  fontsize=10.5)
    ax.legend(loc='lower right', fontsize=9.5, frameon=False, labelcolor=INK)
    ax.set_xlim(-0.5, t[-1] + 3.0)
    ax.set_title('CalibS: the AM5 settling after the %.2f° slew from the Sun\n' % SLEW_DEG
                 + 'Stage-1 alignment record of frames 1–81 (`s1_calibs_ecl`). Frame 1 began '
                 '%.3f s after the last gain-0 coronal frame;\nthe slew completed inside that '
                 'gap. Perpendicular scatter %.2f ″ rms.\nThe record\'s τ = %.1f s (§3n) was '
                 'fitted on the rate, not the displacement.' % (GAP_S, perp_rms, TAU_RECORD),
                 fontsize=10.5, color=INK, loc='left')
    # residuals
    axr.axhline(0, color=INK2, lw=1)
    axr.plot(t, r1, 'o', ms=5, color=FIT1, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label='pure exponential, rms %.2f ″' % r1.std())
    axr.plot(t, r2, 'o', ms=5, color=FIT2, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label='exponential + drift, rms %.2f ″' % r2.std())
    axr.set_ylabel('residual (″)', color=INK, fontsize=10.5)
    axr.set_xlabel('time from CalibS frame 1 (s)', color=INK, fontsize=10.5)
    axr.legend(loc='upper right', fontsize=9, frameon=False, ncol=2, labelcolor=INK)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.09)
    ChartWriter(OUT, REV).save(fig, 'calibs_settling.png')
    hu_record.publish(['calibs_settling.png'], OUT)


if __name__ == '__main__':
    main()
