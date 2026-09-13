"""Drift against time after a slew, with the settling fits overlaid -- CalibS, and 23_44_06.

Douglas, 2026-09-13: "Plot the CalibS drift against time with both fits overlaid."  Then:
"Now do the same per-frame fit for 23_44_06 after its slew."

WHAT IS PLOTTED.  A stack's stage-1 alignment record (`results.txt['alignment']['shifts_px']`):
the global shift the stacker applied to each frame to align it to frame 1, i.e. where the
field had moved to.  The motion is nearly straight, so it is projected onto its own mean
direction and shown in arcsec (2.2028 "/px); the perpendicular scatter is quoted, not drawn.
Time runs from the capture's first used frame.

CALIBS (`--capture calibs`, the default).  Frames 1-81 at 0.3153 s, beginning 3.156 s after the
last gain-0 coronal frame with the 10.17 deg slew completed inside that gap (record section
3n).  Two least-squares fits on the displacement:

    pure exponential      x(t) = A (1 - exp(-t / tau))
    exponential + drift   x(t) = A (1 - exp(-t / tau)) + v t

The record's tau = 9.2 s was fitted on the RATE over 15 windows, not the displacement, so the
pure-exponential displacement fit need not return it; the legend carries what each fit found.

23_44_06 (`--capture h10_g125d`).  Frames 1-49 at 1.3163 s, the first capture after the slew
from pointing B (Dec +23.28) to pointing C (Dec +37.55), 14.3 deg almost entirely in
declination.  The slew is dated from 23_42_43's own frames: frame 47 (opened 21:43:44.6)
aligned to frame 0 and frame 48 (21:43:45.9) did not, so the slew began at ~21:43:45.6; frame 1
of 23_44_06 opened at 21:44:07.0, 21.4 s later, and the slew itself took ~2.5-3 s at the AM5's
~6 deg/s.  So this capture begins ~18-19 s after the mount stopped.  The two exponential models
DO NOT CONVERGE on it -- the displacement is a straight line (2.5 "/min, 0.36 " rms) with no
curvature to fit -- so what is drawn is that line and, over it, what each CalibS fit PREDICTS
should still be happening 18.5 s after a slew: the pure exponential's remaining settle
A exp(-g/tau)(1 - exp(-t/tau)) and the drift model's remaining settle plus its v t.  Neither
prediction is what the data show, and the chart is the record of that.

Chart conventions follow the dataviz reference palette: data in the secondary text ink, the
fitted or predicted curves in categorical slots 1-3 (blue, orange, aqua -- the three slots
that validate all-pairs), 2 px fitted lines, thinner predicted ones, >= 8 px markers with a
surface ring, hairline solid grid, one y axis per panel (arcsec; the pixel conversion is in the
label, never a second axis), a legend because there are several series, text never in a
series colour.

Writes calibs/<name>_settling.png (+ chart_versions/) and publishes through hu_record.publish
so an earlier revision is superseded, never overwritten.

    .venv/Scripts/python.exe tools/husillos2026/hu_settle_chart.py [--capture calibs|h10_g125d]
"""
import argparse
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
OUT = os.path.join(HUS, 'calibs')
REV = os.environ.get('HU_REV', 'rev01')
PS = 2.2028                 # arcsec per px, the cell's plate scale
TAU_RECORD = 9.2            # s, section 3n, fitted on the rate
SETTLED_FIRST = 21          # first frame of the settled CalibS stack (hu_calibs.SETTLED_FIRST)

CAPTURES = {
    'calibs': dict(name='calibs', label='CalibS', src=os.path.join(HUS, 'calibs', 's1_calibs_ecl'),
                   s2=os.path.join(HUS, 'calibs', 's2_calibs_ecl'), dt=0.3153, slew_deg=10.17,
                   gap_note='Frame 1 began 3.156 s after the last gain-0 coronal frame;\nthe '
                            '10.17° slew from the Sun completed inside that gap. ',
                   axis_note='The slew from the Sun was 10.17°, mostly in RA. Each component fitted alone with a pure exponential; the rate over the last 5 s\nis measured, not fitted. Dec is at the tracking floor by 20 s; RA is still creeping at 25 s.'),
    'h10_g125d': dict(name='h10_g125d', label='23_44_06',
                      src=os.path.join(HUS, 'horizon', 's1d_h10_g125d'),
                      s2=os.path.join(HUS, 'horizon', 's2d_h10_g125d'), dt=1.3163, slew_deg=14.3,
                      gap_s=18.5,
                      gap_note='The 14.3° slew (pointing B → C, almost all in Dec)\nbegan at '
                               '21:43:45.6 by 23_42_43\u2019s own frames; frame 1 opened 21.4 s '
                               'later, ~18–19 s after the mount stopped.',
                      axis_note='First capture after the 14.3° slew from pointing B to C, almost all in Dec; frame 1 opened 21.4 s after the slew began,\n~18–19 s after the mount stopped. Neither component has an exponential to fit: both are lines at the tracking floor from frame 1.'),
}
#: the CalibS displacement fits, carried so 23_44_06 can be drawn against their predictions
CALIBS_FITS = dict(pure=(50.10, 7.29), drift=(33.94, 4.59, 0.667))

# dataviz reference palette, light mode
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'


def displacement(src, dt):
    z = glob.glob(os.path.join(src, 'centroid_data*.zip'))[0]
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z).open('results.txt'), encoding='utf-8',
                                   errors='replace'))
    s = np.array(r['alignment']['shifts_px'], float)
    d = s - s[0]
    u = d[-1] / np.linalg.norm(d[-1])
    along = d @ u * PS
    perp = d - np.outer(d @ u, u)
    return np.arange(len(s)) * dt, along, float(np.hypot(*perp.T).std() * PS)


def axis_components(src, s2dir, dt):
    """The alignment drift resolved onto the sky: (time, {'RA*cos(dec)': arcsec, 'Dec': arcsec}).

    The sky frame is the affine of the capture's own matched stars (its stage-2
    CATALOGUE_MATCHED_ERRORS), the construction record_charts.SkyFrame uses; inline because
    only the two direction vectors are needed."""
    z = glob.glob(os.path.join(src, 'centroid_data*.zip'))[0]
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z).open('results.txt'), encoding='utf-8',
                                   errors='replace'))
    s = np.array(r['alignment']['shifts_px'], float)
    d = s - s[0]
    zz = glob.glob(os.path.join(s2dir, '**', 'distortion_data*.zip'), recursive=True)[0]
    zf = zipfile.ZipFile(zz)
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    import pandas as pd
    t = pd.read_csv(zf.open(n))
    t.columns = [c.strip() for c in t.columns]
    ra, de = t['RA(catalog)'].values, t['DEC(catalog)'].values
    c = np.cos(np.radians(de.mean()))
    X = np.column_stack([t['px'].values, t['py'].values, np.ones(len(t))])
    ca, *_ = np.linalg.lstsq(X, (ra - ra.mean()) * c * 3600, rcond=None)
    cd, *_ = np.linalg.lstsq(X, (de - de.mean()) * 3600, rcond=None)
    comps = {'RA*cos(dec)': ca[0] * d[:, 0] + ca[1] * d[:, 1],
             'Dec': cd[0] * d[:, 0] + cd[1] * d[:, 1]}
    return np.arange(len(s)) * dt, comps


#: the AM5's steady drift with tracking on, measured at the zenith (2.48 "/min) and again on
#: 23_44_06 at 15 deg (2.5 "/min), both in DECLINATION and nil in RA -- record section 3n
TRACKING_FLOOR_DEC = 2.5 / 60.0     # arcsec per second


def chart_by_axis(c):
    """Douglas, 2026-09-13: "Plot the CalibS drift split by RA and Dec axis."  2026-09-14: "Let's
    do the RA and DEC drift for the 23_44_06 field."  The two sky components of a capture's
    alignment shifts on one time axis, each fitted alone -- a pure exponential where the data
    constrain one, a straight line where they do not (the 1-sigma error on tau exceeding tau
    is the test; 23_44_06's components are lines) -- with the mount's Dec tracking floor drawn
    as the slope a settled axis shows, both signs, since a Dec drift can run either way."""
    tt, comps = axis_components(c['src'], c['s2'], c['dt'])
    fig, ax, axr = figure()
    tf = np.linspace(0, tt[-1], 400)
    late = tt >= tt[-1] - 5.0
    ends = {}
    for (lab, y), col, short in zip(comps.items(), (S1, S2), ('RA', 'Dec')):
        v_late = np.polyfit(tt[late], y[late], 1)[0]
        model = None
        try:
            p, cov = curve_fit(m1, tt, y, p0=(0.9 * y[-1] if abs(y[-1]) > 1 else 1.0, 8),
                               maxfev=20000)
            e = np.sqrt(np.diag(cov))
            if np.isfinite(e[1]) and e[1] < p[1]:
                model = ('exp', p, e)
        except Exception:                                           # noqa: BLE001
            model = None
        if model:
            _, p, e = model
            fit_y, fit_f = m1(tt, *p), m1(tf, *p)
            desc = 'τ = %.1f ± %.1f s' % (p[1], e[1])
        else:
            lin = np.polyfit(tt, y, 1)
            fit_y, fit_f = np.polyval(lin, tt), np.polyval(lin, tf)
            desc = 'no exponential to fit: a line at %+.1f ″/min' % (lin[0] * 60)
        res = y - fit_y
        ax.plot(tt, y, 'o', ms=5, color=col, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3)
        ax.plot(tf, fit_f, '-', lw=2, color=col, solid_capstyle='round', zorder=4,
                label='%s:  %+.1f ″ total,  %s,  last 5 s %+.0f ″/min'
                      % (lab, y[-1], desc, v_late * 60))
        axr.plot(tt, res, 'o', ms=4.5, color=col, markeredgecolor=SURFACE, markeredgewidth=1.0,
                 label='%s residual, rms %.2f ″' % (short, res.std()))
        ends[short] = y[-1]
    # the tracking floor, both signs, as slopes from the origin: what a settled axis does
    for sign, lab in ((+1, 'the AM5 tracking floor: ±%.1f ″/min, the drift of a settled axis'
                       % (TRACKING_FLOOR_DEC * 60)), (-1, None)):
        ax.plot(tf, sign * TRACKING_FLOOR_DEC * tf, '-', lw=1, color=INK2, alpha=0.7,
                zorder=2, label=lab)
    if c['name'] == 'calibs':
        tcut = (SETTLED_FIRST - 1) * c['dt']
        for a_ in (ax, axr):
            a_.axvline(tcut, color=INK2, lw=1, alpha=0.6)
        ax.text(tcut + 0.25, 2.0, 'settled stack begins\n(frame %d, %.1f s)'
                % (SETTLED_FIRST, tcut), color=INK2, fontsize=9, va='bottom')
    for short, y_end in ends.items():
        ax.text(tt[-1] + 0.02 * tt[-1], y_end, short, color=INK2, fontsize=9, va='center')
    ax.set_ylabel('displacement on the sky (″)\n1 px = %.4f ″' % PS, color=INK, fontsize=10.5)
    # upper left: the only corner the curves and the floor lines leave empty
    ax.legend(loc='upper left', fontsize=9.5, frameon=False, labelcolor=INK)
    ax.set_xlim(-0.02 * tt[-1], tt[-1] * 1.12)
    ax.set_title('%s: the drift split into its RA and Dec components\n' % c['label']
                 + 'Stage-1 alignment record (%s), resolved onto the sky through the affine of '
                 'the capture\u2019s own matched stars.\n' % os.path.basename(c['src'])
                 + c['axis_note'],
                 fontsize=10.5, color=INK, loc='left')
    axr.axhline(0, color=INK2, lw=1)
    axr.set_ylabel('residual (″)', color=INK, fontsize=10.5)
    axr.set_xlabel('time from %s frame 1 (s)' % c['label'], color=INK, fontsize=10.5)
    axr.legend(loc='upper right', fontsize=9, frameon=False, ncol=2, labelcolor=INK)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.09)
    return fig


def by_axis(src, s2dir, dt):
    """The same drift resolved into RA and Dec on the sky, each fitted alone.

    Douglas, 2026-09-13: "So do the last two charts agree with each other or not?"  At face
    value no -- CalibS is still moving at ~36 "/min 20-25 s after its slew and 23_44_06 is at
    the 2.5 "/min tracking floor 18 s after its.  Split by axis they agree: the AM5's steady
    drift is in Dec and nil in RA; CalibS' Dec component is at that floor by 20 s exactly as
    23_44_06's Dec-only slew is, and what is still creeping at 25 s in CalibS is the RA axis,
    which 23_44_06 never exercised.  The sky frame is the affine of the capture's own matched
    stars (its stage-2 CATALOGUE_MATCHED_ERRORS), the same construction record_charts.SkyFrame
    uses; done inline here because only the two direction vectors are needed.
    """
    tt, comps = axis_components(src, s2dir, dt)
    late = tt >= tt[-1] - 5.0
    print('drift resolved onto the sky, per axis (%d frames, %.1f s):' % (len(tt), tt[-1]))
    for lab, y in comps.items():
        try:
            p, cov = curve_fit(m1, tt, y, p0=(0.9 * y[-1] if abs(y[-1]) > 1 else 1.0, 8))
            fit = 'pure-exp tau %.1f +- %.1f s' % (p[1], np.sqrt(cov[1, 1]))
        except Exception:                                           # noqa: BLE001
            fit = 'pure-exp fit did not converge (a straight line)'
        print('   %-12s total %+6.1f "   rate first 3 s %+5.2f "/s   rate last 5 s %+5.2f "/s = '
              '%+4.0f "/min   %s'
              % (lab, y[-1], np.polyfit(tt[tt < 3], y[tt < 3], 1)[0],
                 np.polyfit(tt[late], y[late], 1)[0], np.polyfit(tt[late], y[late], 1)[0] * 60,
                 fit))


def m1(t, A, tau):
    return A * (1 - np.exp(-t / tau))


def m2(t, A, tau, v):
    return A * (1 - np.exp(-t / tau)) + v * t


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1.0, linestyle='-')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)


def figure():
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=(3.2, 1.3), hspace=0.08))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    style(axr)
    return fig, ax, axr


def chart_calibs(c):
    t, x, perp_rms = displacement(c['src'], c['dt'])
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

    fig, ax, axr = figure()
    ax.plot(t, x, 'o', ms=6, color=INK2, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label='field displacement, one point per %.4f s frame' % c['dt'], zorder=3)
    ax.plot(tt, m1(tt, *p1), '-', lw=2, color=S1, solid_capstyle='round',
            label='pure exponential:  A = %.1f ″,  τ = %.1f ± %.1f s' % (p1[0], p1[1], e1[1]),
            zorder=4)
    ax.plot(tt, m2(tt, *p2), '-', lw=2, color=S2, solid_capstyle='round',
            label='exponential + drift:  A = %.1f ″,  τ = %.1f ± %.1f s,  drift %.0f ″/min'
                  % (p2[0], p2[1], e2[1], p2[2] * 60), zorder=4)
    tcut = (SETTLED_FIRST - 1) * c['dt']
    for a in (ax, axr):
        a.axvline(tcut, color=INK2, lw=1, alpha=0.6)
    ax.text(tcut + 0.25, 1.5, 'settled stack begins\n(frame %d, %.1f s)' % (SETTLED_FIRST, tcut),
            color=INK2, fontsize=9, va='bottom')
    ax.text(t[-1] + 0.3, m1(t[-1], *p1), 'exp.', color=INK2, fontsize=9, va='center')
    ax.text(t[-1] + 0.3, m2(t[-1], *p2), 'exp. + drift', color=INK2, fontsize=9, va='center')
    ax.set_ylabel('displacement along the drift direction (″)\n1 px = %.4f ″' % PS, color=INK,
                  fontsize=10.5)
    ax.legend(loc='lower right', fontsize=9.5, frameon=False, labelcolor=INK)
    ax.set_xlim(-0.5, t[-1] + 3.0)
    ax.set_title('CalibS: the AM5 settling after the %.2f° slew from the Sun\n' % c['slew_deg']
                 + 'Stage-1 alignment record of frames 1–81 (`s1_calibs_ecl`). ' + c['gap_note']
                 + 'Perpendicular scatter %.2f ″ rms.\nThe record\u2019s τ = %.1f s (§3n) was '
                 'fitted on the rate, not the displacement.' % (perp_rms, TAU_RECORD),
                 fontsize=10.5, color=INK, loc='left')
    axr.axhline(0, color=INK2, lw=1)
    axr.plot(t, r1, 'o', ms=5, color=S1, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label='pure exponential, rms %.2f ″' % r1.std())
    axr.plot(t, r2, 'o', ms=5, color=S2, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label='exponential + drift, rms %.2f ″' % r2.std())
    axr.set_ylabel('residual (″)', color=INK, fontsize=10.5)
    axr.set_xlabel('time from CalibS frame 1 (s)', color=INK, fontsize=10.5)
    axr.legend(loc='upper right', fontsize=9, frameon=False, ncol=2, labelcolor=INK)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.09)
    return fig


def chart_after_slew(c):
    """A capture that began well after its slew: the linear drift that fits it, and what the
    two CalibS models predict should still have been happening."""
    t, x, perp_rms = displacement(c['src'], c['dt'])
    g = c['gap_s']
    lin = np.polyfit(t, x, 1)
    res = x - np.polyval(lin, t)
    A1, tau1 = CALIBS_FITS['pure']
    A2, tau2, v2 = CALIBS_FITS['drift']
    pred1 = A1 * np.exp(-g / tau1) * (1 - np.exp(-t / tau1))
    pred2 = A2 * np.exp(-g / tau2) * (1 - np.exp(-t / tau2)) + v2 * t
    for lab, m, p0 in (('pure exponential', m1, (3, 8)), ('exponential + drift', m2, (2, 8, 0.03))):
        try:
            p, cov = curve_fit(m, t, x, p0=p0, maxfev=20000)
            print('%-22s ' % lab + '  '.join('%.3g +- %.3g' % (a, b) for a, b in
                                             zip(p, np.sqrt(np.diag(cov)))) + '  (unconstrained)')
        except Exception as ex:                                     # noqa: BLE001
            print('%-22s did not converge (%s)' % (lab, ex))
    print('linear drift %.3f "/s = %.1f "/min, residual rms %.2f "; perpendicular scatter %.2f "; '
          'total %.2f " (%.2f px) over %.1f s'
          % (lin[0], lin[0] * 60, res.std(), perp_rms, x[-1], x[-1] / PS, t[-1]))
    print('CalibS models %.1f s after a slew predict: pure exp %.1f " more settle; exp+drift %.1f " '
          'more settle + %.1f " of drift over this capture'
          % (g, A1 * np.exp(-g / tau1), A2 * np.exp(-g / tau2), v2 * t[-1]))

    fig, ax, axr = figure()
    tt = np.linspace(0, t[-1], 400)
    ax.plot(t, x, 'o', ms=6, color=INK2, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label='field displacement, one point per %.4f s frame' % c['dt'], zorder=3)
    ax.plot(tt, np.polyval(lin, tt), '-', lw=2, color=S3, solid_capstyle='round',
            label='linear drift fitted here:  %.1f ″/min  (the AM5 tracks at 2.5 ″/min overhead)'
                  % (lin[0] * 60), zorder=4)
    ax.plot(tt, A1 * np.exp(-g / tau1) * (1 - np.exp(-tt / tau1)), '-', lw=1.4, color=S1,
            alpha=0.9, label='predicted from CalibS, pure exponential (τ = %.1f s): %.1f ″ of '
                             'settle still to come' % (tau1, A1 * np.exp(-g / tau1)), zorder=4)
    ax.plot(tt, A2 * np.exp(-g / tau2) * (1 - np.exp(-tt / tau2)) + v2 * tt, '-', lw=1.4,
            color=S2, alpha=0.9,
            label='predicted from CalibS, exponential + drift (τ = %.1f s, %.0f ″/min)'
                  % (tau2, v2 * 60), zorder=4)
    ax.text(t[-1] + 0.6, pred1[-1], 'pure exp.', color=INK2, fontsize=9, va='center')
    ax.text(t[-1] + 0.6, pred2[-1], 'exp. + drift', color=INK2, fontsize=9, va='center')
    ax.text(t[-1] + 0.6, np.polyval(lin, t[-1]), 'fitted', color=INK2, fontsize=9, va='center')
    ax.set_ylabel('displacement along the drift direction (″)\n1 px = %.4f ″' % PS, color=INK,
                  fontsize=10.5)
    ax.legend(loc='upper left', fontsize=9.5, frameon=False, labelcolor=INK)
    ax.set_xlim(-1, t[-1] + 9.0)
    ax.set_title('%s: the AM5 %.1f s after a %.1f° slew — nothing left to settle\n'
                 % (c['label'], g, c['slew_deg'])
                 + 'Stage-1 alignment record of frames 1–49 (`s1d_h10_g125d`, tracked). '
                 + c['gap_note'] + '\nPerpendicular scatter %.2f ″ rms. The exponential fits do '
                 'not converge on this capture: the displacement is a straight line.' % perp_rms,
                 fontsize=10.5, color=INK, loc='left')
    axr.axhline(0, color=INK2, lw=1)
    axr.plot(t, res, 'o', ms=5, color=S3, markeredgecolor=SURFACE, markeredgewidth=1.2,
             label='residual from the linear drift, rms %.2f ″' % res.std())
    axr.set_ylabel('residual (″)', color=INK, fontsize=10.5)
    axr.set_xlabel('time from %s frame 1 (s)' % c['label'], color=INK, fontsize=10.5)
    axr.legend(loc='upper right', fontsize=9, frameon=False, labelcolor=INK)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.09)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', default='calibs', choices=sorted(CAPTURES))
    ap.add_argument('--by-axis', action='store_true',
                    help='draw the RA and Dec components separately instead of the total')
    a = ap.parse_args()
    c = CAPTURES[a.capture]
    if a.by_axis:
        fig = chart_by_axis(c)
    else:
        fig = chart_calibs(c) if a.capture == 'calibs' else chart_after_slew(c)
    by_axis(c['src'], c['s2'], c['dt'])
    name = '%s_settling%s.png' % (c['name'], '_by_axis' if a.by_axis else '')
    ChartWriter(OUT, REV).save(fig, name)
    hu_record.publish([name], OUT)


if __name__ == '__main__':
    main()
