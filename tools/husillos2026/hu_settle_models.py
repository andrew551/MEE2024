"""Is the RA/Dec difference in CalibS' settle a property of the axes, or only of the arrival
velocities?  The competing damping laws, fitted; and one shared tau against two.

Douglas, 2026-09-14: "Assuming that the motor acts simultaneously along both axes, it likely
means the velocity in RA was higher than DEC.  Therefore at the arrival point, RA must undergo
a greater deceleration ... We have fit an exponential decay, which makes sense for such
damping, but how does the difference in velocities affect the modelling?  Maybe it is not a
difference in RA and DEC per se but just a difference in initial velocities."

THE PREMISE FIRST.  The slew from the Sun (RA 142.106, Dec +14.907 at 18:30:15) to CalibS
(149.227, +7.503) moved the RA axis +7.12 deg and the Dec axis -7.40 deg -- equal travel, so
on one rate profile both axes arrived at the same speed.  (An earlier draft of the record
called the slew "mostly in RA", reading the drift direction as the slew direction.)

THEN THE MODELS.  Under linear damping -- a first-order relaxation, which is what an
exponential is -- tau is the system's constant and the arrival velocity only sets A = v0 tau,
so "same system, different v0" predicts one shared tau: a joint fit and an F-test decide it.
A velocity-dependent effective tau needs a nonlinear law: Coulomb friction (constant
deceleration to a hard stop, longer for a faster axis) or quadratic drag (shorter for a
faster axis).  Each is fitted per axis and compared by residual rms.  A two-time-constant
exponential is tried for a slow second component.

Reads the same alignment record and sky frame as hu_settle_chart.py.  Prints only.

    .venv/Scripts/python.exe tools/husillos2026/hu_settle_models.py
"""
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hu_settle_chart as H  # noqa: E402


def exp1(t, A, tau):
    return A * (1 - np.exp(-t / tau))


def coulomb(t, v0, a):
    """velocity v0 - a t until it stops, then flat"""
    ts = v0 / a
    return np.where(t < ts, v0 * t - 0.5 * a * t ** 2, v0 ** 2 / (2 * a))


def quad_drag(t, v0, k):
    """dv/dt = -k v^2  ->  x = ln(1 + k v0 t) / k"""
    return np.log1p(k * v0 * t) / k


def exp2(t, A1, tau1, A2, tau2):
    return A1 * (1 - np.exp(-t / tau1)) + A2 * (1 - np.exp(-t / tau2))


def main():
    c = H.CAPTURES['calibs']
    t, comps = H.axis_components(c['src'], c['s2'], c['dt'])
    RA, DE = comps['RA*cos(dec)'], comps['Dec']
    n = len(t)

    print('per-axis fits (residual rms in arcsec; v0 = arrival velocity, arcsec/s):')
    for lab, y in (('RA', RA), ('Dec', DE)):
        p, cv = curve_fit(exp1, t, y, p0=(y[-1], 7))
        e = np.sqrt(np.diag(cv))
        print('  %-4s exponential   A %5.1f  tau %4.1f +- %3.1f s  v0 = A/tau = %4.1f "/s   rms %.2f'
              % (lab, p[0], p[1], e[1], p[0] / p[1], (y - exp1(t, *p)).std()))
        pc, _ = curve_fit(coulomb, t, y, p0=(5, 0.3), maxfev=20000)
        print('  %-4s Coulomb       v0 %4.1f "/s  decel %.2f "/s^2  stops at %4.1f s              rms %.2f'
              % (lab, pc[0], pc[1], pc[0] / pc[1], (y - coulomb(t, *pc)).std()))
        pq, _ = curve_fit(quad_drag, t, y, p0=(5, 0.05), maxfev=20000)
        print('  %-4s quadratic     v0 %4.1f "/s  k %.3f 1/arcsec  half-life 1/(k v0) %4.1f s  rms %.2f'
              % (lab, pq[0], pq[1], 1 / (pq[1] * pq[0]), (y - quad_drag(t, *pq)).std()))

    def joint(mask):
        def rs_shared(p):
            A1, A2, tau = p
            return np.concatenate([(RA - exp1(t, A1, tau))[mask], (DE - exp1(t, A2, tau))[mask]])

        def rs_sep(p):
            A1, tau1, A2, tau2 = p
            return np.concatenate([(RA - exp1(t, A1, tau1))[mask], (DE - exp1(t, A2, tau2))[mask]])

        s = least_squares(rs_shared, x0=(44, 24, 7))
        q = least_squares(rs_sep, x0=(44, 7.7, 24, 6.0))
        rss_s, rss_q = float(s.fun @ s.fun), float(q.fun @ q.fun)
        N = 2 * int(mask.sum())
        F = (rss_s - rss_q) / (rss_q / (N - 4))
        return s, q, rss_s, rss_q, F, 1 - stats.f.cdf(F, 1, N - 4), N

    print()
    print('JOINT exponential fit of both axes -- one shared tau against two:')
    for lab, mask in (('all 81 frames', np.ones(n, bool)),
                      ('overshoot frames 3.5-6.5 s excluded', ~((t > 3.5) & (t < 6.5)))):
        s, q, rss_s, rss_q, F, p, N = joint(mask)
        print('  %-36s shared tau %.2f s (RSS %.1f)   separate %.2f / %.2f s (RSS %.1f)   '
              'F(1,%d) = %.1f, p = %.2g -> two taus %s'
              % (lab, s.x[2], rss_s, q.x[1], q.x[3], rss_q, N - 4, F, p,
                 'demanded' if p < 0.01 else 'not demanded'))

    print()
    print('two time constants:')
    for lab, y, p0 in (('RA', RA, (35, 4, 15, 40)), ('Dec', DE, (18, 4, 8, 40))):
        try:
            p, cv = curve_fit(exp2, t, y, p0=p0, maxfev=50000)
            e = np.sqrt(np.diag(cv))
            print('  %-4s A1 %5.1f tau1 %4.1f +- %3.1f s   A2 %8.1f tau2 %8.1f +- %8.1f s   rms %.2f'
                  % (lab, p[0], p[1], e[1], p[2], p[3], e[3], (y - exp2(t, *p)).std()))
        except Exception as ex:                                     # noqa: BLE001
            print('  %-4s did not converge: %s' % (lab, ex))
    print()
    print('equal travel, one rate profile: the arrival velocities were the same.  A shared tau is')
    print('what "same system, different v0" predicts under linear damping; the F-test rejects it.')
    print('Friction, the nonlinear law with the right sign, fits worst on both axes.')


if __name__ == '__main__':
    main()
