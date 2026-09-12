"""One horizon capture reduced FRAME BY FRAME, the Leon construction, so the 3x can be split.

Douglas, 2026-09-12: "Do the per-frame reduction of one horizon capture."

WHY.  Section 3v found Husillos' 10-degree night residual is 3x Leon's at the bright end
(0.827 " against 0.279 " once model transfer is removed) and that it is structure, not noise.
One construction difference was left unseparated: Leon's map is a per-star MEDIAN over ~45
separately solved 4 s frames (tools/refraction/drive_horizon.py, then
tools/step3_atmosphere_maps.clip_medians), so each frame's own low-order atmospheric
distortion is absorbed by that frame's quadratic before the median; Husillos' map is one
100 s stack with one quadratic at the end.  This tool does the Leon operation on a Husillos
capture, so the same field is reduced both ways and the difference is the construction.

WHICH CAPTURE.  23_34_38 (h10_g125a): the 10-degree pointing with the most stars (345 on the
deep stack), gain 125 so a single 1 s frame carries enough signal to solve on its own.  Frames
1-99; frame 0 is skipped everywhere in this cell (docs/HUSILLOS2026_ZENITH.md).

THE RECIPE, Leon's rung for rung.  Stage 1 on ONE frame (`--frames i-i`), with the synthetic
hot-pixel dark; stage 2 against the same zenith quintic reference at the same rung as the
stack (cubic and above frozen, quadratic and scale free), corrections ON at the FRAME'S OWN
mid-time -- the SER trailer's per-frame timestamp plus half the 1.000 s exposure, exactly
Leon's DATE-OBS + EXPTIME/2.  Then per star: the median over frames of its residual, a star
needing 20 frames; the field's median vector removed; the 3 x MAD clip with a 2.5 " floor;
rms, the alt/az split, and the magnitude bins of hu_maps_bymag.py.

THE STAGE-1 REGIME is chosen by what solves, and both candidates are documented conventions:
`s1` is the zenith star-field preset, which is ALSO Leon's per-frame regime
(drive_horizon.STAGE1: thresh 5.0, min_area 4, sigma_subtract 3.0) and so the like-for-like
choice; `deep` is the eclipse blocks' detection, which on a single read-noise-limited frame
admits thousands of noise candidates and costs ~160 s a frame.  The probe on frame 50 decides
(see the record); the regime used is written into every output directory name.

Resumable, and meant to be run as several workers on disjoint frame ranges:

    .venv/Scripts/python.exe tools/husillos2026/hu_perframe.py run --frames 1-25 [--regime s1]
    .venv/Scripts/python.exe tools/husillos2026/hu_perframe.py medians [--regime s1]
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
import hu_horizon_reduce as H  # noqa: E402
import ser_track  # noqa: E402

#: Chosen on the command line (--capture <tag>, a tag from hu_horizon_reduce.CAPTURES).  The
#: first run was 23_34_38 alone; its result -- and the stack's alignment record it sent us
#: back to -- showed the first three `10 deg` captures were UNTRACKED (770-800 px of drift
#: over 99 frames, 7.8 px/frame = the sidereal rate at Dec +29.6), so every stack of them is
#: smeared and the per-frame median is the only valid reduction.  23_31_59 needs it too.
CAPTURE = None
EXPOSURE_S = 1.000
FIRST, LAST = 1, 99
MIN_FRAMES = 20                       # step3_atmosphere_maps.clip_medians
OUT = None


def select(tag):
    global CAPTURE, OUT
    for t, folder, name, gain, a, b in H.CAPTURES:
        if t == tag:
            CAPTURE = (t, folder, name)
            OUT = os.path.join(H.OUT, 'perframe_' + t)
            os.makedirs(OUT, exist_ok=True)
            return
    raise SystemExit('unknown capture tag %r' % tag)
BINS = [(4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]   # floor_vs_sampling.py's
SITE_LL = (42.09293, -4.52702, 743.0)


def ser_path():
    return os.path.join(H.SRC, CAPTURE[1], CAPTURE[2] + '.ser')


def frame_midtimes():
    """UTC mid-exposure per frame index, from the SER trailer."""
    p = ser_path()
    hdr = ser_track.read_header(p)
    ts = ser_track.read_timestamps(p, hdr)
    if ts is None:
        raise SystemExit('no per-frame timestamps in ' + p)
    return [t + datetime.timedelta(seconds=EXPOSURE_S / 2) for t in ts]


def frame_dir(regime, i):
    return os.path.join(OUT, regime, 'f%03d' % i)


def reduce_frame(i, regime, tmid):
    d1 = os.path.join(frame_dir(regime, i), 's1')
    if not H.czip(d1):
        H.run([H.PY, '-m', 'mee2024.cli', 'stack', ser_path(), '--frames', '%d-%d' % (i, i),
               '--dark', H.DARK, *(H.DEEP if regime == 'deep' else H.S1),
               '--no-display', '--quiet', '-o', d1], os.path.join(d1, 'stage1.log'))
    z = H.czip(d1)
    if not z:
        return None
    d2 = os.path.join(frame_dir(regime, i), 's2')
    if not H.results(d2):
        H.run([H.PY, '-m', 'mee2024.cli', 'distortion', z, '--order', 'quintic',
               '--set', 'distortion_reference_files=' + H.refzip(),
               '--set', 'distortion_fixed_coefficients=quadratic',
               '--set', 'distortion_free_scale=True',
               '--set', 'distortion_fit_tol_initial=20.0', '--set', 'distortion_fit_tol=3.0',
               '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
               *H.SITE, '--set', 'observation_time=' + tmid.strftime('%H:%M:%S'),
               '--no-display', '--quiet', '-o', d2], os.path.join(d2, 'stage2.log'))
    return H.results(d2)


def do_run(args):
    a, b = (int(x) for x in args.frames.split('-'))
    mids = frame_midtimes()
    for i in range(a, b + 1):
        j = reduce_frame(i, args.regime, mids[i])
        print('f%03d %s  %s' % (i, mids[i].strftime('%H:%M:%S.%f')[:12],
                                 ('%3d stars, rms %.3f ", ps %.7f' % (
                                     j['#stars used'], j['final rms error (arcseconds)'],
                                     j['platescale (arcseconds/pixel)'])) if j else 'FAILED'),
              flush=True)


# ---------------------------------------------------------------- the medians
def altaz_basis(regime, i, lat, lon, height):
    """Increasing-altitude and -azimuth unit vectors in sensor px from one solved frame,
    the cell-1 construction (regressed on CATALOGUE positions)."""
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    d2 = os.path.join(frame_dir(regime, i), 's2')
    z = glob.glob(os.path.join(d2, '**', 'distortion_data*.zip'), recursive=True)[0]
    zf = zipfile.ZipFile(z)
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n))
    t.columns = [c.strip() for c in t.columns]
    j = H.results(d2)
    when = Time(j['observation_date'] + 'T' + j['observation_time (UTC)'], scale='utc')
    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    aa = SkyCoord(t['RA(catalog)'].values * u.deg, t['DEC(catalog)'].values * u.deg
                  ).transform_to(AltAz(obstime=when, location=site))
    alt = aa.alt.deg
    azc = aa.az.deg * np.cos(np.radians(alt.mean()))
    A = np.column_stack([azc - azc.mean(), alt - alt.mean(), np.ones(len(t))])
    cx, *_ = np.linalg.lstsq(A, t['px'].values.astype(float), rcond=None)
    cy, *_ = np.linalg.lstsq(A, t['py'].values.astype(float), rcond=None)
    v_az, v_alt = np.array([cx[0], cy[0]]), np.array([cx[1], cy[1]])
    return v_alt / np.linalg.norm(v_alt), v_az / np.linalg.norm(v_az), float(alt.mean())


def by_bin(mag, dx, dy, lo, hi, min_stars=8):
    k = (mag >= lo) & (mag < hi)
    if k.sum() < min_stars:
        return np.nan, int(k.sum())
    return float(np.sqrt(np.mean(dx[k] ** 2 + dy[k] ** 2))), int(k.sum())


def do_medians(args):
    regime = args.regime
    files = sorted(glob.glob(os.path.join(OUT, regime, 'f*', 's2', '**', 'TWOD_RESIDUALS.csv'),
                             recursive=True))
    solved = []
    for f in files:
        m = re.search(r'[\\/]f(\d{3})[\\/]', f)
        if m:
            solved.append(int(m.group(1)))
    print('%d frames solved of %d (regime %s)' % (len(files), LAST - FIRST + 1, regime))
    if not files:
        return
    acc = {}
    per_frame = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d = d[d['magV'] <= BINS[-1][1]]
        per_frame.append(float(np.sqrt(np.mean(d.dx_arcsec ** 2 + d.dy_arcsec ** 2))))
        for sid, px, py, dx, dy, mag in zip(d.ID, d.px, d.py, d.dx_arcsec, d.dy_arcsec, d.magV):
            acc.setdefault(sid, []).append((px, py, dx, dy, mag))
    ids = [k for k, v in acc.items() if len(v) >= MIN_FRAMES]
    P = np.array([[np.median([q[c] for q in acc[i]]) for c in range(4)] + [acc[i][0][4],
                                                                              len(acc[i])]
                  for i in ids])
    px, py, qx, qy, mag, nfr = P.T
    qx, qy = qx - np.median(qx), qy - np.median(qy)
    m = np.hypot(qx, qy)
    lim = max(3 * 1.4826 * np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
    good = m < lim
    px, py, qx, qy, mag, nfr = (v[good] for v in (px, py, qx, qy, mag, nfr))
    v_alt, v_az, alt = altaz_basis(regime, solved[len(solved) // 2], *SITE_LL)
    q_alt, q_az = qx * v_alt[0] + qy * v_alt[1], qx * v_az[0] + qy * v_az[1]
    rms = float(np.sqrt(np.mean(qx ** 2 + qy ** 2)))
    va, ha = float(np.sqrt(np.mean(q_alt ** 2))), float(np.sqrt(np.mean(q_az ** 2)))
    print('per-frame single-fit rms: median %.3f " (range %.3f-%.3f) -- each frame alone'
          % (np.median(per_frame), min(per_frame), max(per_frame)))
    print('per-star MEDIAN over frames (>= %d frames, %d stars, %d clipped, mean %.0f frames/star):'
          % (MIN_FRAMES, len(px), int((~good).sum()), nfr.mean()))
    print('   rms %.3f "   vertical %.3f   horizontal %.3f   V/H %.2f   alt %.2f deg'
          % (rms, va, ha, va / max(ha, 1e-9), alt))
    row = [by_bin(mag, qx, qy, lo, hi) for lo, hi in BINS]
    print('   by G: ' + '  '.join('%d-%d %s' % (b[0], b[1], ('%.3f/%d' % (v, n)) if v == v else '-')
                                   for (v, n), b in zip(row, BINS)))
    bright = float(np.nanmean([v for (v, n), b in zip(row, BINS) if 8 <= b[0] < 10]))
    faint = float(np.nanmean([v for (v, n), b in zip(row, BINS) if b[0] >= 11]))
    print('   bright (8-10) %.3f "   faint (11-13) %.3f "   f/b %.2f' % (bright, faint, faint / bright))
    print()
    print('beside it (record section 3v, bright end): the same capture as ONE STACK, frozen '
          'model 1.202-1.244 " (2-field mean 1.223), whole quintic free 0.827 "; Leon horizon '
          '0.279 ".')
    pd.DataFrame(dict(ID=np.array(ids)[good], px=px, py=py, qx=qx, qy=qy, magV=mag,
                      n_frames=nfr)).to_csv(os.path.join(OUT, regime + '_medians.csv'),
                                            index=False)
    # the same table in TWOD_RESIDUALS.csv's columns, so hu_atmosphere.py and
    # hu_atmos_maps.py can take the per-frame medians exactly where they take a stack's
    # stage-2 residuals -- the untracked captures have no valid stack to offer them
    pd.DataFrame(dict(px=px, py=py, dx_px=qx / 2.2028, dy_px=qy / 2.2028, dx_arcsec=qx,
                      dy_arcsec=qy, error_arcsec=np.hypot(qx, qy), radius_px=np.nan, magV=mag,
                      ID=np.array(ids)[good])).to_csv(
        os.path.join(OUT, regime + '_TWOD_RESIDUALS.csv'), index=False)
    json.dump(dict(regime=regime, frames_solved=len(files), stars=int(len(px)), rms=rms,
                   vertical=va, horizontal=ha, v_over_h=va / max(ha, 1e-9), alt=alt,
                   bright_G8_10=bright, faint_G11_13=faint,
                   per_frame_rms_median=float(np.median(per_frame))),
              open(os.path.join(OUT, regime + '_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'medians'])
    ap.add_argument('--capture', default='h10_g125a', help='a tag from hu_horizon_reduce.CAPTURES')
    ap.add_argument('--frames', default='%d-%d' % (FIRST, LAST))
    ap.add_argument('--regime', default='s1', choices=['s1', 'deep'])
    args = ap.parse_args()
    select(args.capture)
    {'run': do_run, 'medians': do_medians}[args.cmd](args)
