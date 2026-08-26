"""M4: the meridian refraction mosaic (docs/REFRACTION_2026.md sections 1, 5).

80 fields, alt 5 deg (south) -> 88.92 (zenith straddle) -> 5 (north), 5 x 6 s each, all
at transit so the altitude rate is zero by design. Per field: 5-frame stack, then stage 2
corrections ON and OFF (quadratic free, the six 08-12 zenith references frozen). The
corrections-ON plate scale should be flat over 5-89 deg of altitude; where it departs is
the standard model's validity boundary at this site. Matched pairs (field k vs 81-k, equal
zenith distance, opposite azimuth) measure azimuthal asymmetry; fields 62/63 straddle the
pier flip and its 4-step FOCUSPOS change, calibrating the N3 focus confound.

Weather is interpolated per field from the site logger (local = UTC+2): at alt 5 deg the
refraction term is ~37,000 ppm and the 0.5 K drift across the 70-minute run is worth tens
of ppm on the lowest fields.
"""
import datetime
import glob
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = r"G:/Leon Aug 2026/2026-08-13/Refraction mosaic"
RD = r"D:/MEE2024 output/MEE_output/refraction"
REFROOT = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
LOGGER = r"I:/Leon location and weather data/leon_temp_press_humid.csv"

SITE = ['--set', 'observation_lat=42.740470', '--set', 'observation_long=-5.613780',
        '--set', 'observation_height=1101', '--set', 'observation_wavelength=0.62']
STAGE1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
          '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
          '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=True',
          '--set', 'remove_edgy_centroids=True', '--set', 'centroid_refine_window=True',
          '--set', 'centroid_window_sigma=2.0']


def load_logger():
    txt = open(LOGGER, 'rb').read().decode('utf-16')
    recs = []
    for line in txt.splitlines():
        p = line.split('\t')
        if len(p) >= 9 and p[0].strip().isdigit():
            dd, mm, yy = p[1].strip().split('/')
            loc = datetime.datetime(int(yy), int(mm), int(dd),
                                    *map(int, p[2].strip().split(':')))
            try:
                recs.append(((loc - datetime.timedelta(hours=2)).timestamp(),
                             float(p[4]), float(p[6]), float(p[8])))
            except ValueError:
                pass
    a = np.array(recs)
    return a


def wx_at(logger, t_utc):
    ts = t_utc.timestamp()
    T = float(np.interp(ts, logger[:, 0], logger[:, 1]))
    RH = float(np.interp(ts, logger[:, 0], logger[:, 2])) / 100.0
    P = float(np.interp(ts, logger[:, 0], logger[:, 3]))
    return T, P, RH


def run(cmd, log_path, timeout=900):
    with open(log_path, 'w') as log:
        try:
            return subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            log.write('\nTIMEOUT\n')
            return -1


def main():
    logger = load_logger()
    refs = sorted(glob.glob(os.path.join(REFROOT, "08-12_Z*.txt")))
    assert len(refs) == 6, f"expected 6 zenith references, found {len(refs)}"
    fields = sorted(d for d in os.listdir(G) if d.startswith("REFR_M"))
    print(f"{len(fields)} mosaic fields; ~{len(fields) * 3 / 60:.1f} h estimated", flush=True)

    for k, fld in enumerate(fields, 1):
        outdir = os.path.join(RD, "mosaic", fld)
        done = glob.glob(os.path.join(outdir, "corr_*", "**", "distortion_results.txt"),
                         recursive=True)
        if len(done) >= 2:
            continue
        os.makedirs(outdir, exist_ok=True)
        frames = sorted(glob.glob(os.path.join(G, fld, "*", "*.fits")))
        if not frames:
            print(f"[{k}/{len(fields)}] {fld}: no frames, skipped", flush=True)
            continue

        zips = glob.glob(os.path.join(outdir, "centroid_data*.zip"))
        if not zips:
            rc = run([sys.executable, '-m', 'mee2024.cli', 'stack', *frames, *STAGE1,
                      '--no-scan', '--no-display', '--quiet', '-o', outdir],
                     os.path.join(outdir, 'stage1.log'))
            zips = glob.glob(os.path.join(outdir, 'centroid_data*.zip'))
            if rc != 0 or not zips:
                print(f"[{k}/{len(fields)}] {fld}: STAGE1 FAILED", flush=True)
                continue

        mids = []
        for f in frames:
            h = fits.getheader(f)
            mids.append(datetime.datetime.fromisoformat(h['DATE-OBS'])
                        + datetime.timedelta(seconds=float(h['EXPTIME']) / 2))
        t_mid = mids[0] + (mids[-1] - mids[0]) / 2
        T, P, RH = wx_at(logger, t_mid.replace(tzinfo=datetime.timezone.utc))

        ok = []
        for corr in ('True', 'False'):
            d2 = os.path.join(outdir, 'corr_on' if corr == 'True' else 'corr_off')
            if glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
                ok.append(corr)
                continue
            os.makedirs(d2, exist_ok=True)
            run([sys.executable, '-m', 'mee2024.cli', 'distortion', zips[0],
                 '--order', 'cubic', '--date-from-header', '--fix-distortion', *refs,
                 '--set', 'distortion_fixed_coefficients=quadratic',
                 '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
                 '--set', f'enable_corrections={corr}',
                 '--set', f'enable_corrections_ref={corr}',
                 '--set', 'enable_gravitational_def=False',
                 '--set', f"observation_time={t_mid.strftime('%H:%M:%S')}",
                 '--set', f'observation_temp={T:.1f}',
                 '--set', f'observation_pressure={P:.1f}',
                 '--set', f'observation_humidity={RH:.3f}', *SITE,
                 '--no-display', '--quiet', '-o', d2],
                os.path.join(d2, 'stage2.log'))
            if glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
                ok.append(corr)
        print(f"[{k}/{len(fields)}] {fld} T={T:.1f}C P={P:.1f}hPa "
              f"{'ok' if len(ok) == 2 else 'PARTIAL ' + str(ok)}", flush=True)
    print("done: mosaic reductions complete", flush=True)


if __name__ == '__main__':
    main()
