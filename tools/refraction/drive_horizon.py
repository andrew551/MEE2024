"""Per-frame reduction ladder for the Leon 2026 horizon calibration sets (M2).

Each horizon block is 45 x 6 s frames tracking one star field as it descends ~0.87 deg,
so per-frame astrometry gives ~45 same-star samples of plate scale against altitude.
Reduced twice per frame -- corrections ON (residual slope = refraction-model error) and
OFF (raw slope, ~600 ppm/deg at alt 9.6 deg, until now known only from the model).

Frames are reduced individually, NOT stacked: a 45-frame stack smears ~0.5 arcsec of
refraction evolution into the frame edges (measured on the 10-frame pilot), and residual
maps stack better in analysis space than in image space.

Auxiliary, eclipse-2026-specific tooling: shells out to `python -m mee2024.cli`, changes
nothing in the package. Resumable -- finished frames are skipped on rerun.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import subprocess
import sys

from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = r"G:/Leon Aug 2026"
OUT = r"D:/MEE2024 output/MEE_output/refraction/perframe"
REFROOT = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"

# Weather per capture window from the site logger (local = UTC+2), extracted 2026-08-26;
# reproduces LEON_2026-08-11.md section 4.4 where they overlap. N1/N2 are at the exact
# FOCUSPOS of their own night's zenith set (17049 / 17041); N3 sits at 17037, 4 steps from
# the 08-12 set -- inside the EAF's ~15-step backlash, carried as a caveat, not a blocker.
WINDOWS = {
    'N1': dict(date='2026-08-11', refs='08-11', T=23.5, P=896.3, RH=0.371),
    'N2': dict(date='2026-08-12', refs='08-12', T=23.7, P=897.5, RH=0.379),
    'N3': dict(date='2026-08-13', refs='08-12', T=21.5, P=898.4, RH=0.500),
}
FIELDS = {'H1': 'H1_eclipse_altaz', 'H2': 'H2_plus2deg_alt', 'H3': 'H3_calfield_sightline'}

SITE = ['--set', 'observation_lat=42.740470', '--set', 'observation_long=-5.613780',
        '--set', 'observation_height=1101', '--set', 'observation_wavelength=0.62']

# The zenith stage-1 regime (HANDOFF_zenith_cubic README) -- these are night frames.
STAGE1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
          '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
          '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=True',
          '--set', 'remove_edgy_centroids=True', '--set', 'centroid_refine_window=True',
          '--set', 'centroid_window_sigma=2.0']


def run(cmd, log_path, timeout=600):
    with open(log_path, 'w') as log:
        try:
            r = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout)
            return r.returncode
        except subprocess.TimeoutExpired:
            log.write('\nTIMEOUT\n')
            return -1


def reduce_frame(frame, outdir, w, tol):
    """Stage 1 on one frame, then stage 2 corrections ON and OFF. Returns result paths."""
    os.makedirs(outdir, exist_ok=True)
    zips = glob.glob(os.path.join(outdir, 'centroid_data*.zip'))
    if not zips:
        rc = run([sys.executable, '-m', 'mee2024.cli', 'stack', frame, *STAGE1,
                  '--no-scan', '--no-display', '--quiet', '-o', outdir],
                 os.path.join(outdir, 'stage1.log'))
        zips = glob.glob(os.path.join(outdir, 'centroid_data*.zip'))
        if rc != 0 or not zips:
            return None
    zip_path = zips[0]

    h = fits.getheader(frame)
    mid = (datetime.datetime.fromisoformat(h['DATE-OBS'])
           + datetime.timedelta(seconds=float(h['EXPTIME']) / 2))
    refs = sorted(f for f in glob.glob(os.path.join(REFROOT, f"{w['refs']}_Z*.txt")))
    assert len(refs) == 6, f"expected 6 zenith references, found {len(refs)}"

    out = {}
    for corr in ('True', 'False'):
        d2 = os.path.join(outdir, 'corr_on' if corr == 'True' else 'corr_off')
        done = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        if done:
            out[corr] = done[0]
            continue
        os.makedirs(d2, exist_ok=True)
        run([sys.executable, '-m', 'mee2024.cli', 'distortion', zip_path,
             '--order', 'cubic', '--date-from-header',
             '--fix-distortion', *refs,
             '--set', 'distortion_fixed_coefficients=quadratic',
             '--set', f'distortion_fit_tol={tol}', '--set', 'max_star_mag_dist=13',
             '--set', f'enable_corrections={corr}', '--set', f'enable_corrections_ref={corr}',
             '--set', 'enable_gravitational_def=False',
             '--set', f"observation_time={mid.strftime('%H:%M:%S')}",
             '--set', f"observation_temp={w['T']}", '--set', f"observation_pressure={w['P']}",
             '--set', f"observation_humidity={w['RH']}", *SITE,
             '--no-display', '--quiet', '-o', d2],
            os.path.join(d2, 'stage2.log'))
        done = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
        out[corr] = done[0] if done else None
    return out


def harvest(csv_path):
    """One CSV row per frame x correction state, from whatever has completed."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, '*', '*', 'f*', 'corr_*', '**',
                                           'distortion_results.txt'), recursive=True)):
        d = json.load(open(f))
        rel = os.path.relpath(f, OUT).replace('\\', '/').split('/')
        rows.append(dict(
            window=rel[0], field=rel[1], frame=rel[2],
            corrections='on' if rel[3] == 'corr_on' else 'off',
            obs_time_utc=d.get('observation_time (UTC)', ''),
            stars_count=d['#stars used'],
            rms_arcsec=round(d['final rms error (arcseconds)'], 4),
            platescale_arcsec_per_px=round(d['platescale (arcseconds/pixel)'], 7),
            se_hc0_ppm=round(d['platescale_relative_uncertainty'] * 1e6, 2),
            alt_deg=round(d['observation alt (degrees)'], 4)
                    if d.get('observation alt (degrees)') is not None else '',
            az_deg=round(d['observation az (degrees)'], 4)
                   if d.get('observation az (degrees)') is not None else '',
            ra_deg=round(d['RA'], 6), dec_deg=round(d['DEC'], 6),
            roll_deg=round(d['ROLL'], 4),
        ))
    if rows:
        with open(csv_path, 'w', newline='') as fp:
            wtr = csv.DictWriter(fp, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
    print(f'{len(rows)} results -> {csv_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--windows', default='N2,N3')
    ap.add_argument('--fields', default='H1,H2,H3')
    ap.add_argument('--tol', type=float, default=2.0,
                    help='distortion_fit_tol in arcsec (2.0: the pilot showed 1.0 rejects '
                         'half the field at this altitude)')
    ap.add_argument('--limit', type=int, default=0, help='frames per field (0 = all)')
    ap.add_argument('--harvest-only', action='store_true')
    args = ap.parse_args()

    csv_path = os.path.join(os.path.dirname(OUT), 'perframe_results.csv')
    if args.harvest_only:
        harvest(csv_path)
        return

    todo = []
    for wname in args.windows.split(','):
        w = WINDOWS[wname]
        for fname in args.fields.split(','):
            folder = os.path.join(G, w['date'], 'Horizon', FIELDS[fname])
            frames = sorted(glob.glob(os.path.join(folder, '*', '*.fits')))
            if args.limit:
                frames = frames[:args.limit]
            todo += [(wname, w, fname, i, fr) for i, fr in enumerate(frames, 1)]
    print(f'{len(todo)} frames to reduce (x2 correction states); '
          f'~{len(todo) * 85 / 3600:.1f} h estimated', flush=True)

    fails = 0
    for k, (wname, w, fname, i, fr) in enumerate(todo, 1):
        outdir = os.path.join(OUT, wname, fname, f'f{i:02d}')
        r = reduce_frame(fr, outdir, w, args.tol)
        ok = r is not None and all(r.values())
        fails += 0 if ok else 1
        print(f'[{k}/{len(todo)}] {wname}/{fname}/f{i:02d} '
              f'{"ok" if ok else "FAILED"}', flush=True)
        if k % 15 == 0 or k == len(todo):
            harvest(csv_path)
    print(f'done: {len(todo) - fails} ok, {fails} failed', flush=True)


if __name__ == '__main__':
    main()
