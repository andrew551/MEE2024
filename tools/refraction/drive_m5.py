"""M5: the step-3 rehearsal (docs/REFRACTION_2026.md section 5).

Reproduces the real reduction chain at night, where the deflection is absent and truth is
the catalogue: same-night zenith cubic -> step-2-like fit on H3 (the CAL_piLeo analogue;
a mid-block 10-frame stack, corrections ON, quadratic free) -> step-3-like fit on every H1
frame with H3's result as the frozen reference and only the pointing constant free
(distortion_fit_tol=999, as section 16 specifies for the eclipse field: the residual is
the signal). Every arcsecond of structure in the H1 residuals is the class of error the
eclipse reduction will inherit. Stage 1 is reused from the M2 ladder; only stage-2 runs.
"""
import datetime
import glob
import json
import os
import subprocess
import sys

from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = r"G:/Leon Aug 2026"
RD = r"D:/MEE2024 output/MEE_output/refraction"
REFROOT = r"D:/MEE2024 output/MEE_output/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"

WINDOWS = {
    'N1': dict(date='2026-08-11', refs='08-11', T=23.5, P=896.3, RH=0.371),
    'N2': dict(date='2026-08-12', refs='08-12', T=23.7, P=897.5, RH=0.379),
    'N3': dict(date='2026-08-13', refs='08-12', T=21.5, P=898.4, RH=0.500),
}
SITE = ['--set', 'observation_lat=42.740470', '--set', 'observation_long=-5.613780',
        '--set', 'observation_height=1101', '--set', 'observation_wavelength=0.62']
STAGE1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=False',
          '--set', 'centroid_gaussian_thresh=5.0', '--set', 'min_area=4',
          '--set', 'sigma_subtract=3.0', '--set', 'delete_saturated_blob=True',
          '--set', 'remove_edgy_centroids=True', '--set', 'centroid_refine_window=True',
          '--set', 'centroid_window_sigma=2.0']


def run(cmd, log_path, timeout=900):
    with open(log_path, 'w') as log:
        try:
            return subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            log.write('\nTIMEOUT\n')
            return -1


def wx(w, t_hms):
    return ['--set', f"observation_time={t_hms}", '--set', f"observation_temp={w['T']}",
            '--set', f"observation_pressure={w['P']}", '--set', f"observation_humidity={w['RH']}",
            '--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
            '--set', 'enable_gravitational_def=False', *SITE]


def build_h3_reference(wname, w):
    """The CAL_piLeo analogue: mid-block 10-frame H3 stack, quadratic-free fit."""
    outdir = os.path.join(RD, 'm5_rehearsal', wname, 'H3_reference')
    done = glob.glob(os.path.join(outdir, 'stage2', '**', 'distortion_results.txt'),
                     recursive=True)
    if done:
        return done[0]
    os.makedirs(outdir, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(G, w['date'], 'Horizon',
                                           'H3_calfield_sightline', '*', '*.fits')))[17:27]
    mids = []
    for f in frames:
        h = fits.getheader(f)
        mids.append(datetime.datetime.fromisoformat(h['DATE-OBS'])
                    + datetime.timedelta(seconds=float(h['EXPTIME']) / 2))
    t_mid = (mids[0] + (mids[-1] - mids[0]) / 2).strftime('%H:%M:%S')

    rc = run([sys.executable, '-m', 'mee2024.cli', 'stack', *frames, *STAGE1,
              '--no-scan', '--no-display', '--quiet', '-o', outdir],
             os.path.join(outdir, 'stage1.log'))
    zips = glob.glob(os.path.join(outdir, 'centroid_data*.zip'))
    if rc != 0 or not zips:
        raise RuntimeError(f'{wname} H3 reference stage 1 failed')
    zrefs = sorted(glob.glob(os.path.join(REFROOT, f"{w['refs']}_Z*.txt")))
    assert len(zrefs) == 6
    d2 = os.path.join(outdir, 'stage2')
    os.makedirs(d2, exist_ok=True)
    rc = run([sys.executable, '-m', 'mee2024.cli', 'distortion', zips[0],
              '--order', 'cubic', '--date-from-header', '--fix-distortion', *zrefs,
              '--set', 'distortion_fixed_coefficients=quadratic',
              '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
              *wx(w, t_mid), '--no-display', '--quiet', '-o', d2],
             os.path.join(d2, 'stage2.log'))
    done = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if rc != 0 or not done:
        raise RuntimeError(f'{wname} H3 reference stage 2 failed')
    return done[0]


def main():
    for wname, w in WINDOWS.items():
        ref = build_h3_reference(wname, w)
        j = json.load(open(ref))
        print(f"{wname} H3 reference: {j['#stars used']} stars, rms "
              f"{j['final rms error (arcseconds)']:.4f} arcsec, platescale "
              f"{j['platescale (arcseconds/pixel)']:.7f} arcsec/px", flush=True)

        frames = sorted(glob.glob(os.path.join(RD, 'perframe', wname, 'H1', 'f*')))
        for k, fdir in enumerate(frames, 1):
            outdir = os.path.join(RD, 'm5_rehearsal', wname, os.path.basename(fdir))
            if glob.glob(os.path.join(outdir, '**', 'distortion_results.txt'),
                         recursive=True):
                continue
            os.makedirs(outdir, exist_ok=True)
            zips = glob.glob(os.path.join(fdir, 'centroid_data*.zip'))
            prev = glob.glob(os.path.join(fdir, 'corr_on', '**', 'distortion_results.txt'),
                             recursive=True)
            if not zips or not prev:
                print(f'  {wname}/{os.path.basename(fdir)}: missing M2 inputs, skipped',
                      flush=True)
                continue
            t_hms = json.load(open(prev[0]))['observation_time (UTC)']
            rc = run([sys.executable, '-m', 'mee2024.cli', 'distortion', zips[0],
                      '--order', 'cubic', '--date-from-header', '--fix-distortion', ref,
                      '--set', 'distortion_fixed_coefficients=constant',
                      '--set', 'distortion_fit_tol=999', '--set', 'max_star_mag_dist=13',
                      *wx(w, t_hms), '--no-display', '--quiet', '-o', outdir],
                     os.path.join(outdir, 'stage2.log'))
            ok = bool(glob.glob(os.path.join(outdir, '**', 'distortion_results.txt'),
                                recursive=True))
            print(f'  [{k}/{len(frames)}] {wname} rehearsal {os.path.basename(fdir)} '
                  f'{"ok" if ok else "FAILED"}', flush=True)
    print('done: M5 rehearsal fits complete', flush=True)


if __name__ == '__main__':
    main()
