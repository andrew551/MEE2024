"""Leon 2026: the background-mode A/B on Leon ALONE, so `annular` is a measured choice.

Cell 1 decomposed the centroid convention into two axes and measured them on Bruns'
optics: the background-subtraction mode is worth ~19 ppm of plate scale, the estimator
(windowed vs footprint moments) under 2 ppm (docs/MATRIX_2026.md). Leon's headline was
reduced windowed+annular, and the only Leon re-reduction so far
(tools/step3_leon_bruns_convention.py) switched BOTH axes at once to Gaussian+moments
(-0.08 arcsec of L). So the background mode has never been isolated on Leon, and Leon's
`annular` is an inherited default rather than a measured decision. This closes that.

The 2x2 on Leon:

    windowed + annular    = the headline (cal_pileo_step2 canonical + step3_s0_v4 stacks)
    moments  + Gaussian   = step3_bruns_convention/            (already run)
    windowed + Gaussian   = step3_bg_ab/windowed_gaussian/     (THIS tool: the A/B cell)
    moments  + annular    = step3_bg_ab/moments_annular/       (THIS tool: completes the 2x2)

Each variant is applied at BOTH levels that see stars -- CAL_piLeo (the imported plate
scale) and the four science tiers -- because changing only one level is the cross-mix
that produced a meaningless +0.24 arcsec on the Bruns data. The six 08-12 zenith cubic
references stay windowed, and the frozen step3_s0_v4 preprocessed frames are reused, for
the same reasons the convention re-run gave: single-variable comparison against the
headline. The union/estimator is the headline's own machinery (tools/step3_s2_union.py)
with its paths redirected, so the numbers differ only by the stage-1 flags.

Usage:  python tools/step3_background_ab.py [windowed_gaussian|moments_annular|all]
"""
import glob, json, os, subprocess, sys, zipfile

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
ROOT = r"D:/MEE2024 output/MEE_output/step3_bg_ab"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
FRAMES = [l.strip() for l in open(os.path.join(REPO, "calibration", "cal_pileo_frames.txt"),
                                  encoding="utf-8") if l.strip()]
assert len(REFS) == 6 and len(FRAMES) == 16

# the eclipse-day standard of docs/FIELD_PRESETS.md, minus the two axes under test
S1_BASE = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
           '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
           '--set', 'sigma_subtract=0.0', '--set', 'delete_saturated_blob=False',
           '--set', 'remove_edgy_centroids=True', '--set', 'centroid_window_sigma=2.0']
VARIANTS = {
    'windowed_gaussian': ['--set', 'centroid_refine_window=True',
                          '--set', 'background_subtraction_mode=Gaussian'],
    'moments_annular':   ['--set', 'centroid_refine_window=False',
                          '--set', 'background_subtraction_mode=annular'],
}
SITE = ['--set', 'observation_lat=42.740470', '--set', 'observation_long=-5.613780',
        '--set', 'observation_height=1101', '--set', 'observation_humidity=0.208',
        '--set', 'observation_wavelength=0.62']
CAL_MET = ['--set', 'observation_temp=30.5', '--set', 'observation_pressure=896.6']
SCI_MET = ['--set', 'observation_temp=29.2', '--set', 'observation_pressure=896.7']
MIDT = {'0p1s': '18:28:32', '0p3s': '18:28:34', '0p6s': '18:28:33', '1p2s': '18:28:32'}


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def stage1(out, name, frames, s1):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY, '-m', 'mee2024.cli', 'stack', *frames, *s1, '--no-scan', '--no-display',
             '--quiet', '-o', d], os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print(f'{name}: STAGE 1 FAILED', flush=True)
        return None
    r = json.load(zipfile.ZipFile(z[0]).open('results.txt'))
    print(f"{name}: {r['n_centroids']} centroids (estimator {r.get('centroid estimator')}, "
          f"background {r.get('background stubtraction mode')})",   # the program's own key, typo included
          flush=True)
    return z[0]


def reduce_variant(vname):
    OUT = os.path.join(ROOT, vname)
    os.makedirs(OUT, exist_ok=True)
    S1 = S1_BASE + VARIANTS[vname]
    print(f'\n===== variant {vname}: {" ".join(VARIANTS[vname][1::2])} =====', flush=True)

    # ---- 1. CAL_piLeo, the imported plate scale. ORDER is part of the definition (F23).
    cz = stage1(OUT, 'cal_pileo', FRAMES, S1)
    d2 = os.path.join(OUT, 'cal_pileo', 'stage2')
    os.makedirs(d2, exist_ok=True)
    if cz and not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        run([PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', 'cubic',
             '--date-from-header', '--fix-distortion', *REFS,
             '--set', 'distortion_fixed_coefficients=quadratic',
             '--set', 'distortion_fit_tol=1.0', '--set', 'max_star_mag_dist=13',
             '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=True',
             '--set', 'enable_corrections_ref=True', *SITE, *CAL_MET,
             '--set', 'observation_time=18:29:35', '--no-display', '--quiet', '-o', d2],
            os.path.join(d2, 'stage2.log'))
    calres = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not calres:
        print(f'{vname}: CAL_piLeo stage 2 FAILED -- stopping this variant', flush=True)
        return
    j = json.load(open(calres[0], encoding='utf-8'))
    PS_NEW = j['platescale (arcseconds/pixel)']
    print(f"CAL_piLeo [{vname}]: {j['#stars used']} stars, rms "
          f"{j['final rms error (arcseconds)']:.4f}, ps {PS_NEW:.7f} "
          f"(headline windowed+annular 2.2054043: "
          f"{1e6*(PS_NEW-2.2054043)/2.2054043:+.1f} ppm)", flush=True)

    # ---- 2. the science tiers, constant-only against the new CAL
    for tier in ('0p1s', '0p3s', '0p6s', '1p2s'):
        frames = sorted(glob.glob(os.path.join(V4, tier, 'preprocessed', '*.fits')))
        tz = stage1(OUT, tier, frames, S1)
        if not tz:
            continue
        td = os.path.join(OUT, tier, 'stage2_constant')
        os.makedirs(td, exist_ok=True)
        if not glob.glob(os.path.join(td, '**', 'distortion_results.txt'), recursive=True):
            run([PY, '-m', 'mee2024.cli', 'distortion', tz, '--order', 'cubic',
                 '--date-from-header', '--fix-distortion', calres[0],
                 '--set', 'distortion_fixed_coefficients=constant',
                 '--set', 'distortion_fit_tol=2.0', '--set', 'max_star_mag_dist=13',
                 '--set', 'rough_match_threshhold=36', '--set', 'enable_corrections=True',
                 '--set', 'enable_corrections_ref=True', *SITE, *SCI_MET,
                 '--set', 'observation_time=' + MIDT[tier], '--no-display', '--quiet',
                 '-o', td], os.path.join(td, 'stage2.log'))
        r = glob.glob(os.path.join(td, '**', 'distortion_results.txt'), recursive=True)
        if r:
            jj = json.load(open(r[0], encoding='utf-8'))
            print(f"  {tier}: {jj['#stars used']} matched, rms "
                  f"{jj['final rms error (arcseconds)']:.4f}", flush=True)
        else:
            print(f'  {tier}: stage 2 FAILED', flush=True)

    # ---- 3. the union and the estimator, the headline's own machinery
    print(f'\n--- union + estimator [{vname}] ---', flush=True)
    src = open(os.path.join(REPO, 'tools', 'step3_s2_union.py'), encoding='utf-8').read()
    out_fwd = OUT.replace('\\', '/')
    src = src.replace('D:/MEE2024 output/MEE_output/step3_prelim_L', out_fwd)
    src = src.replace('D:/MEE2024 output/MEE_output/step3_s0_v4', out_fwd)
    src = src.replace("PS, NX, NY, W_NORM = 2.2054043", f"PS, NX, NY, W_NORM = {PS_NEW:.7f}")
    exec(compile(src, 'step3_s2_union_' + vname, 'exec'), {'__name__': '__main__'})


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for v in (list(VARIANTS) if which == 'all' else [which]):
        reduce_variant(v)
    print('done all', flush=True)
